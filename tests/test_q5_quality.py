"""Real tiny Parquet/video fixtures verify audit failures and output protection."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from checks import check_q5_quality as quality


class QualityAuditTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="q5-quality-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "dataset"
        self.out = Path(directory.name) / "report"
        (self.root / "meta/timing").mkdir(parents=True)
        (self.root / "data").mkdir()
        (self.root / "videos").mkdir()
        self.frames = 16
        names = [f"joint_{i}" for i in range(38)]
        info = dict(fps=30, total_episodes=1, total_tasks=1, total_frames=self.frames,
                    total_videos=1, chunks_size=1000,
                    data_path="data/episode_{episode_index:06d}.parquet",
                    video_path="videos/episode_{episode_index:06d}.mp4",
                    features={key: dict(names=names, shape=[38]) for key in ["action", "observation.state"]})
        info["features"]["observation.images.image_head"] = dict(shape=[48, 64, 3])
        self.write_json("meta/info.json", info)
        self.write_json("meta/episodes.jsonl", dict(episode_index=0, length=self.frames, tasks=["pick"]))
        self.write_json("meta/tasks.jsonl", dict(task_index=0, task="pick"))
        self.write_json("meta/timing/episode_000000.json",
                        dict(relative_image_stamps=(np.arange(self.frames) / 30).tolist(), qc={"n_img": self.frames}))
        vectors = {key: pa.array(np.zeros((self.frames, 38)).tolist(), type=pa.list_(pa.float32(), 38))
                   for key in ["action", "observation.state"]}
        self.table = pa.table(dict(**vectors, frame_index=range(self.frames), episode_index=[0] * self.frames,
                                  task_index=[0] * self.frames, index=range(self.frames),
                                  timestamp=np.arange(self.frames) / 30))
        pq.write_table(self.table, self.root / "data/episode_000000.parquet")
        self.stats = {key: {**{stat: [0.0] * 38 for stat in ["min", "max", "mean", "std"]},
                            "count": [self.frames]} for key in vectors}
        self.write_stats()
        self.write_video()

    def write_json(self, path, value):
        (self.root / path).write_text(json.dumps(value) + "\n", encoding="utf-8")

    def write_stats(self):
        self.write_json("meta/episodes_stats.jsonl", dict(episode_index=0, stats=self.stats))

    def write_video(self, black=False):
        with av.open(str(self.root / "videos/episode_000000.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for i in range(self.frames):
                pixels = np.full((48, 64, 3), 0 if black else 60 + i, np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def run_audit(self):
        console, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(errors):
            code = quality.main([str(self.root), str(self.out)])
        return code, console.getvalue(), errors.getvalue()

    def report(self):
        text = (self.out / "quality_metrics.json").read_text(encoding="utf-8")
        # A completed report must be portable JSON, including on missing-file paths.
        return json.loads(text, parse_constant=lambda value: self.fail(f"Invalid JSON constant: {value}"))

    def test_valid_data_passes_and_never_changes_source(self):
        fingerprints = {p: hashlib.sha256(p.read_bytes()).digest() for p in self.root.rglob("*") if p.is_file()}
        code, console, _ = self.run_audit()
        self.assertEqual(code, 0)
        self.assertEqual(self.report()["status"], "PASS")
        self.assertIn('"status": "PASS"', console)
        self.assertFalse((self.out / "INCOMPLETE").exists())
        self.assertEqual(self.report()["decoded_video_frames"], self.frames)
        self.assertEqual(fingerprints, {p: hashlib.sha256(p.read_bytes()).digest()
                                       for p in self.root.rglob("*") if p.is_file()})

    def test_bad_global_index_returns_nonzero_from_actual_cli(self):
        column = self.table.schema.get_field_index("index")
        table = self.table.set_column(column, "index", pa.array(np.arange(self.frames) + 100))
        pq.write_table(table, self.root / "data/episode_000000.parquet")
        result = subprocess.run([sys.executable, "-B", str(Path(quality.__file__)), str(self.root), str(self.out)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.report()["status"], "FAIL")
        self.assertIn("global index not consecutive/unique", self.report()["errors"])
        self.assertFalse((self.out / "INCOMPLETE").exists())

    def test_wrong_statistics_fail_and_are_visible_in_summary(self):
        self.stats["action"]["mean"] = [99.0] * 38
        self.write_stats()
        code, console, _ = self.run_audit()
        self.assertEqual(code, 1)
        self.assertEqual(self.report()["status"], "FAIL")
        self.assertIn("stored action.mean differs from data", console)
        self.assertIn('"metadata_stat_mismatches"', console)
        self.assertEqual(self.report()["metadata_stat_mismatches"][0]["max_abs_error"], 99.0)

    def test_stats_episode_ids_must_match_exactly_once(self):
        for label, records in [
            ("missing", []),
            ("unexpected", [dict(episode_index=1, stats=self.stats)]),
            ("duplicate", [dict(episode_index=0, stats=self.stats)] * 2),
        ]:
            with self.subTest(label=label):
                self.out = self.out.parent / f"report-{label}"
                (self.root / "meta/episodes_stats.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
                code, _, error = self.run_audit()
                self.assertEqual(code, 1, error)
                self.assertEqual(self.report()["status"], "FAIL")
                expected = "episode stats duplicate indices" if label == "duplicate" else "episode stats indices missing or unexpected"
                self.assertIn(expected, self.report()["errors"])
                self.assertFalse((self.out / "INCOMPLETE").exists())

    def test_required_stats_and_shapes_cannot_pass_through_broadcasting(self):
        baseline = json.dumps(self.stats)
        cases = [("feature", None)] + [(field, None) for field in ["min", "max", "mean", "std", "count"]]
        cases += [("mean", 0.0), ("mean", [0.0]), ("mean", [[0.0] * 38]),
                  ("mean", [0.0] * 37), ("count", self.frames), ("count", [self.frames] * 38),
                  ("count", []), ("std", [float("inf")] * 38)]
        for i, (field, value) in enumerate(cases):
            with self.subTest(field=field, value=value):
                self.out = self.out.parent / f"report-stats-{i}"
                self.stats = json.loads(baseline)
                if field == "feature":
                    del self.stats["action"]
                elif value is None:
                    del self.stats["action"][field]
                else:
                    self.stats["action"][field] = value
                self.write_stats()
                code, _, error = self.run_audit()
                self.assertEqual(code, 1, error)
                self.assertEqual(self.report()["status"], "FAIL")
                self.assertTrue(any("stored action." in e for e in self.report()["errors"]))
                self.assertFalse((self.out / "INCOMPLETE").exists())

    def test_nonfinite_vectors_complete_with_strict_json_failure(self):
        for i, (key, value) in enumerate([
            ("action", float("nan")), ("observation.state", float("inf")),
            ("action", -float("inf")),
        ]):
            with self.subTest(key=key, value=value):
                self.out = self.out.parent / f"report-nonfinite-{i}"
                values = np.zeros((self.frames, 38))
                values[0, 0] = value
                column = self.table.schema.get_field_index(key)
                table = self.table.set_column(column, key, pa.array(values.tolist(), type=pa.list_(pa.float32(), 38)))
                pq.write_table(table, self.root / "data/episode_000000.parquet")
                code, console, error = self.run_audit()
                self.assertEqual(code, 1, error)
                report = self.report()
                self.assertEqual(report["status"], "FAIL")
                self.assertIn(f"episode 0: {key} NaN/Inf", report["errors"])
                self.assertEqual(report["decoded_video_frames"], self.frames)
                json.loads(console[console.index("{"):], parse_constant=lambda value: self.fail(value))
                self.assertFalse((self.out / "INCOMPLETE").exists())

    def test_nonfinite_timestamps_and_sidecar_complete_with_strict_json_failure(self):
        timestamps = np.arange(self.frames) / 30
        timestamps[0] = float("nan")
        column = self.table.schema.get_field_index("timestamp")
        pq.write_table(self.table.set_column(column, "timestamp", pa.array(timestamps)),
                       self.root / "data/episode_000000.parquet")
        self.write_json("meta/timing/episode_000000.json", dict(
            relative_image_stamps=timestamps.tolist(), qc={"n_img": self.frames, "age_s_max": float("inf")}))
        code, _, error = self.run_audit()
        self.assertEqual(code, 1, error)
        report = self.report()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("episode 0: timestamp NaN/Inf", report["errors"])
        self.assertIsNone(report["source_qc"]["episodes"][0]["age_s_max"])
        self.assertFalse((self.out / "INCOMPLETE").exists())

    def test_source_qc_missing_fields_are_unknown_and_have_coverage(self):
        code, _, error = self.run_audit()
        self.assertEqual(code, 0, error)
        qc = self.report()["source_qc"]
        self.assertEqual(qc["totals"]["source_images"], self.frames)
        self.assertIsNone(qc["totals"]["stale_action"])
        self.assertIsNone(qc["maxima"]["age_s_max"])
        self.assertIsNone(qc["episodes"][0]["no_state"])
        self.assertEqual(qc["coverage"]["observed_episodes"]["source_images"], 1)
        self.assertEqual(qc["coverage"]["observed_episodes"]["stale_action"], 0)
        self.assertEqual(qc["coverage"]["expected_episodes"], 1)

    def test_source_qc_partial_coverage_does_not_report_partial_sum_as_total(self):
        self.write_json("meta/timing/episode_000000.json", dict(
            relative_image_stamps=[0, 1], qc={"n_img": 2, "no_state": 0}))
        qc = quality.source_qc(self.root, [0, 1])
        self.assertIsNone(qc["totals"]["source_images"])
        self.assertIsNone(qc["totals"]["no_state"])
        self.assertEqual(qc["coverage"]["missing_episode_ids"], [1])
        self.assertEqual(qc["coverage"]["observed_episodes"]["no_state"], 1)
        self.assertEqual(qc["episodes"][0]["no_state"], 0)

    def test_short_episode_finishes_failure_report_without_empty_reductions(self):
        self.frames = 1
        info = json.loads((self.root / "meta/info.json").read_text())
        info["total_frames"] = 1
        self.write_json("meta/info.json", info)
        self.write_json("meta/episodes.jsonl", dict(episode_index=0, length=1, tasks=["pick"]))
        self.write_json("meta/timing/episode_000000.json", dict(relative_image_stamps=[0.0], qc={"n_img": 1}))
        for values in self.stats.values():
            values["count"] = [1]
        self.write_stats()
        pq.write_table(self.table.slice(0, 1), self.root / "data/episode_000000.parquet")
        self.write_video()
        code, _, error = self.run_audit()
        self.assertEqual(code, 1, error)
        self.assertEqual(self.report()["status"], "FAIL")
        self.assertIn("episode 0: row count 1 vs 1", self.report()["errors"])
        self.assertEqual(self.report()["decoded_video_frames"], 1)
        self.assertFalse((self.out / "INCOMPLETE").exists())

    def test_later_valid_episode_still_writes_csv_after_invalid_vector(self):
        info = json.loads((self.root / "meta/info.json").read_text())
        info.update(total_episodes=2, total_videos=2, total_frames=2 * self.frames)
        self.write_json("meta/info.json", info)
        (self.root / "meta/episodes.jsonl").write_text("".join(
            json.dumps(dict(episode_index=eid, length=self.frames, tasks=["pick"])) + "\n"
            for eid in range(2)), encoding="utf-8")
        (self.root / "meta/episodes_stats.jsonl").write_text("".join(
            json.dumps(dict(episode_index=eid, stats=self.stats)) + "\n" for eid in range(2)), encoding="utf-8")
        (self.root / "videos/episode_000001.mp4").write_bytes((self.root / "videos/episode_000000.mp4").read_bytes())
        self.write_json("meta/timing/episode_000001.json", dict(
            relative_image_stamps=(np.arange(self.frames) / 30).tolist(), qc={"n_img": self.frames}))
        next_table = self.table.set_column(self.table.schema.get_field_index("episode_index"),
                                          "episode_index", pa.array([1] * self.frames))
        next_table = next_table.set_column(next_table.schema.get_field_index("index"),
                                          "index", pa.array(np.arange(self.frames) + self.frames))
        pq.write_table(next_table, self.root / "data/episode_000001.parquet")
        values = np.zeros((self.frames, 38))
        values[0, 0] = float("nan")
        broken = self.table.set_column(self.table.schema.get_field_index("action"), "action",
                                       pa.array(values.tolist(), type=pa.list_(pa.float32(), 38)))
        pq.write_table(broken, self.root / "data/episode_000000.parquet")
        code, _, error = self.run_audit()
        self.assertEqual(code, 1, error)
        self.assertEqual(self.report()["episodes_checked"], 2)
        self.assertEqual(self.report()["decoded_video_frames"], 2 * self.frames)
        self.assertTrue((self.out / "episodes.csv").is_file())
        self.assertFalse((self.out / "INCOMPLETE").exists())

    def test_visual_warning_does_not_become_structural_failure(self):
        self.write_video(black=True)
        code, console, _ = self.run_audit()
        self.assertEqual(code, 0)
        self.assertIn("black", console)
        self.assertTrue(self.report()["warnings"])
        self.assertEqual(self.report()["errors"], [])

    def test_missing_episode_files_still_produce_completed_failure_report(self):
        (self.root / "videos/episode_000000.mp4").unlink()
        (self.root / "meta/timing/episode_000000.json").unlink()
        code, _, _ = self.run_audit()
        self.assertEqual(code, 1)
        self.assertEqual(self.report()["status"], "FAIL")
        self.assertIn("episode 0: missing parquet/video", self.report()["errors"])
        self.assertFalse((self.out / "INCOMPLETE").exists())

    def test_existing_report_is_rejected_without_touching_anything(self):
        self.out.mkdir()
        previous = self.out / "quality_metrics.json"
        previous.write_bytes(b"previous report")
        with mock.patch.object(quality, "audit") as audit, contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as error:
            quality.main([str(self.root), str(self.out)])
        self.assertEqual(error.exception.code, 2)
        audit.assert_not_called()
        self.assertEqual(previous.read_bytes(), b"previous report")
        self.assertEqual(list(self.out.iterdir()), [previous])

    def test_interrupted_audit_leaves_marker_and_cannot_be_reused(self):
        with mock.patch.object(quality, "audit", side_effect=ValueError("fixture failure")):
            code, _, error = self.run_audit()
        self.assertEqual(code, 1)
        self.assertIn("ValueError: fixture failure", error)
        self.assertTrue((self.out / "INCOMPLETE").is_file())
        self.assertFalse((self.out / "quality_metrics.json").exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            quality.main([str(self.root), str(self.out)])

    def test_output_inside_dataset_is_rejected_before_creating_files(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            quality.main([str(self.root), str(self.root / "reports")])
        self.assertFalse((self.root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
