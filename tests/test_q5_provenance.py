"""Stdlib provenance checks through the real split preparation and runtime guards."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pipeline.q5.prepare_data import prepare_split
from pipeline.q5.runtime import Paths, require_stats, sha256, validate_split
from pipeline.q5.schema import JOINT_NAMES


class Q5ProvenanceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.paths = Paths.create(directory.name)
        source = Path(directory.name) / "source"
        (source / "meta").mkdir(parents=True)
        info = {
            "total_episodes": 65, "chunks_size": 1000, "data_path": "episode_{episode_index:06d}.parquet",
            "video_path": "{video_key}/episode_{episode_index:06d}.mp4",
            "features": {key: {"names": list(JOINT_NAMES), "dtype": "float32"}
                         for key in ["action", "observation.state"]}}
        info["features"]["camera"] = {"dtype": "video"}
        (source / "meta/info.json").write_text(json.dumps(info))
        # Only replace the PyArrow-backed copier; preparation and validation run unchanged.
        with mock.patch.dict(sys.modules, {"pipeline.data.exclude_episodes": SimpleNamespace(run=self.copy_split)}):
            self.manifest = prepare_split(self.paths, source)
        self.paths.stats.mkdir(parents=True)
        (self.paths.stats / "norm_stats.json").write_text("{}")
        (self.paths.stats / "statistics_run.json").write_text(json.dumps(dict(
            status="PASS", training_eligible=True, test_episodes_used=0, validation_episodes_used=0,
            episodes_used=50, frames_used=50, selected_episode_ids=list(range(50)), split="train",
            action_horizon=10, joint_dimension=38, action_representation="absolute_joint_targets",
            delta_transform=False, source_info_sha256=sha256(self.paths.train / "meta/info.json"),
            split_manifest_sha256=sha256(self.paths.split_manifest),
            sha256=sha256(self.paths.stats / "norm_stats.json"), joint_names=list(JOINT_NAMES),
            source_parquet_fingerprints=[{"path": row["data_path"], "sha256": row["sha256"]}
                                         for row in self.manifest["train"]])))

    @staticmethod
    def copy_split(source, destination, excluded):
        included = sorted(set(range(65)) - excluded)
        (destination / "meta/timing").mkdir(parents=True)
        (destination / "camera").mkdir()
        info = json.loads((source / "meta/info.json").read_text())
        info.update(total_episodes=len(included), total_frames=len(included))
        (destination / "meta/info.json").write_text(json.dumps(info))
        (destination / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "pick orange"}\n')
        episodes, mapping = [], []
        for new, old in enumerate(included):
            (destination / f"episode_{new:06d}.parquet").write_bytes(f"fixture-{old}".encode())
            (destination / f"camera/episode_{new:06d}.mp4").write_bytes(f"video-{old}".encode())
            (destination / f"meta/timing/episode_{new:06d}.json").write_text(json.dumps({"source_bag": f"bag-{old}"}))
            episodes.append({"episode_index": new, "length": 1})
            mapping.append({"original_episode_index": old, "new_episode_index": new, "length": 1})
        (destination / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
        return {"episode_mapping": mapping}

    def test_preparation_and_statistics_accept_manifest_without_metadata_fingerprints(self):
        self.assertNotIn("metadata_sha256", self.manifest)
        before = self.paths.split_manifest.read_bytes()
        self.assertEqual(validate_split(self.paths), self.manifest)
        self.assertEqual(require_stats(self.paths)["episodes_used"], 50)
        self.assertEqual(self.paths.split_manifest.read_bytes(), before)

    def test_extra_metadata_fingerprints_are_ignored_without_rewriting_manifest_or_statistics(self):
        self.manifest["metadata_sha256"] = {
            role: {name: "0" * 64 for name in ["meta/tasks.jsonl", "meta/episodes.jsonl"]}
            for role in ["train", "validation", "test"]}
        self.paths.split_manifest.write_text(json.dumps(self.manifest))
        # Model an existing manifest and its matching statistics report.
        report_path = self.paths.stats / "statistics_run.json"
        report = json.loads(report_path.read_text())
        report["split_manifest_sha256"] = sha256(self.paths.split_manifest)
        report_path.write_text(json.dumps(report))
        before = self.paths.split_manifest.read_bytes(), report_path.read_bytes()
        for role in ["train", "validation", "test"]:
            root = getattr(self.paths, role)
            (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "changed prompt"}\n')
            with (root / "meta/episodes.jsonl").open("a") as stream:
                stream.write("\n")
        for check_files in [False, True]:
            validate_split(self.paths, check_files=check_files)
        self.assertEqual(require_stats(self.paths)["episodes_used"], 50)
        self.assertEqual((self.paths.split_manifest.read_bytes(), report_path.read_bytes()), before)

    def test_parquet_video_and_timing_fingerprints_are_still_required(self):
        for role in ["train", "validation", "test"]:
            row = self.manifest[role][0]
            for relative in [row["data_path"], *(asset["path"] for asset in row["assets"])]:
                with self.subTest(role=role, path=relative):
                    path = getattr(self.paths, role) / relative
                    before = path.read_bytes()
                    path.write_bytes(b"changed")
                    try:
                        validate_split(self.paths, check_files=False)
                        for guard in [validate_split, require_stats]:
                            with self.assertRaisesRegex(ValueError, f"Changed {role} (Parquet|asset)"):
                                guard(self.paths)
                    finally:
                        path.write_bytes(before)

    def test_check_files_false_skips_hashes_but_still_checks_episode_boundaries(self):
        with mock.patch("pipeline.q5.runtime.sha256", side_effect=AssertionError("Hashing disabled")):
            validate_split(self.paths, check_files=False)
        for role in ["train", "validation", "test"]:
            path = getattr(self.paths, role) / "meta/episodes.jsonl"
            before = path.read_text()
            rows = [json.loads(line) for line in before.splitlines()]
            for change in ["lengths", "duplicate_id", "missing", "unknown_id"]:
                with self.subTest(role=role, change=change):
                    changed = [dict(row) for row in rows]
                    if change == "lengths":
                        changed[0]["length"], changed[1]["length"] = 0, 2
                    elif change == "duplicate_id":
                        changed[1]["episode_index"] = 0
                    elif change == "missing":
                        changed.pop()
                    else:
                        changed[0]["episode_index"] = 100
                    path.write_text("".join(json.dumps(row) + "\n" for row in changed))
                    try:
                        for guard in [lambda paths: validate_split(paths, check_files=False),
                                      validate_split, require_stats]:
                            with self.assertRaisesRegex(ValueError, f"Incorrect {role} episode IDs/lengths"):
                                guard(self.paths)
                    finally:
                        path.write_text(before)

    def test_missing_or_invalid_episode_metadata_fails_clearly(self):
        path = self.paths.validation / "meta/episodes.jsonl"
        original = path.read_bytes()
        for contents, error in [(None, "Missing"), (b"{invalid json}\n", "Invalid")]:
            with self.subTest(error=error):
                if contents is None:
                    path.unlink()
                else:
                    path.write_bytes(contents)
                try:
                    for check_files in [False, True]:
                        with self.assertRaisesRegex(ValueError, f"{error} validation episode metadata: meta/episodes.jsonl"):
                            validate_split(self.paths, check_files=check_files)
                finally:
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
