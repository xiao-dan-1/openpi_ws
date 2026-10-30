"""Stdlib provenance checks through the real split preparation and runtime guards."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pipeline.q5.prepare_data import prepare_split
from pipeline.q5.runtime import Paths, SPLIT_METADATA_FILES, require_stats, sha256, validate_split
from pipeline.q5.schema import JOINT_NAMES


class Q5MetadataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.paths = Paths.create(directory.name)
        source = Path(directory.name) / "source"
        (source / "meta").mkdir(parents=True)
        (source / "meta/info.json").write_text(json.dumps({
            "total_episodes": 65, "chunks_size": 1000, "data_path": "episode_{episode_index:06d}.parquet",
            "features": {key: {"names": list(JOINT_NAMES), "dtype": "float32"}
                         for key in ["action", "observation.state"]}}))
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
        info = json.loads((source / "meta/info.json").read_text())
        info.update(total_episodes=len(included), total_frames=len(included))
        (destination / "meta/info.json").write_text(json.dumps(info))
        (destination / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "pick orange"}\n')
        episodes, mapping = [], []
        for new, old in enumerate(included):
            (destination / f"episode_{new:06d}.parquet").write_bytes(f"fixture-{old}".encode())
            (destination / f"meta/timing/episode_{new:06d}.json").write_text(json.dumps({"source_bag": f"bag-{old}"}))
            episodes.append({"episode_index": new, "length": 1})
            mapping.append({"original_episode_index": old, "new_episode_index": new, "length": 1})
        (destination / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
        return {"episode_mapping": mapping}

    def test_preparation_records_both_metadata_fingerprints_for_each_role(self):
        for role in ["train", "validation", "test"]:
            root = getattr(self.paths, role)
            self.assertEqual(self.manifest["metadata_sha256"][role],
                             {name: sha256(root / name) for name in SPLIT_METADATA_FILES})
        self.assertEqual(require_stats(self.paths)["episodes_used"], 50)

    def test_metadata_edits_are_rejected_by_split_and_statistics_guards(self):
        for role in ["train", "validation", "test"]:
            for name in SPLIT_METADATA_FILES:
                with self.subTest(role=role, name=name):
                    path = getattr(self.paths, role) / name
                    before = path.read_bytes()
                    if name == "meta/tasks.jsonl":
                        path.write_text('{"task_index": 0, "task": "changed prompt"}\n')
                    else:
                        # Preserve episode boundaries to test the fingerprint independently.
                        path.write_bytes(before + b"\n")
                    try:
                        for guard in [validate_split, require_stats]:
                            with self.assertRaisesRegex(ValueError, f"Changed {role} metadata: {name}"):
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
                            with self.assertRaisesRegex(ValueError, f"Incorrect {role} episode IDs/lengths"):
                                validate_split(self.paths, check_files=False)
                        finally:
                            path.write_text(before)

    def test_old_or_partial_manifests_are_rejected_without_backfilling(self):
        original = self.paths.split_manifest.read_bytes()
        for role in [None, "train", "validation", "test"]:
            for check_files in [False, True]:
                with self.subTest(role=role, check_files=check_files):
                    manifest = json.loads(original)
                    if role is None:
                        del manifest["metadata_sha256"]
                    else:
                        del manifest["metadata_sha256"][role]["meta/tasks.jsonl"]
                    self.paths.split_manifest.write_text(json.dumps(manifest))
                    before = self.paths.split_manifest.read_bytes()
                    with self.assertRaisesRegex(ValueError, "Missing .* metadata fingerprints.*do not backfill hashes"):
                        validate_split(self.paths, check_files=check_files)
                    self.assertEqual(self.paths.split_manifest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
