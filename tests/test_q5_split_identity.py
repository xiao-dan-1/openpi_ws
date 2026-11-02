"""Relocation accepts only explicit, hash-verified training manifests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.q5.runtime import sha256
from pipeline.q5.split_identity import verify_training_split


class SplitIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.local = self.root / "local.json"
        self.reference = self.root / "training.json"
        self.manifest = {"status": "PASS", "seed": 42, "source": "/server/data",
                         "source_info_sha256": "source-digest", "locked_test_source_episode_ids": [2]}
        for i, role in enumerate(["train", "validation", "test"]):
            self.manifest[role] = [{"source_episode_index": i, "original_episode_index": None,
                                    "output_episode_index": 0, "length": 10,
                                    "data_path": "episode.parquet", "sha256": f"parquet-{i}",
                                    "source_bag": f"bag-{i}", "assets": [
                                        {"path": "video.mp4", "sha256": f"video-{i}"},
                                        {"path": "timing.json", "sha256": f"timing-{i}"}]}]
        self.reference.write_text(json.dumps(self.manifest))
        self.expected = sha256(self.reference)
        self.local.write_bytes(self.reference.read_bytes())

    def verify(self):
        return verify_training_split(self.local, self.expected, self.reference)

    def test_exact_manifest_needs_no_reference(self):
        self.assertEqual(verify_training_split(self.local, self.expected)["comparison"], "exact_manifest_sha256")

    def test_relocated_history_requires_reference_and_preserves_files(self):
        relocated = copy.deepcopy(self.manifest)
        relocated.update(source="/local/data", cleaning_history_available=True,
                         cleaning_manifest_sha256="history-digest", episode_identity="history available")
        for role in ["train", "validation", "test"]:
            relocated[role][0]["original_episode_index"] = 100 + relocated[role][0]["source_episode_index"]
        self.local.write_text(json.dumps(relocated, indent=2))
        before = self.local.read_bytes(), self.reference.read_bytes()
        with self.assertRaisesRegex(ValueError, "--training-split-manifest"):
            verify_training_split(self.local, self.expected)
        self.assertEqual(self.verify()["comparison"], "verified_data_identity")
        self.assertEqual(before, (self.local.read_bytes(), self.reference.read_bytes()))

    def test_untrusted_reference_rejected_even_when_local_hash_matches(self):
        self.reference.write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            self.verify()

    def test_missing_reference_rejected(self):
        self.reference.unlink()
        with self.assertRaisesRegex(ValueError, "not found"):
            self.verify()

    def test_changed_training_identity_rejected(self):
        for role in ["train", "validation", "test"]:
            for field in ["source_episode_index", "output_episode_index", "length", "data_path", "sha256", "source_bag"]:
                with self.subTest(role=role, field=field):
                    changed = copy.deepcopy(self.manifest)
                    changed[role][0][field] = "changed"
                    self.local.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, "episode identities"):
                        self.verify()

    def test_changed_assets_and_population_rejected(self):
        for change in ["video", "timing", "population", "seed", "unknown_field", "source_info"]:
            with self.subTest(change=change):
                changed = copy.deepcopy(self.manifest)
                if change in ["video", "timing"]:
                    changed["validation"][0]["assets"][change == "timing"]["sha256"] = "changed"
                elif change == "population":
                    changed["validation"] = []
                elif change == "source_info":
                    changed["source_info_sha256"] = "changed"
                else:
                    changed[change] = 99
                self.local.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "episode identities"):
                    self.verify()

    def test_conflicting_known_original_ids_rejected(self):
        self.manifest["train"][0]["original_episode_index"] = 100
        self.reference.write_text(json.dumps(self.manifest))
        self.expected = sha256(self.reference)
        self.manifest["train"][0]["original_episode_index"] = 101
        self.local.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Conflicting original"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
