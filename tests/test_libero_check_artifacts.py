"""Small fixture checks; OpenPI/model integration is deliberately not exercised."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from checks import check_libero_normalization as normalization
from checks import check_libero_training_input as training_input
from checks import libero_provenance as provenance

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    np = pa = pq = None


class OutputSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "dataset"
        (self.root / "meta").mkdir(parents=True)
        (self.root / "meta/info.json").write_text("{}", encoding="utf-8")

    def test_existing_output_is_preserved_before_heavy_imports(self):
        output = self.directory / "stats"
        output.mkdir()
        original = {"norm_stats.json": b"previous stats", "validation_manifest.json": b"previous pass"}
        for name, data in original.items():
            (output / name).write_bytes(data)
        with self.assertRaises(FileExistsError):
            normalization.main(["--data-root", str(self.root), "--output-dir", str(output)])
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, original)

    def test_output_inside_dataset_is_refused_without_creating_it(self):
        for output in (self.root, self.root / "new" / "stats"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                normalization.prepare_output(output, self.root)
        self.assertFalse((self.root / "new").exists())

    def test_missing_source_does_not_create_output(self):
        output = self.directory / "stats"
        with self.assertRaisesRegex(ValueError, "Dataset is absent"):
            normalization.main(["--data-root", str(self.directory / "missing"),
                                "--output-dir", str(output)])
        self.assertFalse(output.exists())

    @unittest.skipUnless(np is not None, "requires optional numpy/pyarrow")
    def test_failure_leaves_incomplete_marker_and_no_pass(self):
        output = self.directory / "stats"
        modules = {
            "openpi.shared.normalize": SimpleNamespace(RunningStats=object, load=mock.Mock(), save=mock.Mock()),
            "openpi.transforms": SimpleNamespace(Normalize=object, Unnormalize=object),
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ), \
                mock.patch.object(sys, "dont_write_bytecode", sys.dont_write_bytecode), \
                mock.patch.object(normalization, "load_metadata", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                normalization.main(["--data-root", str(self.root), "--output-dir", str(output)])
        self.assertTrue((output / "INCOMPLETE").is_file())
        self.assertFalse((output / "validation_manifest.json").exists())

    def test_import_does_not_change_process_environment_or_bytecode_setting(self):
        before, bytecode = dict(os.environ), sys.dont_write_bytecode
        importlib.reload(normalization)
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(sys.dont_write_bytecode, bytecode)

    def test_direct_scripts_support_help_without_openpi(self):
        for module in (normalization, training_input):
            result = subprocess.run([sys.executable, module.__file__, "--help"],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--data-root", result.stdout)
        self.assertIn("--stats-dir", result.stdout)

    def test_consumer_uses_selected_stats_and_refuses_incomplete_before_model_import(self):
        output = self.directory / "chosen-stats"
        output.mkdir()
        (output / "INCOMPLETE").touch()
        with mock.patch.dict(os.environ), mock.patch.object(sys, "dont_write_bytecode", sys.dont_write_bytecode):
            with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
                training_input.main(["--data-root", str(self.root), "--stats-dir", str(output)])

    def test_missing_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalization check first"):
            provenance.validate_stats(self.directory / "missing-stats", self.root)


@unittest.skipUnless(np is not None, "requires optional numpy/pyarrow")
class NumericProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "dataset"
        (self.root / "meta").mkdir(parents=True)
        self.info = {"fps": 10, "chunks_size": 1000, "data_path": "episode_{episode_index:06d}.parquet"}
        # Deliberately unsorted on disk: the original digest sorts episodes.
        self.episodes = [{"episode_index": 1, "length": 1}, {"episode_index": 0, "length": 2}]
        self.tasks = [{"task_index": 0, "task": "pick cup"}]
        (self.root / "meta/info.json").write_text(json.dumps(self.info), encoding="utf-8")
        (self.root / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.episodes), encoding="utf-8")
        (self.root / "meta/tasks.jsonl").write_text(json.dumps(self.tasks[0]) + "\n", encoding="utf-8")
        self.arrays = {}
        for episode in self.episodes:
            index, length = episode["episode_index"], episode["length"]
            state = (np.arange(length * 8).reshape(length, 8) + index).astype(np.float32)
            actions = (np.arange(length * 7).reshape(length, 7) / 10 + index).astype(np.float32)
            timestamp = np.arange(length, dtype=np.float64) / 10
            self.arrays[index] = state, actions, timestamp
            self.write_episode(index)
        # Compute expected bytes independently using the pre-change algorithm.
        digest = hashlib.sha256()
        digest.update(json.dumps({"tasks": self.tasks,
                                  "episodes": sorted(self.episodes, key=lambda row: row["episode_index"]),
                                  "fps": self.info["fps"]}, sort_keys=True, separators=(",", ":")).encode())
        for index in sorted(self.arrays):
            for array in self.arrays[index]:
                digest.update(np.asarray(array, dtype="<f4").tobytes())
        self.expected_digest = digest.hexdigest()
        self.stats = Path(temporary.name) / "stats"
        self.stats.mkdir()
        (self.stats / "norm_stats.json").write_bytes(b'{"mock": "stats"}')
        self.manifest = {"status": "PASS", "validation_profile": provenance.PROFILE,
                         "numeric_data_sha256": self.expected_digest,
                         "norm_stats_sha256": hashlib.sha256((self.stats / "norm_stats.json").read_bytes()).hexdigest()}
        self.write_manifest()

    def write_episode(self, index):
        state, actions, timestamp = self.arrays[index]
        pq.write_table(pa.table({"state": state.tolist(), "actions": actions.tolist(),
                                 "timestamp": timestamp, "image": [b"unused"] * len(state)}),
                       self.root / self.info["data_path"].format(episode_index=index))

    def write_manifest(self):
        (self.stats / "validation_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_original_digest_is_accepted_using_only_numeric_columns(self):
        original = pq.ParquetFile
        readers = []

        def tracked_reader(path):
            source = original(path)
            reader = mock.MagicMock(wraps=source)
            reader.__enter__.return_value = reader
            reader.__exit__.side_effect = lambda *args: source.close()
            readers.append(reader)
            return reader

        with mock.patch.object(pq, "ParquetFile", side_effect=tracked_reader):
            actual = provenance.validate_stats(self.stats, self.root)
        self.assertEqual(actual["numeric_data_sha256"], self.expected_digest)
        self.assertEqual(len(readers), 2)
        for reader in readers:
            reader.read.assert_called_once_with(columns=["state", "actions", "timestamp"], use_threads=False)

    def test_changes_in_each_numeric_column_are_rejected(self):
        for column in range(3):
            with self.subTest(column=column):
                values = self.arrays[0][column]
                values.flat[0] += 1
                self.write_episode(0)
                with self.assertRaisesRegex(ValueError, "do not match --data-root"):
                    provenance.validate_stats(self.stats, self.root)
                values.flat[0] -= 1
                self.write_episode(0)

    def test_changed_task_metadata_is_rejected(self):
        (self.root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "different"}\n')
        with self.assertRaisesRegex(ValueError, "do not match --data-root"):
            provenance.validate_stats(self.stats, self.root)

    def test_changed_stats_and_missing_provenance_are_rejected(self):
        del self.manifest["numeric_data_sha256"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "do not match --data-root"):
            provenance.validate_stats(self.stats, self.root)
        (self.stats / "norm_stats.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "stats changed"):
            provenance.validate_stats(self.stats, self.root)

    def test_incomplete_artifacts_are_rejected_even_with_pass_manifest(self):
        (self.stats / "INCOMPLETE").touch()
        with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
            provenance.validate_stats(self.stats, self.root)

    def test_failed_or_different_profile_manifest_is_rejected(self):
        for field, value in (("status", "FAIL"), ("validation_profile", "different")):
            with self.subTest(field=field):
                original = self.manifest[field]
                self.manifest[field] = value
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "Wrong candidate normalization profile"):
                    provenance.validate_stats(self.stats, self.root)
                self.manifest[field] = original

    def test_nonfinite_and_wrong_shape_numeric_data_are_rejected(self):
        self.arrays[0][0][0, 0] = np.nan
        self.write_episode(0)
        with self.assertRaisesRegex(ValueError, "invalid numeric"):
            provenance.validate_stats(self.stats, self.root)
        self.arrays[0] = (np.zeros((2, 7)), self.arrays[0][1], self.arrays[0][2])
        self.write_episode(0)
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            provenance.validate_stats(self.stats, self.root)


if __name__ == "__main__":
    unittest.main()
