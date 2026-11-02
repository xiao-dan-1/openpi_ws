"""Regression coverage for shared runtime, admission checks and single-pass preflight."""
import argparse
from collections import Counter
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from pipeline.common import config as common_config, runtime
from pipeline.common.qc import COUNTERS, fields, require_training_qc
from pipeline.common.tokenization import checked_token_length
from pipeline.q5 import runtime as q5_runtime
from pipeline.q5.benchmark_fsdp import parse_args
from pipeline.q5.parallelism import gpu_selectors
from pipeline.q5.schedule import resolve_schedule


class SharedRuntimeTests(unittest.TestCase):
    def test_libero_and_q5_keep_their_own_environment_settings(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ), \
             mock.patch.object(runtime, "verify_upstream_revision", return_value="reviewed") as revision, \
             mock.patch.object(q5_runtime, "configure_video_backend") as video:
            root = Path(temporary).resolve()
            (root / "datasets/libero/meta").mkdir(parents=True)
            (root / "datasets/libero/meta/info.json").write_text("{}")
            with mock.patch.object(common_config, "WORKSPACE", root):
                common_config.prepare_environment(cpu=True)
            self.assertEqual(os.environ["HF_DATASETS_CACHE"], str(root / ".cache/huggingface/datasets"))
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
            self.assertIn("examples/libero/main.py", revision.call_args.kwargs["protected_paths"])
            self.assertEqual(q5_runtime.prepare_environment(q5_runtime.Paths.create(root), cpu=False,
                             gpu="02,3", allow_multiple_gpus=True, memory_fraction=0.65), "reviewed")
            self.assertEqual(os.environ["HF_DATASETS_CACHE"], str(root / ".cache/huggingface/q5"))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.65")
            self.assertNotIn("JAX_PLATFORMS", os.environ)
            video.assert_called_once()

    def test_both_asset_entrypoints_reject_changed_tokenizer(self):
        from pipeline.q5.config import require_base_assets
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(common_config, "WORKSPACE", Path(temporary)):
            paths = q5_runtime.Paths.create(temporary)
            tokenizer = paths.workspace / "openpi_cache/big_vision/paligemma_tokenizer.model"
            tokenizer.parent.mkdir(parents=True)
            tokenizer.write_bytes(b"changed")
            for entrypoint in [lambda: common_config.require_assets(None, stats=False),
                               lambda: require_base_assets(paths)]:
                with self.assertRaisesRegex(ValueError, "Tokenizer SHA256"):
                    entrypoint()

    def test_official_script_location_checked_for_both_callers(self):
        with mock.patch.object(runtime.importlib, "import_module", return_value=SimpleNamespace(__file__=__file__)):
            for entrypoint in [lambda: common_config.load_official_script("train"),
                               lambda: q5_runtime.official_train_module(q5_runtime.Paths.create())]:
                with self.assertRaisesRegex(ImportError, "Unexpected official script"):
                    entrypoint()

    def test_benchmark_rejects_aliases_and_accepts_same_selectors_as_training(self):
        for value in ["2,02", "2,2", "0,1,2", "0, 1"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                gpu_selectors(value)
            with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args(["--gpus", value, "--fsdp-devices", "2"])
            self.assertEqual(error.exception.code, 2)
        for value in ["02,3", "GPU-one,GPU-two"]:
            parsed = parse_args(["--gpus", value, "--fsdp-devices", "2"])
            self.assertEqual(parsed.gpus, gpu_selectors(value))

    def test_gpu_monitor_resolves_uuid_prefixes_and_rejects_physical_aliases(self):
        from checks.check_gpu_communication import snapshot
        rows = [["0", "GPU-a111", "gpu0", "24000", "0", "0"],
                ["1", "GPU-a222", "gpu1", "24000", "0", "0"]]
        with mock.patch("checks.check_gpu_communication.query_smi", side_effect=[rows, []]):
            self.assertEqual([d["index"] for d in snapshot("GPU-a2,0")["devices"]], ["1", "0"])
        for selectors in ["0,GPU-a111", "GPU-a", "GPU-unknown"]:
            with self.subTest(selectors=selectors), \
                 mock.patch("checks.check_gpu_communication.query_smi", return_value=rows), \
                 self.assertRaisesRegex(ValueError, "resolve distinct GPUs"):
                snapshot(selectors)

    def test_q5_reviewed_shallow_checkout_needs_no_ancestry(self):
        reviewed = next(iter(runtime.SUPPORTED_LOCAL_COMMITS))
        with mock.patch.object(runtime.subprocess, "check_output", side_effect=[reviewed, ""]), \
             mock.patch.object(runtime.subprocess, "run") as ancestry, \
             mock.patch.object(q5_runtime, "configure_video_backend"), mock.patch.dict(os.environ):
            self.assertEqual(q5_runtime.prepare_environment(q5_runtime.Paths.create()), reviewed)
            ancestry.assert_not_called()


class PreflightTests(unittest.TestCase):
    def setUp(self):
        from tests.test_q5_provenance import Q5ProvenanceTests
        self.fixture = Q5ProvenanceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_each_payload_hashed_once_and_stats_fingerprints_still_checked(self):
        paths = self.fixture.paths
        with mock.patch.object(q5_runtime, "sha256", wraps=q5_runtime.sha256) as digest:
            report, manifest = q5_runtime.require_stats(paths, with_manifest=True)
        calls = Counter(call.args[0] for call in digest.call_args_list)
        for role in ["train", "validation", "test"]:
            root = getattr(paths, role)
            for row in manifest[role]:
                self.assertEqual(calls[root / row["data_path"]], 1)
                for asset in row["assets"]:
                    self.assertEqual(calls[root / asset["path"]], 1)
        report["source_parquet_fingerprints"][0]["sha256"] = "changed"
        (paths.stats / "statistics_run.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Statistics training data changed"):
            q5_runtime.require_stats(paths)

    def test_fresh_invocation_does_not_reuse_a_previous_validation(self):
        paths = self.fixture.paths
        q5_runtime.require_stats(paths)
        row = self.fixture.manifest["validation"][0]
        (paths.validation / row["assets"][0]["path"]).write_bytes(b"changed after preflight")
        with self.assertRaisesRegex(ValueError, "Changed validation asset"):
            q5_runtime.require_stats(paths)

    def test_duplicate_payload_paths_are_rejected_in_both_manifests(self):
        paths = self.fixture.paths
        report_path = paths.stats / "statistics_run.json"
        report = json.loads(report_path.read_text())
        report["source_parquet_fingerprints"].append(report["source_parquet_fingerprints"][0])
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Duplicate statistics"):
            q5_runtime.require_stats(paths)
        manifest = self.fixture.manifest
        manifest["train"][1]["data_path"] = manifest["train"][0]["data_path"]
        paths.split_manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Duplicate train Parquet paths"):
            q5_runtime.require_stats(paths)


class AdmissionTests(unittest.TestCase):
    def test_qc_missing_is_unknown_and_rejected_for_training(self):
        complete = dict.fromkeys(COUNTERS, 0)
        complete.update(age_s_max=0.02, age_a_max=0.03)
        self.assertEqual(require_training_qc(complete)["stale_action"], 0)
        for key in complete:
            missing = dict(complete)
            del missing[key]
            self.assertIsNone(fields(missing)[key])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                require_training_qc(missing)
        for invalid in [float("nan"), float("inf"), -1, True, "0", 0.5]:
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "stale_action"):
                require_training_qc(dict(complete, stale_action=invalid))

    def test_schedule_defaults_and_explicit_overrides(self):
        self.assertEqual(resolve_schedule("smoke"), (100, 10, 1000))
        self.assertEqual(resolve_schedule("finetune"), (1000, 1000, 30000))
        self.assertEqual(resolve_schedule("finetune", 120, 0, 500), (120, 0, 500))
        for args in [("other",), ("smoke", 0), ("smoke", 100, -1), ("smoke", 100, 10, 10)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_schedule(*args)


@unittest.skipUnless(importlib.util.find_spec("jax"), "Requires the OpenPI CPU dependency environment")
class OfficialApiTests(unittest.TestCase):
    def test_token_audit_reads_each_splits_own_task_table(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from pipeline.q5.check_config import audit_state_token_lengths
        tokenizer = SimpleNamespace(tokenize=lambda prompt, state: (None, np.ones(len(prompt), dtype=bool)))
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / role for role in ["train", "validation"]]
            for root, prompt in zip(roots, ["pick", "validation prompt is longer than the limit"], strict=True):
                (root / "meta").mkdir(parents=True)
                (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}))
                pq.write_table(pa.table({"observation.state": np.zeros((1, 38)).tolist(), "task_index": [0]}),
                               root / "data.parquet")
            records = [{"data_path": "data.parquet"}]
            self.assertEqual(audit_state_token_lengths(roots[0], records, lambda data: data, tokenizer, 20), (4, 1))
            with self.assertRaisesRegex(ValueError, "token limit"):
                audit_state_token_lengths(roots[1], records, lambda data: data, tokenizer, 20)

    def test_describe_rejects_identity_changes_after_preflight(self):
        from pipeline.q5.config import build_config, describe
        from tests.test_q5_provenance import Q5ProvenanceTests
        fixture = Q5ProvenanceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        paths = fixture.paths
        report = q5_runtime.require_stats(paths)
        config = build_config(paths)
        with mock.patch("pipeline.q5.config.require_stats", side_effect=AssertionError("Repeated payload scan")):
            self.assertEqual(describe(paths, config, "smoke", stats_report=report)["norm_stats_sha256"], report["sha256"])
            for path, error in [(paths.split_manifest, "Split manifest changed"),
                                (paths.stats / "norm_stats.json", "Statistics file changed")]:
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                try:
                    with self.assertRaisesRegex(ValueError, error):
                        describe(paths, config, "smoke", stats_report=report)
                finally:
                    path.write_bytes(original)

    def test_real_data_contract_rejects_missing_qc_without_turning_it_into_zero(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from openpi import transforms
        from openpi.shared.normalize import NormStats
        from pipeline.q5.data_contract import check_data_contract
        from pipeline.q5.schema import JOINT_NAMES
        qc = dict.fromkeys(COUNTERS, 0)
        qc.update(age_s_max=0.02, age_a_max=0.03)
        stats = NormStats(mean=np.zeros(38), std=np.ones(38), q01=-np.ones(38), q99=np.ones(38))
        data = SimpleNamespace(norm_stats={"state": stats, "actions": stats},
                               data_transforms=transforms.Group(), model_transforms=transforms.Group())
        with tempfile.TemporaryDirectory() as temporary:
            paths, manifest = q5_runtime.Paths.create(temporary), {}
            for role in ["train", "validation"]:
                root = getattr(paths, role)
                (root / "meta/timing").mkdir(parents=True)
                (root / "meta/info.json").write_text(json.dumps({"fps": 30, "features": {
                    key: {"names": list(JOINT_NAMES)} for key in ["action", "observation.state"]}}))
                pq.write_table(pa.table({"observation.state": np.zeros((2, 38)).tolist(),
                    "action": np.zeros((2, 38)).tolist(), "timestamp": [0, 1/30], "frame_index": [0, 1]}), root / "data.parquet")
                (root / "meta/timing/episode_000000.json").write_text(json.dumps({
                    "relative_image_stamps": [0, 1/30], "qc": qc}))
                manifest[role] = [{"output_episode_index": 0, "length": 2, "data_path": "data.parquet"}]
            self.assertEqual(check_data_contract(paths, data, manifest)["status"], "PASS")
            sidecar = paths.train / "meta/timing/episode_000000.json"
            timing = json.loads(sidecar.read_text())
            del timing["qc"]["stale_action"]
            sidecar.write_text(json.dumps(timing))
            with self.assertRaisesRegex(ValueError, "QC field: stale_action"):
                check_data_contract(paths, data, manifest)

    def test_official_tokenizer_detects_boundary_without_copying_encoding_rules(self):
        import numpy as np
        import sentencepiece
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.models import tokenizer as official
        # Real SentencePiece model, all I/O confined to the temporary fixture.
        buffer = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=iter(["Task: pick orange, State: 0 1 127 128 255; Action:"] * 10),
            model_writer=buffer, vocab_size=100, model_type="char", hard_vocab_limit=False, minloglevel=2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokenizer.model"
            path.write_bytes(buffer.getvalue())
            with mock.patch.object(official.download, "maybe_download", return_value=path):
                prompt, state = " pick_orange\n", np.linspace(-1, 1, 38)
                _, mask = PaligemmaTokenizer(max_len=10000).tokenize(prompt, state)
                actual = int(mask.sum())
                for limit in [actual, actual + 1]:
                    self.assertEqual(checked_token_length(PaligemmaTokenizer(max_len=limit + 1),
                                     prompt, state, limit), actual)
                for limit in [actual - 1, actual - 10]:
                    with self.assertRaisesRegex(ValueError, "exceeds the token limit"):
                        checked_token_length(PaligemmaTokenizer(max_len=limit + 1), prompt, state, limit)

    def test_audit_uses_the_upstream_config_filter_and_schedule(self):
        import flax.nnx as nnx
        from openpi.training.config import TrainConfig
        from pipeline.q5.config import build_config
        with mock.patch.object(TrainConfig, "trainable_filter", new_callable=mock.PropertyMock, return_value=nnx.Nothing):
            config = build_config(q5_runtime.Paths.create(tempfile.gettempdir()), profile="finetune",
                                  steps=120, warmup_steps=0, schedule_steps=500)
        self.assertIs(config.weight_loader.trainable_filter, nnx.Nothing)
        self.assertEqual((config.num_train_steps, config.lr_schedule.warmup_steps, config.lr_schedule.decay_steps),
                         resolve_schedule("finetune", 120, 0, 500))


if __name__ == "__main__":
    unittest.main()
