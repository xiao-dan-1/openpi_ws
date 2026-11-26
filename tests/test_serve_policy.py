"""Shared serving aligns checkpoint selection with upstream (--policy.config / --policy.dir)."""
import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pipeline.common import config as libero_config
from pipeline.inference import serve_policy
from pipeline.q5 import config as q5_config


@dataclasses.dataclass
class Model:
    action_dim: int = 38
    action_horizon: int = 10


class ServePolicyTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.stack.enter_context(mock.patch.dict(os.environ))
        self.stack.enter_context(mock.patch.object(serve_policy, "WORKSPACE", self.root))
        self.verify_revision = serve_policy.runtime.verify_upstream_revision
        self.revision = self.stack.enter_context(mock.patch.object(serve_policy.runtime, "verify_upstream_revision"))
        self.environment = self.stack.enter_context(mock.patch.object(serve_policy.runtime, "prepare_environment"))
        self.stack.enter_context(mock.patch.object(serve_policy.runtime, "require_tokenizer"))
        self.q5_builder = self.stack.enter_context(mock.patch.object(q5_config, "build_config"))
        self.libero_builder = self.stack.enter_context(mock.patch.object(libero_config, "build_config"))
        self.describe = self.stack.enter_context(mock.patch.object(libero_config, "describe"))
        self.report = self.stack.enter_context(mock.patch.object(serve_policy, "emit_report"))
        self.create_policy = mock.Mock(return_value=SimpleNamespace(metadata={"robot_type": "q5"}))
        self.server = mock.Mock()
        self.stack.enter_context(mock.patch.dict(sys.modules, {
            "jax": SimpleNamespace(default_backend=lambda: "gpu", device_count=lambda: 1),
            "openpi.policies.policy_config": SimpleNamespace(create_trained_policy=self.create_policy),
            "openpi.serving.websocket_policy_server": SimpleNamespace(WebsocketPolicyServer=self.server),
        }))

    def fixture(self, policy_config="pi05_q5", *, with_record=True, checkpoint_path=None):
        is_q5 = policy_config == "pi05_q5"
        self.checkpoint = (
            self.root / f"checkpoints/{policy_config}/demo/15999"
            if checkpoint_path is None else Path(checkpoint_path)
        )
        asset_id = "q5_pick_orange_to_basket_v2_train" if is_q5 else "libero"
        self.stats = self.checkpoint / "assets" / asset_id / "norm_stats.json"
        self.stats.parent.mkdir(parents=True, exist_ok=True)
        self.stats.write_text('{"checkpoint_owned":true}')
        (self.checkpoint / "params").mkdir(exist_ok=True)
        self.metadata = self.checkpoint / "_CHECKPOINT_METADATA"
        self.metadata.write_text('{"commit_timestamp_nsecs":1}')
        model = Model(38 if is_q5 else 32)
        self.config = SimpleNamespace(model=model, checkpoint_dir=self.checkpoint.parent)
        self.record_path = self.root / f"runs/{policy_config}/demo/config.json"
        self.record = {
            "profile": "finetune", "batch_size": 16, "model": dataclasses.asdict(model),
            "norm_stats_sha256": hashlib.sha256(self.stats.read_bytes()).hexdigest(),
            "upstream_commit" if is_q5 else "upstream_commit_expected": serve_policy.runtime.UPSTREAM_COMMIT,
        }
        if with_record:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.record_path.write_text(json.dumps(self.record))
        self.q5_builder.return_value = self.libero_builder.return_value = self.config
        self.describe.return_value = {"model": self.record["model"]}

    def invoke(self, *extra, policy_config="pi05_q5", policy_dir=None):
        target = f"checkpoints/{policy_config}/demo/15999" if policy_dir is None else str(policy_dir)
        argv = ["serve_policy", "--gpu", "0", "--policy.config", policy_config, "--policy.dir", target, *extra]
        with mock.patch.object(sys, "argv", argv):
            serve_policy.main()

    def test_q5_uses_38d_config_without_any_training_dataset_or_base_weights(self):
        self.fixture("pi05_q5")
        original = self.record_path.read_bytes()
        self.invoke("--memory-fraction", "0.6", "--host", "0.0.0.0", "--port", "8111")
        self.libero_builder.assert_not_called()
        self.assertEqual(self.q5_builder.call_args.args[0].workspace, self.root)
        self.create_policy.assert_called_once_with(self.config, self.checkpoint)
        self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.6")
        kwargs = self.server.call_args.kwargs
        self.assertEqual((kwargs["host"], kwargs["port"]), ("0.0.0.0", 8111))
        self.assertEqual(kwargs["metadata"]["pipeline"]["model"]["action_dim"], 38)
        self.server.return_value.serve_forever.assert_called_once()
        self.assertEqual(self.record_path.read_bytes(), original)
        self.assertFalse((self.root / "datasets").exists())
        self.assertFalse((self.root / "openpi_cache").exists())
        self.assertIsNone(self.report.call_args.args[1])

    def test_libero_defaults_to_all_interfaces_and_preserves_pipeline_identity(self):
        self.fixture("pi05_libero")
        self.invoke(policy_config="pi05_libero")
        self.q5_builder.assert_not_called()
        self.libero_builder.assert_called_once()
        identity = self.server.call_args.kwargs["metadata"]["pipeline"]
        self.assertEqual(identity, {
            "experiment": "demo", "checkpoint_step": 15999,
            "upstream_commit": serve_policy.runtime.UPSTREAM_COMMIT,
            "model": self.record["model"],
            "norm_stats_sha256": self.record["norm_stats_sha256"],
        })
        self.assertEqual(self.server.call_args.kwargs["host"], "0.0.0.0")

    def test_standalone_checkpoint_dir_loads_without_runs_record(self):
        for policy_config in ("pi05_libero", "pi05_q5"):
            with self.subTest(policy_config=policy_config):
                self.create_policy.reset_mock()
                self.server.reset_mock()
                standalone = self.root / f"imported/{policy_config}_ckpt"
                self.fixture(policy_config, with_record=False, checkpoint_path=standalone)
                self.invoke(policy_config=policy_config, policy_dir=standalone)
                self.create_policy.assert_called_once_with(self.config, standalone)
                identity = self.server.call_args.kwargs["metadata"]["pipeline"]
                self.assertEqual(identity["config"], policy_config)
                self.assertEqual(identity["checkpoint_dir"], str(standalone))
                self.assertEqual(identity["norm_stats_sha256"], self.record["norm_stats_sha256"])

    def test_statistics_or_uncommitted_checkpoint_rejected_before_model_load(self):
        self.fixture("pi05_q5")
        for change in ["statistics", "uncommitted", "missing_params"]:
            with self.subTest(change=change):
                if change == "statistics":
                    self.stats.write_text("changed")
                else:
                    self.stats.write_text('{"checkpoint_owned":true}')
                    self.metadata.write_text("{}" if change == "uncommitted" else '{"commit_timestamp_nsecs":1}')
                if change == "missing_params":
                    (self.checkpoint / "params").rmdir()
                with self.assertRaisesRegex(ValueError, "statistics|finalized"):
                    self.invoke()
                self.create_policy.assert_not_called()
                self.server.assert_not_called()

    def test_wrong_config_for_checkpoint_assets_rejected_before_model_load(self):
        self.fixture("pi05_q5")
        with self.assertRaisesRegex(ValueError, "Missing checkpoint normalization statistics"):
            self.invoke(policy_config="pi05_libero", policy_dir=self.checkpoint)
        self.create_policy.assert_not_called()

    def test_recorded_model_and_upstream_changes_rejected(self):
        self.fixture("pi05_q5")
        for field in ["model", "upstream_commit"]:
            changed = dict(self.record)
            changed[field] = {} if field == "model" else "unknown"
            self.record_path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "model|upstream"):
                self.invoke()
            self.create_policy.assert_not_called()

    def test_q5_accepts_all_reviewed_training_revisions(self):
        self.fixture("pi05_q5")
        for revision in [serve_policy.runtime.UPSTREAM_COMMIT, *sorted(serve_policy.runtime.SUPPORTED_LOCAL_COMMITS)]:
            with self.subTest(revision=revision):
                self.record_path.write_text(json.dumps({**self.record, "upstream_commit": revision}))
                self.invoke()
                self.assertEqual(self.create_policy.call_args.args, (self.config, self.checkpoint))

    def test_q5_reviewed_shallow_checkout_does_not_require_ancestry(self):
        self.fixture("pi05_q5")
        revision = sorted(serve_policy.runtime.SUPPORTED_LOCAL_COMMITS)[0]
        self.record_path.write_text(json.dumps({**self.record, "upstream_commit": revision}))
        self.revision.side_effect = self.verify_revision
        with mock.patch.object(serve_policy.runtime.subprocess, "check_output", side_effect=[revision, ""]), \
             mock.patch.object(serve_policy.runtime.subprocess, "run") as ancestry:
            self.invoke()
        ancestry.assert_not_called()
        self.create_policy.assert_called_once_with(self.config, self.checkpoint)

    def test_q5_still_rejects_unknown_or_dirty_runtime(self):
        self.fixture("pi05_q5")
        self.revision.side_effect = self.verify_revision
        for outputs, message in [(["unknown"], "Expected OpenPI"),
                                 ([sorted(serve_policy.runtime.SUPPORTED_LOCAL_COMMITS)[0],
                                   "src/openpi/policies/policy.py"], "Unreviewed changes")]:
            with self.subTest(outputs=outputs), \
                 mock.patch.object(serve_policy.runtime.subprocess, "check_output", side_effect=outputs), \
                 self.assertRaisesRegex(ValueError, message):
                self.invoke()
            self.environment.assert_not_called()
            self.create_policy.assert_not_called()

    def test_libero_still_requires_reviewed_runtime_ancestry(self):
        self.fixture("pi05_libero")
        self.revision.side_effect = self.verify_revision
        revision = sorted(serve_policy.runtime.SUPPORTED_LOCAL_COMMITS)[0]
        with mock.patch.object(serve_policy.runtime.subprocess, "check_output", return_value=revision), \
             mock.patch.object(serve_policy.runtime.subprocess, "run",
                               side_effect=subprocess.CalledProcessError(1, "git merge-base")) as ancestry, \
             self.assertRaises(subprocess.CalledProcessError):
            self.invoke(policy_config="pi05_libero")
        ancestry.assert_called_once()
        self.environment.assert_not_called()
        self.create_policy.assert_not_called()

    def test_invalid_or_legacy_arguments_rejected_before_environment_changes(self):
        bad_argvs = [
            ["--gpu", "0,1", "--policy.config", "pi05_q5", "--policy.dir", "checkpoints/pi05_q5/demo/15999"],
            ["--gpu", "bad", "--policy.config", "pi05_q5", "--policy.dir", "checkpoints/pi05_q5/demo/15999"],
            ["--gpu", "0", "--policy.config", "unknown", "--policy.dir", "checkpoints/pi05_q5/demo/15999"],
            ["--gpu", "0", "--policy.config", "pi05_q5", "--policy.dir", "checkpoints/pi05_q5/demo/15999", "--port", "0"],
            ["--gpu", "0", "--policy.config", "pi05_q5", "--policy.dir", "checkpoints/pi05_q5/demo/15999", "--memory-fraction", "nan"],
            ["--gpu", "0", "--policy.config", "pi05_q5"],
            ["--gpu", "0", "--policy.dir", "checkpoints/pi05_q5/demo/15999"],
            ["--gpu", "0", "--exp-name", "demo", "--step", "15999"],
            ["--gpu", "0", "--checkpoint", "checkpoints/pi05_q5/demo/15999"],
        ]
        for argv in bad_argvs:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                 mock.patch.object(sys, "argv", ["serve_policy", *argv]), \
                 self.assertRaises(SystemExit):
                serve_policy.main()
            self.environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
