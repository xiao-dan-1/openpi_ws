"""Shared serving routes robots explicitly and uses checkpoint-owned assets."""
import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
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
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(mock.patch.dict(os.environ))
        self.stack.enter_context(mock.patch.object(serve_policy, "WORKSPACE", self.root))
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

    def fixture(self, robot="q5"):
        self.record_path = self.root / f"runs/pi05_{robot}/demo/config.json"
        self.record_path.parent.mkdir(parents=True)
        self.checkpoint = self.root / f"checkpoints/pi05_{robot}/demo/15999"
        asset_id = "q5_pick_orange_to_basket_v2_train" if robot == "q5" else "libero"
        self.stats = self.checkpoint / "assets" / asset_id / "norm_stats.json"
        self.stats.parent.mkdir(parents=True)
        self.stats.write_text('{"checkpoint_owned":true}')
        (self.checkpoint / "params").mkdir()
        self.metadata = self.checkpoint / "_CHECKPOINT_METADATA"
        self.metadata.write_text('{"commit_timestamp_nsecs":1}')
        model = Model(38 if robot == "q5" else 32)
        self.config = SimpleNamespace(model=model, checkpoint_dir=self.checkpoint.parent)
        self.record = {"profile": "finetune", "batch_size": 16, "model": dataclasses.asdict(model),
                       "norm_stats_sha256": hashlib.sha256(self.stats.read_bytes()).hexdigest(),
                       "upstream_commit" if robot == "q5" else "upstream_commit_expected":
                           serve_policy.runtime.UPSTREAM_COMMIT}
        self.record_path.write_text(json.dumps(self.record))
        self.q5_builder.return_value = self.libero_builder.return_value = self.config
        self.describe.return_value = {"model": self.record["model"]}

    def invoke(self, *extra, robot="q5"):
        arguments = ["serve_policy", "--gpu", "0", "--exp-name", "demo", "--step", "15999"]
        if robot is not None:
            arguments += ["--robot", robot]
        with mock.patch.object(sys, "argv", [*arguments, *extra]):
            serve_policy.main()

    def test_q5_uses_38d_config_without_any_training_dataset_or_base_weights(self):
        self.fixture()
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

    def test_legacy_command_defaults_to_libero_and_preserves_pipeline_identity(self):
        self.fixture("libero")
        self.invoke(robot=None)
        self.q5_builder.assert_not_called()
        self.libero_builder.assert_called_once()
        identity = self.server.call_args.kwargs["metadata"]["pipeline"]
        self.assertEqual(identity, {"experiment": "demo", "checkpoint_step": 15999,
                                   "upstream_commit": serve_policy.runtime.UPSTREAM_COMMIT,
                                   "model": self.record["model"],
                                   "norm_stats_sha256": self.record["norm_stats_sha256"]})
        self.assertEqual(self.server.call_args.kwargs["host"], "127.0.0.1")

    def test_statistics_or_uncommitted_checkpoint_rejected_before_model_load(self):
        self.fixture()
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

    def test_recorded_model_and_upstream_changes_rejected(self):
        self.fixture()
        for field in ["model", "upstream_commit"]:
            changed = dict(self.record)
            changed[field] = {} if field == "model" else "unknown"
            self.record_path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "model|upstream"):
                self.invoke()
            self.create_policy.assert_not_called()

    def test_wrong_robot_reports_missing_record_before_any_model_load(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, "Missing libero training record.*--robot"):
            self.invoke(robot=None)
        self.create_policy.assert_not_called()

    def test_invalid_arguments_rejected_before_environment_changes(self):
        for extra in [["--gpu", "0,1"], ["--gpu", "bad"], ["--robot", "unknown"],
                      ["--port", "0"], ["--memory-fraction", "nan"]]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                self.invoke(*extra)
            self.environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
