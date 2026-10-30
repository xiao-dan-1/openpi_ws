"""Exercise entrypoint safeguards without loading GPU or model dependencies."""
import contextlib
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pipeline.common import config as common_config
from pipeline.inference import local_docker_serve, policy_rpc
from pipeline.train import run as train


@dataclasses.dataclass
class Schedule:
    peak_lr: float = 5e-5


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.run_dir = self.root / "runs/pi05_libero/demo"
        self.run_dir.mkdir(parents=True)
        self.manifest = self.run_dir / "config.json"
        self.config = SimpleNamespace(
            name="pi05_libero", checkpoint_dir=self.root / "checkpoints/pi05_libero/demo",
            assets_dirs=self.root / "assets/pi05_libero", batch_size=16, num_workers=0,
            num_train_steps=30000, ema_decay=None, lr_schedule=Schedule(), resume=False,
            model=SimpleNamespace(pi05=True, paligemma_variant="gemma_2b_lora",
                                  action_expert_variant="gemma_300m_lora", action_horizon=10,
                                  action_dim=32, max_token_len=200, discrete_state_input=False),
        )
        self.stack.enter_context(patch.object(common_config, "WORKSPACE", self.root))
        self.stack.enter_context(patch.object(common_config, "verify_upstream_revision",
                                             return_value=common_config.UPSTREAM_COMMIT))
        self.jax = SimpleNamespace(default_backend=lambda: "gpu", device_count=lambda: 1)

    def patch(self, target, name, **kwargs):
        return self.stack.enter_context(patch.object(target, name, **kwargs))


class LocalInferenceManifestTest(RuntimeFixture):
    def setUp(self):
        super().setUp()
        self.digest = "7e19abf21b2f8837e1e27c468f4021efe359e38a9f6158acb83b6f978dad6b5a"
        self.checkpoint = self.config.checkpoint_dir / "999"
        self.stats = self.checkpoint / "assets/libero/norm_stats.json"
        self.stats.parent.mkdir(parents=True)
        self.stats.write_text("checkpoint-owned statistics")
        (self.checkpoint / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs":1}')
        self.create_policy = Mock()
        self.server = Mock()
        self.ts_utils = SimpleNamespace(get_ts_context=Mock())
        self.orbax = SimpleNamespace(PyTreeCheckpointer=Mock())
        self.stack.enter_context(patch.dict(sys.modules, {
            "jax": self.jax, "orbax": SimpleNamespace(checkpoint=self.orbax),
            "orbax.checkpoint": self.orbax,
            "orbax.checkpoint._src.serialization": SimpleNamespace(tensorstore_utils=self.ts_utils),
            "openpi.policies.policy_config": SimpleNamespace(create_trained_policy=self.create_policy),
            "openpi.serving.websocket_policy_server": SimpleNamespace(WebsocketPolicyServer=self.server),
        }))
        self.patch(sys, "argv", new=["local_docker_serve", "--exp-name", "demo"])
        self.patch(local_docker_serve, "WORKSPACE", new=self.root)
        self.patch(local_docker_serve, "prepare_environment")
        self.patch(local_docker_serve, "require_assets")
        self.patch(local_docker_serve, "build_config", return_value=self.config)
        self.patch(local_docker_serve, "emit_report")
        # Simulate the already-validated checkpoint asset; describe still hashes
        # workspace files normally, so differing/missing workspace stats matter.
        self.patch(local_docker_serve, "hashlib", new=SimpleNamespace(
            sha256=Mock(return_value=SimpleNamespace(hexdigest=lambda: self.digest))))
        self.stack.enter_context(patch.dict("os.environ"))

    def training_record(self):
        record = common_config.describe(self.config, "smoke")
        record.update(norm_stats_sha256=self.digest, batch_size=1, workers=4,
                      steps=120, learning_rate={"peak_lr": 1e-5}, extra_evidence={"preserve": True})
        return record

    def test_existing_training_record_is_preserved_byte_for_byte(self):
        original = json.dumps(self.training_record(), indent=4).encode() + b"\n"
        self.manifest.write_bytes(original)
        local_docker_serve.main()
        self.assertEqual(self.manifest.read_bytes(), original)
        self.create_policy.assert_called_once_with(self.config, self.checkpoint)
        self.server.return_value.serve_forever.assert_called_once()

    def test_incompatible_records_fail_before_loading_and_remain_unchanged(self):
        for field, value in [("model", {"pi05": False}), ("norm_stats_sha256", "different"),
                             ("upstream_commit_expected", "unknown")]:
            with self.subTest(field=field):
                record = self.training_record()
                record[field] = value
                original = json.dumps(record).encode()
                self.manifest.write_bytes(original)
                with self.assertRaisesRegex(ValueError, field):
                    local_docker_serve.main()
                self.assertEqual(self.manifest.read_bytes(), original)
                self.create_policy.assert_not_called()
                self.server.assert_not_called()

    def test_import_manifest_uses_checkpoint_stats_and_passes_client_verification(self):
        workspace_stats = self.config.assets_dirs / "libero/norm_stats.json"
        for has_workspace_stats in [False, True]:
            with self.subTest(has_workspace_stats=has_workspace_stats):
                if has_workspace_stats:
                    workspace_stats.parent.mkdir(parents=True)
                    workspace_stats.write_text("different workspace statistics")
                local_docker_serve.main()
                record = json.loads(self.manifest.read_text())
                self.assertEqual(record["norm_stats_sha256"], self.digest)
                self.assertEqual(record["stats_path"], str(self.stats))
                metadata = self.server.call_args.kwargs["metadata"]
                identity = policy_rpc.verify_server_metadata(metadata, "demo", 999, workspace=self.root)
                self.assertEqual(identity["norm_stats_sha256"], self.digest)
                self.manifest.unlink()


class TrainingResumeTest(RuntimeFixture):
    def setUp(self):
        super().setUp()
        self.config.resume = True
        self.config.checkpoint_dir.mkdir(parents=True)
        self.manager = Mock()
        self.manager.__enter__ = Mock(return_value=self.manager)
        self.manager.__exit__ = Mock(return_value=False)
        self.orbax = SimpleNamespace(CheckpointManager=Mock(return_value=self.manager),
                                    CheckpointManagerOptions=Mock())
        self.stack.enter_context(patch.dict(sys.modules, {
            "jax": self.jax, "orbax": SimpleNamespace(checkpoint=self.orbax),
            "orbax.checkpoint": self.orbax,
        }))
        self.patch(sys, "argv", new=["train", "--gpu", "0", "--exp-name", "demo", "--resume"])
        self.patch(train, "WORKSPACE", new=self.root)
        self.patch(train, "prepare_environment")
        self.patch(train, "require_assets")
        self.patch(train, "build_config", return_value=self.config)
        self.official = self.patch(train, "load_official_script")
        self.write_manifest = self.patch(train, "write_manifest")
        self.manifest.write_text(json.dumps(common_config.describe(self.config, "smoke")))

    def test_empty_or_zero_only_checkpoint_refuses_resume(self):
        for steps in [(), (0,)]:
            with self.subTest(steps=steps):
                for step in steps:
                    (self.config.checkpoint_dir / str(step)).mkdir()
                self.manager.latest_step.return_value = max(steps) if steps else None
                with self.assertRaisesRegex(ValueError, "No completed nonzero checkpoint"):
                    train.main()
                self.official.assert_not_called()
                self.write_manifest.assert_not_called()
                self.manager.__exit__.assert_called()

    def test_uncommitted_numbered_directory_cannot_enable_resume(self):
        (self.config.checkpoint_dir / "999").mkdir()
        self.manager.latest_step.return_value = None
        with self.assertRaisesRegex(ValueError, "No completed nonzero checkpoint"):
            train.main()
        self.official.assert_not_called()
        self.write_manifest.assert_not_called()

    def test_committed_nonzero_checkpoint_launches_trainer_read_only(self):
        self.manager.latest_step.return_value = 50
        train.main()
        self.orbax.CheckpointManagerOptions.assert_called_once_with(read_only=True, create=False)
        self.orbax.CheckpointManager.assert_called_once_with(
            self.config.checkpoint_dir, options=self.orbax.CheckpointManagerOptions.return_value)
        self.manager.__exit__.assert_called_once()
        self.official.return_value.main.assert_called_once_with(self.config)
        self.write_manifest.assert_called_once_with(self.config, "smoke", self.run_dir / "resume_config.json")


if __name__ == "__main__":
    unittest.main()
