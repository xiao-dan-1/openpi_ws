"""Train entrypoint guards and real official save/restore on two emulated CPU devices."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from pipeline.q5.parallelism import gpu_selectors, training_layout
from pipeline.q5.train import require_resume_configuration


class TrainingArgumentsTests(unittest.TestCase):
    def test_steps_require_two_before_environment_preparation(self):
        from pipeline.q5 import train
        for steps in [-1, 0, 1]:
            with self.subTest(steps=steps), \
                 mock.patch.object(sys, "argv", ["train", "--gpu", "0", "--exp-name", "demo", "--steps", str(steps)]), \
                 mock.patch.object(train, "prepare_environment") as environment, \
                 contextlib.redirect_stderr(io.StringIO()) as stderr, \
                 self.assertRaises(SystemExit) as exit:
                train.main()
            self.assertEqual(exit.exception.code, 2)
            self.assertIn("--steps must be at least 2", stderr.getvalue())
            environment.assert_not_called()

    def test_two_steps_reaches_environment_preparation(self):
        from pipeline.q5 import train
        with mock.patch.object(sys, "argv", ["train", "--gpu", "0", "--exp-name", "demo", "--steps", "2"]), \
             mock.patch.object(train, "prepare_environment", side_effect=RuntimeError("environment reached")) as environment:
            with self.assertRaisesRegex(RuntimeError, "environment reached"):
                train.main()
            environment.assert_called_once()

    def test_single_gpu_compatible_and_dual_explicit(self):
        self.assertEqual(training_layout("2", None, 1)["fsdp_devices"], 1)
        self.assertEqual(training_layout("2,3", 2, 16)["per_device_batch"], 8)
        self.assertEqual(gpu_selectors("02,3"), "2,3")
        self.assertEqual(gpu_selectors("GPU-one,GPU-two"), "GPU-one,GPU-two")
        for value in ["", "2,2", "2,02", "0,1,2", "-1", "0,x", "0, 1"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                gpu_selectors(value)

    def test_invalid_layout_rejected_before_assets_or_jax(self):
        cases = [["--gpus", "2,3"], ["--gpus", "2,3", "--fsdp-devices", "1"],
                 ["--gpus", "2,3", "--fsdp-devices", "2", "--batch-size", "3"],
                 ["--gpu", "2,3", "--fsdp-devices", "2"], ["--gpu", "2", "--fsdp-devices", "2"]]
        for args in cases:
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-B", "-S", "-m", "pipeline.q5.train",
                                         "--exp-name", "demo", *args], capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_resume_guards_strategy_but_allows_other_physical_cards(self):
        original = {key: "same" for key in ["profile", "model", "freeze_policy", "learning_rate", "optimizer",
                    "batch_size", "seed", "fsdp_devices", "device_count", "split_manifest_sha256",
                    "norm_stats_sha256", "upstream_commit", "implementation_sha256"]}
        current = dict(original, parallelism={"gpu_selectors": "0,1"}, steps=120)
        require_resume_configuration(original, current)
        for key in original:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                require_resume_configuration(original, dict(current, **{key: "changed"}))

    def test_train_delegates_dual_config_and_records_saved_progress(self):
        import jax
        from pipeline.q5 import config as config_module, train
        from pipeline.q5.runtime import Paths
        with tempfile.TemporaryDirectory() as root:
            paths = Paths.create(root)
            (paths.train / "meta").mkdir(parents=True)
            (paths.train / "meta/info.json").write_text('{"total_frames":17382}')
            paths.split_manifest.write_text("{}")
            with mock.patch.object(train, "prepare_environment", return_value="revision") as environment, \
                 mock.patch.object(train, "require_stats"), \
                 mock.patch.object(config_module, "require_stats", return_value={"sha256": "stats"}), \
                 mock.patch.object(config_module, "require_base_assets"), \
                 mock.patch.object(jax, "default_backend", return_value="gpu"), \
                 mock.patch.object(jax, "device_count", return_value=2), \
                 mock.patch.object(jax, "process_count", return_value=1), \
                 mock.patch.object(jax, "local_devices", return_value=["device0", "device1"]), \
                 mock.patch.object(train, "official_train_module") as official, \
                 mock.patch("pipeline.q5.checkpointing.checkpoint_progress", return_value={"state_step": 100}) as progress, \
                 mock.patch.object(sys, "argv", ["train", "--workspace", root, "--gpus", "2,3",
                            "--fsdp-devices", "2", "--batch-size", "16", "--memory-fraction", "0.75", "--exp-name", "dual"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                train.main()
                config = official.return_value.main.call_args.args[0]
                config.checkpoint_dir.mkdir(parents=True)
                progress.side_effect = [{"state_step": 100}, {"state_step": 120}]
                sys.argv += ["--resume", "--steps", "120"]
                train.main()
                self.assertTrue(official.return_value.main.call_args.args[0].resume)
                calls = official.return_value.main.call_count
                progress.side_effect = None
                progress.return_value = {"state_step": 120}
                with self.assertRaisesRegex(ValueError, "--steps must exceed"):
                    train.main()
                self.assertEqual(official.return_value.main.call_count, calls)
            config = official.return_value.main.call_args.args[0]
            self.assertEqual(config.fsdp_devices, 2)
            self.assertEqual(config.batch_size, 16)
            self.assertTrue(environment.call_args.kwargs["allow_multiple_gpus"])
            self.assertEqual(environment.call_args.kwargs["gpu"], "2,3")
            self.assertTrue(all(call.args == (config.checkpoint_dir, "stats") for call in progress.call_args_list))
            run = paths.workspace / "runs/pi05_q5/dual"
            manifest = json.loads((run / "config.json").read_text())
            self.assertEqual(manifest["parallelism"]["mesh_shape"], {"batch": 1, "fsdp": 2})
            self.assertEqual(manifest["fsdp_devices"], 2)
            result = json.loads((run / "training_result.json").read_text())
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["resume_checkpoint"]["state_step"], 100)
            self.assertEqual(result["final_checkpoint"]["state_step"], 120)


class OfficialCheckpointTests(unittest.TestCase):
    def test_two_cpu_official_training_save_resume_matches_uninterrupted(self):
        # The fixed batch checks parameter/optimizer restoration, not real loader continuity.
        # Separate process: force two CPU devices before the first JAX import.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                   XLA_FLAGS="--xla_force_host_platform_device_count=2")
        result = subprocess.run([sys.executable, "-B", "-m", "tests.test_q5_training", "--checkpoint-child"],
                                env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-7000:])
        self.assertIn("PASS two-CPU official save/restore", result.stdout)


def checkpoint_child():
    import dataclasses
    import logging
    from types import SimpleNamespace
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as model_module
    from openpi.training import checkpoints, sharding
    from openpi.shared import normalize
    from pipeline.q5.config import build_config
    from pipeline.q5.runtime import ASSET_ID, Paths, official_train_module, sha256
    from pipeline.q5.checkpointing import checkpoint_progress

    assert jax.default_backend() == "cpu" and jax.device_count() == 2
    logging.basicConfig(level=logging.WARNING)

    class TinyModel(model_module.BaseModel):
        def __init__(self, rng):
            self.action_dim, self.action_horizon, self.max_token_len = 38, 10, 200
            # 8 MiB: above the official 4 MiB sharding threshold.
            self.kernel = nnx.Param(jax.random.normal(rng, (2048, 1024)) * 0.01)

        def compute_loss(self, rng, observation, actions, *, train=False):
            prediction = self.kernel.value.mean()
            return (prediction + observation.state[:, :1] - actions[:, :, 0]) ** 2

        def sample_actions(self, rng, observation, **kwargs):
            return jnp.broadcast_to(self.kernel.value.mean(), (observation.state.shape[0], 10, 38))

    class TinyConfig:
        def create(self, rng):
            return TinyModel(rng)

    class EmptyWeights:
        def load(self, shape):
            return shape

    class Loader:
        def __init__(self, layout, stats):
            self.stats = stats
            observation = model_module.Observation(images={"base_0_rgb": jnp.zeros((4, 2, 2, 3))},
                image_masks={"base_0_rgb": jnp.ones(4, dtype=bool)}, state=jnp.ones((4, 38)))
            self.batch = jax.device_put((observation, jnp.zeros((4, 10, 38))), layout)

        def __iter__(self):
            while True:
                yield self.batch

        def data_config(self):
            return SimpleNamespace(norm_stats=self.stats, asset_id=ASSET_ID)

    with tempfile.TemporaryDirectory() as root:
        paths = Paths.create(root)
        official_root = Path(__import__("openpi").__file__).resolve().parents[2]
        # Load the actual installed upstream trainer; no modifications to its source.
        official = official_train_module(Paths.create(official_root.parent))
        stats = {key: normalize.NormStats(mean=np.zeros(38), std=np.ones(38),
                    q01=-np.ones(38), q99=np.ones(38)) for key in ["state", "actions"]}
        normalize.save(paths.stats, stats)
        stats_sha = sha256(paths.stats / "norm_stats.json")
        base = dataclasses.replace(build_config(paths, batch_size=4, fsdp_devices=2),
                    model=TinyConfig(), freeze_filter=nnx.Nothing, weight_loader=EmptyWeights(),
                    save_interval=2, keep_period=2)
        mesh = sharding.make_mesh(2)
        loader = Loader(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)), stats)
        empty = Path(root) / "empty"
        empty.mkdir()
        try:
            checkpoint_progress(empty, stats_sha)
        except ValueError as exc:
            assert "No completed" in str(exc)
        else:
            raise AssertionError("Empty resume must not initialize a fresh run")

        def run(name, limit, resume=False):
            config = dataclasses.replace(base, exp_name=name, num_train_steps=limit, resume=resume)
            with mock.patch.object(official._data_loader, "create_data_loader", return_value=loader), \
                 mock.patch.object(official, "init_wandb"), mock.patch.object(official.wandb, "log"), \
                 mock.patch.object(official.wandb, "Image"):
                official.main(config)
            progress = checkpoint_progress(config.checkpoint_dir, stats_sha)
            assert progress["state_step"] == limit and progress["checkpoint_label"] == limit - 1
            shape, _ = official.init_train_state(config, jax.random.key(0), mesh, resume=True)
            manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir,
                        keep_period=2, overwrite=False, resume=True)
            assert resuming
            with manager:
                state = checkpoints.restore_state(manager, shape, loader)
                jax.block_until_ready(state)
            assert int(state.step) == limit
            assert not state.params["kernel"].value.is_fully_replicated
            assert len(state.params["kernel"].value.addressable_shards) == 2
            return config, jax.device_get((state.params, state.opt_state)), progress

        _, _, first = run("split", 4)
        assert all(value == 4 for value in first["optimizer_counts"].values())
        try:
            checkpoint_progress(Path(base.checkpoint_base_dir) / "pi05_q5/split", "wrong_statistics_hash")
        except ValueError as exc:
            assert "statistics mismatch" in str(exc)
        else:
            raise AssertionError("Saved statistics mismatch must fail")
        config, resumed, last = run("split", 6, resume=True)
        _, uninterrupted, _ = run("continuous", 6)
        for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(uninterrupted), strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert all(value == 6 for value in last["optimizer_counts"].values())
        assert float(config.lr_schedule.create()(4)) != float(config.lr_schedule.create()(0))
        # Saved inference weights are independently restorable on a single device.
        params = model_module.restore_params(Path(config.checkpoint_dir) / "5/params", restore_type=np.ndarray)
        np.testing.assert_array_equal(params["kernel"], resumed[0]["kernel"].value)
        print("PASS two-CPU official save/restore: step and Adam/schedule counts, all parameter/moment values, "
              "resumed == uninterrupted, FSDP layout and single-device inference weight restoration")


if __name__ == "__main__":
    if "--checkpoint-child" in sys.argv:
        checkpoint_child()
    else:
        unittest.main()
