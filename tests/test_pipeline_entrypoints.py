"""CPU-only regression checks for package entrypoints and workspace isolation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from pipeline.common.validation import experiment_name


class PipelineEntrypointsTest(unittest.TestCase):
    def run_module(self, module, *arguments, cwd=WORKSPACE, env=None):
        return subprocess.run(
            [sys.executable, "-B", "-m", module, *arguments],
            cwd=cwd, env=env, text=True, capture_output=True, timeout=20,
        )

    def test_all_cli_help_without_model_or_dataset(self):
        # -S disables site packages: even the simulator CLI must stay usable
        # without importing JAX, torch, OpenPI, or LIBERO at module import time.
        modules = [
            "pipeline.train.run", "pipeline.train.benchmark_batch",
            "pipeline.inference.serve_policy", "pipeline.inference.local_docker_serve",
            "pipeline.inference.check_policy", "pipeline.eval.eval_libero",
            "pipeline.eval.check_simulation", "pipeline.data.compute_norm_stats",
            "pipeline.data.check_config",
            "pipeline.q5.prepare_data", "pipeline.q5.check_config", "pipeline.q5.check_model", "pipeline.q5.evaluate_validation", "pipeline.q5.train", "pipeline.q5.benchmark_fsdp",
        ]
        with tempfile.TemporaryDirectory() as empty:
            env = dict(os.environ, PYTHONPATH=str(WORKSPACE))
            for module in modules:
                with self.subTest(module=module):
                    result = subprocess.run(
                        [sys.executable, "-B", "-S", "-m", module, "--help"],
                        cwd=empty, env=env, capture_output=True, text=True, timeout=20,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_workspace_resolution_outside_project(self):
        with tempfile.TemporaryDirectory() as empty:
            code = (
                "import sys; from pipeline.common.config import WORKSPACE; "
                "from pipeline.common.validation import experiment_name; "
                "assert 'pipeline.train.run' not in sys.modules; "
                "assert 'jax' not in sys.modules; print(WORKSPACE)"
            )
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-c", code], cwd=empty,
                env=dict(os.environ, PYTHONPATH=str(WORKSPACE)),
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()).resolve(), WORKSPACE)

    def test_invalid_cli_arguments_fail_before_loading_assets(self):
        cases = [
            ("pipeline.train.run", ["--gpu", "0", "--exp-name", "../escape"]),
            ("pipeline.q5.train", ["--gpu", "0", "--exp-name", "../escape"]),
            ("pipeline.q5.train", ["--gpu", "0", "--exp-name", "demo", "--batch-size", "0"]),
            ("pipeline.q5.train", ["--gpu", "0", "--exp-name", "demo", "--profile", "finetune", "--steps", "1000"]),
            ("pipeline.q5.check_model", ["--gpu", "0", "--updates", "1"]),
            ("pipeline.q5.benchmark_fsdp", ["--gpus", "0", "--fsdp-devices", "1", "--num-workers", "-1"]),
            ("pipeline.q5.benchmark_fsdp", ["--gpus", "0", "--fsdp-devices", "1", "--timing-mode", "unknown"]),
            ("pipeline.inference.serve_policy", ["--gpu", "0", "--exp-name", "demo", "--step", "-1"]),
            ("pipeline.inference.local_docker_serve", ["--memory-fraction", "1"]),
            ("pipeline.eval.eval_libero", ["--exp-name", "demo", "--step", "1", "--trials", "0"]),
            ("pipeline.data.compute_norm_stats", ["--batch-size", "0"]),
            ("pipeline.train.benchmark_batch", ["--gpu", "0", "--num-workers", "-1", "--plan"]),
        ]
        for module, arguments in cases:
            with self.subTest(module=module):
                result = self.run_module(module, *arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_benchmark_plan_preserves_arguments(self):
        result = self.run_module(
            "pipeline.train.benchmark_batch", "--gpu", "0", "--plan",
            "--batches", "8", "2", "8", "--num-workers", "2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["batches"], [2, 8])
        self.assertEqual(plan["num_workers"], 2)
        self.assertFalse(plan["checkpoint_saving"])

    def test_shared_experiment_name_rejects_path_traversal(self):
        self.assertEqual(experiment_name("libero_lora_01-test"), "libero_lora_01-test")
        for value in ["", ".", "..", "../demo", "demo/test", "demo\\test", "/tmp/demo", "a b"]:
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    experiment_name(value)


if __name__ == "__main__":
    unittest.main()
