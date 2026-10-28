"""Validate explicit allocator configuration without GPU initialization."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from pipeline.q5.memory import add_memory_argument, configure_memory_fraction, memory_fraction, memory_settings


class Q5MemoryTests(unittest.TestCase):
    def test_default_and_explicit_cli_budget(self):
        parser = argparse.ArgumentParser()
        add_memory_argument(parser)
        self.assertEqual(parser.parse_args([]).memory_fraction, 0.75)
        self.assertEqual(parser.parse_args(["--memory-fraction", "0.65"]).memory_fraction, 0.65)

    def test_invalid_values_are_rejected(self):
        for value in ["nan", "inf", "-0.1", "0", "1", "2", "oops", None]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                memory_fraction(value)

    def test_explicit_budget_replaces_both_environment_spellings_only(self):
        with mock.patch.dict(os.environ, {"XLA_CLIENT_MEM_FRACTION": "0.9",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}):
            configure_memory_fraction(0.65)
            self.assertNotIn("XLA_CLIENT_MEM_FRACTION", os.environ)
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.65")
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "true")
            self.assertEqual(memory_settings()["memory_fraction"], 0.65)

    def test_runtime_applies_requested_budget_for_both_cards(self):
        from pipeline.q5 import runtime
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ), \
             mock.patch.object(runtime.subprocess, "check_output", side_effect=[runtime.UPSTREAM_COMMIT, ""]), \
             mock.patch.object(runtime, "configure_video_backend"):
            runtime.prepare_environment(runtime.Paths.create(directory), cpu=False, gpu="1,2",
                                        allow_multiple_gpus=True, memory_fraction=0.65)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1,2")
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.65")
            self.assertEqual(memory_settings()["memory_fraction"], 0.65)

    def test_all_gpu_entrypoints_reject_bad_fraction_before_loading_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        modules = [("pipeline.q5.benchmark_fsdp", ["--gpus", "1,2", "--fsdp-devices", "2"]),
                   ("pipeline.q5.train", ["--gpu", "2", "--exp-name", "demo"]),
                   ("pipeline.q5.check_model", ["--gpu", "2"]),
                   ("pipeline.q5.evaluate_validation", ["--gpu", "2", "--exp-name", "demo", "--step", "119"])]
        for module, args in modules:
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-B", "-S", "-m", module, *args, "--memory-fraction", "nan"],
                                        cwd=root, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("memory-fraction", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
