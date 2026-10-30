"""Two emulated CPU devices validate collectives; never use local GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from checks.check_gpu_communication import corrected_bus_bytes, gpu_pair, gpu_uuids, positive_size, parse_worker_output


class CommunicationTests(unittest.TestCase):
    def test_native_logs_before_and_after_result_are_preserved(self):
        output = ('NCCL INFO init\nGPU_COMM_RESULT_JSON={"status":"PASS","results":[]}\n'
                  'NCCL INFO Channel 00 via SHM/direct/direct\nNCCL INFO finished\n')
        report, diagnostics = parse_worker_output(output)
        self.assertEqual(report, {"status": "PASS", "results": []})
        self.assertIn("init", diagnostics)
        self.assertIn("SHM/direct/direct", diagnostics)
        self.assertIn("finished", diagnostics)

    def test_missing_or_truncated_result_is_rejected(self):
        for output in ['NCCL INFO only', 'GPU_COMM_RESULT_JSON={"status":', 'GPU_COMM_RESULT_JSON={"status":"FAIL"}']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_worker_output(output)

    def test_invalid_gpu_pairs_and_sizes(self):
        for pair in ["0", "0,0", "0,00", "0,1,2", "a,1", "0,-1"]:
            with self.subTest(pair=pair), self.assertRaises(argparse.ArgumentTypeError):
                gpu_pair(pair)
        for size in ["0", "-1", "nan", "inf", "129"]:
            with self.subTest(size=size), self.assertRaises(argparse.ArgumentTypeError):
                positive_size(size)

    def test_physical_indices_resolve_to_uuid_in_requested_order(self):
        devices = [{"index": "0", "uuid": "GPU-first"}, {"index": "2", "uuid": "GPU-third"}]
        self.assertEqual(gpu_pair("02,0"), "2,0")
        self.assertEqual(gpu_uuids("2,0", devices), "GPU-third,GPU-first")
        for uuids in [("N/A", "GPU-third"), ("GPU-first", "GPU-first")]:
            invalid = [dict(index=str(index), uuid=uuid) for index, uuid in zip([0, 2], uuids)]
            with self.subTest(uuids=uuids), self.assertRaisesRegex(ValueError, "GPU UUIDs"):
                gpu_uuids("0,2", invalid)

    def test_worker_sets_uuid_selection_before_device_discovery(self):
        from checks import check_gpu_communication as communication
        args = SimpleNamespace(gpus="2,0", cpu_fixture=False, debug_transport=False)
        devices = [{"index": "0", "uuid": "GPU-first"}, {"index": "2", "uuid": "GPU-third"}]
        with mock.patch.dict(os.environ), \
             mock.patch.object(communication, "snapshot", return_value={"devices": devices}), \
             mock.patch("jax.local_devices", return_value=[]), \
             self.assertRaisesRegex(RuntimeError, "Expected exactly two"):
            try:
                communication.worker(args)
            finally:
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-third,GPU-first")

    def test_bus_normalization_uses_different_collective_message_sizes(self):
        self.assertEqual(corrected_bus_bytes("all_reduce", 1024), 1024)
        self.assertEqual(corrected_bus_bytes("all_gather", 1024), 1024)
        self.assertEqual(corrected_bus_bytes("reduce_scatter", 1024), 512)

    def test_real_collectives_on_two_cpu_devices(self):
        script = Path(__file__).resolve().parents[1] / "checks/check_gpu_communication.py"
        result = subprocess.run([sys.executable, "-B", str(script), "--worker", "--cpu-fixture",
                                 "--gpus", "0,1", "--sizes-mib", "0.00390625", "--warmup", "1", "--iterations", "3"],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        report, _ = parse_worker_output(result.stdout)
        self.assertEqual(report["backend"], "cpu")
        self.assertEqual({r["operation"] for r in report["results"]}, {"all_reduce", "all_gather", "reduce_scatter"})
        self.assertTrue(all(r["correctness"] == "PASS" and r["median_ms"] > 0 for r in report["results"]))
        self.assertFalse(report["training_configuration_changed"])

    def test_parent_worker_protocol_with_multiple_sizes(self):
        script = Path(__file__).resolve().parents[1] / "checks/check_gpu_communication.py"
        result = subprocess.run([sys.executable, "-B", str(script), "--cpu-fixture", "--gpus", "0,1",
                                 "--sizes-mib", "0.00390625", "0.015625", "--warmup", "1", "--iterations", "3"],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        decoder = json.JSONDecoder()
        first, end = decoder.raw_decode(result.stdout)
        report, _ = decoder.raw_decode(result.stdout[end:].lstrip())
        self.assertEqual(first["selected_pair"], "0,1")
        self.assertEqual(len(report["results"]), 6)
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
