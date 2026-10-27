"""Orchestration failure tests with isolated MOCK workers; no GPU/model test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.train import benchmark_batch as benchmark


class BatchDriverTest(unittest.TestCase):
    def run_scenario(self, stub, num_workers=0):
        with tempfile.TemporaryDirectory(prefix="openpi_batch_driver_") as directory:
            args = argparse.Namespace(gpu="0", batches=[1, 2, 4], warmup_steps=1,
                                      measure_steps=2, timeout_seconds=30, cache_dir=None,
                                      num_workers=num_workers)
            real_popen = subprocess.Popen
            launched = []
            self.forwarded_workers = []

            def launch(command, **kwargs):
                self.assertEqual(command[1:5], ["-B", "-u", "-m", "pipeline.train.benchmark_batch"])
                self.assertEqual(kwargs["cwd"], Path(directory))
                batch = command[command.index("--batch-size") + 1]
                output = command[command.index("--worker-output") + 1]
                workers = command[command.index("--num-workers") + 1]
                launched.append(int(batch))
                self.forwarded_workers.append(int(workers))
                return real_popen([sys.executable, "-c", stub, output, batch, workers], **kwargs)

            gpu = {"index": 0, "uuid": "GPU-test", "total_mib": 24576, "used_mib": 13}
            with mock.patch.object(benchmark, "WORKSPACE", Path(directory)), \
                 mock.patch.object(benchmark, "gpu_snapshot", return_value=gpu), \
                 mock.patch.object(benchmark, "gpu_processes", return_value={}), \
                 mock.patch.object(benchmark.subprocess, "Popen", side_effect=launch):
                exit_code = benchmark.driver(args)
            summary_path = next(Path(directory).glob("checks/output/batch_benchmark/*/summary.json"))
            summary = json.loads(summary_path.read_text())
            self.assertFalse((Path(directory) / "checkpoints").exists())
            return exit_code, launched, summary

    def test_oom_stops_larger_batches(self):
        code, launched, summary = self.run_scenario(
            "import json,sys; from pathlib import Path; b=int(sys.argv[2]); "
            "Path(sys.argv[1]).write_text(json.dumps({'batch_size':b,'status':'PASS' if b==1 else 'OOM'})); "
            "sys.exit(0 if b==1 else 1)")
        self.assertEqual(code, 1)
        self.assertEqual(launched, [1, 2])
        self.assertEqual(summary["status"], "PARTIAL")
        self.assertEqual(summary["passed_batches"], [1])

    def test_nonzero_exit_cannot_be_pass(self):
        code, launched, summary = self.run_scenario(
            "import json,sys; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(json.dumps({'batch_size':int(sys.argv[2]),'status':'PASS'})); "
            "sys.exit(3)")
        self.assertEqual(code, 1)
        self.assertEqual(launched, [1])
        self.assertEqual(summary["results"][0]["status"], "FAIL")

    def test_worker_without_report_cannot_be_pass(self):
        code, launched, summary = self.run_scenario("import sys; sys.exit(2)")
        self.assertEqual(code, 1)
        self.assertEqual(launched, [1])
        self.assertEqual(summary["results"][0]["status"], "FAIL")

    def test_busy_gpu_is_rejected(self):
        gpu = {"index": 0, "uuid": "GPU-test", "total_mib": 24576, "used_mib": 6000}
        with mock.patch.object(benchmark, "gpu_snapshot", return_value=gpu), \
             mock.patch.object(benchmark, "gpu_processes", return_value={1234: 6000}):
            with self.assertRaisesRegex(RuntimeError, "GPU is occupied"):
                benchmark.require_available_gpu("0")

    def test_worker_count_is_forwarded_to_isolated_children(self):
        code, launched, summary = self.run_scenario(
            "import json,sys; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(json.dumps({'batch_size':int(sys.argv[2]),'status':'PASS',"
            "'received_workers':int(sys.argv[3])}))", num_workers=2)
        self.assertEqual(code, 0)
        self.assertEqual(launched, [1, 2, 4])
        self.assertEqual(self.forwarded_workers, [2, 2, 2])
        self.assertEqual(summary["num_workers"], 2)
        self.assertTrue(all(result["received_workers"] == 2 for result in summary["results"]))

    @unittest.skipUnless(os.name == "posix", "Process-group cleanup is a Linux path")
    def test_stop_includes_spawned_child(self):
        with tempfile.TemporaryDirectory(prefix="openpi_batch_cleanup_") as directory:
            pid_file = Path(directory) / "child.pid"
            code = ("import subprocess,sys,time; from pathlib import Path; "
                    "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)")
            process = subprocess.Popen([sys.executable, "-c", code, str(pid_file)], start_new_session=True)
            try:
                deadline = time.monotonic() + 10
                while not pid_file.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(pid_file.exists())
                child_pid = int(pid_file.read_text())
                benchmark.stop_test_process(process)
                for _ in range(40):
                    stat = Path(f"/proc/{child_pid}/stat")
                    if not stat.exists() or stat.read_text().split()[2] == "Z":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("Spawned test child was still running after cleanup")
            finally:
                benchmark.stop_test_process(process)


if __name__ == "__main__":
    unittest.main()
