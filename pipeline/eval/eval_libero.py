"""Run the official LIBERO evaluator against a WebSocket policy server."""
import argparse
import atexit
import concurrent.futures
import contextlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

from pipeline.common.config import WORKSPACE
from pipeline.common.reporting import artifact_directory, emit_report
from pipeline.eval.runtime import add_runtime_arguments, dispatch_simulation
from pipeline.inference.policy_rpc import read_server_metadata

SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
}


class EvaluationLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.errors = []
        self.total_episodes = None
        self.total_successes = None
        self.success_rate = None

    def emit(self, record):
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            self.errors.append(message)
        if message.startswith("# successes: "):
            self.total_successes = int(message.split(": ", 1)[1].split()[0])
        if message.startswith("Total episodes: "):
            self.total_episodes = int(message.split(": ", 1)[1])
        if message.startswith("Total success rate: "):
            self.success_rate = float(message.split(": ", 1)[1])


def prepare_simulation(backend):
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("LIBERO simulation requires Python 3.8")
    if backend == "osmesa":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = backend
    os.environ["PYNPUT_BACKEND"] = "dummy"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("LP_NUM_THREADS", "4")
    root = WORKSPACE / "openpi/third_party/libero/libero/libero"
    required_paths = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "assets": root / "assets",
    }
    for key, path in required_paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing LIBERO {key}: {path}")
    temporary = tempfile.TemporaryDirectory(prefix="openpi-libero-runtime-")
    # LIBERO caches its config path at import time, so keep it until the process exits.
    atexit.register(temporary.cleanup)
    temp_root = Path(temporary.name)
    numba_dir = temp_root / "numba"
    config_dir = temp_root / "config"
    datasets_dir = temp_root / "datasets"
    for directory in (numba_dir, config_dir, datasets_dir):
        directory.mkdir()
    os.environ["NUMBA_CACHE_DIR"] = str(numba_dir)
    paths = {**required_paths, "datasets": datasets_dir}
    # JSON is valid YAML; writing before LIBERO import avoids its interactive prompt.
    (config_dir / "config.yaml").write_text(
        json.dumps({key: str(path) for key, path in paths.items()}, indent=2), encoding="utf-8"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    for extra_path in (
        str(WORKSPACE / "openpi/packages/openpi-client/src"),
        str(WORKSPACE / "openpi/third_party/libero"),
    ):
        if extra_path not in sys.path:
            sys.path.insert(0, extra_path)


class _SingleTaskSuite:
    n_tasks = 1

    def __init__(self, suite, task_id):
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(f"task-id must be in 0..{suite.n_tasks - 1}, got {task_id}")
        self._suite = suite
        self._task_id = task_id

    def get_task(self, _index):
        return self._suite.get_task(self._task_id)

    def get_task_init_states(self, _index):
        return self._suite.get_task_init_states(self._task_id)

    def __getattr__(self, name):
        return getattr(self._suite, name)


def _forward_args(args):
    forwarded = [
        "--host", args.host,
        "--port", str(args.port),
        "--suite", args.suite,
        "--trials", str(args.trials),
        "--seed", str(args.seed),
    ]
    if args.task_id is not None:
        forwarded += ["--task-id", str(args.task_id)]
    if args.workers > 1:
        forwarded += ["--workers", str(args.workers)]
    if args.no_videos:
        forwarded.append("--no-videos")
    return forwarded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port (default: 8000)")
    parser.add_argument("--suite", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        default="libero_spatial")
    parser.add_argument("--trials", type=int, default=50,
                        help="Rollouts per task using the first N official initial states (default: %(default)s)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Optional single task index within the suite (default: evaluate all 10 tasks)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent task worker processes for full-suite evaluation (default: %(default)s)")
    parser.add_argument("--no-videos", action="store_true",
                        help="Skip MP4 rollout video encoding during evaluation")
    parser.add_argument("--seed", type=int, default=7)
    add_runtime_arguments(parser)
    parser.add_argument("--output-dir", type=Path,
                        help="Optionally keep evaluation videos and report; otherwise use temporary files")
    args = parser.parse_args()
    if not 1 <= args.trials <= 50 or not 1 <= args.port <= 65535 or (args.task_id is not None and args.task_id < 0):
        parser.error("trials must be 1..50; port 1..65535; task-id >= 0")
    if args.workers < 1:
        parser.error("workers must be >= 1")
    if args.workers > 1 and args.task_id is not None:
        parser.error("--workers > 1 cannot be combined with --task-id")
    if dispatch_simulation(parser, args, module="pipeline.eval.eval_libero", workspace=WORKSPACE,
                           docker_args=_forward_args(args), host=args.host):
        return
    with artifact_directory(args.output_dir) as output:
        if args.workers > 1:
            run_parallel_evaluation(args, output)
        else:
            run_evaluation(args, output)


def _terminate_processes(processes):
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def run_parallel_evaluation(args, output):
    logging.basicConfig(level=logging.INFO, force=True)
    metadata = read_server_metadata(args.host, args.port)
    num_tasks = SUITE_TASK_COUNTS[args.suite]
    effective_workers = min(args.workers, num_tasks)
    started = time.monotonic()
    tasks_root = output / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "FAIL",
        "host": args.host,
        "port": args.port,
        "server_metadata": metadata,
        "suite": args.suite,
        "task_id": None,
        "trials_per_task": args.trials,
        "seed": args.seed,
        "render_backend": args.render_backend,
        "workers": effective_workers,
        "no_videos": args.no_videos,
        "output_dir": str(output) if args.output_dir is not None else None,
    }
    child_env = os.environ.copy()
    child_env["OMP_NUM_THREADS"] = "1"
    child_env["LP_NUM_THREADS"] = "1"
    if args.render_backend == "osmesa":
        child_env["CUDA_VISIBLE_DEVICES"] = ""

    active_lock = threading.Lock()
    active_procs = {}

    def run_task(task_id):
        task_dir = tasks_root / f"task_{task_id}"
        log_path = tasks_root / f"task_{task_id}.log"
        cmd = [
            sys.executable, "-B", "-u", "-m", "pipeline.eval.eval_libero",
            "--host", args.host,
            "--port", str(args.port),
            "--suite", args.suite,
            "--task-id", str(task_id),
            "--trials", str(args.trials),
            "--seed", str(args.seed),
            "--render-backend", args.render_backend,
            "--workers", "1",
            "--output-dir", str(task_dir),
        ]
        if args.no_videos:
            cmd.append("--no-videos")
        task_started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(cmd, cwd=str(WORKSPACE), env=child_env, stdout=log_file, stderr=subprocess.STDOUT)
            with active_lock:
                active_procs[id(proc)] = proc
            try:
                returncode = proc.wait()
            finally:
                with active_lock:
                    active_procs.pop(id(proc), None)
        task_elapsed = round(time.monotonic() - task_started, 3)
        result_path = task_dir / "result.json"
        if result_path.is_file():
            task_report = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:] if log_path.is_file() else ""
            task_report = {
                "status": "FAIL",
                "task_id": task_id,
                "episodes_completed": 0,
                "episodes_succeeded": 0,
                "task_success_rate": None,
                "error": f"Worker exited with code {returncode} without result.json: {tail}",
            }
        summary = {
            "task_id": task_id,
            "status": task_report.get("status", "FAIL"),
            "returncode": returncode,
            "episodes_completed": task_report.get("episodes_completed", 0),
            "episodes_succeeded": task_report.get("episodes_succeeded", 0),
            "task_success_rate": task_report.get("task_success_rate"),
            "elapsed_seconds": task_report.get("elapsed_seconds", task_elapsed),
        }
        if task_report.get("error"):
            summary["error"] = task_report["error"]
        return summary, task_dir

    task_results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(run_task, task_id): task_id for task_id in range(num_tasks)}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                summary, task_dir = future.result()
                completed += 1
                logging.info(
                    "Task %d (%d/%d) finished: status=%s, rate=%s, elapsed=%.1fs",
                    summary["task_id"], completed, num_tasks, summary["status"],
                    summary["task_success_rate"], summary["elapsed_seconds"],
                )
                if args.output_dir is not None and not args.no_videos:
                    src_videos = task_dir / "videos"
                    if src_videos.is_dir():
                        dst_videos = output / "videos"
                        dst_videos.mkdir(parents=True, exist_ok=True)
                        for mp4 in src_videos.glob("*.mp4"):
                            shutil.move(str(mp4), str(dst_videos / mp4.name))
                task_results.append(summary)
        task_results.sort(key=lambda item: item["task_id"])
        report["tasks"] = task_results
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        failures = [
            item for item in task_results
            if item["returncode"] != 0 or item["status"] != "PASS" or item["episodes_completed"] != args.trials
        ]
        if failures:
            failed_ids = [item["task_id"] for item in failures]
            raise RuntimeError(f"Parallel LIBERO tasks failed: {failed_ids}")
        total_episodes = sum(item["episodes_completed"] for item in task_results)
        total_succeeded = sum(item["episodes_succeeded"] for item in task_results)
        report.update(
            status="PASS",
            episodes_completed=total_episodes,
            episodes_succeeded=total_succeeded,
            task_success_rate=total_succeeded / total_episodes,
            scope="Parallel task evaluation completed; PASS describes execution, not policy quality",
        )
    except BaseException as error:
        with active_lock:
            remaining = list(active_procs.values())
        if remaining:
            _terminate_processes(remaining)
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        emit_report(report, output / "result.json" if args.output_dir is not None else None)


def run_evaluation(args, output):
    prepare_simulation(args.render_backend)
    logging.basicConfig(level=logging.INFO, force=True)
    metadata = read_server_metadata(args.host, args.port)
    capture = EvaluationLog()
    logging.getLogger().addHandler(capture)
    started = time.monotonic()
    expected_episodes = (1 if args.task_id is not None else 10) * args.trials
    report = {
        "status": "FAIL",
        "host": args.host,
        "port": args.port,
        "server_metadata": metadata,
        "suite": args.suite,
        "task_id": args.task_id,
        "trials_per_task": args.trials,
        "seed": args.seed,
        "render_backend": args.render_backend,
        "workers": 1,
        "no_videos": args.no_videos,
        "output_dir": str(output) if args.output_dir is not None else None,
    }
    try:
        path = WORKSPACE / "openpi/examples/libero/main.py"
        spec = importlib.util.spec_from_file_location("_pipeline_libero_evaluator", path)
        official = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = official
        spec.loader.exec_module(official)
        eval_args = official.Args(
            host=args.host,
            port=args.port,
            task_suite_name=args.suite,
            num_trials_per_task=args.trials,
            seed=args.seed,
            video_out_path=str(output / "videos"),
        )
        with contextlib.ExitStack() as stack:
            if args.task_id is not None:
                real_get_benchmark_dict = official.benchmark.get_benchmark_dict

                def filtered_benchmark_dict():
                    original = real_get_benchmark_dict()
                    factory = original[args.suite]
                    return {**original, args.suite: lambda: _SingleTaskSuite(factory(), args.task_id)}

                stack.enter_context(patch.object(official.benchmark, "get_benchmark_dict", filtered_benchmark_dict))
            if args.no_videos and hasattr(official, "imageio"):
                stack.enter_context(patch.object(official.imageio, "mimwrite", lambda *a, **kw: None))
            official.eval_libero(eval_args)
        if capture.errors:
            raise RuntimeError("Official evaluator logged runtime errors; these are not task failures")
        if capture.total_episodes != expected_episodes or capture.success_rate is None:
            raise RuntimeError("Official evaluator did not report the expected completed episodes")
        episodes_succeeded = (
            capture.total_successes
            if capture.total_successes is not None
            else round(capture.success_rate * capture.total_episodes)
        )
        report.update(
            status="PASS",
            episodes_completed=capture.total_episodes,
            episodes_succeeded=episodes_succeeded,
            task_success_rate=capture.success_rate,
            elapsed_seconds=round(time.monotonic() - started, 3),
            scope="Evaluation completed; PASS describes execution, not policy quality",
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        logging.getLogger().removeHandler(capture)
        report["runtime_errors"] = capture.errors
        emit_report(report, output / "result.json" if args.output_dir is not None else None)


if __name__ == "__main__":
    main()
