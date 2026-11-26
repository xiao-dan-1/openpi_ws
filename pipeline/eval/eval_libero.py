"""Run the official LIBERO evaluator against a WebSocket policy server."""
import argparse
import atexit
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from pipeline.common.config import WORKSPACE
from pipeline.common.reporting import artifact_directory, emit_report
from pipeline.eval.runtime import add_runtime_arguments, dispatch_simulation
from pipeline.inference.policy_rpc import read_server_metadata


class EvaluationLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.errors = []
        self.total_episodes = None
        self.success_rate = None

    def emit(self, record):
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            self.errors.append(message)
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
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["LP_NUM_THREADS"] = "4"
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
    return forwarded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port (default: 8000)")
    parser.add_argument("--suite", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        default="libero_spatial")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--task-id", type=int, default=None,
                        help="Optional single task index within the suite (default: evaluate all 10 tasks)")
    parser.add_argument("--seed", type=int, default=7)
    add_runtime_arguments(parser)
    parser.add_argument("--output-dir", type=Path,
                        help="Optionally keep evaluation videos and report; otherwise use temporary files")
    args = parser.parse_args()
    if not 1 <= args.trials <= 50 or not 1 <= args.port <= 65535 or (args.task_id is not None and args.task_id < 0):
        parser.error("trials must be 1..50; port 1..65535; task-id >= 0")
    if dispatch_simulation(parser, args, module="pipeline.eval.eval_libero", workspace=WORKSPACE,
                           docker_args=_forward_args(args), host=args.host):
        return
    with artifact_directory(args.output_dir) as output:
        run_evaluation(args, output)


def run_evaluation(args, output):
    prepare_simulation(args.render_backend)
    logging.basicConfig(level=logging.INFO, force=True)
    metadata = read_server_metadata(args.host, args.port)
    capture = EvaluationLog()
    logging.getLogger().addHandler(capture)
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
        if args.task_id is not None:
            real_get_benchmark_dict = official.benchmark.get_benchmark_dict

            def filtered_benchmark_dict():
                original = real_get_benchmark_dict()
                factory = original[args.suite]
                return {**original, args.suite: lambda: _SingleTaskSuite(factory(), args.task_id)}

            with patch.object(official.benchmark, "get_benchmark_dict", filtered_benchmark_dict):
                official.eval_libero(eval_args)
        else:
            official.eval_libero(eval_args)
        if capture.errors:
            raise RuntimeError("Official evaluator logged runtime errors; these are not task failures")
        if capture.total_episodes != expected_episodes or capture.success_rate is None:
            raise RuntimeError("Official evaluator did not report the expected completed episodes")
        report.update(
            status="PASS",
            episodes_completed=capture.total_episodes,
            task_success_rate=capture.success_rate,
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
