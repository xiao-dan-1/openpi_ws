"""Run the official LIBERO evaluator in the separate Python 3.8 environment."""
import argparse
import datetime
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

from pipeline.common.config import WORKSPACE
from pipeline.inference.policy_rpc import read_server_metadata, verify_server_metadata
from pipeline.common.validation import experiment_name


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
        raise RuntimeError("Use tools/libero-env/bin/python (Python 3.8), not the training environment")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = backend
    os.environ["PYNPUT_BACKEND"] = "dummy"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["LP_NUM_THREADS"] = "4"
    os.environ["NUMBA_CACHE_DIR"] = str(WORKSPACE / ".cache/numba")
    root = WORKSPACE / "openpi/third_party/libero/libero/libero"
    paths = {"benchmark_root": root, "bddl_files": root / "bddl_files",
             "init_states": root / "init_files", "assets": root / "assets",
             "datasets": WORKSPACE / "datasets"}
    for key, path in paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing LIBERO {key}: {path}")
    config_dir = WORKSPACE / ".cache/libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML; writing before LIBERO import avoids its interactive prompt.
    (config_dir / "config.yaml").write_text(
        json.dumps({key: str(path) for key, path in paths.items()}, indent=2), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(WORKSPACE / "openpi/third_party/libero"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        default="libero_spatial")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-backend", choices=["osmesa", "egl"], default="osmesa")
    args = parser.parse_args()
    if not 1 <= args.trials <= 50 or args.step < 0 or not 1 <= args.port <= 65535:
        parser.error("trials must be 1..50; step >= 0; port 1..65535")
    prepare_simulation(args.render_backend)
    logging.basicConfig(level=logging.INFO, force=True)
    metadata = read_server_metadata(args.host, args.port)
    identity = verify_server_metadata(metadata, args.exp_name, args.step)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = WORKSPACE / "runs/pi05_libero" / args.exp_name / f"eval_{args.step}" / f"{args.suite}_{stamp}"
    output.mkdir(parents=True)
    capture = EvaluationLog()
    logging.getLogger().addHandler(capture)
    report = {"status": "FAIL", "policy": identity, "suite": args.suite, "trials_per_task": args.trials,
              "seed": args.seed, "render_backend": args.render_backend, "output_dir": str(output)}
    try:
        path = WORKSPACE / "openpi/examples/libero/main.py"
        spec = importlib.util.spec_from_file_location("_pipeline_libero_evaluator", path)
        official = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = official
        spec.loader.exec_module(official)
        official.eval_libero(official.Args(host=args.host, port=args.port, task_suite_name=args.suite,
                                          num_trials_per_task=args.trials, seed=args.seed,
                                          video_out_path=str(output / "videos")))
        if capture.errors:
            raise RuntimeError("Official evaluator logged runtime errors; these are not task failures")
        if capture.total_episodes != 10 * args.trials or capture.success_rate is None:
            raise RuntimeError("Official evaluator did not report the expected completed episodes")
        report.update(status="PASS", episodes_completed=capture.total_episodes,
                      task_success_rate=capture.success_rate,
                      scope="Evaluation completed; PASS describes execution, not policy quality")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        logging.getLogger().removeHandler(capture)
        report["runtime_errors"] = capture.errors
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
