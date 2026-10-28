"""Check one real LIBERO scene, 10 stabilization steps, cameras and robot state."""
import argparse
import importlib.util
import logging
import sys
from pathlib import Path

from pipeline.common.config import WORKSPACE
from pipeline.eval.eval_libero import prepare_simulation
from pipeline.common.reporting import emit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, help="Optionally save camera images and report; otherwise print results only")
    args = parser.parse_args()
    if args.task_id < 0:
        parser.error("task-id must be non-negative")
    # Use the same private LIBERO paths and CPU OSMesa backend as the evaluator.
    prepare_simulation("osmesa")
    logging.basicConfig(level=logging.INFO, force=True)
    output = args.output_dir.resolve() if args.output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True)
    report = {"status": "FAIL", "suite": args.suite, "task_id": args.task_id,
              "seed": args.seed, "render_backend": "osmesa", "output_dir": str(output) if output is not None else None}
    env = None
    try:
        import numpy as np
        from PIL import Image

        path = WORKSPACE / "openpi/examples/libero/main.py"
        spec = importlib.util.spec_from_file_location("_pipeline_libero_render_check", path)
        official = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = official
        spec.loader.exec_module(official)
        np.random.seed(args.seed)
        suite = official.benchmark.get_benchmark_dict()[args.suite]()
        if args.task_id >= suite.n_tasks:
            raise ValueError(f"task-id must be below {suite.n_tasks}")
        task = suite.get_task(args.task_id)
        initial_states = suite.get_task_init_states(args.task_id)
        if len(initial_states) == 0:
            raise ValueError("No task initial states found")
        print("Creating offscreen environment and loading scene assets...", flush=True)
        env, prompt = official._get_libero_env(task, official.LIBERO_ENV_RESOLUTION, args.seed)
        env.reset()
        env.set_init_state(initial_states[0])
        for _ in range(10):
            observation, _, _, _ = env.step(official.LIBERO_DUMMY_ACTION)
        cameras = {}
        for key, filename in [("agentview_image", "agentview.png"),
                              ("robot0_eye_in_hand_image", "wrist.png")]:
            raw = np.asarray(observation[key])
            if raw.shape != (256, 256, 3) or raw.dtype != np.uint8:
                raise ValueError(f"Invalid camera {key}: {raw.shape}, {raw.dtype}")
            if np.ptp(raw) == 0:
                raise ValueError(f"Camera {key} returned a uniform image")
            # The official evaluator rotates simulator frames by 180 degrees.
            pixels = np.ascontiguousarray(raw[::-1, ::-1])
            if output is not None:
                Image.fromarray(pixels).save(output / filename)
            cameras[key] = {"shape": list(raw.shape), "dtype": str(raw.dtype),
                            "pixel_min": int(raw.min()), "pixel_max": int(raw.max()),
                            "image_path": str(output / filename) if output is not None else None}
        state = np.concatenate((observation["robot0_eef_pos"],
                                official._quat2axisangle(observation["robot0_eef_quat"].copy()),
                                observation["robot0_gripper_qpos"]))
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Invalid 8-dimensional robot state")
        env.close()
        env = None
        report.update(status="PASS", prompt=prompt, initial_state_index=0,
                      stabilization_steps=10, cameras=cameras, state_shape=list(state.shape),
                      state_values=state.tolist(),
                      scope="Real scene/reset/initial state/10 dummy steps/CPU render; no policy, checkpoint or task-success evaluation")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            emit_report(report, output / "result.json" if output is not None else None)


if __name__ == "__main__":
    main()
