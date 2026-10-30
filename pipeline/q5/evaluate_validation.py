"""Server-only offline validation of saved Q5 policies; never reads test observations."""
import argparse
import dataclasses
import json
from pathlib import Path

from pipeline.common.validation import experiment_name
from pipeline.common.reporting import emit_report
from pipeline.q5.runtime import Paths, prepare_environment, require, require_stats, validate_split, sha256, ASSET_ID, VALIDATION_ID
from pipeline.q5.memory import add_memory_argument, memory_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--samples-per-episode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, help="Optionally save the validation report; otherwise print results only")
    add_memory_argument(parser)
    args = parser.parse_args()
    if args.step < 0 or args.samples_per_episode < 1:
        parser.error("Step must be nonnegative and sample count positive")
    paths = Paths.create(args.workspace)
    prepare_environment(paths, cpu=False, gpu=args.gpu, memory_fraction=args.memory_fraction)
    statistics = require_stats(paths)
    manifest = validate_split(paths)
    import jax
    import numpy as np
    from openpi.policies import policy_config
    from openpi.training import data_loader
    from pipeline.q5.config import build_config
    require(jax.default_backend() == "gpu" and jax.device_count() == 1, "Run validation on one server GPU")
    run = json.loads((paths.workspace / "runs/pi05_q5" / args.exp_name / "config.json").read_text())
    require(run["split_manifest_sha256"] == sha256(paths.split_manifest), "Checkpoint split differs")
    require(run["norm_stats_sha256"] == statistics["sha256"], "Checkpoint training statistics differ")
    checkpoint = paths.workspace / "checkpoints/pi05_q5" / args.exp_name / str(args.step)
    require(checkpoint.is_dir(), "Checkpoint not found")
    require(sha256(checkpoint / "assets" / ASSET_ID / "norm_stats.json") == statistics["sha256"], "Saved checkpoint statistics mismatch")
    config = build_config(paths, profile=run["profile"], exp_name=args.exp_name)
    require(dataclasses.asdict(config.model) == run["model"], "Checkpoint model specification differs")
    if args.output is not None:
        require(not args.output.exists(), "Validation output exists; select a new --output")
    policy = policy_config.create_trained_policy(config, checkpoint)
    data = dataclasses.replace(config.data.create(config.assets_dirs, config.model), repo_id=VALIDATION_ID)
    if hasattr(data, "repo_root"):
        data = dataclasses.replace(data, repo_root=str(paths.validation))
    raw = data_loader.create_torch_dataset(data, 10, config.model)
    groups = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "left_hand": slice(14, 26), "right_hand": slice(26, 38)}
    rng = np.random.default_rng(args.seed)
    rows = []
    start = 0
    for record in manifest["validation"]:
        n = record["length"]
        sums = {name: [0.0, 0, 0.0, 0] for name in groups}
        indices = np.unique(np.rint(np.linspace(0, n - 1, min(args.samples_per_episode, n))).astype(int))
        for frame in indices:
            item = raw[int(start + frame)]
            # Dataset-only repack produces the exact external Q5 observation contract.
            from openpi import transforms
            packed = transforms.compose(data.repack_transforms.inputs)(item)
            expected = np.asarray(packed.pop("actions"), dtype=float)
            require(
                expected.shape == (10, 38) and np.isfinite(expected).all(),
                f"Invalid validation actions: episode={record['output_episode_index']}, frame={frame}; "
                "expected shape (10, 38) with finite values",
            )
            noise = rng.standard_normal((10, 38)).astype(np.float32)
            predicted = np.asarray(policy.infer(packed, noise=noise)["actions"])
            require(predicted.shape == (10, 38) and np.isfinite(predicted).all(), "Invalid predicted Q5 actions")
            valid = min(10, n - int(frame))
            for name, sl in groups.items():
                difference = np.abs(predicted[:valid, sl] - expected[:valid, sl])
                sums[name][0] += float(difference.sum())
                sums[name][1] += difference.size
                first = np.abs(predicted[0, sl] - expected[0, sl])
                sums[name][2] += float(first.sum())
                sums[name][3] += first.size
        rows.append({"validation_episode": record["output_episode_index"], "source_bag": record["source_bag"],
                     "frame_indices": indices.tolist(), "mae_rad": {k: v[0] / v[1] for k, v in sums.items()},
                     "first_action_mae_rad": {k: v[2] / v[3] for k, v in sums.items()}})
        start += n
    report = {"status": "PASS", "allocator_settings": memory_settings(), "split": "validation", "episodes": len(rows), "seed": args.seed,
              "samples_per_episode_limit": args.samples_per_episode, "checkpoint": str(checkpoint),
              "norm_stats_sha256": statistics["sha256"], "test_observations_read": 0,
              "equal_episode_mae_rad": {k: float(np.mean([r["mae_rad"][k] for r in rows])) for k in groups},
              "equal_episode_first_action_mae_rad": {k: float(np.mean([r["first_action_mae_rad"][k] for r in rows])) for k in groups},
              "per_episode": rows, "padding_excluded_from_metrics": True,
              "note": "Offline demonstration-action error under fixed sample/noise schedule; not a closed-loop success rate."}
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
