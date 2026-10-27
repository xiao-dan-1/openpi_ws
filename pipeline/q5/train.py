"""Server-only execution of the official trainer with Q5-specific configuration."""
import argparse
import json
import logging
from pathlib import Path

from pipeline.common.validation import experiment_name
from pipeline.q5.runtime import Paths, prepare_environment, official_train_module, require_stats, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--profile", choices=["smoke", "finetune"], default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--continue-after-segment", action="store_true",
                        help="Acknowledge that a finetune segment ending during warmup is only the first segment")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or (args.steps is not None and args.steps < 1):
        parser.error("Batch/steps must be positive; workers must be non-negative")
    from pipeline.q5.schedule import validate_schedule
    try:
        schedule_plan = validate_schedule(args.profile, args.steps or (100 if args.profile == "smoke" else 1000),
                    args.warmup_steps if args.warmup_steps is not None else (10 if args.profile == "smoke" else 1000),
                    args.schedule_steps if args.schedule_steps is not None else (1000 if args.profile == "smoke" else 30000),
                    args.continue_after_segment)
    except ValueError as exc:
        parser.error(str(exc))
    paths = Paths.create(args.workspace)
    revision = prepare_environment(paths, cpu=False, gpu=args.gpu)
    require_stats(paths)
    from pipeline.q5.config import build_config, describe, require_base_assets
    require_base_assets(paths)
    import jax
    require(jax.default_backend() == "gpu" and jax.device_count() == 1, "Exactly one server GPU is required")
    run_dir = paths.workspace / "runs/pi05_q5" / args.exp_name
    config = build_config(paths, profile=args.profile, exp_name=args.exp_name, batch_size=args.batch_size,
                          steps=args.steps, num_workers=args.num_workers, resume=args.resume,
                          warmup_steps=args.warmup_steps, schedule_steps=args.schedule_steps,
                          weight_audit_path=run_dir / "weight_loading_report.json")
    manifest = describe(paths, config, args.profile)
    manifest["upstream_commit"] = revision
    manifest["schedule_plan"] = schedule_plan
    train_frames = json.loads((paths.train / "meta/info.json").read_text())["total_frames"]
    manifest["sample_budget"] = {"training_start_frames": train_frames,
                    "nominal_window_samples": config.num_train_steps * config.batch_size,
                    "nominal_passes": config.num_train_steps * config.batch_size / train_frames,
                    "note": "Sample budget ratio, not unique start coverage or task success"}
    print(json.dumps({"schedule_plan": schedule_plan, "sample_budget": manifest["sample_budget"]}, indent=2), flush=True)
    run_dir = paths.workspace / "runs" / config.name / args.exp_name
    if args.resume:
        require(config.checkpoint_dir.exists(), "No checkpoint directory to resume")
        original = json.loads((run_dir / "config.json").read_text())
        for key in ["profile", "model", "freeze_policy", "learning_rate", "optimizer", "batch_size", "seed",
                    "split_manifest_sha256", "norm_stats_sha256", "upstream_commit", "implementation_sha256"]:
            require(original.get(key) == manifest.get(key), f"Resume configuration differs: {key}")
    else:
        require(not config.checkpoint_dir.exists() and not run_dir.exists(), "Experiment exists; use --resume or another name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ("resume_config.json" if args.resume else "config.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    logging.basicConfig(level=logging.INFO)
    official_train_module(paths).main(config)


if __name__ == "__main__":
    main()
