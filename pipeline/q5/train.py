"""Server-only execution of the official trainer with Q5-specific configuration."""
import argparse
import json
import logging
from pathlib import Path

from pipeline.common.validation import experiment_name
from pipeline.q5.runtime import Paths, prepare_environment, official_train_module, require_stats, require
from pipeline.q5.memory import add_memory_argument, memory_settings
from pipeline.q5.parallelism import gpu_selectors, training_layout


def require_resume_configuration(original, manifest):
    for key in ["profile", "model", "freeze_policy", "learning_rate", "optimizer", "batch_size", "seed",
                "fsdp_devices", "device_count", "split_manifest_sha256", "norm_stats_sha256",
                "upstream_commit", "implementation_sha256"]:
        require(original.get(key) == manifest.get(key), f"Resume configuration differs: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--gpu", type=gpu_selectors, help="One physical GPU index or UUID (existing single-GPU syntax)")
    selectors.add_argument("--gpus", type=gpu_selectors, help="One/two physical GPU indices or UUIDs, e.g. 2,3")
    parser.add_argument("--fsdp-devices", type=int, choices=[1, 2],
                        help="Single GPU defaults to 1; two GPUs require explicit 2")
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--profile", choices=["smoke", "finetune"], default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="Global batch across selected GPUs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--continue-after-segment", action="store_true",
                        help="Acknowledge that a finetune segment ending during warmup is only the first segment")
    add_memory_argument(parser)
    args = parser.parse_args()
    gpus = args.gpus or args.gpu
    if args.gpu and "," in args.gpu:
        parser.error("--gpu selects one GPU; use --gpus for two GPUs")
    try:
        layout = training_layout(gpus, args.fsdp_devices, args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))
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
    revision = prepare_environment(paths, cpu=False, gpu=gpus, allow_multiple_gpus=layout["device_count"] > 1,
                                   memory_fraction=args.memory_fraction)
    require_stats(paths)
    from pipeline.q5.config import build_config, describe, require_base_assets
    require_base_assets(paths)
    import jax
    require(jax.default_backend() == "gpu" and jax.device_count() == layout["device_count"]
            and jax.process_count() == 1, "Expected selected GPUs on one server process")
    run_dir = paths.workspace / "runs/pi05_q5" / args.exp_name
    config = build_config(paths, profile=args.profile, exp_name=args.exp_name, batch_size=args.batch_size,
                          steps=args.steps, num_workers=args.num_workers, resume=args.resume,
                          warmup_steps=args.warmup_steps, schedule_steps=args.schedule_steps,
                          weight_audit_path=run_dir / "weight_loading_report.json", fsdp_devices=layout["fsdp_devices"])
    manifest = describe(paths, config, args.profile)
    manifest["upstream_commit"] = revision
    manifest["schedule_plan"] = schedule_plan
    manifest["allocator_settings"] = memory_settings()
    manifest["device_count"] = layout["device_count"]
    manifest["parallelism"] = {**layout, "gpu_selectors": gpus,
                               "devices": [str(device) for device in jax.local_devices()]}
    train_frames = json.loads((paths.train / "meta/info.json").read_text())["total_frames"]
    manifest["sample_budget"] = {"training_start_frames": train_frames,
                    "nominal_window_samples": config.num_train_steps * config.batch_size,
                    "nominal_passes": config.num_train_steps * config.batch_size / train_frames,
                    "note": "Sample budget ratio, not unique start coverage or task success"}
    print(json.dumps({"schedule_plan": schedule_plan, "parallelism": manifest["parallelism"],
                      "sample_budget": manifest["sample_budget"],
                      "allocator_settings": manifest["allocator_settings"]}, indent=2), flush=True)
    run_dir = paths.workspace / "runs" / config.name / args.exp_name
    if args.resume:
        require(config.checkpoint_dir.exists(), "No checkpoint directory to resume")
        original = json.loads((run_dir / "config.json").read_text())
        require_resume_configuration(original, manifest)
        from pipeline.q5.checkpointing import checkpoint_progress
        progress = checkpoint_progress(config.checkpoint_dir, manifest["norm_stats_sha256"])
        require(progress["state_step"] < config.num_train_steps, "--steps must exceed the restored state.step")
        manifest["resume_checkpoint"] = progress
        print(json.dumps({"resume_checkpoint": progress,
                          "next_learning_rate": float(config.lr_schedule.create()(progress["state_step"]))}, indent=2), flush=True)
    else:
        require(not config.checkpoint_dir.exists() and not run_dir.exists(), "Experiment exists; use --resume or another name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ("resume_config.json" if args.resume else "config.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    logging.basicConfig(level=logging.INFO)
    official_train_module(paths).main(config)
    from pipeline.q5.checkpointing import checkpoint_progress
    progress = checkpoint_progress(config.checkpoint_dir, manifest["norm_stats_sha256"])
    require(progress["state_step"] == config.num_train_steps, "Final checkpoint did not reach --steps")
    result = {"status": "PASS", "parallelism": manifest["parallelism"], "allocator_settings": manifest["allocator_settings"],
              "resume_checkpoint": manifest.get("resume_checkpoint"), "final_checkpoint": progress,
              "next_learning_rate": float(config.lr_schedule.create()(progress["state_step"]))}
    (run_dir / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
