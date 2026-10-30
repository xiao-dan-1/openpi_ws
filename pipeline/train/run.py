"""Launch the pinned official JAX trainer with workspace-local LoRA config."""
import argparse
import logging
import json

from pipeline.common.validation import experiment_name
from pipeline.common.config import WORKSPACE, build_config, describe, load_official_script, prepare_environment, require_assets, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="One GPU index or UUID allocated to you")
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--profile", choices=["smoke", "finetune"], default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.steps is not None and args.steps < 1):
        parser.error("batch-size and steps must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    prepare_environment(cpu=False, gpu=args.gpu, cache_dir=args.cache_dir)
    import jax
    config = build_config(profile=args.profile, exp_name=args.exp_name,
                          batch_size=args.batch_size, steps=args.steps, resume=args.resume,
                          num_workers=args.num_workers)
    require_assets(config, weights=True, full_stats=True)
    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU; refusing CPU fallback")
    checkpoint = config.checkpoint_dir
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"Experiment exists: {checkpoint}; use --resume or a new name")
    if args.resume and not checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint directory is absent: {checkpoint}")
    if args.resume:
        import orbax.checkpoint as ocp

        # Orbax ignores unfinished saves; read-only prevents retention/deletion.
        with ocp.CheckpointManager(checkpoint, options=ocp.CheckpointManagerOptions(
            read_only=True, create=False
        )) as manager:
            latest = manager.latest_step()
        if latest is None or latest <= 0:
            raise ValueError("No completed nonzero checkpoint; refusing to restart as --resume")
    run_dir = WORKSPACE / "runs" / config.name / args.exp_name
    if args.resume:
        original = json.loads((run_dir / "config.json").read_text())
        current = describe(config, args.profile)
        for key in ["profile", "model", "learning_rate", "norm_stats_sha256", "local_dataset_alias"]:
            if original.get(key) != current.get(key):
                raise ValueError(f"Resume configuration changed: {key}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(config, args.profile, run_dir / ("resume_config.json" if args.resume else "config.json"))
    logging.basicConfig(level=logging.INFO)
    official = load_official_script("train")
    official.main(config)


if __name__ == "__main__":
    main()
