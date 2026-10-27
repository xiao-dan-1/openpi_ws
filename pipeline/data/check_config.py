"""Check the real final-config DataLoader on CPU, without loading model weights."""
import argparse
import json

from pipeline.common.config import WORKSPACE, build_config, describe, prepare_environment, require_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "finetune"], default="smoke")
    parser.add_argument("--cache-dir")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-stats", action="store_true", help="Validate with partial probe stats, not training ready")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    prepare_environment(cpu=True, cache_dir=args.cache_dir)
    import jax
    import numpy as np
    import pyarrow as pa
    import torch
    from openpi.training import data_loader

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    torch.set_num_threads(2)
    assets = WORKSPACE / "checks/output/official_norm_smoke/assets" if args.probe_stats else None
    config = build_config(profile=args.profile, assets_base_dir=assets, num_workers=args.num_workers)
    require_assets(config, weights=True, full_stats=not args.probe_stats)
    loader = data_loader.create_data_loader(config, shuffle=False, num_batches=2)
    shapes = None
    batches = 0
    for observation, actions in loader:
        b = config.batch_size
        if observation.state.shape != (b, 32) or actions.shape != (b, 10, 32):
            raise ValueError("Wrong state/action batch shape")
        if observation.tokenized_prompt.shape != (b, 200):
            raise ValueError("Wrong token shape")
        for value in [observation.state, actions, *observation.images.values()]:
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError("Non-finite batch values")
        for image in observation.images.values():
            if image.shape != (b, 224, 224, 3):
                raise ValueError("Wrong camera shape")
        shapes = {"state": list(observation.state.shape), "actions": list(actions.shape),
                  "tokens": list(observation.tokenized_prompt.shape)}
        batches += 1
    if batches != 2 or jax.default_backend() != "cpu":
        raise ValueError("Expected two real CPU DataLoader batches")
    print(json.dumps({"status": "PASS", "probe_stats_only": args.probe_stats,
                      "backend": jax.default_backend(), "batches_checked": batches,
                      "shapes": shapes, "config": describe(config, args.profile),
                      "scope": "Real LoRA-config DataLoader; model initialization/weights/GPU/gradients untested"}, indent=2))


if __name__ == "__main__":
    main()
