"""Call the upstream compute_norm_stats.main with a process-local configuration."""
import argparse
import hashlib
import json
import time

from pipeline.common.config import WORKSPACE, LOCAL_REPO_ID, build_config, load_official_script, prepare_environment, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel data reader processes; start with 2 on the server")
    parser.add_argument("--max-frames", type=int, help="Probe only; writes outside active training assets")
    parser.add_argument("--cache-dir", help="Optional existing Hugging Face datasets cache")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_frames is not None and args.max_frames < args.batch_size):
        parser.error("batch-size must be positive and max-frames must be at least one batch")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    prepare_environment(cpu=True, cache_dir=args.cache_dir)
    import pyarrow as pa
    import torch
    from openpi.training import config as upstream
    from openpi.shared import normalize

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    torch.set_num_threads(2)
    partial = args.max_frames is not None
    assets = WORKSPACE / "checks/output/official_norm_smoke/assets" if partial else WORKSPACE / "assets"
    config = build_config(profile="stats", batch_size=args.batch_size, assets_base_dir=assets,
                          num_workers=args.num_workers)
    from pipeline.common.config import require_assets
    require_assets(config, stats=False)
    official = load_official_script("compute_norm_stats")
    # The pinned official CLI resolves names through this dictionary. Register
    # our local-path configuration in this PROCESS only, leaving source files intact.
    original = upstream._CONFIGS_DICT[config.name]
    started = time.monotonic()
    try:
        upstream._CONFIGS_DICT[config.name] = config
        official.main(config.name, max_frames=args.max_frames)
    finally:
        upstream._CONFIGS_DICT[config.name] = original
    elapsed = time.monotonic() - started
    output = config.assets_dirs / LOCAL_REPO_ID
    stats = normalize.load(output)
    import numpy as np
    for key, dimension in [("state", 8), ("actions", 7)]:
        for field in ["mean", "std", "q01", "q99"]:
            value = getattr(stats[key], field)
            if value is None or value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid official {key}.{field}")
        if not np.all(stats[key].std >= 0) or not np.all(stats[key].q99 >= stats[key].q01):
            raise ValueError(f"Invalid official {key} statistics ranges")
    write_manifest(config, "stats_probe" if partial else "stats_full", output / "pipeline_config.json")
    metadata = json.loads((WORKSPACE / "datasets/libero/meta/info.json").read_text())
    total = metadata["total_frames"]
    requested = min(args.max_frames, total) if partial else total
    processed = requested // config.batch_size * config.batch_size
    report = {"status": "PASS", "official_entry": "scripts/compute_norm_stats.py:main",
              "probe_only": partial, "batch_size": config.batch_size,
              "num_workers": config.num_workers, "elapsed_seconds": round(elapsed, 3),
              "frames_used": processed, "frames_available": total,
              "tail_frames_dropped": requested - processed,
              "action_vectors_used": processed * config.model.action_horizon,
              "stats_path": str(output / "norm_stats.json"),
              "sha256": hashlib.sha256((output / "norm_stats.json").read_bytes()).hexdigest()}
    (output / "statistics_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
