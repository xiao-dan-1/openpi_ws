"""Compute/validate candidate pi05 LIBERO h10 stats using bounded-memory reads.

Numeric transforms match pi05_libero: no extra delta transform; state/actions
are unchanged before normalization. Uses all frames, per-episode RunningStats
updates; not bitwise-equivalent to the official batched script which drops tail
frames. Writes validation artifacts only, not active training assets/config.
Requires a new output directory outside the dataset; failed runs retain an
INCOMPLETE marker and previous output directories are never overwritten.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if __package__:
    from .libero_provenance import (PROFILE, load_metadata, new_numeric_digest,
                                    numeric_episodes, update_numeric_digest)
else:
    from libero_provenance import (PROFILE, load_metadata, new_numeric_digest,
                                   numeric_episodes, update_numeric_digest)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_output(output, root):
    output, root = output.resolve(), root.resolve()
    require(not output.is_relative_to(root), "Output directory must be outside --data-root")
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").write_text("Normalization validation has not completed.\n",
                                       encoding="utf-8")
    return output


def main(argv=None):
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=workspace / "datasets/libero")
    parser.add_argument("--output-dir", type=Path,
                        default=workspace / "checks/output/pi05_libero_h10",
                        help="New directory outside --data-root; existing paths are refused")
    args = parser.parse_args(argv)
    root = args.data_root.resolve()
    require((root / "meta/info.json").is_file(), f"Dataset is absent: {root}")
    output = prepare_output(args.output_dir, root)
    sys.dont_write_bytecode = True
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})

    import numpy as np
    import pyarrow as pa
    from openpi.shared.normalize import RunningStats, load, save
    from openpi.transforms import Normalize, Unnormalize

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    info, episodes, tasks = load_metadata(root)
    require(len(episodes) == 1693 and info["fps"] == 10, "Unexpected data profile")
    accumulators = {key: RunningStats() for key in ["state", "actions"]}
    sums = {"state": np.zeros(8), "actions": np.zeros(7)}
    counts = {"state": 0, "actions": 0}
    source_digest = new_numeric_digest(info, episodes, tasks)
    samples = []
    for index, state, actions, timestamp in numeric_episodes(root, info, episodes):
        length = len(state)
        indices = np.minimum(np.arange(length)[:, None] + np.arange(10), length - 1)
        windows = actions[indices]
        for key, values in [("state", state), ("actions", windows)]:
            require(np.isfinite(values).all(), f"Episode {index}: invalid {key}")
            accumulators[key].update(values)
            flat = values.reshape(-1, values.shape[-1])
            sums[key] += flat.sum(axis=0, dtype=np.float64)
            counts[key] += len(flat)
        update_numeric_digest(source_digest, state, actions, timestamp)
        if index in [0, 846, 1692]:
            for frame in sorted({0, length // 2, length - 1}):
                samples.append({"state": state[frame], "actions": windows[frame]})
        if (index + 1) % 100 == 0 or index == 1692:
            print(f"Statistics: {index + 1}/1693 episodes", file=sys.stderr, flush=True)

    require(counts == {"state": 273465, "actions": 2734650}, "Incorrect statistics coverage")
    stats = {key: value.get_statistics() for key, value in accumulators.items()}
    for key, dimension in [("state", 8), ("actions", 7)]:
        entry = stats[key]
        for name in ["mean", "std", "q01", "q99"]:
            array = getattr(entry, name)
            require(array.shape == (dimension,) and np.isfinite(array).all(),
                    f"Invalid {key}.{name}")
        require(np.all(entry.std >= 0) and np.all(entry.q99 >= entry.q01), "Invalid stats ranges")
        require(np.allclose(entry.mean, sums[key] / counts[key], atol=5e-5, rtol=1e-5),
                f"{key} mean differs from independent float64 sum")

    normalize = Normalize(stats, use_quantiles=True, strict=True)
    restore = Unnormalize(stats, use_quantiles=True)
    max_error = 0.0
    for sample in samples:
        normalized = normalize(sample)
        recovered = restore(normalized)
        for key in ["state", "actions"]:
            require(np.isfinite(normalized[key]).all(), "Non-finite normalized values")
            error = float(np.max(np.abs(recovered[key] - sample[key])))
            require(error < 1e-5, f"Excessive {key} roundtrip error: {error}")
            max_error = max(max_error, error)

    save(output, stats)
    reloaded = load(output)
    for key in stats:
        for name in ["mean", "std", "q01", "q99"]:
            require(np.array_equal(getattr(stats[key], name), getattr(reloaded[key], name)),
                    "Stats changed during serialization")
    report = {
        "status": "PASS", "validation_profile": PROFILE,
        "cpu_only": True, "episodes": len(episodes), "vector_counts": counts,
        "stat_dimensions": {"state": 8, "actions": 7},
        "numeric_data_sha256": source_digest.hexdigest(),
        "norm_stats_sha256": hashlib.sha256((output / "norm_stats.json").read_bytes()).hexdigest(),
        "roundtrip_samples": len(samples), "max_roundtrip_error": max_error,
        "stats_path": str(output / "norm_stats.json"),
        "aggregation": "All frames, per-episode official RunningStats; histogram-approximate quantiles",
        "scope": "Candidate normalization stats; final training configuration, tokenizer and model not tested",
    }
    (output / "validation_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    (output / "INCOMPLETE").unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
