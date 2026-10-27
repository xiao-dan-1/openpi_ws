"""Compute/validate candidate pi05 LIBERO h10 stats using bounded-memory reads.

Numeric transforms match pi05_libero: no extra delta transform; state/actions
are unchanged before normalization. Uses all frames, per-episode RunningStats
updates; not bitwise-equivalent to the official batched script which drops tail
frames. Writes validation artifacts only, not active training assets/config.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                   "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from openpi.shared.normalize import RunningStats, load, save
    from openpi.transforms import Normalize, Unnormalize

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=workspace / "datasets/libero")
    parser.add_argument("--output-dir", type=Path,
                        default=workspace / "checks/output/pi05_libero_h10")
    args = parser.parse_args()
    root = args.data_root.resolve()
    with (root / "meta/info.json").open() as stream:
        info = json.load(stream)
    with (root / "meta/episodes.jsonl").open() as stream:
        episodes = [json.loads(line) for line in stream if line.strip()]
    with (root / "meta/tasks.jsonl").open() as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    episodes.sort(key=lambda item: item["episode_index"])
    require(len(episodes) == 1693 and info["fps"] == 10, "Unexpected data profile")
    accumulators = {key: RunningStats() for key in ["state", "actions"]}
    sums = {"state": np.zeros(8), "actions": np.zeros(7)}
    counts = {"state": 0, "actions": 0}
    source_digest = hashlib.sha256()
    source_digest.update(json.dumps({"tasks": tasks, "episodes": episodes, "fps": info["fps"]},
                                    sort_keys=True, separators=(",", ":")).encode())
    samples = []
    for episode in episodes:
        index = episode["episode_index"]
        path = root / info["data_path"].format(episode_index=index,
                                               episode_chunk=index // info["chunks_size"])
        table = pq.ParquetFile(path).read(columns=["state", "actions", "timestamp"])
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        length = table.num_rows
        require(length == episode["length"] and state.shape == (length, 8)
                and actions.shape == (length, 7), f"Episode {index}: wrong shape")
        indices = np.minimum(np.arange(length)[:, None] + np.arange(10), length - 1)
        windows = actions[indices]
        for key, values in [("state", state), ("actions", windows)]:
            require(np.isfinite(values).all(), f"Episode {index}: invalid {key}")
            accumulators[key].update(values)
            flat = values.reshape(-1, values.shape[-1])
            sums[key] += flat.sum(axis=0, dtype=np.float64)
            counts[key] += len(flat)
        for array in [state, actions, table["timestamp"].to_numpy()]:
            source_digest.update(np.asarray(array, dtype="<f4").tobytes())
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

    output = args.output_dir.resolve()
    save(output, stats)
    reloaded = load(output)
    for key in stats:
        for name in ["mean", "std", "q01", "q99"]:
            require(np.array_equal(getattr(stats[key], name), getattr(reloaded[key], name)),
                    "Stats changed during serialization")
    report = {
        "status": "PASS", "validation_profile": "pi05_libero, h10, extra_delta_transform=False",
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


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
