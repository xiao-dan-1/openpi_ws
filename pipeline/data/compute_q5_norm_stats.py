"""Compute OpenPI statistics for explicit Q5 absolute-joint action windows.

Reads Parquet only; uses upstream RunningStats for moments and exact weighted
quantiles by default, with upstream serialization. Includes every selected frame.
This is not the upstream compute_norm_stats.main nor a Q5 model configuration.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", default="train")
    parser.add_argument("--quantiles", choices=["exact", "histogram"], default="exact")
    args = parser.parse_args(argv)
    require(args.action_horizon > 0 and args.batch_size > 0, "Horizon and batch size must be positive")
    root, upstream, out = args.data_root.resolve(), args.openpi_root.resolve(), args.output_dir.resolve()
    require(not out.exists(), f"Refusing to overwrite an existing output: {out}")
    require(out != root and root not in out.parents, "Statistics must be stored outside the source dataset")
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", HF_HUB_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      OPENBLAS_NUM_THREADS="2", HF_HUB_DISABLE_TELEMETRY="1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream / "src"))
    import numpy as np
    import pyarrow.parquet as pq
    from openpi.shared import normalize
    from openpi import transforms

    require(Path(normalize.__file__).resolve() == upstream / "src/openpi/shared/normalize.py",
            "Unexpected OpenPI normalization implementation")
    start_time = time.monotonic()
    info = json.loads((root / "meta/info.json").read_text())
    require(info["robot_type"] == "q5" and info["codebase_version"] == "v2.1", "Expected Q5 LeRobot v2.1")
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    begin, end = map(int, info["splits"][args.split].split(":"))
    selected = [e for e in episodes if begin <= e["episode_index"] < end]
    require(bool(selected), "Selected split is empty")
    action_names = info["features"]["action"]["names"]
    state_names = info["features"]["observation.state"]["names"]
    require(action_names == state_names and len(action_names) == 38, "Q5 joint order/size mismatch")
    states, actions, window_ids, fingerprints = [], [], [], []
    offset = 0
    for episode in selected:
        eid, n = episode["episode_index"], episode["length"]
        path = root / info["data_path"].format(episode_index=eid, episode_chunk=eid // info["chunks_size"])
        t = pq.read_table(path)
        require(len(t) == n and np.all(t["episode_index"].to_numpy() == eid), f"Episode metadata mismatch: {eid}")
        require(np.array_equal(t["frame_index"].to_numpy(), np.arange(n)), f"Frame index mismatch: {eid}")
        for key, chunks in [("observation.state", states), ("action", actions)]:
            a = np.asarray(t[key].to_pylist(), dtype=np.float64)
            require(a.shape == (n, 38) and np.isfinite(a).all(), f"Invalid {key}: {eid}")
            chunks.append(a)
        # Matches the already validated LeRobot forward window and end padding.
        window_ids.append(offset + np.minimum(np.arange(n)[:, None] + np.arange(args.action_horizon), n - 1))
        offset += n
        fingerprints.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
    state = np.concatenate(states)
    action = np.concatenate(actions)
    ids = np.concatenate(window_ids)
    weights = np.bincount(ids.ravel(), minlength=len(action)).astype(np.int64)
    require(int(weights.sum()) == len(state) * args.action_horizon, "Action window count mismatch")
    running = {key: normalize.RunningStats() for key in ["state", "actions"]}
    for start in range(0, len(state), args.batch_size):
        stop = min(start + args.batch_size, len(state))
        running["state"].update(state[start:stop])
        running["actions"].update(action[ids[start:stop]])
    stats = {k: value.get_statistics() for k, value in running.items()}

    def exact_quantiles(x, w):
        count = int(w.sum())
        result = np.empty((2, x.shape[1]))
        for j in range(x.shape[1]):
            order = np.argsort(x[:, j], kind="stable")
            values, cumulative = x[order, j], np.cumsum(w[order])
            for k, q in enumerate([0.01, 0.99]):
                position = q * (count - 1)
                lo, hi = int(np.floor(position)), int(np.ceil(position))
                low = values[np.searchsorted(cumulative, lo, side="right")]
                high = values[np.searchsorted(cumulative, hi, side="right")]
                result[k, j] = low + (high - low) * (position - lo)
        return result

    validations = {}
    for key, x, w in [("state", state, np.ones(len(state), dtype=np.int64)), ("actions", action, weights)]:
        s = stats[key]
        for name in ["mean", "std", "q01", "q99"]:
            value = getattr(s, name)
            require(value.shape == (38,) and np.isfinite(value).all(), f"Invalid {key}.{name}")
        require(np.all(s.std >= 0) and np.all(s.q99 >= s.q01), f"Invalid ranges: {key}")
        mean = np.average(x, axis=0, weights=w)
        std = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
        require(np.allclose(s.mean, mean, atol=1e-9, rtol=1e-9), f"Independent mean mismatch: {key}")
        require(np.allclose(s.std, std, atol=1e-7, rtol=1e-7), f"Independent std mismatch: {key}")
        quantiles = exact_quantiles(x, w)
        q_error = float(np.max(np.abs(np.stack([s.q01, s.q99]) - quantiles)))
        per_joint_error = np.max(np.abs(np.stack([s.q01, s.q99]) - quantiles), axis=0)
        if args.quantiles == "exact":
            s.q01, s.q99 = quantiles[0], quantiles[1]
        require(np.all(s.q01 >= x.min(0) - 1e-8) and np.all(s.q99 <= x.max(0) + 1e-8), f"Quantiles outside observed bounds: {key}")
        validations[key] = {"mean_max_abs_error": float(np.max(np.abs(s.mean - mean))),
                            "std_max_abs_error": float(np.max(np.abs(s.std - std))),
                            "histogram_quantile_max_abs_error_vs_exact": q_error,
                            "histogram_largest_error_joint": action_names[int(np.argmax(per_joint_error))],
                            "saved_quantile_max_abs_error_vs_exact": float(np.max(np.abs(np.stack([s.q01, s.q99]) - quantiles)))}
    left_hand = np.array([i for i, name in enumerate(action_names) if name.startswith("left_hand_")])
    require(len(left_hand) == 12 and np.all(action[:, left_hand] == 0), "Expected left-hand zero commands")
    require(np.all(stats["actions"].std[left_hand] == 0), "Zero-action channels should retain zero std")
    for quantile in [False, True]:
        values = {"state": state, "actions": action}
        normalized = transforms.Normalize(stats, use_quantiles=quantile, strict=True)(values)
        require(all(np.isfinite(v).all() for v in normalized.values()), "Non-finite normalized data")
        recovered = transforms.Unnormalize(stats, use_quantiles=quantile)(normalized)
        for key, expected in [("state", state), ("actions", action)]:
            error = float(np.max(np.abs(recovered[key] - expected)))
            require(error < 1e-9, f"Normalize/unnormalize roundtrip mismatch: {key}")
            validations[key][f"{'quantile' if quantile else 'zscore'}_roundtrip_max_abs_error"] = error
    # Source fingerprints are also checked after computation, before publishing assets.
    for f in fingerprints:
        require(digest(root / f["path"]) == f["sha256"], "Source changed during computation")
    out.mkdir(parents=True)
    marker = out / "INCOMPLETE"
    marker.write_text("Do not use until validation completes.\n")
    normalize.save(out, stats)
    restored = normalize.load(out)
    for key in stats:
        for field in ["mean", "std", "q01", "q99"]:
            require(np.array_equal(getattr(stats[key], field), getattr(restored[key], field)), "Serialization mismatch")
    report = {"status": "PASS", "dataset": str(root), "split": args.split,
              "selected_episode_ids": [e["episode_index"] for e in selected],
              "episodes_used": len(selected), "frames_used": len(state), "frames_available": info["total_frames"],
              "tail_frames_dropped": 0, "state_vectors_used": len(state),
              "action_vectors_used": int(weights.sum()), "action_horizon": args.action_horizon,
              "action_representation": "absolute_joint_targets", "delta_transform": False,
              "joint_dimension": 38, "joint_names": action_names, "padding": "repeat final action within each episode",
              "stats_keys": ["state", "actions"], "batch_size": args.batch_size,
              "statistics_implementation": "openpi.shared.normalize.RunningStats moments and serialization",
              "quantile_method": "exact weighted linear quantiles" if args.quantiles == "exact" else "OpenPI histogram quantiles (5000 bins)",
              "upstream_main_called": False, "all_selected_frames_used": True,
              "upstream_commit": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
              "normalize_source_sha256": digest(Path(normalize.__file__)),
              "transform_source_sha256": digest(Path(transforms.__file__)),
              "stats_path": str(out / "norm_stats.json"), "sha256": digest(out / "norm_stats.json"),
              "zero_std_action_joint_names": [action_names[i] for i in np.flatnonzero(stats["actions"].std == 0)],
              "validation": validations, "source_parquet_fingerprints": fingerprints,
              "source_info_sha256": digest(root / "meta/info.json"),
              "elapsed_seconds": round(time.monotonic() - start_time, 3),
              "scope": "Q5 absolute commands, all currently marked training episodes, horizon as recorded; no model training/configuration.",
              "reuse_conditions": ["Joint mapping, action representation, horizon and training split must match.",
                                   "Recompute for delta actions or a changed train/validation split.",
                                   "38-dimensional statistics alone do not adapt a 32-dimensional pretrained model."]}
    (out / "statistics_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    marker.unlink()
    print(json.dumps({k: v for k, v in report.items() if k not in ["source_parquet_fingerprints", "selected_episode_ids", "joint_names"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
