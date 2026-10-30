"""Numeric provenance shared by the LIBERO stats producer and consumer.

The digest retains the original stats check's encoding: canonical metadata,
then state, actions and timestamps as little-endian float32, episode by episode.
Only numeric Parquet columns are read, with one episode resident at a time.
"""
import hashlib
import json


PROFILE = "pi05_libero, h10, extra_delta_transform=False"


def load_metadata(root):
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = [json.loads(line) for line in
                (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    tasks = [json.loads(line) for line in
             (root / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    episodes.sort(key=lambda item: item["episode_index"])
    return info, episodes, tasks


def new_numeric_digest(info, episodes, tasks):
    digest = hashlib.sha256()
    digest.update(json.dumps({"tasks": tasks, "episodes": episodes, "fps": info["fps"]},
                             sort_keys=True, separators=(",", ":")).encode())
    return digest


def numeric_episodes(root, info, episodes):
    import numpy as np
    import pyarrow.parquet as pq

    for episode in episodes:
        index = episode["episode_index"]
        path = root / info["data_path"].format(
            episode_index=index, episode_chunk=index // info["chunks_size"])
        with pq.ParquetFile(path) as parquet:
            table = parquet.read(columns=["state", "actions", "timestamp"], use_threads=False)
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        timestamp = table["timestamp"].to_numpy()
        length = table.num_rows
        if not (length > 0 and length == episode["length"]
                and state.shape == (length, 8) and actions.shape == (length, 7)
                and timestamp.shape == (length,)):
            raise ValueError(f"Episode {index}: wrong shape")
        if not all(np.isfinite(array).all() for array in (state, actions, timestamp)):
            raise ValueError(f"Episode {index}: invalid numeric data")
        yield index, state, actions, timestamp


def update_numeric_digest(digest, state, actions, timestamp):
    import numpy as np

    for array in (state, actions, timestamp):
        digest.update(np.asarray(array, dtype="<f4").tobytes())


def validate_stats(stats_dir, data_root):
    """Reject unfinished/changed artifacts and bind stats to current source data."""
    if (stats_dir / "INCOMPLETE").exists():
        raise ValueError("Candidate normalization artifacts are INCOMPLETE")
    manifest_path = stats_dir / "validation_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Run the normalization check first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("validation_profile") != PROFILE:
        raise ValueError("Wrong candidate normalization profile")
    if hashlib.sha256((stats_dir / "norm_stats.json").read_bytes()).hexdigest() != manifest.get(
            "norm_stats_sha256"):
        raise ValueError("Candidate stats changed since validation")
    info, episodes, tasks = load_metadata(data_root)
    digest = new_numeric_digest(info, episodes, tasks)
    for _, state, actions, timestamp in numeric_episodes(data_root, info, episodes):
        update_numeric_digest(digest, state, actions, timestamp)
    if digest.hexdigest() != manifest.get("numeric_data_sha256"):
        raise ValueError("Candidate stats do not match --data-root numeric data and metadata")
    return manifest
