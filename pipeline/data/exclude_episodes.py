"""Create an independent LeRobot v2.1 copy with selected episodes removed.

Never edits the source or overwrites an existing destination. Episode numbers,
global frame indices, splits and associated statistics are rebuilt together.
"""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar_stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"min": [float(a.min())], "max": [float(a.max())],
            "mean": [float(a.mean())], "std": [float(a.std())], "count": [len(a)]}


def run(source, destination, excluded):
    source, destination = source.resolve(), destination.resolve()
    require(source.is_dir(), f"Source is absent: {source}")
    require(source != destination and source not in destination.parents,
            "Destination must be outside the source dataset")
    require(not destination.exists(), f"Destination already exists; refusing to overwrite: {destination}")
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    require(info["codebase_version"] == "v2.1", "Only LeRobot v2.1 is supported")
    # Do not silently drop alternate assets, global statistics or unknown metadata.
    require({p.name for p in source.iterdir()} <= {"data", "meta", "videos", "cleaning_manifest.json"},
            "Unexpected source files; review their migration before proceeding")
    require({p.name for p in (source / "meta").iterdir()} <=
            {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl", "timing"},
            "Unsupported extra metadata (including global stats); review before proceeding")
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    original_ids = [e["episode_index"] for e in episodes]
    require(original_ids == list(range(info["total_episodes"])), "Source episode IDs must be consecutive")
    require(excluded <= set(original_ids), f"Unknown excluded episodes: {excluded - set(original_ids)}")
    kept = [e for e in episodes if e["episode_index"] not in excluded]
    require(bool(kept), "Cannot exclude every episode")
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    chunk_size = info["chunks_size"]

    def relative(key, eid, video_key=None):
        p = Path(info[key].format(episode_index=eid, episode_chunk=eid // chunk_size, video_key=video_key))
        require(not p.is_absolute() and ".." not in p.parts, "Unsafe source path template")
        return p

    old_stats = {e["episode_index"]: e for e in read_jsonl(source / "meta/episodes_stats.jsonl")}
    require(set(old_stats) == set(original_ids), "Source episode stats are incomplete")
    remap = {e["episode_index"]: i for i, e in enumerate(kept)}
    new_info = copy.deepcopy(info)
    new_info.update(total_episodes=len(kept), total_frames=sum(e["length"] for e in kept),
                    total_videos=len(kept) * len(video_keys), total_chunks=(len(kept) - 1) // chunk_size + 1)
    new_info["splits"] = {}
    for split, span in info["splits"].items():
        start, stop = map(int, span.split(":"))
        selected = [new for old, new in remap.items() if start <= old < stop]
        if selected:
            require(selected == list(range(min(selected), max(selected) + 1)), f"Split {split} became noncontiguous")
            new_info["splits"][split] = f"{min(selected)}:{max(selected) + 1}"

    # Preflight every referenced asset before creating any output.
    for episode in episodes:
        eid = episode["episode_index"]
        require((source / relative("data_path", eid)).is_file(), f"Missing source Parquet {eid}")
        for key in video_keys:
            require((source / relative("video_path", eid, key)).is_file(), f"Missing source video {eid}: {key}")
    destination.mkdir(parents=True)
    marker = destination / "INCOMPLETE"
    marker.write_text("Do not use this directory until validation completes.\n", encoding="utf-8")
    (destination / "meta").mkdir()
    shutil.copy2(source / "meta/tasks.jsonl", destination / "meta/tasks.jsonl")
    output_episodes, output_stats, mappings = [], [], []
    global_start = 0
    checked_videos = 0
    for episode in kept:
        old, n = episode["episode_index"], episode["length"]
        new = remap[old]
        src_parquet = source / relative("data_path", old)
        dst_parquet = destination / relative("data_path", new)
        table = pq.read_table(src_parquet)
        require(len(table) == n, f"Source length mismatch: {old}")
        require(np.array_equal(table["frame_index"].to_numpy(), np.arange(n)), f"Source frame indices: {old}")
        require(np.all(table["episode_index"].to_numpy() == old), f"Source episode indices: {old}")
        original = table
        for key, values in [("episode_index", np.full(n, new, dtype=np.int64)),
                            ("index", np.arange(global_start, global_start + n, dtype=np.int64))]:
            column = table.schema.get_field_index(key)
            table = table.set_column(column, table.schema.field(key), pa.array(values, type=table.schema.field(key).type))
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_parquet)
        roundtrip = pq.read_table(dst_parquet)
        require(table.equals(roundtrip), f"Parquet write verification failed: {new}")
        for key in original.column_names:
            if key not in {"episode_index", "index"}:
                require(original[key].equals(roundtrip[key]), f"Unexpected content change: {new}/{key}")
        new_episode = copy.deepcopy(episode)
        new_episode["episode_index"] = new
        output_episodes.append(new_episode)
        stats = copy.deepcopy(old_stats[old])
        stats["episode_index"] = new
        for key in ["episode_index", "index"]:
            stats["stats"][key] = scalar_stats(table[key].to_numpy())
        output_stats.append(stats)
        for key in video_keys:
            src_video = source / relative("video_path", old, key)
            dst_video = destination / relative("video_path", new, key)
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)
            require(file_hash(src_video) == file_hash(dst_video), f"Video copy verification failed: {new}")
            checked_videos += 1
        timing = source / "meta/timing" / f"episode_{old:06d}.json"
        if timing.is_file():
            dst_timing = destination / "meta/timing" / f"episode_{new:06d}.json"
            dst_timing.parent.mkdir(exist_ok=True)
            shutil.copy2(timing, dst_timing)
            require(file_hash(timing) == file_hash(dst_timing), f"Timing copy verification failed: {new}")
        mappings.append({"original_episode_index": old, "new_episode_index": new,
                         "length": n, "new_global_start": global_start})
        global_start += n
    write_json(destination / "meta/info.json", new_info)
    write_jsonl(destination / "meta/episodes.jsonl", output_episodes)
    write_jsonl(destination / "meta/episodes_stats.jsonl", output_stats)
    require(global_start == new_info["total_frames"], "Incorrect output frame count")
    require(len(list((destination / "data").rglob("*.parquet"))) == len(kept), "Incorrect output Parquet count")
    require(len(list((destination / "videos").rglob("*.mp4"))) == checked_videos, "Incorrect output video count")
    manifest = {"source": str(source), "destination": str(destination), "source_modified": False,
                "excluded_original_episodes": sorted(excluded), "episode_mapping": mappings,
                "validation": {"parquet_roundtrips": len(kept), "unchanged_video_sha256": checked_videos,
                               "non_index_columns_unchanged": True, "episodes": len(kept), "frames": global_start},
                "note": "Timing sidecars retain original source_bag paths for provenance. No new normalization assets generated."}
    write_json(destination / "cleaning_manifest.json", manifest)
    marker.unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude", required=True, nargs="+", type=int)
    args = parser.parse_args()
    result = run(args.source, args.output, set(args.exclude))
    print(json.dumps({k: v for k, v in result.items() if k != "episode_mapping"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
