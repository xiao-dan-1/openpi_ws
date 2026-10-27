"""Read-only LIBERO sample check; no model imports, downloads, or output files."""

import argparse
import io
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_episode(root: Path, info: dict, episode: dict, tasks: dict) -> dict:
    episode_id = episode["episode_index"]
    relative_path = info["data_path"].format(
        episode_chunk=episode_id // info["chunks_size"],
        episode_index=episode_id,
    )
    path = root / relative_path
    columns = ["image", "wrist_image", "state", "actions", "episode_index",
               "frame_index", "task_index", "timestamp"]
    table = pq.ParquetFile(path).read(columns=columns)
    count = table.num_rows
    require(count == episode["length"] and count > 0,
            f"Episode {episode_id}: frame count differs from metadata")
    require(set(table["episode_index"].to_pylist()) == {episode_id},
            f"Episode {episode_id}: incorrect episode_index")

    shapes = {}
    for key, dimension in [("state", 8), ("actions", 7)]:
        stored_type = table.schema.field(key).type
        require(getattr(stored_type, "value_type", None) == pa.float32(),
                f"Episode {episode_id}: {key} stored type is {stored_type}, expected float32 list")
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        require(values.shape == (count, dimension),
                f"Episode {episode_id}: unexpected {key} shape {values.shape}")
        require(bool(np.isfinite(values).all()),
                f"Episode {episode_id}: {key} contains NaN or infinity")
        shapes[key] = list(values.shape)

    frame_indices = np.asarray(table["frame_index"].to_pylist())
    require(bool(np.array_equal(frame_indices, np.arange(count))),
            f"Episode {episode_id}: frame_index is not consecutive from zero")
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    require(bool(np.allclose(timestamps, np.arange(count) / info["fps"], atol=1e-4, rtol=0)),
            f"Episode {episode_id}: timestamps do not match fps")

    task_indices = sorted(set(table["task_index"].to_pylist()))
    require(all(index in tasks for index in task_indices),
            f"Episode {episode_id}: unknown task_index")
    texts = sorted({tasks[index] for index in task_indices})
    require(set(texts) == set(episode["tasks"]),
            f"Episode {episode_id}: task text differs from episode metadata")

    sampled_frames = sorted({0, count // 2, count - 1})
    for frame_index in sampled_frames:
        row = table.slice(frame_index, 1).to_pylist()[0]
        for key in ["image", "wrist_image"]:
            cell = row[key]
            require(isinstance(cell, dict) and bool(cell.get("bytes")),
                    f"Episode {episode_id}, frame {frame_index}: {key} has no embedded image bytes")
            with Image.open(io.BytesIO(cell["bytes"])) as image:
                image.load()
                pixels = np.asarray(image)
                require(image.mode == "RGB" and pixels.shape == (256, 256, 3)
                        and pixels.dtype == np.uint8,
                        f"Episode {episode_id}, frame {frame_index}: unexpected {key} image format")

    return {
        "episode_index": episode_id,
        "frames": count,
        "state_shape": shapes["state"],
        "actions_shape": shapes["actions"],
        "task_indices": task_indices,
        "tasks": texts,
        "decoded_frame_indices": sampled_frames,
        "both_cameras": "RGB uint8, 256x256x3",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets" / "libero")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    root = args.data_root.resolve()
    with (root / "meta/info.json").open(encoding="utf-8") as stream:
        info = json.load(stream)
    require(info["codebase_version"] == "v2.0" and info["fps"] == 10,
            "Expected LIBERO LeRobot v2.0 data at 10 fps")
    episodes = {item["episode_index"]: item for item in read_jsonl(root / "meta/episodes.jsonl")}
    tasks = {item["task_index"]: item["task"] for item in read_jsonl(root / "meta/tasks.jsonl")}
    require(len(episodes) == info["total_episodes"] and len(tasks) == info["total_tasks"],
            "Metadata record counts differ from info.json")
    require(all(index in episodes for index in args.episodes), "Requested episode is absent")

    report = {
        "status": "PASS",
        "scope": "Raw data sample check; model loader, normalization, and GPU not tested",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "data_root": str(root),
        "episodes": [check_episode(root, info, episodes[index], tasks)
                     for index in dict.fromkeys(args.episodes)],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
