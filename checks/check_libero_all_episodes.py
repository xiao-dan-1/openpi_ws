"""Bounded-memory numeric/metadata and LeRobot window-index checks; CPU/offline.

Reads numeric columns one episode at a time. Does not build an Arrow cache or
instantiate the full dataset loader. Image decoding is covered by earlier checks.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import get_episode_data_index

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets/libero")
    root = parser.parse_args().data_root.resolve()
    with (root / "meta/info.json").open() as stream:
        info = json.load(stream)
    with (root / "meta/episodes.jsonl").open() as stream:
        episodes = {item["episode_index"]: item for item in map(json.loads, stream)}
    with (root / "meta/tasks.jsonl").open() as stream:
        tasks = {item["task_index"]: item["task"] for item in map(json.loads, stream)}
    require(len(episodes) == info["total_episodes"] == 1693, "Incorrect episode count")
    require(len(tasks) == info["total_tasks"] == 40, "Incorrect task count")
    require(sorted(episodes) == list(range(len(episodes))), "Non-consecutive episode IDs")
    # Exercise the actual pinned LeRobot index planner with full boundary arrays.
    # This is not a substitute for constructing its complete Dataset object.
    planner = SimpleNamespace(episode_data_index=get_episode_data_index(episodes),
                              delta_indices={"actions": list(range(10))})
    total = windows = 0
    for episode_id in sorted(episodes):
        metadata = episodes[episode_id]
        path = root / info["data_path"].format(
            episode_index=episode_id, episode_chunk=episode_id // info["chunks_size"])
        parquet = pq.ParquetFile(path)
        require({"image", "wrist_image"}.issubset(parquet.schema_arrow.names),
                f"Episode {episode_id}: missing camera columns")
        table = parquet.read(columns=["state", "actions", "episode_index", "frame_index",
                                      "task_index", "timestamp"])
        count = table.num_rows
        require(count == metadata["length"] and count > 0, f"Episode {episode_id}: wrong length")
        vectors = {}
        for key, dimension in [("state", 8), ("actions", 7)]:
            require(table.schema.field(key).type.value_type == pa.float32(), f"Incorrect {key} dtype")
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            require(values.shape == (count, dimension) and np.isfinite(values).all(),
                    f"Episode {episode_id}: invalid {key} shape or values")
            vectors[key] = values
        require(set(table["episode_index"].to_pylist()) == {episode_id}, "Mixed episode IDs")
        require(np.array_equal(table["frame_index"].to_numpy(), np.arange(count)), "Wrong frame order")
        require(np.allclose(table["timestamp"].to_numpy(), np.arange(count) / info["fps"],
                            atol=1e-4, rtol=0), f"Episode {episode_id}: wrong timestamps")
        texts = {tasks[index] for index in table["task_index"].to_pylist()}
        require(texts == set(metadata["tasks"]), f"Episode {episode_id}: wrong tasks")
        require(int(planner.episode_data_index["from"][episode_id]) == total
                and int(planner.episode_data_index["to"][episode_id]) == total + count,
                f"Episode {episode_id}: incorrect global offsets")
        for local_index in sorted({0, count - 1}):
            query, padding = LeRobotDataset._get_query_indices(planner, total + local_index, episode_id)
            expected = np.minimum(np.arange(local_index, local_index + 10), count - 1)
            require(np.array_equal(query["actions"], total + expected),
                    f"Episode {episode_id}: cross-episode query or wrong padding indices")
            require(np.array_equal(padding["actions_is_pad"].numpy(),
                                   np.arange(local_index, local_index + 10) >= count),
                    f"Episode {episode_id}: wrong padding mask")
            require(vectors["actions"][expected].shape == (10, 7), "Invalid boundary action shape")
            windows += 1
        total += count
        if (episode_id + 1) % 100 == 0 or episode_id == 1692:
            print(f"Checked {episode_id + 1}/1693 episodes", file=sys.stderr, flush=True)
    require(total == info["total_frames"] == 273465, "Incorrect total frame count")
    print(json.dumps({"status": "PASS", "data_root": str(root), "cpu_only": True,
                      "offline": True, "episodes_checked": len(episodes), "frames_checked": total,
                      "tasks": len(tasks), "boundary_windows_checked": windows,
                      "window_shape": [10, 7], "cross_episode_query": False,
                      "scope": "All numeric rows, metadata, and real LeRobot index planner; "
                               "complete Dataset construction, all-frame image decoding, normalization, "
                               "tokenizer and model execution are not covered"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
