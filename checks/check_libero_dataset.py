"""Construct the complete local LeRobot Dataset, then read real action windows.

CPU and offline. Reuses --cache-dir or creates reusable Arrow cache there.
Source Parquet/metadata and training code are not modified.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=workspace / "datasets/libero")
    parser.add_argument("--cache-dir", type=Path,
                        default=workspace / ".cache/huggingface/datasets")
    args = parser.parse_args()
    root = args.data_root.resolve()
    require((root / "meta/info.json").is_file(), f"Dataset is absent: {root}")
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_CACHE": str(args.cache_dir.resolve()),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    })
    os.environ.pop("LEROBOT_HOME", None)

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.transforms import PromptFromLeRobotTask

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    torch.set_num_threads(2)
    print("Constructing complete Dataset (first run may build a ~35 GB cache)...",
          file=sys.stderr, flush=True)
    dataset = LeRobotDataset(
        "physical-intelligence/libero", root=root, episodes=None,
        revision="a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        download_videos=False, delta_timestamps={"actions": [i / 10 for i in range(10)]},
    )
    require(len(dataset) == dataset.meta.total_frames == 273465, "Incorrect frame count")
    require(dataset.num_episodes == 1693 and dataset.episodes is None, "Dataset is not complete")
    prompt_transform = PromptFromLeRobotTask(dataset.meta.tasks)
    results = []
    for episode_id in [0, 846, 1692]:
        length = dataset.meta.episodes[episode_id]["length"]
        start = int(dataset.episode_data_index["from"][episode_id])
        reference = np.asarray(pq.ParquetFile(root / dataset.meta.get_data_file_path(episode_id))
                               .read(columns=["actions"])["actions"].to_pylist(), dtype=np.float32)
        frames = sorted({0, length // 2, length - 1})
        for frame in frames:
            item = prompt_transform(dataset[start + frame])
            require(int(item["episode_index"]) == episode_id and int(item["frame_index"]) == frame,
                    "Dataset returned the wrong frame")
            expected_indices = np.minimum(np.arange(frame, frame + 10), length - 1)
            require(tuple(item["actions"].shape) == (10, 7)
                    and np.allclose(item["actions"].numpy(), reference[expected_indices], atol=1e-7, rtol=0),
                    f"Episode {episode_id}, frame {frame}: wrong actions")
            require(np.array_equal(item["actions_is_pad"].numpy(), np.arange(frame, frame + 10) >= length),
                    "Wrong end-of-episode padding")
            require(item["prompt"] in dataset.meta.episodes[episode_id]["tasks"], "Wrong prompt")
            for key in ["image", "wrist_image"]:
                require(tuple(item[key].shape) == (3, 256, 256) and torch.isfinite(item[key]).all(),
                        f"Invalid decoded {key}")
        results.append({"episode_index": episode_id, "checked_frames": frames,
                        "real_action_windows_checked": len(frames)})
    print(json.dumps({
        "status": "PASS", "full_dataset_constructed": True,
        "frames_loaded": len(dataset), "episodes_loaded": dataset.num_episodes,
        "cpu_only": True, "offline": True,
        "cache_dir": str(args.cache_dir.resolve()), "real_window_shape": [10, 7],
        "episodes": results,
        "scope": "Complete LeRobot Dataset construction and 9 actual frame/window reads; "
                 "normalization, tokenizer, configured complete OpenPI batch and model not tested",
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
