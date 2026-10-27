"""Offline CPU validation of all metadata/numeric rows and every episode boundary.

Creates a stable dataset symlink and reusable Arrow cache under workspace/.cache.
Does not modify source data or training code. No model/tokenizer is constructed.
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(workspace, root):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    torch.set_num_threads(2)
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import DataConfig
    from openpi.training.data_loader import create_torch_dataset

    print("Loading all episodes through OpenPI create_torch_dataset...", file=sys.stderr, flush=True)
    dataset = create_torch_dataset(
        DataConfig(repo_id="physical-intelligence/libero", prompt_from_task=True),
        action_horizon=10,
        model_config=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    )
    raw = dataset._dataset
    info = raw.meta.info
    require(raw.root.resolve() == root, "Loader did not use the requested local dataset")
    require(len(dataset) == info["total_frames"] == 273465, "Incorrect total frame count")
    require(raw.num_episodes == info["total_episodes"] == 1693, "Incorrect episode count")
    require(len(raw.meta.tasks) == info["total_tasks"] == 40, "Incorrect task count")
    episode_ids = sorted(raw.meta.episodes)
    require(episode_ids == list(range(1693)), "Episode IDs are not contiguous")

    arrow = raw.hf_dataset.data.table
    global_episode_ids = arrow["episode_index"].to_numpy()
    global_frame_ids = arrow["frame_index"].to_numpy()
    timestamps = arrow["timestamp"].to_numpy()
    task_ids = arrow["task_index"].to_numpy()
    require(np.isfinite(timestamps).all(), "Non-finite timestamps")
    require(set(task_ids.tolist()).issubset(raw.meta.tasks), "Unknown task ID")
    numeric_shapes = {}
    for key, dimension in [("state", 8), ("actions", 7)]:
        column = arrow[key]
        require(column.type.value_type == pa.float32(), f"Unexpected {key} dtype")
        require(column.null_count == 0, f"Null {key} rows")
        require(np.all(pc.list_value_length(column).to_numpy() == dimension),
                f"Incorrect {key} dimensions")
        flat = column.combine_chunks().values
        require(flat.null_count == 0 and np.isfinite(flat.to_numpy()).all(),
                f"Non-finite or null {key} values")
        numeric_shapes[key] = [len(dataset), dimension]

    offset = 0
    windows = 0
    for episode_id in episode_ids:
        episode = raw.meta.episodes[episode_id]
        length = episode["length"]
        end = offset + length
        require(int(raw.episode_data_index["from"][episode_id]) == offset
                and int(raw.episode_data_index["to"][episode_id]) == end,
                f"Episode {episode_id}: incorrect global boundaries")
        require(np.all(global_episode_ids[offset:end] == episode_id),
                f"Episode {episode_id}: mixed episode IDs")
        require(np.array_equal(global_frame_ids[offset:end], np.arange(length)),
                f"Episode {episode_id}: incorrect frame order")
        require(np.allclose(timestamps[offset:end], np.arange(length) / info["fps"],
                            atol=1e-4, rtol=0), f"Episode {episode_id}: incorrect timestamps")
        texts = {raw.meta.tasks[int(task)] for task in task_ids[offset:end]}
        require(texts == set(episode["tasks"]), f"Episode {episode_id}: incorrect task text")

        # Reference actions come directly from that episode's original Parquet.
        path = root / raw.meta.get_data_file_path(episode_id)
        actions = np.asarray(pq.ParquetFile(path).read(columns=["actions"])["actions"].to_pylist(),
                             dtype=np.float32)
        for local_index in sorted({0, length - 1}):
            sample = dataset[offset + local_index]
            expected_indices = np.minimum(np.arange(local_index, local_index + 10), length - 1)
            require(tuple(sample["actions"].shape) == (10, 7), "Incorrect action window shape")
            require(np.allclose(sample["actions"].numpy(), actions[expected_indices],
                                atol=1e-7, rtol=0),
                    f"Episode {episode_id}, frame {local_index}: wrong or cross-episode actions")
            expected_pad = np.arange(local_index, local_index + 10) >= length
            require(np.array_equal(sample["actions_is_pad"].numpy(), expected_pad),
                    f"Episode {episode_id}: incorrect action padding mask")
            require(sample["prompt"] in episode["tasks"], "Incorrect injected prompt")
            windows += 1
        offset = end
        if (episode_id + 1) % 100 == 0 or episode_id == episode_ids[-1]:
            print(f"Validated boundaries: {episode_id + 1}/{len(episode_ids)}", file=sys.stderr, flush=True)

    require(offset == len(dataset), "Episode lengths do not cover the full dataset")
    return {
        "status": "PASS", "cpu_only": True, "offline": True,
        "data_root": str(root), "total_episodes": len(episode_ids),
        "total_frames": len(dataset), "total_tasks": len(raw.meta.tasks),
        "numeric_shapes": numeric_shapes, "boundary_windows_checked": windows,
        "action_window_shape": [10, 7],
        "cross_episode_action_leakage": False,
        "cache_dir": str(workspace / ".cache/huggingface/datasets"),
        "cpu_threads": 2,
        "scope": "All numeric rows/indices/tasks and all episode start/end action windows; "
                 "images decoded at boundary samples, not at every frame; "
                 "normalization, tokenizer, full transformed batch, model and GPU not tested",
    }


def main():
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=workspace / "datasets/libero")
    args = parser.parse_args()
    root = args.data_root.resolve()
    require((root / "meta/info.json").is_file(), f"Missing dataset metadata: {root}")
    cache = workspace / ".cache"
    alias = cache / "lerobot/physical-intelligence/libero"
    if alias.exists() or alias.is_symlink():
        require(alias.resolve() == root, f"Existing dataset alias points elsewhere: {alias}")
    else:
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(root, target_is_directory=True)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "huggingface/datasets"),
        "HF_LEROBOT_HOME": str(cache / "lerobot"),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    })
    # Deprecated inherited settings must not override this process's explicit root.
    os.environ.pop("LEROBOT_HOME", None)
    stop = threading.Event()
    def report_memory():
        while not stop.wait(30):
            status = Path("/proc/self/status")
            if status.is_file():
                rss = next((line for line in status.read_text().splitlines()
                            if line.startswith("VmRSS:")), "VmRSS unavailable")
                print(f"Still running; {rss}", file=sys.stderr, flush=True)
    threading.Thread(target=report_memory, daemon=True).start()
    try:
        report = run(workspace, root)
    finally:
        stop.set()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
