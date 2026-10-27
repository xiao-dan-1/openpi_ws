"""CPU/offline check of local LeRobot samples and OpenPI LIBERO transforms."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(data_root, episode_ids):
    import numpy as np
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.policies.libero_policy import LiberoInputs

    require(data_root.is_dir(), f"Dataset directory is absent: {data_root}")
    repack = transforms.RepackTransform({
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    })
    adapter = LiberoInputs(model_type=ModelType.PI05)
    resize = transforms.ResizeImages(224, 224)
    pad = transforms.PadStatesAndActions(32)
    results = []
    for episode_id in episode_ids:
        # The pinned reader indexes delta-window bounds by global episode ID,
        # which fails for isolated nonzero-ID subsets. Exercise real windows on
        # episode 0, and genuine per-frame reads on the other isolated episodes.
        windowed = episode_id == 0
        dataset = LeRobotDataset(
            "physical-intelligence/libero", root=data_root,
            episodes=[episode_id], download_videos=False,
            revision="a4336d589d589045d1c56423ffdf3b88a0e19b1f",
            delta_timestamps={"actions": [i / 10 for i in range(10)]} if windowed else None,
        )
        prompt_transform = transforms.PromptFromLeRobotTask(dataset.meta.tasks)
        frames = sorted({0, len(dataset) // 2, len(dataset) - 1})
        last_prompt = None
        for frame in frames:
            item = dataset[frame]
            item = {key: value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor) else value
                    for key, value in item.items()}
            require(item["state"].shape == (8,), "Unexpected raw state shape")
            expected_actions = (10, 7) if windowed else (7,)
            require(item["actions"].shape == expected_actions, "Unexpected raw action shape")
            for key in ["image", "wrist_image"]:
                image = np.asarray(item[key])
                require(image.shape == (3, 256, 256), f"Unexpected {key} tensor shape")
                require(np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1,
                        f"Unexpected {key} tensor values")
            for key in ["state", "actions"]:
                require(np.isfinite(item[key]).all(), f"Non-finite {key}")
            if windowed and frame == len(dataset) - 1:
                require(np.allclose(item["actions"], item["actions"][0]),
                        "End-of-episode action padding is unexpected")

            adapted = pad(resize(adapter(repack(prompt_transform(item)))))
            require(adapted["state"].shape == (32,), "Incorrect padded state shape")
            expected_padded_actions = (10, 32) if windowed else (32,)
            require(adapted["actions"].shape == expected_padded_actions, "Incorrect padded action shape")
            require(np.all(adapted["state"][8:] == 0), "State padding is not zero")
            require(np.all(adapted["actions"][..., 7:] == 0), "Action padding is not zero")
            for key, image in adapted["image"].items():
                require(image.shape == (224, 224, 3) and image.dtype == np.uint8,
                        f"Incorrect adapted {key} image format")
            require(bool(adapted["image_mask"]["base_0_rgb"]), "Scene camera is masked")
            require(bool(adapted["image_mask"]["left_wrist_0_rgb"]), "Wrist camera is masked")
            require(not bool(adapted["image_mask"]["right_wrist_0_rgb"]),
                    "Missing camera should be masked for pi05")
            require(np.all(adapted["image"]["right_wrist_0_rgb"] == 0),
                    "Missing camera should contain zeros")
            last_prompt = adapted["prompt"]
            require(last_prompt in dataset.meta.episodes[episode_id]["tasks"],
                    "Incorrect task prompt")
        results.append({"episode_index": episode_id, "frames": len(dataset),
                        "checked_frames": frames, "prompt": last_prompt,
                        "action_window_checked": windowed,
                        "adapted_actions_shape": list(expected_padded_actions)})

    return {
        "status": "PASS", "data_root": str(data_root), "cpu_only": True,
        "offline": True, "episodes": results,
        "raw_state_shape": [8], "raw_action_chunk_shape": [10, 7],
        "adapted_state_shape": [32], "adapted_action_chunk_shape": [10, 32],
        "adapted_camera_shape": [224, 224, 3],
        "subset_limitation": "Pinned LeRobot delta-window indexing fails on isolated nonzero "
                             "episode IDs; windows checked on episode 0, other episodes read per frame",
        "scope": "LeRobot reader + explicit LIBERO repack/adapter/resize/pad; "
                 "normalization, tokenizer, configured full loader, model and GPU not tested",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets/libero")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with tempfile.TemporaryDirectory(prefix="openpi-loader-check-") as cache:
        os.environ["HF_HOME"] = str(Path(cache) / "hf")
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache) / "datasets")
        report = run(args.data_root.resolve(), list(dict.fromkeys(args.episodes)))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
