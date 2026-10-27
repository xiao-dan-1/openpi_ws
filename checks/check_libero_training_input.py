"""Offline CPU check of tokenizer and official pi05_libero input transforms.

Uses candidate normalization artifacts from check_libero_normalization.py.
No model weights, optimizer, gradients or training configuration edits.
Asset resolution is restricted to already-present, verified local files.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

sys.dont_write_bytecode = True
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=workspace / "datasets/libero")
    parser.add_argument("--cache-dir", type=Path, default=workspace / ".cache/huggingface/datasets")
    args = parser.parse_args()
    root = args.data_root.resolve()
    stats_dir = workspace / "checks/output/pi05_libero_h10"
    tokenizer_file = workspace / "openpi_cache/big_vision/paligemma_tokenizer.model"
    require(tokenizer_file.is_file(), f"Missing tokenizer asset: {tokenizer_file}")
    require(hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() == TOKENIZER_SHA256,
            "Tokenizer SHA256 does not match the verified official asset")
    require((stats_dir / "validation_manifest.json").is_file(), "Run the normalization check first")
    manifest = json.loads((stats_dir / "validation_manifest.json").read_text())
    require(manifest["status"] == "PASS"
            and manifest["validation_profile"] == "pi05_libero, h10, extra_delta_transform=False",
            "Wrong candidate normalization profile")
    require(hashlib.sha256((stats_dir / "norm_stats.json").read_bytes()).hexdigest()
            == manifest["norm_stats_sha256"], "Candidate stats changed since validation")
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_CACHE": str(args.cache_dir.resolve()),
        "OPENPI_DATA_HOME": str(workspace / "openpi_cache"),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    })
    os.environ.pop("LEROBOT_HOME", None)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pyarrow as pa
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.models.model import Observation
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import download
    from openpi.training import config, data_loader
    from openpi.transforms import PromptFromLeRobotTask

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    torch.set_num_threads(2)
    require(jax.default_backend() == "cpu", "Unexpected JAX backend")

    # Use the real transforms and SentencePiece implementation, but never fetch
    # missing assets. This override affects this check process only.
    def local_asset(url, **kwargs):
        if str(url) == "gs://big_vision/paligemma_tokenizer.model":
            return tokenizer_file
        require(not urlparse(str(url)).scheme, f"Network asset resolution forbidden: {url}")
        path = Path(url).resolve()
        require(path.exists(), f"Local asset is absent: {path}")
        return path

    baseline = config.get_config("pi05_libero")
    require(baseline.model.pi05 and baseline.model.action_horizon == 10
            and baseline.model.action_dim == 32 and not baseline.model.discrete_state_input
            and not baseline.data.extra_delta_transform, "Baseline profile changed")
    max_tokens = baseline.model.max_token_len
    with mock.patch.object(download, "maybe_download", side_effect=local_asset):
        tokenizer = PaligemmaTokenizer(max_tokens)
        factory = dataclasses.replace(
            baseline.data,
            assets=config.AssetsConfig(assets_dir=str(stats_dir.parent), asset_id=stats_dir.name),
        )
        data_config = factory.create(baseline.assets_dirs, baseline.model)
    require(data_config.norm_stats is not None and data_config.use_quantile_norm,
            "Candidate quantile normalization was not loaded")
    with (root / "meta/tasks.jsonl").open() as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    token_lengths = []
    for task in tasks:
        text = task["task"].strip().replace("_", " ").replace("\n", " ")
        expected = tokenizer._tokenizer.encode(text, add_bos=True) + tokenizer._tokenizer.encode("\n")
        require(0 < len(expected) <= max_tokens, "A task prompt would be truncated")
        tokens, mask = tokenizer.tokenize(task["task"])
        require(tokens.shape == mask.shape == (max_tokens,) and mask.dtype == np.bool_,
                "Wrong tokenizer output shape or mask")
        require(int(mask.sum()) == len(expected)
                and np.array_equal(tokens[mask], expected), "Incorrect task tokens or mask")
        token_lengths.append(len(expected))
    require(len(tasks) == 40, "Incorrect task count")

    print("Constructing full cached Dataset and applying configured input transforms...",
          file=sys.stderr, flush=True)
    raw = LeRobotDataset("physical-intelligence/libero", root=root, episodes=None,
                         revision="a4336d589d589045d1c56423ffdf3b88a0e19b1f",
                         download_videos=False,
                         delta_timestamps={"actions": [i / 10 for i in range(10)]})
    require(len(raw) == 273465 and raw.num_episodes == 1693, "Incomplete source Dataset")
    prompted = data_loader.TransformedDataset(raw, [PromptFromLeRobotTask(raw.meta.tasks)])
    transformed = data_loader.transform_dataset(prompted, data_config)
    examples = []
    for episode_id in [0, 846, 1692]:
        count = raw.meta.episodes[episode_id]["length"]
        start = int(raw.episode_data_index["from"][episode_id])
        for frame in sorted({0, count // 2, count - 1}):
            examples.append(transformed[start + frame])
    batch = jax.tree.map(lambda *values: np.stack(values), *examples)
    batch = jax.tree.map(jnp.asarray, batch)
    observation = Observation.from_dict(batch)
    batch_size = len(examples)
    require(observation.state.shape == (batch_size, 32), "Wrong batched state shape")
    require(batch["actions"].shape == (batch_size, 10, 32), "Wrong batched action shape")
    for values in [observation.state, batch["actions"]]:
        require(np.isfinite(np.asarray(values)).all() and values.dtype == jnp.float32,
                "Invalid normalized numeric batch")
    require(np.all(np.asarray(observation.state)[..., 8:] == 0)
            and np.all(np.asarray(batch["actions"])[..., 7:] == 0), "Wrong zero padding")
    require(observation.tokenized_prompt.shape == observation.tokenized_prompt_mask.shape
            == (batch_size, max_tokens), "Wrong batched token shape")
    for key, image in observation.images.items():
        values = np.asarray(image)
        require(values.shape == (batch_size, 224, 224, 3) and values.dtype == np.float32
                and np.isfinite(values).all() and values.min() >= -1 and values.max() <= 1,
                f"Invalid model observation image: {key}")
    require(np.asarray(observation.image_masks["base_0_rgb"]).all()
            and np.asarray(observation.image_masks["left_wrist_0_rgb"]).all()
            and not np.asarray(observation.image_masks["right_wrist_0_rgb"]).any(), "Wrong camera masks")
    print(json.dumps({
        "status": "PASS", "backend": jax.default_backend(), "offline": True,
        "tokenizer_sha256": TOKENIZER_SHA256, "task_prompts_checked": len(tasks),
        "max_prompt_tokens": max(token_lengths), "max_token_len": max_tokens,
        "batch_size_checked": batch_size, "state_shape": list(observation.state.shape),
        "actions_shape": list(batch["actions"].shape),
        "token_shape": list(observation.tokenized_prompt.shape),
        "camera_shape": [batch_size, 224, 224, 3],
        "norm_stats_sha256": manifest["norm_stats_sha256"],
        "scope": "Official pi05_libero input transforms with candidate stats and local assets; "
                 "full final-config DataLoader iteration, weights, GPU, gradients and training not tested",
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
