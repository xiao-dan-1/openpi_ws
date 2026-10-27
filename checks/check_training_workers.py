"""Real episode-0 training transforms with spawned workers, on CPU only."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common.config import WORKSPACE, LOCAL_REPO_ID, build_config, prepare_environment


def main():
    prepare_environment(cpu=True, cache_dir=str(Path.home() / ".cache/huggingface/datasets"))
    import jax
    import numpy as np
    import pyarrow as pa
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.training import data_loader
    from openpi.transforms import PromptFromLeRobotTask

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    torch.set_num_threads(2)
    config = build_config(profile="finetune", batch_size=16,
                          assets_base_dir=WORKSPACE / "checks/output/official_norm_smoke/assets")
    data = config.data.create(config.assets_dirs, config.model)
    raw = LeRobotDataset(LOCAL_REPO_ID, root=WORKSPACE / "datasets/libero", episodes=[0],
                         delta_timestamps={"actions": [i / 10 for i in range(10)]})
    prompted = data_loader.TransformedDataset(raw, [PromptFromLeRobotTask(raw.meta.tasks)])
    dataset = data_loader.transform_dataset(prompted, data)
    runs = []
    for workers in (0, 2):
        loader = data_loader.TorchDataLoader(dataset, local_batch_size=16, num_workers=workers,
                                             shuffle=False, num_batches=2)
        batches = []
        for batch in loader:
            if batch["state"].shape != (16, 32) or batch["actions"].shape != (16, 10, 32):
                raise ValueError("Unexpected state/action shapes")
            if batch["tokenized_prompt"].shape != (16, 200):
                raise ValueError("Unexpected token shape")
            host = jax.tree.map(np.asarray, batch)
            for leaf in jax.tree.leaves(host):
                if not np.isfinite(leaf).all():
                    raise ValueError("Non-finite transformed training input")
            batches.append(host)
        if len(batches) != 2:
            raise ValueError("Expected two batches")
        runs.append(batches)
        del loader
    for first, second in zip(jax.tree.leaves(runs[0]), jax.tree.leaves(runs[1])):
        np.testing.assert_array_equal(first, second)
    print(json.dumps({"status": "PASS", "workers_compared": [0, 2], "episode": 0,
                      "batch_size": 16, "batches_per_run": 2, "inputs_exactly_equal": True,
                      "scope": "Real episode-0 input transforms including normalization/tokenizer/images, CPU only with probe stats; full dataset/GPU/RAM impact untested"}, indent=2))


if __name__ == "__main__":
    main()
