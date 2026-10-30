"""Compare sequential and spawned official loaders on real episode-0 data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common.config import WORKSPACE, LOCAL_REPO_ID, build_config, load_official_script, prepare_environment


def main():
    prepare_environment(cpu=True)
    import numpy as np
    import pyarrow as pa
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.shared.normalize import RunningStats
    from openpi.transforms import PromptFromLeRobotTask
    from openpi.training.data_loader import TransformedDataset, TorchDataLoader

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    torch.set_num_threads(2)
    config = build_config(profile="stats")
    data = config.data.create(config.assets_dirs, config.model)
    official = load_official_script("compute_norm_stats")
    raw = LeRobotDataset(LOCAL_REPO_ID, root=WORKSPACE / "datasets/libero", episodes=[0],
                         delta_timestamps={"actions": [i / 10 for i in range(10)]})
    dataset = TransformedDataset(raw, [PromptFromLeRobotTask(raw.meta.tasks),
                                      *data.repack_transforms.inputs,
                                      *data.data_transforms.inputs, official.RemoveStrings()])
    results = []
    for workers in (0, 2):
        loader = TorchDataLoader(dataset, local_batch_size=32, num_workers=workers,
                                 shuffle=False, num_batches=4)
        accumulators = {key: RunningStats() for key in ("state", "actions")}
        count = 0
        for batch in loader:
            for key, accumulator in accumulators.items():
                accumulator.update(np.asarray(batch[key]))
            count += 1
        if count != 4:
            raise ValueError(f"Expected four batches, got {count}")
        results.append({key: accumulator.get_statistics() for key, accumulator in accumulators.items()})
        del loader
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            np.testing.assert_array_equal(getattr(results[0][key], field), getattr(results[1][key], field))
    print(json.dumps({"status": "PASS", "workers_compared": [0, 2], "episode": 0,
                      "frames_per_run": 128, "batches_per_run": 4,
                      "statistics_exactly_equal": True,
                      "scope": "Real episode-0 reader + official transforms + spawned workers; full Dataset and server speed untested"}, indent=2))


if __name__ == "__main__":
    main()
