"""Read committed checkpoint counters without loading model or optimizer tensors."""
import re
from pathlib import Path

from pipeline.q5.runtime import ASSET_ID, require, sha256


def checkpoint_progress(checkpoint_dir, norm_stats_sha256):
    import numpy as np
    import orbax.checkpoint as ocp

    checkpoint_dir = Path(checkpoint_dir)
    require(checkpoint_dir.is_dir(), "No checkpoint directory to resume/check")
    # Orbax excludes temporary/incomplete saves. Read-only also prevents retention/deletion.
    with ocp.CheckpointManager(checkpoint_dir, options=ocp.CheckpointManagerOptions(
        read_only=True, create=False
    )) as manager:
        label = manager.latest_step()
    require(label is not None and label > 0, "No completed nonzero checkpoint; refusing to restart as --resume")
    directory = checkpoint_dir / str(label)
    require(sha256(directory / "assets" / ASSET_ID / "norm_stats.json") == norm_stats_sha256,
            "Saved checkpoint statistics mismatch")
    require((directory / "params").is_dir(), "Missing saved model parameters")
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = ocp.tree.to_flat_dict(checkpointer.metadata(directory / "train_state"), sep="/")
        counts = sorted(key for key in metadata if key.startswith("opt_state/") and key.endswith("/count"))
        require("step" in metadata and counts, "Missing training/optimizer counters")
        sources = {"step": "step", **{f"count_{i}": key for i, key in enumerate(counts)}}
        restored = checkpointer.restore(directory / "train_state", args=ocp.args.PyTreeRestore(
            item={key: np.zeros((), np.int32) for key in sources},
            restore_args={key: ocp.RestoreArgs(restore_type=np.ndarray) for key in sources},
            transforms={key: ocp.Transform(original_key=re.escape(source)) for key, source in sources.items()},
            transforms_default_to_original=False,
        ))
    step = int(restored["step"])
    optimizer_counts = {source: int(restored[key]) for key, source in sources.items() if key != "step"}
    require(step == label + 1, "Checkpoint label/state.step mismatch")
    require(all(value == step for value in optimizer_counts.values()), "Optimizer/schedule counters differ from state.step")
    return {"checkpoint_label": int(label), "state_step": step, "optimizer_counts": optimizer_counts,
            "norm_stats_sha256": norm_stats_sha256,
            "scope": "Committed checkpoint, saved statistics and scalar counters; tensor restoration is performed by the official trainer"}
