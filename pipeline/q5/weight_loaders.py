"""Migrate the pretrained trunk; reinitialize exactly two complete action projections."""
import dataclasses
import json
import re
from pathlib import Path

import flax.traverse_util
import numpy as np

PROJECTION_KEYS = frozenset(f"action_{direction}_proj/{leaf}" for direction in ("in", "out") for leaf in ("kernel", "bias"))


def merge_trunk(loaded, reference):
    source = flax.traverse_util.flatten_dict(loaded, sep="/")
    target = flax.traverse_util.flatten_dict(reference, sep="/")
    unknown = set(source) - set(target)
    if unknown:
        raise ValueError(f"Unexpected checkpoint keys: {sorted(unknown)}")
    hidden = target["action_in_proj/kernel"].shape[1]
    shapes = {"action_in_proj/kernel": ((32, hidden), (38, hidden)),
              "action_in_proj/bias": ((hidden,), (hidden,)),
              "action_out_proj/kernel": ((hidden, 32), (hidden, 38)),
              "action_out_proj/bias": ((32,), (38,))}
    for key, (old, new) in shapes.items():
        if key not in source or source[key].shape != old or target[key].shape != new:
            raise ValueError(f"Unexpected migration shape: {key}, expected {old} -> {new}")
    result = {}
    for key, ref in target.items():
        if key in PROJECTION_KEYS:
            # During official initialization these are ShapeDtypeStruct leaves.
            # The official trainer removes placeholders and retains model RNG initialization.
            result[key] = ref
        elif key not in source:
            if not re.fullmatch(r".*lora.*", key):
                raise ValueError(f"Missing pretrained trunk weight: {key}")
            result[key] = ref
        else:
            value = source[key]
            if value.shape != ref.shape:
                raise ValueError(f"Pretrained trunk shape mismatch: {key}: {value.shape} != {ref.shape}")
            result[key] = value.astype(ref.dtype) if value.dtype != ref.dtype else value
    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class Q5WeightLoader:
    params_path: str
    audit_path: str | None = None
    trainable_filter: object = None

    def load(self, params):
        from openpi.models import model
        loaded = model.restore_params(self.params_path, restore_type=np.ndarray)
        merged = merge_trunk(loaded, params)
        if self.audit_path:
            from pipeline.q5.parameter_audit import audit_parameters
            report = audit_parameters(params, self.trainable_filter)
            source = flax.traverse_util.flatten_dict(loaded, sep="/")
            rows = report["parameters"]
            for row in rows:
                key = row["path"]
                row["initialization"] = ("model_rng_action_projection" if key in PROJECTION_KEYS else
                                         "model_rng_new_lora" if key not in source else "pretrained_checkpoint")
                row["source_shape"] = list(source[key].shape) if key in source else None
            report.update(full_checkpoint_arrays_read=True, migration_shapes_checked=True,
                          source_checkpoint=self.params_path,
                          reinitialized_projection_paths=sorted(PROJECTION_KEYS),
                          note="Checkpoint arrays read and merged; model initialization/forward/update are validated separately.")
            path = Path(self.audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        return merged
