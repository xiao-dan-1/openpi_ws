"""Whole-tree parameter inventory for server initialization; metadata-only helpers."""
import math
import re
import ast
import json

import flax.nnx as nnx
import flax.traverse_util
import numpy as np


def component(key):
    if key.startswith("PaliGemma/img/"):
        return "vision"
    if key.startswith("PaliGemma/llm/"):
        return "action_expert" if any(re.search(r"_1$", part) for part in key.split("/")) else "language"
    if key.startswith(("action_in_proj/", "action_out_proj/")):
        return "action_projection"
    if key.startswith(("time_mlp_in/", "time_mlp_out/")):
        return "time_mlp"
    return "other"


def audit_parameters(reference, trainable_filter):
    predicate = nnx.filterlib.to_predicate(trainable_filter)
    rows, groups = [], {}
    for key, value in flax.traverse_util.flatten_dict(reference, sep="/").items():
        # The actual complete tree supplies names, shapes and dtypes. Predicate
        # values only need to be nnx.Param for the standard LIBERO filter.
        trainable = bool(predicate(tuple(key.split("/")), nnx.Param(np.zeros(1, np.float32))))
        group = component(key)
        lora = "lora" in key
        expected = lora if key.startswith("PaliGemma/llm/") else True
        if trainable != expected:
            raise ValueError(f"Unexpected parameter freeze state: {key}")
        count = math.prod(value.shape)
        rows.append({"path": key, "shape": list(value.shape), "dtype": str(value.dtype),
                     "component": group, "lora": lora, "trainable": trainable, "elements": count})
        totals = groups.setdefault(group, {"trainable_elements": 0, "frozen_elements": 0, "leaves": 0})
        totals["trainable_elements" if trainable else "frozen_elements"] += count
        totals["leaves"] += 1
    action_non_lora = [r for r in rows if r["component"] == "action_expert" and not r["lora"]]
    if not action_non_lora or any(r["trainable"] for r in action_non_lora):
        raise ValueError("No frozen action-expert non-LoRA parameters were verified")
    return {"status": "PASS", "scope": "complete parameter tree, not representative dummy paths",
            "action_expert_non_lora_leaves_checked": len(action_non_lora), "groups": groups, "parameters": rows}


def audit_checkpoint_metadata(metadata_path, trainable_filter):
    """Inspect every real base-checkpoint path without reading tensor payloads."""
    import jax
    metadata = json.loads(metadata_path.read_text())["tree_metadata"]
    flat = {}
    for encoded, entry in metadata.items():
        parts = list(ast.literal_eval(encoded))
        if parts[0] == "params":
            parts = parts[1:]
        if parts[-1] == "value":
            parts = parts[:-1]
        shape = entry["value_metadata"].get("write_shape")
        if shape is not None:
            flat["/".join(parts)] = jax.ShapeDtypeStruct(tuple(shape), np.float32)
    report = audit_parameters(flax.traverse_util.unflatten_dict(flat, sep="/"), trainable_filter)
    report.update(scope="all real base-checkpoint metadata paths; new target-model LoRA leaves not represented",
                  tensor_payloads_read=False, dtype_note="float32 placeholders for freeze predicate; stored dtypes not checked here")
    return report
