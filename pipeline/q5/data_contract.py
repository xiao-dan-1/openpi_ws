"""Read-only train/validation numeric and timing checks; no model or raw-bag access."""
import json
import numpy as np
import pyarrow.parquet as pq

from pipeline.q5.runtime import require
from pipeline.q5.schema import JOINT_NAMES, STATE_UNITS, ACTION_UNITS, ACTION_REPRESENTATION


def check_data_contract(paths, data_config, manifest):
    from openpi import transforms
    all_transforms = [*data_config.data_transforms.inputs, *data_config.data_transforms.outputs,
                      *data_config.model_transforms.inputs, *data_config.model_transforms.outputs]
    require(not any(isinstance(t, (transforms.DeltaActions, transforms.AbsoluteActions)) for t in all_transforms),
            "Q5 absolute targets must not receive an extra delta/absolute conversion")
    normalizer = transforms.Normalize(data_config.norm_stats, use_quantiles=True, strict=True)
    unnormalizer = transforms.Unnormalize(data_config.norm_stats, use_quantiles=True)
    reports = {}
    for role, root in [("train", paths.train), ("validation", paths.validation)]:
        info = json.loads((root / "meta/info.json").read_text())
        require(info["fps"] == 30, "Unexpected Q5 control/data rate")
        for key in ["action", "observation.state"]:
            require(info["features"][key]["names"] == list(JOINT_NAMES), "Joint names/order mismatch")
        counters = {k: 0 for k in ["no_state", "no_action", "stale_state", "stale_action", "disorder", "age_a_future"]}
        max_drift = max_state_age = max_action_age = max_pts_error = 0.0
        intervals, frames, max_error = [], 0, 0.0
        for row in manifest[role]:
            eid, n = row["output_episode_index"], row["length"]
            table = pq.read_table(root / row["data_path"])
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            require(state.shape == actions.shape == (n, 38), "State/action shape mismatch")
            require(np.isfinite(state).all() and np.isfinite(actions).all(), "Non-finite joints")
            require(np.all(actions[:, 14:26] == 0), "Expected zero left-hand commands")
            timestamp = table["timestamp"].to_numpy().astype(float)
            require(np.array_equal(table["frame_index"].to_numpy(), np.arange(n)), "Nonconsecutive frame index")
            max_pts_error = max(max_pts_error, float(np.abs(timestamp - np.arange(n) / 30).max()))
            require(max_pts_error < 1e-4, "Data timestamp/FPS mismatch")
            values = {"state": state, "actions": actions}
            normalized = normalizer(values)
            recovered = unnormalizer(normalized)
            require(all(np.isfinite(v).all() for v in normalized.values()), "Non-finite normalized joint values")
            max_error = max(max_error, *(float(np.abs(recovered[k] - values[k]).max()) for k in values))
            require(max_error < 2e-6, "Normalization changed joint units/order or roundtrip failed")
            timing = json.loads((root / f"meta/timing/episode_{eid:06d}.json").read_text())
            stamps = np.asarray(timing["relative_image_stamps"])
            require(len(stamps) == n and np.isfinite(stamps).all() and np.all(np.diff(stamps) > 0), "Invalid source image timestamps")
            intervals.extend(np.diff(stamps).tolist())
            max_drift = max(max_drift, float(np.abs(stamps - np.arange(n) / 30).max()))
            qc = timing["qc"]
            for key in counters:
                counters[key] += qc.get(key, 0)
            max_state_age = max(max_state_age, qc["age_s_max"] or 0)
            max_action_age = max(max_action_age, qc["age_a_max"] or 0)
            frames += n
        require(all(counters[k] == 0 for k in ["stale_state", "stale_action", "disorder", "age_a_future"]),
                f"Source converter QC flags in {role}: {counters}")
        reports[role] = dict(episodes=len(manifest[role]), frames=frames, normalization_roundtrip_max_error=max_error,
                              nominal_timestamp_max_error_s=max_pts_error, source_image_interval_min_s=min(intervals),
                              source_image_interval_max_s=max(intervals), source_vs_constant_fps_max_drift_s=max_drift,
                              converter_counters=counters, converter_state_age_max_s=max_state_age,
                              converter_action_age_max_s=max_action_age)
    return {"status": "PASS", "joint_order": list(JOINT_NAMES), "expected_state_units": STATE_UNITS,
            "expected_action_units": ACTION_UNITS, "unit_basis": "existing Q5 HybridJointCommand documentation; dataset itself has no unit field",
            "action_representation": ACTION_REPRESENTATION, "delta_transform": False,
            "same_training_and_inference_data_transforms": True, "train_validation": reports,
            "raw_bags_revalidated": False, "live_robot_execution_verified": False,
            "limitations": ["Per-message joint/command stamps are unavailable here; ages come from converter QC sidecars.",
                            "Robot-side joint reorder, radians and absolute-target execution still require live/bag verification.",
                            "Image-vs-state transport/physical response delay cannot be inferred conclusively from converter counters."]}
