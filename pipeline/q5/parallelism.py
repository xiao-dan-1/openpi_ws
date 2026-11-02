"""Validate explicit single/dual GPU selection before importing JAX."""
from pipeline.common.validation import gpu_selectors


def training_layout(gpus, fsdp_devices, batch_size):
    count = len(gpus.split(","))
    fsdp = 1 if fsdp_devices is None and count == 1 else fsdp_devices
    if fsdp != count:
        raise ValueError("Use one GPU with fsdp=1, or two GPUs with explicit --fsdp-devices 2")
    if batch_size < 1 or batch_size % count:
        raise ValueError("Global batch must be positive and divisible by selected GPU count")
    return {"device_count": count, "fsdp_devices": fsdp,
            "global_batch": batch_size, "per_device_batch": batch_size // count,
            "mesh_shape": {"batch": 1, "fsdp": fsdp}}
