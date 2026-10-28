"""Validate explicit single/dual GPU selection before importing JAX."""
import argparse


def gpu_selectors(value):
    parts = value.split(",")
    if len(parts) not in (1, 2) or len(parts) != len(set(parts)) or not all(
        p.isdigit() or (p.startswith("GPU-") and len(p) > 4 and p.strip() == p) for p in parts
    ):
        raise argparse.ArgumentTypeError("Use one GPU or two distinct GPUs, e.g. 2 or 2,3 (UUIDs also accepted)")
    # Reject index aliases such as 2,02, which refer to the same device.
    canonical = [str(int(p)) if p.isdigit() else p for p in parts]
    if len(set(canonical)) != len(parts):
        raise argparse.ArgumentTypeError("GPU selectors must refer to distinct devices")
    return ",".join(canonical)


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
