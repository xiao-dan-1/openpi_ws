"""Explicit single/dual-GPU Q5 capacity/throughput probe; no saved checkpoints."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import time

from pipeline.q5.runtime import Paths, prepare_environment, require, require_stats, official_train_module
from pipeline.q5.memory import add_memory_argument, memory_settings


def gpu_selectors(value):
    parts = value.split(",")
    if len(parts) not in (1, 2) or len(parts) != len(set(parts)) or not all(p.isdigit() for p in parts):
        raise argparse.ArgumentTypeError("Use one index or two distinct indices, e.g. 2 or 2,3")
    return value


def sharded_state_storage(params):
    """Sum logical tensor bytes and actual addressable shard bytes by local device."""
    import jax
    logical = 0
    per_device = {}
    leaves_sharded = leaves_replicated = 0
    for value in jax.tree.leaves(params):
        if not hasattr(value, "addressable_shards"):
            continue
        logical += value.size * value.dtype.itemsize
        leaves_replicated += int(value.is_fully_replicated)
        leaves_sharded += int(not value.is_fully_replicated)
        for shard in value.addressable_shards:
            name = str(shard.device)
            per_device[name] = per_device.get(name, 0) + shard.data.size * shard.data.dtype.itemsize
    return {"logical_bytes": logical, "per_device_bytes": per_device,
            "sharded_array_leaves": leaves_sharded, "replicated_array_leaves": leaves_replicated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--gpus", type=gpu_selectors, required=True)
    parser.add_argument("--fsdp-devices", type=int, choices=[1, 2], required=True)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch across the selected GPUs")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Includes the first compile/update call")
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Optional JSON report; default prints only")
    add_memory_argument(parser)
    args = parser.parse_args()
    count = len(args.gpus.split(","))
    if args.fsdp_devices != count:
        parser.error("For this probe use one GPU/fsdp=1 or two GPUs/fsdp=2; ordinary replicated data parallelism is not tested")
    if args.batch_size < 1 or args.batch_size % count or args.warmup_steps < 3 or args.measure_steps < 2:
        parser.error("Global batch must divide evenly by GPU count; require warmup>=3 and measurement>=2")
    if args.output is not None and args.output.exists():
        parser.error("Report already exists; choose another --output")
    paths = Paths.create(args.workspace)
    prepare_environment(paths, cpu=False, gpu=args.gpus, allow_multiple_gpus=True, memory_fraction=args.memory_fraction)
    stats = require_stats(paths)
    from pipeline.q5.config import build_config, require_base_assets
    require_base_assets(paths)
    import jax
    import numpy as np
    import torch
    from openpi.training import data_loader, sharding
    from pipeline.q5.check_model import compile_training_step
    from pipeline.common.reporting import emit_report
    from checks.check_gpu_communication import snapshot
    torch.set_num_threads(2)
    require(jax.default_backend() == "gpu" and jax.device_count() == count, "Unexpected GPU backend/count")
    before = snapshot(args.gpus)
    initial_other = [p for p in before["compute_processes"] if int(p["pid"]) != os.getpid()]
    print(json.dumps({"stage": "starting", "gpus": args.gpus, "global_batch": args.batch_size,
                      "per_device_batch": args.batch_size // count, "fsdp_devices": args.fsdp_devices,
                      "memory_fraction": args.memory_fraction, "allocator_settings": memory_settings(), "other_processes": initial_other,
                      "checkpoints_saved": False}, indent=2), flush=True)
    config = dataclasses.replace(build_config(paths, profile="smoke", batch_size=args.batch_size, num_workers=0),
                                 fsdp_devices=args.fsdp_devices)
    official = official_train_module(paths)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True,
                num_batches=args.warmup_steps + args.measure_steps)
    iterator = iter(loader)
    batch = next(iterator)
    jax.block_until_ready(batch)
    require(tuple(batch[0].state.shape) == (args.batch_size, 38) and
            tuple(batch[1].shape) == (args.batch_size, 10, 38), "Wrong global batch shape")
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    print("stage=model_initialization", flush=True)
    state, state_sharding = official.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    storage = sharded_state_storage((state.params, state.opt_state))
    require(count == 1 or storage["sharded_array_leaves"] > 0, "Expected FSDP-sharded weights/optimizer arrays")
    print(json.dumps({"state_storage_before_updates": storage}, indent=2), flush=True)
    step = compile_training_step(official, config, mesh, state_sharding)
    devices = jax.local_devices()

    def memory():
        result = {}
        for device in devices:
            values = device.memory_stats()
            result[str(device)] = ({k: int(v) for k, v in values.items() if isinstance(v, (int, float))}
                                   if values else None)
        return result

    def update(current, current_batch):
        with sharding.set_mesh(mesh):
            current, metrics = step(train_rng, current, current_batch)
        jax.block_until_ready((current, metrics))
        metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        require(all(np.isfinite(v) for v in metrics.values()), "Non-finite training metric")
        return current, metrics

    after_init = memory()
    started = time.perf_counter()
    state, metrics = update(state, batch)
    first_update_seconds = time.perf_counter() - started
    print(f"compile_and_first_update_seconds={first_update_seconds:.3f}", flush=True)
    for _ in range(1, args.warmup_steps):
        batch = next(iterator)
        jax.block_until_ready(batch)
        state, metrics = update(state, batch)
    samples = []
    for i in range(args.measure_steps):
        begin = time.perf_counter()
        batch = next(iterator)
        jax.block_until_ready(batch)
        loaded = time.perf_counter()
        state, metrics = update(state, batch)
        finish = time.perf_counter()
        row = {"index": i, "data_seconds": loaded - begin, "compute_seconds": finish - loaded,
               "total_seconds": finish - begin, **metrics}
        samples.append(row)
        print(f"measured={i+1}/{args.measure_steps} compute={row['compute_seconds']:.3f}s "
              f"total={row['total_seconds']:.3f}s loss={metrics['loss']:.6f} grad={metrics['grad_norm']:.6f}", flush=True)
    require(int(jax.device_get(state.step)) == args.warmup_steps + args.measure_steps, "Wrong update count")
    require(any(row["grad_norm"] > 0 for row in samples), "No nonzero measured gradient")
    after = snapshot(args.gpus)
    other_end = [p for p in after["compute_processes"] if int(p["pid"]) != os.getpid()]
    report = {"status": "PASS", "gpus": args.gpus, "fsdp_devices": count, "global_batch": args.batch_size,
              "per_device_batch": args.batch_size // count, "warmup_steps": args.warmup_steps,
              "measured_steps": args.measure_steps, "first_update_compile_included_seconds": first_update_seconds,
              "median_compute_seconds": statistics.median(s["compute_seconds"] for s in samples),
              "median_total_seconds": statistics.median(s["total_seconds"] for s in samples),
              "compute_samples_per_second": args.batch_size * len(samples) / sum(s["compute_seconds"] for s in samples),
              "total_samples_per_second": args.batch_size * len(samples) / sum(s["total_seconds"] for s in samples),
              "state_storage": storage, "memory_after_init": after_init, "memory_after_measurement": memory(),
              "memory_fraction": args.memory_fraction, "allocator_settings": memory_settings(), "samples": samples, "other_processes_start": initial_other,
              "other_processes_end": other_end,
              "shared_load_observed": bool(initial_other or other_end),
              "norm_stats_sha256": stats["sha256"], "checkpoint_save_restore_tested": False,
              "scope": "real Q5 weights/data/updates with official FSDP; no checkpoints or task-quality evaluation",
              "note": "Memory allocator statistics include retained pool buffers; compare bytes_in_use/peak when available, not only nvidia-smi."}
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
