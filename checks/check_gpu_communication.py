"""Synthetic two-GPU JAX collectives; no models, training, or persistent reports.

Each invocation isolates one GPU pair in a subprocess. The measured intervals
include synchronized host dispatch, exclude compilation, warmup and host copies.
"""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter")


def parse_worker_output(stdout):
    """Native NCCL logs may be flushed before or after the JSON result."""
    prefix, marker, result = stdout.rpartition("GPU_COMM_RESULT_JSON=")
    if not marker:
        raise ValueError("Worker returned no result marker")
    result = result.lstrip()
    report, end = json.JSONDecoder().raw_decode(result)
    if not isinstance(report, dict) or report.get("status") != "PASS":
        raise ValueError("Worker returned an invalid result")
    diagnostics = "\n".join(part.strip() for part in [prefix, result[end:]] if part.strip())
    return report, diagnostics


def gpu_pair(value):
    parts = value.split(",")
    if len(parts) != 2 or not all(p.isdigit() for p in parts) or len({int(p) for p in parts}) != 2:
        raise argparse.ArgumentTypeError("Specify two distinct physical GPU indices, e.g. 0,1")
    return ",".join(str(int(p)) for p in parts)


def positive_size(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Size must be a finite number in (0, 128] MiB") from exc
    if not math.isfinite(number) or not 0 < number <= 128:
        raise argparse.ArgumentTypeError("Size must be a finite number in (0, 128] MiB")
    return number


def corrected_bus_bytes(operation, input_bytes, ranks=2):
    """NCCL-style normalization; not a measurement of physical PCIe traffic.

    AllGather size is the full gathered output; ReduceScatter size is full input.
    AllReduce size is each rank's full input.
    """
    if operation == "all_reduce":
        return input_bytes * 2 * (ranks - 1) / ranks
    if operation == "all_gather":
        return input_bytes * ranks * (ranks - 1) / ranks
    if operation == "reduce_scatter":
        return input_bytes * (ranks - 1) / ranks
    raise ValueError(operation)


def query_smi(arguments):
    result = subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return list(csv.reader(io.StringIO(result.stdout)))


def snapshot(pair):
    gpu_rows = query_smi(["--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    selected = set(pair.split(","))
    devices = [{"index": r[0].strip(), "uuid": r[1].strip(), "name": r[2].strip(),
                "total_mib": r[3].strip(), "used_mib": r[4].strip(), "utilization_percent": r[5].strip()}
               for r in gpu_rows if r and r[0].strip() in selected]
    if len(devices) != len(selected):
        raise ValueError(f"Could not resolve GPU pair {pair}")
    uuids = {d["uuid"] for d in devices}
    processes = query_smi(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    active = [{"gpu_uuid": r[0].strip(), "pid": r[1].strip(), "name": r[2].strip(), "used_mib": r[3].strip()}
              for r in processes if len(r) >= 4 and r[0].strip() in uuids]
    return {"devices": devices, "compute_processes": active}


def gpu_uuids(pair, devices):
    """Keep requested nvidia-smi order; UUIDs avoid CUDA ordinal differences."""
    by_index = {device["index"]: device["uuid"] for device in devices}
    uuids = [by_index[index] for index in pair.split(",")]
    if len(set(uuids)) != 2 or not all(uuid.startswith("GPU-") and len(uuid) > 4 for uuid in uuids):
        raise ValueError("Cannot resolve two distinct GPU UUIDs; no communication test started")
    return ",".join(uuids)


def worker(args):
    # This standalone benchmark never calls the Q5 runtime or changes its settings.
    # Small synthetic buffers use their own modest, on-demand JAX memory pool.
    os.environ["CUDA_VISIBLE_DEVICES"] = ("" if args.cpu_fixture else
                                         gpu_uuids(args.gpus, snapshot(args.gpus)["devices"]))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
    os.environ.pop("XLA_CLIENT_MEM_FRACTION", None)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    if args.cpu_fixture:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    else:
        os.environ["JAX_PLATFORMS"] = "cuda"
    if args.debug_transport:
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,GRAPH,P2P,SHM,NET"
    import jax
    import numpy as np
    import jaxlib
    devices = jax.local_devices()
    if len(devices) != 2 or (not args.cpu_fixture and any(d.platform != "gpu" for d in devices)):
        raise RuntimeError(f"Expected exactly two {'CPU fixture' if args.cpu_fixture else 'GPU'} devices, got {devices}")
    rows = []
    for size in args.sizes_mib:
        # float32, even vector length for two-rank ReduceScatter.
        count = max(2, int(size * 2**20 / 4) // 2 * 2)
        input_bytes = count * 4
        x = jax.device_put_sharded([np.full(count, rank + 1, np.float32) for rank in range(2)], devices)
        jax.block_until_ready(x)
        for operation in OPERATIONS:
            def collective(value):
                if operation == "all_reduce":
                    return jax.lax.psum(value, "ranks")
                if operation == "all_gather":
                    return jax.lax.all_gather(value, "ranks", axis=0, tiled=True)
                return jax.lax.psum_scatter(value, "ranks", scatter_dimension=0, tiled=True)
            mapped = jax.pmap(collective, axis_name="ranks", devices=devices)
            started = time.perf_counter()
            lowered = mapped.lower(x)
            hlo = lowered.as_text().lower().replace("_", "-")
            hlo_name = {"all_reduce": "all-reduce", "all_gather": "all-gather", "reduce_scatter": "reduce-scatter"}[operation]
            if hlo_name not in hlo:
                raise RuntimeError(f"Expected collective missing from lowered graph: {operation}")
            compiled = lowered.compile()
            compile_seconds = time.perf_counter() - started
            for _ in range(args.warmup):
                y = compiled(x)
                jax.block_until_ready(y)
            elapsed = []
            for _ in range(args.iterations):
                started = time.perf_counter_ns()
                y = compiled(x)
                jax.block_until_ready(y)
                elapsed.append((time.perf_counter_ns() - started) / 1e9)
            # Check rank-dependent contents outside the timing region.
            expected_length = count * 2 if operation == "all_gather" else count // 2 if operation == "reduce_scatter" else count
            if y.shape != (2, expected_length):
                raise ValueError(f"Collective shape mismatch: {operation}: {y.shape}")
            for shard in y.addressable_shards:
                local = shard.data.reshape(-1)
                indices = np.unique([0, expected_length // 2, expected_length - 1])
                values = np.asarray(jax.device_get(local[indices]))
                expected = np.where(indices < count, 1.0, 2.0) if operation == "all_gather" else np.full(len(indices), 3.0)
                np.testing.assert_array_equal(values, expected)
            median = statistics.median(elapsed)
            normalized_bytes = corrected_bus_bytes(operation, input_bytes)
            row = {"operation": operation, "input_mib_per_gpu": input_bytes / 2**20,
                   "output_mib_per_gpu": expected_length * 4 / 2**20,
                   "iterations": args.iterations, "median_ms": median * 1000,
                   "p90_ms": float(np.percentile(elapsed, 90) * 1000),
                   "minimum_ms": min(elapsed) * 1000,
                   "normalized_bus_GB_per_s": normalized_bytes / median / 1e9,
                   "compile_seconds_excluded": compile_seconds, "correctness": "PASS"}
            rows.append(row)
            print(f"{operation:14} input={row['input_mib_per_gpu']:8.4f} MiB/GPU "
                  f"median={row['median_ms']:9.3f} ms p90={row['p90_ms']:9.3f} ms "
                  f"normalized={row['normalized_bus_GB_per_s']:7.3f} GB/s", file=sys.stderr, flush=True)
            del y, compiled, mapped, lowered
            jax.clear_caches()
        del x
    return {"status": "PASS", "device_pair": args.gpus, "backend": jax.default_backend(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "jax_version": jax.__version__, "jaxlib_version": jaxlib.__version__,
            "devices": [str(d) for d in devices], "results": rows,
            "scope": "synchronized host-dispatched synthetic collectives; compilation/warmup/host transfers excluded",
            "bandwidth_definition": "NCCL-style bus normalization; decimal GB/s, not physical link throughput",
            "transport": "not inferred from timing; use --debug-transport to inspect NCCL diagnostics",
            "training_configuration_changed": False, "model_loaded": False, "gpu_training_tested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, type=gpu_pair, help="One pair per run, e.g. 0,1 or 2,3 or 0,2")
    parser.add_argument("--sizes-mib", nargs="+", type=positive_size, default=[1 / 256, 1, 16, 64])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=300, help="Timeout seconds for this pair, including JAX startup/compilation")
    parser.add_argument("--debug-transport", action="store_true", help="Print NCCL routing diagnostics; use for diagnosis, omit for timing comparisons")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cpu-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 2 or args.timeout < 1:
        parser.error("Require warmup>=1, iterations>=2, timeout>=1")
    if args.worker:
        print("GPU_COMM_RESULT_JSON=" + json.dumps(worker(args)), flush=True)
        return
    before = None if args.cpu_fixture else snapshot(args.gpus)
    print(json.dumps({"selected_pair": args.gpus, "before": before}, indent=2), flush=True)
    if before and before["compute_processes"]:
        print("Other compute processes are present: results describe shared load, not isolated link performance.", flush=True)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--gpus", args.gpus,
               "--warmup", str(args.warmup), "--iterations", str(args.iterations),
               "--sizes-mib", *map(str, args.sizes_mib)]
    if args.debug_transport:
        command.append("--debug-transport")
    if args.cpu_fixture:
        command.append("--cpu-fixture")
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit("FAIL: communication worker timed out and was terminated; no model or training process was touched")
    if process.returncode:
        if process.stdout:
            print(process.stdout, flush=True)
        raise SystemExit(f"FAIL: communication worker exited with code {process.returncode}")
    # NCCL diagnostic lines can go to stdout; keep them without corrupting JSON.
    report, diagnostics = parse_worker_output(process.stdout)
    if diagnostics:
        print(diagnostics, flush=True)
    after = None if args.cpu_fixture else snapshot(args.gpus)
    report.update(before=before, after=after)
    external = bool(before and (before["compute_processes"] or after["compute_processes"]))
    report["other_compute_processes_observed"] = external
    report["measurement_conditions"] = "shared_load" if external else "no_compute_process_seen_at_endpoints; concurrent activity between snapshots not excluded"
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
