"""Isolated single-GPU LoRA batch-capacity tests using official train functions."""
import argparse
import csv
import datetime
import functools
import json
import logging
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

from pipeline.common.config import WORKSPACE

BENCHMARK_VERSION = 3


def system_available_mib():
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return None


def stop_test_process(process):
    # Each candidate has its own session: include its spawned DataLoader workers.
    def send(sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                process.terminate() if sig == signal.SIGTERM else process.kill()
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        process.wait(timeout=10)


def synchronized_update(step, mesh, rng, state, batch):
    """Use the official trainer's normal JIT invocation on the real TrainState.

    Explicit lower().compile() reconstructs it with ArgInfo objects in JAX 0.5.3,
    which conflicts with the pinned dataclass runtime type checker.
    """
    import jax
    from openpi.training import sharding

    with sharding.set_mesh(mesh):
        new_state, info = step(rng, state, batch)
    jax.block_until_ready((new_state, info))
    metrics = {key: float(value) for key, value in jax.device_get(info).items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Non-finite training metrics: {metrics}")
    return new_state, metrics


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def is_oom(message):
    text = message.lower()
    return any(token in text for token in (
        "out of memory", "out_of_memory", "resource_exhausted", "failed to allocate"))


def gpu_snapshot(selector):
    output = subprocess.check_output([
        "nvidia-smi", "-i", selector,
        "--query-gpu=index,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True)
    rows = list(csv.reader(output.strip().splitlines()))
    if len(rows) != 1:
        raise RuntimeError("Specify exactly one GPU")
    index, uuid, total, used = [value.strip() for value in rows[0]]
    return {"index": int(index), "uuid": uuid, "total_mib": float(total), "used_mib": float(used)}


def gpu_processes(uuid):
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True)
    result = {}
    for row in csv.reader(output.strip().splitlines()):
        if len(row) == 3 and row[0].strip() == uuid:
            memory = row[2].strip()
            result[int(row[1])] = float(memory) if memory.replace(".", "", 1).isdigit() else None
    return result


def require_available_gpu(selector, previous_pid=None):
    deadline = time.monotonic() + 10
    while True:
        gpu = gpu_snapshot(selector)
        processes = gpu_processes(gpu["uuid"])
        if not processes and gpu["used_mib"] < 512:
            return gpu
        # Allow the previous child to finish releasing its CUDA context.
        if previous_pid is not None and set(processes) <= {previous_pid} and time.monotonic() < deadline:
            time.sleep(0.5)
            continue
        raise RuntimeError(f"GPU is occupied: used={gpu['used_mib']} MiB, PIDs={list(processes)}; choose an allocated idle GPU")


def device_memory(device):
    stats = device.memory_stats()
    if stats is None:
        return None
    return {key: int(value) for key, value in stats.items() if isinstance(value, (int, float))}


def worker(args):
    output = Path(args.worker_output)
    report = {"status": "FAIL", "batch_size": args.batch_size, "num_workers": args.num_workers, "stage": "starting",
              "benchmark_version": BENCHMARK_VERSION}

    def stage(name):
        report["stage"] = name
        print(f"[batch={args.batch_size}] {name}", flush=True)

    try:
        from pipeline.common.config import prepare_environment, build_config, require_assets, load_official_script, describe
        prepare_environment(cpu=False, gpu=args.gpu, cache_dir=args.cache_dir)
        # Use the same allocator reservation policy as our formal trainer.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        import jax
        import numpy as np
        import pyarrow as pa
        import torch
        from openpi.training import data_loader, sharding

        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
        torch.set_num_threads(2)
        logging.basicConfig(level=logging.INFO)
        if jax.default_backend() != "gpu" or jax.device_count() != 1:
            raise RuntimeError("Expected exactly one visible GPU; no CPU fallback")
        device = jax.devices()[0]
        config = build_config(profile="finetune", exp_name="batch_capacity_probe",
                              batch_size=args.batch_size, num_workers=args.num_workers)
        require_assets(config, weights=True, full_stats=True)
        report["config"] = describe(config, "finetune")
        report["allocator"] = {"preallocate": True, "memory_fraction": 0.9,
                               "allocator_env": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default")}
        official = load_official_script("train")
        mesh = sharding.make_mesh(config.fsdp_devices)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rng = jax.random.key(config.seed)
        train_rng, init_rng = jax.random.split(rng)
        stage("data_loader")
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True,
                                               num_batches=args.warmup_steps + args.measure_steps)
        iterator = iter(loader)
        batch = next(iterator)
        jax.block_until_ready(batch)
        stage("model_initialization")
        state, state_sharding = official.init_train_state(config, init_rng, mesh, resume=False)
        jax.block_until_ready(state)
        report["memory_after_init"] = device_memory(device)
        step = jax.jit(functools.partial(official.train_step, config),
                       in_shardings=(replicated, state_sharding, data_sharding),
                       out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        stage("compile_and_first_update")
        started = time.monotonic()
        state, _ = synchronized_update(step, mesh, train_rng, state, batch)
        report["compile_and_first_update_seconds"] = round(time.monotonic() - started, 3)

        def update(current_state, current_batch):
            return synchronized_update(step, mesh, train_rng, current_state, current_batch)

        stage("warmup")
        # The first compiled call already counted as warmup update 1.
        for index in range(1, args.warmup_steps):
            batch = next(iterator)
            jax.block_until_ready(batch)
            state, _ = update(state, batch)
        stage("measurement")
        samples = []
        for index in range(args.measure_steps):
            started = time.monotonic()
            batch = next(iterator)
            jax.block_until_ready(batch)
            loaded = time.monotonic()
            state, metrics = update(state, batch)
            finished = time.monotonic()
            sample = {"index": index, "data_seconds": loaded - started,
                      "compute_seconds": finished - loaded, "total_seconds": finished - started,
                      **metrics}
            samples.append(sample)
            print(f"batch={args.batch_size} measured={index+1}/{args.measure_steps} "
                  f"compute={sample['compute_seconds']:.3f}s total={sample['total_seconds']:.3f}s "
                  f"loss={metrics['loss']:.6f} grad_norm={metrics['grad_norm']:.6f}", flush=True)
        updates = int(jax.device_get(state.step))
        if updates != args.warmup_steps + args.measure_steps or not any(s["grad_norm"] > 0 for s in samples):
            raise ValueError("Expected real optimizer updates and nonzero measured gradients")
        report.update(status="PASS", stage="completed", warmup_steps=args.warmup_steps,
                      measured_steps=args.measure_steps, optimizer_updates=updates, samples=samples,
                      median_compute_seconds=statistics.median(s["compute_seconds"] for s in samples),
                      median_total_seconds=statistics.median(s["total_seconds"] for s in samples),
                      compute_samples_per_second=args.batch_size * len(samples) / sum(s["compute_seconds"] for s in samples),
                      total_samples_per_second=args.batch_size * len(samples) / sum(s["total_seconds"] for s in samples),
                      memory_after_measurement=device_memory(device),
                      scope="Real LoRA weights/data/forward/backward/updates; synchronized timing; no checkpoints or task-quality assessment")
    except Exception as error:
        report.update(status="OOM" if is_oom(str(error)) else "FAIL",
                      error=f"{type(error).__name__}: {str(error)[:2000]}",
                      error_truncated=len(str(error)) > 2000)
        traceback.print_exc()
    finally:
        save_json(output, report)
        print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "config")}, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


def driver(args):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = WORKSPACE / "checks/output/batch_benchmark" / stamp
    output.mkdir(parents=True)
    summary = {"status": "RUNNING", "benchmark_version": BENCHMARK_VERSION,
               "requested_batches": args.batches, "num_workers": args.num_workers, "results": [],
               "output_dir": str(output), "scope": "Short capacity/throughput tests, not training-quality validation"}
    previous_pid = None
    print(f"Running batch benchmark version {BENCHMARK_VERSION}", flush=True)
    try:
        for batch in args.batches:
            gpu = require_available_gpu(args.gpu, previous_pid)
            summary["gpu"] = gpu
            result_path = output / f"batch_{batch}.json"
            log_path = output / f"batch_{batch}.log"
            command = [sys.executable, "-B", "-u", "-m", "pipeline.train.benchmark_batch",
                       "--worker-output", str(result_path), "--gpu", gpu["uuid"],
                       "--batch-size", str(batch), "--warmup-steps", str(args.warmup_steps),
                       "--measure-steps", str(args.measure_steps), "--num-workers", str(args.num_workers)]
            if args.cache_dir:
                command += ["--cache-dir", args.cache_dir]
            print(f"Starting batch={batch}, workers={args.num_workers}; log={log_path}", flush=True)
            peak_mib = 0.0
            ram_before = system_available_mib()
            ram_min = ram_before
            invalid_reason = None
            process = None
            with log_path.open("w") as log:
                try:
                    process = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=(os.name == "posix"))
                    started = time.monotonic()
                    last_progress = started
                    while process.poll() is None:
                        active = gpu_processes(gpu["uuid"])
                        if set(active) - {process.pid}:
                            invalid_reason = "Another GPU compute process appeared during the test"
                            break
                        peak_mib = max(peak_mib, active.get(process.pid) or 0)
                        available = system_available_mib()
                        if available is not None:
                            ram_min = available if ram_min is None else min(ram_min, available)
                        if time.monotonic() - last_progress >= 10:
                            print(f"batch={batch} running; elapsed={time.monotonic()-started:.0f}s; "
                                  f"sampled_peak={peak_mib:.0f} MiB; details={log_path.name}", flush=True)
                            last_progress = time.monotonic()
                        if time.monotonic() - started > args.timeout_seconds:
                            invalid_reason = "Candidate exceeded timeout"
                            break
                        time.sleep(0.5)
                finally:
                    if process is not None:
                        stop_test_process(process)
            previous_pid = process.pid
            result = json.loads(result_path.read_text()) if result_path.is_file() else {
                "status": "OOM" if is_oom(log_path.read_text(errors="replace")) else "FAIL",
                "batch_size": batch, "error": "Worker exited without a result; inspect candidate log"}
            if invalid_reason:
                result.update(status="INVALID", error=invalid_reason)
            elif process.returncode != 0 and result.get("status") == "PASS":
                result.update(status="FAIL", error="Worker exit code disagrees with PASS")
            result.update(exit_code=process.returncode, num_workers=args.num_workers,
                          sampled_process_memory_peak_mib=peak_mib or None,
                          system_available_before_mib=ram_before, system_available_min_mib=ram_min,
                          log_path=str(log_path))
            save_json(result_path, result)
            summary["results"].append(result)
            save_json(output / "summary.json", summary)
            print(f"batch={batch}: {result['status']}; compute_samples/s={result.get('compute_samples_per_second')}; "
                  f"total_samples/s={result.get('total_samples_per_second')}", flush=True)
            if result["status"] != "PASS":
                summary["stopped_after_batch"] = batch
                break
        summary["status"] = "COMPLETED" if len(summary["results"]) == len(args.batches) and all(
            result["status"] == "PASS" for result in summary["results"]) else "PARTIAL"
    except BaseException as error:
        summary.update(status="ABORTED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["passed_batches"] = [r["batch_size"] for r in summary["results"] if r["status"] == "PASS"]
        save_json(output / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2), flush=True)
    return 0 if summary["status"] == "COMPLETED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--cache-dir")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plan", action="store_true", help="Print test plan without GPU access or loading model")
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (args.gpu.isdigit() or (args.gpu.startswith("GPU-") and "," not in args.gpu)):
        parser.error("gpu must identify one GPU index or UUID")
    if min(args.batches) < 1 or args.batch_size < 1 or args.warmup_steps < 1 or args.measure_steps < 2 or args.timeout_seconds < 1:
        parser.error("Positive batch/warmup/timeout and at least two measured steps required")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    args.batches = sorted(set(args.batches))
    if args.plan:
        print(json.dumps({"batches": args.batches, "warmup_steps": args.warmup_steps,
                          "measure_steps": args.measure_steps, "profile": "finetune",
                          "num_workers": args.num_workers, "benchmark_version": BENCHMARK_VERSION,
                          "checkpoint_saving": False,
                          "scope": "Plan only; no GPU/model/data test performed"}, indent=2))
        return 0
    if args.worker_output:
        return worker(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    return driver(args)


if __name__ == "__main__":
    sys.exit(main())
