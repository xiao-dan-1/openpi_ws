"""Local Docker adapter for a downloaded checkpoint and constrained WSL memory."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

from pipeline.common.config import WORKSPACE, UPSTREAM_COMMIT, build_config, describe, prepare_environment, require_assets
from pipeline.common.validation import experiment_name
from pipeline.common.reporting import emit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--exp-name", type=experiment_name, default="libero_lora_01")
    parser.add_argument("--step", type=int, default=999)
    parser.add_argument("--output", type=Path, help="Optionally save the service record; otherwise print it only")
    parser.add_argument("--memory-fraction", type=float, default=0.70)
    parser.add_argument("--restore-concurrent-gb", type=int, default=3)
    parser.add_argument("--file-io-concurrency", type=int, default=1,
                        help="TensorStore I/O concurrency; 1 avoids transient host-memory spikes on WSL")
    args = parser.parse_args()
    if args.step < 0 or not 0 < args.memory_fraction < 1 or args.restore_concurrent_gb < 1 or args.file_io_concurrency < 1 or not 1 <= args.port <= 65535:
        parser.error("Invalid GPU memory fraction, restore limit, or port")
    prepare_environment(cpu=False, gpu=args.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.memory_fraction)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    logging.basicConfig(level=logging.INFO, force=True)
    import jax
    import orbax.checkpoint as ocp
    from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError("Expected one local inference GPU")
    config = build_config(profile="finetune", exp_name=args.exp_name, batch_size=16)
    require_assets(config, stats=False)
    checkpoint = config.checkpoint_dir / str(args.step)
    metadata = json.loads((checkpoint / "_CHECKPOINT_METADATA").read_text())
    if not metadata.get("commit_timestamp_nsecs"):
        raise ValueError("Downloaded checkpoint is not finalized")
    digest = hashlib.sha256((checkpoint / "assets/libero/norm_stats.json").read_bytes()).hexdigest()
    if digest != "7e19abf21b2f8837e1e27c468f4021efe359e38a9f6158acb83b6f978dad6b5a":
        raise ValueError("Downloaded statistics differ from the validated server run")
    current = describe(config, "finetune")
    current.update(norm_stats_sha256=digest, stats_path=str(checkpoint / "assets/libero/norm_stats.json"))
    run_dir = WORKSPACE / "runs/pi05_libero" / args.exp_name
    manifest = run_dir / "config.json"
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ["upstream_commit_expected", "baseline", "model", "local_dataset_alias", "norm_stats_sha256"]:
            if record.get(key) != current[key]:
                raise ValueError(f"Checkpoint inference differs from the training record: {key}")
    else:
        record = current
        record.update(record_origin="Reconstructed for local inference from validated pipeline configuration; not a downloaded server config.json",
                      imported_checkpoint=str(checkpoint), trained_updates_from_checkpoint_label=args.step + 1)
        run_dir.mkdir(parents=True, exist_ok=True)
        with manifest.open("x", encoding="utf-8") as output:
            output.write(json.dumps(record, indent=2))

    def bounded_checkpointer():
        return ocp.Checkpointer(ocp.PyTreeCheckpointHandler(restore_concurrent_gb=args.restore_concurrent_gb))

    original_ts_context = ts_utils.get_ts_context

    def bounded_ts_context(**kwargs):
        kwargs["file_io_concurrency_limit"] = args.file_io_concurrency
        kwargs["data_copy_concurrency_limit"] = args.file_io_concurrency
        return original_ts_context(**kwargs)

    logging.info("Loading downloaded LoRA checkpoint with bounded restore: %s", checkpoint)
    # Same official restoration logic, with a smaller host-memory concurrency budget.
    with patch.object(ts_utils, "get_ts_context", bounded_ts_context), patch.object(ocp, "PyTreeCheckpointer", bounded_checkpointer):
        policy = create_trained_policy(config, checkpoint)
    identity = {"experiment": args.exp_name, "checkpoint_step": args.step,
                "upstream_commit": UPSTREAM_COMMIT, "model": record["model"], "norm_stats_sha256": digest}
    info = {"pipeline": identity}
    emit_report({"metadata": info, "memory_fraction": args.memory_fraction,
                 "restore_concurrent_gb": args.restore_concurrent_gb,
                 "file_io_concurrency": args.file_io_concurrency}, args.output)
    logging.info("Local checkpoint loaded; serving on 127.0.0.1:%s", args.port)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=info).serve_forever()


if __name__ == "__main__":
    main()
