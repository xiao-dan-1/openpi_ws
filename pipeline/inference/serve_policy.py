"""Serve this experiment's LoRA checkpoint using the official policy and server."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

from pipeline.common.config import WORKSPACE, UPSTREAM_COMMIT, build_config, describe, prepare_environment, require_assets
from pipeline.common.validation import experiment_name
from pipeline.common.reporting import emit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, help="Optionally save the service record; otherwise print it only")
    args = parser.parse_args()
    if args.step < 0 or not 1 <= args.port <= 65535:
        parser.error("step must be non-negative; port must be 1..65535")
    prepare_environment(cpu=False, gpu=args.gpu)
    logging.basicConfig(level=logging.INFO, force=True)
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError("Expected one visible GPU for policy inference")
    run_dir = WORKSPACE / "runs/pi05_libero" / args.exp_name
    original = json.loads((run_dir / "config.json").read_text())
    if original.get("upstream_commit_expected") != UPSTREAM_COMMIT:
        raise ValueError("Training record has a different upstream commit")
    config = build_config(profile=original["profile"], exp_name=args.exp_name,
                          batch_size=original["batch_size"])
    if describe(config, original["profile"])["model"] != original["model"]:
        raise ValueError("Inference model differs from recorded training model")
    require_assets(config, stats=False)
    checkpoint = config.checkpoint_dir / str(args.step)
    if not (checkpoint / "_CHECKPOINT_METADATA").is_file() or not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Missing finalized checkpoint: {checkpoint}")
    stats_path = checkpoint / "assets/libero/norm_stats.json"
    stats_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    if stats_sha != original["norm_stats_sha256"]:
        raise ValueError("Checkpoint statistics differ from the training record")
    logging.info("Loading LoRA policy from %s with checkpoint-owned statistics", checkpoint)
    policy = create_trained_policy(config, checkpoint)
    metadata = dict(policy.metadata)
    metadata["pipeline"] = {
        "experiment": args.exp_name, "checkpoint_step": args.step,
        "upstream_commit": UPSTREAM_COMMIT, "model": original["model"],
        "norm_stats_sha256": stats_sha,
    }
    emit_report({"checkpoint": str(checkpoint), "gpu": args.gpu, "port": args.port,
                 "metadata": metadata}, args.output)
    logging.info("Policy loaded; starting WebSocket server at 127.0.0.1:%s", args.port)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
