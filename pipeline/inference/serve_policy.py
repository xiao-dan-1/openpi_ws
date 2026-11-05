"""Serve a Q5 or LIBERO checkpoint using the official policy and WebSocket server."""
import argparse
import dataclasses
import json
import logging
from pathlib import Path
import sys

from pipeline.common import runtime
from pipeline.common.validation import experiment_name, gpu_selectors, require
from pipeline.common.reporting import emit_report
from pipeline.q5.memory import add_memory_argument, configure_memory_fraction

WORKSPACE = Path(__file__).resolve().parents[2]


def experiment_config(robot, exp_name):
    record_path = WORKSPACE / "runs" / f"pi05_{robot}" / exp_name / "config.json"
    require(record_path.is_file(), f"Missing {robot} training record: {record_path}; check --robot")
    original = json.loads(record_path.read_text())
    upstream = original.get("upstream_commit" if robot == "q5" else "upstream_commit_expected")
    supported = (runtime.UPSTREAM_COMMIT, *runtime.SUPPORTED_LOCAL_COMMITS) if robot == "q5" else (runtime.UPSTREAM_COMMIT,)
    require(upstream in supported, f"Training record has an unsupported upstream commit: {upstream}")
    if robot == "q5":
        from pipeline.q5.config import build_config
        from pipeline.q5.runtime import ASSET_ID, Paths
        config = build_config(Paths.create(WORKSPACE), profile=original["profile"], exp_name=exp_name,
                              batch_size=original["batch_size"])
        model = dataclasses.asdict(config.model)
        asset_id = ASSET_ID
    else:
        from pipeline.common.config import build_config, describe
        config = build_config(profile=original["profile"], exp_name=exp_name,
                              batch_size=original["batch_size"])
        model = describe(config, original["profile"])["model"]
        asset_id = "libero"
    require(model == original["model"], "Inference model differs from recorded training model")
    return config, original, asset_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["libero", "q5"], default="libero",
                        help="Checkpoint configuration; default libero preserves existing commands")
    parser.add_argument("--gpu", required=True, type=gpu_selectors)
    parser.add_argument("--exp-name", required=True, type=experiment_name)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="Listen address; use 0.0.0.0 for remote clients")
    parser.add_argument("--output", type=Path, help="Optionally save the service record; otherwise print it only")
    add_memory_argument(parser)
    args = parser.parse_args()
    if args.step < 0 or not 1 <= args.port <= 65535:
        parser.error("step must be non-negative; port must be 1..65535")
    if "," in args.gpu:
        parser.error("Policy inference requires one GPU")
    # Match Q5 training's exact revision whitelist, including shallow checkouts.
    runtime.verify_upstream_revision(WORKSPACE / "openpi", verify_ancestry=args.robot != "q5")
    runtime.prepare_environment(WORKSPACE, cpu=False, gpu=args.gpu)
    configure_memory_fraction(args.memory_fraction)
    # Resolve both upstream packages from this checkout, even after moving the workspace.
    sys.path[:0] = [
        str(WORKSPACE / "openpi/src"),
        str(WORKSPACE / "openpi/packages/openpi-client/src"),
    ]
    config, original, asset_id = experiment_config(args.robot, args.exp_name)
    runtime.require_tokenizer(WORKSPACE)
    checkpoint = config.checkpoint_dir / str(args.step)
    checkpoint_metadata = checkpoint / "_CHECKPOINT_METADATA"
    require(checkpoint_metadata.is_file() and (checkpoint / "params").is_dir(),
            f"Missing finalized checkpoint: {checkpoint}")
    require(json.loads(checkpoint_metadata.read_text()).get("commit_timestamp_nsecs"),
            f"Checkpoint is not finalized: {checkpoint}")
    stats_sha = runtime.sha256(checkpoint / "assets" / asset_id / "norm_stats.json")
    require(stats_sha == original["norm_stats_sha256"], "Checkpoint statistics differ from the training record")
    logging.basicConfig(level=logging.INFO, force=True)
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError("Expected one visible GPU for policy inference")
    logging.info("Loading LoRA policy from %s with checkpoint-owned statistics", checkpoint)
    policy = create_trained_policy(config, checkpoint)
    metadata = dict(policy.metadata)
    metadata["pipeline"] = {
        "experiment": args.exp_name, "checkpoint_step": args.step,
        "upstream_commit": runtime.UPSTREAM_COMMIT, "model": original["model"],
        "norm_stats_sha256": stats_sha,
    }
    emit_report({"robot": args.robot, "checkpoint": str(checkpoint), "gpu": args.gpu,
                 "host": args.host, "port": args.port,
                 "metadata": metadata}, args.output)
    logging.info("Policy loaded; starting WebSocket server at %s:%s", args.host, args.port)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
