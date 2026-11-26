"""Serve a Q5 or LIBERO checkpoint using the official policy and WebSocket server."""
import argparse
import dataclasses
import json
import logging
from pathlib import Path
import sys

from pipeline.common import runtime
from pipeline.common.reporting import emit_report
from pipeline.common.validation import experiment_name, gpu_selectors, require
from pipeline.q5.memory import add_memory_argument, configure_memory_fraction

WORKSPACE = Path(__file__).resolve().parents[2]
POLICY_CONFIGS = ("pi05_libero", "pi05_q5")


def resolve_checkpoint_dir(policy_dir, *, workspace=None):
    root = (WORKSPACE if workspace is None else workspace).resolve()
    raw = Path(policy_dir)
    if raw.is_absolute():
        return raw.resolve()
    for candidate in (root / raw, Path.cwd() / raw):
        if candidate.exists():
            return candidate.resolve()
    return (root / raw).resolve()


def find_training_record(policy_config, checkpoint, *, workspace=None):
    root = (WORKSPACE if workspace is None else workspace).resolve()
    try:
        rel = checkpoint.resolve().relative_to(root / "checkpoints" / policy_config)
    except ValueError:
        return None, None, None
    if len(rel.parts) != 2 or not rel.parts[1].isdigit():
        return None, None, None
    try:
        exp_name = experiment_name(rel.parts[0])
    except argparse.ArgumentTypeError:
        return None, None, None
    record_path = root / "runs" / policy_config / exp_name / "config.json"
    if not record_path.is_file():
        return None, None, None
    return exp_name, int(rel.parts[1]), json.loads(record_path.read_text())


def build_policy_config(policy_config, checkpoint, *, workspace=None):
    root = WORKSPACE if workspace is None else workspace
    exp_name, step, original = find_training_record(policy_config, checkpoint, workspace=root)
    profile = original["profile"] if original is not None else "finetune"
    config_exp = exp_name or "serve"
    if policy_config == "pi05_q5":
        from pipeline.q5.config import build_config
        from pipeline.q5.runtime import ASSET_ID, Paths
        if original is not None:
            upstream = original.get("upstream_commit")
            supported = (runtime.UPSTREAM_COMMIT, *runtime.SUPPORTED_LOCAL_COMMITS)
            require(upstream in supported, f"Training record has an unsupported upstream commit: {upstream}")
        batch_size = original["batch_size"] if original is not None else 1
        config = build_config(Paths.create(root), profile=profile, exp_name=config_exp, batch_size=batch_size)
        model = dataclasses.asdict(config.model)
        asset_id = ASSET_ID
    else:
        from pipeline.common.config import build_config, describe
        if original is not None:
            upstream = original.get("upstream_commit_expected")
            require(upstream == runtime.UPSTREAM_COMMIT,
                    f"Training record has an unsupported upstream commit: {upstream}")
        batch_size = original["batch_size"] if original is not None else 16
        config = build_config(profile=profile, exp_name=config_exp, batch_size=batch_size)
        model = describe(config, profile)["model"]
        asset_id = "libero"
    if original is not None:
        require(model == original["model"], "Inference model differs from recorded training model")
    return config, model, asset_id, exp_name, step, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=gpu_selectors)
    parser.add_argument("--policy.config", dest="policy_config", required=True, choices=list(POLICY_CONFIGS),
                        help="Policy config name, aligned with upstream serve_policy (pi05_libero or pi05_q5)")
    parser.add_argument("--policy.dir", dest="policy_dir", required=True, type=Path,
                        help="Checkpoint directory to load, aligned with upstream serve_policy")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0", help="Listen address (default: 0.0.0.0)")
    parser.add_argument("--output", type=Path, help="Optionally save the service record; otherwise print it only")
    add_memory_argument(parser)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    if "," in args.gpu:
        parser.error("Policy inference requires one GPU")
    # Match Q5 training's exact revision whitelist, including shallow checkouts.
    runtime.verify_upstream_revision(WORKSPACE / "openpi", verify_ancestry=args.policy_config != "pi05_q5")
    runtime.prepare_environment(WORKSPACE, cpu=False, gpu=args.gpu)
    configure_memory_fraction(args.memory_fraction)
    # Resolve both upstream packages from this checkout, even after moving the workspace.
    sys.path[:0] = [
        str(WORKSPACE / "openpi/src"),
        str(WORKSPACE / "openpi/packages/openpi-client/src"),
    ]
    checkpoint = resolve_checkpoint_dir(args.policy_dir, workspace=WORKSPACE)
    config, model, asset_id, exp_name, step, original = build_policy_config(
        args.policy_config, checkpoint, workspace=WORKSPACE
    )
    runtime.require_tokenizer(WORKSPACE)
    checkpoint_metadata = checkpoint / "_CHECKPOINT_METADATA"
    require(checkpoint_metadata.is_file() and (checkpoint / "params").is_dir(),
            f"Missing finalized checkpoint: {checkpoint}")
    require(json.loads(checkpoint_metadata.read_text()).get("commit_timestamp_nsecs"),
            f"Checkpoint is not finalized: {checkpoint}")
    stats_path = checkpoint / "assets" / asset_id / "norm_stats.json"
    require(stats_path.is_file(), f"Missing checkpoint normalization statistics: {stats_path}")
    stats_sha = runtime.sha256(stats_path)
    if original is not None:
        require(stats_sha == original["norm_stats_sha256"], "Checkpoint statistics differ from the training record")
    logging.basicConfig(level=logging.INFO, force=True)
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError("Expected one visible GPU for policy inference")
    logging.info("Loading LoRA policy (%s) from %s with checkpoint-owned statistics", args.policy_config, checkpoint)
    policy = create_trained_policy(config, checkpoint)
    metadata = dict(policy.metadata)
    if original is not None:
        metadata["pipeline"] = {
            "experiment": exp_name, "checkpoint_step": step,
            "upstream_commit": runtime.UPSTREAM_COMMIT, "model": original["model"],
            "norm_stats_sha256": stats_sha,
        }
    else:
        metadata["pipeline"] = {
            "config": args.policy_config, "checkpoint_dir": str(checkpoint),
            "upstream_commit": runtime.UPSTREAM_COMMIT, "model": model,
            "norm_stats_sha256": stats_sha,
        }
    emit_report({"config": args.policy_config, "checkpoint": str(checkpoint), "gpu": args.gpu,
                 "host": args.host, "port": args.port,
                 "metadata": metadata}, args.output)
    logging.info("Policy loaded; starting WebSocket server at %s:%s", args.host, args.port)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
