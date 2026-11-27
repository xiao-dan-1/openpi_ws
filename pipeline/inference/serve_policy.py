"""Serve a Q5 or LIBERO checkpoint using the official policy and WebSocket server."""
import argparse
import dataclasses
import json
import logging
from pathlib import Path
import sys
import time

from pipeline.common import runtime
from pipeline.common.memory import add_memory_argument, configure_memory_fraction
from pipeline.common.reporting import emit_report
from pipeline.common.validation import experiment_name, gpu_selectors, require

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


def verify_checkpoint_layout(checkpoint):
    """Validate that a checkpoint directory is complete and finalized."""
    require((checkpoint / "params").is_dir(), f"Missing finalized checkpoint: {checkpoint}")
    checkpoint_metadata = checkpoint / "_CHECKPOINT_METADATA"
    if checkpoint_metadata.is_file():
        require(json.loads(checkpoint_metadata.read_text()).get("commit_timestamp_nsecs"),
                f"Checkpoint is not finalized: {checkpoint}")
        return
    if (checkpoint / "train_state").exists():
        raise ValueError(f"Checkpoint is not finalized: {checkpoint}")
    require((checkpoint / "params/_METADATA").is_file() and (checkpoint / "params/manifest.ocdbt").is_file(),
            f"Missing finalized checkpoint: {checkpoint}")


def detect_libero_variant(checkpoint):
    """Distinguish official full-parameter checkpoints from local LoRA checkpoints."""
    params_metadata = checkpoint / "params/_METADATA"
    if params_metadata.is_file():
        tree_metadata = json.loads(params_metadata.read_text()).get("tree_metadata", {})
        if tree_metadata:
            return "lora" if any("lora" in key for key in tree_metadata) else "official"
    official_stats = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    local_stats = checkpoint / "assets/libero/norm_stats.json"
    if official_stats.is_file() and not local_stats.is_file():
        return "official"
    return "lora"


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
        from pipeline.common.config import build_config, build_official_config, describe, describe_model
        if original is not None:
            upstream = original.get("upstream_commit_expected")
            require(upstream == runtime.UPSTREAM_COMMIT,
                    f"Training record has an unsupported upstream commit: {upstream}")
            batch_size = original["batch_size"]
            config = build_config(profile=profile, exp_name=config_exp, batch_size=batch_size)
            model = describe(config, profile)["model"]
            asset_id = "libero"
            params_metadata = checkpoint / "params/_METADATA"
            if params_metadata.is_file() and json.loads(params_metadata.read_text()).get("tree_metadata"):
                detected = detect_libero_variant(checkpoint)
                recorded_model = original["model"]
                expected_variant = ("lora" if "lora" in str(recorded_model.get("paligemma", ""))
                                    or "lora" in str(recorded_model.get("action_expert", "")) else "official")
                require(detected == expected_variant,
                        f"Checkpoint variant ({detected}) differs from recorded training model ({expected_variant})")
        else:
            variant = detect_libero_variant(checkpoint)
            if variant == "official":
                official_stats = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
                asset_id = "physical-intelligence/libero" if official_stats.is_file() else "libero"
                config = build_official_config(asset_id=asset_id)
                model = describe_model(config)
            else:
                config = build_config(profile=profile, exp_name=config_exp, batch_size=16)
                model = describe(config, profile)["model"]
                asset_id = "libero"
    if original is not None:
        require(model == original["model"], "Inference model differs from recorded training model")
    return config, model, asset_id, exp_name, step, original


def make_warmup_observation(policy_config):
    import numpy as np

    if policy_config == "pi05_libero":
        return {
            "observation/state": np.zeros(8, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "warmup",
        }
    return {
        "state": np.zeros(38, dtype=np.float32),
        "images": {"image_head": np.zeros((224, 224, 3), dtype=np.uint8)},
        "prompt": "warmup",
    }


def warmup_policy(policy, policy_config):
    """Run one dummy inference before opening the WebSocket port to trigger JIT compilation."""
    import numpy as np

    started = time.monotonic()
    result = policy.infer(make_warmup_observation(policy_config))
    actions = np.asarray(result["actions"])
    require(actions.size > 0 and bool(np.all(np.isfinite(actions))),
            "Warmup inference returned empty or non-finite actions")
    return round(time.monotonic() - started, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=gpu_selectors)
    parser.add_argument("--policy.config", dest="policy_config", required=True, choices=list(POLICY_CONFIGS),
                        help="Policy config name, aligned with upstream serve_policy (pi05_libero or pi05_q5)")
    parser.add_argument("--policy.dir", dest="policy_dir", required=True, type=Path,
                        help="Checkpoint directory to load, aligned with upstream serve_policy")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0", help="Listen address (default: 0.0.0.0)")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip dummy warmup inference before opening the WebSocket server")
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
    verify_checkpoint_layout(checkpoint)
    config, model, asset_id, exp_name, step, original = build_policy_config(
        args.policy_config, checkpoint, workspace=WORKSPACE
    )
    runtime.require_tokenizer(WORKSPACE)
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
    variant = detect_libero_variant(checkpoint) if original is None and args.policy_config == "pi05_libero" else None
    label = f"{args.policy_config}/{variant}" if variant is not None else args.policy_config
    logging.info("Loading policy (%s) from %s with checkpoint-owned statistics", label, checkpoint)
    policy = create_trained_policy(config, checkpoint)
    warmup_seconds = None
    if not args.skip_warmup:
        logging.info("Running dummy warmup inference for %s...", label)
        warmup_seconds = warmup_policy(policy, args.policy_config)
        logging.info("Warmup inference completed in %.3fs", warmup_seconds)
    metadata = dict(policy.metadata)
    if original is not None:
        metadata["pipeline"] = {
            "experiment": exp_name, "checkpoint_step": step,
            "upstream_commit": runtime.UPSTREAM_COMMIT, "model": original["model"],
            "norm_stats_sha256": stats_sha,
        }
    else:
        pipeline_meta = {
            "config": args.policy_config, "checkpoint_dir": str(checkpoint),
            "upstream_commit": runtime.UPSTREAM_COMMIT, "model": model,
            "norm_stats_sha256": stats_sha,
        }
        if variant is not None:
            pipeline_meta["variant"] = variant
        metadata["pipeline"] = pipeline_meta
    emit_report({"config": args.policy_config, "checkpoint": str(checkpoint), "gpu": args.gpu,
                 "host": args.host, "port": args.port, "warmup_seconds": warmup_seconds,
                 "metadata": metadata}, args.output)
    logging.info("Policy loaded; starting WebSocket server at %s:%s", args.host, args.port)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
