"""Server GPU preflight: full weights, whole-tree freeze audit, forward and optional updates."""
import argparse
import functools
import json
from pathlib import Path

from pipeline.q5.runtime import Paths, prepare_environment, require, require_stats, official_train_module


def compile_training_step(official, config, mesh, state_sharding):
    """Match official scripts/train.py, including ownership of old state buffers."""
    import jax
    from openpi.training import sharding
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    return jax.jit(
        functools.partial(official.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--updates", type=int, default=0, help="0: load and forward only; >=3: also inspect real update samples")
    parser.add_argument("--output-dir", type=Path, help="Optionally save reports to a new directory; otherwise print results only")
    args = parser.parse_args()
    if args.updates < 0 or 0 < args.updates < 3:
        parser.error("Use --updates 0 or at least 3 (first update has zero LR during warmup)")
    paths = Paths.create(args.workspace)
    prepare_environment(paths, cpu=False, gpu=args.gpu)
    stats = require_stats(paths)
    from pipeline.q5.config import build_config, require_base_assets
    require_base_assets(paths)
    import flax.nnx as nnx
    import jax
    import numpy as np
    from openpi.training import data_loader, sharding
    from pipeline.q5.parameter_audit import audit_parameters, component
    require(jax.default_backend() == "gpu" and jax.device_count() == 1, "Run this preflight on one server GPU")
    out = args.output_dir.resolve() if args.output_dir is not None else None
    if out is not None:
        require(not out.exists(), "Output directory already exists; select a fresh --output-dir")
        out.mkdir(parents=True)
        (out / "INCOMPLETE").write_text("Server model checks in progress\n")
    config = build_config(paths, batch_size=1,
                          weight_audit_path=out / "weight_loading_report.json" if out is not None else None)
    official = official_train_module(paths)
    mesh = sharding.make_mesh(config.fsdp_devices)
    state, state_sharding = official.init_train_state(config, jax.random.key(config.seed), mesh, resume=False)
    jax.block_until_ready(state)
    # Audit the initialized tree in memory; checking never depends on a saved report.
    # Official initialization reads/merges the checkpoint and validates every shape.
    audit = audit_parameters(state.params.to_pure_dict(), config.trainable_filter)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False, num_batches=max(args.updates, 1))
    iterator = iter(loader)
    batch = next(iterator)
    graphdef = state.model_def
    def forward(params, observation, actions):
        model = nnx.merge(graphdef, params)
        model.eval()
        return model.compute_loss(jax.random.key(123), observation, actions, train=False)
    forward_step = jax.jit(forward)
    with sharding.set_mesh(mesh):
        loss = np.array(forward_step(state.params, *batch), copy=True)
    require(np.isfinite(loss).all(), "Model forward produced non-finite loss")
    # Standalone forward is only a check. Do not retain its compiled cache during
    # backward compilation; subsequent checks need no reference to its buffers.
    forward_step.clear_cache()
    del forward_step

    def sampled_values(params):
        result = {}
        for path, value in params.flat_state().items():
            key = "/".join(map(str, path))
            # Spread samples across each tensor; this is not a full-tensor equality proof.
            flat = value.value.reshape(-1)
            indices = np.unique(np.linspace(0, flat.size - 1, min(flat.size, 32)).astype(int))
            result[key] = np.asarray(jax.device_get(flat[indices])).astype(float)
        return result

    before = sampled_values(state.params) if args.updates else None
    metrics = []
    if args.updates:
        step = compile_training_step(official, config, mesh, state_sharding)
        for i in range(args.updates):
            if i:
                batch = next(iterator)
            with sharding.set_mesh(mesh):
                state, info = step(jax.random.key(config.seed), state, batch)
            numeric = {k: float(v) for k, v in jax.device_get(info).items()}
            require(all(np.isfinite(v) for v in numeric.values()), "Non-finite training metric")
            metrics.append(numeric)
        after = sampled_values(state.params)
        rows = {r["path"]: r for r in audit["parameters"]}
        changed_groups = set()
        frozen_checked = 0
        for key, old in before.items():
            changed = not np.array_equal(old, after[key])
            if not rows[key]["trainable"]:
                require(not changed, f"Frozen parameter sample changed: {key}")
                frozen_checked += 1
            elif changed:
                group = component(key)
                changed_groups.add(group + ("_lora" if "lora" in key else ""))
        require({"action_projection", "vision", "time_mlp", "language_lora", "action_expert_lora"} <= changed_groups,
                f"Expected trainable groups did not change in sampled values: {changed_groups}")
    else:
        changed_groups, frozen_checked = set(), 0
    report = {"status": "PASS", "full_weight_loading_tested": True, "model_forward_tested": True,
              "training_state_buffer_donation": bool(args.updates), "training_jit_matches_official": True,
              "whole_parameter_tree_audited": True, "action_expert_non_lora_leaves_checked": audit["action_expert_non_lora_leaves_checked"],
              "forward_loss_mean": float(loss.mean()), "real_updates": args.updates, "update_metrics": metrics,
              "changed_groups_in_samples": sorted(changed_groups), "frozen_leaf_samples_checked": frozen_checked,
              "update_check_scope": "up to 32 distributed elements per tensor; metadata freeze audit covers every parameter",
              "state_shape": list(batch[0].state.shape), "action_shape": list(batch[1].shape),
              "norm_stats_sha256": stats["sha256"], "uses_training_data_only": True,
              "checkpoint_save_restore_tested": False, "note": "No checkpoint saved; use a smoke experiment to test persistence."}
    if out is not None:
        (out / "model_check.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "INCOMPLETE").unlink()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
