"""38-dimensional server training configuration, independent from LIBERO."""
import dataclasses
import json
from pathlib import Path

from pipeline.common.runtime import require_tokenizer, require_base_params
from pipeline.common.validation import require
from pipeline.q5.runtime import NAME, ASSET_ID, TRAIN_ID, FREEZE_POLICY, VIDEO_BACKEND, require_stats, sha256
from pipeline.q5.schema import JOINT_NAMES
from pipeline.q5.schedule import resolve_schedule


def build_config(paths, *, profile="smoke", exp_name="q5_smoke", batch_size=1, steps=None, num_workers=0,
                 resume=False, warmup_steps=None, schedule_steps=None, weight_audit_path=None, fsdp_devices=1):
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as upstream, optimizer
    from pipeline.q5.data import Q5DataConfig
    from pipeline.q5.weight_loaders import Q5WeightLoader
    if profile not in {"smoke", "finetune"} or batch_size < 1 or num_workers < 0 or (steps is not None and steps < 1):
        raise ValueError("Invalid Q5 training arguments")
    if fsdp_devices not in (1, 2) or batch_size % fsdp_devices:
        raise ValueError("Q5 supports fsdp_devices=1 or 2 with divisible global batch")
    model = Pi0Config(pi05=True, action_dim=38, action_horizon=10, max_token_len=200,
                      discrete_state_input=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    smoke = profile == "smoke"
    steps, warmup_steps, schedule_steps = resolve_schedule(profile, steps, warmup_steps, schedule_steps)
    freeze_filter = model.get_freeze_filter()
    config = upstream.TrainConfig(
        name=NAME, project_name="openpi_q5", exp_name=exp_name, model=model,
        data=Q5DataConfig(repo_id=TRAIN_ID, repo_root=str(paths.train), assets=upstream.AssetsConfig(asset_id=ASSET_ID)),
        freeze_filter=freeze_filter,
        assets_base_dir=str(paths.workspace / "assets"), checkpoint_base_dir=str(paths.workspace / "checkpoints"),
        batch_size=batch_size, num_workers=num_workers, num_train_steps=steps,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=warmup_steps,
            peak_lr=5e-5, decay_steps=schedule_steps, decay_lr=5e-6),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0), ema_decay=None,
        log_interval=10 if smoke else 100, save_interval=50 if smoke else 1000,
        keep_period=50 if smoke else 5000, wandb_enabled=False, fsdp_devices=fsdp_devices, seed=42, resume=resume, overwrite=False,
        policy_metadata={"robot_type": "q5", "joint_names": list(JOINT_NAMES), "action_representation": "absolute_joint_targets"},
    )
    return dataclasses.replace(config, weight_loader=Q5WeightLoader(
        str(paths.base_params), str(weight_audit_path) if weight_audit_path else None, config.trainable_filter))


def describe(paths, config, profile, *, stats_report=None):
    # Callers may pass the result of this invocation's preflight; never cache across runs.
    report = require_stats(paths) if stats_report is None else stats_report
    split_digest = sha256(paths.split_manifest)
    require(split_digest == report["split_manifest_sha256"], "Split manifest changed after preflight")
    require(sha256(paths.stats / "norm_stats.json") == report["sha256"], "Statistics file changed after preflight")
    return dict(profile=profile, name=config.name, exp_name=config.exp_name,
                model=dataclasses.asdict(config.model), freeze_policy=FREEZE_POLICY, video_backend=VIDEO_BACKEND,
                learning_rate=dataclasses.asdict(config.lr_schedule), batch_size=config.batch_size,
                fsdp_devices=config.fsdp_devices,
                optimizer=dataclasses.asdict(config.optimizer),
                num_workers=config.num_workers, seed=config.seed, ema_decay=config.ema_decay,
                train_repo_id=TRAIN_ID, train_data_root=str(paths.train),
                split_manifest_sha256=split_digest,
                norm_stats_sha256=report["sha256"], base_params=str(paths.base_params),
                checkpoint_dir=str(config.checkpoint_dir), steps=config.num_train_steps,
                implementation_sha256={name: sha256(Path(__file__).parent / name) for name in
                                       ["config.py", "data.py", "weight_loaders.py", "train.py", "schema.py", "runtime.py", "schedule.py", "parameter_audit.py", "memory.py", "parallelism.py", "checkpointing.py",
                                        "../common/runtime.py", "../common/validation.py"]},
                weight_migration="pretrained trunk + reinitialized 38-dimensional action projections")


def require_base_assets(paths):
    require_tokenizer(paths.workspace)
    require_base_params(paths.base_params)
    metadata = paths.base_params / "_METADATA"
    tree = json.loads(metadata.read_text())["tree_metadata"]
    shapes = {key: value["value_metadata"].get("write_shape") for key, value in tree.items()}
    for key, shape in {"('params', 'action_in_proj', 'kernel')": [32, 1024],
                       "('params', 'action_out_proj', 'kernel')": [1024, 32],
                       "('params', 'action_out_proj', 'bias')": [32]}.items():
        require(shapes.get(key) == shape, f"Unexpected base checkpoint shape: {key}")
