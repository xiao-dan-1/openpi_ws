"""38-dimensional server training configuration, independent from LIBERO."""
import dataclasses
import json
from pathlib import Path

from pipeline.q5.runtime import NAME, ASSET_ID, TRAIN_ID, FREEZE_POLICY, VIDEO_BACKEND, require_stats, sha256
from pipeline.q5.schema import JOINT_NAMES


def build_config(paths, *, profile="smoke", exp_name="q5_smoke", batch_size=1, steps=None, num_workers=0,
                 resume=False, warmup_steps=None, schedule_steps=None, weight_audit_path=None):
    import flax.nnx as nnx
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as upstream, optimizer
    from pipeline.q5.data import Q5DataConfig
    from pipeline.q5.weight_loaders import Q5WeightLoader
    if profile not in {"smoke", "finetune"} or batch_size < 1 or num_workers < 0 or (steps is not None and steps < 1):
        raise ValueError("Invalid Q5 training arguments")
    model = Pi0Config(pi05=True, action_dim=38, action_horizon=10, max_token_len=200,
                      discrete_state_input=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    smoke = profile == "smoke"
    warmup_steps = (10 if smoke else 1000) if warmup_steps is None else warmup_steps
    schedule_steps = (1000 if smoke else 30000) if schedule_steps is None else schedule_steps
    if warmup_steps < 0 or schedule_steps <= warmup_steps:
        raise ValueError("Schedule length must exceed the nonnegative warmup")
    freeze_filter = model.get_freeze_filter()
    return upstream.TrainConfig(
        name=NAME, project_name="openpi_q5", exp_name=exp_name, model=model,
        data=Q5DataConfig(repo_id=TRAIN_ID, repo_root=str(paths.train), assets=upstream.AssetsConfig(asset_id=ASSET_ID)),
        freeze_filter=freeze_filter,
        weight_loader=Q5WeightLoader(str(paths.base_params), str(weight_audit_path) if weight_audit_path else None,
                                    nnx.All(nnx.Param, nnx.Not(freeze_filter))),
        assets_base_dir=str(paths.workspace / "assets"), checkpoint_base_dir=str(paths.workspace / "checkpoints"),
        batch_size=batch_size, num_workers=num_workers, num_train_steps=steps or (100 if smoke else 1000),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=warmup_steps,
            peak_lr=5e-5, decay_steps=schedule_steps, decay_lr=5e-6),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0), ema_decay=None,
        log_interval=10 if smoke else 100, save_interval=50 if smoke else 1000,
        keep_period=50 if smoke else 5000, wandb_enabled=False, fsdp_devices=1, seed=42, resume=resume, overwrite=False,
        policy_metadata={"robot_type": "q5", "joint_names": list(JOINT_NAMES), "action_representation": "absolute_joint_targets"},
    )


def describe(paths, config, profile):
    report = require_stats(paths)
    return dict(profile=profile, name=config.name, exp_name=config.exp_name,
                model=dataclasses.asdict(config.model), freeze_policy=FREEZE_POLICY, video_backend=VIDEO_BACKEND,
                learning_rate=dataclasses.asdict(config.lr_schedule), batch_size=config.batch_size,
                optimizer=dataclasses.asdict(config.optimizer),
                num_workers=config.num_workers, seed=config.seed, ema_decay=config.ema_decay,
                train_repo_id=TRAIN_ID, train_data_root=str(paths.train),
                split_manifest_sha256=sha256(paths.split_manifest),
                norm_stats_sha256=report["sha256"], base_params=str(paths.base_params),
                checkpoint_dir=str(config.checkpoint_dir), steps=config.num_train_steps,
                implementation_sha256={name: sha256(Path(__file__).parent / name) for name in
                                       ["config.py", "data.py", "weight_loaders.py", "train.py", "schema.py", "runtime.py", "schedule.py", "parameter_audit.py"]},
                weight_migration="pretrained trunk + reinitialized 38-dimensional action projections")


def require_base_assets(paths):
    from pipeline.q5.runtime import require
    tokenizer = paths.workspace / "openpi_cache/big_vision/paligemma_tokenizer.model"
    require(tokenizer.is_file() and sha256(tokenizer) == "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6", "Missing or changed tokenizer")
    metadata = paths.base_params / "_METADATA"
    require(metadata.is_file() and (paths.base_params / "manifest.ocdbt").is_file(), "Missing local pi05_base checkpoint")
    tree = json.loads(metadata.read_text())["tree_metadata"]
    shapes = {key: value["value_metadata"].get("write_shape") for key, value in tree.items()}
    for key, shape in {"('params', 'action_in_proj', 'kernel')": [32, 1024],
                       "('params', 'action_out_proj', 'kernel')": [1024, 32],
                       "('params', 'action_out_proj', 'bias')": [32]}.items():
        require(shapes.get(key) == shape, f"Unexpected base checkpoint shape: {key}")
