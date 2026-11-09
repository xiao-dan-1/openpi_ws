"""Official pi05_libero baseline with explicit local-path and LoRA adaptations.

Import this module before JAX; call prepare_environment before importing OpenPI.
No upstream repository files are modified.
"""
import dataclasses
import hashlib
import json
from pathlib import Path

from pipeline.common import runtime
from pipeline.common.memory import DEFAULT_MEMORY_FRACTION, configure_memory_fraction
from pipeline.common.runtime import UPSTREAM_COMMIT, SUPPORTED_LOCAL_COMMITS
from pipeline.common.validation import gpu_selectors

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE_REPO_ID = "physical-intelligence/libero"
LOCAL_REPO_ID = "libero"


def verify_upstream_revision():
    return runtime.verify_upstream_revision(
        WORKSPACE / "openpi", protected_paths=(*runtime.RUNTIME_PATHS, "examples/libero/main.py"))


def prepare_environment(*, cpu=True, gpu=None, cache_dir=None, memory_fraction=DEFAULT_MEMORY_FRACTION):
    if not (WORKSPACE / "datasets/libero/meta/info.json").is_file():
        raise FileNotFoundError("Expected workspace/datasets/libero/meta/info.json")
    verify_upstream_revision()
    if not cpu:
        gpu = gpu_selectors(gpu or "")
        if "," in gpu:
            raise ValueError("--gpu must specify one allocated GPU index or GPU-UUID")
    runtime.prepare_environment(WORKSPACE, cpu=cpu, gpu=gpu, cache_dir=cache_dir)
    if not cpu:
        configure_memory_fraction(memory_fraction)


def build_config(*, profile="smoke", exp_name="smoke", batch_size=None, steps=None,
                 resume=False, assets_base_dir=None, num_workers=0):
    from openpi.training import config as upstream
    from openpi.training import optimizer, weight_loaders

    if (batch_size is not None and batch_size < 1) or (steps is not None and steps < 1):
        raise ValueError("batch_size and steps must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    baseline = upstream.get_config("pi05_libero")
    if not (baseline.model.pi05 and baseline.model.action_horizon == 10
            and baseline.model.action_dim == 32 and not baseline.model.discrete_state_input
            and not baseline.data.extra_delta_transform):
        raise ValueError("Official pi05_libero baseline changed; review pipeline assumptions")
    assets = Path(assets_base_dir).resolve() if assets_base_dir else WORKSPACE / "assets"
    # Local ID 'libero' resolves directly to workspace/datasets/libero. It is a
    # filesystem alias, NOT a different dataset or a published Hugging Face repo.
    data_changes = {
        "repo_id": LOCAL_REPO_ID,
        "assets": upstream.AssetsConfig(assets_dir=str(assets / baseline.name), asset_id=LOCAL_REPO_ID),
    }
    # Disable the WSL checkout's pre-existing root extension so both machines
    # exercise the same HF_LEROBOT_HOME/local-alias path resolution.
    if "repo_root" in {field.name for field in dataclasses.fields(baseline.data)}:
        data_changes["repo_root"] = None
    data = dataclasses.replace(baseline.data, **data_changes)
    common = {
        "data": data, "assets_base_dir": str(assets),
        "checkpoint_base_dir": str(WORKSPACE / "checkpoints"),
        "exp_name": exp_name, "num_workers": num_workers, "wandb_enabled": False,
        "resume": resume, "overwrite": False, "fsdp_devices": 1,
        "pytorch_weight_path": None,
        "weight_loader": weight_loaders.CheckpointWeightLoader(
            str(WORKSPACE / "openpi_cache/openpi-assets/checkpoints/pi05_base/params")),
    }
    if profile == "stats":
        return dataclasses.replace(baseline, batch_size=batch_size or 32, **common)
    if profile not in {"smoke", "finetune"}:
        raise ValueError(f"Unknown profile: {profile}")
    model = dataclasses.replace(baseline.model, paligemma_variant="gemma_2b_lora",
                                action_expert_variant="gemma_300m_lora")
    smoke = profile == "smoke"
    return dataclasses.replace(
        baseline, model=model, freeze_filter=model.get_freeze_filter(), ema_decay=None,
        batch_size=batch_size or 1, num_train_steps=steps or (100 if smoke else 30000),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10 if smoke else 1000,
                                                  peak_lr=5e-5, decay_steps=1000 if smoke else 30000,
                                                  decay_lr=5e-6),
        log_interval=10 if smoke else 100, save_interval=50 if smoke else 1000,
        keep_period=50 if smoke else 5000, **common,
    )


def require_assets(config, *, stats=True, weights=False, full_stats=False):
    runtime.require_tokenizer(WORKSPACE)
    if stats:
        path = config.assets_dirs / LOCAL_REPO_ID / "norm_stats.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run python -m pipeline.data.compute_norm_stats first: {path}")
        if full_stats:
            report_path = path.parent / "statistics_run.json"
            if not report_path.is_file():
                raise FileNotFoundError(f"Missing official full-statistics run record: {report_path}")
            report = json.loads(report_path.read_text())
            if report.get("status") != "PASS" or report.get("probe_only") is not False:
                raise ValueError("Training requires completed full official statistics, not probe statistics")
            if report.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Active statistics changed since their official computation")
    if weights:
        runtime.require_base_params(WORKSPACE / "openpi_cache/openpi-assets/checkpoints/pi05_base/params")


def load_official_script(name):
    return runtime.load_official_script(WORKSPACE, name)


def describe(config, profile):
    stats = config.assets_dirs / LOCAL_REPO_ID / "norm_stats.json"
    return {
        "upstream_commit_expected": UPSTREAM_COMMIT,
        "upstream_commit_actual": verify_upstream_revision(), "baseline": "pi05_libero",
        "profile": profile, "source_dataset": SOURCE_REPO_ID,
        "local_dataset_alias": LOCAL_REPO_ID, "data_root": str(WORKSPACE / "datasets/libero"),
        "model": {"pi05": config.model.pi05, "paligemma": config.model.paligemma_variant,
                  "action_expert": config.model.action_expert_variant,
                  "action_horizon": config.model.action_horizon, "action_dim": config.model.action_dim,
                  "max_token_len": config.model.max_token_len,
                  "discrete_state_input": config.model.discrete_state_input},
        "batch_size": config.batch_size, "workers": config.num_workers,
        "steps": config.num_train_steps, "ema_decay": config.ema_decay,
        "learning_rate": dataclasses.asdict(config.lr_schedule),
        "stats_path": str(config.assets_dirs / LOCAL_REPO_ID / "norm_stats.json"),
        "norm_stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest() if stats.is_file() else None,
        "checkpoint_dir": str(config.checkpoint_dir), "resume": config.resume,
        "adaptation": "Local paths; single-device LoRA; batch/steps/LR/EMA differ from official full finetuning",
    }


def write_manifest(config, profile, destination, *, allocator_settings=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = describe(config, profile)
    if allocator_settings is not None:
        manifest["allocator_settings"] = allocator_settings
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
