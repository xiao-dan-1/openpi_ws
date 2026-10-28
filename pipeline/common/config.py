"""Official pi05_libero baseline with explicit local-path and LoRA adaptations.

Import this module before JAX; call prepare_environment before importing OpenPI.
No upstream repository files are modified.
"""
import dataclasses
import importlib
import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
# Reviewed local descendant: dataset-root support, bounded Orbax I/O, and WSLg
# viewer/deployment tools. Model, policy, trainer and evaluator code match the
# pinned baseline. Unknown descendants still require an explicit review.
SUPPORTED_LOCAL_COMMITS = frozenset({
    "13a95ea16adeee33b7efd49f8f3dadc6aebe20b9",
    # Packaged pipeline entrypoint and lower I/O default; runtime code unchanged.
    "eacc6a831df78bd8de7ef0804f48ef32f6de0052",
})
SOURCE_REPO_ID = "physical-intelligence/libero"
LOCAL_REPO_ID = "libero"


def verify_upstream_revision():
    repo = WORKSPACE / "openpi"
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        if actual not in SUPPORTED_LOCAL_COMMITS:
            raise ValueError(f"Expected OpenPI {UPSTREAM_COMMIT} or a reviewed local revision, found {actual}")
        subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", UPSTREAM_COMMIT, actual],
            check=True, capture_output=True, text=True,
        )
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD", "--name-only", "--",
             "src/openpi", "scripts/train.py", "scripts/compute_norm_stats.py", "examples/libero/main.py"],
            text=True,
        ).strip()
        if dirty:
            raise ValueError(f"Unreviewed changes in local OpenPI runtime code: {dirty}")
    return actual


def prepare_environment(*, cpu=True, gpu=None, cache_dir=None):
    sys.dont_write_bytecode = True
    if not (WORKSPACE / "datasets/libero/meta/info.json").is_file():
        raise FileNotFoundError("Expected workspace/datasets/libero/meta/info.json")
    verify_upstream_revision()
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_LEROBOT_HOME": str(WORKSPACE / "datasets"),
        "HF_DATASETS_CACHE": str(Path(cache_dir).resolve() if cache_dir else
                                 WORKSPACE / ".cache/huggingface/datasets"),
        "OPENPI_DATA_HOME": str(WORKSPACE / "openpi_cache"),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    })
    os.environ.pop("LEROBOT_HOME", None)
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        if not gpu or not (gpu.isdigit() or (gpu.startswith("GPU-") and "," not in gpu)):
            raise ValueError("--gpu must specify one allocated GPU index or GPU-UUID")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"


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
    tokenizer = WORKSPACE / "openpi_cache/big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Missing official tokenizer: {tokenizer}")
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6":
        raise ValueError("Tokenizer SHA256 does not match the validated official asset")
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
        params = WORKSPACE / "openpi_cache/openpi-assets/checkpoints/pi05_base/params"
        if not (params / "_METADATA").is_file() or not (params / "manifest.ocdbt").is_file():
            raise FileNotFoundError(f"Missing pi05_base Orbax parameters: {params}")


def load_official_script(name):
    # A normal importable module name lets spawned DataLoader workers unpickle
    # the official RemoveStrings transform. A process-only importlib alias cannot.
    root = WORKSPACE / "openpi"
    sys.path.insert(0, str(root))
    module = importlib.import_module(f"scripts.{name}")
    if Path(module.__file__).resolve() != (root / "scripts" / f"{name}.py").resolve():
        raise ImportError(f"Unexpected official script location: {module.__file__}")
    return module


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


def write_manifest(config, profile, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(describe(config, profile), indent=2), encoding="utf-8")
