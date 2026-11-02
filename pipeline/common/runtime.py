"""Workspace-independent runtime primitives; safe to import before JAX."""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

UPSTREAM_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
# Reviewed descendants: local dataset roots, bounded Orbax I/O and deployment.
SUPPORTED_LOCAL_COMMITS = frozenset({
    "13a95ea16adeee33b7efd49f8f3dadc6aebe20b9",
    "eacc6a831df78bd8de7ef0804f48ef32f6de0052",
})
RUNTIME_PATHS = ("src/openpi", "scripts/train.py", "scripts/compute_norm_stats.py")
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream_revision(repo, *, protected_paths=RUNTIME_PATHS, verify_ancestry=True):
    repo = Path(repo)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        if actual not in SUPPORTED_LOCAL_COMMITS:
            raise ValueError(f"Expected OpenPI {UPSTREAM_COMMIT} or a reviewed local revision, found {actual}")
        if verify_ancestry:
            subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", UPSTREAM_COMMIT, actual],
                           check=True, capture_output=True, text=True)
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--name-only", "--", *protected_paths], text=True).strip()
    if dirty:
        raise ValueError(f"Unreviewed changes in local OpenPI runtime code: {dirty}")
    return actual


def prepare_environment(workspace, *, cpu=True, gpu=None, cache_dir=None):
    """Set shared offline/cache/thread/device settings after caller validation."""
    workspace = Path(workspace)
    sys.dont_write_bytecode = True
    os.environ.update(
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false", HF_LEROBOT_HOME=str(workspace / "datasets"),
        HF_DATASETS_CACHE=str(Path(cache_dir).resolve() if cache_dir else workspace / ".cache/huggingface/datasets"),
        OPENPI_DATA_HOME=str(workspace / "openpi_cache"),
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    os.environ.pop("LEROBOT_HOME", None)
    if cpu:
        os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ.pop("JAX_PLATFORMS", None)


def require_tokenizer(workspace):
    tokenizer = Path(workspace) / "openpi_cache/big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Missing official tokenizer: {tokenizer}")
    if sha256(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer SHA256 does not match the validated official asset")


def require_base_params(params):
    params = Path(params)
    if not (params / "_METADATA").is_file() or not (params / "manifest.ocdbt").is_file():
        raise FileNotFoundError(f"Missing pi05_base Orbax parameters: {params}")


def load_official_script(workspace, name):
    # Keep an importable module name so spawned DataLoader workers can unpickle it.
    root = (Path(workspace) / "openpi").resolve()
    sys.path.insert(0, str(root))
    module = importlib.import_module(f"scripts.{name}")
    if Path(module.__file__).resolve() != root / "scripts" / f"{name}.py":
        raise ImportError(f"Unexpected official script location: {module.__file__}")
    return module
