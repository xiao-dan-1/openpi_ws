#!/usr/bin/env bash
# Download JAX/Orbax weights used by this workspace.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/.." && pwd)"
venv_python="$workspace_root/.venv/bin/python"
python_bin="$venv_python"
if [[ ! -x "$python_bin" ]]; then
  # A system Python can show help before the environment is installed.
  python_bin="$(command -v python3)" || {
    echo "OpenPI environment is missing. Run: bash scripts/setup_env.sh" >&2
    exit 1
  }
fi

"$python_bin" -B - "$workspace_root" "$venv_python" "$@" <<'PY'
import argparse
import os
import sys
from pathlib import Path

BASE_MODEL = "pi05_base"
BASE_REPO = "robotgeneralist/openpi_checkpoint_mirrors2"
BASE_REVISION = "1fda6b04e83d625eb1a55d68db32050e248cac2b"
HF_MIRROR = "https://hf-mirror.com"
LIBERO_MODEL = "pi05_libero"
LIBERO_URL = f"gs://openpi-assets/checkpoints/{LIBERO_MODEL}"


def download_base(cache_root):
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT") or HF_MIRROR
    # Use HTTP resume; hf_transfer can discard existing .incomplete files.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = os.environ.get("HF_HUB_ETAG_TIMEOUT") or "30"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") or "60"

    from huggingface_hub import HfApi, snapshot_download

    print(f"Weight download endpoint: {os.environ['HF_ENDPOINT']}", flush=True)
    prefix = f"{BASE_MODEL}/params/"
    destination = cache_root / "openpi-assets/checkpoints"
    # Fetch the manifest first: snapshot_download can silently reuse a partial
    # local directory when the Hub is unreachable.
    info = HfApi().model_info(
        BASE_REPO, revision=BASE_REVISION, files_metadata=True,
        timeout=int(os.environ["HF_HUB_ETAG_TIMEOUT"]),
    )
    files = [entry for entry in info.siblings if entry.rfilename.startswith(prefix)]
    if not files:
        sys.exit(f"No checkpoint files found at {BASE_REPO}@{BASE_REVISION}/{prefix}")
    snapshot_download(
        repo_id=BASE_REPO,
        revision=BASE_REVISION,
        allow_patterns=[f"{prefix}*"],
        local_dir=destination,
    )
    for entry in files:
        path = destination / entry.rfilename
        if not path.is_file() or path.stat().st_size != entry.size:
            sys.exit(f"Incomplete checkpoint: {path} (expected {entry.size} bytes). Rerun to resume.")
    print(f"Weights ready: {destination / prefix} ({len(files)} files; names and sizes verified)")


def download_libero(cache_root):
    from openpi.shared import download

    os.environ["OPENPI_DATA_HOME"] = str(cache_root)
    checkpoint = download.maybe_download(LIBERO_URL, token="anon")
    for relative in ("params/manifest.ocdbt", "assets/physical-intelligence/libero/norm_stats.json"):
        path = checkpoint / relative
        if not path.is_file():
            sys.exit(f"Incomplete checkpoint: missing {path}. Move {checkpoint} aside and rerun.")
    print(f"Weights ready: {checkpoint}")


parser = argparse.ArgumentParser(
    prog="bash scripts/download_weights.sh",
    description="Download JAX/Orbax weights into the workspace's openpi_cache.",
    epilog=f"HF_ENDPOINT overrides the default base-weight mirror ({HF_MIRROR}).",
)
parser.add_argument(
    "model", nargs="?", choices=(BASE_MODEL, LIBERO_MODEL), default=BASE_MODEL,
    help="checkpoint to download (default: %(default)s)",
)
args = parser.parse_args(sys.argv[3:])
if not os.access(sys.argv[2], os.X_OK):
    sys.exit("OpenPI environment is missing. Run: bash scripts/setup_env.sh")

cache_root = Path(sys.argv[1]) / "openpi_cache"
if args.model == BASE_MODEL:
    download_base(cache_root)
else:
    download_libero(cache_root)
PY
