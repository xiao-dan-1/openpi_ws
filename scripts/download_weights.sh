#!/usr/bin/env bash
# Download the JAX/Orbax pi05_base checkpoint used by this workspace.
set -euo pipefail

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Usage: bash scripts/download_weights.sh
Download pinned pi05_base JAX/Orbax parameters (~12.4 GB).
Destination: <workspace>/openpi_cache/openpi-assets/checkpoints/pi05_base/params
HF_ENDPOINT defaults to https://hf-mirror.com; override it to switch sites.
Completed files and Hugging Face partial downloads are reused on reruns.
Remote file names and sizes must be verified before reporting success.
Tokenizer and datasets are prepared separately.
HELP
    exit 0 ;;
esac
if (( $# != 0 )); then
  echo "Unexpected arguments. Use --help for usage." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/.." && pwd)"
python_bin="$workspace_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Training environment is missing. Run: bash scripts/setup_env.sh" >&2
  exit 1
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Use HTTP resume; hf_transfer can discard existing .incomplete files.
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
echo "Weight download endpoint: $HF_ENDPOINT"
"$python_bin" -B - "$workspace_root" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo_id = "robotgeneralist/openpi_checkpoint_mirrors2"
revision = "1fda6b04e83d625eb1a55d68db32050e248cac2b"
prefix = "pi05_base/params"
destination = Path(sys.argv[1]) / "openpi_cache/openpi-assets/checkpoints"
# Fetch the manifest first: snapshot_download alone can silently reuse a partial
# local directory when the Hub is unreachable. A failed lookup must fail here.
info = HfApi().model_info(
    repo_id, revision=revision, files_metadata=True,
    timeout=int(os.environ["HF_HUB_ETAG_TIMEOUT"]),
)
files = [entry for entry in info.siblings if entry.rfilename.startswith(f"{prefix}/")]
if not files:
    sys.exit(f"No checkpoint files found at {repo_id}@{revision}/{prefix}")
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    allow_patterns=[f"{prefix}/*"],
    local_dir=destination,
)
for entry in files:
    path = destination / entry.rfilename
    if not path.is_file() or path.stat().st_size != entry.size:
        sys.exit(f"Incomplete checkpoint: {path} (expected {entry.size} bytes). Rerun to resume.")
print(f"Weights ready: {destination / prefix} ({len(files)} files; names and sizes verified)")
PY
