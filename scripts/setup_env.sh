#!/usr/bin/env bash
# Install the training environment outside the upstream OpenPI submodule.
set -euo pipefail

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Usage: bash scripts/setup_env.sh
Linux/WSL: create or sync <workspace>/.venv using Python 3.11 and openpi/uv.lock.
If uv is missing, install it in ~/.local/bin without changing shell profiles.
Existing uv settings (package mirrors, proxy, cache, etc.) are inherited.
Existing .venv must be a usable Python 3.11 virtual environment, not a symlink.
Sync removes extra packages not in the lockfile; incompatible environments are
left unchanged and must be moved aside explicitly before retrying.
HELP
    exit 0 ;;
esac
if (( $# != 0 )); then
  echo "Unexpected arguments. Use --help for usage." >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this CUDA training setup on Linux or inside WSL." >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/.." && pwd -P)"
if [[ ! -f "$workspace_root/openpi/pyproject.toml" || ! -f "$workspace_root/openpi/uv.lock" ]]; then
  echo "OpenPI is missing. From the workspace, run: git submodule update --init --recursive" >&2
  exit 1
fi

env_dir="$workspace_root/.venv"
if [[ -L "$env_dir" ]]; then
  echo "Refusing symlinked environment: $env_dir. Use a local .venv directory." >&2
  exit 1
fi
if [[ -e "$env_dir" ]]; then
  if [[ ! -f "$env_dir/pyvenv.cfg" || ! -x "$env_dir/bin/python" ]] || \
      ! "$env_dir/bin/python" -I -B -c \
        'import sys; sys.exit(sys.version_info[:2] != (3, 11) or sys.prefix == sys.base_prefix)'; then
    echo "Expected a usable Python 3.11 virtual environment: $env_dir" >&2
    echo "Move the existing path aside explicitly before retrying; it has not been rebuilt." >&2
    exit 1
  fi
fi

uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  command -v curl >/dev/null || { echo "curl is required to install uv." >&2; exit 1; }
  installer="$(mktemp)"
  trap 'rm -f -- "$installer"' EXIT
  echo "Installing uv in $HOME/.local/bin ..."
  curl --fail --location --show-error --silent --retry 3 --connect-timeout 10 \
    --max-time 120 https://astral.sh/uv/install.sh --output "$installer"
  UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$installer"
  uv_bin="$HOME/.local/bin/uv"
fi

cd -- "$workspace_root"
echo "Workspace: $workspace_root"
echo "Environment: $env_dir"
echo "Syncing locked dependencies; extra packages will be removed."
# uv sync also installs OpenPI and openpi-client as editable packages.
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$env_dir" \
  "$uv_bin" sync --project "$workspace_root/openpi" --frozen --python 3.11

# Keep this import check on CPU; GPU checks belong to the training pipeline.
JAX_PLATFORMS=cpu "$env_dir/bin/python" -B - <<'PY'
import sys

import av
import flax
import jax
import openpi
import pyarrow
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"Environment: {sys.prefix}")
print(f"JAX: {jax.__version__}; Torch: {torch.__version__}")
print("Training dependencies imported successfully.")
PY
echo "Ready. Next: bash scripts/download_weights.sh"
