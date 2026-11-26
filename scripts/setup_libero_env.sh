#!/usr/bin/env bash
# Uses the official pinned LIBERO dependency lists in an independent Python 3.8 environment.
set -euo pipefail

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Usage: bash scripts/setup_libero_env.sh
Create or sync tools/libero-env using Python 3.8 and the pinned LIBERO dependencies.
Install uv in tools/bin if missing. Sync removes packages outside the resolved lockfile.
LIBERO_NETWORK_MODE=direct|inherit (default direct): package download route.
LIBERO_PYTHON_NETWORK_MODE=direct|inherit: optional independent Python download route.
UV_DEFAULT_INDEX: optional PyPI mirror. PyTorch cu113 uses its official extra index.
HELP
    exit 0 ;;
esac
if (( $# != 0 )); then
  echo "Unexpected arguments. Use --help for usage." >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/.." && pwd -P)"
openpi_dir="$workspace_root/openpi"
if [[ ! -f "$openpi_dir/examples/libero/requirements.txt" || \
      ! -f "$openpi_dir/third_party/libero/requirements.txt" || \
      ! -f "$openpi_dir/packages/openpi-client/pyproject.toml" ]]; then
  echo "OpenPI or LIBERO is missing. From the workspace, run: git submodule update --init --recursive" >&2
  exit 1
fi
actual_commit="$(git -C "$openpi_dir" rev-parse HEAD)"
case "$actual_commit" in
  215abfb217dbac7d5f1273282331b9b1866c0479) ;;
  13a95ea16adeee33b7efd49f8f3dadc6aebe20b9|eacc6a831df78bd8de7ef0804f48ef32f6de0052)
    git -C "$openpi_dir" merge-base --is-ancestor 215abfb217dbac7d5f1273282331b9b1866c0479 "$actual_commit"
    ;;
  *) echo "Unexpected OpenPI version: $actual_commit" >&2; exit 1 ;;
esac

export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$workspace_root/tools/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$workspace_root/.cache/uv}"
export UV_BUILD_CONSTRAINT="$workspace_root/pipeline/ops/libero_build_constraints.txt"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export GIT_LFS_SKIP_SMUDGE=1
network_mode="${LIBERO_NETWORK_MODE:-direct}"
python_network_mode="${LIBERO_PYTHON_NETWORK_MODE:-$network_mode}"
for mode in "$network_mode" "$python_network_mode"; do
  case "$mode" in direct|inherit) ;; *) echo "Network mode must be direct or inherit" >&2; exit 1 ;; esac
done
run_with_network() (
  local selected_mode="$1"
  shift
  if [[ "$selected_mode" == "direct" ]]; then export NO_PROXY='*' no_proxy='*'; fi
  "$@"
)
echo "Package download route: $network_mode; Python download route: $python_network_mode"

uv_bin="$workspace_root/tools/bin/uv"
if [[ ! -x "$uv_bin" ]]; then uv_bin="$(command -v uv || true)"; fi
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then uv_bin="$HOME/.local/bin/uv"; fi
if [[ -z "$uv_bin" ]]; then
  command -v curl >/dev/null || { echo "curl is required to install uv." >&2; exit 1; }
  installer="$(mktemp)"
  trap 'unlink -- "$installer"' EXIT
  uv_bin="$workspace_root/tools/bin/uv"
  echo "Installing uv in $workspace_root/tools/bin ..."
  run_with_network "$network_mode" curl --fail --location --show-error --silent --retry 3 \
    --connect-timeout 10 --max-time 120 https://astral.sh/uv/install.sh --output "$installer"
  run_with_network "$network_mode" env UV_INSTALL_DIR="$workspace_root/tools/bin" \
    UV_NO_MODIFY_PATH=1 sh "$installer"
fi

env_dir="$workspace_root/tools/libero-env"
env_python="$env_dir/bin/python"
if [[ -e "$env_dir" || -L "$env_dir" ]]; then
  if [[ ! -x "$env_python" ]] || ! "$env_python" -I -B -c \
      'import sys; sys.exit(sys.version_info[:2] != (3, 8) or sys.prefix == sys.base_prefix)'; then
    echo "Expected a usable Python 3.8 virtual environment at $env_dir. Move the existing path aside before retrying." >&2
    exit 1
  fi
else
  run_with_network "$python_network_mode" "$uv_bin" venv --python 3.8 "$env_dir"
fi
# Resolve the combined official lists first: the submodule list includes direct
# dependencies whose transitive dependencies are not in the example's lockfile.
lock_file="$workspace_root/tools/libero_requirements.lock"
run_with_network "$network_mode" "$uv_bin" pip compile --python "$env_python" \
  "$openpi_dir/examples/libero/requirements.txt" \
  "$openpi_dir/third_party/libero/requirements.txt" \
  "$openpi_dir/packages/openpi-client/pyproject.toml" \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match \
  --output-file "$lock_file"
run_with_network "$network_mode" "$uv_bin" pip sync --python "$env_python" "$lock_file" \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
run_with_network "$network_mode" "$uv_bin" pip install --python "$env_python" --no-deps \
  -e "$openpi_dir/packages/openpi-client" \
  -e "$openpi_dir/third_party/libero"
"$uv_bin" pip check --python "$env_python"

if ! torch_error="$("$env_python" -I -B -c 'import torch' 2>&1)"; then
  if [[ "$torch_error" != *"libtorch_cpu.so: cannot enable executable stack"* ]]; then
    echo "$torch_error" >&2
    exit 1
  fi
  patchelf_bin="$workspace_root/tools/patchelf/usr/bin/patchelf"
  if [[ ! -x "$patchelf_bin" ]]; then patchelf_bin="$(command -v patchelf || true)"; fi
  if [[ -z "$patchelf_bin" ]] || [[ "$("$patchelf_bin" --help 2>&1)" != *"--clear-execstack"* ]]; then
    echo "$torch_error" >&2
    echo "Install patchelf 0.18 or later to load the pinned PyTorch on this system, then rerun setup." >&2
    exit 1
  fi
  echo "Clearing the executable-stack flag from libtorch_cpu.so in $env_dir"
  "$env_python" -I -B - "$patchelf_bin" <<'PY'
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

library = Path(sysconfig.get_path("platlib")) / "torch/lib/libtorch_cpu.so"
# uv may link the installed file into its cache. Patch a separate copy.
with tempfile.TemporaryDirectory(prefix=".torch-patch-", dir=library.parent) as directory:
    patched = Path(directory) / library.name
    subprocess.run([sys.argv[1], "--clear-execstack", "--output", patched, library], check=True)
    shutil.copymode(library, patched)
    os.replace(patched, library)
PY
  "$env_python" -I -B -c 'import torch'
fi

"$uv_bin" pip freeze --python "$env_python" > "$workspace_root/tools/libero_installed.txt"
echo "LIBERO Python environment installed: $env_dir"
echo "Verify rendering from the workspace: python3 -B -u -m pipeline.eval.check_simulation"
