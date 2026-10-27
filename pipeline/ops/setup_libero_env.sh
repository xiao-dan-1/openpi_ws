#!/usr/bin/env bash
# Uses the official pinned LIBERO dependency lists in an independent Python 3.8 environment.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash pipeline/ops/setup_libero_env.sh"
  echo "Creates tools/libero-env using Python 3.8; does not modify openpi/.venv."
  echo "LIBERO_NETWORK_MODE=direct|inherit (default direct): package download route."
  echo "LIBERO_PYTHON_NETWORK_MODE=direct|inherit: optional independent Python download route."
  echo "UV_DEFAULT_INDEX: optional verified PyPI mirror; PyTorch cu113 still uses its official extra index."
  exit 0
fi
ops_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$ops_dir/../.." && pwd)"
actual_commit="$(git -C "$workspace_root/openpi" rev-parse HEAD)"
case "$actual_commit" in
  215abfb217dbac7d5f1273282331b9b1866c0479) ;;
  13a95ea16adeee33b7efd49f8f3dadc6aebe20b9|eacc6a831df78bd8de7ef0804f48ef32f6de0052)
    git -C "$workspace_root/openpi" merge-base --is-ancestor 215abfb217dbac7d5f1273282331b9b1866c0479 "$actual_commit"
    ;;
  *) echo "Unexpected OpenPI version: $actual_commit" >&2; exit 1 ;;
esac
uv_bin="$workspace_root/tools/bin/uv"
if [[ ! -x "$uv_bin" ]]; then uv_bin="$(command -v uv)"; fi
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$workspace_root/tools/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$workspace_root/.cache/uv}"
export UV_BUILD_CONSTRAINT="$ops_dir/libero_build_constraints.txt"
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
env_dir="$workspace_root/tools/libero-env"
if [[ ! -x "$env_dir/bin/python" ]]; then
  run_with_network "$python_network_mode" "$uv_bin" venv --python 3.8 "$env_dir"
fi
"$env_dir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 8), "Expected dedicated Python 3.8 environment"'
# Resolve the combined official lists first: the submodule list includes direct
# dependencies whose transitive dependencies are not in the example's lockfile.
lock_file="$workspace_root/tools/libero_requirements.lock"
run_with_network "$network_mode" "$uv_bin" pip compile --python "$env_dir/bin/python" \
  "$workspace_root/openpi/examples/libero/requirements.txt" \
  "$workspace_root/openpi/third_party/libero/requirements.txt" \
  "$workspace_root/openpi/packages/openpi-client/pyproject.toml" \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match \
  --output-file "$lock_file"
run_with_network "$network_mode" "$uv_bin" pip sync --python "$env_dir/bin/python" "$lock_file" \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
run_with_network "$network_mode" "$uv_bin" pip install --python "$env_dir/bin/python" --no-deps \
  -e "$workspace_root/openpi/packages/openpi-client" \
  -e "$workspace_root/openpi/third_party/libero"
"$uv_bin" pip check --python "$env_dir/bin/python"
"$uv_bin" pip freeze --python "$env_dir/bin/python" > "$workspace_root/tools/libero_installed.txt"
echo "LIBERO Python environment installed: $env_dir"
echo "Next: verify simulation system libraries and run python -m pipeline.eval.eval_libero."
