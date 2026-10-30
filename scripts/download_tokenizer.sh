#!/usr/bin/env bash
# Download the fixed PaliGemma tokenizer required by the training pipeline.
set -euo pipefail

if [[ $# == 1 && ( "$1" == -h || "$1" == --help ) ]]; then
  cat <<'HELP'
Usage: bash scripts/download_tokenizer.sh
Download and SHA-256 verify the PaliGemma tokenizer from official GCS.
Destination: <workspace>/openpi_cache/big_vision/paligemma_tokenizer.model
Valid existing files are reused; unexpected files are never overwritten.
Q5_TOKENIZER_URL can override the URL; the required SHA-256 stays fixed.
Requires curl for downloads and sha256sum for verification; no Python needed.
HELP
  exit 0
fi
if (( $# != 0 )); then
  echo "Unexpected arguments. Use --help for usage." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/.." && pwd -P)"
tokenizer="$workspace_root/openpi_cache/big_vision/paligemma_tokenizer.model"
expected_sha="8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
command -v sha256sum >/dev/null || { echo "Required command is missing: sha256sum" >&2; exit 1; }

verify_tokenizer() {
  if ! printf '%s  %s\n' "$expected_sha" "$1" | sha256sum --check --status; then
    echo "Tokenizer SHA-256 verification failed: $1" >&2
    echo "Existing files have not been overwritten; move an invalid target aside before retrying." >&2
    exit 1
  fi
}

if [[ -e "$tokenizer" || -L "$tokenizer" ]]; then
  verify_tokenizer "$tokenizer"
else
  command -v curl >/dev/null || { echo "Required command is missing: curl" >&2; exit 1; }
  mkdir -p -- "$(dirname -- "$tokenizer")"
  temporary="$(mktemp "${tokenizer}.download.XXXXXX")"
  trap 'rm -f -- "$temporary"' EXIT
  curl --fail --location --show-error --retry 3 --connect-timeout 10 --max-time 120 \
    "${Q5_TOKENIZER_URL:-https://storage.googleapis.com/big_vision/paligemma_tokenizer.model}" \
    --output "$temporary"
  verify_tokenizer "$temporary"
  # Publish only verified bytes; preserve a target created by another process.
  mv -nT -- "$temporary" "$tokenizer"
  verify_tokenizer "$tokenizer"
fi
echo "Tokenizer ready: $tokenizer (SHA-256 verified)"
