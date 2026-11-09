#!/usr/bin/env bash
set -euo pipefail

if [[ $# == 1 && ( "$1" == -h || "$1" == --help ) ]]; then
  cat <<'HELP'
Usage: bash scripts/download_libero.sh
Download LIBERO LeRobot v2.0 (~34.9 GB) from ModelScope by default.
Destination: <workspace>/datasets/libero
Set HF_ENDPOINT to use Hugging Face or a compatible mirror instead.
Requires the workspace environment from scripts/setup_env.sh and curl for ModelScope.
LIBERO_JOBS sets ModelScope concurrency (1-16, default 2).
Terminals show file sizes at start and one updating progress line. Logs show completed files.
Rerun to resume in the same staging directory. Files are verified before publishing.
An existing dataset is checked and never overwritten.
HELP
  exit 0
fi
if (( $# != 0 )); then
  echo "Unexpected arguments. Use --help for usage." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="$workspace_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Training environment is missing. Run: bash scripts/setup_env.sh" >&2
  exit 1
fi

# Download configuration. HF_ENDPOINT opts into the Hugging Face source.
destination="$workspace_root/datasets/libero"
export LIBERO_JOBS="${LIBERO_JOBS:-2}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export LIBERO_SOURCE="huggingface"
  export LIBERO_ENDPOINT="$HF_ENDPOINT"
  export LIBERO_REPO_ID="physical-intelligence/libero"
  export LIBERO_REVISION="9dfa69510ea9e1613fc54112bc706444b686a231"
  export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
  staging="$workspace_root/datasets/.libero-download"
else
  export LIBERO_SOURCE="modelscope"
  export LIBERO_ENDPOINT="https://www.modelscope.cn"
  export LIBERO_REPO_ID="buxiangqimingzi/physical-intelligence-libero"
  export LIBERO_REVISION="8bbb233c0f1451d16bf2c9c5bbd5bb218fc88451"
  export LIBERO_MODELSCOPE_ID="217553"
  staging="$workspace_root/datasets/.libero-modelscope-download"
fi

# Download implementation.
exec "$python_bin" -B - "$destination" "$staging" <<'PY'
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

use_hf = os.environ["LIBERO_SOURCE"] == "huggingface"
endpoint = os.environ["LIBERO_ENDPOINT"]
repo_id = os.environ["LIBERO_REPO_ID"]
revision = os.environ["LIBERO_REVISION"]
destination, staging = map(Path, sys.argv[1:])
datasets = destination.parent
if datasets.is_symlink():
    sys.exit(f"Expected a directory inside the workspace: {datasets}")
datasets.mkdir(exist_ok=True)

# Keep the directory locked for this process without leaving a lock file behind.
lock = os.open(datasets, os.O_RDONLY | os.O_DIRECTORY)
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(f"Another download is already running in {datasets}")

for directory in (destination, staging):
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        sys.exit(f"Expected a regular directory: {directory}")

print(f"Destination: {destination}", flush=True)
print(f"Source: {endpoint} ({repo_id}@{revision})", flush=True)
if use_hf:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi(endpoint=endpoint).dataset_info(
        repo_id, revision=revision, files_metadata=True, token=False,
        timeout=int(os.environ["HF_HUB_ETAG_TIMEOUT"]),
    )
    if info.sha != revision:
        sys.exit("Hugging Face returned an unexpected LIBERO revision")
    files = {f.rfilename: (f.size, None) for f in info.siblings
             if f.rfilename.startswith(("meta/", "data/"))}
else:
    try:
        jobs = int(os.environ["LIBERO_JOBS"])
        if not 1 <= jobs <= 16:
            raise ValueError
    except ValueError:
        sys.exit("LIBERO_JOBS must be an integer from 1 to 16")
    curl = ["curl", "-q", "-fsSL", "--connect-timeout", "10", "--retry", "2"]
    dataset_id = os.environ["LIBERO_MODELSCOPE_ID"]
    tree_url = (f"{endpoint}/api/v1/datasets/{dataset_id}/repo/tree?Revision={revision}"
                "&Root=%2F&Recursive=True&PageNumber=1&PageSize=2000")
    tree = json.loads(subprocess.check_output([*curl, "--max-time", "60", tree_url]))
    if tree.get("Code") != 200:
        sys.exit(f"ModelScope file listing failed: {tree.get('Message', tree.get('Code'))}")
    files = {f["Path"]: (f["Size"], f.get("Sha256")) for f in tree["Data"]["Files"]
             if f["Type"] == "blob" and f["Path"].startswith(("meta/", "data/"))}
    if any(not digest or len(digest) != 64 for _, digest in files.values()):
        sys.exit("Missing SHA-256 in ModelScope file listing")

expected = {f"meta/{name}" for name in ("info.json", "episodes.jsonl", "stats.json", "tasks.jsonl")}
expected.update(f"data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet" for i in range(1693))
if files.keys() != expected or any(size is None or size <= 0 for size, _ in files.values()):
    sys.exit("Unexpected LIBERO file listing")


def verify_file(path, size, digest):
    if path.resolve() != path or not path.is_file() or path.stat().st_size != size:
        sys.exit(f"Missing or invalid file: {path} (expected {size} bytes)")
    if digest:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                sys.exit(f"SHA-256 mismatch: {path}. Move this file aside before retrying.")


def download_modelscope(root, files, jobs):
    mib = 1024 ** 2
    terminal = sys.stdout.isatty()
    clear_line = "\r\x1b[K" if terminal else ""
    pending = iter(enumerate(sorted(files.items()), 1))
    active = {}
    verified = transferred = 0
    started = time.monotonic()
    print(f"{len(files)} files, {sum(size for size, _ in files.values()) / 1024**3:.2f} GiB, "
          f"{jobs} concurrent downloads", flush=True)
    try:
        while True:
            while len(active) < jobs:
                item = next(pending, None)
                if item is None:
                    break
                index, (name, (size, digest)) = item
                path = root / name
                if path.exists():
                    verify_file(path, size, digest)
                    verified += 1
                    continue
                part = path.with_suffix(path.suffix + ".part")
                offset = part.stat().st_size if part.is_file() else 0
                if offset >= size:
                    verify_file(part, size, digest)
                    part.rename(path)
                    verified += 1
                    continue
                label = f"[{index}/{len(files)}] {size / mib:.1f} MiB  {path.name}"
                if terminal:
                    print(f"{clear_line}{label}", flush=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{name}"
                command = [*curl, "--speed-limit", "1", "--speed-time", "60",
                           "--continue-at", "-", "--output", str(part), "--write-out", "%{speed_download}", url]
                transfer = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                active[transfer] = (path, part, size, digest, offset, label)

            if not active:
                break
            finished = [transfer for transfer in active if transfer.poll() is not None]
            for transfer in finished:
                path, part, size, digest, offset, label = active[transfer]
                speed, error = transfer.communicate()
                if transfer.returncode:
                    sys.exit(f"{label} download failed (curl {transfer.returncode}). Rerun to resume.\n{error.strip()}")
                verify_file(part, size, digest)
                part.rename(path)
                del active[transfer]
                verified += 1
                transferred += size - offset
                if not terminal:
                    print(f"{label}  {float(speed) / mib:.2f} MiB/s", flush=True)
            if finished:
                continue
            if terminal:
                received = total = new_bytes = 0
                for _, part, size, _, offset, _ in active.values():
                    count = part.stat().st_size if part.is_file() else 0
                    received += count
                    total += size
                    new_bytes += max(count - offset, 0)
                speed = (transferred + new_bytes) / mib / (time.monotonic() - started)
                print(f"{clear_line}{verified}/{len(files)} verified | {len(active)} active | "
                      f"{received / mib:.1f}/{total / mib:.1f} MiB | {speed:.2f} MiB/s", end="", flush=True)
            time.sleep(1)
    finally:
        for transfer in active:
            if transfer.poll() is None:
                transfer.terminate()
        for transfer in active:
            transfer.communicate()
        if terminal:
            print(clear_line, end="", flush=True)


root = destination if destination.exists() else staging
if root == staging:
    if any(path.is_symlink() for path in staging.rglob("*")):
        sys.exit(f"Unexpected symbolic link in download directory: {staging}")
    if use_hf:
        snapshot_download(
            repo_id, repo_type="dataset", revision=revision, endpoint=endpoint, token=False,
            allow_patterns=["data/*", "meta/*"], local_dir=staging,
        )

if root == staging and not use_hf:
    download_modelscope(root, files, jobs)
else:
    for name, (size, digest) in files.items():
        verify_file(root / name, size, digest)

if root == staging:
    if destination.exists() or destination.is_symlink():
        sys.exit(f"Destination appeared during download: {destination}. Verified files remain at {staging}")
    staging.rename(destination)

checks = "paths and sizes" if use_hf else "paths, sizes and SHA-256"
print(f"LIBERO ready: {destination} ({len(files)} files; verified {checks})")
PY
