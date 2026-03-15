#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_ID="${1:-Qwen/Qwen3-8B}"
LOCAL_DIR="${2:-/root/autodl-tmp/models/Qwen3-8B}"
CACHE_DIR="${3:-/root/autodl-tmp/hf-cache}"

mkdir -p "${LOCAL_DIR}" "${CACHE_DIR}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
# Prefer resumable parallel HTTP downloads for large sharded local staging.
# The installed Xet path has been much slower here, and hf_transfer does not
# handle resumed partial files well.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

python - <<'PY' "${MODEL_ID}" "${LOCAL_DIR}" "${CACHE_DIR}"
import json
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

model_id = sys.argv[1]
local_dir = Path(sys.argv[2])
cache_dir = Path(sys.argv[3])
download_cache_dir = local_dir / ".cache" / "huggingface" / "download"

index_path = local_dir / "model.safetensors.index.json"
single_file_path = local_dir / "model.safetensors"
config_path = local_dir / "config.json"

def _missing_indexed_weights(index_file: Path, target_dir: Path) -> list[str]:
    index_payload = json.loads(index_file.read_text())
    weight_map = index_payload.get("weight_map", {})
    if not weight_map:
        raise SystemExit(f"empty weight_map in {index_file}")
    return sorted(
        {
            filename
            for filename in weight_map.values()
            if not (target_dir / filename).exists()
        }
    )

def _clear_lockfiles() -> None:
    if not download_cache_dir.exists():
        return
    for lock_path in sorted(download_cache_dir.glob("*.lock")):
        lock_path.unlink(missing_ok=True)

def _download_with_retries(filename: str, *, max_attempts: int = 8) -> None:
    target_path = local_dir / filename
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        _clear_lockfiles()
        try:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=str(local_dir),
                    cache_dir=str(cache_dir),
                    resume_download=True,
                    local_dir_use_symlinks=False,
                )
            )
            if not target_path.exists() and downloaded_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded_path, target_path)
            if not target_path.exists():
                raise RuntimeError(f"download finished without materializing {target_path}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = min(60, 2 ** attempt)
            print(
                f"retrying model file download for {filename} "
                f"(attempt {attempt}/{max_attempts}) after error: {exc!r}"
            )
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise SystemExit(f"failed to download {filename} after {max_attempts} attempts: {last_error!r}")

if config_path.exists():
    if single_file_path.exists():
        print(f"local model already present: {local_dir}")
        raise SystemExit(0)
    if index_path.exists():
        missing = _missing_indexed_weights(index_path, local_dir)
        if not missing:
            print(f"local model already present: {local_dir}")
            raise SystemExit(0)
        print(
            f"partial model detected under {local_dir}; "
            f"resuming {len(missing)} missing shard(s): {missing[:5]}"
        )

required_small_files = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
]
for filename in required_small_files:
    if not (local_dir / filename).exists():
        _download_with_retries(filename)

if not config_path.exists():
    raise SystemExit(f"missing config.json after snapshot download: {local_dir}")
if not index_path.exists() and not single_file_path.exists():
    raise SystemExit(
        f"missing model.safetensors or model.safetensors.index.json after snapshot download: {local_dir}"
    )

PY

resolve_redirect_url() {
  local filename="$1"
  python - <<'PY' "${MODEL_ID}" "${filename}"
import sys

import requests
from huggingface_hub import hf_hub_url

model_id = sys.argv[1]
filename = sys.argv[2]
url = hf_hub_url(model_id, filename)
response = requests.get(url, allow_redirects=False, timeout=60)
response.raise_for_status()
print(response.headers.get("location", url))
PY
}

download_large_file_with_retries() {
  local filename="$1"
  local target_path="${LOCAL_DIR}/${filename}"
  local attempt
  for attempt in $(seq 1 8); do
    find "${LOCAL_DIR}/.cache/huggingface/download" -maxdepth 1 -name '*.lock' -delete 2>/dev/null || true
    if url="$(resolve_redirect_url "${filename}")" && wget -c -O "${target_path}" "${url}"; then
      return 0
    fi
    sleep_seconds=$((attempt < 6 ? 2 ** attempt : 60))
    echo "retrying large shard download for ${filename} (attempt ${attempt}/8)" >&2
    sleep "${sleep_seconds}"
  done
  echo "failed to download large shard ${filename} after 8 attempts" >&2
  return 1
}

mapfile -t missing_shards < <(
  python - <<'PY' "${LOCAL_DIR}"
import json
import sys
from pathlib import Path

local_dir = Path(sys.argv[1])
index_path = local_dir / "model.safetensors.index.json"
if not index_path.exists():
    raise SystemExit(0)
payload = json.loads(index_path.read_text())
weight_map = payload.get("weight_map", {})
for filename in sorted({name for name in weight_map.values() if not (local_dir / name).exists()}):
    print(filename)
PY
)

for filename in "${missing_shards[@]}"; do
  download_large_file_with_retries "${filename}"
done

python - <<'PY' "${LOCAL_DIR}"
import json
import sys
from pathlib import Path

local_dir = Path(sys.argv[1])
config_path = local_dir / "config.json"
index_path = local_dir / "model.safetensors.index.json"
single_file_path = local_dir / "model.safetensors"

if not config_path.exists():
    raise SystemExit(f"missing config.json after local staging: {local_dir}")
if single_file_path.exists():
    print(f"local model ready: {local_dir}")
    raise SystemExit(0)
if not index_path.exists():
    raise SystemExit(f"missing model.safetensors or model.safetensors.index.json after local staging: {local_dir}")

payload = json.loads(index_path.read_text())
weight_map = payload.get("weight_map", {})
missing = sorted({name for name in weight_map.values() if not (local_dir / name).exists()})
if missing:
    raise SystemExit(f"missing sharded model weights under {local_dir}: {missing[:5]}")
print(f"local model ready: {local_dir}")
PY
