#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_ID="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
LOCAL_DIR="${2:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
CACHE_DIR="${3:-/root/autodl-tmp/hf-cache}"

mkdir -p "${LOCAL_DIR}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"

python - <<'PY' "${MODEL_ID}" "${LOCAL_DIR}" "${CACHE_DIR}"
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

model_id = sys.argv[1]
local_dir = Path(sys.argv[2])
cache_dir = Path(sys.argv[3])

small_files = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
]

for filename in small_files:
    downloaded = Path(hf_hub_download(model_id, filename, cache_dir=str(cache_dir)))
    shutil.copy2(downloaded, local_dir / filename)
PY

if [[ ! -f "${LOCAL_DIR}/model.safetensors" ]]; then
  URL="$(
    python - <<'PY' "${MODEL_ID}"
import sys
import requests
from huggingface_hub import hf_hub_url

model_id = sys.argv[1]
url = hf_hub_url(model_id, "model.safetensors")
response = requests.get(url, allow_redirects=False, timeout=30)
response.raise_for_status()
print(response.headers["location"])
PY
  )"
  wget -c -O "${LOCAL_DIR}/model.safetensors" "${URL}"
fi

python - <<'PY' "${LOCAL_DIR}"
import sys
from pathlib import Path

local_dir = Path(sys.argv[1])
required = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]
missing = [name for name in required if not (local_dir / name).exists()]
if missing:
    raise SystemExit(f"missing local qwen files: {missing}")
print(f"local qwen model ready: {local_dir}")
PY
