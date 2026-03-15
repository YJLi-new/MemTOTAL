#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_ID="${1:-Qwen/Qwen3-4B}"
LOCAL_DIR="${2:-/root/autodl-tmp/models/Qwen3-4B}"
CACHE_DIR="${3:-/root/autodl-tmp/hf-cache}"

bash scripts/prepare_local_qwen3_model.sh \
  "${MODEL_ID}" \
  "${LOCAL_DIR}" \
  "${CACHE_DIR}"
