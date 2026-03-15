#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED="${1:-1707}"
OUTPUT_DIR="${2:-runs/verify/m3-core4-qwen25/stage-c}"
RESUME_DIR="${3:-runs/verify/m3-core4-qwen25/stage-b}"
CONFIG_PATH="${4:-configs/exp/m3_stage_c_core4_qwen25_smoke.yaml}"

./scripts/run_train.sh \
  --config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${RESUME_DIR}"
