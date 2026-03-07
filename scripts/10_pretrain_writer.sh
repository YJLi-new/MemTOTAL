#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED="${1:-1701}"
OUTPUT_DIR="${2:-runs/verify/m3-core4-qwen25/stage-a}"
CONFIG_PATH="${3:-configs/exp/m3_stage_a_core4_qwen25_smoke.yaml}"

./scripts/run_train.sh \
  --config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}"
