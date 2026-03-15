#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${ROOT_DIR}/runs/smoke/${STAMP}"
RESULT_ROOT="${ROOT_DIR}/results/generated/${STAMP}"
mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

"${ROOT_DIR}/scripts/setup_env.sh"
"${ROOT_DIR}/scripts/setup_data.sh"

./scripts/run_train.sh \
  --config "${ROOT_DIR}/configs/exp/smoke_qwen25.yaml" \
  --seed 123 \
  --output_dir "${RUN_ROOT}/train" \
  --dry-run

./scripts/run_eval.sh \
  --config "${ROOT_DIR}/configs/exp/smoke_qwen25.yaml" \
  --seed 123 \
  --output_dir "${RUN_ROOT}/eval" \
  --checkpoint "${RUN_ROOT}/train/checkpoint.pt" \
  --dry-run

./scripts/run_analysis.sh \
  --config "${ROOT_DIR}/configs/exp/smoke_qwen25.yaml" \
  --seed 123 \
  --output_dir "${RESULT_ROOT}" \
  --input_root "${RUN_ROOT}" \
  --dry-run

"${ROOT_DIR}/scripts/run_ci_checks.sh"

echo "smoke-ok ${RUN_ROOT}"
