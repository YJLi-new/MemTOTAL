#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED_BASE="${1:-3401}"
OUTPUT_ROOT="${2:-results/generated/m3-stage-c-sensitivity-audit}"
QWEN25_STAGEB="${3:-runs/verify/m3-core4-qwen25/stage-b}"
QWEN3_STAGEB="${4:-runs/verify/m3-core4-qwen3/stage-b}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

mkdir -p "${OUTPUT_ROOT}"

./scripts/run_analysis.sh \
  --config configs/exp/m3_stage_c_sensitivity_audit_qwen25.yaml \
  --seed "$((SEED_BASE + 0))" \
  --output_dir "${OUTPUT_ROOT}/qwen25" \
  --resume "${QWEN25_STAGEB}" \
  "${EXTRA_ARGS[@]}"

./scripts/run_analysis.sh \
  --config configs/exp/m3_stage_c_sensitivity_audit_qwen3.yaml \
  --seed "$((SEED_BASE + 10))" \
  --output_dir "${OUTPUT_ROOT}/qwen3" \
  --resume "${QWEN3_STAGEB}" \
  "${EXTRA_ARGS[@]}"
