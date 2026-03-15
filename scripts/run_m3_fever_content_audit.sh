#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-7301}"
INPUT_ROOT="${2:-runs/review/m3-fever-real-pilot-qwen25}"
RESULT_ROOT="${3:-results/generated/review/m3-fever-real-pilot-qwen25}"

python -m analysis \
  --config configs/exp/stage_c_real_pilot_content_audit_fever.yaml \
  --seed "${SEED}" \
  --output_dir "${RESULT_ROOT}/content-audit" \
  --input_root "${INPUT_ROOT}"

echo "fever content audit complete"
