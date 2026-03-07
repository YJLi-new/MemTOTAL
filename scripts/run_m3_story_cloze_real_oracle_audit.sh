#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-4101}"
RUN_ROOT="${2:-runs/verify/m3-story-cloze-real-pilot-qwen25}"
RESULT_ROOT="${3:-results/generated/m3-story-cloze-real-pilot-qwen25}"

./scripts/run_analysis.sh \
  --config configs/exp/stage_c_real_pilot_oracle_audit.yaml \
  --seed "${BASE_SEED}" \
  --output_dir "${RESULT_ROOT}/oracle" \
  --input_root "${RUN_ROOT}"

./scripts/publish_review_artifacts.sh

echo "story-cloze real oracle audit complete"
