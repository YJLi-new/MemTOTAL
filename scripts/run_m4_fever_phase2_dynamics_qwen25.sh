#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-9701}"
RUN_ROOT="${2:-runs/verify/m4-fever-phase2-dynamics-qwen25}"
RESULT_ROOT="${3:-results/generated/m4-fever-phase2-dynamics-qwen25}"
PHASE0_METRICS="${4:-results/generated/review/m4-fever-shared-injection-qwen25/phase0-gate-sweep/metrics.json}"
RESUME_STAGE_B_ROOT="${5:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

declare -a SUITES=(
  "raw8 example_blocks_raw8"
  "triad6 triad_curated6"
)

SUITE_INDEX=0
for SUITE in "${SUITES[@]}"; do
  SUITE_NAME="$(awk '{print $1}' <<<"${SUITE}")"
  SUPPORT_VARIANT="$(awk '{print $2}' <<<"${SUITE}")"
  SUITE_SEED="$((BASE_SEED + SUITE_INDEX * 100))"
  SUITE_RUN_ROOT="${RUN_ROOT}/${SUITE_NAME}/phase2-selected"
  SUITE_RESULT_ROOT="${RESULT_ROOT}/${SUITE_NAME}"

  python scripts/run_m4_selected_shared_injection_suite.py \
    --config configs/exp/m4_fever_qwen25_phase2_common.yaml \
    --phase0_metrics "${PHASE0_METRICS}" \
    --support-serialization "${SUPPORT_VARIANT}" \
    --resume "${RESUME_STAGE_B_ROOT}" \
    --output_root "${SUITE_RUN_ROOT}" \
    --seed "${SUITE_SEED}"

  ./scripts/run_analysis.sh \
    --config configs/exp/m4_fever_qwen25_phase2_compare.yaml \
    --seed "$((SUITE_SEED + 40))" \
    --output_dir "${SUITE_RESULT_ROOT}/phase2-compare" \
    --input_root "${SUITE_RUN_ROOT}"

  SUITE_INDEX="$((SUITE_INDEX + 1))"
done

./scripts/run_analysis.sh \
  --config configs/exp/m4_fever_qwen25_phase2_dynamics_audit.yaml \
  --seed "$((BASE_SEED + 900))" \
  --output_dir "${RESULT_ROOT}/dynamics-audit" \
  --input_root "${RUN_ROOT}"

mkdir -p runs/review results/generated/review
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RUN_ROOT}/" "runs/review/m4-fever-phase2-dynamics-qwen25/"
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RESULT_ROOT}/" "results/generated/review/m4-fever-phase2-dynamics-qwen25/"

./scripts/publish_review_artifacts.sh

echo "m4 FEVER phase2 dynamics suite complete"
