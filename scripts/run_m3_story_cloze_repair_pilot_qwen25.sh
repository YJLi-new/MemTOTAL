#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-9101}"
RUN_ROOT="${2:-runs/verify/m3-story-cloze-real-pilot-qwen25}"
RESULT_ROOT="${3:-results/generated/m3-story-cloze-real-pilot-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_shared_repair_fixed100.yaml \
  --seed "${BASE_SEED}" \
  --output_dir "${RUN_ROOT}/pilot-B-new-shared-repair" \
  --resume "${RESUME_STAGE_B_ROOT}"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_candidate_repair_fixed100.yaml \
  --seed "$((BASE_SEED + 2))" \
  --output_dir "${RUN_ROOT}/pilot-R-real" \
  --resume "${RESUME_STAGE_B_ROOT}"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_candidate_repair_shuffled_fixed100.yaml \
  --seed "$((BASE_SEED + 4))" \
  --output_dir "${RUN_ROOT}/pilot-R-shuffle" \
  --resume "${RESUME_STAGE_B_ROOT}"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_candidate_repair_zero_fixed100.yaml \
  --seed "$((BASE_SEED + 6))" \
  --output_dir "${RUN_ROOT}/pilot-R-zero" \
  --resume "${RESUME_STAGE_B_ROOT}"

./scripts/run_analysis.sh \
  --config configs/exp/stage_c_real_pilot_compare_story_repair.yaml \
  --seed "$((BASE_SEED + 8))" \
  --output_dir "${RESULT_ROOT}/repair-compare" \
  --input_root "${RUN_ROOT}"

mkdir -p runs/review results/generated/review
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RUN_ROOT}/" "runs/review/m3-story-cloze-real-pilot-qwen25/"
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RESULT_ROOT}/" "results/generated/review/m3-story-cloze-real-pilot-qwen25/"

./scripts/publish_review_artifacts.sh

echo "story-cloze repair pilot complete"
