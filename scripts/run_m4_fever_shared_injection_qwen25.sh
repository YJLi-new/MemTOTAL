#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-9101}"
RUN_ROOT="${2:-runs/verify/m4-fever-shared-injection-qwen25}"
RESULT_ROOT="${3:-results/generated/m4-fever-shared-injection-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

./scripts/run_train.sh \
  --config configs/exp/m4_fever_qwen25_base_only.yaml \
  --seed "${BASE_SEED}" \
  --output_dir "${RUN_ROOT}/pilot-A-base-only"

./scripts/run_train.sh \
  --config configs/exp/m4_fever_qwen25_teacher_text.yaml \
  --seed "$((BASE_SEED + 2))" \
  --output_dir "${RUN_ROOT}/pilot-T-teacher-text"

./scripts/run_analysis.sh \
  --config configs/exp/m4_fever_qwen25_writer_audit.yaml \
  --seed "$((BASE_SEED + 4))" \
  --output_dir "${RESULT_ROOT}/writer-audit" \
  --input_root "${RUN_ROOT}" \
  --resume "${RESUME_STAGE_B_ROOT}"

PHASE1_PASSED="$(python -c 'import json,sys; print("1" if json.loads(open(sys.argv[1]).read()).get("phase1_gate_passed") else "0")' "${RESULT_ROOT}/writer-audit/metrics.json")"

if [[ "${PHASE1_PASSED}" == "1" ]]; then
  ./scripts/run_train.sh \
    --config configs/exp/m4_fever_qwen25_injected_real.yaml \
    --seed "$((BASE_SEED + 10))" \
    --output_dir "${RUN_ROOT}/pilot-I-real" \
    --resume "${RESUME_STAGE_B_ROOT}"

  ./scripts/run_train.sh \
    --config configs/exp/m4_fever_qwen25_injected_shuffle.yaml \
    --seed "$((BASE_SEED + 12))" \
    --output_dir "${RUN_ROOT}/pilot-I-shuffle" \
    --resume "${RESUME_STAGE_B_ROOT}"

  ./scripts/run_train.sh \
    --config configs/exp/m4_fever_qwen25_injected_zero.yaml \
    --seed "$((BASE_SEED + 14))" \
    --output_dir "${RUN_ROOT}/pilot-I-zero" \
    --resume "${RESUME_STAGE_B_ROOT}"

  ./scripts/run_analysis.sh \
    --config configs/exp/m4_fever_qwen25_shared_injection_compare.yaml \
    --seed "$((BASE_SEED + 16))" \
    --output_dir "${RESULT_ROOT}/compare" \
    --input_root "${RUN_ROOT}"
fi

mkdir -p runs/review results/generated/review
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RUN_ROOT}/" "runs/review/m4-fever-shared-injection-qwen25/"
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RESULT_ROOT}/" "results/generated/review/m4-fever-shared-injection-qwen25/"

./scripts/publish_review_artifacts.sh

echo "m4 FEVER shared injection complete (phase1_gate_passed=${PHASE1_PASSED})"
