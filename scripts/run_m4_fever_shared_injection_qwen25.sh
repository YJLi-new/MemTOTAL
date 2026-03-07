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

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

./scripts/run_analysis.sh \
  --config configs/exp/m4_fever_qwen25_phase0_gate_sweep.yaml \
  --seed "${BASE_SEED}" \
  --output_dir "${RESULT_ROOT}/phase0-gate-sweep"

./scripts/run_analysis.sh \
  --config configs/exp/m4_fever_qwen25_phase1_writer_audit.yaml \
  --seed "$((BASE_SEED + 4))" \
  --output_dir "${RESULT_ROOT}/phase1-writer-audit" \
  --input_root "${RESULT_ROOT}/phase0-gate-sweep" \
  --resume "${RESUME_STAGE_B_ROOT}"

PHASE0_PASSED="$(python -c 'import json,sys; print("1" if json.loads(open(sys.argv[1]).read()).get("phase0_gate_passed") else "0")' "${RESULT_ROOT}/phase0-gate-sweep/metrics.json")"
PHASE1_PASSED="$(python -c 'import json,sys; print("1" if json.loads(open(sys.argv[1]).read()).get("phase1_gate_passed") else "0")' "${RESULT_ROOT}/phase1-writer-audit/metrics.json")"

if [[ "${PHASE0_PASSED}" == "1" && "${PHASE1_PASSED}" == "1" ]]; then
  python scripts/run_m4_selected_shared_injection_suite.py \
    --config configs/exp/m4_fever_qwen25_phase2_common.yaml \
    --phase0_metrics "${RESULT_ROOT}/phase0-gate-sweep/metrics.json" \
    --resume "${RESUME_STAGE_B_ROOT}" \
    --output_root "${RUN_ROOT}/phase2-selected" \
    --seed "$((BASE_SEED + 20))"

  ./scripts/run_analysis.sh \
    --config configs/exp/m4_fever_qwen25_phase2_compare.yaml \
    --seed "$((BASE_SEED + 24))" \
    --output_dir "${RESULT_ROOT}/phase2-compare" \
    --input_root "${RUN_ROOT}/phase2-selected"
fi

mkdir -p runs/review results/generated/review
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RUN_ROOT}/" "runs/review/m4-fever-shared-injection-qwen25/"
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RESULT_ROOT}/" "results/generated/review/m4-fever-shared-injection-qwen25/"

./scripts/publish_review_artifacts.sh

echo "m4 FEVER shared injection recovery complete (phase0=${PHASE0_PASSED}, phase1=${PHASE1_PASSED})"
