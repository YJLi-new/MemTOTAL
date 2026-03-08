#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-12701}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/m4-fever-deep-prompt-recovery-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/m4-fever-deep-prompt-recovery-qwen25}"
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

./scripts/run_analysis.sh \
  --config configs/exp/m4_fever_qwen25_split_prep.yaml \
  --seed "${BASE_SEED}" \
  --output_dir "${RESULT_ROOT}/split-prep"

python scripts/run_m4_selected_shared_injection_suite.py \
  --config configs/exp/m4_fever_qwen25_phase2_val_deep_common.yaml \
  --phase0_metrics "${PHASE0_METRICS}" \
  --prompt-variant answer_slot_labels \
  --support-serialization triad_curated6 \
  --resume "${RESUME_STAGE_B_ROOT}" \
  --output_root "${RUN_ROOT}/triad6/phase2-selected" \
  --seed "$((BASE_SEED + 100))" \
  --train-steps 96 \
  --warmup-steps 32

./scripts/run_analysis.sh \
  --config configs/exp/m4_fever_qwen25_dynamics_recovery_deep.yaml \
  --seed "$((BASE_SEED + 900))" \
  --output_dir "${RESULT_ROOT}/dynamics-recovery" \
  --input_root "${RUN_ROOT}"

SELECTION_JSON="${RESULT_ROOT}/dynamics-recovery/selection.json"
SELECTION_PASSED="$(python -c 'import json,sys; print("1" if json.loads(open(sys.argv[1]).read()).get("selection_passed") else "0")' "${SELECTION_JSON}")"
SCREEN248_COMPARE_METRICS=""
FIXED64_COMPARE_METRICS=""

if [[ "${SELECTION_PASSED}" == "1" ]]; then
  python scripts/run_m4_gate_from_selection.py \
    --config configs/exp/m4_fever_qwen25_screen248_test_gate_deep_common.yaml \
    --selection_json "${SELECTION_JSON}" \
    --resume "${RESUME_STAGE_B_ROOT}" \
    --output_root "${RUN_ROOT}/screen248-test-gate" \
    --seed "$((BASE_SEED + 1200))" \
    --gate_name screen248_test

  ./scripts/run_analysis.sh \
    --config configs/exp/m4_fever_qwen25_screen248_test_compare.yaml \
    --seed "$((BASE_SEED + 1224))" \
    --output_dir "${RESULT_ROOT}/screen248-test-gate" \
    --input_root "${RUN_ROOT}/screen248-test-gate"
  SCREEN248_COMPARE_METRICS="${RESULT_ROOT}/screen248-test-gate/metrics.json"

  python scripts/run_m4_gate_from_selection.py \
    --config configs/exp/m4_fever_qwen25_fixed64_gate_deep_common.yaml \
    --selection_json "${SELECTION_JSON}" \
    --resume "${RESUME_STAGE_B_ROOT}" \
    --output_root "${RUN_ROOT}/fixed64-gate" \
    --seed "$((BASE_SEED + 1300))" \
    --gate_name fixed64

  ./scripts/run_analysis.sh \
    --config configs/exp/m4_fever_qwen25_fixed64_compare_deep.yaml \
    --seed "$((BASE_SEED + 1324))" \
    --output_dir "${RESULT_ROOT}/fixed64-gate" \
    --input_root "${RUN_ROOT}/fixed64-gate"
  FIXED64_COMPARE_METRICS="${RESULT_ROOT}/fixed64-gate/metrics.json"
fi

DUAL_GATE_ARGS=(
  --selection_json "${SELECTION_JSON}"
  --output_json "${RESULT_ROOT}/dynamics-recovery/dual_gate_summary.json"
  --overwrite-selection
)
if [[ -n "${SCREEN248_COMPARE_METRICS}" ]]; then
  DUAL_GATE_ARGS+=(--screen248_test_metrics "${SCREEN248_COMPARE_METRICS}")
fi
if [[ -n "${FIXED64_COMPARE_METRICS}" ]]; then
  DUAL_GATE_ARGS+=(--fixed64_metrics "${FIXED64_COMPARE_METRICS}")
fi
python scripts/update_m4_dual_gate_summary.py "${DUAL_GATE_ARGS[@]}"

mkdir -p runs/review results/generated/review
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RUN_ROOT}/" "runs/review/m4-fever-deep-prompt-recovery-qwen25/"
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RESULT_ROOT}/" "results/generated/review/m4-fever-deep-prompt-recovery-qwen25/"

./scripts/publish_review_artifacts.sh

echo "m4 FEVER deep-prompt recovery complete (selection_passed=${SELECTION_PASSED})"
