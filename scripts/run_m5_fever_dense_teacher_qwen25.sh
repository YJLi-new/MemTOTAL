#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-14291}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/m5-fever-dense-teacher-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/m5-fever-dense-teacher-qwen25}"
PHASE0_METRICS="${4:-results/generated/review/m4-fever-shared-injection-qwen25/phase0-gate-sweep/metrics.json}"
RESUME_STAGE_B_ROOT="${5:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
WARM_START="${6:-/root/autodl-tmp/runs/verify/m5-fever-writer-reasoner-alignment-qwen25/canonical/phase2-selected/pilot-I-real/snapshot_evals/step_0000/checkpoint.pt}"
RUN_HINGE_OFF_AUDIT="${7:-0}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

if [[ ! -f "${WARM_START}" ]]; then
  echo "missing warm-start checkpoint: ${WARM_START}" >&2
  exit 1
fi

python - "${RESULT_ROOT}/warm_start_manifest.json" "${WARM_START}" "${RUN_HINGE_OFF_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
warm_start = Path(sys.argv[2]).resolve()
run_hinge_off_audit = str(sys.argv[3]).strip() not in {"", "0", "false", "False"}
payload = {
    "canonical-dense-teacher": str(warm_start),
    "control-safe-hinge": str(warm_start),
}
if run_hinge_off_audit:
    payload["control-no-hinge-audit"] = str(warm_start)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

./scripts/run_analysis.sh \
  --config configs/exp/m4_fever_qwen25_split_prep.yaml \
  --seed "${BASE_SEED}" \
  --output_dir "${RESULT_ROOT}/split-prep"

./scripts/run_analysis.sh \
  --config configs/exp/m4_fever_qwen25_support_bank_prep.yaml \
  --seed "$((BASE_SEED + 10))" \
  --output_dir "${RESULT_ROOT}/support-bank-prep"

run_suite() {
  local run_name="$1"
  local suite_config="$2"
  local run_seed_base="$3"
  local run_root="${RUN_ROOT}/${run_name}"
  local result_root="${RESULT_ROOT}/${run_name}"
  local selection_json=""
  local screen248_metrics=""
  local heldout_a_metrics=""
  local heldout_b_metrics=""
  local fixed64_metrics=""
  mkdir -p "${run_root}" "${result_root}"

  python scripts/run_m4_selected_shared_injection_suite.py \
    --config "${suite_config}" \
    --phase0_metrics "${PHASE0_METRICS}" \
    --prompt-variant answer_slot_labels \
    --support-serialization triad_curated6 \
    --resume "${RESUME_STAGE_B_ROOT}" \
    --output_root "${run_root}/phase2-selected" \
    --seed "${run_seed_base}"

  ./scripts/run_analysis.sh \
    --config configs/exp/m5_fever_qwen25_dynamics_recovery_structured.yaml \
    --seed "$((run_seed_base + 900))" \
    --output_dir "${result_root}/dynamics-recovery" \
    --input_root "${run_root}"

  selection_json="${result_root}/dynamics-recovery/selection.json"
  local selection_passed
  selection_passed="$(python -c 'import json,sys; print("1" if json.loads(open(sys.argv[1]).read()).get("selection_passed") else "0")' "${selection_json}")"
  if [[ "${selection_passed}" == "1" ]]; then
    python scripts/run_m4_gate_from_selection.py \
      --config configs/exp/m5_fever_qwen25_screen248_test_gate_structured_common.yaml \
      --selection_json "${selection_json}" \
      --resume "${RESUME_STAGE_B_ROOT}" \
      --output_root "${run_root}/screen248-test-canonical" \
      --seed "$((run_seed_base + 1200))" \
      --gate_name screen248_test_canonical

    ./scripts/run_analysis.sh \
      --config configs/exp/m5_fever_qwen25_compare_alignment.yaml \
      --seed "$((run_seed_base + 1224))" \
      --output_dir "${result_root}/screen248-test-canonical" \
      --input_root "${run_root}/screen248-test-canonical"
    screen248_metrics="${result_root}/screen248-test-canonical/metrics.json"

    local canonical_passed
    canonical_passed="$(python -c 'import json,sys; print("1" if json.loads(open(sys.argv[1]).read()).get("gate_passed") else "0")' "${screen248_metrics}")"
    if [[ "${canonical_passed}" == "1" ]]; then
      python scripts/run_m4_gate_from_selection.py \
        --config configs/exp/m5_fever_qwen25_screen248_test_heldout_a_gate_structured_common.yaml \
        --selection_json "${selection_json}" \
        --resume "${RESUME_STAGE_B_ROOT}" \
        --output_root "${run_root}/screen248-test-heldout-a" \
        --seed "$((run_seed_base + 1300))" \
        --gate_name screen248_test_heldout_a

      ./scripts/run_analysis.sh \
        --config configs/exp/m5_fever_qwen25_compare_alignment.yaml \
        --seed "$((run_seed_base + 1324))" \
        --output_dir "${result_root}/screen248-test-heldout-a" \
        --input_root "${run_root}/screen248-test-heldout-a"
      heldout_a_metrics="${result_root}/screen248-test-heldout-a/metrics.json"

      python scripts/run_m4_gate_from_selection.py \
        --config configs/exp/m5_fever_qwen25_screen248_test_heldout_b_gate_structured_common.yaml \
        --selection_json "${selection_json}" \
        --resume "${RESUME_STAGE_B_ROOT}" \
        --output_root "${run_root}/screen248-test-heldout-b" \
        --seed "$((run_seed_base + 1400))" \
        --gate_name screen248_test_heldout_b

      ./scripts/run_analysis.sh \
        --config configs/exp/m5_fever_qwen25_compare_alignment.yaml \
        --seed "$((run_seed_base + 1424))" \
        --output_dir "${result_root}/screen248-test-heldout-b" \
        --input_root "${run_root}/screen248-test-heldout-b"
      heldout_b_metrics="${result_root}/screen248-test-heldout-b/metrics.json"
    fi
  fi

  local summary_args=(
    --selection_json "${selection_json}"
    --run_metrics_json "${run_root}/phase2-selected/pilot-I-real/metrics.json"
    --dynamics_summary_csv "${result_root}/dynamics-recovery/dynamics_recovery_summary.csv"
    --prefix_norm_csv "${result_root}/dynamics-recovery/prefix_norm_drift.csv"
    --output_json "${result_root}/run-summary.json"
    --output_report "${result_root}/run-summary.md"
    --overwrite-selection
  )
  if [[ -n "${screen248_metrics}" ]]; then
    summary_args+=(--screen248_test_metrics "${screen248_metrics}")
  fi
  if [[ -n "${heldout_a_metrics}" ]]; then
    summary_args+=(--heldout_a_metrics "${heldout_a_metrics}")
  fi
  if [[ -n "${heldout_b_metrics}" ]]; then
    summary_args+=(--heldout_b_metrics "${heldout_b_metrics}")
  fi
  python scripts/update_m4_run_summary.py "${summary_args[@]}"

  local promote_to_fixed64
  promote_to_fixed64="$(python -c 'import json,sys; data=json.loads(open(sys.argv[1]).read()); print("1" if data.get("screen248_test_gate_passed") and not data.get("support_bank_brittle") else "0")' "${result_root}/run-summary.json")"
  if [[ "${promote_to_fixed64}" == "1" ]]; then
    python scripts/run_m4_gate_from_selection.py \
      --config configs/exp/m5_fever_qwen25_fixed64_gate_structured_legacy_common.yaml \
      --selection_json "${selection_json}" \
      --resume "${RESUME_STAGE_B_ROOT}" \
      --output_root "${run_root}/fixed64-legacy" \
      --seed "$((run_seed_base + 1500))" \
      --gate_name fixed64_legacy

    ./scripts/run_analysis.sh \
      --config configs/exp/m5_fever_qwen25_compare_alignment.yaml \
      --seed "$((run_seed_base + 1524))" \
      --output_dir "${result_root}/fixed64-legacy" \
      --input_root "${run_root}/fixed64-legacy"
    fixed64_metrics="${result_root}/fixed64-legacy/metrics.json"

    python scripts/update_m4_run_summary.py \
      --selection_json "${selection_json}" \
      --run_metrics_json "${run_root}/phase2-selected/pilot-I-real/metrics.json" \
      --dynamics_summary_csv "${result_root}/dynamics-recovery/dynamics_recovery_summary.csv" \
      --prefix_norm_csv "${result_root}/dynamics-recovery/prefix_norm_drift.csv" \
      --screen248_test_metrics "${screen248_metrics}" \
      --heldout_a_metrics "${heldout_a_metrics}" \
      --heldout_b_metrics "${heldout_b_metrics}" \
      --fixed64_metrics "${fixed64_metrics}" \
      --output_json "${result_root}/run-summary.json" \
      --output_report "${result_root}/run-summary.md" \
      --overwrite-selection
  fi
}

run_suite \
  "control-safe-hinge" \
  "configs/exp/m5_fever_qwen25_phase2_val_objective_dense_teacher_control_common.yaml" \
  "$((BASE_SEED + 100))"

run_suite \
  "canonical-dense-teacher" \
  "configs/exp/m5_fever_qwen25_phase2_val_objective_dense_teacher_canonical_common.yaml" \
  "$((BASE_SEED + 400))"

if [[ "${RUN_HINGE_OFF_AUDIT}" != "0" ]]; then
  run_suite \
    "control-no-hinge-audit" \
    "configs/exp/m5_fever_qwen25_phase2_val_objective_dense_teacher_hinge_off_audit_common.yaml" \
    "$((BASE_SEED + 700))"
fi

summary_args=(
  --canonical_summary_json "${RESULT_ROOT}/canonical-dense-teacher/run-summary.json"
  --control_summary_json "${RESULT_ROOT}/control-safe-hinge/run-summary.json"
  --output_json "${RESULT_ROOT}/dense-teacher-summary.json"
  --output_report "${RESULT_ROOT}/dense-teacher-summary.md"
)
if [[ "${RUN_HINGE_OFF_AUDIT}" != "0" ]]; then
  summary_args+=(--hinge_off_audit_summary_json "${RESULT_ROOT}/control-no-hinge-audit/run-summary.json")
fi
python scripts/update_m5_dense_teacher_summary.py "${summary_args[@]}"

mkdir -p runs/review results/generated/review
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RUN_ROOT}/" "runs/review/m5-fever-dense-teacher-qwen25/"
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RESULT_ROOT}/" "results/generated/review/m5-fever-dense-teacher-qwen25/"

./scripts/publish_review_artifacts.sh

echo "m5.3 FEVER dense teacher complete"
