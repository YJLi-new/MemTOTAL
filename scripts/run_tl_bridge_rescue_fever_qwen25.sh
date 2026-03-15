#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-18291}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/tl-bridge-rescue-fever-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/tl-bridge-rescue-fever-qwen25}"
PHASE0_METRICS="${4:-results/generated/review/m4-fever-shared-injection-qwen25/phase0-gate-sweep/metrics.json}"
RESUME_STAGE_B_ROOT="${5:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
WARM_START_INPUT="${6:-/root/autodl-tmp/runs/verify/tl-poc-fever-qwen25/sl8/phase2-selected/pilot-I-real/snapshot_evals/step_0002/checkpoint.pt}"
BASELINE_SL8_SUMMARY="${7:-results/generated/review/tl-poc-fever-qwen25/sl8/run-summary.json}"
BASELINE_TL_SUMMARY="${8:-results/generated/review/tl-poc-fever-qwen25/tl-h4-k8/run-summary.json}"
BASELINE_TL_READER_CSV="${9:-results/generated/review/tl-poc-fever-qwen25/tl-h4-k8/dynamics-recovery/reader_query_diagnostics.csv}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

resolve_path() {
  python - "$1" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_absolute():
    path = (Path.cwd() / path).resolve()
print(path)
PY
}

WARM_START="$(resolve_path "${WARM_START_INPUT}")"
BASELINE_SL8_SUMMARY="$(resolve_path "${BASELINE_SL8_SUMMARY}")"
BASELINE_TL_SUMMARY="$(resolve_path "${BASELINE_TL_SUMMARY}")"
BASELINE_TL_READER_CSV="$(resolve_path "${BASELINE_TL_READER_CSV}")"

if [[ ! -f "${WARM_START}" ]]; then
  echo "missing warm-start checkpoint: ${WARM_START}" >&2
  exit 1
fi

MATERIALIZED_CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${MATERIALIZED_CONFIG_ROOT}"

python - "${RESULT_ROOT}/warm_start_manifest.json" "${WARM_START}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
warm_start = Path(sys.argv[2]).resolve()
output_path.write_text(json.dumps({"tl-h4-k8-rescue": str(warm_start)}, indent=2, sort_keys=True) + "\n")
PY

python - "configs/exp/tl_bridge_rescue_fever_qwen25_h4_k8.yaml" "${MATERIALIZED_CONFIG_ROOT}/tl_bridge_rescue_fever_qwen25_h4_k8.json" "${WARM_START}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

source_config = sys.argv[1]
output_config = Path(sys.argv[2])
warm_start = str(Path(sys.argv[3]).resolve())
config = load_config(source_config)
config.setdefault("runtime", {})
config["runtime"]["pilot_init_checkpoint_path"] = warm_start
output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

RUN_NAME="tl-h4-k8-rescue"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
RESULT_DIR="${RESULT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RESULT_DIR}"

if [[ ! -f "${RUN_DIR}/phase2-selected/suite_metrics.json" ]]; then
  python scripts/run_m4_selected_shared_injection_suite.py \
    --config "${MATERIALIZED_CONFIG_ROOT}/tl_bridge_rescue_fever_qwen25_h4_k8.json" \
    --phase0_metrics "${PHASE0_METRICS}" \
    --prompt-variant answer_slot_labels \
    --support-serialization triad_curated6 \
    --resume "${RESUME_STAGE_B_ROOT}" \
    --output_root "${RUN_DIR}/phase2-selected" \
    --seed "${BASE_SEED}"
fi

if [[ ! -f "${RESULT_DIR}/dynamics-recovery/selection.json" ]]; then
  ./scripts/run_analysis.sh \
    --config configs/exp/tl_bridge_rescue_fever_qwen25_dynamics_recovery_h4_k8.yaml \
    --seed "$((BASE_SEED + 900))" \
    --output_dir "${RESULT_DIR}/dynamics-recovery" \
    --input_root "${RUN_DIR}"
fi

SELECTION_JSON="${RESULT_DIR}/dynamics-recovery/selection.json"
SCREEN248_METRICS=""
HELDOUT_A_METRICS=""
HELDOUT_B_METRICS=""
FIXED64_METRICS=""

SELECTION_PASSED="$(python -c 'import json,sys; print("1" if json.loads(open(sys.argv[1]).read()).get("selection_passed") else "0")' "${SELECTION_JSON}")"
if [[ "${SELECTION_PASSED}" == "1" ]]; then
  if [[ ! -f "${RUN_DIR}/screen248-test-canonical/pilot-I-zero-gate/metrics.json" ]]; then
    python scripts/run_m4_gate_from_selection.py \
      --config "configs/exp/tl_bridge_rescue_fever_qwen25_screen248_test_gate_h4_k8.yaml" \
      --selection_json "${SELECTION_JSON}" \
      --resume "${RESUME_STAGE_B_ROOT}" \
      --output_root "${RUN_DIR}/screen248-test-canonical" \
      --seed "$((BASE_SEED + 1200))" \
      --gate_name screen248_test_canonical
  fi

  if [[ ! -f "${RESULT_DIR}/screen248-test-canonical/metrics.json" ]]; then
    ./scripts/run_analysis.sh \
      --config configs/exp/m5_fever_qwen25_compare_alignment.yaml \
      --seed "$((BASE_SEED + 1224))" \
      --output_dir "${RESULT_DIR}/screen248-test-canonical" \
      --input_root "${RUN_DIR}/screen248-test-canonical"
  fi
  SCREEN248_METRICS="${RESULT_DIR}/screen248-test-canonical/metrics.json"

  CANONICAL_PASSED="$(python -c 'import json,sys; print("1" if json.loads(open(sys.argv[1]).read()).get("gate_passed") else "0")' "${SCREEN248_METRICS}")"
  if [[ "${CANONICAL_PASSED}" == "1" ]]; then
    if [[ ! -f "${RUN_DIR}/screen248-test-heldout-a/pilot-I-zero-gate/metrics.json" ]]; then
      python scripts/run_m4_gate_from_selection.py \
        --config "configs/exp/tl_bridge_rescue_fever_qwen25_screen248_test_heldout_a_gate_h4_k8.yaml" \
        --selection_json "${SELECTION_JSON}" \
        --resume "${RESUME_STAGE_B_ROOT}" \
        --output_root "${RUN_DIR}/screen248-test-heldout-a" \
        --seed "$((BASE_SEED + 1300))" \
        --gate_name screen248_test_heldout_a
    fi

    if [[ ! -f "${RESULT_DIR}/screen248-test-heldout-a/metrics.json" ]]; then
      ./scripts/run_analysis.sh \
        --config configs/exp/m5_fever_qwen25_compare_alignment.yaml \
        --seed "$((BASE_SEED + 1324))" \
        --output_dir "${RESULT_DIR}/screen248-test-heldout-a" \
        --input_root "${RUN_DIR}/screen248-test-heldout-a"
    fi
    HELDOUT_A_METRICS="${RESULT_DIR}/screen248-test-heldout-a/metrics.json"

    if [[ ! -f "${RUN_DIR}/screen248-test-heldout-b/pilot-I-zero-gate/metrics.json" ]]; then
      python scripts/run_m4_gate_from_selection.py \
        --config "configs/exp/tl_bridge_rescue_fever_qwen25_screen248_test_heldout_b_gate_h4_k8.yaml" \
        --selection_json "${SELECTION_JSON}" \
        --resume "${RESUME_STAGE_B_ROOT}" \
        --output_root "${RUN_DIR}/screen248-test-heldout-b" \
        --seed "$((BASE_SEED + 1400))" \
        --gate_name screen248_test_heldout_b
    fi

    if [[ ! -f "${RESULT_DIR}/screen248-test-heldout-b/metrics.json" ]]; then
      ./scripts/run_analysis.sh \
        --config configs/exp/m5_fever_qwen25_compare_alignment.yaml \
        --seed "$((BASE_SEED + 1424))" \
        --output_dir "${RESULT_DIR}/screen248-test-heldout-b" \
        --input_root "${RUN_DIR}/screen248-test-heldout-b"
    fi
    HELDOUT_B_METRICS="${RESULT_DIR}/screen248-test-heldout-b/metrics.json"
  fi
fi

SUMMARY_ARGS=(
  --selection_json "${SELECTION_JSON}"
  --run_metrics_json "${RUN_DIR}/phase2-selected/pilot-I-real/metrics.json"
  --dynamics_summary_csv "${RESULT_DIR}/dynamics-recovery/dynamics_recovery_summary.csv"
  --prefix_norm_csv "${RESULT_DIR}/dynamics-recovery/prefix_norm_drift.csv"
  --output_json "${RESULT_DIR}/run-summary.json"
  --output_report "${RESULT_DIR}/run-summary.md"
  --overwrite-selection
)
if [[ -n "${SCREEN248_METRICS}" ]]; then
  SUMMARY_ARGS+=(--screen248_test_metrics "${SCREEN248_METRICS}")
fi
if [[ -n "${HELDOUT_A_METRICS}" ]]; then
  SUMMARY_ARGS+=(--heldout_a_metrics "${HELDOUT_A_METRICS}")
fi
if [[ -n "${HELDOUT_B_METRICS}" ]]; then
  SUMMARY_ARGS+=(--heldout_b_metrics "${HELDOUT_B_METRICS}")
fi
python scripts/update_m4_run_summary.py "${SUMMARY_ARGS[@]}"

PROMOTE_TO_FIXED64="$(python -c 'import json,sys; data=json.loads(open(sys.argv[1]).read()); print("1" if data.get("screen248_test_gate_passed") and not data.get("support_bank_brittle") else "0")' "${RESULT_DIR}/run-summary.json")"
if [[ "${PROMOTE_TO_FIXED64}" == "1" ]]; then
  if [[ ! -f "${RUN_DIR}/fixed64-legacy/pilot-I-zero-gate/metrics.json" ]]; then
    python scripts/run_m4_gate_from_selection.py \
      --config "configs/exp/tl_bridge_rescue_fever_qwen25_fixed64_gate_h4_k8.yaml" \
      --selection_json "${SELECTION_JSON}" \
      --resume "${RESUME_STAGE_B_ROOT}" \
      --output_root "${RUN_DIR}/fixed64-legacy" \
      --seed "$((BASE_SEED + 1500))" \
      --gate_name fixed64_legacy
  fi

  if [[ ! -f "${RESULT_DIR}/fixed64-legacy/metrics.json" ]]; then
    ./scripts/run_analysis.sh \
      --config configs/exp/m5_fever_qwen25_compare_alignment.yaml \
      --seed "$((BASE_SEED + 1524))" \
      --output_dir "${RESULT_DIR}/fixed64-legacy" \
      --input_root "${RUN_DIR}/fixed64-legacy"
  fi
  FIXED64_METRICS="${RESULT_DIR}/fixed64-legacy/metrics.json"

  python scripts/update_m4_run_summary.py \
    --selection_json "${SELECTION_JSON}" \
    --run_metrics_json "${RUN_DIR}/phase2-selected/pilot-I-real/metrics.json" \
    --dynamics_summary_csv "${RESULT_DIR}/dynamics-recovery/dynamics_recovery_summary.csv" \
    --prefix_norm_csv "${RESULT_DIR}/dynamics-recovery/prefix_norm_drift.csv" \
    --screen248_test_metrics "${SCREEN248_METRICS}" \
    --heldout_a_metrics "${HELDOUT_A_METRICS}" \
    --heldout_b_metrics "${HELDOUT_B_METRICS}" \
    --fixed64_metrics "${FIXED64_METRICS}" \
    --output_json "${RESULT_DIR}/run-summary.json" \
    --output_report "${RESULT_DIR}/run-summary.md" \
    --overwrite-selection
fi

python scripts/update_tl_bridge_rescue_summary.py \
  --sl8_summary_json "${BASELINE_SL8_SUMMARY}" \
  --tl_h4_k8_summary_json "${BASELINE_TL_SUMMARY}" \
  --tl_h4_k8_rescue_summary_json "${RESULT_DIR}/run-summary.json" \
  --tl_h4_k8_reader_query_csv "${BASELINE_TL_READER_CSV}" \
  --tl_h4_k8_rescue_reader_query_csv "${RESULT_DIR}/dynamics-recovery/reader_query_diagnostics.csv" \
  --output_json "${RESULT_ROOT}/bridge-rescue-summary.json" \
  --output_report "${RESULT_ROOT}/bridge-rescue-summary.md"

./scripts/publish_review_artifacts.sh

echo "TL bridge rescue complete"
