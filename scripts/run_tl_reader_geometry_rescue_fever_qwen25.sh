#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-19291}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/tl-reader-geometry-rescue-fever-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/tl-reader-geometry-rescue-fever-qwen25}"
PHASE0_METRICS="${4:-results/generated/review/m4-fever-shared-injection-qwen25/phase0-gate-sweep/metrics.json}"
RESUME_STAGE_B_ROOT="${5:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
WARM_START_INPUT="${6:-/root/autodl-tmp/runs/verify/tl-poc-fever-qwen25/sl8/phase2-selected/pilot-I-real/snapshot_evals/step_0002/checkpoint.pt}"
BASELINE_SUMMARY="${7:-/root/autodl-tmp/results/generated/tl-slot-basis-rescue-fever-qwen25/tl-h4-k8-slot-basis-rescue/run-summary.json}"
BASELINE_READER_CSV="${8:-/root/autodl-tmp/results/generated/tl-slot-basis-rescue-fever-qwen25/tl-h4-k8-slot-basis-rescue/dynamics-recovery/reader_query_diagnostics.csv}"
BASELINE_TRAIN_EVENTS="${9:-/root/autodl-tmp/runs/verify/tl-slot-basis-rescue-fever-qwen25/tl-h4-k8-slot-basis-rescue/phase2-selected/pilot-I-real/train_events.json}"

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
BASELINE_SUMMARY="$(resolve_path "${BASELINE_SUMMARY}")"
BASELINE_READER_CSV="$(resolve_path "${BASELINE_READER_CSV}")"
BASELINE_TRAIN_EVENTS="$(resolve_path "${BASELINE_TRAIN_EVENTS}")"

for required_file in "${WARM_START}" "${BASELINE_SUMMARY}" "${BASELINE_READER_CSV}" "${BASELINE_TRAIN_EVENTS}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing required input: ${required_file}" >&2
    exit 1
  fi
done

MATERIALIZED_CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${MATERIALIZED_CONFIG_ROOT}"

python - "${RESULT_ROOT}/warm_start_manifest.json" "${WARM_START}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
warm_start = Path(sys.argv[2]).resolve()
payload = {
    "rg1a-ctxoff-h4-k8": str(warm_start),
    "rg1b-ctxoff-h4-k4": str(warm_start),
    "rg1c-ctxoff-h4-k4-linear": str(warm_start),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

materialize_config() {
  local src_config="$1"
  local output_config="$2"
  python - "${src_config}" "${output_config}" "${WARM_START}" <<'PY'
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
}

materialize_config \
  "configs/exp/tl_reader_geometry_rescue_fever_qwen25_ctxnone_h4_k8.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_geometry_rescue_fever_qwen25_ctxnone_h4_k8.json"
materialize_config \
  "configs/exp/tl_reader_geometry_rescue_fever_qwen25_ctxnone_h4_k4.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_geometry_rescue_fever_qwen25_ctxnone_h4_k4.json"
materialize_config \
  "configs/exp/tl_reader_geometry_rescue_fever_qwen25_ctxnone_linear_h4_k4.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_geometry_rescue_fever_qwen25_ctxnone_linear_h4_k4.json"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

run_arm() {
  local run_name="$1"
  local suite_config="$2"
  local dynamics_config="$3"
  local run_seed_base="$4"
  local run_dir="${RUN_ROOT}/${run_name}"
  local result_dir="${RESULT_ROOT}/${run_name}"
  mkdir -p "${run_dir}" "${result_dir}"

  if [[ ! -f "${run_dir}/phase2-selected/suite_metrics.json" ]]; then
    python scripts/run_m4_selected_shared_injection_suite.py \
      --config "${suite_config}" \
      --phase0_metrics "${PHASE0_METRICS}" \
      --prompt-variant answer_slot_labels \
      --support-serialization triad_curated6 \
      --resume "${RESUME_STAGE_B_ROOT}" \
      --output_root "${run_dir}/phase2-selected" \
      --seed "${run_seed_base}"
  fi

  if [[ ! -f "${result_dir}/dynamics-recovery/selection.json" ]]; then
    ./scripts/run_analysis.sh \
      --config "${dynamics_config}" \
      --seed "$((run_seed_base + 900))" \
      --output_dir "${result_dir}/dynamics-recovery" \
      --input_root "${run_dir}"
  fi

  python scripts/update_m4_run_summary.py \
    --selection_json "${result_dir}/dynamics-recovery/selection.json" \
    --run_metrics_json "${run_dir}/phase2-selected/pilot-I-real/metrics.json" \
    --dynamics_summary_csv "${result_dir}/dynamics-recovery/dynamics_recovery_summary.csv" \
    --prefix_norm_csv "${result_dir}/dynamics-recovery/prefix_norm_drift.csv" \
    --output_json "${result_dir}/run-summary.json" \
    --output_report "${result_dir}/run-summary.md" \
    --overwrite-selection
}

run_arm \
  "rg1a-ctxoff-h4-k8" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_geometry_rescue_fever_qwen25_ctxnone_h4_k8.json" \
  "configs/exp/tl_reader_geometry_rescue_fever_qwen25_dynamics_recovery_ctxnone_h4_k8.yaml" \
  "${BASE_SEED}"
run_arm \
  "rg1b-ctxoff-h4-k4" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_geometry_rescue_fever_qwen25_ctxnone_h4_k4.json" \
  "configs/exp/tl_reader_geometry_rescue_fever_qwen25_dynamics_recovery_ctxnone_h4_k4.yaml" \
  "$((BASE_SEED + 1000))"
run_arm \
  "rg1c-ctxoff-h4-k4-linear" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_geometry_rescue_fever_qwen25_ctxnone_linear_h4_k4.json" \
  "configs/exp/tl_reader_geometry_rescue_fever_qwen25_dynamics_recovery_ctxnone_linear_h4_k4.yaml" \
  "$((BASE_SEED + 2000))"

python scripts/update_tl_reader_geometry_summary.py \
  --baseline_summary_json "${BASELINE_SUMMARY}" \
  --rg1a_summary_json "${RESULT_ROOT}/rg1a-ctxoff-h4-k8/run-summary.json" \
  --rg1b_summary_json "${RESULT_ROOT}/rg1b-ctxoff-h4-k4/run-summary.json" \
  --rg1c_summary_json "${RESULT_ROOT}/rg1c-ctxoff-h4-k4-linear/run-summary.json" \
  --baseline_reader_query_csv "${BASELINE_READER_CSV}" \
  --rg1a_reader_query_csv "${RESULT_ROOT}/rg1a-ctxoff-h4-k8/dynamics-recovery/reader_query_diagnostics.csv" \
  --rg1b_reader_query_csv "${RESULT_ROOT}/rg1b-ctxoff-h4-k4/dynamics-recovery/reader_query_diagnostics.csv" \
  --rg1c_reader_query_csv "${RESULT_ROOT}/rg1c-ctxoff-h4-k4-linear/dynamics-recovery/reader_query_diagnostics.csv" \
  --baseline_train_events_json "${BASELINE_TRAIN_EVENTS}" \
  --rg1a_train_events_json "${RUN_ROOT}/rg1a-ctxoff-h4-k8/phase2-selected/pilot-I-real/train_events.json" \
  --rg1b_train_events_json "${RUN_ROOT}/rg1b-ctxoff-h4-k4/phase2-selected/pilot-I-real/train_events.json" \
  --rg1c_train_events_json "${RUN_ROOT}/rg1c-ctxoff-h4-k4-linear/phase2-selected/pilot-I-real/train_events.json" \
  --output_json "${RESULT_ROOT}/rg1-summary.json" \
  --output_report "${RESULT_ROOT}/rg1-summary.md"

./scripts/publish_review_artifacts.sh

echo "TL reader geometry RG-1 complete"
