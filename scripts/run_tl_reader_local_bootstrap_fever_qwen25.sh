#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-23291}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/tl-reader-local-bootstrap-fever-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/tl-reader-local-bootstrap-fever-qwen25}"
PHASE0_METRICS="${4:-results/generated/review/m4-fever-shared-injection-qwen25/phase0-gate-sweep/metrics.json}"
RESUME_STAGE_B_ROOT="${5:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
WARM_START_INPUT="${6:-/root/autodl-tmp/runs/verify/tl-reader-symmetry-break-fever-qwen25/rg2-partition-none-linear-h4-k4/phase2-selected/pilot-I-real/checkpoint.pt}"
CONTROL_SUMMARY_JSON="${7:-results/generated/review/tl-reader-symmetry-break-fever-qwen25/rg2-partition-none-linear-h4-k4/run-summary.json}"
CONTROL_TRAIN_EVENTS_JSON="${8:-runs/review/tl-reader-symmetry-break-fever-qwen25/rg2-partition-none-linear-h4-k4/phase2-selected/pilot-I-real/train_events.json}"

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
CONTROL_SUMMARY="$(resolve_path "${CONTROL_SUMMARY_JSON}")"
CONTROL_TRAIN_EVENTS="$(resolve_path "${CONTROL_TRAIN_EVENTS_JSON}")"

if [[ ! -f "${WARM_START}" ]]; then
  echo "missing required input: ${WARM_START}" >&2
  exit 1
fi
if [[ ! -f "${CONTROL_SUMMARY}" ]]; then
  echo "missing required control summary: ${CONTROL_SUMMARY}" >&2
  exit 1
fi
if [[ ! -f "${CONTROL_TRAIN_EVENTS}" ]]; then
  echo "missing required control train events: ${CONTROL_TRAIN_EVENTS}" >&2
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
payload = {
    "rg3-bootstrap-partition-none-linear-h4-k4": str(warm_start),
    "rg3-bootstrap-reconstruction-partition-none-linear-h4-k4": str(warm_start),
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
  "configs/exp/tl_reader_local_bootstrap_fever_qwen25_bootstrap_partition_none_linear_h4_k4.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_local_bootstrap_fever_qwen25_bootstrap_partition_none_linear_h4_k4.json"
materialize_config \
  "configs/exp/tl_reader_local_bootstrap_fever_qwen25_bootstrap_reconstruction_partition_none_linear_h4_k4.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_local_bootstrap_fever_qwen25_bootstrap_reconstruction_partition_none_linear_h4_k4.json"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

ensure_suite_complete() {
  local suite_config="$1"
  local run_seed_base="$2"
  local run_dir="$3"
  local suite_root="${run_dir}/phase2-selected"
  local lock_fd
  mkdir -p "${suite_root}"
  exec {lock_fd}> "${run_dir}/.suite.lock"
  flock "${lock_fd}"
  if [[ ! -f "${suite_root}/suite_metrics.json" ]]; then
    python scripts/run_m4_selected_shared_injection_suite.py \
      --config "${suite_config}" \
      --phase0_metrics "${PHASE0_METRICS}" \
      --prompt-variant answer_slot_labels \
      --support-serialization triad_curated6 \
      --resume "${RESUME_STAGE_B_ROOT}" \
      --output_root "${suite_root}" \
      --seed "${run_seed_base}"
  fi
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

ensure_analysis_complete() {
  local dynamics_config="$1"
  local run_seed_base="$2"
  local run_dir="$3"
  local result_dir="$4"
  local recovery_root="${result_dir}/dynamics-recovery"
  local lock_fd
  mkdir -p "${recovery_root}"
  exec {lock_fd}> "${result_dir}/.analysis.lock"
  flock "${lock_fd}"
  if [[ ! -f "${recovery_root}/selection.json" ]]; then
    python -m analysis \
      --config "${dynamics_config}" \
      --seed "$((run_seed_base + 900))" \
      --output_dir "${recovery_root}" \
      --input_root "${run_dir}"
  fi
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

run_arm() {
  local run_name="$1"
  local suite_config="$2"
  local dynamics_config="$3"
  local run_seed_base="$4"
  local run_dir="${RUN_ROOT}/${run_name}"
  local result_dir="${RESULT_ROOT}/${run_name}"
  mkdir -p "${run_dir}" "${result_dir}"

  ensure_suite_complete "${suite_config}" "${run_seed_base}" "${run_dir}"
  ensure_analysis_complete "${dynamics_config}" "${run_seed_base}" "${run_dir}" "${result_dir}"

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
  "rg3-bootstrap-partition-none-linear-h4-k4" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_local_bootstrap_fever_qwen25_bootstrap_partition_none_linear_h4_k4.json" \
  "configs/exp/tl_reader_local_bootstrap_fever_qwen25_dynamics_recovery_bootstrap_partition_none_linear_h4_k4.yaml" \
  "${BASE_SEED}"
run_arm \
  "rg3-bootstrap-reconstruction-partition-none-linear-h4-k4" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_reader_local_bootstrap_fever_qwen25_bootstrap_reconstruction_partition_none_linear_h4_k4.json" \
  "configs/exp/tl_reader_local_bootstrap_fever_qwen25_dynamics_recovery_bootstrap_reconstruction_partition_none_linear_h4_k4.yaml" \
  "$((BASE_SEED + 1000))"

python scripts/update_tl_reader_rg3_summary.py \
  --control_summary_json "${CONTROL_SUMMARY}" \
  --bootstrap_summary_json "${RESULT_ROOT}/rg3-bootstrap-partition-none-linear-h4-k4/run-summary.json" \
  --bootstrap_reconstruction_summary_json "${RESULT_ROOT}/rg3-bootstrap-reconstruction-partition-none-linear-h4-k4/run-summary.json" \
  --control_train_events_json "${CONTROL_TRAIN_EVENTS}" \
  --bootstrap_train_events_json "${RUN_ROOT}/rg3-bootstrap-partition-none-linear-h4-k4/phase2-selected/pilot-I-real/train_events.json" \
  --bootstrap_reconstruction_train_events_json "${RUN_ROOT}/rg3-bootstrap-reconstruction-partition-none-linear-h4-k4/phase2-selected/pilot-I-real/train_events.json" \
  --output_json "${RESULT_ROOT}/rg3-summary.json" \
  --output_report "${RESULT_ROOT}/rg3-summary.md"

./scripts/publish_review_artifacts.sh

echo "TL reader RG-3 complete"
