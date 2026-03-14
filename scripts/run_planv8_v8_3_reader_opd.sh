#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PRIMARY_BACKBONE_NAME="${PLANV8_PRIMARY_BACKBONE_NAME:-Qwen3-8B}"
PRIMARY_BACKBONE_KEY="${PLANV8_PRIMARY_BACKBONE_KEY:-qwen3}"
PRIMARY_MODEL_ID="${PLANV8_PRIMARY_MODEL_ID:-Qwen/Qwen3-8B}"
PRIMARY_PREP_SCRIPT="${PLANV8_PRIMARY_PREP_SCRIPT:-scripts/prepare_local_qwen3_model.sh}"
PRIMARY_MODEL_DIR_DEFAULT="${PLANV8_PRIMARY_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-8B}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-3-reader-opd}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-3-reader-opd}"
PRIMARY_MODEL_DIR="${4:-${PRIMARY_MODEL_DIR_DEFAULT}}"
V82_RUN_ROOT="${5:-/root/autodl-tmp/runs/verify/planv8-v8-2-reader-sweep}"
V82_SUMMARY_PATH="${6:-results/generated/review/planv8-v8-2-reader-sweep/v8-2-summary.json}"
SELECTED_PROMPTS_PATH="${7:-results/generated/review/planv8-v8-2-reader-sweep/selected-prompt-modes.json}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PLANV8_EXPERIMENT_PREFIX="${PLANV8_EXPERIMENT_PREFIX:-planv8}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}" "${HF_HOME}"

CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${CONFIG_ROOT}"

python - "${V82_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step not in {"open_v8_3_reader_opd", "open_v8_3_reader_opd_last_consumer_attempt"}:
    raise SystemExit(
        f"V8-2 did not authorize V8-3; recommended_next_step={next_step!r}"
    )
PY

bash "${PRIMARY_PREP_SCRIPT}" \
  "${PRIMARY_MODEL_ID}" \
  "${PRIMARY_MODEL_DIR}" \
  "${HF_HOME}"

BASE_ARM_ID="$(
python - "${V82_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
base_arm_id = str(summary.get("base_for_v8_3_arm_id") or summary.get("best_arm_id") or "").strip()
if not base_arm_id:
    raise SystemExit("V8-2 summary did not provide base_for_v8_3_arm_id or best_arm_id.")
print(base_arm_id)
PY
)"

if [[ -f "${SELECTED_PROMPTS_PATH}" ]]; then
  cp "${SELECTED_PROMPTS_PATH}" "${RESULT_ROOT}/selected-prompt-modes.json"
fi
cp "${V82_SUMMARY_PATH}" "${RESULT_ROOT}/v8-2-summary.reference.json"

materialize_config() {
  local task_name="$1"
  local arm_id="$2"
  local output_config="$3"
  local base_config="${V82_RUN_ROOT}/materialized-configs/${task_name}-${BASE_ARM_ID}.json"
  local base_checkpoint="${V82_RUN_ROOT}/${BASE_ARM_ID}-${task_name}/checkpoint.pt"
  python scripts/planv8_v8_3_config.py \
    --base_config "${base_config}" \
    --base_checkpoint "${base_checkpoint}" \
    --arm_id "${arm_id}" \
    --output_config "${output_config}" \
    --v82_summary_path "${V82_SUMMARY_PATH}" \
    --primary_model_dir "${PRIMARY_MODEL_DIR}" \
    --primary_backbone_name "${PRIMARY_BACKBONE_NAME}"
}

run_single_pilot() {
  local suite_config="$1"
  local run_seed="$2"
  local run_dir="$3"
  mkdir -p "${run_dir}"
  local lock_fd
  exec {lock_fd}> "${run_dir}/.suite.lock"
  flock "${lock_fd}"
  if [[ ! -f "${run_dir}/metrics.json" ]]; then
    python - "${suite_config}" "${run_seed}" "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.training.m4_shared_injection import run_shared_injection_pilot

config_path = Path(sys.argv[1])
seed = int(sys.argv[2])
run_dir = Path(sys.argv[3])
config = json.loads(config_path.read_text())
run_shared_injection_pilot(
    config=config,
    seed=seed,
    output_dir=run_dir,
    resume=None,
    dry_run=False,
)
PY
  fi
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

copy_run_artifacts() {
  local dst_dir="$1"
  local run_dir="$2"
  mkdir -p "${dst_dir}"
  cp "${run_dir}/metrics.json" "${dst_dir}/metrics.json"
  if [[ -f "${run_dir}/train_events.json" ]]; then
    cp "${run_dir}/train_events.json" "${dst_dir}/train_events.json"
  else
    printf '[]\n' > "${dst_dir}/train_events.json"
  fi
  if [[ -f "${run_dir}/task_case_dump.jsonl" ]]; then
    cp "${run_dir}/task_case_dump.jsonl" "${dst_dir}/task_case_dump.jsonl"
  fi
}

for arm_id in \
  p0_ce_only \
  p1_teacher_choice_kl \
  p2_opd_ansonly_w01 \
  p3_opd_ansonly_w03 \
  p4_opd_ansplusctx_w03 \
  p5_opd_ansplusctx_centered
do
  for task_name in gsm8k triviaqa fever; do
    materialize_config \
      "${task_name}" \
      "${arm_id}" \
      "${CONFIG_ROOT}/${task_name}-${arm_id}.json"
  done
done

seed_offset=0
for arm_id in \
  p0_ce_only \
  p1_teacher_choice_kl \
  p2_opd_ansonly_w01 \
  p3_opd_ansonly_w03 \
  p4_opd_ansplusctx_w03 \
  p5_opd_ansplusctx_centered
do
  for task_name in gsm8k triviaqa fever; do
    run_single_pilot \
      "${CONFIG_ROOT}/${task_name}-${arm_id}.json" \
      "$((BASE_SEED + seed_offset))" \
      "${RUN_ROOT}/${arm_id}-${task_name}"
    copy_run_artifacts \
      "${RESULT_ROOT}/${arm_id}/${task_name}" \
      "${RUN_ROOT}/${arm_id}-${task_name}"
    seed_offset=$((seed_offset + 10))
  done
done

python scripts/update_planv8_v8_3_summary.py \
  --result_root "${RESULT_ROOT}" \
  --v82_summary "${V82_SUMMARY_PATH}" \
  --output_json "${RESULT_ROOT}/v8-3-summary.json" \
  --output_report "${RESULT_ROOT}/v8-3-summary.md"

bash scripts/publish_review_artifacts.sh

echo "PLANv8 V8-3 reader OPD complete."
