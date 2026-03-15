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
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-8-multiseed-confirmation}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-8-multiseed-confirmation}"
PRIMARY_MODEL_DIR="${4:-${PRIMARY_MODEL_DIR_DEFAULT}}"
V80_SUMMARY_PATH="${5:-results/generated/review/planv8-v8-0-${PRIMARY_BACKBONE_KEY}-baselines-oracles/v8-0-summary.json}"
SELECTED_PROMPTS_PATH="${6:-results/generated/review/planv8-v8-0-${PRIMARY_BACKBONE_KEY}-baselines-oracles/selected-prompt-modes.json}"
V83_RUN_ROOT="${7:-/root/autodl-tmp/runs/verify/planv8-v8-3-reader-opd-${PRIMARY_BACKBONE_KEY}}"
V83_SUMMARY_PATH="${8:-results/generated/review/planv8-v8-3-reader-opd-${PRIMARY_BACKBONE_KEY}/v8-3-summary.json}"
V85_RUN_ROOT="${9:-/root/autodl-tmp/runs/verify/planv8-v8-5-bridge-revisit-${PRIMARY_BACKBONE_KEY}}"
V85_SUMMARY_PATH="${10:-results/generated/review/planv8-v8-5-bridge-revisit-${PRIMARY_BACKBONE_KEY}/v8-5-summary.json}"
V86_RUN_ROOT="${11:-/root/autodl-tmp/runs/verify/planv8-v8-6-writer-aux-${PRIMARY_BACKBONE_KEY}}"
V86_SUMMARY_PATH="${12:-results/generated/review/planv8-v8-6-writer-aux-${PRIMARY_BACKBONE_KEY}/v8-6-summary.json}"
V87_SUMMARY_PATH="${13:-results/generated/review/planv8-v8-7-comparators-${PRIMARY_BACKBONE_KEY}/v8-7-summary.json}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PLANV8_EXPERIMENT_PREFIX="${PLANV8_EXPERIMENT_PREFIX:-planv8}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}" "${HF_HOME}"

CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
MANIFEST_ROOT="${RUN_ROOT}/materialized-manifests"
mkdir -p "${CONFIG_ROOT}" "${MANIFEST_ROOT}"

python - "${V87_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step != "open_v8_8_multiseed_confirmation":
    raise SystemExit(
        f"V8-7 did not authorize V8-8; recommended_next_step={next_step!r}"
    )
PY

bash "${PRIMARY_PREP_SCRIPT}" \
  "${PRIMARY_MODEL_ID}" \
  "${PRIMARY_MODEL_DIR}" \
  "${HF_HOME}"

SELECTION_MANIFEST="${MANIFEST_ROOT}/selection-manifest.json"
python scripts/planv8_v8_8_selection_manifest.py \
  --v83_summary "${V83_SUMMARY_PATH}" \
  --v83_run_root "${V83_RUN_ROOT}" \
  --v87_summary "${V87_SUMMARY_PATH}" \
  --output_json "${SELECTION_MANIFEST}" \
  --v85_summary "${V85_SUMMARY_PATH}" \
  --v85_run_root "${V85_RUN_ROOT}" \
  --v86_summary "${V86_SUMMARY_PATH}" \
  --v86_run_root "${V86_RUN_ROOT}"

if [[ -f "${SELECTED_PROMPTS_PATH}" ]]; then
  cp "${SELECTED_PROMPTS_PATH}" "${RESULT_ROOT}/selected-prompt-modes.json"
fi
cp "${V80_SUMMARY_PATH}" "${RESULT_ROOT}/v8-0-summary.reference.json"
cp "${V83_SUMMARY_PATH}" "${RESULT_ROOT}/v8-3-summary.reference.json"
if [[ -f "${V85_SUMMARY_PATH}" ]]; then
  cp "${V85_SUMMARY_PATH}" "${RESULT_ROOT}/v8-5-summary.reference.json"
fi
if [[ -f "${V86_SUMMARY_PATH}" ]]; then
  cp "${V86_SUMMARY_PATH}" "${RESULT_ROOT}/v8-6-summary.reference.json"
fi
cp "${V87_SUMMARY_PATH}" "${RESULT_ROOT}/v8-7-summary.reference.json"
cp "${SELECTION_MANIFEST}" "${RESULT_ROOT}/selection-manifest.json"

materialize_config() {
  local variant_id="$1"
  local source_phase="$2"
  local source_run_root="$3"
  local source_arm_id="$4"
  local task_name="$5"
  local output_config="$6"
  local base_config="${source_run_root}/materialized-configs/${task_name}-${source_arm_id}.json"
  python scripts/planv8_v8_8_config.py \
    --base_config "${base_config}" \
    --output_config "${output_config}" \
    --variant_id "${variant_id}" \
    --source_phase "${source_phase}" \
    --source_arm_id "${source_arm_id}" \
    --primary_model_dir "${PRIMARY_MODEL_DIR}" \
    --primary_backbone_name "${PRIMARY_BACKBONE_NAME}" \
    --train_steps 400
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

mapfile -t CANDIDATE_ROWS < <(
  python - "${SELECTION_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
for row in payload.get("promoted_variants", []):
    print("|".join(
        [
            str(row.get("variant_id", "")).strip(),
            str(row.get("source_phase", "")).strip(),
            str(row.get("source_run_root", "")).strip(),
            str(row.get("arm_id", "")).strip(),
        ]
    ))
PY
)

mapfile -t CONFIRMATION_SEEDS < <(
  python - "${SELECTION_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
for seed in payload.get("seeds", []):
    print(int(seed))
PY
)

for candidate_row in "${CANDIDATE_ROWS[@]}"; do
  IFS='|' read -r variant_id source_phase source_run_root source_arm_id <<< "${candidate_row}"
  for task_name in gsm8k triviaqa fever; do
    materialize_config \
      "${variant_id}" \
      "${source_phase}" \
      "${source_run_root}" \
      "${source_arm_id}" \
      "${task_name}" \
      "${CONFIG_ROOT}/${variant_id}-${task_name}.json"
  done
done

for candidate_row in "${CANDIDATE_ROWS[@]}"; do
  IFS='|' read -r variant_id source_phase source_run_root source_arm_id <<< "${candidate_row}"
  for seed in "${CONFIRMATION_SEEDS[@]}"; do
    for task_name in gsm8k triviaqa fever; do
      run_single_pilot \
        "${CONFIG_ROOT}/${variant_id}-${task_name}.json" \
        "${seed}" \
        "${RUN_ROOT}/${variant_id}/seed_${seed}/${task_name}"
      copy_run_artifacts \
        "${RESULT_ROOT}/${variant_id}/seed_${seed}/${task_name}" \
        "${RUN_ROOT}/${variant_id}/seed_${seed}/${task_name}"
    done
  done
done

python scripts/update_planv8_v8_8_summary.py \
  --result_root "${RESULT_ROOT}" \
  --selection_manifest "${SELECTION_MANIFEST}" \
  --v80_summary "${V80_SUMMARY_PATH}" \
  --v87_summary "${V87_SUMMARY_PATH}" \
  --output_json "${RESULT_ROOT}/v8-8-summary.json" \
  --output_report "${RESULT_ROOT}/v8-8-summary.md"

bash scripts/publish_review_artifacts.sh

echo "PLANv8 V8-8 multiseed confirmation complete."
