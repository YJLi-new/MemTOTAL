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
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-7-comparators}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-7-comparators}"
PRIMARY_MODEL_DIR="${4:-${PRIMARY_MODEL_DIR_DEFAULT}}"
V80_SUMMARY_PATH="${5:-results/generated/review/planv8-v8-0-${PRIMARY_BACKBONE_KEY}-baselines-oracles/v8-0-summary.json}"
SELECTED_PROMPTS_PATH="${6:-results/generated/review/planv8-v8-0-${PRIMARY_BACKBONE_KEY}-baselines-oracles/selected-prompt-modes.json}"
V76_SUMMARY_PATH="${7:-results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json}"
V86_SUMMARY_PATH="${8:-results/generated/review/planv8-v8-6-writer-aux-${PRIMARY_BACKBONE_KEY}/v8-6-summary.json}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PLANV8_EXPERIMENT_PREFIX="${PLANV8_EXPERIMENT_PREFIX:-planv8}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}" "${HF_HOME}"

DATA_ROOT="${RUN_ROOT}/materialized-datasets"
SOURCE_ROOT="${RUN_ROOT}/materialized-sources"
MANIFEST_ROOT="${RUN_ROOT}/materialized-manifests"
CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${DATA_ROOT}" "${SOURCE_ROOT}" "${MANIFEST_ROOT}" "${CONFIG_ROOT}"

python - "${V86_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step != "open_v8_7_comparators":
    raise SystemExit(
        f"V8-6 did not authorize V8-7; recommended_next_step={next_step!r}"
    )
PY

SPLIT_PLAN_JSON="${MANIFEST_ROOT}/v8-7-split-plan.json"
python - <<'PY' "${SPLIT_PLAN_JSON}"
import json
import sys
from pathlib import Path

split_plan_path = Path(sys.argv[1])
split_plan_path.write_text(
    json.dumps(
        {
            "gsm8k": {
                "source_examples": 136,
                "support_examples": 8,
                "train_examples": 64,
                "eval_examples": 64,
            },
            "triviaqa": {
                "source_examples": 136,
                "support_examples": 8,
                "train_examples": 64,
                "eval_examples": 64,
            },
            "fever": {
                "source_examples": 256,
                "support_examples": 8,
                "train_examples": 48,
                "eval_examples": 48,
            },
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

python -m memtotal.tasks.writer_jointpeft_data \
  --output_root "${DATA_ROOT}" \
  --source_output_root "${SOURCE_ROOT}" \
  --manifest_root "${MANIFEST_ROOT}" \
  --seed "${BASE_SEED}" \
  --benchmarks "gsm8k,triviaqa,fever" \
  --split_plan_json "${SPLIT_PLAN_JSON}"

bash "${PRIMARY_PREP_SCRIPT}" \
  "${PRIMARY_MODEL_ID}" \
  "${PRIMARY_MODEL_DIR}" \
  "${HF_HOME}"

cp "${V80_SUMMARY_PATH}" "${RESULT_ROOT}/v8-0-summary.reference.json"
cp "${V76_SUMMARY_PATH}" "${RESULT_ROOT}/v7-6-summary.reference.json"
cp "${V86_SUMMARY_PATH}" "${RESULT_ROOT}/v8-6-summary.reference.json"
if [[ -f "${SELECTED_PROMPTS_PATH}" ]]; then
  cp "${SELECTED_PROMPTS_PATH}" "${RESULT_ROOT}/selected-prompt-modes.json"
fi

copy_run_artifacts() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "${dst_dir}"
  rsync -a \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    --exclude='.analysis.lock' \
    --exclude='.suite.lock' \
    "${src_dir}/" "${dst_dir}/"
}

run_rag_eval() {
  local config_path="$1"
  local run_seed="$2"
  local run_dir="$3"
  mkdir -p "${run_dir}"
  python scripts/planv8_v8_7_retrieval_eval.py \
    --config "${config_path}" \
    --seed "${run_seed}" \
    --output_dir "${run_dir}"
}

run_memgen_eval() {
  local config_path="$1"
  local run_seed="$2"
  local run_dir="$3"
  mkdir -p "${run_dir}"
  bash scripts/run_memgen.sh \
    --config "${config_path}" \
    --seed "${run_seed}" \
    --output_dir "${run_dir}"
}

seed_offset=0
for task_name in gsm8k triviaqa fever; do
  python scripts/planv8_v8_7_config.py rag \
    --task_name "${task_name}" \
    --output_config "${CONFIG_ROOT}/${task_name}-m1_text_rag.json" \
    --eval_path "${DATA_ROOT}/${task_name}/eval.jsonl" \
    --support_path "${DATA_ROOT}/${task_name}/support.jsonl" \
    --primary_model_dir "${PRIMARY_MODEL_DIR}" \
    --primary_backbone_name "${PRIMARY_BACKBONE_NAME}"
  run_rag_eval \
    "${CONFIG_ROOT}/${task_name}-m1_text_rag.json" \
    "$((BASE_SEED + seed_offset))" \
    "${RUN_ROOT}/m1_text_rag-${task_name}"
  copy_run_artifacts \
    "${RUN_ROOT}/m1_text_rag-${task_name}" \
    "${RESULT_ROOT}/m1_text_rag/${task_name}"
  seed_offset=$((seed_offset + 10))
done

for task_name in gsm8k triviaqa; do
  python scripts/planv8_v8_7_config.py memgen \
    --task_name "${task_name}" \
    --output_config "${CONFIG_ROOT}/${task_name}-m2_memgen.json" \
    --primary_model_dir "${PRIMARY_MODEL_DIR}" \
    --primary_backbone_name "${PRIMARY_BACKBONE_NAME}"
  run_memgen_eval \
    "${CONFIG_ROOT}/${task_name}-m2_memgen.json" \
    "$((BASE_SEED + seed_offset))" \
    "${RUN_ROOT}/m2_memgen-${task_name}"
  copy_run_artifacts \
    "${RUN_ROOT}/m2_memgen-${task_name}" \
    "${RESULT_ROOT}/m2_memgen/${task_name}"
  seed_offset=$((seed_offset + 10))
done

python scripts/update_planv8_v8_7_summary.py \
  --result_root "${RESULT_ROOT}" \
  --v80_summary "${V80_SUMMARY_PATH}" \
  --v76_summary "${V76_SUMMARY_PATH}" \
  --v86_summary "${V86_SUMMARY_PATH}" \
  --output_json "${RESULT_ROOT}/v8-7-summary.json" \
  --output_report "${RESULT_ROOT}/v8-7-summary.md"

bash scripts/publish_review_artifacts.sh

echo "PLANv8 V8-7 comparators complete."
