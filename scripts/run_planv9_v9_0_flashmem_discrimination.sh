#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PRIMARY_BACKBONE_NAME="${PLANV9_PRIMARY_BACKBONE_NAME:-Qwen3-4B}"
PRIMARY_BACKBONE_KEY="${PLANV9_PRIMARY_BACKBONE_KEY:-qwen34}"
PRIMARY_MODEL_ID="${PLANV9_PRIMARY_MODEL_ID:-Qwen/Qwen3-4B}"
PRIMARY_PREP_SCRIPT="${PLANV9_PRIMARY_PREP_SCRIPT:-scripts/prepare_local_qwen34_model.sh}"
PRIMARY_MODEL_DIR_DEFAULT="${PLANV9_PRIMARY_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-4B}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv9-v9-0-flashmem-discrimination-${PRIMARY_BACKBONE_KEY}}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv9-v9-0-flashmem-discrimination-${PRIMARY_BACKBONE_KEY}}"
PRIMARY_MODEL_DIR="${4:-${PRIMARY_MODEL_DIR_DEFAULT}}"
V80_REVIEW_RUN_ROOT="${5:-runs/review/planv8-v8-0-qwen34-baselines-oracles}"
V80_SUMMARY_PATH="${6:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/v8-0-summary.json}"
SELECTED_PROMPTS_PATH="${7:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/selected-prompt-modes.json}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}" "${HF_HOME}"

DATA_ROOT="${RUN_ROOT}/materialized-datasets"
SOURCE_ROOT="${RUN_ROOT}/materialized-sources"
MANIFEST_ROOT="${RUN_ROOT}/materialized-manifests"
CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${DATA_ROOT}" "${SOURCE_ROOT}" "${MANIFEST_ROOT}" "${CONFIG_ROOT}"

if [[ ! -f "${V80_REVIEW_RUN_ROOT}/materialized-datasets/gsm8k/eval.jsonl" ]]; then
  echo "missing V8-0 qwen34 GSM8K eval split under ${V80_REVIEW_RUN_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SELECTED_PROMPTS_PATH}" ]]; then
  echo "missing selected prompt modes: ${SELECTED_PROMPTS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${V80_SUMMARY_PATH}" ]]; then
  echo "missing V8-0 summary: ${V80_SUMMARY_PATH}" >&2
  exit 1
fi

rsync -a "${V80_REVIEW_RUN_ROOT}/materialized-datasets/gsm8k/" "${DATA_ROOT}/gsm8k/"
if [[ -d "${V80_REVIEW_RUN_ROOT}/materialized-sources/gsm8k" ]]; then
  rsync -a "${V80_REVIEW_RUN_ROOT}/materialized-sources/gsm8k/" "${SOURCE_ROOT}/gsm8k/"
fi
if [[ -f "${V80_REVIEW_RUN_ROOT}/materialized-manifests/gsm8k.json" ]]; then
  cp "${V80_REVIEW_RUN_ROOT}/materialized-manifests/gsm8k.json" "${MANIFEST_ROOT}/gsm8k.json"
fi
if [[ -f "${V80_REVIEW_RUN_ROOT}/materialized-manifests/v8-0-split-plan.json" ]]; then
  cp "${V80_REVIEW_RUN_ROOT}/materialized-manifests/v8-0-split-plan.json" "${MANIFEST_ROOT}/v8-0-split-plan.json"
fi
cp "${SELECTED_PROMPTS_PATH}" "${MANIFEST_ROOT}/selected-prompt-modes.json"
cp "${V80_SUMMARY_PATH}" "${MANIFEST_ROOT}/v8-0-summary.json"

bash "${PRIMARY_PREP_SCRIPT}" \
  "${PRIMARY_MODEL_ID}" \
  "${PRIMARY_MODEL_DIR}" \
  "${HF_HOME}"

SELECTED_PROMPT_VARIANT="$(
  python - "${SELECTED_PROMPTS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
gsm8k = payload.get("gsm8k", {})
print(str(gsm8k.get("selected_prompt_variant", "")))
PY
)"
if [[ -z "${SELECTED_PROMPT_VARIANT}" ]]; then
  echo "failed to resolve selected GSM8K prompt variant from ${SELECTED_PROMPTS_PATH}" >&2
  exit 1
fi

SUPPORT_PATH="${DATA_ROOT}/gsm8k/support.jsonl"
TRAIN_PATH="${DATA_ROOT}/gsm8k/train.jsonl"
EVAL_PATH="${DATA_ROOT}/gsm8k/eval.jsonl"

ARMS=(
  "a0_nomemory_control"
  "a1_legacy_prefix_oracle"
  "a2_precache_latent_oracle"
  "a3_sequence_replay_oracle"
)

for arm_id in "${ARMS[@]}"; do
  python scripts/planv9_v9_0_config.py \
    --arm_id "${arm_id}" \
    --output_config "${CONFIG_ROOT}/${arm_id}.json" \
    --support_path "${SUPPORT_PATH}" \
    --train_path "${TRAIN_PATH}" \
    --eval_path "${EVAL_PATH}" \
    --selected_prompt_variant "${SELECTED_PROMPT_VARIANT}" \
    --primary_model_dir "${PRIMARY_MODEL_DIR}" \
    --primary_backbone_name "${PRIMARY_BACKBONE_NAME}" \
    --hf_cache_dir "${HF_HOME}"
done

run_single_pilot() {
  local suite_config="$1"
  local run_seed="$2"
  local run_dir="$3"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    return 0
  fi
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
}

for arm_id in "${ARMS[@]}"; do
  echo "[planv9-v9-0] running ${arm_id}"
  run_single_pilot "${CONFIG_ROOT}/${arm_id}.json" "${BASE_SEED}" "${RUN_ROOT}/${arm_id}"
done

python scripts/update_planv9_v9_0_summary.py \
  --run_root "${RUN_ROOT}" \
  --v80_summary_path "${V80_SUMMARY_PATH}" \
  --selected_prompt_modes_path "${SELECTED_PROMPTS_PATH}" \
  --output_json "${RESULT_ROOT}/v9-0-summary.json" \
  --output_md "${RESULT_ROOT}/v9-0-summary.md"

echo "[planv9-v9-0] wrote ${RESULT_ROOT}/v9-0-summary.json"
