#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PRIMARY_BACKBONE_NAME="${PLANV9_PRIMARY_BACKBONE_NAME:-Qwen3-4B}"
PRIMARY_BACKBONE_KEY="${PLANV9_PRIMARY_BACKBONE_KEY:-qwen34}"
PRIMARY_MODEL_ID="${PLANV9_PRIMARY_MODEL_ID:-Qwen/Qwen3-4B}"
PRIMARY_PREP_SCRIPT="${PLANV9_PRIMARY_PREP_SCRIPT:-scripts/prepare_local_qwen34_model.sh}"
PRIMARY_MODEL_DIR_DEFAULT="${PLANV9_PRIMARY_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-4B}"
V90_SUMMARY_PATH_DEFAULT="${PLANV9_V90_SUMMARY_PATH:-results/generated/review/planv9-v9-0-flashmem-discrimination-qwen34/v9-0-summary.json}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv9-v9-1-longhorizon-baselines-${PRIMARY_BACKBONE_KEY}}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv9-v9-1-longhorizon-baselines-${PRIMARY_BACKBONE_KEY}}"
PRIMARY_MODEL_DIR="${4:-${PRIMARY_MODEL_DIR_DEFAULT}}"
V90_SUMMARY_PATH="${5:-${V90_SUMMARY_PATH_DEFAULT}}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-tmp/.cache/huggingface/hub}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}" "${HF_HOME}"

DATA_ROOT="${RUN_ROOT}/materialized-datasets"
MANIFEST_ROOT="${RUN_ROOT}/materialized-manifests"
CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${DATA_ROOT}" "${MANIFEST_ROOT}" "${CONFIG_ROOT}"

if [[ ! -f "${V90_SUMMARY_PATH}" ]]; then
  echo "missing V9-0 summary: ${V90_SUMMARY_PATH}" >&2
  exit 1
fi

python - "${V90_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step != "hard_fail_a2_shift_mainline_consumer_to_c0_or_c2":
    raise SystemExit(
        f"V9-0 did not authorize the C0/C2-only V9-1 path; recommended_next_step={next_step!r}"
    )
PY

bash "${PRIMARY_PREP_SCRIPT}" \
  "${PRIMARY_MODEL_ID}" \
  "${PRIMARY_MODEL_DIR}" \
  "${HF_HOME}"

python scripts/planv9_v9_1_config.py materialize \
  --output_root "${RUN_ROOT}" \
  --seed "${BASE_SEED}"

run_static_eval() {
  local benchmark_name="$1"
  local baseline_id="$2"
  local dataset_path="$3"
  local eval_examples="$4"
  local evaluator_type="$5"
  local metric_name="$6"
  local support_dataset_path="${7:-}"
  local config_path="${CONFIG_ROOT}/${benchmark_name}-${baseline_id}.json"
  local run_dir="${RUN_ROOT}/${benchmark_name}/${baseline_id}"
  local config_args=()
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    return 0
  fi
  if [[ -n "${support_dataset_path}" ]]; then
    config_args+=(--support_dataset_path "${support_dataset_path}")
  fi
  python scripts/planv9_v9_1_config.py static-config \
    --benchmark_name "${benchmark_name}" \
    --baseline_id "${baseline_id}" \
    --dataset_path "${dataset_path}" \
    --output_config "${config_path}" \
    --primary_model_dir "${PRIMARY_MODEL_DIR}" \
    --eval_examples "${eval_examples}" \
    --evaluator_type "${evaluator_type}" \
    --metric_name "${metric_name}" \
    "${config_args[@]}" \
    --primary_backbone_name "${PRIMARY_BACKBONE_NAME}" \
    --hf_cache_dir "${HF_HOME}"

  if [[ "${baseline_id}" == "b3_text_rag" ]]; then
    python scripts/planv9_v9_1_retrieval_eval.py \
      --config "${config_path}" \
      --seed "${BASE_SEED}" \
      --output_dir "${run_dir}"
  else
    python -m eval \
      --config "${config_path}" \
      --seed "${BASE_SEED}" \
      --output_dir "${run_dir}"
  fi
}

declare -a STATIC_BENCHMARKS=(
  "memoryagentbench memoryagentbench memoryagent_score 100"
  "longmemeval qa_f1 f1 100"
)

declare -a STATIC_BASELINES=(
  "b0_short_window b0_short_window_eval"
  "b1_full_history b1_full_history_eval"
  "b2_text_summary b2_text_summary_eval"
  "b3_text_rag b3_rag_eval"
)

for benchmark_spec in "${STATIC_BENCHMARKS[@]}"; do
  read -r benchmark_name evaluator_type metric_name eval_examples <<<"${benchmark_spec}"
  for baseline_spec in "${STATIC_BASELINES[@]}"; do
    read -r baseline_id dataset_key <<<"${baseline_spec}"
    dataset_path="${DATA_ROOT}/${benchmark_name}/${dataset_key}.jsonl"
    support_dataset_path=""
    if [[ "${baseline_id}" == "b3_text_rag" ]]; then
      support_dataset_path="${DATA_ROOT}/${benchmark_name}/b3_rag_support.jsonl"
    fi
    echo "[planv9-v9-1] running ${benchmark_name}/${baseline_id}"
    run_static_eval \
      "${benchmark_name}" \
      "${baseline_id}" \
      "${dataset_path}" \
      "${eval_examples}" \
      "${evaluator_type}" \
      "${metric_name}" \
      "${support_dataset_path}"
  done
done

ALFWORLD_MANIFEST="${MANIFEST_ROOT}/alfworld-pilot.json"
for baseline_id in b0_short_window b1_full_history b2_text_summary b3_text_rag; do
  run_dir="${RUN_ROOT}/alfworld/${baseline_id}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    continue
  fi
  echo "[planv9-v9-1] running alfworld/${baseline_id}"
  python scripts/planv9_v9_1_alfworld_eval.py \
    --manifest_path "${ALFWORLD_MANIFEST}" \
    --baseline_id "${baseline_id}" \
    --primary_model_dir "${PRIMARY_MODEL_DIR}" \
    --primary_backbone_name "${PRIMARY_BACKBONE_NAME}" \
    --seed "${BASE_SEED}" \
    --output_dir "${run_dir}" \
    --eval_episodes 120 \
    --max_steps 12 \
    --hf_cache_dir "${HF_HOME}"
done

cp "${V90_SUMMARY_PATH}" "${RESULT_ROOT}/v9-0-summary.reference.json"

python scripts/update_planv9_v9_1_summary.py \
  --run_root "${RUN_ROOT}" \
  --v90_summary_path "${V90_SUMMARY_PATH}" \
  --output_json "${RESULT_ROOT}/v9-1-summary.json" \
  --output_md "${RESULT_ROOT}/v9-1-summary.md"

echo "[planv9-v9-1] wrote ${RESULT_ROOT}/v9-1-summary.json"
