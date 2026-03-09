#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-60917}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/writer-deep-prefix-jointpeft-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/writer-deep-prefix-jointpeft-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

DATA_ROOT="${RUN_ROOT}/materialized-datasets"
SOURCE_ROOT="${RUN_ROOT}/materialized-sources"
MANIFEST_ROOT="${RUN_ROOT}/materialized-manifests"
CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${DATA_ROOT}" "${SOURCE_ROOT}" "${MANIFEST_ROOT}" "${CONFIG_ROOT}"

python -m memtotal.tasks.writer_jointpeft_data \
  --output_root "${DATA_ROOT}" \
  --source_output_root "${SOURCE_ROOT}" \
  --manifest_root "${MANIFEST_ROOT}" \
  --seed "${BASE_SEED}"

materialize_config() {
  local src_config="$1"
  local output_config="$2"
  local support_path="$3"
  local train_path="$4"
  local eval_path="$5"
  local variant="$6"
  python - "${src_config}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${variant}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

source_config = sys.argv[1]
output_config = Path(sys.argv[2])
support_path = str(Path(sys.argv[3]).resolve())
train_path = str(Path(sys.argv[4]).resolve())
eval_path = str(Path(sys.argv[5]).resolve())
variant = sys.argv[6]

config = load_config(source_config)
config.setdefault("task", {})
config.setdefault("method", {})
config.setdefault("runtime", {})
config["task"]["support_dataset_path"] = support_path
config["task"]["train_dataset_path"] = train_path
config["task"]["train_support_dataset_path"] = support_path
config["task"]["dataset_path"] = eval_path
config["task"]["support_lookup_dataset_paths"] = []
config["task"]["train_support_episode_bank_path"] = ""
config["task"]["pilot_split"] = str(config["task"].get("smoke_subset", config["task"].get("split", "eval")))

receiver_lora_disabled = {
    "enabled": False,
    "target_layers": [],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 0,
    "alpha": 4.0,
    "dropout": 0.0,
}
receiver_lora_early4 = {
    "enabled": True,
    "target_layers": [0, 1, 2, 3],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 2,
    "alpha": 4.0,
    "dropout": 0.0,
}

if variant == "control":
    config["runtime"]["shared_injection_arm"] = "base_only"
    config["runtime"]["writer_memory_control"] = "real"
    config["runtime"]["pilot_arm_alias"] = "A"
    config["method"]["receiver_lora"] = receiver_lora_disabled
elif variant == "writer":
    config["runtime"]["shared_injection_arm"] = "injected"
    config["runtime"]["writer_memory_control"] = "real"
    config["runtime"]["pilot_arm_alias"] = "II_writer_direct"
elif variant == "source_stub_health":
    config["runtime"]["shared_injection_arm"] = "injected"
    config["runtime"]["writer_memory_control"] = "real"
    config["runtime"]["pilot_arm_alias"] = "I_source_stub_health"
    config["runtime"]["pilot_prefix_source_mode"] = "source_stub"
    config["runtime"]["pilot_deep_prefix_init_mode"] = "kv_stat_match"
    config["runtime"]["pilot_train_steps"] = 32
    config["runtime"]["pilot_snapshot_steps"] = [0, 8, 16, 32]
    config["runtime"]["pilot_lr_schedule"] = "constant_with_linear_warmup"
    config["runtime"]["pilot_lr_warmup_steps"] = 8
    config["runtime"]["pilot_projector_warmup_steps"] = 0
    config["runtime"]["pilot_source_stub_learning_rate"] = 1.0e-4
    config["runtime"]["pilot_projector_learning_rate"] = 1.5e-4
    config["runtime"]["pilot_receiver_lora_learning_rate"] = 5.0e-5
    config["runtime"]["pilot_gradient_clip_norm"] = 1.0
    config["method"]["receiver_lora"] = receiver_lora_early4
else:
    raise ValueError(f"unsupported materialization variant: {variant}")

output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

materialize_config \
  "configs/exp/writer_circuit_g1_source_stub_gsm8k_template.yaml" \
  "${CONFIG_ROOT}/gsm8k-source-stub-health.json" \
  "${DATA_ROOT}/gsm8k/support.jsonl" \
  "${DATA_ROOT}/gsm8k/train.jsonl" \
  "${DATA_ROOT}/gsm8k/eval.jsonl" \
  "source_stub_health"

for task_name in gsm8k narrativeqa fever; do
  materialize_config \
    "configs/exp/writer_circuit_g2_writer_direct_${task_name}_template.yaml" \
    "${CONFIG_ROOT}/${task_name}-control.json" \
    "${DATA_ROOT}/${task_name}/support.jsonl" \
    "${DATA_ROOT}/${task_name}/train.jsonl" \
    "${DATA_ROOT}/${task_name}/eval.jsonl" \
    "control"
  materialize_config \
    "configs/exp/writer_circuit_g2_writer_direct_${task_name}_template.yaml" \
    "${CONFIG_ROOT}/${task_name}-writer.json" \
    "${DATA_ROOT}/${task_name}/support.jsonl" \
    "${DATA_ROOT}/${task_name}/train.jsonl" \
    "${DATA_ROOT}/${task_name}/eval.jsonl" \
    "writer"
done

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

ensure_suite_complete() {
  local suite_config="$1"
  local run_seed="$2"
  local run_dir="$3"
  local arm_spec="$4"
  mkdir -p "${run_dir}"
  local lock_fd
  exec {lock_fd}> "${run_dir}/.suite.lock"
  flock "${lock_fd}"
  if [[ ! -f "${run_dir}/suite_metrics.json" ]]; then
    python scripts/run_m4_selected_shared_injection_suite.py \
      --config "${suite_config}" \
      --resume "${RESUME_STAGE_B_ROOT}" \
      --output_root "${run_dir}" \
      --seed "${run_seed}" \
      --arm-spec "${arm_spec}"
  fi
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

ensure_suite_complete "${CONFIG_ROOT}/gsm8k-source-stub-health.json" "${BASE_SEED}" \
  "${RUN_ROOT}/gsm8k-source-stub-health" "pilot-I-source-stub-health:I_source_stub_health:injected:real:0"

ensure_suite_complete "${CONFIG_ROOT}/gsm8k-control.json" "$((BASE_SEED + 1000))" \
  "${RUN_ROOT}/gsm8k-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/gsm8k-writer.json" "$((BASE_SEED + 2000))" \
  "${RUN_ROOT}/gsm8k-writer" "pilot-I-writer-direct:II_writer_direct:injected:real:0"

ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-control.json" "$((BASE_SEED + 3000))" \
  "${RUN_ROOT}/narrativeqa-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-writer.json" "$((BASE_SEED + 4000))" \
  "${RUN_ROOT}/narrativeqa-writer" "pilot-I-writer-direct:II_writer_direct:injected:real:0"

ensure_suite_complete "${CONFIG_ROOT}/fever-control.json" "$((BASE_SEED + 5000))" \
  "${RUN_ROOT}/fever-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/fever-writer.json" "$((BASE_SEED + 6000))" \
  "${RUN_ROOT}/fever-writer" "pilot-I-writer-direct:II_writer_direct:injected:real:0"

cp "${DATA_ROOT}/split-manifest.json" "${RESULT_ROOT}/split-manifest.json"

copy_run_artifacts() {
  local dst_dir="$1"
  local run_dir="$2"
  local pilot_subdir="$3"
  mkdir -p "${dst_dir}"
  cp "${run_dir}/${pilot_subdir}/metrics.json" "${dst_dir}/metrics.json"
  cp "${run_dir}/${pilot_subdir}/train_events.json" "${dst_dir}/train_events.json"
  cp "${run_dir}/${pilot_subdir}/task_case_dump.jsonl" "${dst_dir}/task_case_dump.jsonl"
  cp "${run_dir}/suite_metrics.json" "${dst_dir}/suite_metrics.json"
}

copy_run_artifacts \
  "${RESULT_ROOT}/source-stub-health/gsm8k" \
  "${RUN_ROOT}/gsm8k-source-stub-health" \
  "pilot-I-source-stub-health"
copy_run_artifacts "${RESULT_ROOT}/gsm8k/control" "${RUN_ROOT}/gsm8k-control" "pilot-A-selected"
copy_run_artifacts "${RESULT_ROOT}/gsm8k/writer" "${RUN_ROOT}/gsm8k-writer" "pilot-I-writer-direct"
copy_run_artifacts "${RESULT_ROOT}/narrativeqa/control" "${RUN_ROOT}/narrativeqa-control" "pilot-A-selected"
copy_run_artifacts "${RESULT_ROOT}/narrativeqa/writer" "${RUN_ROOT}/narrativeqa-writer" "pilot-I-writer-direct"
copy_run_artifacts "${RESULT_ROOT}/fever/control" "${RUN_ROOT}/fever-control" "pilot-A-selected"
copy_run_artifacts "${RESULT_ROOT}/fever/writer" "${RUN_ROOT}/fever-writer" "pilot-I-writer-direct"

python scripts/update_writer_deep_prefix_jointpeft_summary.py \
  --source_stub_health_metrics_json "${RUN_ROOT}/gsm8k-source-stub-health/pilot-I-source-stub-health/metrics.json" \
  --source_stub_health_train_events_json "${RUN_ROOT}/gsm8k-source-stub-health/pilot-I-source-stub-health/train_events.json" \
  --gsm8k_control_metrics_json "${RUN_ROOT}/gsm8k-control/pilot-A-selected/metrics.json" \
  --gsm8k_writer_metrics_json "${RUN_ROOT}/gsm8k-writer/pilot-I-writer-direct/metrics.json" \
  --gsm8k_writer_train_events_json "${RUN_ROOT}/gsm8k-writer/pilot-I-writer-direct/train_events.json" \
  --narrativeqa_control_metrics_json "${RUN_ROOT}/narrativeqa-control/pilot-A-selected/metrics.json" \
  --narrativeqa_writer_metrics_json "${RUN_ROOT}/narrativeqa-writer/pilot-I-writer-direct/metrics.json" \
  --narrativeqa_writer_train_events_json "${RUN_ROOT}/narrativeqa-writer/pilot-I-writer-direct/train_events.json" \
  --fever_control_metrics_json "${RUN_ROOT}/fever-control/pilot-A-selected/metrics.json" \
  --fever_writer_metrics_json "${RUN_ROOT}/fever-writer/pilot-I-writer-direct/metrics.json" \
  --fever_writer_train_events_json "${RUN_ROOT}/fever-writer/pilot-I-writer-direct/train_events.json" \
  --output_json "${RESULT_ROOT}/writer-deep-prefix-jointpeft-summary.json" \
  --output_report "${RESULT_ROOT}/writer-deep-prefix-jointpeft-summary.md"

first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

HISTORICAL_RESULT_DIR=""
if HISTORICAL_RESULT_DIR="$(first_existing_dir \
  "results/generated/review/writer-circuit-opening-qwen25" \
  "results/generated/writer-circuit-opening-qwen25" \
  "/root/autodl-tmp/results/generated/writer-circuit-opening-qwen25" \
  "/root/autodl-tmp/results/generated/review/writer-circuit-opening-qwen25")"; then
  :
fi
HISTORICAL_RUN_DIR=""
if HISTORICAL_RUN_DIR="$(first_existing_dir \
  "runs/review/writer-circuit-opening-qwen25" \
  "runs/verify/writer-circuit-opening-qwen25" \
  "/root/autodl-tmp/runs/verify/writer-circuit-opening-qwen25" \
  "/root/autodl-tmp/runs/review/writer-circuit-opening-qwen25")"; then
  :
fi

if [[ -n "${HISTORICAL_RESULT_DIR}" && -n "${HISTORICAL_RUN_DIR}" ]]; then
  mkdir -p "results/generated/review/writer-circuit-opening-qwen25"
  python scripts/update_writer_circuit_opening_addendum_summary.py \
    --gsm8k_control_metrics_json "${HISTORICAL_RESULT_DIR}/gsm8k/control/metrics.json" \
    --gsm8k_p1a_metrics_json "${HISTORICAL_RESULT_DIR}/gsm8k/p1a/metrics.json" \
    --gsm8k_p1a_train_events_json "${HISTORICAL_RUN_DIR}/gsm8k-p1a/pilot-I-source-stub/train_events.json" \
    --gsm8k_p2a_metrics_json "${HISTORICAL_RESULT_DIR}/gsm8k/p2a/metrics.json" \
    --gsm8k_p2a_train_events_json "${HISTORICAL_RUN_DIR}/gsm8k-p2a/pilot-I-source-stub-lora/train_events.json" \
    --narrativeqa_control_metrics_json "${HISTORICAL_RESULT_DIR}/narrativeqa/control/metrics.json" \
    --narrativeqa_p1a_metrics_json "${HISTORICAL_RESULT_DIR}/narrativeqa/p1a/metrics.json" \
    --narrativeqa_p1a_train_events_json "${HISTORICAL_RUN_DIR}/narrativeqa-p1a/pilot-I-source-stub/train_events.json" \
    --narrativeqa_p2a_metrics_json "${HISTORICAL_RESULT_DIR}/narrativeqa/p2a/metrics.json" \
    --narrativeqa_p2a_train_events_json "${HISTORICAL_RUN_DIR}/narrativeqa-p2a/pilot-I-source-stub-lora/train_events.json" \
    --fever_control_metrics_json "${HISTORICAL_RESULT_DIR}/fever/control/metrics.json" \
    --fever_p1a_metrics_json "${HISTORICAL_RESULT_DIR}/fever/p1a/metrics.json" \
    --fever_p1a_train_events_json "${HISTORICAL_RUN_DIR}/fever-p1a/pilot-I-source-stub/train_events.json" \
    --fever_p2a_metrics_json "${HISTORICAL_RESULT_DIR}/fever/p2a/metrics.json" \
    --fever_p2a_train_events_json "${HISTORICAL_RUN_DIR}/fever-p2a/pilot-I-source-stub-lora/train_events.json" \
    --output_json "results/generated/review/writer-circuit-opening-qwen25/writer-circuit-opening-addendum-summary.json" \
    --output_report "results/generated/review/writer-circuit-opening-qwen25/writer-circuit-opening-addendum-summary.md"
else
  echo "warning: historical writer-circuit-opening artifacts not found; skipped addendum summary refresh" >&2
fi

./scripts/publish_review_artifacts.sh
