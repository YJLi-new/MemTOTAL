#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-53191}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/writer-weaver-qwen25-f1a}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/writer-weaver-qwen25-f1a}"
W0_RUN_ROOT_INPUT="${4:-runs/review/writer-weaver-qwen25-smoke}"
W0_RESULT_ROOT_INPUT="${5:-results/generated/review/writer-weaver-qwen25-smoke}"
RESUME_STAGE_B_ROOT="${6:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"

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

W0_RUN_ROOT="$(resolve_path "${W0_RUN_ROOT_INPUT}")"
W0_RESULT_ROOT="$(resolve_path "${W0_RESULT_ROOT_INPUT}")"

for required_path in \
  "${W0_RUN_ROOT}/materialized-datasets/split-manifest.json" \
  "${W0_RESULT_ROOT}/w0-summary.json" \
  "${W0_RESULT_ROOT}/gsm8k/control/metrics.json" \
  "${W0_RESULT_ROOT}/gsm8k/support-context/metrics.json" \
  "${W0_RESULT_ROOT}/narrativeqa/control/metrics.json" \
  "${W0_RESULT_ROOT}/narrativeqa/support-context/metrics.json" \
  "${W0_RESULT_ROOT}/fever/control/metrics.json" \
  "${W0_RESULT_ROOT}/fever/support-context/metrics.json"
do
  if [[ ! -f "${required_path}" ]]; then
    echo "missing required W0 reference: ${required_path}" >&2
    exit 1
  fi
done

CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${CONFIG_ROOT}"

materialize_config() {
  local src_config="$1"
  local output_config="$2"
  local support_path="$3"
  local train_path="$4"
  local eval_path="$5"
  python - "${src_config}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

source_config = sys.argv[1]
output_config = Path(sys.argv[2])
support_path = str(Path(sys.argv[3]).resolve())
train_path = str(Path(sys.argv[4]).resolve())
eval_path = str(Path(sys.argv[5]).resolve())

config = load_config(source_config)
config.setdefault("task", {})
config["task"]["support_dataset_path"] = support_path
config["task"]["train_dataset_path"] = train_path
config["task"]["train_support_dataset_path"] = support_path
config["task"]["dataset_path"] = eval_path
config["task"]["support_lookup_dataset_paths"] = []
config["task"]["train_support_episode_bank_path"] = ""
config["task"]["pilot_split"] = str(config["task"].get("smoke_subset", config["task"].get("split", "eval")))
output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

materialize_config \
  "configs/exp/writer_weaver_qwen25_f1a_gsm8k_template.yaml" \
  "${CONFIG_ROOT}/gsm8k-f1a.json" \
  "${W0_RUN_ROOT}/materialized-datasets/gsm8k/support.jsonl" \
  "${W0_RUN_ROOT}/materialized-datasets/gsm8k/train.jsonl" \
  "${W0_RUN_ROOT}/materialized-datasets/gsm8k/eval.jsonl"
materialize_config \
  "configs/exp/writer_weaver_qwen25_f1a_narrativeqa_template.yaml" \
  "${CONFIG_ROOT}/narrativeqa-f1a.json" \
  "${W0_RUN_ROOT}/materialized-datasets/narrativeqa/support.jsonl" \
  "${W0_RUN_ROOT}/materialized-datasets/narrativeqa/train.jsonl" \
  "${W0_RUN_ROOT}/materialized-datasets/narrativeqa/eval.jsonl"
materialize_config \
  "configs/exp/writer_weaver_qwen25_f1a_fever_labelgen_template.yaml" \
  "${CONFIG_ROOT}/fever-f1a.json" \
  "${W0_RUN_ROOT}/materialized-datasets/fever/support.jsonl" \
  "${W0_RUN_ROOT}/materialized-datasets/fever/train.jsonl" \
  "${W0_RUN_ROOT}/materialized-datasets/fever/eval.jsonl"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

ensure_suite_complete() {
  local suite_config="$1"
  local run_seed="$2"
  local run_dir="$3"
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
      --arm-spec "pilot-I-big-context:I_f1a:injected:real:0"
  fi
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

ensure_suite_complete "${CONFIG_ROOT}/gsm8k-f1a.json" "${BASE_SEED}" \
  "${RUN_ROOT}/gsm8k-f1a"
ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-f1a.json" "$((BASE_SEED + 1000))" \
  "${RUN_ROOT}/narrativeqa-f1a"
ensure_suite_complete "${CONFIG_ROOT}/fever-f1a.json" "$((BASE_SEED + 2000))" \
  "${RUN_ROOT}/fever-f1a"

copy_task_artifacts() {
  local task_name="$1"
  local run_dir="$2"
  local dst_dir="${RESULT_ROOT}/${task_name}/f1a"
  mkdir -p "${dst_dir}"
  cp "${run_dir}/pilot-I-big-context/metrics.json" "${dst_dir}/metrics.json"
  cp "${run_dir}/pilot-I-big-context/train_events.json" "${dst_dir}/train_events.json"
  cp "${run_dir}/suite_metrics.json" "${dst_dir}/suite_metrics.json"
}

copy_task_artifacts "gsm8k" "${RUN_ROOT}/gsm8k-f1a"
copy_task_artifacts "narrativeqa" "${RUN_ROOT}/narrativeqa-f1a"
copy_task_artifacts "fever" "${RUN_ROOT}/fever-f1a"

python scripts/update_writer_weaver_f1a_summary.py \
  --w0_summary_json "${W0_RESULT_ROOT}/w0-summary.json" \
  --gsm8k_control_metrics_json "${W0_RESULT_ROOT}/gsm8k/control/metrics.json" \
  --gsm8k_w0_metrics_json "${W0_RESULT_ROOT}/gsm8k/support-context/metrics.json" \
  --gsm8k_f1a_metrics_json "${RUN_ROOT}/gsm8k-f1a/pilot-I-big-context/metrics.json" \
  --narrativeqa_control_metrics_json "${W0_RESULT_ROOT}/narrativeqa/control/metrics.json" \
  --narrativeqa_w0_metrics_json "${W0_RESULT_ROOT}/narrativeqa/support-context/metrics.json" \
  --narrativeqa_f1a_metrics_json "${RUN_ROOT}/narrativeqa-f1a/pilot-I-big-context/metrics.json" \
  --fever_control_metrics_json "${W0_RESULT_ROOT}/fever/control/metrics.json" \
  --fever_w0_metrics_json "${W0_RESULT_ROOT}/fever/support-context/metrics.json" \
  --fever_f1a_metrics_json "${RUN_ROOT}/fever-f1a/pilot-I-big-context/metrics.json" \
  --output_json "${RESULT_ROOT}/f1a-summary.json" \
  --output_report "${RESULT_ROOT}/f1a-summary.md"

./scripts/publish_review_artifacts.sh

echo "Writer-weaver F1a suite complete"
