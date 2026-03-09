#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-41791}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/writer-weaver-qwen25-smoke}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/writer-weaver-qwen25-smoke}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

DATA_ROOT="${RUN_ROOT}/materialized-datasets"
CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${DATA_ROOT}" "${CONFIG_ROOT}"

python - "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))


data_root = Path(sys.argv[1]).resolve()
gsm8k_rows = load_jsonl(Path("data/benchmarks/materialized/gsm8k/eval-real-smoke8.jsonl").resolve())
narrativeqa_rows = load_jsonl(Path("data/benchmarks/materialized/narrativeqa/eval-real-smoke4.jsonl").resolve())
fever_support_rows = load_jsonl(Path("data/benchmarks/pilots/fever/pilot-support8.jsonl").resolve())
fever_eval_rows = load_jsonl(Path("data/benchmarks/materialized/fever/eval-real-smoke4.jsonl").resolve())

manifest = {}
task_splits = {
    "gsm8k": {
        "support": gsm8k_rows[:4],
        "train": gsm8k_rows[4:6],
        "eval": gsm8k_rows[6:8],
    },
    "narrativeqa": {
        "support": narrativeqa_rows[:2],
        "train": narrativeqa_rows[2:3],
        "eval": narrativeqa_rows[3:4],
    },
    "fever": {
        "support": fever_support_rows[:6],
        "train": fever_eval_rows[:2],
        "eval": fever_eval_rows[2:4],
    },
}
for task_name, splits in task_splits.items():
    manifest[task_name] = {}
    for split_name, rows in splits.items():
        output_path = data_root / task_name / f"{split_name}.jsonl"
        write_jsonl(output_path, rows)
        manifest[task_name][split_name] = {
            "path": str(output_path),
            "rows": len(rows),
            "ids": [str(row.get("id", "")) for row in rows],
        }
(data_root / "split-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

materialize_config() {
  local src_config="$1"
  local output_config="$2"
  local support_path="$3"
  local train_path="$4"
  local eval_path="$5"
  local stimulus_mode="$6"
  python - "${src_config}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${stimulus_mode}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

source_config = sys.argv[1]
output_config = Path(sys.argv[2])
support_path = str(Path(sys.argv[3]).resolve())
train_path = str(Path(sys.argv[4]).resolve())
eval_path = str(Path(sys.argv[5]).resolve())
stimulus_mode = sys.argv[6]

config = load_config(source_config)
config.setdefault("runtime", {})
config["runtime"]["pilot_writer_stimulus_mode"] = stimulus_mode
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
  "configs/exp/writer_weaver_qwen25_smoke_gsm8k_template.yaml" \
  "${CONFIG_ROOT}/gsm8k-support-only.json" \
  "${DATA_ROOT}/gsm8k/support.jsonl" \
  "${DATA_ROOT}/gsm8k/train.jsonl" \
  "${DATA_ROOT}/gsm8k/eval.jsonl" \
  "support_only"
materialize_config \
  "configs/exp/writer_weaver_qwen25_smoke_gsm8k_template.yaml" \
  "${CONFIG_ROOT}/gsm8k-support-context.json" \
  "${DATA_ROOT}/gsm8k/support.jsonl" \
  "${DATA_ROOT}/gsm8k/train.jsonl" \
  "${DATA_ROOT}/gsm8k/eval.jsonl" \
  "support_and_context"
materialize_config \
  "configs/exp/writer_weaver_qwen25_smoke_narrativeqa_template.yaml" \
  "${CONFIG_ROOT}/narrativeqa-support-only.json" \
  "${DATA_ROOT}/narrativeqa/support.jsonl" \
  "${DATA_ROOT}/narrativeqa/train.jsonl" \
  "${DATA_ROOT}/narrativeqa/eval.jsonl" \
  "support_only"
materialize_config \
  "configs/exp/writer_weaver_qwen25_smoke_narrativeqa_template.yaml" \
  "${CONFIG_ROOT}/narrativeqa-support-context.json" \
  "${DATA_ROOT}/narrativeqa/support.jsonl" \
  "${DATA_ROOT}/narrativeqa/train.jsonl" \
  "${DATA_ROOT}/narrativeqa/eval.jsonl" \
  "support_and_context"
materialize_config \
  "configs/exp/writer_weaver_qwen25_smoke_fever_labelgen_template.yaml" \
  "${CONFIG_ROOT}/fever-support-only.json" \
  "${DATA_ROOT}/fever/support.jsonl" \
  "${DATA_ROOT}/fever/train.jsonl" \
  "${DATA_ROOT}/fever/eval.jsonl" \
  "support_only"
materialize_config \
  "configs/exp/writer_weaver_qwen25_smoke_fever_labelgen_template.yaml" \
  "${CONFIG_ROOT}/fever-support-context.json" \
  "${DATA_ROOT}/fever/support.jsonl" \
  "${DATA_ROOT}/fever/train.jsonl" \
  "${DATA_ROOT}/fever/eval.jsonl" \
  "support_and_context"

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

ensure_suite_complete "${CONFIG_ROOT}/gsm8k-support-context.json" "${BASE_SEED}" \
  "${RUN_ROOT}/gsm8k-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/gsm8k-support-only.json" "$((BASE_SEED + 1000))" \
  "${RUN_ROOT}/gsm8k-support-only" "pilot-I-real:I_real:injected:real:0"
ensure_suite_complete "${CONFIG_ROOT}/gsm8k-support-context.json" "$((BASE_SEED + 2000))" \
  "${RUN_ROOT}/gsm8k-support-context" "pilot-I-real:I_real:injected:real:0"

ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-support-context.json" "$((BASE_SEED + 3000))" \
  "${RUN_ROOT}/narrativeqa-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-support-only.json" "$((BASE_SEED + 4000))" \
  "${RUN_ROOT}/narrativeqa-support-only" "pilot-I-real:I_real:injected:real:0"
ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-support-context.json" "$((BASE_SEED + 5000))" \
  "${RUN_ROOT}/narrativeqa-support-context" "pilot-I-real:I_real:injected:real:0"

ensure_suite_complete "${CONFIG_ROOT}/fever-support-context.json" "$((BASE_SEED + 6000))" \
  "${RUN_ROOT}/fever-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/fever-support-only.json" "$((BASE_SEED + 7000))" \
  "${RUN_ROOT}/fever-support-only" "pilot-I-real:I_real:injected:real:0"
ensure_suite_complete "${CONFIG_ROOT}/fever-support-context.json" "$((BASE_SEED + 8000))" \
  "${RUN_ROOT}/fever-support-context" "pilot-I-real:I_real:injected:real:0"

mkdir -p "${RESULT_ROOT}"

copy_task_artifacts() {
  local task_name="$1"
  local arm_name="$2"
  local run_dir="$3"
  local pilot_subdir="$4"
  local dst_dir="${RESULT_ROOT}/${task_name}/${arm_name}"
  mkdir -p "${dst_dir}"
  cp "${run_dir}/${pilot_subdir}/metrics.json" "${dst_dir}/metrics.json"
  cp "${run_dir}/${pilot_subdir}/train_events.json" "${dst_dir}/train_events.json"
  cp "${run_dir}/suite_metrics.json" "${dst_dir}/suite_metrics.json"
}

copy_task_artifacts "gsm8k" "control" "${RUN_ROOT}/gsm8k-control" "pilot-A-selected"
copy_task_artifacts "gsm8k" "support-only" "${RUN_ROOT}/gsm8k-support-only" "pilot-I-real"
copy_task_artifacts "gsm8k" "support-context" "${RUN_ROOT}/gsm8k-support-context" "pilot-I-real"
copy_task_artifacts "narrativeqa" "control" "${RUN_ROOT}/narrativeqa-control" "pilot-A-selected"
copy_task_artifacts "narrativeqa" "support-only" "${RUN_ROOT}/narrativeqa-support-only" "pilot-I-real"
copy_task_artifacts "narrativeqa" "support-context" "${RUN_ROOT}/narrativeqa-support-context" "pilot-I-real"
copy_task_artifacts "fever" "control" "${RUN_ROOT}/fever-control" "pilot-A-selected"
copy_task_artifacts "fever" "support-only" "${RUN_ROOT}/fever-support-only" "pilot-I-real"
copy_task_artifacts "fever" "support-context" "${RUN_ROOT}/fever-support-context" "pilot-I-real"

python scripts/update_writer_weaver_summary.py \
  --gsm8k_control_metrics_json "${RUN_ROOT}/gsm8k-control/pilot-A-selected/metrics.json" \
  --gsm8k_support_only_metrics_json "${RUN_ROOT}/gsm8k-support-only/pilot-I-real/metrics.json" \
  --gsm8k_support_context_metrics_json "${RUN_ROOT}/gsm8k-support-context/pilot-I-real/metrics.json" \
  --narrativeqa_control_metrics_json "${RUN_ROOT}/narrativeqa-control/pilot-A-selected/metrics.json" \
  --narrativeqa_support_only_metrics_json "${RUN_ROOT}/narrativeqa-support-only/pilot-I-real/metrics.json" \
  --narrativeqa_support_context_metrics_json "${RUN_ROOT}/narrativeqa-support-context/pilot-I-real/metrics.json" \
  --fever_control_metrics_json "${RUN_ROOT}/fever-control/pilot-A-selected/metrics.json" \
  --fever_support_only_metrics_json "${RUN_ROOT}/fever-support-only/pilot-I-real/metrics.json" \
  --fever_support_context_metrics_json "${RUN_ROOT}/fever-support-context/pilot-I-real/metrics.json" \
  --output_json "${RESULT_ROOT}/w0-summary.json" \
  --output_report "${RESULT_ROOT}/w0-summary.md"

./scripts/publish_review_artifacts.sh

echo "Writer-weaver W0 smoke suite complete"
