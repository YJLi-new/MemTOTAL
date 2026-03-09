#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-26291}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/tl-bridge-multitask-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/tl-bridge-multitask-qwen25}"
FEVER_SUMMARY_INPUT="${4:-results/generated/review/tl-micro-lora-fever-qwen25/v2-summary.json}"
RESUME_STAGE_B_ROOT="${5:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
WARM_START_INPUT="${6:-/root/autodl-tmp/runs/verify/tl-micro-lora-fever-qwen25/v2-microlora-r2-late3-partition-none-linear-h4-k4/phase2-selected/pilot-I-real/checkpoint.pt}"

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

FEVER_SUMMARY="$(resolve_path "${FEVER_SUMMARY_INPUT}")"
WARM_START="$(resolve_path "${WARM_START_INPUT}")"

if [[ ! -f "${FEVER_SUMMARY}" ]]; then
  echo "missing required input: ${FEVER_SUMMARY}" >&2
  exit 1
fi
if [[ ! -f "${WARM_START}" ]]; then
  echo "missing required input: ${WARM_START}" >&2
  exit 1
fi

DATA_ROOT="${RUN_ROOT}/materialized-datasets"
mkdir -p "${DATA_ROOT}"

python - "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))


data_root = Path(sys.argv[1]).resolve()
gsm8k_source = Path("data/benchmarks/materialized/gsm8k/eval-real-smoke8.jsonl").resolve()
narrativeqa_source = Path("data/benchmarks/materialized/narrativeqa/eval-real-smoke4.jsonl").resolve()

gsm8k_rows = load_jsonl(gsm8k_source)
narrativeqa_rows = load_jsonl(narrativeqa_source)

if len(gsm8k_rows) < 8:
    raise ValueError(f"GSM8K real-smoke source requires at least 8 rows, got {len(gsm8k_rows)}.")
if len(narrativeqa_rows) < 4:
    raise ValueError(f"NarrativeQA real-smoke source requires at least 4 rows, got {len(narrativeqa_rows)}.")

gsm8k_splits = {
    "support": gsm8k_rows[:4],
    "train": gsm8k_rows[4:6],
    "eval": gsm8k_rows[6:8],
}
narrativeqa_splits = {
    "support": narrativeqa_rows[:2],
    "train": narrativeqa_rows[2:3],
    "eval": narrativeqa_rows[3:4],
}

manifest = {"gsm8k": {}, "narrativeqa": {}}
for task_name, splits in {
    "gsm8k": gsm8k_splits,
    "narrativeqa": narrativeqa_splits,
}.items():
    task_root = data_root / task_name
    task_root.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        split_path = task_root / f"{split_name}.jsonl"
        write_jsonl(split_path, rows)
        manifest[task_name][split_name] = {
            "path": str(split_path),
            "rows": len(rows),
            "ids": [str(row.get("id", "")) for row in rows],
        }

(data_root / "split-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

MATERIALIZED_CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${MATERIALIZED_CONFIG_ROOT}"

materialize_config() {
  local src_config="$1"
  local output_config="$2"
  local support_path="$3"
  local train_path="$4"
  local eval_path="$5"
  python - "${src_config}" "${output_config}" "${WARM_START}" "${support_path}" "${train_path}" "${eval_path}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

source_config = sys.argv[1]
output_config = Path(sys.argv[2])
warm_start = str(Path(sys.argv[3]).resolve())
support_path = str(Path(sys.argv[4]).resolve())
train_path = str(Path(sys.argv[5]).resolve())
eval_path = str(Path(sys.argv[6]).resolve())

config = load_config(source_config)
config.setdefault("runtime", {})
config["runtime"]["pilot_init_checkpoint_path"] = warm_start
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
  "configs/exp/tl_multitask_qwen25_control_narrativeqa_real_smoke_template.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_multitask_qwen25_control_narrativeqa_real_smoke.json" \
  "${DATA_ROOT}/narrativeqa/support.jsonl" \
  "${DATA_ROOT}/narrativeqa/train.jsonl" \
  "${DATA_ROOT}/narrativeqa/eval.jsonl"
materialize_config \
  "configs/exp/tl_multitask_qwen25_microlora_r2_late3_narrativeqa_real_smoke_template.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_multitask_qwen25_microlora_r2_late3_narrativeqa_real_smoke.json" \
  "${DATA_ROOT}/narrativeqa/support.jsonl" \
  "${DATA_ROOT}/narrativeqa/train.jsonl" \
  "${DATA_ROOT}/narrativeqa/eval.jsonl"
materialize_config \
  "configs/exp/tl_multitask_qwen25_control_gsm8k_real_smoke_template.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_multitask_qwen25_control_gsm8k_real_smoke.json" \
  "${DATA_ROOT}/gsm8k/support.jsonl" \
  "${DATA_ROOT}/gsm8k/train.jsonl" \
  "${DATA_ROOT}/gsm8k/eval.jsonl"
materialize_config \
  "configs/exp/tl_multitask_qwen25_microlora_r2_late3_gsm8k_real_smoke_template.yaml" \
  "${MATERIALIZED_CONFIG_ROOT}/tl_multitask_qwen25_microlora_r2_late3_gsm8k_real_smoke.json" \
  "${DATA_ROOT}/gsm8k/support.jsonl" \
  "${DATA_ROOT}/gsm8k/train.jsonl" \
  "${DATA_ROOT}/gsm8k/eval.jsonl"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

ensure_suite_complete() {
  local suite_config="$1"
  local run_seed="$2"
  local run_dir="$3"
  local arm_spec="$4"
  local lock_fd
  mkdir -p "${run_dir}"
  exec {lock_fd}> "${run_dir}/.suite.lock"
  flock "${lock_fd}"
  if [[ ! -f "${run_dir}/suite_metrics.json" ]]; then
    python scripts/run_m4_selected_shared_injection_suite.py \
      --config "${suite_config}" \
      --resume "${RESUME_STAGE_B_ROOT}" \
      --output_root "${run_dir}" \
      --seed "${run_seed}" \
      --prompt-variant task_native \
      --support-serialization example_blocks_raw8 \
      --arm-spec "${arm_spec}"
  fi
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

ensure_suite_complete \
  "${MATERIALIZED_CONFIG_ROOT}/tl_multitask_qwen25_control_narrativeqa_real_smoke.json" \
  "${BASE_SEED}" \
  "${RUN_ROOT}/narrativeqa-control" \
  "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete \
  "${MATERIALIZED_CONFIG_ROOT}/tl_multitask_qwen25_microlora_r2_late3_narrativeqa_real_smoke.json" \
  "$((BASE_SEED + 1000))" \
  "${RUN_ROOT}/narrativeqa-microlora-r2-late3" \
  "pilot-I-real:I_real:injected:real:10"
ensure_suite_complete \
  "${MATERIALIZED_CONFIG_ROOT}/tl_multitask_qwen25_control_gsm8k_real_smoke.json" \
  "$((BASE_SEED + 2000))" \
  "${RUN_ROOT}/gsm8k-control" \
  "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete \
  "${MATERIALIZED_CONFIG_ROOT}/tl_multitask_qwen25_microlora_r2_late3_gsm8k_real_smoke.json" \
  "$((BASE_SEED + 3000))" \
  "${RUN_ROOT}/gsm8k-microlora-r2-late3" \
  "pilot-I-real:I_real:injected:real:10"

mkdir -p \
  "${RESULT_ROOT}/narrativeqa/control" \
  "${RESULT_ROOT}/narrativeqa/microlora-r2-late3" \
  "${RESULT_ROOT}/gsm8k/control" \
  "${RESULT_ROOT}/gsm8k/microlora-r2-late3"

cp "${RUN_ROOT}/narrativeqa-control/pilot-A-selected/metrics.json" "${RESULT_ROOT}/narrativeqa/control/metrics.json"
cp "${RUN_ROOT}/narrativeqa-control/pilot-A-selected/train_events.json" "${RESULT_ROOT}/narrativeqa/control/train_events.json"
cp "${RUN_ROOT}/narrativeqa-control/suite_metrics.json" "${RESULT_ROOT}/narrativeqa/control/suite_metrics.json"
cp "${RUN_ROOT}/narrativeqa-microlora-r2-late3/pilot-I-real/metrics.json" "${RESULT_ROOT}/narrativeqa/microlora-r2-late3/metrics.json"
cp "${RUN_ROOT}/narrativeqa-microlora-r2-late3/pilot-I-real/train_events.json" "${RESULT_ROOT}/narrativeqa/microlora-r2-late3/train_events.json"
cp "${RUN_ROOT}/narrativeqa-microlora-r2-late3/suite_metrics.json" "${RESULT_ROOT}/narrativeqa/microlora-r2-late3/suite_metrics.json"
cp "${RUN_ROOT}/gsm8k-control/pilot-A-selected/metrics.json" "${RESULT_ROOT}/gsm8k/control/metrics.json"
cp "${RUN_ROOT}/gsm8k-control/pilot-A-selected/train_events.json" "${RESULT_ROOT}/gsm8k/control/train_events.json"
cp "${RUN_ROOT}/gsm8k-control/suite_metrics.json" "${RESULT_ROOT}/gsm8k/control/suite_metrics.json"
cp "${RUN_ROOT}/gsm8k-microlora-r2-late3/pilot-I-real/metrics.json" "${RESULT_ROOT}/gsm8k/microlora-r2-late3/metrics.json"
cp "${RUN_ROOT}/gsm8k-microlora-r2-late3/pilot-I-real/train_events.json" "${RESULT_ROOT}/gsm8k/microlora-r2-late3/train_events.json"
cp "${RUN_ROOT}/gsm8k-microlora-r2-late3/suite_metrics.json" "${RESULT_ROOT}/gsm8k/microlora-r2-late3/suite_metrics.json"

python scripts/update_tl_multitask_summary.py \
  --fever_summary_json "${FEVER_SUMMARY}" \
  --narrativeqa_control_metrics_json "${RUN_ROOT}/narrativeqa-control/pilot-A-selected/metrics.json" \
  --narrativeqa_bridge_metrics_json "${RUN_ROOT}/narrativeqa-microlora-r2-late3/pilot-I-real/metrics.json" \
  --gsm8k_control_metrics_json "${RUN_ROOT}/gsm8k-control/pilot-A-selected/metrics.json" \
  --gsm8k_bridge_metrics_json "${RUN_ROOT}/gsm8k-microlora-r2-late3/pilot-I-real/metrics.json" \
  --output_json "${RESULT_ROOT}/v4-summary.json" \
  --output_report "${RESULT_ROOT}/v4-summary.md"

./scripts/publish_review_artifacts.sh

echo "TL bridge multitask V4 complete"
