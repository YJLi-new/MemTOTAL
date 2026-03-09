#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-58431}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/writer-circuit-opening-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/writer-circuit-opening-qwen25}"
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
config["task"]["support_dataset_path"] = support_path
config["task"]["train_dataset_path"] = train_path
config["task"]["train_support_dataset_path"] = support_path
config["task"]["dataset_path"] = eval_path
config["task"]["support_lookup_dataset_paths"] = []
config["task"]["train_support_episode_bank_path"] = ""
config["task"]["pilot_split"] = str(config["task"].get("smoke_subset", config["task"].get("split", "eval")))
config.setdefault("method", {})
config.setdefault("runtime", {})
config["runtime"]["pilot_prefix_source_mode"] = "source_stub"
config["runtime"]["pilot_deep_prefix_init_mode"] = "kv_stat_match"
if variant == "p1a":
    config["runtime"]["pilot_arm_alias"] = "I_circuit_p1a"
    config["method"]["receiver_lora"] = {
        "enabled": False,
        "target_layers": [],
        "target_modules": ["k_proj", "v_proj"],
        "rank": 0,
        "alpha": 4.0,
        "dropout": 0.0,
    }
elif variant == "p2a":
    config["runtime"]["pilot_arm_alias"] = "I_circuit_p2a"
    config["method"]["receiver_lora"] = {
        "enabled": True,
        "target_layers": [0, 1, 2, 3],
        "target_modules": ["k_proj", "v_proj"],
        "rank": 2,
        "alpha": 4.0,
        "dropout": 0.0,
    }
else:
    raise ValueError(f"unsupported variant: {variant}")
output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

materialize_config \
  "configs/exp/writer_circuit_g1_source_stub_gsm8k_template.yaml" \
  "${CONFIG_ROOT}/gsm8k-p1a.json" \
  "${DATA_ROOT}/gsm8k/support.jsonl" \
  "${DATA_ROOT}/gsm8k/train.jsonl" \
  "${DATA_ROOT}/gsm8k/eval.jsonl" \
  "p1a"
materialize_config \
  "configs/exp/writer_circuit_g1_source_stub_gsm8k_template.yaml" \
  "${CONFIG_ROOT}/gsm8k-p2a.json" \
  "${DATA_ROOT}/gsm8k/support.jsonl" \
  "${DATA_ROOT}/gsm8k/train.jsonl" \
  "${DATA_ROOT}/gsm8k/eval.jsonl" \
  "p2a"
materialize_config \
  "configs/exp/writer_circuit_g1_source_stub_narrativeqa_template.yaml" \
  "${CONFIG_ROOT}/narrativeqa-p1a.json" \
  "${DATA_ROOT}/narrativeqa/support.jsonl" \
  "${DATA_ROOT}/narrativeqa/train.jsonl" \
  "${DATA_ROOT}/narrativeqa/eval.jsonl" \
  "p1a"
materialize_config \
  "configs/exp/writer_circuit_g1_source_stub_narrativeqa_template.yaml" \
  "${CONFIG_ROOT}/narrativeqa-p2a.json" \
  "${DATA_ROOT}/narrativeqa/support.jsonl" \
  "${DATA_ROOT}/narrativeqa/train.jsonl" \
  "${DATA_ROOT}/narrativeqa/eval.jsonl" \
  "p2a"
materialize_config \
  "configs/exp/writer_circuit_g1_source_stub_fever_template.yaml" \
  "${CONFIG_ROOT}/fever-p1a.json" \
  "${DATA_ROOT}/fever/support.jsonl" \
  "${DATA_ROOT}/fever/train.jsonl" \
  "${DATA_ROOT}/fever/eval.jsonl" \
  "p1a"
materialize_config \
  "configs/exp/writer_circuit_g1_source_stub_fever_template.yaml" \
  "${CONFIG_ROOT}/fever-p2a.json" \
  "${DATA_ROOT}/fever/support.jsonl" \
  "${DATA_ROOT}/fever/train.jsonl" \
  "${DATA_ROOT}/fever/eval.jsonl" \
  "p2a"

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

ensure_suite_complete "${CONFIG_ROOT}/gsm8k-p1a.json" "${BASE_SEED}" \
  "${RUN_ROOT}/gsm8k-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/gsm8k-p1a.json" "$((BASE_SEED + 1000))" \
  "${RUN_ROOT}/gsm8k-p1a" "pilot-I-source-stub:I_p1a:injected:real:0"
ensure_suite_complete "${CONFIG_ROOT}/gsm8k-p2a.json" "$((BASE_SEED + 2000))" \
  "${RUN_ROOT}/gsm8k-p2a" "pilot-I-source-stub-lora:I_p2a:injected:real:0"

ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-p1a.json" "$((BASE_SEED + 3000))" \
  "${RUN_ROOT}/narrativeqa-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-p1a.json" "$((BASE_SEED + 4000))" \
  "${RUN_ROOT}/narrativeqa-p1a" "pilot-I-source-stub:I_p1a:injected:real:0"
ensure_suite_complete "${CONFIG_ROOT}/narrativeqa-p2a.json" "$((BASE_SEED + 5000))" \
  "${RUN_ROOT}/narrativeqa-p2a" "pilot-I-source-stub-lora:I_p2a:injected:real:0"

ensure_suite_complete "${CONFIG_ROOT}/fever-p1a.json" "$((BASE_SEED + 6000))" \
  "${RUN_ROOT}/fever-control" "pilot-A-selected:A:base_only:real:0"
ensure_suite_complete "${CONFIG_ROOT}/fever-p1a.json" "$((BASE_SEED + 7000))" \
  "${RUN_ROOT}/fever-p1a" "pilot-I-source-stub:I_p1a:injected:real:0"
ensure_suite_complete "${CONFIG_ROOT}/fever-p2a.json" "$((BASE_SEED + 8000))" \
  "${RUN_ROOT}/fever-p2a" "pilot-I-source-stub-lora:I_p2a:injected:real:0"

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
copy_task_artifacts "gsm8k" "p1a" "${RUN_ROOT}/gsm8k-p1a" "pilot-I-source-stub"
copy_task_artifacts "gsm8k" "p2a" "${RUN_ROOT}/gsm8k-p2a" "pilot-I-source-stub-lora"
copy_task_artifacts "narrativeqa" "control" "${RUN_ROOT}/narrativeqa-control" "pilot-A-selected"
copy_task_artifacts "narrativeqa" "p1a" "${RUN_ROOT}/narrativeqa-p1a" "pilot-I-source-stub"
copy_task_artifacts "narrativeqa" "p2a" "${RUN_ROOT}/narrativeqa-p2a" "pilot-I-source-stub-lora"
copy_task_artifacts "fever" "control" "${RUN_ROOT}/fever-control" "pilot-A-selected"
copy_task_artifacts "fever" "p1a" "${RUN_ROOT}/fever-p1a" "pilot-I-source-stub"
copy_task_artifacts "fever" "p2a" "${RUN_ROOT}/fever-p2a" "pilot-I-source-stub-lora"

python scripts/update_writer_circuit_opening_summary.py \
  --gsm8k_control_metrics_json "${RUN_ROOT}/gsm8k-control/pilot-A-selected/metrics.json" \
  --gsm8k_p1a_metrics_json "${RUN_ROOT}/gsm8k-p1a/pilot-I-source-stub/metrics.json" \
  --gsm8k_p2a_metrics_json "${RUN_ROOT}/gsm8k-p2a/pilot-I-source-stub-lora/metrics.json" \
  --narrativeqa_control_metrics_json "${RUN_ROOT}/narrativeqa-control/pilot-A-selected/metrics.json" \
  --narrativeqa_p1a_metrics_json "${RUN_ROOT}/narrativeqa-p1a/pilot-I-source-stub/metrics.json" \
  --narrativeqa_p2a_metrics_json "${RUN_ROOT}/narrativeqa-p2a/pilot-I-source-stub-lora/metrics.json" \
  --fever_control_metrics_json "${RUN_ROOT}/fever-control/pilot-A-selected/metrics.json" \
  --fever_p1a_metrics_json "${RUN_ROOT}/fever-p1a/pilot-I-source-stub/metrics.json" \
  --fever_p2a_metrics_json "${RUN_ROOT}/fever-p2a/pilot-I-source-stub-lora/metrics.json" \
  --output_json "${RESULT_ROOT}/writer-circuit-opening-summary.json" \
  --output_report "${RESULT_ROOT}/writer-circuit-opening-summary.md"

./scripts/publish_review_artifacts.sh

echo "Writer circuit opening P1a/P2a suite complete"
