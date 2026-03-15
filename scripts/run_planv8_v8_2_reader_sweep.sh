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
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-2-reader-sweep}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-2-reader-sweep}"
PRIMARY_MODEL_DIR="${4:-${PRIMARY_MODEL_DIR_DEFAULT}}"
SELECTED_PROMPTS_PATH="${5:-results/generated/review/planv8-v8-0-${PRIMARY_BACKBONE_KEY}-baselines-oracles/selected-prompt-modes.json}"
V81_SUMMARY_PATH="${6:-results/generated/review/planv8-v8-1-reader-interface-scout/v8-1-summary.json}"

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

python - "${V81_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step not in {"open_v8_2_reader_sweep", "open_v8_2_reader_sweep_last_chance"}:
    raise SystemExit(
        f"V8-1 did not authorize V8-2; recommended_next_step={next_step!r}"
    )
PY

SPLIT_PLAN_JSON="${MANIFEST_ROOT}/v8-2-split-plan.json"
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

PROMPT_EXPORTS="$(
python - "${SELECTED_PROMPTS_PATH}" "${V81_SUMMARY_PATH}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

selected_path = Path(sys.argv[1])
v81_summary_path = Path(sys.argv[2])
fallback = {
    "gsm8k": "q3_gsm8k_nonthink",
    "triviaqa": "q3_trivia_think",
    "fever": "answer_slot_labels",
}

if selected_path.exists():
    payload = json.loads(selected_path.read_text())
    for task_name in fallback:
        task_payload = payload.get(task_name, {})
        prompt = str(task_payload.get("selected_prompt_variant", "")).strip()
        if prompt:
            fallback[task_name] = prompt
elif v81_summary_path.exists():
    payload = json.loads(v81_summary_path.read_text())
    prompt_modes = payload.get("selected_prompt_modes_by_task", {})
    for task_name in fallback:
        prompt = str(prompt_modes.get(task_name, "")).strip()
        if prompt:
            fallback[task_name] = prompt

for env_key, task_name in (
    ("GSM8K_PROMPT", "gsm8k"),
    ("TRIVIAQA_PROMPT", "triviaqa"),
    ("FEVER_PROMPT", "fever"),
):
    print(f"{env_key}={shlex.quote(fallback[task_name])}")
PY
)"
eval "${PROMPT_EXPORTS}"

if [[ -f "${SELECTED_PROMPTS_PATH}" ]]; then
  cp "${SELECTED_PROMPTS_PATH}" "${RESULT_ROOT}/selected-prompt-modes.json"
else
  python - <<'PY' "${RESULT_ROOT}/selected-prompt-modes.json" "${GSM8K_PROMPT}" "${TRIVIAQA_PROMPT}" "${FEVER_PROMPT}"
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
output_path.write_text(
    json.dumps(
        {
            "gsm8k": {"selected_prompt_variant": sys.argv[2]},
            "triviaqa": {"selected_prompt_variant": sys.argv[3]},
            "fever": {"selected_prompt_variant": sys.argv[4]},
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
fi

cp "${V81_SUMMARY_PATH}" "${RESULT_ROOT}/v8-1-summary.reference.json"

materialize_config() {
  local task_name="$1"
  local arm_id="$2"
  local prompt_variant="$3"
  local output_config="$4"
  local support_path="$5"
  local train_path="$6"
  local eval_path="$7"
  local v81_summary_path="$8"
  python scripts/planv8_v8_2_config.py \
    --task_name "${task_name}" \
    --arm_id "${arm_id}" \
    --prompt_variant "${prompt_variant}" \
    --output_config "${output_config}" \
    --support_path "${support_path}" \
    --train_path "${train_path}" \
    --eval_path "${eval_path}" \
    --primary_model_dir "${PRIMARY_MODEL_DIR}" \
    --primary_backbone_name "${PRIMARY_BACKBONE_NAME}" \
    --v81_summary_path "${v81_summary_path}"
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

for task_name in gsm8k triviaqa fever; do
  case "${task_name}" in
    gsm8k)
      prompt_variant="${GSM8K_PROMPT}"
      ;;
    triviaqa)
      prompt_variant="${TRIVIAQA_PROMPT}"
      ;;
    fever)
      prompt_variant="${FEVER_PROMPT}"
      ;;
    *)
      echo "unsupported task ${task_name}" >&2
      exit 1
      ;;
  esac
  materialize_config \
    "${task_name}" \
    "control" \
    "${prompt_variant}" \
    "${CONFIG_ROOT}/${task_name}-control.json" \
    "${DATA_ROOT}/${task_name}/support.jsonl" \
    "${DATA_ROOT}/${task_name}/train.jsonl" \
    "${DATA_ROOT}/${task_name}/eval.jsonl" \
    "${V81_SUMMARY_PATH}"
done

for arm_id in \
  r0_mid8_r32_lr5e5 \
  r1_mid8_r64_lr1e4 \
  r2_mid12_r64_lr1e4 \
  r3_mid12_r64_lr2e4 \
  r4_late8_r32_lr1e4 \
  r5_mid8_r16_lr5e5
do
  for task_name in gsm8k triviaqa fever; do
    case "${task_name}" in
      gsm8k)
        prompt_variant="${GSM8K_PROMPT}"
        ;;
      triviaqa)
        prompt_variant="${TRIVIAQA_PROMPT}"
        ;;
      fever)
        prompt_variant="${FEVER_PROMPT}"
        ;;
    esac
    materialize_config \
      "${task_name}" \
      "${arm_id}" \
      "${prompt_variant}" \
      "${CONFIG_ROOT}/${task_name}-${arm_id}.json" \
      "${DATA_ROOT}/${task_name}/support.jsonl" \
      "${DATA_ROOT}/${task_name}/train.jsonl" \
      "${DATA_ROOT}/${task_name}/eval.jsonl" \
      "${V81_SUMMARY_PATH}"
  done
done

seed_offset=0
for task_name in gsm8k triviaqa fever; do
  run_single_pilot \
    "${CONFIG_ROOT}/${task_name}-control.json" \
    "$((BASE_SEED + seed_offset))" \
    "${RUN_ROOT}/control-${task_name}"
  copy_run_artifacts \
    "${RESULT_ROOT}/control/${task_name}" \
    "${RUN_ROOT}/control-${task_name}"
  seed_offset=$((seed_offset + 10))
done

for arm_id in \
  r0_mid8_r32_lr5e5 \
  r1_mid8_r64_lr1e4 \
  r2_mid12_r64_lr1e4 \
  r3_mid12_r64_lr2e4 \
  r4_late8_r32_lr1e4 \
  r5_mid8_r16_lr5e5
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

python scripts/update_planv8_v8_2_summary.py \
  --result_root "${RESULT_ROOT}" \
  --v81_summary "${V81_SUMMARY_PATH}" \
  --output_json "${RESULT_ROOT}/v8-2-summary.json" \
  --output_report "${RESULT_ROOT}/v8-2-summary.md"

bash scripts/publish_review_artifacts.sh

echo "PLANv8 V8-2 reader sweep complete."
