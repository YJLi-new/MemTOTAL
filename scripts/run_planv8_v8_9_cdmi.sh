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
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-9-cdmi}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-9-cdmi}"
PRIMARY_MODEL_DIR="${4:-${PRIMARY_MODEL_DIR_DEFAULT}}"
V80_SUMMARY_PATH="${5:-results/generated/review/planv8-v8-0-${PRIMARY_BACKBONE_KEY}-baselines-oracles/v8-0-summary.json}"
SELECTED_PROMPTS_PATH="${6:-results/generated/review/planv8-v8-0-${PRIMARY_BACKBONE_KEY}-baselines-oracles/selected-prompt-modes.json}"
V88_RESULT_ROOT="${7:-/root/autodl-tmp/results/generated/planv8-v8-8-multiseed-confirmation-${PRIMARY_BACKBONE_KEY}}"

V88_SUMMARY_PATH="${V88_RESULT_ROOT}/v8-8-summary.json"
SELECTION_MANIFEST_PATH="${V88_RESULT_ROOT}/selection-manifest.json"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PLANV8_EXPERIMENT_PREFIX="${PLANV8_EXPERIMENT_PREFIX:-planv8}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}" "${HF_HOME}"

CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
DATA_ROOT="${RUN_ROOT}/materialized-datasets"
MANIFEST_ROOT="${RUN_ROOT}/materialized-manifests"
mkdir -p "${CONFIG_ROOT}" "${DATA_ROOT}" "${MANIFEST_ROOT}"

python - "${V88_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step != "open_v8_9_cdmi":
    raise SystemExit(
        f"V8-8 did not authorize V8-9; recommended_next_step={next_step!r}"
    )
PY

bash "${PRIMARY_PREP_SCRIPT}" \
  "${PRIMARY_MODEL_ID}" \
  "${PRIMARY_MODEL_DIR}" \
  "${HF_HOME}"

CDMI_MANIFEST="${MANIFEST_ROOT}/cdmi-manifest.json"
python - "${V88_SUMMARY_PATH}" "${SELECTION_MANIFEST_PATH}" "${SELECTED_PROMPTS_PATH}" "${DATA_ROOT}" "${CDMI_MANIFEST}" <<'PY'
import copy
import json
import sys
from pathlib import Path

from memtotal.tasks.registry import load_task_dataset
from memtotal.training.m4_shared_injection import _resolve_prompt_text


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def render_rows(config: dict, dataset_path: str, prompt_variant: str) -> list[dict]:
    config_copy = copy.deepcopy(config)
    config_copy["task"]["dataset_path"] = dataset_path
    rows = load_task_dataset(config_copy)
    rendered = []
    for row in rows:
        row_copy = dict(row)
        row_copy["segment"] = _resolve_prompt_text(row_copy, prompt_variant=prompt_variant)
        row_copy["cdmi_source_prompt_variant"] = prompt_variant
        rendered.append(row_copy)
    return rendered


def interleave_rows(*groups: list[dict]) -> list[dict]:
    combined: list[dict] = []
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                combined.append(dict(group[index]))
    return combined


v88_summary = load_json(Path(sys.argv[1]))
selection_manifest = load_json(Path(sys.argv[2]))
selected_prompts = load_json(Path(sys.argv[3])) if Path(sys.argv[3]).exists() else {}
data_root = Path(sys.argv[4]).resolve()
output_path = Path(sys.argv[5])

best_variant_id = str(v88_summary.get("best_confirmed_variant_id", "")).strip()
if not best_variant_id:
    raise SystemExit("V8-8 summary is missing best_confirmed_variant_id.")

selected_variant = None
for row in selection_manifest.get("promoted_variants", []):
    if str(row.get("variant_id", "")).strip() == best_variant_id:
        selected_variant = dict(row)
        break
if selected_variant is None:
    raise SystemExit(f"V8-8 selection manifest is missing variant {best_variant_id!r}.")

source_run_root = Path(str(selected_variant.get("source_run_root", "")).strip()).resolve()
source_arm_id = str(selected_variant.get("arm_id", "")).strip()
if not source_arm_id:
    raise SystemExit("Selected V8-8 variant is missing arm_id.")

base_configs = {
    "gsm8k": source_run_root / "materialized-configs" / f"gsm8k-{source_arm_id}.json",
    "triviaqa": source_run_root / "materialized-configs" / f"triviaqa-{source_arm_id}.json",
}
for task_name, path in base_configs.items():
    if not path.exists():
        raise SystemExit(f"Missing source config for {task_name}: {path}")

task_payload: dict[str, dict] = {}
for task_name in ("gsm8k", "triviaqa"):
    config = load_json(base_configs[task_name])
    prompt_variant = str(
        config.get("runtime", {}).get("pilot_prompt_variant")
        or selected_prompts.get(task_name, {}).get("selected_prompt_variant")
        or "task_native"
    ).strip()
    support_rows = render_rows(config, str(config["task"]["support_dataset_path"]), prompt_variant)
    train_rows = render_rows(config, str(config["task"]["train_dataset_path"]), prompt_variant)
    eval_rows = render_rows(config, str(config["task"]["dataset_path"]), prompt_variant)

    task_root = data_root / task_name
    support_path = task_root / "support.jsonl"
    train_path = task_root / "train.jsonl"
    eval_path = task_root / "eval.jsonl"
    write_jsonl(support_path, support_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    task_payload[task_name] = {
        "base_config_path": str(base_configs[task_name]),
        "source_prompt_variant": prompt_variant,
        "support_path": str(support_path),
        "support_rows": len(support_rows),
        "train_path": str(train_path),
        "train_rows": len(train_rows),
        "eval_path": str(eval_path),
        "eval_rows": len(eval_rows),
    }

joint_root = data_root / "joint"
joint_support_rows = interleave_rows(
    render_rows(load_json(base_configs["gsm8k"]), task_payload["gsm8k"]["support_path"], "task_native"),
    render_rows(load_json(base_configs["triviaqa"]), task_payload["triviaqa"]["support_path"], "task_native"),
)
joint_train_rows = interleave_rows(
    render_rows(load_json(base_configs["gsm8k"]), task_payload["gsm8k"]["train_path"], "task_native"),
    render_rows(load_json(base_configs["triviaqa"]), task_payload["triviaqa"]["train_path"], "task_native"),
)
joint_support_path = joint_root / "support.jsonl"
joint_train_path = joint_root / "train.jsonl"
write_jsonl(joint_support_path, joint_support_rows)
write_jsonl(joint_train_path, joint_train_rows)

payload = {
    "phase": "V8-9",
    "best_confirmed_variant_id": best_variant_id,
    "source_phase": str(selected_variant.get("source_phase", "")).strip(),
    "source_run_root": str(source_run_root),
    "source_arm_id": source_arm_id,
    "source_interface_family": str(selected_variant.get("interface_family", "")).strip(),
    "source_bridge_family": str(selected_variant.get("bridge_family", "")).strip(),
    "source_auxiliary_family": str(selected_variant.get("auxiliary_family", "")).strip(),
    "source_prompt_variants": {
        task_name: task_payload[task_name]["source_prompt_variant"]
        for task_name in ("gsm8k", "triviaqa")
    },
    "base_configs": {
        task_name: task_payload[task_name]["base_config_path"]
        for task_name in ("gsm8k", "triviaqa")
    },
    "splits": {
        "gsm8k": {
            "support": {
                "path": task_payload["gsm8k"]["support_path"],
                "rows": task_payload["gsm8k"]["support_rows"],
            },
            "train": {
                "path": task_payload["gsm8k"]["train_path"],
                "rows": task_payload["gsm8k"]["train_rows"],
            },
            "eval": {
                "path": task_payload["gsm8k"]["eval_path"],
                "rows": task_payload["gsm8k"]["eval_rows"],
            },
        },
        "triviaqa": {
            "support": {
                "path": task_payload["triviaqa"]["support_path"],
                "rows": task_payload["triviaqa"]["support_rows"],
            },
            "train": {
                "path": task_payload["triviaqa"]["train_path"],
                "rows": task_payload["triviaqa"]["train_rows"],
            },
            "eval": {
                "path": task_payload["triviaqa"]["eval_path"],
                "rows": task_payload["triviaqa"]["eval_rows"],
            },
        },
        "joint": {
            "support": {
                "path": str(joint_support_path),
                "rows": len(joint_support_rows),
            },
            "train": {
                "path": str(joint_train_path),
                "rows": len(joint_train_rows),
            },
        },
    },
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

cp "${V80_SUMMARY_PATH}" "${RESULT_ROOT}/v8-0-summary.reference.json"
cp "${V88_SUMMARY_PATH}" "${RESULT_ROOT}/v8-8-summary.reference.json"
cp "${SELECTION_MANIFEST_PATH}" "${RESULT_ROOT}/selection-manifest.json"
cp "${CDMI_MANIFEST}" "${RESULT_ROOT}/cdmi-manifest.json"
if [[ -f "${SELECTED_PROMPTS_PATH}" ]]; then
  cp "${SELECTED_PROMPTS_PATH}" "${RESULT_ROOT}/selected-prompt-modes.json"
fi

mapfile -t MANIFEST_VALUES < <(
  python - "${CDMI_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(str(payload.get("best_confirmed_variant_id", "")).strip())
print(str(payload.get("source_phase", "")).strip())
print(str(payload.get("source_arm_id", "")).strip())
print(str(payload.get("source_interface_family", "")).strip())
print(str(payload.get("source_bridge_family", "")).strip())
print(str(payload.get("source_auxiliary_family", "")).strip())
print(str(payload.get("base_configs", {}).get("gsm8k", "")).strip())
print(str(payload.get("base_configs", {}).get("triviaqa", "")).strip())
print(str(payload.get("source_prompt_variants", {}).get("gsm8k", "")).strip())
print(str(payload.get("source_prompt_variants", {}).get("triviaqa", "")).strip())
print(str(payload.get("splits", {}).get("gsm8k", {}).get("support", {}).get("path", "")).strip())
print(str(payload.get("splits", {}).get("gsm8k", {}).get("train", {}).get("path", "")).strip())
print(str(payload.get("splits", {}).get("gsm8k", {}).get("eval", {}).get("path", "")).strip())
print(str(payload.get("splits", {}).get("triviaqa", {}).get("support", {}).get("path", "")).strip())
print(str(payload.get("splits", {}).get("triviaqa", {}).get("train", {}).get("path", "")).strip())
print(str(payload.get("splits", {}).get("triviaqa", {}).get("eval", {}).get("path", "")).strip())
print(str(payload.get("splits", {}).get("joint", {}).get("support", {}).get("path", "")).strip())
print(str(payload.get("splits", {}).get("joint", {}).get("train", {}).get("path", "")).strip())
PY
)

BEST_VARIANT_ID="${MANIFEST_VALUES[0]}"
SOURCE_PHASE="${MANIFEST_VALUES[1]}"
SOURCE_ARM_ID="${MANIFEST_VALUES[2]}"
SOURCE_INTERFACE_FAMILY="${MANIFEST_VALUES[3]}"
SOURCE_BRIDGE_FAMILY="${MANIFEST_VALUES[4]}"
SOURCE_AUX_FAMILY="${MANIFEST_VALUES[5]}"
GSM8K_BASE_CONFIG="${MANIFEST_VALUES[6]}"
TRIVIA_BASE_CONFIG="${MANIFEST_VALUES[7]}"
GSM8K_PROMPT_VARIANT="${MANIFEST_VALUES[8]}"
TRIVIA_PROMPT_VARIANT="${MANIFEST_VALUES[9]}"
GSM8K_SUPPORT_PATH="${MANIFEST_VALUES[10]}"
GSM8K_TRAIN_PATH="${MANIFEST_VALUES[11]}"
GSM8K_EVAL_PATH="${MANIFEST_VALUES[12]}"
TRIVIA_SUPPORT_PATH="${MANIFEST_VALUES[13]}"
TRIVIA_TRAIN_PATH="${MANIFEST_VALUES[14]}"
TRIVIA_EVAL_PATH="${MANIFEST_VALUES[15]}"
JOINT_SUPPORT_PATH="${MANIFEST_VALUES[16]}"
JOINT_TRAIN_PATH="${MANIFEST_VALUES[17]}"

JOINT_CHECKPOINT_PATH="${RUN_ROOT}/c2_joint_math/checkpoint.pt"

materialize_config() {
  local base_config="$1"
  local condition_id="$2"
  local eval_task_name="$3"
  local support_path="$4"
  local train_path="$5"
  local eval_path="$6"
  local source_prompt_variant="$7"
  local checkpoint_path="${8:-}"
  python scripts/planv8_v8_9_config.py \
    --base_config "${base_config}" \
    --output_config "${CONFIG_ROOT}/${condition_id}.json" \
    --condition_id "${condition_id}" \
    --eval_task_name "${eval_task_name}" \
    --support_path "${support_path}" \
    --train_path "${train_path}" \
    --eval_path "${eval_path}" \
    --source_variant_id "${BEST_VARIANT_ID}" \
    --source_phase "${SOURCE_PHASE}" \
    --source_arm_id "${SOURCE_ARM_ID}" \
    --source_interface_family "${SOURCE_INTERFACE_FAMILY}" \
    --source_bridge_family "${SOURCE_BRIDGE_FAMILY}" \
    --source_auxiliary_family "${SOURCE_AUX_FAMILY}" \
    --source_prompt_variant "${source_prompt_variant}" \
    --checkpoint_path "${checkpoint_path}" \
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

materialize_config \
  "${GSM8K_BASE_CONFIG}" \
  "c0_math_self" \
  "gsm8k" \
  "${GSM8K_SUPPORT_PATH}" \
  "${GSM8K_TRAIN_PATH}" \
  "${GSM8K_EVAL_PATH}" \
  "${GSM8K_PROMPT_VARIANT}"
materialize_config \
  "${TRIVIA_BASE_CONFIG}" \
  "c1_trivia_self" \
  "triviaqa" \
  "${TRIVIA_SUPPORT_PATH}" \
  "${TRIVIA_TRAIN_PATH}" \
  "${TRIVIA_EVAL_PATH}" \
  "${TRIVIA_PROMPT_VARIANT}"
materialize_config \
  "${GSM8K_BASE_CONFIG}" \
  "c2_joint_math" \
  "gsm8k" \
  "${JOINT_SUPPORT_PATH}" \
  "${JOINT_TRAIN_PATH}" \
  "${GSM8K_EVAL_PATH}" \
  "${GSM8K_PROMPT_VARIANT}"
materialize_config \
  "${TRIVIA_BASE_CONFIG}" \
  "c3_joint_trivia" \
  "triviaqa" \
  "${JOINT_SUPPORT_PATH}" \
  "${JOINT_TRAIN_PATH}" \
  "${TRIVIA_EVAL_PATH}" \
  "${TRIVIA_PROMPT_VARIANT}" \
  "${JOINT_CHECKPOINT_PATH}"
materialize_config \
  "${TRIVIA_BASE_CONFIG}" \
  "c4_math_support_on_trivia" \
  "triviaqa" \
  "${GSM8K_SUPPORT_PATH}" \
  "${GSM8K_TRAIN_PATH}" \
  "${TRIVIA_EVAL_PATH}" \
  "${TRIVIA_PROMPT_VARIANT}" \
  "${JOINT_CHECKPOINT_PATH}"
materialize_config \
  "${GSM8K_BASE_CONFIG}" \
  "c5_trivia_support_on_math" \
  "gsm8k" \
  "${TRIVIA_SUPPORT_PATH}" \
  "${TRIVIA_TRAIN_PATH}" \
  "${GSM8K_EVAL_PATH}" \
  "${GSM8K_PROMPT_VARIANT}" \
  "${JOINT_CHECKPOINT_PATH}"

run_single_pilot "${CONFIG_ROOT}/c0_math_self.json" "${BASE_SEED}" "${RUN_ROOT}/c0_math_self"
copy_run_artifacts "${RESULT_ROOT}/c0_math_self" "${RUN_ROOT}/c0_math_self"

run_single_pilot "${CONFIG_ROOT}/c1_trivia_self.json" "$((BASE_SEED + 1))" "${RUN_ROOT}/c1_trivia_self"
copy_run_artifacts "${RESULT_ROOT}/c1_trivia_self" "${RUN_ROOT}/c1_trivia_self"

run_single_pilot "${CONFIG_ROOT}/c2_joint_math.json" "$((BASE_SEED + 2))" "${RUN_ROOT}/c2_joint_math"
copy_run_artifacts "${RESULT_ROOT}/c2_joint_math" "${RUN_ROOT}/c2_joint_math"

if [[ ! -f "${JOINT_CHECKPOINT_PATH}" ]]; then
  echo "missing required joint checkpoint: ${JOINT_CHECKPOINT_PATH}" >&2
  exit 1
fi

run_single_pilot "${CONFIG_ROOT}/c3_joint_trivia.json" "$((BASE_SEED + 3))" "${RUN_ROOT}/c3_joint_trivia"
copy_run_artifacts "${RESULT_ROOT}/c3_joint_trivia" "${RUN_ROOT}/c3_joint_trivia"

run_single_pilot "${CONFIG_ROOT}/c4_math_support_on_trivia.json" "$((BASE_SEED + 4))" "${RUN_ROOT}/c4_math_support_on_trivia"
copy_run_artifacts "${RESULT_ROOT}/c4_math_support_on_trivia" "${RUN_ROOT}/c4_math_support_on_trivia"

run_single_pilot "${CONFIG_ROOT}/c5_trivia_support_on_math.json" "$((BASE_SEED + 5))" "${RUN_ROOT}/c5_trivia_support_on_math"
copy_run_artifacts "${RESULT_ROOT}/c5_trivia_support_on_math" "${RUN_ROOT}/c5_trivia_support_on_math"

python scripts/update_planv8_v8_9_summary.py \
  --result_root "${RESULT_ROOT}" \
  --manifest "${CDMI_MANIFEST}" \
  --v80_summary "${V80_SUMMARY_PATH}" \
  --v88_summary "${V88_SUMMARY_PATH}" \
  --output_json "${RESULT_ROOT}/v8-9-summary.json" \
  --output_report "${RESULT_ROOT}/v8-9-summary.md"

bash scripts/publish_review_artifacts.sh

echo "PLANv8 V8-9 CDMI complete."
