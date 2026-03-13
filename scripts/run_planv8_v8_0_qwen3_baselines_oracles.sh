#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-0-qwen3-baselines-oracles}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-0-qwen3-baselines-oracles}"
QWEN25_MODEL_DIR="${4:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
QWEN3_MODEL_DIR="${5:-/root/autodl-tmp/models/Qwen3-8B}"
QWEN25_REFERENCE_SUMMARY="${6:-results/generated/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25/v7-0-summary.json}"

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

SPLIT_PLAN_JSON="${MANIFEST_ROOT}/v8-0-split-plan.json"
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

if [[ ! -f "${QWEN25_MODEL_DIR}/config.json" ]]; then
  bash scripts/prepare_local_qwen25_model.sh \
    "Qwen/Qwen2.5-1.5B-Instruct" \
    "${QWEN25_MODEL_DIR}" \
    "${HF_HOME}"
fi

bash scripts/prepare_local_qwen3_model.sh \
  "Qwen/Qwen3-8B" \
  "${QWEN3_MODEL_DIR}" \
  "${HF_HOME}"

materialize_config() {
  local task_name="$1"
  local arm_id="$2"
  local prompt_variant="$3"
  local output_config="$4"
  local support_path="$5"
  local train_path="$6"
  local eval_path="$7"
  python - "${task_name}" "${arm_id}" "${prompt_variant}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${QWEN25_MODEL_DIR}" "${QWEN3_MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

task_name = sys.argv[1]
arm_id = sys.argv[2]
prompt_variant = sys.argv[3]
output_config = Path(sys.argv[4])
support_path = str(Path(sys.argv[5]).resolve())
train_path = str(Path(sys.argv[6]).resolve())
eval_path = str(Path(sys.argv[7]).resolve())
qwen25_model_dir = str(Path(sys.argv[8]).resolve())
qwen3_model_dir = str(Path(sys.argv[9]).resolve())

template_path = Path(f"configs/exp/writer_circuit_g2_writer_direct_{task_name}_template.yaml")
config = load_config(template_path)
config.setdefault("experiment", {})
config.setdefault("backbone", {})
config.setdefault("method", {})
config.setdefault("runtime", {})
config.setdefault("task", {})
config.setdefault("method", {}).setdefault("receiver_lora", {})
config.setdefault("task", {}).setdefault("evaluator", {})

use_qwen25 = arm_id.startswith("o0_")
config["backbone"]["name"] = "Qwen2.5-1.5B-Instruct" if use_qwen25 else "Qwen3-8B"
config["backbone"]["model_id"] = qwen25_model_dir if use_qwen25 else qwen3_model_dir
config["backbone"]["dtype"] = "bfloat16"
config["backbone"]["cache_dir"] = "/root/autodl-tmp/hf-cache"
config["backbone"]["gradient_checkpointing"] = bool(arm_id.startswith("o4_"))
config["backbone"]["max_new_tokens"] = 32
if not use_qwen25:
    config["backbone"]["use_chat_template"] = True
    config["backbone"]["chat_template_enable_thinking"] = False
    if task_name == "gsm8k":
        config["backbone"]["max_new_tokens"] = 192
    elif task_name == "triviaqa":
        config["backbone"]["max_new_tokens"] = 64

config["task"]["support_dataset_path"] = support_path
config["task"]["train_dataset_path"] = train_path
config["task"]["train_support_dataset_path"] = support_path
config["task"]["dataset_path"] = eval_path
config["task"]["support_lookup_dataset_paths"] = []
config["task"]["train_support_episode_bank_path"] = ""
config["task"]["pilot_split"] = "eval"
if task_name == "gsm8k":
    config["task"]["evaluator"]["normalizer"] = "gsm8k_final_answer"

config["experiment"]["name"] = f"{Path().resolve().name}_{arm_id}"
config["experiment"]["stage"] = "V8-0"
config["experiment"]["method_variant"] = arm_id

receiver_cfg = {
    "enabled": False,
    "target_layers": [],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 0,
    "alpha": 4.0,
    "dropout": 0.0,
}
config["method"]["receiver_lora"] = receiver_cfg

runtime = config["runtime"]
runtime["device"] = "cuda"
runtime["shared_injection_arm"] = "base_only"
runtime["writer_memory_control"] = "real"
runtime["pilot_arm_alias"] = arm_id
runtime["pilot_prompt_variant"] = prompt_variant
runtime["pilot_support_serialization"] = "example_blocks_raw8"
runtime["pilot_train_support_mode"] = "static_support_rows"
runtime["pilot_support_examples"] = 8
runtime["pilot_train_steps"] = 0
runtime["pilot_snapshot_steps"] = [0]
runtime["pilot_gradient_probe_enabled"] = False
runtime["pilot_aux_loss_mode"] = "off"
runtime["pilot_alignment_aux_mode"] = "off"
runtime["pilot_writer_gain_margin_weight"] = 0.0
runtime["pilot_writer_common_mode_penalty_weight"] = 0.0
runtime["pilot_writer_covariance_diversity_weight"] = 0.0
runtime["pilot_writer_slot_energy_balance_weight"] = 0.0
runtime["pilot_writer_learning_rate"] = 0.0
runtime["pilot_projector_learning_rate"] = 0.0
runtime["pilot_support_encoder_learning_rate"] = 0.0
runtime["pilot_receiver_lora_learning_rate"] = 0.0
runtime["pilot_writer_weight_decay"] = 0.0
runtime["pilot_projector_weight_decay"] = 0.0
runtime["pilot_support_encoder_weight_decay"] = 0.0
runtime["pilot_receiver_lora_weight_decay"] = 0.0
runtime["pilot_lr_schedule"] = "constant"
runtime["pilot_lr_warmup_steps"] = 0
runtime["pilot_gradient_accumulation_steps"] = 1
runtime["pilot_groupwise_grad_clip"] = True
runtime["pilot_gradient_clip_norm"] = 1.0
runtime["pilot_memory_path_variant"] = "single_level"
runtime["pilot_bridge_mode"] = "writer_direct"
runtime["pilot_injection_mode"] = "sparse_deep_prefix"
runtime["pilot_support_encoder_mode"] = "pooled_block"
runtime["pilot_writer_stimulus_mode"] = "support_only"
runtime["pilot_writer_context_tokens"] = 8
runtime["pilot_backbone_prompt_mask_mode"] = "none"
runtime["pilot_writer_context_prompt_mode"] = "same_as_backbone"
runtime["pilot_prefix_source_mode"] = "writer"
runtime["pilot_memory_consumer_mode"] = "legacy_prefix"
runtime["pilot_memory_segment_mode"] = "prepend_block"

if arm_id.startswith("b0_") or arm_id.startswith("b1_") or arm_id.startswith("b2_") or arm_id.startswith("b3_") or arm_id.startswith("b4_"):
    runtime["shared_injection_arm"] = "base_only"
elif arm_id.startswith("o0_"):
    runtime["shared_injection_arm"] = "injected"
    runtime["pilot_prefix_source_mode"] = "oracle_support_echo"
    runtime["pilot_deep_prefix_layers"] = [12, 13, 14, 15]
elif arm_id.startswith("o1_"):
    runtime["shared_injection_arm"] = "injected"
    runtime["pilot_prefix_source_mode"] = "oracle_hidden_state_slots"
    runtime["pilot_memory_consumer_mode"] = "legacy_prefix"
    runtime["pilot_deep_prefix_layers"] = [16, 17, 18, 19]
    runtime["pilot_oracle_extract_layer"] = 18
    runtime["pilot_oracle_slot_pool_window"] = 16
    runtime["pilot_oracle_slot_cap"] = 16
elif arm_id.startswith("o2_"):
    runtime["shared_injection_arm"] = "injected"
    runtime["pilot_prefix_source_mode"] = "oracle_hidden_state_slots"
    runtime["pilot_memory_consumer_mode"] = "reader_lora_sequence"
    runtime["pilot_memory_segment_mode"] = "prepend_block"
    runtime["pilot_oracle_extract_layer"] = 18
    runtime["pilot_oracle_slot_pool_window"] = 16
    runtime["pilot_oracle_slot_cap"] = 16
elif arm_id.startswith("o3_"):
    runtime["shared_injection_arm"] = "injected"
    runtime["pilot_prefix_source_mode"] = "oracle_hidden_state_slots"
    runtime["pilot_memory_consumer_mode"] = "reader_lora_sequence"
    runtime["pilot_memory_segment_mode"] = "prepend_block"
    runtime["pilot_oracle_extract_layer"] = 18
    runtime["pilot_oracle_slot_pool_window"] = 16
    runtime["pilot_oracle_slot_cap"] = 32
elif arm_id.startswith("o4_"):
    runtime["shared_injection_arm"] = "injected"
    runtime["pilot_prefix_source_mode"] = "oracle_hidden_state_slots"
    runtime["pilot_memory_consumer_mode"] = "reader_cross_attn"
    runtime["pilot_oracle_extract_layer"] = 18
    runtime["pilot_oracle_slot_pool_window"] = 16
    runtime["pilot_oracle_slot_cap"] = 16
    runtime["pilot_train_steps"] = 50
    runtime["pilot_snapshot_steps"] = [0, 10, 25, 50]
    runtime["pilot_trainable_variant"] = "projector_only"
    runtime["pilot_gradient_probe_enabled"] = True
    runtime["pilot_gradient_probe_interval"] = 5
    runtime["pilot_gradient_probe_max_steps"] = 50
    runtime["pilot_gradient_probe_modules"] = ["reader_cross_attn"]
    runtime["pilot_reader_cross_attn_layers"] = [16, 17, 18, 19]
    runtime["pilot_reader_cross_attn_heads"] = 16
    runtime["pilot_reader_cross_attn_gate_init"] = 0.0
    runtime["pilot_reader_cross_attn_learning_rate"] = 5.0e-5
    runtime["pilot_reader_cross_attn_weight_decay"] = 0.0
else:
    raise ValueError(f"Unsupported arm_id={arm_id}")

output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
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

select_prompt_variant() {
  local task_name="$1"
  local arm_a="$2"
  local arm_b="$3"
  python - "${RUN_ROOT}" "${task_name}" "${arm_a}" "${arm_b}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
task_name = sys.argv[2]
arm_a = sys.argv[3]
arm_b = sys.argv[4]

def load_metrics(arm_id: str) -> dict:
    return json.loads((run_root / arm_id / "metrics.json").read_text())

def prompt_variant(payload: dict) -> str:
    return str(
        payload.get("prompt_variant")
        or payload.get("pilot_prompt_variant")
        or ""
    )

def score_tuple(payload: dict) -> tuple[float, float]:
    return (
        float(payload.get("best_adapt_task_score", 0.0)),
        float(payload.get("best_adapt_macro_f1", 0.0)),
    )

metrics_a = load_metrics(arm_a)
metrics_b = load_metrics(arm_b)
candidate_rows = [
    (score_tuple(metrics_a), arm_a, prompt_variant(metrics_a)),
    (score_tuple(metrics_b), arm_b, prompt_variant(metrics_b)),
]
candidate_rows.sort(key=lambda item: (item[0][0], item[0][1], item[1]), reverse=True)
selected = {
    "task_name": task_name,
    "selected_arm_id": candidate_rows[0][1],
    "selected_prompt_variant": candidate_rows[0][2],
    "arm_scores": {
        arm_a: {
            "task_score": score_tuple(metrics_a)[0],
            "macro_f1": score_tuple(metrics_a)[1],
            "prompt_variant": prompt_variant(metrics_a),
        },
        arm_b: {
            "task_score": score_tuple(metrics_b)[0],
            "macro_f1": score_tuple(metrics_b)[1],
            "prompt_variant": prompt_variant(metrics_b),
        },
    },
}
print(json.dumps(selected))
PY
}

BASELINE_ARMS=(
  "gsm8k:b0_q3_gsm8k_nonthink:q3_gsm8k_nonthink"
  "gsm8k:b1_q3_gsm8k_think_boxed:q3_gsm8k_think_boxed"
  "triviaqa:b2_q3_trivia_nonthink:q3_trivia_nonthink"
  "triviaqa:b3_q3_trivia_think:q3_trivia_think"
  "fever:b4_q3_fever_nonthink:answer_slot_labels"
)

for spec in "${BASELINE_ARMS[@]}"; do
  IFS=":" read -r task_name arm_id prompt_variant <<<"${spec}"
  materialize_config \
    "${task_name}" \
    "${arm_id}" \
    "${prompt_variant}" \
    "${CONFIG_ROOT}/${arm_id}.json" \
    "${DATA_ROOT}/${task_name}/support.jsonl" \
    "${DATA_ROOT}/${task_name}/train.jsonl" \
    "${DATA_ROOT}/${task_name}/eval.jsonl"
  run_single_pilot "${CONFIG_ROOT}/${arm_id}.json" "${BASE_SEED}" "${RUN_ROOT}/${arm_id}"
done

GSM8K_SELECTION="$(select_prompt_variant "gsm8k" "b0_q3_gsm8k_nonthink" "b1_q3_gsm8k_think_boxed")"
TRIVIAQA_SELECTION="$(select_prompt_variant "triviaqa" "b2_q3_trivia_nonthink" "b3_q3_trivia_think")"
python - <<'PY' "${RESULT_ROOT}" "${GSM8K_SELECTION}" "${TRIVIAQA_SELECTION}"
import json
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
gsm8k_selection = json.loads(sys.argv[2])
triviaqa_selection = json.loads(sys.argv[3])
payload = {
    "gsm8k": gsm8k_selection,
    "triviaqa": triviaqa_selection,
    "fever": {
        "selected_arm_id": "b4_q3_fever_nonthink",
        "selected_prompt_variant": "answer_slot_labels",
    },
}
(result_root / "selected-prompt-modes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

GSM8K_PROMPT="$(python - <<'PY' "${RESULT_ROOT}/selected-prompt-modes.json"
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["gsm8k"]["selected_prompt_variant"])
PY
)"
TRIVIAQA_PROMPT="$(python - <<'PY' "${RESULT_ROOT}/selected-prompt-modes.json"
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["triviaqa"]["selected_prompt_variant"])
PY
)"

ORACLE_ARMS=(
  "gsm8k:o0_q25_prefix_replay:${GSM8K_PROMPT}"
  "triviaqa:o0_q25_prefix_replay:${TRIVIAQA_PROMPT}"
  "gsm8k:o1_q3_prefix_oracle_mid4:${GSM8K_PROMPT}"
  "triviaqa:o1_q3_prefix_oracle_mid4:${TRIVIAQA_PROMPT}"
  "gsm8k:o2_q3_seq_oracle16:${GSM8K_PROMPT}"
  "triviaqa:o2_q3_seq_oracle16:${TRIVIAQA_PROMPT}"
  "gsm8k:o3_q3_seq_oracle32:${GSM8K_PROMPT}"
  "triviaqa:o3_q3_seq_oracle32:${TRIVIAQA_PROMPT}"
  "gsm8k:o4_q3_xattn_oracle_smoke:${GSM8K_PROMPT}"
  "triviaqa:o4_q3_xattn_oracle_smoke:${TRIVIAQA_PROMPT}"
)

for spec in "${ORACLE_ARMS[@]}"; do
  IFS=":" read -r task_name arm_prefix prompt_variant <<<"${spec}"
  arm_id="${arm_prefix}_${task_name}"
  materialize_config \
    "${task_name}" \
    "${arm_id}" \
    "${prompt_variant}" \
    "${CONFIG_ROOT}/${arm_id}.json" \
    "${DATA_ROOT}/${task_name}/support.jsonl" \
    "${DATA_ROOT}/${task_name}/train.jsonl" \
    "${DATA_ROOT}/${task_name}/eval.jsonl"
  run_single_pilot "${CONFIG_ROOT}/${arm_id}.json" "${BASE_SEED}" "${RUN_ROOT}/${arm_id}"
done

python scripts/update_planv8_v8_0_summary.py \
  --run-root "${RUN_ROOT}" \
  --output-root "${RESULT_ROOT}" \
  --qwen25-reference-summary "${QWEN25_REFERENCE_SUMMARY}" \
  --selected-prompt-modes "${RESULT_ROOT}/selected-prompt-modes.json"

echo "planv8 v8-0 completed"
