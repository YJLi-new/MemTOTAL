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
  python - "${task_name}" "${arm_id}" "${prompt_variant}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${PRIMARY_MODEL_DIR}" "${PRIMARY_BACKBONE_NAME}" "${v81_summary_path}" <<'PY'
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
primary_model_dir = str(Path(sys.argv[8]).resolve())
primary_backbone_name = sys.argv[9]
v81_summary_path = Path(sys.argv[10])

template_path = Path(f"configs/exp/writer_circuit_g2_writer_direct_{task_name}_template.yaml")
config = load_config(template_path)
config.setdefault("experiment", {})
config.setdefault("backbone", {})
config.setdefault("method", {})
config.setdefault("runtime", {})
config.setdefault("task", {})
config.setdefault("task", {}).setdefault("evaluator", {})

layer_bands = {
    "mid8": [14, 15, 16, 17, 18, 19, 20, 21],
    "mid12": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "late8": [20, 21, 22, 23, 24, 25, 26, 27],
}
arm_sweep = {
    "r0_mid8_r32_lr5e5": {"layer_band": "mid8", "rank": 32, "learning_rate": 5.0e-5, "train_steps": 300},
    "r1_mid8_r64_lr1e4": {"layer_band": "mid8", "rank": 64, "learning_rate": 1.0e-4, "train_steps": 300},
    "r2_mid12_r64_lr1e4": {"layer_band": "mid12", "rank": 64, "learning_rate": 1.0e-4, "train_steps": 300},
    "r3_mid12_r64_lr2e4": {"layer_band": "mid12", "rank": 64, "learning_rate": 2.0e-4, "train_steps": 300},
    "r4_late8_r32_lr1e4": {"layer_band": "late8", "rank": 32, "learning_rate": 1.0e-4, "train_steps": 300},
    "r5_mid8_r16_lr5e5": {"layer_band": "mid8", "rank": 16, "learning_rate": 5.0e-5, "train_steps": 300},
}
receiver_disabled = {
    "enabled": False,
    "target_layers": [],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 0,
    "alpha": 4.0,
    "dropout": 0.0,
}

v81_summary = json.loads(v81_summary_path.read_text())
selected_interface_family = str(
    v81_summary.get("selected_interface_family_for_v8_2")
    or v81_summary.get("best_interface_family")
    or ""
).strip()
base_arm_id = str(v81_summary.get("base_for_v8_2_arm_id") or v81_summary.get("best_arm_id") or "").strip()
base_arm_summary = v81_summary.get("arm_summaries", {}).get(base_arm_id, {})
slot_cap = int(base_arm_summary.get("memory_slots", 16))

config["backbone"]["name"] = primary_backbone_name
config["backbone"]["model_id"] = primary_model_dir
config["backbone"]["dtype"] = "bfloat16"
config["backbone"]["cache_dir"] = "/root/autodl-tmp/hf-cache"
config["backbone"]["use_chat_template"] = True
config["backbone"]["chat_template_enable_thinking"] = False
config["backbone"]["gradient_checkpointing"] = arm_id != "control"
config["backbone"]["max_new_tokens"] = 32
if task_name == "gsm8k":
    config["backbone"]["max_new_tokens"] = 192
    config["task"]["evaluator"]["normalizer"] = "gsm8k_final_answer"
elif task_name == "triviaqa":
    config["backbone"]["max_new_tokens"] = 64

config["task"]["support_dataset_path"] = support_path
config["task"]["train_dataset_path"] = train_path
config["task"]["train_support_dataset_path"] = support_path
config["task"]["dataset_path"] = eval_path
config["task"]["support_lookup_dataset_paths"] = []
config["task"]["train_support_episode_bank_path"] = ""
config["task"]["pilot_split"] = "eval"

config["experiment"]["name"] = f"{Path().resolve().name}_{arm_id}_{task_name}"
config["experiment"]["stage"] = "V8-2"
config["experiment"]["method_variant"] = arm_id

runtime = config["runtime"]
runtime["device"] = "cuda"
runtime["writer_memory_control"] = "real"
runtime["pilot_arm_alias"] = arm_id
runtime["pilot_prompt_variant"] = prompt_variant
runtime["pilot_support_serialization"] = "flat_raw8" if task_name == "fever" else "example_blocks_raw8"
runtime["pilot_train_support_mode"] = "static_support_rows"
runtime["pilot_support_examples"] = 8
runtime["pilot_lr_schedule"] = "constant_with_linear_warmup"
runtime["pilot_lr_warmup_steps"] = 30
runtime["pilot_gradient_accumulation_steps"] = 8
runtime["pilot_groupwise_grad_clip"] = True
runtime["pilot_gradient_clip_norm"] = 1.0
runtime["pilot_writer_grad_clip_norm"] = 1.0
runtime["pilot_projector_grad_clip_norm"] = 1.0
runtime["pilot_support_encoder_grad_clip_norm"] = 1.0
runtime["pilot_receiver_lora_grad_clip_norm"] = 1.0
runtime["pilot_reader_cross_attn_grad_clip_norm"] = 1.0
runtime["pilot_choice_ce_weight"] = 1.0
runtime["pilot_competitor_hinge_weight_max"] = 0.0
runtime["pilot_alignment_aux_mode"] = "off"
runtime["pilot_aux_loss_mode"] = "off"
runtime["pilot_writer_gain_margin_weight"] = 0.0
runtime["pilot_writer_common_mode_penalty_weight"] = 0.0
runtime["pilot_writer_covariance_diversity_weight"] = 0.0
runtime["pilot_writer_slot_energy_balance_weight"] = 0.0
runtime["pilot_memory_long_diversity_weight"] = 0.0
runtime["pilot_memory_short_diversity_weight"] = 0.0
runtime["pilot_reader_attention_diversity_weight"] = 0.0
runtime["pilot_reader_conditioned_query_orthogonality_weight"] = 0.0
runtime["pilot_reader_short_reconstruction_weight"] = 0.0
runtime["pilot_reader_fuser_bootstrap_steps"] = 0
runtime["pilot_writer_learning_rate"] = 0.0
runtime["pilot_projector_learning_rate"] = 0.0
runtime["pilot_support_encoder_learning_rate"] = 0.0
runtime["pilot_receiver_lora_learning_rate"] = 0.0
runtime["pilot_reader_cross_attn_learning_rate"] = 0.0
runtime["pilot_writer_weight_decay"] = 0.01
runtime["pilot_projector_weight_decay"] = 0.01
runtime["pilot_support_encoder_weight_decay"] = 0.01
runtime["pilot_receiver_lora_weight_decay"] = 0.01
runtime["pilot_reader_cross_attn_weight_decay"] = 0.01
runtime["pilot_memory_path_variant"] = "single_level"
runtime["pilot_bridge_mode"] = "writer_direct"
runtime["pilot_support_encoder_mode"] = "pooled_block"
runtime["pilot_writer_stimulus_mode"] = "support_only"
runtime["pilot_writer_context_tokens"] = 8
runtime["pilot_backbone_prompt_mask_mode"] = "none"
runtime["pilot_writer_context_prompt_mode"] = "same_as_backbone"
runtime["pilot_prefix_source_mode"] = "writer"
runtime["pilot_deep_prefix_init_mode"] = "kv_stat_match"
runtime["pilot_memory_consumer_mode"] = "legacy_prefix"
runtime["pilot_memory_segment_mode"] = "prepend_block"
runtime["pilot_train_steps"] = 0
runtime["pilot_snapshot_steps"] = [0]
runtime["pilot_gradient_probe_enabled"] = False
runtime["pilot_gradient_probe_interval"] = 5
runtime["pilot_gradient_probe_max_steps"] = 0
runtime["pilot_gradient_probe_modules"] = []
runtime["pilot_trainable_variant"] = "full"
runtime["pilot_oracle_extract_layer"] = 18
runtime["pilot_oracle_slot_pool_window"] = 16
runtime["pilot_oracle_slot_cap"] = slot_cap
runtime["pilot_deep_prefix_layers"] = [16, 17, 18, 19]
runtime["pilot_deep_prefix_rank"] = 32
runtime["pilot_deep_prefix_projector_mode"] = "shared_low_rank"
runtime["pilot_reader_cross_attn_layers"] = []
runtime["pilot_reader_cross_attn_heads"] = 16
runtime["pilot_reader_cross_attn_gate_init"] = 0.0
runtime["pilot_reader_cross_attn_ff_hidden_dim"] = 4096
config["method"]["receiver_lora"] = dict(receiver_disabled)

if arm_id == "control":
    runtime["shared_injection_arm"] = "base_only"
else:
    if arm_id not in arm_sweep:
        raise ValueError(f"Unsupported V8-2 arm {arm_id}.")
    spec = arm_sweep[arm_id]
    learning_rate = float(spec["learning_rate"])
    reader_layers = list(layer_bands[str(spec["layer_band"])])
    rank = int(spec["rank"])
    runtime["shared_injection_arm"] = "injected"
    runtime["pilot_prefix_source_mode"] = "oracle_hidden_state_slots"
    runtime["pilot_oracle_slot_cap"] = slot_cap
    runtime["pilot_train_steps"] = int(spec["train_steps"])
    runtime["pilot_snapshot_steps"] = [0, 10, 25, 50, 100, 150, 200, 250, int(spec["train_steps"])]
    runtime["pilot_gradient_probe_enabled"] = True
    runtime["pilot_gradient_probe_interval"] = 5
    runtime["pilot_gradient_probe_max_steps"] = int(spec["train_steps"])
    runtime["pilot_trainable_variant"] = "reader_only"
    runtime["pilot_projector_learning_rate"] = learning_rate
    runtime["pilot_receiver_lora_learning_rate"] = learning_rate
    runtime["pilot_reader_cross_attn_learning_rate"] = learning_rate
    if selected_interface_family == "ri0_legacy_prefix":
        runtime["pilot_memory_consumer_mode"] = "legacy_prefix"
        runtime["pilot_gradient_probe_modules"] = ["projector", "receiver_lora"]
        runtime["pilot_deep_prefix_layers"] = reader_layers
        runtime["pilot_deep_prefix_rank"] = rank
        config["method"]["receiver_lora"] = {
            "enabled": True,
            "target_layers": reader_layers,
            "target_modules": ["k_proj", "v_proj"],
            "rank": rank,
            "alpha": float(2 * rank),
            "dropout": 0.0,
        }
    elif selected_interface_family == "ri1_prepend_block":
        runtime["pilot_memory_consumer_mode"] = "reader_lora_sequence"
        runtime["pilot_gradient_probe_modules"] = ["receiver_lora"]
        config["method"]["receiver_lora"] = {
            "enabled": True,
            "target_layers": reader_layers,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "rank": rank,
            "alpha": float(2 * rank),
            "dropout": 0.05,
        }
    elif selected_interface_family == "ri2_cross_attn":
        runtime["pilot_memory_consumer_mode"] = "reader_cross_attn"
        runtime["pilot_gradient_probe_modules"] = ["reader_cross_attn"]
        runtime["pilot_reader_cross_attn_layers"] = reader_layers
        runtime["pilot_reader_cross_attn_heads"] = 16
        runtime["pilot_reader_cross_attn_gate_init"] = 0.0
        runtime["pilot_reader_cross_attn_ff_hidden_dim"] = int(rank * 64)
    else:
        raise ValueError(
            f"Unsupported selected_interface_family_for_v8_2={selected_interface_family!r}."
        )

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
