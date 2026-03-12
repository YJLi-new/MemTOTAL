#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv7-v7-3-bridge-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv7-v7-3-bridge-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
TRAIN_STEPS="${5:-300}"
MODEL_DIR="${6:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
V72_SUMMARY_JSON="${7:-results/generated/review/planv7-v7-2-direct-bandwidth-qwen25/v7-2-summary.json}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PLANV7_PROJECTOR_LR="${PLANV7_PROJECTOR_LR:-7.5e-6}"
export PLANV7_OWNER_LOCKED_PROJECTOR_LR="${PLANV7_OWNER_LOCKED_PROJECTOR_LR:-${PLANV7_PROJECTOR_LR}}"
export PLANV7_REPO_CONFIRMED_V65_PROJECTOR_LR_REFERENCE="${PLANV7_REPO_CONFIRMED_V65_PROJECTOR_LR_REFERENCE:-7.5e-5}"
export PLANV7_OWNER_OVERRIDE_NOTE="${PLANV7_OWNER_OVERRIDE_NOTE:-true}"
export PLANV7_EXPERIMENT_PREFIX="${PLANV7_EXPERIMENT_PREFIX:-planv7}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

DATA_ROOT="${RUN_ROOT}/materialized-datasets"
SOURCE_ROOT="${RUN_ROOT}/materialized-sources"
MANIFEST_ROOT="${RUN_ROOT}/materialized-manifests"
CONFIG_ROOT="${RUN_ROOT}/materialized-configs"
mkdir -p "${DATA_ROOT}" "${SOURCE_ROOT}" "${MANIFEST_ROOT}" "${CONFIG_ROOT}"

read -r DIRECT_CONTROL_ARM_ID WINNING_DEPTH WINNING_DEPTH_LABEL < <(
  python - "${V72_SUMMARY_JSON}" <<'PY'
import json
import os
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
payload = json.loads(summary_path.read_text())
ranking = payload.get("primary_arm_ranking", [])
if not ranking:
    raise SystemExit(f"Missing primary_arm_ranking in {summary_path}")
control_arm_id = str(ranking[0].get("arm_id", "")).strip()
winning_depth = str(payload.get("winning_depth", "D1")).strip() or "D1"
winning_depth_label = str(payload.get("winning_depth_label", "mid4")).strip() or "mid4"
print(control_arm_id, winning_depth, winning_depth_label)
PY
)

python -m memtotal.tasks.writer_jointpeft_data \
  --output_root "${DATA_ROOT}" \
  --source_output_root "${SOURCE_ROOT}" \
  --manifest_root "${MANIFEST_ROOT}" \
  --seed "${BASE_SEED}" \
  --benchmarks "gsm8k,triviaqa"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "${MODEL_DIR}" \
  "${HF_HOME}"

materialize_config() {
  local task_name="$1"
  local arm_id="$2"
  local output_config="$3"
  local support_path="$4"
  local train_path="$5"
  local eval_path="$6"
  python - "${task_name}" "${arm_id}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${TRAIN_STEPS}" "${MODEL_DIR}" "${DIRECT_CONTROL_ARM_ID}" "${WINNING_DEPTH}" <<'PY'
import json
import os
import sys
from pathlib import Path

from memtotal.utils.config import load_config

task_name = sys.argv[1]
arm_id = sys.argv[2]
output_config = Path(sys.argv[3])
support_path = str(Path(sys.argv[4]).resolve())
train_path = str(Path(sys.argv[5]).resolve())
eval_path = str(Path(sys.argv[6]).resolve())
train_steps = max(0, int(sys.argv[7]))
model_dir = str(Path(sys.argv[8]).resolve())
direct_control_arm_id = sys.argv[9]
winning_depth = sys.argv[10]

config = load_config(f"configs/exp/writer_circuit_g2_writer_direct_{task_name}_template.yaml")
config.setdefault("task", {})
config.setdefault("method", {})
config.setdefault("runtime", {})

depth_specs = {
    "D0": {"depth_layers": [0, 1, 2, 3], "receiver_layers": [0, 1, 2, 3]},
    "D1": {"depth_layers": [12, 13, 14, 15], "receiver_layers": [12, 13, 14, 15]},
    "D2": {"depth_layers": [10, 11, 12, 13, 14, 15], "receiver_layers": [12, 13, 14, 15]},
    "D3": {"depth_layers": [0, 1, 2, 3, 12, 13, 14, 15], "receiver_layers": [12, 13, 14, 15]},
}
if winning_depth not in depth_specs:
    raise ValueError(f"Unsupported PLANv7 depth family {winning_depth!r}.")
depth_spec = depth_specs[winning_depth]

w1_writer = {
    "arch": "transformer",
    "memory_slots": 16,
    "hidden_dim": 512,
    "num_heads": 4,
    "transformer_layers": 2,
    "conditioning_layers": 2,
    "dropout": 0.0,
}
w2_writer = {
    "arch": "transformer",
    "memory_slots": 32,
    "hidden_dim": 1536,
    "num_heads": 8,
    "transformer_layers": 4,
    "conditioning_layers": 2,
    "dropout": 0.0,
}
w3_writer = {
    "arch": "transformer",
    "memory_slots": 64,
    "hidden_dim": 3072,
    "num_heads": 8,
    "transformer_layers": 4,
    "conditioning_layers": 3,
    "dropout": 0.0,
}
w4_writer = {
    "arch": "transformer",
    "memory_slots": 96,
    "hidden_dim": 3072,
    "num_heads": 8,
    "transformer_layers": 4,
    "conditioning_layers": 3,
    "dropout": 0.0,
}

direct_control_specs = {
    "d_w1_shared": {
        "writer_family": "W1",
        "bridge_family": "B0",
        "projector_family": "P1_shared_rank64",
        "writer": w1_writer,
        "projector_rank": 64,
        "projector_mode": "shared_low_rank",
    },
    "d_w2_shared": {
        "writer_family": "W2",
        "bridge_family": "B0",
        "projector_family": "shared_rank64_control",
        "writer": w2_writer,
        "projector_rank": 64,
        "projector_mode": "shared_low_rank",
    },
    "d_w2_perlayer": {
        "writer_family": "W2",
        "bridge_family": "B0",
        "projector_family": "P2_per_layer_rank128",
        "writer": w2_writer,
        "projector_rank": 128,
        "projector_mode": "per_layer_low_rank",
    },
}
bridge_specs = {
    "b_w3_q8": {
        "writer_family": "W3",
        "bridge_family": "B1",
        "projector_family": "P2",
        "writer": w3_writer,
        "projector_rank": 128,
        "projector_mode": "per_layer_low_rank",
        "reader_queries": 8,
        "short_slots": 8,
    },
    "b_w3_q16": {
        "writer_family": "W3",
        "bridge_family": "B2",
        "projector_family": "P2",
        "writer": w3_writer,
        "projector_rank": 128,
        "projector_mode": "per_layer_low_rank",
        "reader_queries": 16,
        "short_slots": 16,
    },
    "b_w3_q16_s8": {
        "writer_family": "W3",
        "bridge_family": "B3",
        "projector_family": "P2",
        "writer": w3_writer,
        "projector_rank": 128,
        "projector_mode": "per_layer_low_rank",
        "reader_queries": 16,
        "short_slots": 8,
    },
    "b_w4_q16": {
        "writer_family": "W4",
        "bridge_family": "B2",
        "projector_family": "P3",
        "writer": w4_writer,
        "projector_rank": 256,
        "projector_mode": "per_layer_low_rank",
        "reader_queries": 16,
        "short_slots": 16,
    },
}

if direct_control_arm_id not in direct_control_specs:
    raise ValueError(
        f"Unsupported PLANv7 V7-2 direct control arm {direct_control_arm_id!r}; "
        f"expected one of {sorted(direct_control_specs)}."
    )

config["backbone"]["model_id"] = model_dir
config["task"]["support_dataset_path"] = support_path
config["task"]["train_dataset_path"] = train_path
config["task"]["train_support_dataset_path"] = support_path
config["task"]["dataset_path"] = eval_path
config["task"]["support_lookup_dataset_paths"] = []
config["task"]["train_support_episode_bank_path"] = ""
config["task"]["pilot_split"] = str(config["task"].get("split", config["task"].get("smoke_subset", "eval")))

config["runtime"]["pilot_bridge_mode"] = "writer_direct"
config["runtime"]["pilot_injection_mode"] = "sparse_deep_prefix"
config["runtime"]["pilot_deep_prefix_init_mode"] = "kv_stat_match"
config["runtime"]["pilot_prefix_source_mode"] = "writer"
config["runtime"]["pilot_support_encoder_mode"] = "multi_item_cross_attn_raw"
config["runtime"]["pilot_support_serialization"] = "example_blocks_raw8"
config["runtime"]["pilot_writer_stimulus_mode"] = "support_and_context"
config["runtime"]["pilot_context_support_balance_mode"] = "layernorm_learned_scalar"
config["runtime"]["pilot_context_balance_scale_init"] = 0.75
config["runtime"]["pilot_support_balance_scale_init"] = 1.25
config["runtime"]["pilot_aux_loss_mode"] = "orthogonality_coverage"
config["runtime"]["pilot_writer_slot_orthogonality_weight"] = 0.05
config["runtime"]["pilot_writer_support_coverage_weight"] = 0.05
config["runtime"]["pilot_writer_gain_margin"] = 0.05
config["runtime"]["pilot_writer_gain_margin_weight"] = 0.25
config["runtime"]["pilot_writer_common_mode_penalty_weight"] = 0.1
config["runtime"]["pilot_writer_covariance_diversity_weight"] = 0.05
config["runtime"]["pilot_writer_slot_energy_balance_weight"] = 0.01
config["runtime"]["pilot_writer_context_tokens"] = 8
config["runtime"]["pilot_train_support_mode"] = "static_support_rows"
config["runtime"]["pilot_support_examples"] = 8
config["runtime"]["pilot_lr_schedule"] = "constant_with_linear_warmup"
config["runtime"]["pilot_lr_warmup_steps"] = 0
config["runtime"]["pilot_projector_warmup_steps"] = 0
config["runtime"]["pilot_writer_learning_rate"] = 1.0e-4
config["runtime"]["pilot_projector_learning_rate"] = float(os.environ["PLANV7_PROJECTOR_LR"])
config["runtime"]["pilot_receiver_lora_learning_rate"] = 5.0e-5
config["runtime"]["owner_locked_projector_lr"] = float(os.environ["PLANV7_OWNER_LOCKED_PROJECTOR_LR"])
config["runtime"]["repo_confirmed_v65_projector_lr_reference"] = float(
    os.environ["PLANV7_REPO_CONFIRMED_V65_PROJECTOR_LR_REFERENCE"]
)
config["runtime"]["owner_override_note"] = (
    os.environ["PLANV7_OWNER_OVERRIDE_NOTE"].strip().lower() == "true"
)
config["runtime"]["pilot_writer_weight_decay"] = 0.0
config["runtime"]["pilot_projector_weight_decay"] = 0.0
config["runtime"]["pilot_receiver_lora_weight_decay"] = 0.0
config["runtime"]["pilot_gradient_accumulation_steps"] = 4
config["runtime"]["pilot_groupwise_grad_clip"] = True
config["runtime"]["pilot_gradient_clip_norm"] = 1.0
config["runtime"]["pilot_writer_grad_clip_norm"] = 1.0
config["runtime"]["pilot_projector_grad_clip_norm"] = 0.5
config["runtime"]["pilot_receiver_lora_grad_clip_norm"] = 0.5
config["runtime"]["pilot_gradient_probe_enabled"] = True
config["runtime"]["pilot_gradient_probe_interval"] = 5
config["runtime"]["pilot_gradient_probe_max_steps"] = min(150, max(1, train_steps))
config["runtime"]["pilot_gradient_probe_modules"] = [
    "writer",
    "projector",
    "receiver_lora",
]
config["runtime"]["pilot_prompt_variant"] = "task_native"
config["runtime"]["pilot_active_support_family"] = "S3"
config["runtime"]["pilot_active_context_family"] = "C2"
config["runtime"]["pilot_active_aux_family"] = "L5"
config["runtime"]["pilot_active_depth_family"] = winning_depth
config["runtime"]["pilot_deep_prefix_layers"] = list(depth_spec["depth_layers"])

receiver_lora_config = {
    "enabled": True,
    "target_layers": list(depth_spec["receiver_layers"]),
    "target_modules": ["k_proj", "v_proj"],
    "rank": 2,
    "alpha": 4.0,
    "dropout": 0.0,
}

config["runtime"]["shared_injection_arm"] = "injected"
config["runtime"]["pilot_train_steps"] = train_steps
config["runtime"]["pilot_snapshot_steps"] = [0, 10, 25, 50, 100, 150, 200, 250, train_steps]

if arm_id == "control":
    spec = direct_control_specs[direct_control_arm_id]
    config["runtime"]["pilot_arm_alias"] = "B_CTRL"
    config["runtime"]["pilot_memory_path_variant"] = "single_level"
    config["runtime"]["pilot_projector_token_source"] = "writer_slots"
    config["runtime"]["pilot_active_writer_family"] = spec["writer_family"]
    config["runtime"]["pilot_active_projector_family"] = spec["projector_family"]
    config["runtime"]["pilot_active_bridge_family"] = spec["bridge_family"]
    config["runtime"]["pilot_active_direct_control_arm_id"] = direct_control_arm_id
    config["runtime"]["pilot_deep_prefix_rank"] = int(spec["projector_rank"])
    config["runtime"]["pilot_deep_prefix_projector_mode"] = spec["projector_mode"]
    config["method"]["writer"] = dict(spec["writer"])
    config["method"]["receiver_lora"] = dict(receiver_lora_config)
    config["method"].pop("reader", None)
    config["method"].pop("fuser", None)
else:
    spec = bridge_specs[arm_id]
    config["runtime"]["pilot_arm_alias"] = arm_id.upper()
    config["runtime"]["pilot_memory_path_variant"] = "two_level"
    config["runtime"]["pilot_projector_token_source"] = "short_slots"
    config["runtime"]["pilot_reader_context_mode"] = "prompt_summary"
    config["runtime"]["pilot_reader_num_queries"] = int(spec["reader_queries"])
    config["runtime"]["pilot_fuser_short_slots"] = int(spec["short_slots"])
    config["runtime"]["pilot_active_writer_family"] = spec["writer_family"]
    config["runtime"]["pilot_active_projector_family"] = spec["projector_family"]
    config["runtime"]["pilot_active_bridge_family"] = spec["bridge_family"]
    config["runtime"]["pilot_deep_prefix_rank"] = int(spec["projector_rank"])
    config["runtime"]["pilot_deep_prefix_projector_mode"] = spec["projector_mode"]
    config["method"]["writer"] = dict(spec["writer"])
    config["method"]["reader"] = {
        "num_queries": int(spec["reader_queries"]),
        "use_query_gating": False,
        "condition_on_context": True,
        "conditioning_mode": "add",
        "attention_mode": "standard",
        "dropout": 0.0,
        "query_residual_scale": 0.0,
        "num_heads": 8,
    }
    config["method"]["fuser"] = {
        "short_slots": int(spec["short_slots"]),
        "arch": "resampler",
        "hidden_dim": 1536,
        "num_heads": 8,
        "dropout": 0.0,
    }
    config["method"]["receiver_lora"] = dict(receiver_lora_config)

config["experiment"]["name"] = f"{os.environ['PLANV7_EXPERIMENT_PREFIX']}_v7_3_{task_name}_{arm_id}"
output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

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

copy_run_artifacts() {
  local dst_dir="$1"
  local run_dir="$2"
  local pilot_subdir="$3"
  mkdir -p "${dst_dir}"
  cp "${run_dir}/${pilot_subdir}/metrics.json" "${dst_dir}/metrics.json"
  if [[ -f "${run_dir}/${pilot_subdir}/train_events.json" ]]; then
    cp "${run_dir}/${pilot_subdir}/train_events.json" "${dst_dir}/train_events.json"
  else
    printf '[]\n' > "${dst_dir}/train_events.json"
  fi
  if [[ -f "${run_dir}/${pilot_subdir}/task_case_dump.jsonl" ]]; then
    cp "${run_dir}/${pilot_subdir}/task_case_dump.jsonl" "${dst_dir}/task_case_dump.jsonl"
  fi
  cp "${run_dir}/suite_metrics.json" "${dst_dir}/suite_metrics.json"
}

for task_name in gsm8k triviaqa; do
  materialize_config \
    "${task_name}" \
    "control" \
    "${CONFIG_ROOT}/${task_name}-control.json" \
    "${DATA_ROOT}/${task_name}/support.jsonl" \
    "${DATA_ROOT}/${task_name}/train.jsonl" \
    "${DATA_ROOT}/${task_name}/eval.jsonl"
done

for arm_id in b_w3_q8 b_w3_q16 b_w3_q16_s8 b_w4_q16; do
  for task_name in gsm8k triviaqa; do
    materialize_config \
      "${task_name}" \
      "${arm_id}" \
      "${CONFIG_ROOT}/${task_name}-${arm_id}.json" \
      "${DATA_ROOT}/${task_name}/support.jsonl" \
      "${DATA_ROOT}/${task_name}/train.jsonl" \
      "${DATA_ROOT}/${task_name}/eval.jsonl"
  done
done

seed_offset=0
for task_name in gsm8k triviaqa; do
  ensure_suite_complete \
    "${CONFIG_ROOT}/${task_name}-control.json" \
    "$((BASE_SEED + seed_offset))" \
    "${RUN_ROOT}/${task_name}-control" \
    "pilot-I-real:B_CTRL:injected:real:0"
  copy_run_artifacts \
    "${RESULT_ROOT}/control/${task_name}" \
    "${RUN_ROOT}/${task_name}-control" \
    "pilot-I-real"
  seed_offset=$((seed_offset + 20))
done

for arm_id in b_w3_q8 b_w3_q16 b_w3_q16_s8 b_w4_q16; do
  for task_name in gsm8k triviaqa; do
    ensure_suite_complete \
      "${CONFIG_ROOT}/${task_name}-${arm_id}.json" \
      "$((BASE_SEED + seed_offset))" \
      "${RUN_ROOT}/${task_name}-${arm_id}" \
      "pilot-I-real:${arm_id^^}:injected:real:0"
    copy_run_artifacts \
      "${RESULT_ROOT}/${arm_id}/${task_name}" \
      "${RUN_ROOT}/${task_name}-${arm_id}" \
      "pilot-I-real"
    seed_offset=$((seed_offset + 20))
  done
done

python scripts/update_planv7_v7_3_bridge_summary.py \
  --result_root "${RESULT_ROOT}" \
  --v72_summary "${V72_SUMMARY_JSON}" \
  --output_json "${RESULT_ROOT}/v7-3-summary.json" \
  --output_report "${RESULT_ROOT}/v7-3-summary.md"

bash scripts/publish_review_artifacts.sh

echo "PLANv7 V7-3 bridge sweep complete."
