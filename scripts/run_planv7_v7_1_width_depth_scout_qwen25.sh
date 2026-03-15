#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv7-v7-1-width-depth-scout-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv7-v7-1-width-depth-scout-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
TRAIN_STEPS="${5:-200}"
MODEL_DIR="${6:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"

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
  python - "${task_name}" "${arm_id}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${TRAIN_STEPS}" "${MODEL_DIR}" <<'PY'
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

config = load_config(f"configs/exp/writer_circuit_g2_writer_direct_{task_name}_template.yaml")
config.setdefault("task", {})
config.setdefault("method", {})
config.setdefault("runtime", {})

arm_specs = {
    "s00": {
        "writer_family": "W0",
        "depth_family": "D0",
        "projector_family": "P0",
        "writer": {
            "arch": "transformer",
            "memory_slots": 8,
            "hidden_dim": 128,
            "num_heads": 4,
            "transformer_layers": 2,
            "conditioning_layers": 1,
            "dropout": 0.0,
        },
        "depth_layers": [0, 1, 2, 3],
        "receiver_layers": [0, 1, 2, 3],
        "projector_rank": 32,
    },
    "s01": {
        "writer_family": "W0",
        "depth_family": "D1",
        "projector_family": "P0",
        "writer": {
            "arch": "transformer",
            "memory_slots": 8,
            "hidden_dim": 128,
            "num_heads": 4,
            "transformer_layers": 2,
            "conditioning_layers": 1,
            "dropout": 0.0,
        },
        "depth_layers": [12, 13, 14, 15],
        "receiver_layers": [12, 13, 14, 15],
        "projector_rank": 32,
    },
    "s10": {
        "writer_family": "W1",
        "depth_family": "D0",
        "projector_family": "P1",
        "writer": {
            "arch": "transformer",
            "memory_slots": 16,
            "hidden_dim": 512,
            "num_heads": 4,
            "transformer_layers": 2,
            "conditioning_layers": 2,
            "dropout": 0.0,
        },
        "depth_layers": [0, 1, 2, 3],
        "receiver_layers": [0, 1, 2, 3],
        "projector_rank": 64,
    },
    "s11": {
        "writer_family": "W1",
        "depth_family": "D1",
        "projector_family": "P1",
        "writer": {
            "arch": "transformer",
            "memory_slots": 16,
            "hidden_dim": 512,
            "num_heads": 4,
            "transformer_layers": 2,
            "conditioning_layers": 2,
            "dropout": 0.0,
        },
        "depth_layers": [12, 13, 14, 15],
        "receiver_layers": [12, 13, 14, 15],
        "projector_rank": 64,
    },
}

config["backbone"]["model_id"] = model_dir
config["task"]["support_dataset_path"] = support_path
config["task"]["train_dataset_path"] = train_path
config["task"]["train_support_dataset_path"] = support_path
config["task"]["dataset_path"] = eval_path
config["task"]["support_lookup_dataset_paths"] = []
config["task"]["train_support_episode_bank_path"] = ""
config["task"]["pilot_split"] = str(config["task"].get("split", config["task"].get("smoke_subset", "eval")))

config["runtime"]["pilot_bridge_mode"] = "writer_direct"
config["runtime"]["pilot_memory_path_variant"] = "single_level"
config["runtime"]["pilot_injection_mode"] = "sparse_deep_prefix"
config["runtime"]["pilot_projector_token_source"] = "writer_slots"
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
config["runtime"]["pilot_writer_orthogonalize_slot_basis"] = False
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

receiver_disabled = {
    "enabled": False,
    "target_layers": [],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 0,
    "alpha": 4.0,
    "dropout": 0.0,
}

if arm_id == "control":
    config["runtime"]["shared_injection_arm"] = "base_only"
    config["runtime"]["pilot_arm_alias"] = "C0"
    config["runtime"]["pilot_train_steps"] = 0
    config["runtime"]["pilot_snapshot_steps"] = [0]
    config["runtime"]["pilot_gradient_probe_enabled"] = False
    config["runtime"]["pilot_aux_loss_mode"] = "off"
    config["runtime"]["pilot_active_writer_family"] = "control"
    config["runtime"]["pilot_active_depth_family"] = "none"
    config["runtime"]["pilot_active_projector_family"] = "none"
    config["method"]["receiver_lora"] = receiver_disabled
else:
    spec = arm_specs[arm_id]
    config["runtime"]["shared_injection_arm"] = "injected"
    config["runtime"]["pilot_arm_alias"] = arm_id.upper()
    config["runtime"]["pilot_train_steps"] = train_steps
    config["runtime"]["pilot_snapshot_steps"] = [0, 10, 25, 50, 100, 150, train_steps]
    config["runtime"]["pilot_deep_prefix_layers"] = list(spec["depth_layers"])
    config["runtime"]["pilot_deep_prefix_rank"] = int(spec["projector_rank"])
    config["runtime"]["pilot_active_writer_family"] = spec["writer_family"]
    config["runtime"]["pilot_active_depth_family"] = spec["depth_family"]
    config["runtime"]["pilot_active_projector_family"] = spec["projector_family"]
    config["runtime"]["pilot_projector_mode"] = "shared_low_rank"
    config["method"]["writer"] = dict(spec["writer"])
    config["method"]["receiver_lora"] = {
        "enabled": True,
        "target_layers": list(spec["receiver_layers"]),
        "target_modules": ["k_proj", "v_proj"],
        "rank": 2,
        "alpha": 4.0,
        "dropout": 0.0,
    }

config["experiment"]["name"] = f"{os.environ['PLANV7_EXPERIMENT_PREFIX']}_v7_1_{task_name}_{arm_id}"
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

for arm_id in s00 s01 s10 s11; do
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
    "pilot-A-selected:C0:base_only:real:0"
  copy_run_artifacts \
    "${RESULT_ROOT}/control/${task_name}" \
    "${RUN_ROOT}/${task_name}-control" \
    "pilot-A-selected"
  seed_offset=$((seed_offset + 10))
done

for arm_id in s00 s01 s10 s11; do
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
    seed_offset=$((seed_offset + 10))
  done
done

python scripts/update_planv7_v7_1_width_depth_summary.py \
  --result_root "${RESULT_ROOT}" \
  --output_json "${RESULT_ROOT}/v7-1-summary.json" \
  --output_report "${RESULT_ROOT}/v7-1-summary.md"

bash scripts/publish_review_artifacts.sh

echo "PLANv7 V7-1 width-depth scout complete."
