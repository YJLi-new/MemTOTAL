#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/tmp/memtotal-runs/planv6-v6-2-support-screening-qwen25}"
RESULT_ROOT="${3:-/tmp/memtotal-results/planv6-v6-2-support-screening-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
FREEZE_STEPS="${5:-10}"
TRAIN_STEPS="${6:-200}"
MODEL_DIR="${7:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"

export HF_HOME="${HF_HOME:-/tmp/memtotal-hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  --seed "${BASE_SEED}"

materialize_config() {
  local src_config="$1"
  local output_config="$2"
  local support_path="$3"
  local train_path="$4"
  local eval_path="$5"
  local variant="$6"
  local task_name="$7"
  python - "${src_config}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${variant}" "${task_name}" "${FREEZE_STEPS}" "${TRAIN_STEPS}" "${MODEL_DIR}" <<'PY'
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
task_name = sys.argv[7]
freeze_steps = max(0, int(sys.argv[8]))
train_steps = max(1, int(sys.argv[9]))
model_dir = str(Path(sys.argv[10]).resolve())

config = load_config(source_config)
config.setdefault("task", {})
config.setdefault("method", {})
config.setdefault("runtime", {})
config.setdefault("backbone", {})
config["backbone"]["model_id"] = model_dir
config["task"]["support_dataset_path"] = support_path
config["task"]["train_dataset_path"] = train_path
config["task"]["train_support_dataset_path"] = support_path
config["task"]["dataset_path"] = eval_path
config["task"]["support_lookup_dataset_paths"] = []
config["task"]["train_support_episode_bank_path"] = ""
config["task"]["pilot_split"] = str(config["task"].get("smoke_subset", config["task"].get("split", "eval")))

receiver_lora_disabled = {
    "enabled": False,
    "target_layers": [],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 0,
    "alpha": 4.0,
    "dropout": 0.0,
}
receiver_lora_early4 = {
    "enabled": True,
    "target_layers": [0, 1, 2, 3],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 2,
    "alpha": 4.0,
    "dropout": 0.0,
}

if variant == "control":
    config["runtime"]["shared_injection_arm"] = "base_only"
    config["runtime"]["writer_memory_control"] = "real"
    config["runtime"]["pilot_arm_alias"] = "A"
    config["method"]["receiver_lora"] = receiver_lora_disabled
else:
    screen_variants = {
        "s0_pooled_block_legacy": {
            "arm_alias": "V6_2_S0_C1_L0",
            "support_mode": "pooled_block",
            "stimulus_mode": "support_and_context",
            "balance_mode": "off",
            "context_scale_init": 1.0,
            "support_scale_init": 1.0,
        },
        "s1_pooled_block_gated": {
            "arm_alias": "V6_2_S1_C2_L0",
            "support_mode": "pooled_block",
            "stimulus_mode": "support_and_context",
            "balance_mode": "layernorm_learned_scalar",
            "context_scale_init": 0.75,
            "support_scale_init": 1.25,
        },
        "s2_structured_support_set": {
            "arm_alias": "V6_2_S2_C1_L0",
            "support_mode": "structured_support_set",
            "stimulus_mode": "support_and_context",
            "balance_mode": "off",
            "context_scale_init": 1.0,
            "support_scale_init": 1.0,
        },
        "s3_multi_item_cross_attn_raw": {
            "arm_alias": "V6_2_S3_C0_L0",
            "support_mode": "multi_item_cross_attn_raw",
            "stimulus_mode": "support_only",
            "balance_mode": "off",
            "context_scale_init": 1.0,
            "support_scale_init": 1.0,
        },
        "s4_multi_item_cross_attn_encoded": {
            "arm_alias": "V6_2_S4_C1_L0",
            "support_mode": "multi_item_cross_attn_encoded",
            "stimulus_mode": "support_and_context",
            "balance_mode": "off",
            "context_scale_init": 1.0,
            "support_scale_init": 1.0,
        },
        "s5_hybrid_pooled_plus_items": {
            "arm_alias": "V6_2_S5_C2_L0",
            "support_mode": "hybrid_pooled_plus_items",
            "stimulus_mode": "support_and_context",
            "balance_mode": "layernorm_learned_scalar",
            "context_scale_init": 0.75,
            "support_scale_init": 1.25,
        },
    }
    if variant not in screen_variants:
        raise ValueError(f"unsupported materialization variant: {variant}")
    variant_config = screen_variants[variant]
    config["runtime"]["shared_injection_arm"] = "injected"
    config["runtime"]["writer_memory_control"] = "real"
    config["runtime"]["pilot_arm_alias"] = variant_config["arm_alias"]
    config["runtime"]["pilot_prefix_source_mode"] = "writer"
    config["runtime"]["pilot_support_encoder_mode"] = variant_config["support_mode"]
    config["runtime"]["pilot_writer_stimulus_mode"] = variant_config["stimulus_mode"]
    config["runtime"]["pilot_context_support_balance_mode"] = variant_config["balance_mode"]
    config["runtime"]["pilot_context_balance_scale_init"] = variant_config["context_scale_init"]
    config["runtime"]["pilot_support_balance_scale_init"] = variant_config["support_scale_init"]
    config["runtime"]["pilot_train_steps"] = train_steps
    config["runtime"]["pilot_snapshot_steps"] = sorted({0, 10, 25, 50, 100, 150, train_steps})
    config["runtime"]["pilot_lr_schedule"] = "constant_with_linear_warmup"
    config["runtime"]["pilot_lr_warmup_steps"] = min(10, train_steps)
    config["runtime"]["pilot_projector_warmup_steps"] = min(freeze_steps, train_steps)
    config["runtime"]["pilot_writer_learning_rate"] = 1.0e-4
    config["runtime"]["pilot_support_encoder_learning_rate"] = 7.5e-5
    config["runtime"]["pilot_projector_learning_rate"] = 5.0e-5
    config["runtime"]["pilot_receiver_lora_learning_rate"] = 5.0e-5
    config["runtime"]["pilot_writer_weight_decay"] = 0.0
    config["runtime"]["pilot_support_encoder_weight_decay"] = 0.0
    config["runtime"]["pilot_projector_weight_decay"] = 0.0
    config["runtime"]["pilot_receiver_lora_weight_decay"] = 0.0
    config["runtime"]["pilot_gradient_clip_norm"] = 1.0
    config["runtime"]["pilot_groupwise_grad_clip"] = True
    config["runtime"]["pilot_writer_grad_clip_norm"] = 1.0
    config["runtime"]["pilot_projector_grad_clip_norm"] = 0.5
    config["runtime"]["pilot_receiver_lora_grad_clip_norm"] = 0.5
    config["runtime"]["pilot_gradient_probe_enabled"] = True
    config["runtime"]["pilot_gradient_probe_interval"] = 5
    config["runtime"]["pilot_gradient_probe_max_steps"] = min(120, train_steps)
    config["runtime"]["pilot_gradient_probe_modules"] = [
        "writer",
        "support_encoder",
        "projector",
        "receiver_lora",
    ]
    config["runtime"]["pilot_writer_gain_margin"] = 0.0
    config["runtime"]["pilot_writer_gain_margin_weight"] = 0.0
    config["runtime"]["pilot_writer_common_mode_penalty_weight"] = 0.0
    config["runtime"]["pilot_writer_covariance_diversity_weight"] = 0.0
    config["runtime"]["pilot_writer_slot_energy_balance_weight"] = 0.0
    config["runtime"]["pilot_support_encoder_num_heads"] = int(
        config["runtime"].get("pilot_support_encoder_num_heads", 4)
    )
    config["method"]["receiver_lora"] = receiver_lora_early4

config["experiment"]["name"] = f"planv6_v6_2_{task_name}_{variant}"
output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

for task_name in gsm8k narrativeqa fever; do
  materialize_config \
    "configs/exp/writer_circuit_g2_writer_direct_${task_name}_template.yaml" \
    "${CONFIG_ROOT}/${task_name}-control.json" \
    "${DATA_ROOT}/${task_name}/support.jsonl" \
    "${DATA_ROOT}/${task_name}/train.jsonl" \
    "${DATA_ROOT}/${task_name}/eval.jsonl" \
    "control" \
    "${task_name}"
done

SUPPORT_VARIANTS=(
  "s0_pooled_block_legacy"
  "s1_pooled_block_gated"
  "s2_structured_support_set"
  "s3_multi_item_cross_attn_raw"
  "s4_multi_item_cross_attn_encoded"
  "s5_hybrid_pooled_plus_items"
)

for task_name in gsm8k narrativeqa fever; do
  for variant in "${SUPPORT_VARIANTS[@]}"; do
    materialize_config \
      "configs/exp/writer_circuit_g2_writer_direct_${task_name}_template.yaml" \
      "${CONFIG_ROOT}/${task_name}-${variant}.json" \
      "${DATA_ROOT}/${task_name}/support.jsonl" \
      "${DATA_ROOT}/${task_name}/train.jsonl" \
      "${DATA_ROOT}/${task_name}/eval.jsonl" \
      "${variant}" \
      "${task_name}"
  done
done

if [[ ! -f "${MODEL_DIR}/model.safetensors" || ! -f "${MODEL_DIR}/config.json" ]]; then
  ./scripts/prepare_local_qwen25_model.sh \
    "Qwen/Qwen2.5-1.5B-Instruct" \
    "${MODEL_DIR}" \
    "${HF_HOME}"
fi

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

TASKS=(fever gsm8k narrativeqa)
task_seed_offset() {
  case "$1" in
    fever) echo 0 ;;
    gsm8k) echo 1000 ;;
    narrativeqa) echo 2000 ;;
    *) echo 9000 ;;
  esac
}

variant_seed_offset() {
  case "$1" in
    s0_pooled_block_legacy) echo 100 ;;
    s1_pooled_block_gated) echo 200 ;;
    s2_structured_support_set) echo 300 ;;
    s3_multi_item_cross_attn_raw) echo 400 ;;
    s4_multi_item_cross_attn_encoded) echo 500 ;;
    s5_hybrid_pooled_plus_items) echo 600 ;;
    *) echo 900 ;;
  esac
}

for task_name in "${TASKS[@]}"; do
  task_offset="$(task_seed_offset "${task_name}")"
  ensure_suite_complete \
    "${CONFIG_ROOT}/${task_name}-control.json" \
    "$((BASE_SEED + task_offset))" \
    "${RUN_ROOT}/${task_name}-control" \
    "pilot-A-selected:A:base_only:real:0"
  for variant in "${SUPPORT_VARIANTS[@]}"; do
    ensure_suite_complete \
      "${CONFIG_ROOT}/${task_name}-${variant}.json" \
      "$((BASE_SEED + task_offset + $(variant_seed_offset "${variant}")))" \
      "${RUN_ROOT}/${task_name}-${variant}" \
      "pilot-I-${variant}:$(python - "${CONFIG_ROOT}/${task_name}-${variant}.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(str(payload["runtime"]["pilot_arm_alias"]))
PY
):injected:real:0"
  done
done

cp "${DATA_ROOT}/split-manifest.json" "${RESULT_ROOT}/split-manifest.json"

copy_run_artifacts() {
  local dst_dir="$1"
  local run_dir="$2"
  local pilot_subdir="$3"
  mkdir -p "${dst_dir}"
  cp "${run_dir}/${pilot_subdir}/metrics.json" "${dst_dir}/metrics.json"
  cp "${run_dir}/${pilot_subdir}/train_events.json" "${dst_dir}/train_events.json"
  cp "${run_dir}/${pilot_subdir}/task_case_dump.jsonl" "${dst_dir}/task_case_dump.jsonl"
  cp "${run_dir}/suite_metrics.json" "${dst_dir}/suite_metrics.json"
}

for task_name in "${TASKS[@]}"; do
  copy_run_artifacts "${RESULT_ROOT}/${task_name}/control" "${RUN_ROOT}/${task_name}-control" "pilot-A-selected"
  for variant in "${SUPPORT_VARIANTS[@]}"; do
    copy_run_artifacts \
      "${RESULT_ROOT}/${task_name}/${variant}" \
      "${RUN_ROOT}/${task_name}-${variant}" \
      "pilot-I-${variant}"
  done
done

python scripts/update_planv6_v6_2_support_screening_summary.py \
  --result-root "${RESULT_ROOT}" \
  --output-json "${RESULT_ROOT}/v6-2-summary.json" \
  --output-report "${RESULT_ROOT}/v6-2-summary.md"
