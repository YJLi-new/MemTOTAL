#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv6-v6-3-loss-screening-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv6-v6-3-loss-screening-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
FREEZE_STEPS="${5:-10}"
TRAIN_STEPS="${6:-200}"
MODEL_DIR="${7:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
V62_SUMMARY_JSON="${8:-results/generated/review/planv6-v6-2-support-screening-qwen25/v6-2-summary.json}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
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

mapfile -t TOP_SUPPORT_VARIANTS < <(python - "${V62_SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
default = ["s3_multi_item_cross_attn_raw", "s5_hybrid_pooled_plus_items"]
if not summary_path.exists():
    print("\n".join(default))
    raise SystemExit
payload = json.loads(summary_path.read_text())
values = payload.get("top_two_support_modes", default)
if not isinstance(values, list) or len(values) < 2:
    values = default
for value in values[:2]:
    print(str(value))
PY
)
if [[ "${#TOP_SUPPORT_VARIANTS[@]}" -lt 2 ]]; then
  TOP_SUPPORT_VARIANTS=("s3_multi_item_cross_attn_raw" "s5_hybrid_pooled_plus_items")
fi

LOSS_VARIANTS=(
  "l0_task_only"
  "l1_legacy"
  "l2_contrastive"
  "l3_vicreg"
  "l5_orthogonality_coverage"
)

materialize_config() {
  local src_config="$1"
  local output_config="$2"
  local support_path="$3"
  local train_path="$4"
  local eval_path="$5"
  local support_variant="$6"
  local loss_variant="$7"
  local task_name="$8"
  python - "${src_config}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${support_variant}" "${loss_variant}" "${task_name}" "${FREEZE_STEPS}" "${TRAIN_STEPS}" "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

source_config = sys.argv[1]
output_config = Path(sys.argv[2])
support_path = str(Path(sys.argv[3]).resolve())
train_path = str(Path(sys.argv[4]).resolve())
eval_path = str(Path(sys.argv[5]).resolve())
support_variant = sys.argv[6]
loss_variant = sys.argv[7]
task_name = sys.argv[8]
freeze_steps = max(0, int(sys.argv[9]))
train_steps = max(1, int(sys.argv[10]))
model_dir = str(Path(sys.argv[11]).resolve())

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

support_variants = {
    "s3_multi_item_cross_attn_raw": {
        "arm_alias": "V6_3_S3",
        "support_mode": "multi_item_cross_attn_raw",
        "stimulus_mode": "support_only",
        "balance_mode": "off",
        "context_scale_init": 1.0,
        "support_scale_init": 1.0,
    },
    "s5_hybrid_pooled_plus_items": {
        "arm_alias": "V6_3_S5",
        "support_mode": "hybrid_pooled_plus_items",
        "stimulus_mode": "support_and_context",
        "balance_mode": "layernorm_learned_scalar",
        "context_scale_init": 0.75,
        "support_scale_init": 1.25,
    },
}

loss_variants = {
    "l0_task_only": {
        "arm_suffix": "L0",
        "aux_mode": "task_only",
        "aux_projection_dim": 0,
        "aux_projection_hidden_dim": None,
        "support_row_dropout": 0.0,
        "context_token_dropout": 0.0,
        "contrastive_loss_weight": 0.0,
        "vicreg_loss_weight": 0.0,
        "barlow_loss_weight": 0.0,
        "writer_slot_orthogonality_weight": 0.0,
        "writer_support_coverage_weight": 0.0,
        "writer_gain_margin_weight": 0.0,
        "writer_common_mode_penalty_weight": 0.0,
        "writer_covariance_diversity_weight": 0.0,
        "writer_slot_energy_balance_weight": 0.0,
    },
    "l1_legacy": {
        "arm_suffix": "L1",
        "aux_mode": "legacy",
        "aux_projection_dim": 0,
        "aux_projection_hidden_dim": None,
        "support_row_dropout": 0.0,
        "context_token_dropout": 0.0,
        "contrastive_loss_weight": 0.0,
        "vicreg_loss_weight": 0.0,
        "barlow_loss_weight": 0.0,
        "writer_slot_orthogonality_weight": 0.0,
        "writer_support_coverage_weight": 0.0,
        "writer_gain_margin_weight": 0.25,
        "writer_common_mode_penalty_weight": 0.10,
        "writer_covariance_diversity_weight": 0.05,
        "writer_slot_energy_balance_weight": 0.01,
    },
    "l2_contrastive": {
        "arm_suffix": "L2",
        "aux_mode": "contrastive",
        "aux_projection_dim": 128,
        "aux_projection_hidden_dim": 256,
        "support_row_dropout": 0.25,
        "context_token_dropout": 0.15,
        "contrastive_loss_weight": 0.05,
        "vicreg_loss_weight": 0.0,
        "barlow_loss_weight": 0.0,
        "writer_slot_orthogonality_weight": 0.0,
        "writer_support_coverage_weight": 0.0,
        "writer_gain_margin_weight": 0.0,
        "writer_common_mode_penalty_weight": 0.0,
        "writer_covariance_diversity_weight": 0.0,
        "writer_slot_energy_balance_weight": 0.0,
    },
    "l3_vicreg": {
        "arm_suffix": "L3",
        "aux_mode": "vicreg",
        "aux_projection_dim": 128,
        "aux_projection_hidden_dim": 256,
        "support_row_dropout": 0.25,
        "context_token_dropout": 0.15,
        "contrastive_loss_weight": 0.0,
        "vicreg_loss_weight": 0.05,
        "barlow_loss_weight": 0.0,
        "writer_slot_orthogonality_weight": 0.0,
        "writer_support_coverage_weight": 0.0,
        "writer_gain_margin_weight": 0.0,
        "writer_common_mode_penalty_weight": 0.0,
        "writer_covariance_diversity_weight": 0.0,
        "writer_slot_energy_balance_weight": 0.0,
    },
    "l5_orthogonality_coverage": {
        "arm_suffix": "L5",
        "aux_mode": "orthogonality_coverage",
        "aux_projection_dim": 0,
        "aux_projection_hidden_dim": None,
        "support_row_dropout": 0.0,
        "context_token_dropout": 0.0,
        "contrastive_loss_weight": 0.0,
        "vicreg_loss_weight": 0.0,
        "barlow_loss_weight": 0.0,
        "writer_slot_orthogonality_weight": 0.05,
        "writer_support_coverage_weight": 0.05,
        "writer_gain_margin_weight": 0.0,
        "writer_common_mode_penalty_weight": 0.0,
        "writer_covariance_diversity_weight": 0.0,
        "writer_slot_energy_balance_weight": 0.0,
    },
}

if support_variant == "control":
    config["runtime"]["shared_injection_arm"] = "base_only"
    config["runtime"]["writer_memory_control"] = "real"
    config["runtime"]["pilot_arm_alias"] = "A"
    config["method"]["receiver_lora"] = receiver_lora_disabled
else:
    support_config = support_variants[support_variant]
    loss_config = loss_variants[loss_variant]
    config["runtime"]["shared_injection_arm"] = "injected"
    config["runtime"]["writer_memory_control"] = "real"
    config["runtime"]["pilot_arm_alias"] = f"{support_config['arm_alias']}_{loss_config['arm_suffix']}"
    config["runtime"]["pilot_prefix_source_mode"] = "writer"
    config["runtime"]["pilot_support_encoder_mode"] = support_config["support_mode"]
    config["runtime"]["pilot_writer_stimulus_mode"] = support_config["stimulus_mode"]
    config["runtime"]["pilot_context_support_balance_mode"] = support_config["balance_mode"]
    config["runtime"]["pilot_context_balance_scale_init"] = support_config["context_scale_init"]
    config["runtime"]["pilot_support_balance_scale_init"] = support_config["support_scale_init"]
    config["runtime"]["pilot_aux_loss_mode"] = loss_config["aux_mode"]
    config["runtime"]["pilot_aux_projection_dim"] = loss_config["aux_projection_dim"]
    config["runtime"]["pilot_aux_projection_hidden_dim"] = loss_config["aux_projection_hidden_dim"]
    config["runtime"]["pilot_support_row_dropout"] = loss_config["support_row_dropout"]
    config["runtime"]["pilot_context_token_dropout"] = loss_config["context_token_dropout"]
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
    config["runtime"]["pilot_gradient_probe_max_steps"] = min(150, train_steps)
    config["runtime"]["pilot_gradient_probe_modules"] = [
        "writer",
        "support_encoder",
        "projector",
        "receiver_lora",
    ]
    config["runtime"]["pilot_contrastive_temperature"] = 0.1
    config["runtime"]["pilot_contrastive_loss_weight"] = loss_config["contrastive_loss_weight"]
    config["runtime"]["pilot_contrastive_queue_size"] = 64
    config["runtime"]["pilot_vicreg_invariance_weight"] = 1.0
    config["runtime"]["pilot_vicreg_variance_weight"] = 1.0
    config["runtime"]["pilot_vicreg_covariance_weight"] = 1.0
    config["runtime"]["pilot_vicreg_variance_target"] = 1.0
    config["runtime"]["pilot_vicreg_loss_weight"] = loss_config["vicreg_loss_weight"]
    config["runtime"]["pilot_barlow_loss_weight"] = loss_config["barlow_loss_weight"]
    config["runtime"]["pilot_barlow_lambda"] = 0.005
    config["runtime"]["pilot_writer_slot_orthogonality_weight"] = loss_config["writer_slot_orthogonality_weight"]
    config["runtime"]["pilot_writer_support_coverage_weight"] = loss_config["writer_support_coverage_weight"]
    config["runtime"]["pilot_writer_gain_margin"] = 0.0
    config["runtime"]["pilot_writer_gain_margin_weight"] = loss_config["writer_gain_margin_weight"]
    config["runtime"]["pilot_writer_common_mode_penalty_weight"] = loss_config["writer_common_mode_penalty_weight"]
    config["runtime"]["pilot_writer_covariance_diversity_weight"] = loss_config["writer_covariance_diversity_weight"]
    config["runtime"]["pilot_writer_slot_energy_balance_weight"] = loss_config["writer_slot_energy_balance_weight"]
    config["runtime"]["pilot_support_encoder_num_heads"] = int(
        config["runtime"].get("pilot_support_encoder_num_heads", 4)
    )
    config["method"]["receiver_lora"] = receiver_lora_early4

config["experiment"]["name"] = f"planv6_v6_3_{task_name}_{support_variant}_{loss_variant}"
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
    "l0_task_only" \
    "${task_name}"
done

for task_name in gsm8k narrativeqa fever; do
  for support_variant in "${TOP_SUPPORT_VARIANTS[@]}"; do
    for loss_variant in "${LOSS_VARIANTS[@]}"; do
      materialize_config \
        "configs/exp/writer_circuit_g2_writer_direct_${task_name}_template.yaml" \
        "${CONFIG_ROOT}/${task_name}-${support_variant}-${loss_variant}.json" \
        "${DATA_ROOT}/${task_name}/support.jsonl" \
        "${DATA_ROOT}/${task_name}/train.jsonl" \
        "${DATA_ROOT}/${task_name}/eval.jsonl" \
        "${support_variant}" \
        "${loss_variant}" \
        "${task_name}"
    done
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

for task_name in gsm8k narrativeqa fever; do
  ensure_suite_complete "${CONFIG_ROOT}/${task_name}-control.json" "$((BASE_SEED + 1000))" \
    "${RUN_ROOT}/${task_name}-control" "pilot-A-selected:A:base_only:real:0"
  copy_run_artifacts \
    "${RESULT_ROOT}/${task_name}/control" \
    "${RUN_ROOT}/${task_name}-control" \
    "pilot-A-selected"
done

suite_index=0
for task_name in gsm8k narrativeqa fever; do
  for support_variant in "${TOP_SUPPORT_VARIANTS[@]}"; do
    for loss_variant in "${LOSS_VARIANTS[@]}"; do
      support_tag="${support_variant%%_*}"
      loss_tag="${loss_variant%%_*}"
      combo_id="${support_variant}__${loss_variant}"
      suite_seed=$((BASE_SEED + 2000 + suite_index))
      arm_alias="$(python - "${support_variant}" "${loss_variant}" <<'PY'
import sys
support_variant = sys.argv[1]
loss_variant = sys.argv[2]
support_alias = {
    "s3_multi_item_cross_attn_raw": "V6_3_S3",
    "s5_hybrid_pooled_plus_items": "V6_3_S5",
}[support_variant]
loss_alias = {
    "l0_task_only": "L0",
    "l1_legacy": "L1",
    "l2_contrastive": "L2",
    "l3_vicreg": "L3",
    "l5_orthogonality_coverage": "L5",
}[loss_variant]
print(f"{support_alias}_{loss_alias}")
PY
)"
      ensure_suite_complete \
        "${CONFIG_ROOT}/${task_name}-${support_variant}-${loss_variant}.json" \
        "${suite_seed}" \
        "${RUN_ROOT}/${task_name}-${combo_id}" \
        "pilot-I-${support_tag}-${loss_tag}:${arm_alias}:injected:real:0"
      copy_run_artifacts \
        "${RESULT_ROOT}/${task_name}/${combo_id}" \
        "${RUN_ROOT}/${task_name}-${combo_id}" \
        "pilot-I-${support_tag}-${loss_tag}"
      suite_index=$((suite_index + 1))
    done
  done
done

cp "${DATA_ROOT}/split-manifest.json" "${RESULT_ROOT}/split-manifest.json"

python scripts/update_planv6_v6_3_loss_screening_summary.py \
  --result-root "${RESULT_ROOT}" \
  --output-json "${RESULT_ROOT}/v6-3-summary.json" \
  --output-report "${RESULT_ROOT}/v6-3-summary.md"

./scripts/publish_review_artifacts.sh
