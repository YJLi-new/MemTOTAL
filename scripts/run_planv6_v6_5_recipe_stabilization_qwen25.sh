#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv6-v6-5-recipe-stabilization-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv6-v6-5-recipe-stabilization-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
TRAIN_STEPS="${5:-200}"
MODEL_DIR="${6:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
V64_SUMMARY_JSON="${7:-results/generated/review/planv6-v6-4-mixed-matrix-qwen25/v6-4-summary.json}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SYSTEM_TMPDIR="${SYSTEM_TMPDIR:-/tmp/memtotal-tmp}"
FALLBACK_TMPDIR="${FALLBACK_TMPDIR:-/root/autodl-tmp/tmp}"
MIN_SYSTEM_TMP_FREE_KB="${MIN_SYSTEM_TMP_FREE_KB:-4194304}"

select_tmpdir() {
  if [[ "${FORCE_DATA_TMPDIR:-0}" == "1" ]]; then
    printf '%s\n' "${FALLBACK_TMPDIR}"
    return
  fi
  mkdir -p "${SYSTEM_TMPDIR}" "${FALLBACK_TMPDIR}"
  if [[ -w "${SYSTEM_TMPDIR}" ]]; then
    local free_kb
    free_kb="$(df -Pk "${SYSTEM_TMPDIR}" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -n "${free_kb}" && "${free_kb}" -ge "${MIN_SYSTEM_TMP_FREE_KB}" ]]; then
      printf '%s\n' "${SYSTEM_TMPDIR}"
      return
    fi
  fi
  printf '%s\n' "${FALLBACK_TMPDIR}"
}

TMPDIR="${TMPDIR:-$(select_tmpdir)}"
export TMPDIR
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}" "${TMPDIR}" "${FALLBACK_TMPDIR}"

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

python - "${V64_SUMMARY_JSON}" "${RESULT_ROOT}/screen-manifest.json" "${BASE_SEED}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
base_seed = int(sys.argv[3])
default_finalists = [
    "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage",
    "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg",
    "s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive",
]
if summary_path.exists():
    payload = json.loads(summary_path.read_text())
    finalists = payload.get("finalist_configs", default_finalists)
    if not isinstance(finalists, list) or not finalists:
        finalists = default_finalists
else:
    finalists = default_finalists

finalist_entries = []
for index, combo_id in enumerate(finalists[:3], start=1):
    support_mode_id, stimulus_id, loss_id = combo_id.split("__", 2)
    finalist_entries.append(
        {
            "alias": f"F{index}",
            "combo_id": combo_id,
            "support_mode_id": support_mode_id,
            "stimulus_id": stimulus_id,
            "loss_id": loss_id,
        }
    )

warmup_variants = [
    {"alias": "w0", "steps": 0},
    {"alias": "w10", "steps": 10},
    {"alias": "w20", "steps": 20},
]
clipping_variants = [
    {"alias": "clip_global", "scheme": "global"},
    {"alias": "clip_groupwise", "scheme": "groupwise"},
]
projector_lr_variants = [
    {"alias": "plr5e5", "value": 5.0e-5},
    {"alias": "plr75e6", "value": 7.5e-5},
]
accumulation_variants = [
    {"alias": "acc1", "value": 1},
    {"alias": "acc4", "value": 4},
]
layer_variants = [
    {
        "alias": "layers_base",
        "deep_prefix_layers": [0, 1, 2, 3],
        "receiver_lora_layers": [0, 1, 2, 3],
    },
    {
        "alias": "layers_additive",
        "deep_prefix_layers": [0, 1, 2, 3, 4, 8, 14],
        "receiver_lora_layers": [0, 1, 2, 3, 4],
    },
]

screen_recipes = []
for finalist in finalist_entries:
    for warmup in warmup_variants:
        for clipping in clipping_variants:
            for projector_lr in projector_lr_variants:
                for accumulation in accumulation_variants:
                    for layer_variant in layer_variants:
                        recipe_id = "__".join(
                            [
                                finalist["alias"],
                                warmup["alias"],
                                clipping["alias"],
                                projector_lr["alias"],
                                accumulation["alias"],
                                layer_variant["alias"],
                            ]
                        )
                        screen_recipes.append(
                            {
                                **finalist,
                                "recipe_id": recipe_id,
                                "warmup_alias": warmup["alias"],
                                "warmup_steps": warmup["steps"],
                                "clipping_alias": clipping["alias"],
                                "clipping_scheme": clipping["scheme"],
                                "projector_lr_alias": projector_lr["alias"],
                                "projector_learning_rate": projector_lr["value"],
                                "accumulation_alias": accumulation["alias"],
                                "accumulation_steps": accumulation["value"],
                                "layer_alias": layer_variant["alias"],
                                "layer_variant": layer_variant["alias"],
                                "deep_prefix_layers": layer_variant["deep_prefix_layers"],
                                "receiver_lora_layers": layer_variant["receiver_lora_layers"],
                            }
                        )

manifest = {
    "finalists": finalist_entries,
    "screen_recipes": screen_recipes,
    "confirmation_recipe_ids": [],
    "confirmation_seeds": [base_seed + offset for offset in range(3)],
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

materialize_config() {
  local output_config="$1"
  local support_path="$2"
  local train_path="$3"
  local eval_path="$4"
  local screen_manifest="$5"
  local recipe_id="$6"
  local task_name="$7"
  local train_steps="$8"
  local model_dir="$9"
  python - "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${screen_manifest}" "${recipe_id}" "${task_name}" "${train_steps}" "${model_dir}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

output_config = Path(sys.argv[1])
support_path = str(Path(sys.argv[2]).resolve())
train_path = str(Path(sys.argv[3]).resolve())
eval_path = str(Path(sys.argv[4]).resolve())
screen_manifest = Path(sys.argv[5])
recipe_id = sys.argv[6]
task_name = sys.argv[7]
train_steps = max(1, int(sys.argv[8]))
model_dir = str(Path(sys.argv[9]).resolve())

manifest = json.loads(screen_manifest.read_text())
recipe = next(recipe for recipe in manifest["screen_recipes"] if recipe["recipe_id"] == recipe_id)

config = load_config(f"configs/exp/writer_circuit_g2_writer_direct_{task_name}_template.yaml")
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

config["runtime"]["pilot_train_steps"] = train_steps
config["runtime"]["pilot_snapshot_steps"] = sorted({0, 10, 25, 50, 100, 150, train_steps})
config["runtime"]["pilot_lr_schedule"] = "constant_with_linear_warmup"
config["runtime"]["pilot_lr_warmup_steps"] = int(recipe["warmup_steps"])
config["runtime"]["pilot_projector_warmup_steps"] = int(recipe["warmup_steps"])
config["runtime"]["pilot_writer_learning_rate"] = 1.0e-4
config["runtime"]["pilot_support_encoder_learning_rate"] = 7.5e-5
config["runtime"]["pilot_projector_learning_rate"] = float(recipe["projector_learning_rate"])
config["runtime"]["pilot_receiver_lora_learning_rate"] = 5.0e-5
config["runtime"]["pilot_writer_weight_decay"] = 0.0
config["runtime"]["pilot_support_encoder_weight_decay"] = 0.0
config["runtime"]["pilot_projector_weight_decay"] = 0.0
config["runtime"]["pilot_receiver_lora_weight_decay"] = 0.0
config["runtime"]["pilot_gradient_clip_norm"] = 1.0
config["runtime"]["pilot_groupwise_grad_clip"] = bool(recipe["clipping_scheme"] == "groupwise")
config["runtime"]["pilot_writer_grad_clip_norm"] = 1.0
config["runtime"]["pilot_projector_grad_clip_norm"] = 0.5
config["runtime"]["pilot_receiver_lora_grad_clip_norm"] = 0.5
config["runtime"]["pilot_gradient_probe_enabled"] = True
config["runtime"]["pilot_gradient_probe_interval"] = 5
config["runtime"]["pilot_gradient_probe_max_steps"] = min(150, train_steps)
config["runtime"]["pilot_gradient_accumulation_steps"] = int(recipe["accumulation_steps"])
config["runtime"]["pilot_prefix_source_mode"] = "writer"
config["runtime"]["pilot_support_encoder_mode"] = (
    "multi_item_cross_attn_raw"
    if recipe["support_mode_id"] == "s3_multi_item_cross_attn_raw"
    else "hybrid_pooled_plus_items"
)
config["runtime"]["pilot_writer_stimulus_mode"] = (
    "support_and_context" if recipe["stimulus_id"] == "c2_support_and_context_gated" else "support_only"
)
config["runtime"]["pilot_context_support_balance_mode"] = (
    "layernorm_learned_scalar" if recipe["stimulus_id"] == "c2_support_and_context_gated" else "off"
)
config["runtime"]["pilot_context_balance_scale_init"] = (
    0.75 if recipe["stimulus_id"] == "c2_support_and_context_gated" else 1.0
)
config["runtime"]["pilot_support_balance_scale_init"] = (
    1.25 if recipe["stimulus_id"] == "c2_support_and_context_gated" else 1.0
)
config["runtime"]["pilot_aux_loss_mode"] = str(recipe["loss_id"])
config["runtime"]["pilot_deep_prefix_layers"] = list(recipe["deep_prefix_layers"])
config["method"]["receiver_lora"] = {
    "enabled": True,
    "target_layers": list(recipe["receiver_lora_layers"]),
    "target_modules": ["k_proj", "v_proj"],
    "rank": 2,
    "alpha": 4.0,
    "dropout": 0.0,
}
config["runtime"]["shared_injection_arm"] = "injected"
config["runtime"]["writer_memory_control"] = "real"
config["runtime"]["pilot_arm_alias"] = str(recipe["recipe_id"])
config["experiment"]["name"] = f"planv6_v6_5_{task_name}_{recipe['recipe_id']}"

loss_id = str(recipe["loss_id"])
if loss_id == "l2_contrastive":
    config["runtime"]["pilot_aux_loss_mode"] = "contrastive"
    config["runtime"]["pilot_aux_projection_dim"] = 128
    config["runtime"]["pilot_aux_projection_hidden_dim"] = 256
    config["runtime"]["pilot_support_row_dropout"] = 0.25
    config["runtime"]["pilot_context_token_dropout"] = 0.15
    config["runtime"]["pilot_contrastive_loss_weight"] = 0.05
    config["runtime"]["pilot_vicreg_loss_weight"] = 0.0
    config["runtime"]["pilot_barlow_loss_weight"] = 0.0
    config["runtime"]["pilot_writer_slot_orthogonality_weight"] = 0.0
    config["runtime"]["pilot_writer_support_coverage_weight"] = 0.0
elif loss_id == "l3_vicreg":
    config["runtime"]["pilot_aux_loss_mode"] = "vicreg"
    config["runtime"]["pilot_aux_projection_dim"] = 128
    config["runtime"]["pilot_aux_projection_hidden_dim"] = 256
    config["runtime"]["pilot_support_row_dropout"] = 0.25
    config["runtime"]["pilot_context_token_dropout"] = 0.15
    config["runtime"]["pilot_contrastive_loss_weight"] = 0.0
    config["runtime"]["pilot_vicreg_loss_weight"] = 0.05
    config["runtime"]["pilot_barlow_loss_weight"] = 0.0
    config["runtime"]["pilot_writer_slot_orthogonality_weight"] = 0.0
    config["runtime"]["pilot_writer_support_coverage_weight"] = 0.0
elif loss_id == "l5_orthogonality_coverage":
    config["runtime"]["pilot_aux_loss_mode"] = "orthogonality_coverage"
    config["runtime"]["pilot_aux_projection_dim"] = 0
    config["runtime"]["pilot_aux_projection_hidden_dim"] = None
    config["runtime"]["pilot_support_row_dropout"] = 0.0
    config["runtime"]["pilot_context_token_dropout"] = 0.0
    config["runtime"]["pilot_contrastive_loss_weight"] = 0.0
    config["runtime"]["pilot_vicreg_loss_weight"] = 0.0
    config["runtime"]["pilot_barlow_loss_weight"] = 0.0
    config["runtime"]["pilot_writer_slot_orthogonality_weight"] = 0.05
    config["runtime"]["pilot_writer_support_coverage_weight"] = 0.05
else:
    raise ValueError(f"Unsupported loss_id={loss_id}")

output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

write_control_config() {
  local output_config="$1"
  local support_path="$2"
  local train_path="$3"
  local eval_path="$4"
  local task_name="$5"
  local model_dir="$6"
  python - "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${task_name}" "${model_dir}" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

output_config = Path(sys.argv[1])
support_path = str(Path(sys.argv[2]).resolve())
train_path = str(Path(sys.argv[3]).resolve())
eval_path = str(Path(sys.argv[4]).resolve())
task_name = sys.argv[5]
model_dir = str(Path(sys.argv[6]).resolve())

config = load_config(f"configs/exp/writer_circuit_g2_writer_direct_{task_name}_template.yaml")
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
config["runtime"]["shared_injection_arm"] = "base_only"
config["runtime"]["writer_memory_control"] = "real"
config["runtime"]["pilot_arm_alias"] = "A"
config["method"]["receiver_lora"] = {
    "enabled": False,
    "target_layers": [],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 0,
    "alpha": 4.0,
    "dropout": 0.0,
}
config["experiment"]["name"] = f"planv6_v6_5_{task_name}_control"
output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

write_control_config \
  "${CONFIG_ROOT}/fever-control.json" \
  "${DATA_ROOT}/fever/support.jsonl" \
  "${DATA_ROOT}/fever/train.jsonl" \
  "${DATA_ROOT}/fever/eval.jsonl" \
  "fever" \
  "${MODEL_DIR}"

mapfile -t RECIPE_IDS < <(python - "${RESULT_ROOT}/screen-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for recipe in manifest.get("screen_recipes", []):
    print(recipe["recipe_id"])
PY
)

for recipe_id in "${RECIPE_IDS[@]}"; do
  materialize_config \
    "${CONFIG_ROOT}/${recipe_id}.json" \
    "${DATA_ROOT}/fever/support.jsonl" \
    "${DATA_ROOT}/fever/train.jsonl" \
    "${DATA_ROOT}/fever/eval.jsonl" \
    "${RESULT_ROOT}/screen-manifest.json" \
    "${recipe_id}" \
    "fever" \
    "${TRAIN_STEPS}" \
    "${MODEL_DIR}"
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

ensure_suite_complete \
  "${CONFIG_ROOT}/fever-control.json" \
  "$((BASE_SEED + 1000))" \
  "${RUN_ROOT}/control" \
  "pilot-A-selected:A:base_only:real:0"
copy_run_artifacts \
  "${RESULT_ROOT}/control" \
  "${RUN_ROOT}/control" \
  "pilot-A-selected"

screen_index=0
for recipe_id in "${RECIPE_IDS[@]}"; do
  screen_seed=$((BASE_SEED + 4000 + screen_index))
  ensure_suite_complete \
    "${CONFIG_ROOT}/${recipe_id}.json" \
    "${screen_seed}" \
    "${RUN_ROOT}/screen/${recipe_id}" \
    "pilot-I-screen-${screen_index}:${recipe_id}:injected:real:0"
  copy_run_artifacts \
    "${RESULT_ROOT}/screen/${recipe_id}" \
    "${RUN_ROOT}/screen/${recipe_id}" \
    "pilot-I-screen-${screen_index}"
  screen_index=$((screen_index + 1))
done

cp "${DATA_ROOT}/split-manifest.json" "${RESULT_ROOT}/split-manifest.json"

python scripts/update_planv6_v6_5_recipe_stabilization_summary.py \
  --result-root "${RESULT_ROOT}" \
  --output-json "${RESULT_ROOT}/v6-5-summary.json" \
  --output-report "${RESULT_ROOT}/v6-5-summary.md"

./scripts/publish_review_artifacts.sh

./scripts/publish_review_artifacts.sh

mapfile -t TOP_CONFIRM_RECIPE_IDS < <(python - "${RESULT_ROOT}/v6-5-summary.json" "${RESULT_ROOT}/screen-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
manifest_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
top_two = summary.get("screen_top_two_recipes", [])[:2]
manifest["confirmation_recipe_ids"] = [str(value) for value in top_two]
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
for recipe_id in top_two:
    print(str(recipe_id))
PY
)

for confirm_offset in 0 1 2; do
  confirm_seed=$((BASE_SEED + confirm_offset))
  ensure_suite_complete \
    "${CONFIG_ROOT}/fever-control.json" \
    "${confirm_seed}" \
    "${RUN_ROOT}/confirm/seed_${confirm_seed}/control" \
    "pilot-A-selected:A:base_only:real:0"
  copy_run_artifacts \
    "${RESULT_ROOT}/confirm/seed_${confirm_seed}/control" \
    "${RUN_ROOT}/confirm/seed_${confirm_seed}/control" \
    "pilot-A-selected"
  for recipe_id in "${TOP_CONFIRM_RECIPE_IDS[@]}"; do
    ensure_suite_complete \
      "${CONFIG_ROOT}/${recipe_id}.json" \
      "${confirm_seed}" \
      "${RUN_ROOT}/confirm/seed_${confirm_seed}/${recipe_id}" \
      "pilot-I-confirm-${confirm_seed}:${recipe_id}:injected:real:0"
    copy_run_artifacts \
      "${RESULT_ROOT}/confirm/seed_${confirm_seed}/${recipe_id}" \
      "${RUN_ROOT}/confirm/seed_${confirm_seed}/${recipe_id}" \
      "pilot-I-confirm-${confirm_seed}"
  done
done

python scripts/update_planv6_v6_5_recipe_stabilization_summary.py \
  --result-root "${RESULT_ROOT}" \
  --output-json "${RESULT_ROOT}/v6-5-summary.json" \
  --output-report "${RESULT_ROOT}/v6-5-summary.md"
