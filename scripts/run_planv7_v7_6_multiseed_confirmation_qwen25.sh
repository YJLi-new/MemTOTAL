#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv7-v7-6-multiseed-confirmation-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
TRAIN_STEPS="${5:-300}"
MODEL_DIR="${6:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
V75_SUMMARY_JSON="${7:-results/generated/review/planv7-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json}"
V70_SUMMARY_JSON="${8:-results/generated/review/planv7-v7-0-metrics-oracle-qwen25/v7-0-summary.json}"

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
  --seed "${BASE_SEED}" \
  --benchmarks "gsm8k,triviaqa,fever"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "${MODEL_DIR}" \
  "${HF_HOME}"

python - "${V75_SUMMARY_JSON}" "${RESULT_ROOT}/selection-manifest.json" "${BASE_SEED}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
base_seed = int(sys.argv[3])
payload = json.loads(summary_path.read_text())
ranking = payload.get("aux_arm_ranking", [])
arms = payload.get("arms", {})
if not isinstance(ranking, list) or not ranking:
    raise SystemExit(f"Missing aux_arm_ranking in {summary_path}")

qualified_ids = [
    str(item.get("arm_id", "")).strip()
    for item in ranking
    if str(item.get("arm_id", "")).strip()
    and bool(arms.get(str(item.get("arm_id", "")).strip(), {}).get("acceptance_qualified", False))
]
selected_ids = qualified_ids[:2]
if not selected_ids:
    selected_ids = [str(ranking[0].get("arm_id", "")).strip()]
if len(selected_ids) == 1 and len(ranking) > 1:
    next_id = str(ranking[1].get("arm_id", "")).strip()
    if next_id and next_id not in selected_ids:
        selected_ids.append(next_id)

base_arm_id = str(payload.get("base_from_v7_4_arm_id", "")).strip()
base_source_phase = str(payload.get("base_from_v7_4_source_phase", "v7_3")).strip() or "v7_3"
control_source_arm_id = str(payload.get("control_source_arm_id", "")).strip()
direct_control_arm_id = str(payload.get("direct_control_arm_id", "")).strip()
winner_uses_bridge = bool(
    direct_control_arm_id
    and (
        base_arm_id.startswith("b_")
        or base_arm_id.startswith("f")
        or control_source_arm_id.startswith("b_")
        or base_source_phase in {"v7_3", "v7_4"}
    )
)

manifest = {
    "seeds": [base_seed + offset for offset in range(3)],
    "promoted_arms": selected_ids,
    "variants": [
        "c0_frozen_no_memory",
        "c1_additive_continuity",
        *(["c2_best_direct"] if winner_uses_bridge else []),
        *[f"p{index + 1}_{arm_id}" for index, arm_id in enumerate(selected_ids)],
    ],
    "branch_variant_map": {
        f"p{index + 1}_{arm_id}": arm_id for index, arm_id in enumerate(selected_ids)
    },
    "base_arm_id": base_arm_id,
    "base_source_phase": base_source_phase,
    "control_source_arm_id": control_source_arm_id,
    "direct_control_arm_id": direct_control_arm_id,
    "winning_depth": str(payload.get("winning_depth", "D1")).strip() or "D1",
    "winning_depth_label": str(payload.get("winning_depth_label", "mid4")).strip() or "mid4",
    "winner_uses_bridge": winner_uses_bridge,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

materialize_config() {
  local task_name="$1"
  local variant_id="$2"
  local output_config="$3"
  local support_path="$4"
  local train_path="$5"
  local eval_path="$6"
  python - "${task_name}" "${variant_id}" "${output_config}" "${support_path}" "${train_path}" "${eval_path}" "${TRAIN_STEPS}" "${MODEL_DIR}" "${RESULT_ROOT}/selection-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from memtotal.utils.config import load_config

task_name = sys.argv[1]
variant_id = sys.argv[2]
output_config = Path(sys.argv[3])
support_path = str(Path(sys.argv[4]).resolve())
train_path = str(Path(sys.argv[5]).resolve())
eval_path = str(Path(sys.argv[6]).resolve())
train_steps = max(0, int(sys.argv[7]))
model_dir = str(Path(sys.argv[8]).resolve())
manifest_path = Path(sys.argv[9]).resolve()
manifest = json.loads(manifest_path.read_text())

config = load_config(f"configs/exp/writer_circuit_g2_writer_direct_{task_name}_template.yaml")
config.setdefault("task", {})
config.setdefault("method", {})
config.setdefault("runtime", {})
config.setdefault("experiment", {})

depth_specs = {
    "D0": {"depth_layers": [0, 1, 2, 3], "receiver_layers": [0, 1, 2, 3]},
    "D1": {"depth_layers": [12, 13, 14, 15], "receiver_layers": [12, 13, 14, 15]},
    "D2": {"depth_layers": [10, 11, 12, 13, 14, 15], "receiver_layers": [12, 13, 14, 15]},
    "D3": {"depth_layers": [0, 1, 2, 3, 12, 13, 14, 15], "receiver_layers": [12, 13, 14, 15]},
}
winning_depth = str(manifest.get("winning_depth", "D1")).strip() or "D1"
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
f4_dynamic_budget_specs = {
    "gsm8k": {
        "writer_family": "W3",
        "bridge_family": "B2",
        "projector_family": "P2",
        "writer": w3_writer,
        "projector_rank": 128,
        "projector_mode": "per_layer_low_rank",
        "reader_queries": 16,
        "short_slots": 16,
    },
    "triviaqa": {
        "writer_family": "W2",
        "bridge_family": "B1_dyn",
        "projector_family": "P2_dyn",
        "writer": w2_writer,
        "projector_rank": 128,
        "projector_mode": "per_layer_low_rank",
        "reader_queries": 8,
        "short_slots": 8,
    },
    "fever": {
        "writer_family": "W2",
        "bridge_family": "B1_dyn",
        "projector_family": "P2_dyn",
        "writer": w2_writer,
        "projector_rank": 128,
        "projector_mode": "per_layer_low_rank",
        "reader_queries": 8,
        "short_slots": 8,
    },
}

receiver_disabled = {
    "enabled": False,
    "target_layers": [],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 0,
    "alpha": 4.0,
    "dropout": 0.0,
}
receiver_lora_config = {
    "enabled": True,
    "target_layers": list(depth_spec["receiver_layers"]),
    "target_modules": ["k_proj", "v_proj"],
    "rank": 2,
    "alpha": 4.0,
    "dropout": 0.0,
}


def apply_direct_spec(spec: dict[str, object]) -> None:
    config["runtime"]["pilot_memory_path_variant"] = "single_level"
    config["runtime"]["pilot_projector_token_source"] = "writer_slots"
    config["runtime"]["pilot_reader_context_mode"] = "prompt_summary"
    config["runtime"]["pilot_reader_num_queries"] = 4
    config["runtime"]["pilot_fuser_short_slots"] = int(spec["writer"]["memory_slots"])
    config["runtime"]["pilot_deep_prefix_rank"] = int(spec["projector_rank"])
    config["runtime"]["pilot_deep_prefix_projector_mode"] = spec["projector_mode"]
    config["runtime"]["pilot_active_writer_family"] = spec["writer_family"]
    config["runtime"]["pilot_active_projector_family"] = spec["projector_family"]
    config["runtime"]["pilot_active_bridge_family"] = spec["bridge_family"]
    config["method"]["writer"] = dict(spec["writer"])
    config["method"]["receiver_lora"] = dict(receiver_lora_config)
    config["method"].pop("reader", None)
    config["method"].pop("fuser", None)


def apply_bridge_spec(spec: dict[str, object]) -> None:
    config["runtime"]["pilot_memory_path_variant"] = "two_level"
    config["runtime"]["pilot_projector_token_source"] = "short_slots"
    config["runtime"]["pilot_reader_context_mode"] = "prompt_summary"
    config["runtime"]["pilot_reader_num_queries"] = int(spec["reader_queries"])
    config["runtime"]["pilot_fuser_short_slots"] = int(spec["short_slots"])
    config["runtime"]["pilot_deep_prefix_rank"] = int(spec["projector_rank"])
    config["runtime"]["pilot_deep_prefix_projector_mode"] = spec["projector_mode"]
    config["runtime"]["pilot_active_writer_family"] = spec["writer_family"]
    config["runtime"]["pilot_active_projector_family"] = spec["projector_family"]
    config["runtime"]["pilot_active_bridge_family"] = spec["bridge_family"]
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


def apply_base_arm(base_arm: str) -> None:
    if base_arm in direct_control_specs:
        apply_direct_spec(direct_control_specs[base_arm])
        return
    if base_arm in bridge_specs:
        apply_bridge_spec(bridge_specs[base_arm])
        return
    raise ValueError(
        f"Unsupported PLANv7 V7-6 base arm {base_arm!r}; expected one of "
        f"{sorted(direct_control_specs) + sorted(bridge_specs)}."
    )


def apply_v74_overlay(base_arm: str) -> None:
    if base_arm == "f1_num_mask":
        if task_name == "gsm8k":
            config["runtime"]["pilot_backbone_prompt_mask_mode"] = "gsm8k_numbers"
            config["runtime"]["pilot_writer_context_prompt_mode"] = "full_unmasked_prompt"
        return
    if base_arm == "f2_rx_only":
        config["runtime"]["pilot_trainable_variant"] = "receiver_then_joint"
        config["runtime"]["stage_a_steps"] = 75
        config["runtime"]["stage_b_steps"] = 125
        config["runtime"]["pilot_receiver_lora_learning_rate"] = 5.0e-5
        config["method"]["receiver_lora"] = {
            "enabled": True,
            "target_layers": list(depth_spec["receiver_layers"]),
            "target_modules": ["k_proj", "v_proj"],
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
        }
        return
    if base_arm == "f3_anneal":
        if task_name == "gsm8k":
            config["runtime"]["pilot_writer_context_prompt_mode"] = "full_unmasked_prompt"
            config["runtime"]["pilot_train_backbone_prompt_mask_schedule"] = "gsm8k_number_starvation_anneal"
        return
    if base_arm == "f4_dyn_budget":
        config["runtime"]["pilot_dynamic_budget_profile"] = (
            "gsm8k_64_16" if task_name == "gsm8k" else "triviaqa_32_8"
        )
        apply_bridge_spec(f4_dynamic_budget_specs[task_name])
        return


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
config["runtime"]["pilot_writer_stimulus_mode"] = "support_and_context"
config["runtime"]["pilot_context_support_balance_mode"] = "layernorm_learned_scalar"
config["runtime"]["pilot_context_balance_scale_init"] = 0.75
config["runtime"]["pilot_support_balance_scale_init"] = 1.25
config["runtime"]["pilot_writer_context_tokens"] = 8
config["runtime"]["pilot_train_support_mode"] = "static_support_rows"
config["runtime"]["pilot_support_examples"] = 8
config["runtime"]["pilot_lr_schedule"] = "constant_with_linear_warmup"
config["runtime"]["pilot_lr_warmup_steps"] = 0
config["runtime"]["pilot_projector_warmup_steps"] = 0
config["runtime"]["pilot_writer_learning_rate"] = 1.0e-4
config["runtime"]["pilot_projector_learning_rate"] = 7.5e-6
config["runtime"]["pilot_receiver_lora_learning_rate"] = 5.0e-5
config["runtime"]["owner_locked_projector_lr"] = 7.5e-6
config["runtime"]["repo_confirmed_v65_projector_lr_reference"] = 7.5e-5
config["runtime"]["owner_override_note"] = True
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
config["runtime"]["pilot_active_support_family"] = "S3"
config["runtime"]["pilot_active_context_family"] = "C2"
config["runtime"]["pilot_active_depth_family"] = winning_depth
config["runtime"]["pilot_active_depth_label"] = str(manifest.get("winning_depth_label", "mid4"))
config["runtime"]["pilot_deep_prefix_layers"] = list(depth_spec["depth_layers"])
config["runtime"]["pilot_train_steps"] = train_steps
config["runtime"]["pilot_snapshot_steps"] = [0, 10, 25, 50, 100, 150, 200, 250, train_steps]
config["runtime"]["shared_injection_arm"] = "injected"
config["runtime"]["pilot_backbone_prompt_mask_mode"] = "none"
config["runtime"]["pilot_writer_context_prompt_mode"] = "same_as_backbone"
config["runtime"]["pilot_train_backbone_prompt_mask_schedule"] = "none"
config["runtime"]["pilot_trainable_variant"] = "full"
config["runtime"]["stage_a_steps"] = 0
config["runtime"]["stage_b_steps"] = 0
config["runtime"]["pilot_aux_loss_mode"] = "orthogonality_coverage"
config["runtime"]["pilot_writer_slot_orthogonality_weight"] = 0.05
config["runtime"]["pilot_writer_support_coverage_weight"] = 0.05
config["runtime"]["pilot_reconstruction_aux_mode"] = "off"
config["runtime"]["pilot_reconstruction_aux_weight"] = 0.0
config["runtime"]["pilot_reconstruction_vocab_size"] = 1024
config["runtime"]["pilot_reconstruction_hidden_dim"] = 1024
config["runtime"]["pilot_reconstruction_weight_schedule"] = "constant"
config["runtime"]["pilot_aux_projection_dim"] = 0
config["runtime"]["pilot_aux_projection_hidden_dim"] = None
config["runtime"]["pilot_vicreg_loss_weight"] = 0.0
config["runtime"]["pilot_vicreg_invariance_weight"] = 1.0
config["runtime"]["pilot_vicreg_variance_weight"] = 1.0
config["runtime"]["pilot_vicreg_covariance_weight"] = 1.0
config["runtime"]["pilot_vicreg_variance_target"] = 1.0
config["runtime"]["pilot_contrastive_loss_weight"] = 0.0
config["runtime"]["pilot_contrastive_temperature"] = 0.10
config["runtime"]["pilot_contrastive_queue_size"] = 64
config["runtime"]["pilot_barlow_loss_weight"] = 0.0
config["runtime"]["pilot_barlow_lambda"] = 5.0e-3

if task_name == "fever":
    config["runtime"]["pilot_support_serialization"] = "flat_raw8"
    config["runtime"]["pilot_prompt_variant"] = "inline_short_labels"
else:
    config["runtime"]["pilot_support_serialization"] = "example_blocks_raw8"
    config["runtime"]["pilot_prompt_variant"] = "task_native"

base_arm_id = str(manifest.get("base_arm_id", ""))
base_source_phase = str(manifest.get("base_source_phase", "v7_3"))
control_source_arm_id = str(manifest.get("control_source_arm_id", ""))
underlying_base_arm = base_arm_id
if base_source_phase == "v7_4" and base_arm_id.startswith("f"):
    underlying_base_arm = control_source_arm_id

branch_variant_map = dict(manifest.get("branch_variant_map", {}))
promoted_arm_id = str(branch_variant_map.get(variant_id, ""))

if variant_id == "c0_frozen_no_memory":
    config["runtime"]["shared_injection_arm"] = "base_only"
    config["runtime"]["pilot_arm_alias"] = "C0"
    config["runtime"]["pilot_train_steps"] = 0
    config["runtime"]["pilot_snapshot_steps"] = [0]
    config["runtime"]["pilot_gradient_probe_enabled"] = False
    config["runtime"]["pilot_aux_loss_mode"] = "off"
    config["runtime"]["pilot_active_aux_family"] = "no_memory_control"
    config["method"]["receiver_lora"] = receiver_disabled
elif variant_id == "c1_additive_continuity":
    config["runtime"]["pilot_memory_path_variant"] = "single_level"
    config["runtime"]["pilot_projector_token_source"] = "writer_slots"
    config["runtime"]["pilot_arm_alias"] = "C_ADD"
    config["runtime"]["pilot_active_context_family"] = "C_add"
    config["runtime"]["pilot_active_aux_family"] = "L5_additive_continuity"
    config["runtime"]["pilot_deep_prefix_layers"] = [0, 1, 2, 3, 4, 8, 14]
    config["runtime"]["pilot_reader_num_queries"] = 4
    config["runtime"]["pilot_fuser_short_slots"] = 2
    config["method"]["receiver_lora"] = {
        "enabled": True,
        "target_layers": [0, 1, 2, 3, 4],
        "target_modules": ["k_proj", "v_proj"],
        "rank": 2,
        "alpha": 4.0,
        "dropout": 0.0,
    }
elif variant_id == "c2_best_direct":
    direct_control_arm_id = str(manifest.get("direct_control_arm_id", ""))
    if direct_control_arm_id not in direct_control_specs:
        raise ValueError(
            f"Unsupported PLANv7 V7-6 direct control arm {direct_control_arm_id!r}; "
            f"expected one of {sorted(direct_control_specs)}."
        )
    apply_direct_spec(direct_control_specs[direct_control_arm_id])
    config["runtime"]["pilot_arm_alias"] = "DIRECT_CONTROL"
    config["runtime"]["pilot_active_aux_family"] = "direct_control_comparator"
else:
    apply_base_arm(underlying_base_arm)
    if base_source_phase == "v7_4":
        apply_v74_overlay(base_arm_id)
    if promoted_arm_id == "a0_baseline":
        config["runtime"]["pilot_arm_alias"] = "A0_L5_BASELINE"
        config["runtime"]["pilot_active_aux_arm"] = "A0"
        config["runtime"]["pilot_active_aux_family"] = "A0_L5_baseline"
    elif promoted_arm_id == "a1_reconstruction":
        config["runtime"]["pilot_arm_alias"] = "A1_RECONSTRUCTION"
        config["runtime"]["pilot_active_aux_arm"] = "A1"
        config["runtime"]["pilot_active_aux_family"] = "A1_L5_plus_reconstruction"
        config["runtime"]["pilot_reconstruction_aux_mode"] = "hashed_bow"
        config["runtime"]["pilot_reconstruction_aux_weight"] = 0.02
        config["runtime"]["pilot_reconstruction_weight_schedule"] = "three_stage_decay"
    elif promoted_arm_id == "a2_vicreg":
        config["runtime"]["pilot_arm_alias"] = "A2_VICREG"
        config["runtime"]["pilot_active_aux_arm"] = "A2"
        config["runtime"]["pilot_active_aux_family"] = "A2_L5_plus_vicreg"
        config["runtime"]["pilot_aux_projection_dim"] = 128
        config["runtime"]["pilot_aux_projection_hidden_dim"] = 256
        config["runtime"]["pilot_vicreg_loss_weight"] = 0.02
    elif promoted_arm_id == "a3_contrastive":
        config["runtime"]["pilot_arm_alias"] = "A3_CONTRASTIVE"
        config["runtime"]["pilot_active_aux_arm"] = "A3"
        config["runtime"]["pilot_active_aux_family"] = "A3_L5_plus_contrastive"
        config["runtime"]["pilot_aux_projection_dim"] = 128
        config["runtime"]["pilot_aux_projection_hidden_dim"] = 256
        config["runtime"]["pilot_contrastive_loss_weight"] = 0.02
    elif promoted_arm_id == "a4_reconstruction_vicreg":
        config["runtime"]["pilot_arm_alias"] = "A4_RECONSTRUCTION_VICREG"
        config["runtime"]["pilot_active_aux_arm"] = "A4"
        config["runtime"]["pilot_active_aux_family"] = "A4_L5_plus_reconstruction_plus_vicreg"
        config["runtime"]["pilot_aux_projection_dim"] = 128
        config["runtime"]["pilot_aux_projection_hidden_dim"] = 256
        config["runtime"]["pilot_reconstruction_aux_mode"] = "hashed_bow"
        config["runtime"]["pilot_reconstruction_aux_weight"] = 0.02
        config["runtime"]["pilot_reconstruction_weight_schedule"] = "three_stage_decay"
        config["runtime"]["pilot_vicreg_loss_weight"] = 0.02
    elif promoted_arm_id == "a5_barlow":
        config["runtime"]["pilot_arm_alias"] = "A5_BARLOW"
        config["runtime"]["pilot_active_aux_arm"] = "A5"
        config["runtime"]["pilot_active_aux_family"] = "A5_L5_plus_barlow"
        config["runtime"]["pilot_aux_projection_dim"] = 128
        config["runtime"]["pilot_aux_projection_hidden_dim"] = 256
        config["runtime"]["pilot_barlow_loss_weight"] = 0.02
    else:
        raise ValueError(
            f"Unsupported PLANv7 V7-6 promoted arm {promoted_arm_id!r} from variant {variant_id!r}."
        )

config["experiment"]["name"] = f"planv7_v7_6_{task_name}_{variant_id}"
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

mapfile -t ALL_VARIANTS < <(
  python - "${RESULT_ROOT}/selection-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
for variant in payload.get("variants", []):
    variant_id = str(variant).strip()
    if variant_id:
        print(variant_id)
PY
)

read -r WINNER_USES_BRIDGE DIRECT_CONTROL_ARM_ID < <(
  python - "${RESULT_ROOT}/selection-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(
    "1" if bool(payload.get("winner_uses_bridge", False)) else "0",
    str(payload.get("direct_control_arm_id", "")).strip(),
)
PY
)

for variant_id in "${ALL_VARIANTS[@]}"; do
  if [[ "${variant_id}" == "c2_best_direct" && ( "${WINNER_USES_BRIDGE}" != "1" || -z "${DIRECT_CONTROL_ARM_ID}" ) ]]; then
    continue
  fi
  for task_name in gsm8k triviaqa fever; do
    materialize_config \
      "${task_name}" \
      "${variant_id}" \
      "${CONFIG_ROOT}/${task_name}-${variant_id}.json" \
      "${DATA_ROOT}/${task_name}/support.jsonl" \
      "${DATA_ROOT}/${task_name}/train.jsonl" \
      "${DATA_ROOT}/${task_name}/eval.jsonl"
  done
done

for seed_offset in 0 1 2; do
  run_seed=$((BASE_SEED + seed_offset))
  seed_dir_name="seed_${run_seed}"
  for variant_id in "${ALL_VARIANTS[@]}"; do
    if [[ "${variant_id}" == "c2_best_direct" && ( "${WINNER_USES_BRIDGE}" != "1" || -z "${DIRECT_CONTROL_ARM_ID}" ) ]]; then
      continue
    fi
    for task_name in gsm8k triviaqa fever; do
      arm_spec="pilot-I-real:${variant_id^^}:injected:real:0"
      pilot_subdir="pilot-I-real"
      if [[ "${variant_id}" == "c0_frozen_no_memory" ]]; then
        arm_spec="pilot-C0:C0:base_only:real:0"
        pilot_subdir="pilot-C0"
      elif [[ "${variant_id}" == "c1_additive_continuity" ]]; then
        arm_spec="pilot-I-real:C_ADD:injected:real:0"
      elif [[ "${variant_id}" == "c2_best_direct" ]]; then
        arm_spec="pilot-I-real:DIRECT_CONTROL:injected:real:0"
      fi
      ensure_suite_complete \
        "${CONFIG_ROOT}/${task_name}-${variant_id}.json" \
        "${run_seed}" \
        "${RUN_ROOT}/${variant_id}/${seed_dir_name}/${task_name}" \
        "${arm_spec}"
      copy_run_artifacts \
        "${RESULT_ROOT}/${variant_id}/${seed_dir_name}/${task_name}" \
        "${RUN_ROOT}/${variant_id}/${seed_dir_name}/${task_name}" \
        "${pilot_subdir}"
    done
  done
done

python scripts/update_planv7_v7_6_multiseed_confirmation_summary.py \
  --result_root "${RESULT_ROOT}" \
  --v75_summary "${V75_SUMMARY_JSON}" \
  --v70_summary "${V70_SUMMARY_JSON}" \
  --output_json "${RESULT_ROOT}/v7-6-summary.json" \
  --output_report "${RESULT_ROOT}/v7-6-summary.md"

bash scripts/publish_review_artifacts.sh
