#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memtotal.utils.config import load_config

LAYER_BANDS = {
    "mid8": [14, 15, 16, 17, 18, 19, 20, 21],
    "mid12": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "late8": [20, 21, 22, 23, 24, 25, 26, 27],
}
ARM_SWEEP = {
    "r0_mid8_r32_lr5e5": {"layer_band": "mid8", "rank": 32, "learning_rate": 5.0e-5, "train_steps": 300},
    "r1_mid8_r64_lr1e4": {"layer_band": "mid8", "rank": 64, "learning_rate": 1.0e-4, "train_steps": 300},
    "r2_mid12_r64_lr1e4": {"layer_band": "mid12", "rank": 64, "learning_rate": 1.0e-4, "train_steps": 300},
    "r3_mid12_r64_lr2e4": {"layer_band": "mid12", "rank": 64, "learning_rate": 2.0e-4, "train_steps": 300},
    "r4_late8_r32_lr1e4": {"layer_band": "late8", "rank": 32, "learning_rate": 1.0e-4, "train_steps": 300},
    "r5_mid8_r16_lr5e5": {"layer_band": "mid8", "rank": 16, "learning_rate": 5.0e-5, "train_steps": 300},
}
RECEIVER_DISABLED = {
    "enabled": False,
    "target_layers": [],
    "target_modules": ["k_proj", "v_proj"],
    "rank": 0,
    "alpha": 4.0,
    "dropout": 0.0,
}
LEGACY_PREFIX_DEFAULT_READER_LAYERS = [16, 17, 18, 19]
LEGACY_PREFIX_DEFAULT_RECEIVER_RANK = 2


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _rank_from_label(label: Any, default: int) -> int:
    raw = str(label or "").strip().lower()
    if raw.startswith("r"):
        raw = raw[1:]
    return _safe_int(raw, default)


def _legacy_prefix_receiver_lora(base_arm_summary: dict[str, Any]) -> dict[str, Any]:
    target_layers = [
        int(layer_index)
        for layer_index in base_arm_summary.get("reader_layers", LEGACY_PREFIX_DEFAULT_READER_LAYERS)
    ]
    if not target_layers:
        target_layers = list(LEGACY_PREFIX_DEFAULT_READER_LAYERS)
    rank = _rank_from_label(
        base_arm_summary.get("rank_label"),
        default=LEGACY_PREFIX_DEFAULT_RECEIVER_RANK,
    )
    if len(target_layers) > 5:
        raise ValueError(
            "V8-2 ri0_legacy_prefix requires the V8-1 base receiver micro-LoRA to target at most 5 layers."
        )
    if rank <= 0 or rank > 4:
        raise ValueError(
            "V8-2 ri0_legacy_prefix requires the V8-1 base receiver micro-LoRA rank to stay in [1, 4]."
        )
    return {
        "enabled": True,
        "target_layers": target_layers,
        "target_modules": ["k_proj", "v_proj"],
        "rank": rank,
        "alpha": float(min(2 * rank, 8)),
        "dropout": 0.0,
    }


def materialize_planv8_v8_2_config(
    *,
    task_name: str,
    arm_id: str,
    prompt_variant: str,
    output_config: Path,
    support_path: str,
    train_path: str,
    eval_path: str,
    primary_model_dir: str,
    primary_backbone_name: str,
    v81_summary_path: Path,
) -> dict[str, Any]:
    template_path = Path(f"configs/exp/writer_circuit_g2_writer_direct_{task_name}_template.yaml")
    config = load_config(template_path)
    config.setdefault("experiment", {})
    config.setdefault("backbone", {})
    config.setdefault("method", {})
    config.setdefault("runtime", {})
    config.setdefault("task", {})
    config.setdefault("task", {}).setdefault("evaluator", {})

    v81_summary = json.loads(v81_summary_path.read_text())
    selected_interface_family = str(
        v81_summary.get("selected_interface_family_for_v8_2")
        or v81_summary.get("best_interface_family")
        or ""
    ).strip()
    base_arm_id = str(v81_summary.get("base_for_v8_2_arm_id") or v81_summary.get("best_arm_id") or "").strip()
    base_arm_summary = dict(v81_summary.get("arm_summaries", {}).get(base_arm_id, {}))
    slot_cap = _safe_int(base_arm_summary.get("memory_slots"), 16)

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
    config["method"]["receiver_lora"] = dict(RECEIVER_DISABLED)

    if arm_id == "control":
        runtime["shared_injection_arm"] = "base_only"
    else:
        if arm_id not in ARM_SWEEP:
            raise ValueError(f"Unsupported V8-2 arm {arm_id}.")
        spec = ARM_SWEEP[arm_id]
        learning_rate = float(spec["learning_rate"])
        reader_layers = list(LAYER_BANDS[str(spec["layer_band"])])
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
            config["method"]["receiver_lora"] = _legacy_prefix_receiver_lora(base_arm_summary)
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
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--arm_id", required=True)
    parser.add_argument("--prompt_variant", required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--support_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--primary_model_dir", required=True)
    parser.add_argument("--primary_backbone_name", required=True)
    parser.add_argument("--v81_summary_path", required=True)
    args = parser.parse_args()
    materialize_planv8_v8_2_config(
        task_name=args.task_name,
        arm_id=args.arm_id,
        prompt_variant=args.prompt_variant,
        output_config=Path(args.output_config),
        support_path=str(Path(args.support_path).resolve()),
        train_path=str(Path(args.train_path).resolve()),
        eval_path=str(Path(args.eval_path).resolve()),
        primary_model_dir=str(Path(args.primary_model_dir).resolve()),
        primary_backbone_name=args.primary_backbone_name,
        v81_summary_path=Path(args.v81_summary_path),
    )


if __name__ == "__main__":
    main()
