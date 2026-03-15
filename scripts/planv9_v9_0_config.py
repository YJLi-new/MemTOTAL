#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memtotal.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "configs" / "exp" / "writer_circuit_g2_writer_direct_gsm8k_template.yaml"
HF_CACHE_DIR = "/root/autodl-tmp/hf-cache"

ARM_SPECS: dict[str, dict[str, Any]] = {
    "a0_nomemory_control": {
        "shared_injection_arm": "base_only",
        "memory_consumer_mode": "legacy_prefix",
        "prefix_source_mode": "writer",
    },
    "a1_legacy_prefix_oracle": {
        "shared_injection_arm": "injected",
        "memory_consumer_mode": "legacy_prefix",
        "prefix_source_mode": "oracle_hidden_state_slots",
        "deep_prefix_layers": [16, 17, 18, 19],
        "oracle_extract_layer": 18,
        "oracle_slot_pool_window": 16,
        "oracle_slot_cap": 16,
    },
    "a2_precache_latent_oracle": {
        "shared_injection_arm": "injected",
        "memory_consumer_mode": "precache_latent",
        "prefix_source_mode": "oracle_hidden_state_slots",
        "oracle_extract_layer": 16,
        "oracle_slot_pool_window": 16,
        "oracle_slot_cap": 8,
    },
    "a3_sequence_replay_oracle": {
        "shared_injection_arm": "injected",
        "memory_consumer_mode": "reader_lora_sequence",
        "memory_segment_mode": "prepend_block",
        "prefix_source_mode": "oracle_hidden_state_slots",
        "oracle_extract_layer": 16,
        "oracle_slot_pool_window": 16,
        "oracle_slot_cap": 8,
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def materialize_planv9_v9_0_config(
    *,
    arm_id: str,
    output_config: Path,
    support_path: str,
    train_path: str,
    eval_path: str,
    selected_prompt_variant: str,
    primary_model_dir: str,
    primary_backbone_name: str = "Qwen3-4B",
    hf_cache_dir: str = HF_CACHE_DIR,
) -> dict[str, Any]:
    if arm_id not in ARM_SPECS:
        raise ValueError(f"Unsupported V9-0 arm_id: {arm_id}")
    config = load_config(TEMPLATE_PATH)
    arm_spec = ARM_SPECS[arm_id]
    config.setdefault("experiment", {})
    config.setdefault("backbone", {})
    config.setdefault("method", {})
    config.setdefault("runtime", {})
    config.setdefault("task", {})
    config.setdefault("task", {}).setdefault("evaluator", {})
    config.setdefault("method", {}).setdefault("receiver_lora", {})

    config["experiment"]["name"] = f"planv9_v9_0_{arm_id}"
    config["experiment"]["stage"] = "V9-0"
    config["experiment"]["method_variant"] = arm_id

    config["backbone"]["name"] = str(primary_backbone_name)
    config["backbone"]["model_id"] = str(primary_model_dir)
    config["backbone"]["dtype"] = "bfloat16"
    config["backbone"]["cache_dir"] = str(hf_cache_dir)
    config["backbone"]["attn_implementation"] = "sdpa"
    config["backbone"]["gradient_checkpointing"] = False
    config["backbone"]["use_chat_template"] = True
    config["backbone"]["chat_template_enable_thinking"] = False
    config["backbone"]["max_new_tokens"] = 192

    config["task"]["support_dataset_path"] = str(Path(support_path).resolve())
    config["task"]["train_dataset_path"] = str(Path(train_path).resolve())
    config["task"]["train_support_dataset_path"] = str(Path(support_path).resolve())
    config["task"]["dataset_path"] = str(Path(eval_path).resolve())
    config["task"]["support_lookup_dataset_paths"] = []
    config["task"]["train_support_episode_bank_path"] = ""
    config["task"]["pilot_split"] = "eval"
    config["task"]["benchmark_id"] = "gsm8k"
    config["task"]["evaluator"]["type"] = "exact_match"
    config["task"]["evaluator"]["normalizer"] = "gsm8k_final_answer"

    config["method"]["receiver_lora"] = {
        "enabled": False,
        "target_layers": [],
        "target_modules": ["k_proj", "v_proj"],
        "rank": 0,
        "alpha": 4.0,
        "dropout": 0.0,
    }

    runtime = config["runtime"]
    runtime["device"] = "cuda"
    runtime["shared_injection_arm"] = str(arm_spec["shared_injection_arm"])
    runtime["writer_memory_control"] = "real"
    runtime["pilot_arm_alias"] = arm_id
    runtime["pilot_prompt_variant"] = str(selected_prompt_variant)
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
    runtime["pilot_prefix_source_mode"] = str(arm_spec.get("prefix_source_mode", "writer"))
    runtime["pilot_memory_consumer_mode"] = str(arm_spec.get("memory_consumer_mode", "legacy_prefix"))
    runtime["pilot_memory_segment_mode"] = str(arm_spec.get("memory_segment_mode", "prepend_block"))

    if "deep_prefix_layers" in arm_spec:
        runtime["pilot_deep_prefix_layers"] = list(arm_spec["deep_prefix_layers"])
    if "oracle_extract_layer" in arm_spec:
        runtime["pilot_oracle_extract_layer"] = int(arm_spec["oracle_extract_layer"])
    if "oracle_slot_pool_window" in arm_spec:
        runtime["pilot_oracle_slot_pool_window"] = int(arm_spec["oracle_slot_pool_window"])
    if "oracle_slot_cap" in arm_spec:
        runtime["pilot_oracle_slot_cap"] = int(arm_spec["oracle_slot_cap"])

    _write_json(output_config, config)
    return config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize PLANv9 V9-0 qwen34 configs.")
    parser.add_argument("--arm_id", required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--support_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--selected_prompt_variant", required=True)
    parser.add_argument("--primary_model_dir", required=True)
    parser.add_argument("--primary_backbone_name", default="Qwen3-4B")
    parser.add_argument("--hf_cache_dir", default=HF_CACHE_DIR)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    materialize_planv9_v9_0_config(
        arm_id=args.arm_id,
        output_config=Path(args.output_config),
        support_path=args.support_path,
        train_path=args.train_path,
        eval_path=args.eval_path,
        selected_prompt_variant=args.selected_prompt_variant,
        primary_model_dir=args.primary_model_dir,
        primary_backbone_name=args.primary_backbone_name,
        hf_cache_dir=args.hf_cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
