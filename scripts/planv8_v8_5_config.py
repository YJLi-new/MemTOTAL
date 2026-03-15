#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARM_ORDER = (
    "b0_no_bridge",
    "b1_q16_s16",
    "b2_q32_s16",
    "b3_q32_s8",
    "b4_q48_s16_x96",
)

ARM_METADATA = {
    "b0_no_bridge": {
        "bridge_family": "BR0",
        "reader_queries": 0,
        "short_slots": 0,
        "target_input_slots": None,
    },
    "b1_q16_s16": {
        "bridge_family": "BR1",
        "reader_queries": 16,
        "short_slots": 16,
        "target_input_slots": None,
    },
    "b2_q32_s16": {
        "bridge_family": "BR2",
        "reader_queries": 32,
        "short_slots": 16,
        "target_input_slots": None,
    },
    "b3_q32_s8": {
        "bridge_family": "BR3",
        "reader_queries": 32,
        "short_slots": 8,
        "target_input_slots": None,
    },
    "b4_q48_s16_x96": {
        "bridge_family": "BR2",
        "reader_queries": 48,
        "short_slots": 16,
        "target_input_slots": 96,
    },
}

BRIDGE_READER_CONFIG = {
    "use_query_gating": False,
    "condition_on_context": True,
    "conditioning_mode": "add",
    "attention_mode": "standard",
    "dropout": 0.05,
    "query_residual_scale": 0.0,
    "num_heads": 8,
}

BRIDGE_FUSER_CONFIG = {
    "arch": "resampler",
    "hidden_dim": 1536,
    "num_heads": 8,
    "dropout": 0.05,
}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def materialize_planv8_v8_5_config(
    *,
    base_config_path: Path,
    base_checkpoint_path: Path,
    arm_id: str,
    output_config: Path,
    v84_summary_path: Path,
    primary_model_dir: str = "",
    primary_backbone_name: str = "",
) -> dict[str, Any]:
    if arm_id not in ARM_METADATA:
        raise ValueError(f"Unsupported V8-5 arm {arm_id}.")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing V8-4 base config: {base_config_path}")
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing V8-4 base checkpoint: {base_checkpoint_path}")

    config = json.loads(base_config_path.read_text())
    config.setdefault("experiment", {})
    config.setdefault("backbone", {})
    config.setdefault("runtime", {})
    config.setdefault("method", {})
    config["method"].setdefault("writer", {})

    v84_summary = json.loads(v84_summary_path.read_text())
    selected_interface_family = str(
        v84_summary.get("selected_interface_family_for_v8_5")
        or v84_summary.get("best_interface_family")
        or ""
    ).strip()
    base_arm_id = str(v84_summary.get("base_for_v8_5_arm_id") or v84_summary.get("best_arm_id") or "").strip()

    experiment = config["experiment"]
    runtime = config["runtime"]
    backbone = config["backbone"]
    writer_cfg = config["method"]["writer"]
    arm_spec = ARM_METADATA[arm_id]
    task_name = str(config.get("task", {}).get("benchmark_id", "")).strip() or base_config_path.stem.split("-")[0]

    if primary_model_dir:
        backbone["model_id"] = primary_model_dir
    if primary_backbone_name:
        backbone["name"] = primary_backbone_name

    base_writer_slots = _safe_int(writer_cfg.get("memory_slots"), 64)
    target_input_slots = (
        base_writer_slots
        if arm_spec["target_input_slots"] is None
        else int(arm_spec["target_input_slots"])
    )

    warm_start_enabled = True
    warm_start_status = "full_from_v84"
    if arm_id == "b4_q48_s16_x96" and base_writer_slots != target_input_slots:
        writer_cfg["memory_slots"] = int(target_input_slots)
        warm_start_enabled = False
        warm_start_status = "cold_start_writer_slot_mismatch"

    experiment["name"] = f"{Path().resolve().name}_{arm_id}_{task_name}"
    experiment["stage"] = "V8-5"
    experiment["method_variant"] = arm_id

    runtime["pilot_arm_alias"] = arm_id
    runtime["pilot_v84_base_arm_id"] = base_arm_id
    runtime["pilot_v84_selected_interface_family"] = selected_interface_family
    runtime["pilot_active_bridge_family"] = str(arm_spec["bridge_family"])
    runtime["pilot_bridge_expected_input_slots"] = int(target_input_slots)
    runtime["pilot_bridge_expected_short_slots"] = int(arm_spec["short_slots"])
    runtime["pilot_bridge_expected_queries"] = int(arm_spec["reader_queries"])
    runtime["pilot_init_checkpoint_path"] = (
        str(base_checkpoint_path.resolve()) if warm_start_enabled else ""
    )
    runtime["pilot_init_checkpoint_mode"] = "full"
    runtime["pilot_checkpoint_path"] = ""
    runtime["pilot_train_steps"] = 300
    runtime["pilot_snapshot_steps"] = [0, 10, 25, 50, 100, 150, 200, 250, 300]
    runtime["pilot_gradient_probe_enabled"] = True
    runtime["pilot_gradient_probe_interval"] = 5
    runtime["pilot_gradient_probe_max_steps"] = 300
    runtime["pilot_trainable_variant"] = "reader_only"
    runtime["stage_a_steps"] = 0
    runtime["stage_b_steps"] = 0
    runtime["pilot_reader_fuser_bootstrap_steps"] = 0
    runtime["pilot_v84_warm_start_status"] = warm_start_status

    if arm_id == "b0_no_bridge":
        runtime["pilot_memory_path_variant"] = "single_level"
        runtime["pilot_projector_token_source"] = "writer_slots"
        runtime.pop("pilot_reader_context_mode", None)
        runtime.pop("pilot_reader_num_queries", None)
        runtime.pop("pilot_fuser_short_slots", None)
        config["method"].pop("reader", None)
        config["method"].pop("fuser", None)
    else:
        runtime["pilot_memory_path_variant"] = "two_level"
        runtime["pilot_projector_token_source"] = "short_slots"
        runtime["pilot_reader_context_mode"] = "prompt_summary"
        runtime["pilot_reader_num_queries"] = int(arm_spec["reader_queries"])
        runtime["pilot_fuser_short_slots"] = int(arm_spec["short_slots"])
        config["method"]["reader"] = {
            "num_queries": int(arm_spec["reader_queries"]),
            **BRIDGE_READER_CONFIG,
        }
        config["method"]["fuser"] = {
            "short_slots": int(arm_spec["short_slots"]),
            **BRIDGE_FUSER_CONFIG,
        }

    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=Path, required=True)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--arm_id", type=str, required=True, choices=ARM_ORDER)
    parser.add_argument("--output_config", type=Path, required=True)
    parser.add_argument("--v84_summary_path", type=Path, required=True)
    parser.add_argument("--primary_model_dir", type=str, default="")
    parser.add_argument("--primary_backbone_name", type=str, default="")
    args = parser.parse_args()

    materialize_planv8_v8_5_config(
        base_config_path=args.base_config,
        base_checkpoint_path=args.base_checkpoint,
        arm_id=args.arm_id,
        output_config=args.output_config,
        v84_summary_path=args.v84_summary_path,
        primary_model_dir=args.primary_model_dir,
        primary_backbone_name=args.primary_backbone_name,
    )


if __name__ == "__main__":
    main()
