#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARM_ORDER = (
    "w0_oracle64",
    "w1_ext2layer64_lr2e5",
    "w2_ext3layer64_lr2e5",
    "w3_ext3layer96_lr1e5",
    "w4_ext3layer64_lr5e5",
)

ARM_METADATA = {
    "w0_oracle64": {
        "writer_family": "EW0",
        "memory_slots": 64,
        "transformer_layers": 2,
        "writer_learning_rate": 0.0,
        "trainable_variant": "reader_only",
        "prefix_source_mode": "oracle_hidden_state_slots",
        "stage_a_steps": 0,
        "stage_b_steps": 0,
    },
    "w1_ext2layer64_lr2e5": {
        "writer_family": "EW1",
        "memory_slots": 64,
        "transformer_layers": 2,
        "writer_learning_rate": 2.0e-5,
        "trainable_variant": "writer_then_joint",
        "prefix_source_mode": "writer",
        "stage_a_steps": 80,
        "stage_b_steps": 220,
    },
    "w2_ext3layer64_lr2e5": {
        "writer_family": "EW2",
        "memory_slots": 64,
        "transformer_layers": 3,
        "writer_learning_rate": 2.0e-5,
        "trainable_variant": "writer_then_joint",
        "prefix_source_mode": "writer",
        "stage_a_steps": 80,
        "stage_b_steps": 220,
    },
    "w3_ext3layer96_lr1e5": {
        "writer_family": "EW3",
        "memory_slots": 96,
        "transformer_layers": 3,
        "writer_learning_rate": 1.0e-5,
        "trainable_variant": "writer_then_joint",
        "prefix_source_mode": "writer",
        "stage_a_steps": 80,
        "stage_b_steps": 220,
    },
    "w4_ext3layer64_lr5e5": {
        "writer_family": "EW2",
        "memory_slots": 64,
        "transformer_layers": 3,
        "writer_learning_rate": 5.0e-5,
        "trainable_variant": "writer_then_joint",
        "prefix_source_mode": "writer",
        "stage_a_steps": 80,
        "stage_b_steps": 220,
    },
}


def materialize_planv8_v8_4_config(
    *,
    base_config_path: Path,
    base_checkpoint_path: Path,
    arm_id: str,
    output_config: Path,
    v83_summary_path: Path,
    primary_model_dir: str = "",
    primary_backbone_name: str = "",
) -> dict[str, Any]:
    if arm_id not in ARM_METADATA:
        raise ValueError(f"Unsupported V8-4 arm {arm_id}.")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing V8-3 base config: {base_config_path}")
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing V8-3 base checkpoint: {base_checkpoint_path}")

    config = json.loads(base_config_path.read_text())
    config.setdefault("experiment", {})
    config.setdefault("backbone", {})
    config.setdefault("runtime", {})
    config.setdefault("method", {})
    config["method"].setdefault("writer", {})

    v83_summary = json.loads(v83_summary_path.read_text())
    selected_interface_family = str(
        v83_summary.get("selected_interface_family_for_v8_4")
        or v83_summary.get("best_interface_family")
        or ""
    ).strip()
    base_arm_id = str(v83_summary.get("base_for_v8_4_arm_id") or v83_summary.get("best_arm_id") or "").strip()

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

    writer_cfg["memory_slots"] = int(arm_spec["memory_slots"])
    writer_cfg["transformer_layers"] = int(arm_spec["transformer_layers"])

    experiment["name"] = f"{Path().resolve().name}_{arm_id}_{task_name}"
    experiment["stage"] = "V8-4"
    experiment["method_variant"] = arm_id

    runtime["pilot_arm_alias"] = arm_id
    runtime["pilot_v83_base_arm_id"] = base_arm_id
    runtime["pilot_v83_selected_interface_family"] = selected_interface_family
    runtime["pilot_init_checkpoint_path"] = str(base_checkpoint_path.resolve())
    runtime["pilot_init_checkpoint_mode"] = "consumer_only"
    runtime["pilot_checkpoint_path"] = ""
    runtime["pilot_train_steps"] = 300
    runtime["pilot_snapshot_steps"] = [0, 10, 25, 50, 100, 150, 200, 250, 300]
    runtime["pilot_gradient_probe_enabled"] = True
    runtime["pilot_gradient_probe_interval"] = 5
    runtime["pilot_gradient_probe_max_steps"] = 300
    runtime["pilot_trainable_variant"] = str(arm_spec["trainable_variant"])
    runtime["stage_a_steps"] = int(arm_spec["stage_a_steps"])
    runtime["stage_b_steps"] = int(arm_spec["stage_b_steps"])
    runtime["pilot_prefix_source_mode"] = str(arm_spec["prefix_source_mode"])
    runtime["pilot_oracle_slot_cap"] = int(arm_spec["memory_slots"])
    runtime["pilot_writer_learning_rate"] = float(arm_spec["writer_learning_rate"])
    runtime["pilot_writer_family"] = str(arm_spec["writer_family"])
    runtime["pilot_writer_memory_slots"] = int(arm_spec["memory_slots"])
    runtime["pilot_writer_transformer_layers"] = int(arm_spec["transformer_layers"])

    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=Path, required=True)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--arm_id", type=str, required=True, choices=ARM_ORDER)
    parser.add_argument("--output_config", type=Path, required=True)
    parser.add_argument("--v83_summary_path", type=Path, required=True)
    parser.add_argument("--primary_model_dir", type=str, default="")
    parser.add_argument("--primary_backbone_name", type=str, default="")
    args = parser.parse_args()

    materialize_planv8_v8_4_config(
        base_config_path=args.base_config,
        base_checkpoint_path=args.base_checkpoint,
        arm_id=args.arm_id,
        output_config=args.output_config,
        v83_summary_path=args.v83_summary_path,
        primary_model_dir=args.primary_model_dir,
        primary_backbone_name=args.primary_backbone_name,
    )


if __name__ == "__main__":
    main()
