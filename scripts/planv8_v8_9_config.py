#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _snapshot_steps(train_steps: int) -> list[int]:
    checkpoints = {0, 10, 25, 50, 100, 150, 200, 300, 400, train_steps}
    return sorted(step for step in checkpoints if 0 <= step <= train_steps)


def materialize_planv8_v8_9_config(
    *,
    base_config_path: Path,
    output_config: Path,
    condition_id: str,
    eval_task_name: str,
    support_path: str,
    train_path: str,
    eval_path: str,
    source_variant_id: str,
    source_phase: str,
    source_arm_id: str,
    source_interface_family: str,
    source_bridge_family: str,
    source_auxiliary_family: str,
    source_prompt_variant: str,
    checkpoint_path: str = "",
    primary_model_dir: str = "",
    primary_backbone_name: str = "",
    train_steps: int = 400,
) -> dict[str, Any]:
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing V8-9 base config: {base_config_path}")

    config = json.loads(base_config_path.read_text())
    config.setdefault("experiment", {})
    config.setdefault("backbone", {})
    config.setdefault("runtime", {})
    config.setdefault("task", {})

    experiment = config["experiment"]
    backbone = config["backbone"]
    runtime = config["runtime"]
    task = config["task"]

    if primary_model_dir:
        backbone["model_id"] = primary_model_dir
    if primary_backbone_name:
        backbone["name"] = primary_backbone_name

    task["dataset_path"] = str(Path(eval_path).resolve())
    task["support_dataset_path"] = str(Path(support_path).resolve())
    task["train_dataset_path"] = str(Path(train_path).resolve())
    task["train_support_dataset_path"] = str(Path(support_path).resolve())
    task["support_lookup_dataset_paths"] = []
    task["train_support_episode_bank_path"] = ""

    experiment["name"] = f"{Path().resolve().name}_{condition_id}_{eval_task_name}"
    experiment["stage"] = "V8-9"
    experiment["method_variant"] = condition_id

    runtime["pilot_arm_alias"] = condition_id
    runtime["pilot_cdmi_condition"] = str(condition_id)
    runtime["pilot_cdmi_eval_task"] = str(eval_task_name)
    runtime["pilot_cdmi_source_variant_id"] = str(source_variant_id)
    runtime["pilot_cdmi_source_phase"] = str(source_phase)
    runtime["pilot_cdmi_source_arm_id"] = str(source_arm_id)
    runtime["pilot_cdmi_source_interface_family"] = str(source_interface_family)
    runtime["pilot_cdmi_source_bridge_family"] = str(source_bridge_family)
    runtime["pilot_cdmi_source_auxiliary_family"] = str(source_auxiliary_family)
    runtime["pilot_cdmi_source_prompt_variant"] = str(source_prompt_variant)
    runtime["pilot_prompt_variant"] = "task_native"

    checkpoint_path = str(checkpoint_path).strip()
    if checkpoint_path:
        runtime["pilot_init_checkpoint_path"] = ""
        runtime["pilot_checkpoint_path"] = str(Path(checkpoint_path).resolve())
        runtime["pilot_train_steps"] = 0
        runtime["pilot_snapshot_steps"] = [0]
        runtime["pilot_gradient_probe_enabled"] = False
        runtime["pilot_gradient_probe_interval"] = 5
        runtime["pilot_gradient_probe_max_steps"] = 0
    else:
        runtime["pilot_checkpoint_path"] = ""
        runtime["pilot_train_steps"] = int(train_steps)
        runtime["pilot_snapshot_steps"] = _snapshot_steps(int(train_steps))
        runtime["pilot_gradient_probe_enabled"] = True
        runtime["pilot_gradient_probe_interval"] = 5
        runtime["pilot_gradient_probe_max_steps"] = int(train_steps)

    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize PLANv8 V8-9 CDMI configs.")
    parser.add_argument("--base_config", type=Path, required=True)
    parser.add_argument("--output_config", type=Path, required=True)
    parser.add_argument("--condition_id", type=str, required=True)
    parser.add_argument("--eval_task_name", type=str, required=True)
    parser.add_argument("--support_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--source_variant_id", type=str, required=True)
    parser.add_argument("--source_phase", type=str, required=True)
    parser.add_argument("--source_arm_id", type=str, required=True)
    parser.add_argument("--source_interface_family", type=str, required=True)
    parser.add_argument("--source_bridge_family", type=str, required=True)
    parser.add_argument("--source_auxiliary_family", type=str, required=True)
    parser.add_argument("--source_prompt_variant", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--primary_model_dir", type=str, default="")
    parser.add_argument("--primary_backbone_name", type=str, default="")
    parser.add_argument("--train_steps", type=int, default=400)
    args = parser.parse_args()

    materialize_planv8_v8_9_config(
        base_config_path=args.base_config,
        output_config=args.output_config,
        condition_id=args.condition_id,
        eval_task_name=args.eval_task_name,
        support_path=args.support_path,
        train_path=args.train_path,
        eval_path=args.eval_path,
        source_variant_id=args.source_variant_id,
        source_phase=args.source_phase,
        source_arm_id=args.source_arm_id,
        source_interface_family=args.source_interface_family,
        source_bridge_family=args.source_bridge_family,
        source_auxiliary_family=args.source_auxiliary_family,
        source_prompt_variant=args.source_prompt_variant,
        checkpoint_path=args.checkpoint_path,
        primary_model_dir=args.primary_model_dir,
        primary_backbone_name=args.primary_backbone_name,
        train_steps=args.train_steps,
    )


if __name__ == "__main__":
    main()
