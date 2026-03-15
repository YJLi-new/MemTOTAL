#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _snapshot_steps(train_steps: int) -> list[int]:
    checkpoints = {0, 10, 25, 50, 100, 150, 200, 250, 300, 350, train_steps}
    return sorted(step for step in checkpoints if 0 <= step <= train_steps)


def materialize_planv8_v8_8_config(
    *,
    base_config_path: Path,
    output_config: Path,
    variant_id: str,
    source_phase: str,
    source_arm_id: str,
    primary_model_dir: str = "",
    primary_backbone_name: str = "",
    train_steps: int = 400,
) -> dict[str, Any]:
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing V8-8 base config: {base_config_path}")

    config = json.loads(base_config_path.read_text())
    config.setdefault("experiment", {})
    config.setdefault("backbone", {})
    config.setdefault("runtime", {})

    task_name = str(config.get("task", {}).get("benchmark_id", "")).strip() or base_config_path.stem
    experiment = config["experiment"]
    backbone = config["backbone"]
    runtime = config["runtime"]

    if primary_model_dir:
        backbone["model_id"] = primary_model_dir
    if primary_backbone_name:
        backbone["name"] = primary_backbone_name

    experiment["name"] = f"{Path().resolve().name}_{variant_id}_{task_name}"
    experiment["stage"] = "V8-8"
    experiment["method_variant"] = variant_id

    runtime["pilot_arm_alias"] = variant_id
    runtime["pilot_confirmation_source_phase"] = str(source_phase)
    runtime["pilot_confirmation_source_arm_id"] = str(source_arm_id)
    runtime["pilot_train_steps"] = int(train_steps)
    runtime["pilot_snapshot_steps"] = _snapshot_steps(int(train_steps))
    runtime["pilot_gradient_probe_enabled"] = True
    runtime["pilot_gradient_probe_interval"] = 5
    runtime["pilot_gradient_probe_max_steps"] = int(train_steps)
    runtime["pilot_checkpoint_path"] = ""

    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize PLANv8 V8-8 confirmation configs.")
    parser.add_argument("--base_config", type=Path, required=True)
    parser.add_argument("--output_config", type=Path, required=True)
    parser.add_argument("--variant_id", type=str, required=True)
    parser.add_argument("--source_phase", type=str, required=True)
    parser.add_argument("--source_arm_id", type=str, required=True)
    parser.add_argument("--primary_model_dir", type=str, default="")
    parser.add_argument("--primary_backbone_name", type=str, default="")
    parser.add_argument("--train_steps", type=int, default=400)
    args = parser.parse_args()

    materialize_planv8_v8_8_config(
        base_config_path=args.base_config,
        output_config=args.output_config,
        variant_id=args.variant_id,
        source_phase=args.source_phase,
        source_arm_id=args.source_arm_id,
        primary_model_dir=args.primary_model_dir,
        primary_backbone_name=args.primary_backbone_name,
        train_steps=args.train_steps,
    )


if __name__ == "__main__":
    main()
