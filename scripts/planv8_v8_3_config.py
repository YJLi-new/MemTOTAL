#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARM_ORDER = (
    "p0_ce_only",
    "p1_teacher_choice_kl",
    "p2_opd_ansonly_w01",
    "p3_opd_ansonly_w03",
    "p4_opd_ansplusctx_w03",
    "p5_opd_ansplusctx_centered",
)

ARM_METADATA = {
    "p0_ce_only": {
        "alignment_aux_mode": "off",
        "opd_weight_max": 0.0,
        "start_step": 0,
        "ramp_steps": 0,
        "gsm8k_hint_mode": "answer_only",
        "triviaqa_hint_mode": "answer_only",
        "fever_hint_mode": "label_only",
    },
    "p1_teacher_choice_kl": {
        "alignment_aux_mode": "teacher_choice_kl",
        "alignment_aux_weight_max": 0.1,
        "start_step": 60,
        "ramp_steps": 80,
        "gsm8k_hint_mode": "answer_only",
        "triviaqa_hint_mode": "answer_only",
        "fever_hint_mode": "label_only",
    },
    "p2_opd_ansonly_w01": {
        "alignment_aux_mode": "opd_token_ce",
        "opd_weight_max": 0.1,
        "start_step": 60,
        "ramp_steps": 80,
        "gsm8k_hint_mode": "answer_only",
        "triviaqa_hint_mode": "answer_only",
        "fever_hint_mode": "label_only",
    },
    "p3_opd_ansonly_w03": {
        "alignment_aux_mode": "opd_token_ce",
        "opd_weight_max": 0.3,
        "start_step": 60,
        "ramp_steps": 80,
        "gsm8k_hint_mode": "answer_only",
        "triviaqa_hint_mode": "answer_only",
        "fever_hint_mode": "label_only",
    },
    "p4_opd_ansplusctx_w03": {
        "alignment_aux_mode": "opd_token_ce",
        "opd_weight_max": 0.3,
        "start_step": 60,
        "ramp_steps": 80,
        "gsm8k_hint_mode": "answer_plus_rationale",
        "triviaqa_hint_mode": "answer_plus_evidence",
        "fever_hint_mode": "label_plus_evidence",
    },
    "p5_opd_ansplusctx_centered": {
        "alignment_aux_mode": "opd_token_ce_centered",
        "opd_weight_max": 0.3,
        "start_step": 60,
        "ramp_steps": 80,
        "gsm8k_hint_mode": "answer_plus_rationale",
        "triviaqa_hint_mode": "answer_plus_evidence",
        "fever_hint_mode": "label_plus_evidence",
    },
}


def materialize_planv8_v8_3_config(
    *,
    base_config_path: Path,
    base_checkpoint_path: Path,
    arm_id: str,
    output_config: Path,
    v82_summary_path: Path,
    primary_model_dir: str = "",
    primary_backbone_name: str = "",
) -> dict[str, Any]:
    if arm_id not in ARM_METADATA:
        raise ValueError(f"Unsupported V8-3 arm {arm_id}.")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing V8-2 base config: {base_config_path}")
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing V8-2 base checkpoint: {base_checkpoint_path}")

    config = json.loads(base_config_path.read_text())
    config.setdefault("experiment", {})
    config.setdefault("backbone", {})
    config.setdefault("runtime", {})
    config.setdefault("method", {})

    v82_summary = json.loads(v82_summary_path.read_text())
    selected_interface_family = str(
        v82_summary.get("selected_interface_family_for_v8_3")
        or v82_summary.get("best_interface_family")
        or ""
    ).strip()
    base_arm_id = str(v82_summary.get("base_for_v8_3_arm_id") or v82_summary.get("best_arm_id") or "").strip()

    experiment = config["experiment"]
    runtime = config["runtime"]
    backbone = config["backbone"]
    arm_spec = ARM_METADATA[arm_id]
    task_name = str(config.get("task", {}).get("benchmark_id", "")).strip() or base_config_path.stem.split("-")[0]

    if primary_model_dir:
        backbone["model_id"] = primary_model_dir
    if primary_backbone_name:
        backbone["name"] = primary_backbone_name

    experiment["name"] = f"{Path().resolve().name}_{arm_id}_{task_name}"
    experiment["stage"] = "V8-3"
    experiment["method_variant"] = arm_id

    runtime["pilot_arm_alias"] = arm_id
    runtime["pilot_v82_base_arm_id"] = base_arm_id
    runtime["pilot_v82_selected_interface_family"] = selected_interface_family
    runtime["pilot_init_checkpoint_path"] = str(base_checkpoint_path.resolve())
    runtime["pilot_checkpoint_path"] = ""
    runtime["pilot_train_steps"] = 300
    runtime["pilot_snapshot_steps"] = [0, 10, 25, 50, 100, 150, 200, 250, 300]
    runtime["pilot_gradient_probe_enabled"] = True
    runtime["pilot_gradient_probe_interval"] = 5
    runtime["pilot_gradient_probe_max_steps"] = 300
    runtime["pilot_trainable_variant"] = "reader_only"
    runtime["pilot_aux_loss_mode"] = "off"
    runtime["pilot_choice_ce_weight"] = 1.0
    runtime["pilot_alignment_aux_mode"] = str(arm_spec["alignment_aux_mode"])
    runtime["pilot_alignment_aux_weight"] = 0.0
    runtime["pilot_alignment_aux_weight_max"] = float(arm_spec.get("alignment_aux_weight_max", 0.0))
    runtime["pilot_alignment_aux_start_step"] = int(arm_spec["start_step"])
    runtime["pilot_alignment_aux_ramp_steps"] = int(arm_spec["ramp_steps"])
    runtime["pilot_opd_weight_max"] = float(arm_spec.get("opd_weight_max", 0.0))
    runtime["pilot_opd_start_step"] = int(arm_spec["start_step"])
    runtime["pilot_opd_ramp_steps"] = int(arm_spec["ramp_steps"])
    runtime["pilot_opd_scope"] = "reader_only"
    runtime["pilot_opd_teacher_force_gold"] = True
    runtime["pilot_opd_mask_mode"] = "target_only"
    runtime["pilot_opd_advantage_clip"] = 5.0
    runtime["pilot_opd_center"] = 0.0
    runtime["pilot_opd_scale"] = 0.5
    runtime["pilot_opd_hint_mode_gsm8k"] = str(arm_spec["gsm8k_hint_mode"])
    runtime["pilot_opd_hint_mode_triviaqa"] = str(arm_spec["triviaqa_hint_mode"])
    runtime["pilot_opd_hint_mode_fever"] = str(arm_spec["fever_hint_mode"])

    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=Path, required=True)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--arm_id", type=str, required=True, choices=ARM_ORDER)
    parser.add_argument("--output_config", type=Path, required=True)
    parser.add_argument("--v82_summary_path", type=Path, required=True)
    parser.add_argument("--primary_model_dir", type=str, default="")
    parser.add_argument("--primary_backbone_name", type=str, default="")
    args = parser.parse_args()

    materialize_planv8_v8_3_config(
        base_config_path=args.base_config,
        base_checkpoint_path=args.base_checkpoint,
        arm_id=args.arm_id,
        output_config=args.output_config,
        v82_summary_path=args.v82_summary_path,
        primary_model_dir=args.primary_model_dir,
        primary_backbone_name=args.primary_backbone_name,
    )


if __name__ == "__main__":
    main()
