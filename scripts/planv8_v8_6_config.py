#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARM_ORDER = (
    "a0_none",
    "a1_barlow",
    "a2_recon_bow",
    "a3_writer_opd_ans",
    "a4_writer_opd_ansctx",
    "a5_writer_opd_plus_recon",
)

ARM_METADATA = {
    "a0_none": {
        "auxiliary_family": "none",
        "aux_loss_mode": "task_only",
        "aux_projection_dim": 0,
        "aux_projection_hidden_dim": None,
        "barlow_loss_weight": 0.0,
        "barlow_lambda": 5.0e-3,
        "reconstruction_aux_mode": "off",
        "reconstruction_aux_weight": 0.0,
        "reconstruction_vocab_size": 1024,
        "reconstruction_hidden_dim": 1024,
        "reconstruction_weight_schedule": "constant",
        "alignment_aux_mode": "off",
        "alignment_aux_weight_max": 0.0,
        "start_step": 0,
        "ramp_steps": 0,
        "opd_weight_max": 0.0,
        "gsm8k_hint_mode": "answer_only",
        "triviaqa_hint_mode": "answer_only",
        "fever_hint_mode": "label_only",
    },
    "a1_barlow": {
        "auxiliary_family": "barlow_lite",
        "aux_loss_mode": "barlow",
        "aux_projection_dim": 128,
        "aux_projection_hidden_dim": 256,
        "barlow_loss_weight": 0.02,
        "barlow_lambda": 5.0e-3,
        "reconstruction_aux_mode": "off",
        "reconstruction_aux_weight": 0.0,
        "reconstruction_vocab_size": 1024,
        "reconstruction_hidden_dim": 1024,
        "reconstruction_weight_schedule": "constant",
        "alignment_aux_mode": "off",
        "alignment_aux_weight_max": 0.0,
        "start_step": 0,
        "ramp_steps": 0,
        "opd_weight_max": 0.0,
        "gsm8k_hint_mode": "answer_only",
        "triviaqa_hint_mode": "answer_only",
        "fever_hint_mode": "label_only",
    },
    "a2_recon_bow": {
        "auxiliary_family": "reconstruction_lite",
        "aux_loss_mode": "task_only",
        "aux_projection_dim": 0,
        "aux_projection_hidden_dim": None,
        "barlow_loss_weight": 0.0,
        "barlow_lambda": 5.0e-3,
        "reconstruction_aux_mode": "hashed_bow",
        "reconstruction_aux_weight": 0.02,
        "reconstruction_vocab_size": 1024,
        "reconstruction_hidden_dim": 1024,
        "reconstruction_weight_schedule": "constant",
        "alignment_aux_mode": "off",
        "alignment_aux_weight_max": 0.0,
        "start_step": 0,
        "ramp_steps": 0,
        "opd_weight_max": 0.0,
        "gsm8k_hint_mode": "answer_only",
        "triviaqa_hint_mode": "answer_only",
        "fever_hint_mode": "label_only",
    },
    "a3_writer_opd_ans": {
        "auxiliary_family": "writer_opd_answer_only",
        "aux_loss_mode": "task_only",
        "aux_projection_dim": 0,
        "aux_projection_hidden_dim": None,
        "barlow_loss_weight": 0.0,
        "barlow_lambda": 5.0e-3,
        "reconstruction_aux_mode": "off",
        "reconstruction_aux_weight": 0.0,
        "reconstruction_vocab_size": 1024,
        "reconstruction_hidden_dim": 1024,
        "reconstruction_weight_schedule": "constant",
        "alignment_aux_mode": "opd_token_ce",
        "alignment_aux_weight_max": 0.1,
        "start_step": 80,
        "ramp_steps": 80,
        "opd_weight_max": 0.1,
        "gsm8k_hint_mode": "answer_only",
        "triviaqa_hint_mode": "answer_only",
        "fever_hint_mode": "label_only",
    },
    "a4_writer_opd_ansctx": {
        "auxiliary_family": "writer_opd_answer_plus_context",
        "aux_loss_mode": "task_only",
        "aux_projection_dim": 0,
        "aux_projection_hidden_dim": None,
        "barlow_loss_weight": 0.0,
        "barlow_lambda": 5.0e-3,
        "reconstruction_aux_mode": "off",
        "reconstruction_aux_weight": 0.0,
        "reconstruction_vocab_size": 1024,
        "reconstruction_hidden_dim": 1024,
        "reconstruction_weight_schedule": "constant",
        "alignment_aux_mode": "opd_token_ce",
        "alignment_aux_weight_max": 0.1,
        "start_step": 80,
        "ramp_steps": 80,
        "opd_weight_max": 0.1,
        "gsm8k_hint_mode": "answer_plus_rationale",
        "triviaqa_hint_mode": "answer_plus_evidence",
        "fever_hint_mode": "label_plus_evidence",
    },
    "a5_writer_opd_plus_recon": {
        "auxiliary_family": "writer_opd_plus_reconstruction",
        "aux_loss_mode": "task_only",
        "aux_projection_dim": 0,
        "aux_projection_hidden_dim": None,
        "barlow_loss_weight": 0.0,
        "barlow_lambda": 5.0e-3,
        "reconstruction_aux_mode": "hashed_bow",
        "reconstruction_aux_weight": 0.02,
        "reconstruction_vocab_size": 1024,
        "reconstruction_hidden_dim": 1024,
        "reconstruction_weight_schedule": "constant",
        "alignment_aux_mode": "opd_token_ce",
        "alignment_aux_weight_max": 0.1,
        "start_step": 80,
        "ramp_steps": 80,
        "opd_weight_max": 0.1,
        "gsm8k_hint_mode": "answer_plus_rationale",
        "triviaqa_hint_mode": "answer_plus_evidence",
        "fever_hint_mode": "label_plus_evidence",
    },
}


def materialize_planv8_v8_6_config(
    *,
    base_config_path: Path,
    base_checkpoint_path: Path,
    arm_id: str,
    output_config: Path,
    v85_summary_path: Path,
    primary_model_dir: str = "",
    primary_backbone_name: str = "",
) -> dict[str, Any]:
    if arm_id not in ARM_METADATA:
        raise ValueError(f"Unsupported V8-6 arm {arm_id}.")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing V8-5 base config: {base_config_path}")
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing V8-5 base checkpoint: {base_checkpoint_path}")

    config = json.loads(base_config_path.read_text())
    config.setdefault("experiment", {})
    config.setdefault("backbone", {})
    config.setdefault("runtime", {})
    config.setdefault("method", {})
    config["method"].setdefault("writer", {})

    v85_summary = json.loads(v85_summary_path.read_text())
    selected_interface_family = str(
        v85_summary.get("selected_interface_family_for_v8_6")
        or v85_summary.get("selected_interface_family_for_v8_7")
        or ""
    ).strip()
    selected_bridge_family = str(
        v85_summary.get("selected_bridge_family_for_v8_6")
        or v85_summary.get("selected_bridge_family_for_v8_7")
        or ""
    ).strip()
    base_arm_id = str(v85_summary.get("base_for_v8_6_arm_id") or v85_summary.get("best_arm_id") or "").strip()

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
    experiment["stage"] = "V8-6"
    experiment["method_variant"] = arm_id

    runtime["pilot_arm_alias"] = arm_id
    runtime["pilot_v85_base_arm_id"] = base_arm_id
    runtime["pilot_v85_selected_interface_family"] = selected_interface_family
    runtime["pilot_v85_selected_bridge_family"] = selected_bridge_family
    runtime["pilot_active_aux_family"] = str(arm_spec["auxiliary_family"])
    runtime["pilot_init_checkpoint_path"] = str(base_checkpoint_path.resolve())
    runtime["pilot_init_checkpoint_mode"] = "full"
    runtime["pilot_checkpoint_path"] = ""
    runtime["pilot_train_steps"] = 300
    runtime["pilot_snapshot_steps"] = [0, 10, 25, 50, 100, 150, 200, 250, 300]
    runtime["pilot_gradient_probe_enabled"] = True
    runtime["pilot_gradient_probe_interval"] = 5
    runtime["pilot_gradient_probe_max_steps"] = 300
    runtime["pilot_trainable_variant"] = "writer_then_joint"
    runtime["stage_a_steps"] = 80
    runtime["stage_b_steps"] = 220
    runtime["pilot_aux_loss_mode"] = str(arm_spec["aux_loss_mode"])
    runtime["pilot_aux_projection_dim"] = int(arm_spec["aux_projection_dim"])
    runtime["pilot_aux_projection_hidden_dim"] = arm_spec["aux_projection_hidden_dim"]
    runtime["pilot_barlow_loss_weight"] = float(arm_spec["barlow_loss_weight"])
    runtime["pilot_barlow_lambda"] = float(arm_spec["barlow_lambda"])
    runtime["pilot_reconstruction_aux_mode"] = str(arm_spec["reconstruction_aux_mode"])
    runtime["pilot_reconstruction_aux_weight"] = float(arm_spec["reconstruction_aux_weight"])
    runtime["pilot_reconstruction_vocab_size"] = int(arm_spec["reconstruction_vocab_size"])
    runtime["pilot_reconstruction_hidden_dim"] = int(arm_spec["reconstruction_hidden_dim"])
    runtime["pilot_reconstruction_weight_schedule"] = str(arm_spec["reconstruction_weight_schedule"])
    runtime["pilot_alignment_aux_mode"] = str(arm_spec["alignment_aux_mode"])
    runtime["pilot_alignment_aux_weight"] = 0.0
    runtime["pilot_alignment_aux_weight_max"] = float(arm_spec["alignment_aux_weight_max"])
    runtime["pilot_alignment_aux_start_step"] = int(arm_spec["start_step"])
    runtime["pilot_alignment_aux_ramp_steps"] = int(arm_spec["ramp_steps"])
    runtime["pilot_opd_weight_max"] = float(arm_spec["opd_weight_max"])
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
    parser.add_argument("--v85_summary_path", type=Path, required=True)
    parser.add_argument("--primary_model_dir", type=str, default="")
    parser.add_argument("--primary_backbone_name", type=str, default="")
    args = parser.parse_args()

    materialize_planv8_v8_6_config(
        base_config_path=args.base_config,
        base_checkpoint_path=args.base_checkpoint,
        arm_id=args.arm_id,
        output_config=args.output_config,
        v85_summary_path=args.v85_summary_path,
        primary_model_dir=args.primary_model_dir,
        primary_backbone_name=args.primary_backbone_name,
    )


if __name__ == "__main__":
    main()
