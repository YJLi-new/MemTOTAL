#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from memtotal.training.m4_shared_injection import run_shared_injection_pilot
from memtotal.utils.config import load_config
from memtotal.utils.io import write_json
from memtotal.utils.repro import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the selected M4 fixed64 gate from a preregistered selection.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection_json", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    selection = json.loads(Path(args.selection_json).read_text())
    if not bool(selection.get("selection_passed")):
        raise ValueError(f"Selection did not pass: {args.selection_json}")
    prompt_variant = str(selection["selected_prompt_variant"])
    support_serialization = str(selection["selected_support_serialization"])
    i_real_checkpoint = str(selection["i_real_checkpoint_path"])
    i_shuffle_checkpoint = str(selection["i_shuffle_checkpoint_path"])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    arm_specs = [
        ("pilot-A-gate", "A", "base_only", "real", "", 0),
        ("pilot-T-gate", "T", "teacher_text", "real", "", 2),
        ("pilot-I-real-gate", "I_real", "injected", "real", i_real_checkpoint, 10),
        ("pilot-I-shuffle-gate", "I_shuffle", "injected", "shuffled", i_shuffle_checkpoint, 12),
        ("pilot-I-zero-gate", "I_zero", "injected", "zero", "", 14),
    ]
    suite_rows: list[dict[str, object]] = []
    for subdir, alias, arm, memory_control, checkpoint_path, seed_offset in arm_specs:
        arm_config = copy.deepcopy(config)
        arm_config["runtime"]["pilot_prompt_variant"] = prompt_variant
        arm_config["runtime"]["pilot_support_serialization"] = support_serialization
        arm_config["runtime"]["pilot_arm_alias"] = alias
        arm_config["runtime"]["shared_injection_arm"] = arm
        arm_config["runtime"]["writer_memory_control"] = memory_control
        arm_config["runtime"]["pilot_train_steps"] = 0
        arm_config["runtime"]["pilot_projector_warmup_steps"] = 0
        if checkpoint_path:
            arm_config["runtime"]["pilot_checkpoint_path"] = checkpoint_path
        elif "pilot_checkpoint_path" in arm_config["runtime"]:
            arm_config["runtime"].pop("pilot_checkpoint_path")
        arm_seed = args.seed + seed_offset
        set_seed(arm_seed)
        metrics = run_shared_injection_pilot(
            config=arm_config,
            seed=arm_seed,
            output_dir=output_root / subdir,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        suite_rows.append(
            {
                "subdir": subdir,
                "alias": alias,
                "arm": arm,
                "writer_memory_control": memory_control,
                "task_score": float(metrics["best_adapt_task_score"]),
                "macro_f1": float(metrics.get("best_adapt_macro_f1", 0.0)),
            }
        )
    write_json(
        output_root / "suite_metrics.json",
        {
            "selected_prompt_variant": prompt_variant,
            "selected_support_serialization": support_serialization,
            "selection_json": str(Path(args.selection_json).resolve()),
            "rows": suite_rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
