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
    parser = argparse.ArgumentParser(description="Run the selected M4 shared injection suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase0_metrics")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--prompt-variant")
    parser.add_argument("--support-serialization")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument(
        "--arm-spec",
        action="append",
        default=[],
        help=(
            "Override the default suite arms. Format: "
            "subdir:alias:arm:writer_memory_control:seed_offset"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_arm_specs(raw_specs: list[str]) -> list[tuple[str, str, str, str, int]]:
    if not raw_specs:
        return [
            ("pilot-A-selected", "A", "base_only", "real", 0),
            ("pilot-T-selected", "T", "teacher_text", "real", 2),
            ("pilot-I-real", "I_real", "injected", "real", 10),
            ("pilot-I-shuffle", "I_shuffle", "injected", "shuffled", 12),
            ("pilot-I-zero", "I_zero", "injected", "zero", 14),
        ]
    parsed: list[tuple[str, str, str, str, int]] = []
    for raw_spec in raw_specs:
        parts = raw_spec.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"Invalid --arm-spec={raw_spec!r}. Expected subdir:alias:arm:writer_memory_control:seed_offset."
            )
        subdir, alias, arm, memory_control, seed_offset = parts
        parsed.append((subdir, alias, arm, memory_control, int(seed_offset)))
    return parsed


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    phase0_metrics = json.loads(Path(args.phase0_metrics).read_text()) if args.phase0_metrics else {}
    prompt_variant = str(
        args.prompt_variant
        or phase0_metrics.get("selected_prompt_variant")
        or config.get("runtime", {}).get("pilot_prompt_variant", "inline_short_labels")
    )
    support_serialization_variant = str(
        args.support_serialization
        or phase0_metrics.get("selected_support_serialization")
        or config.get("runtime", {}).get("pilot_support_serialization", "flat_raw8")
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    arm_specs = _parse_arm_specs(args.arm_spec)
    suite_rows: list[dict[str, object]] = []
    for subdir, alias, arm, memory_control, seed_offset in arm_specs:
        arm_config = copy.deepcopy(config)
        arm_config["runtime"]["pilot_prompt_variant"] = prompt_variant
        arm_config["runtime"]["pilot_support_serialization"] = support_serialization_variant
        arm_config["runtime"]["pilot_arm_alias"] = alias
        arm_config["runtime"]["shared_injection_arm"] = arm
        arm_config["runtime"]["writer_memory_control"] = memory_control
        if args.train_steps is not None:
            arm_config["runtime"]["pilot_train_steps"] = int(args.train_steps)
        if args.warmup_steps is not None:
            arm_config["runtime"]["pilot_projector_warmup_steps"] = int(args.warmup_steps)
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
            "selected_support_serialization": support_serialization_variant,
            "rows": suite_rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
