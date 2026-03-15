#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_tl_micro_lora_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv3 V2 micro-LoRA FEVER runs.")
    parser.add_argument("--control_metrics_json", required=True)
    parser.add_argument("--control_run_summary_json", required=True)
    parser.add_argument("--late3_metrics_json", required=True)
    parser.add_argument("--late3_run_summary_json", required=True)
    parser.add_argument("--control_train_events_json")
    parser.add_argument("--late3_train_events_json")
    parser.add_argument("--all5_metrics_json")
    parser.add_argument("--all5_run_summary_json")
    parser.add_argument("--all5_train_events_json")
    parser.add_argument("--rank4_metrics_json")
    parser.add_argument("--rank4_run_summary_json")
    parser.add_argument("--rank4_train_events_json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def _arm_section(summary: dict[str, object], prefix: str, title: str) -> list[str]:
    if f"{prefix}_receiver_lora_enabled" not in summary:
        return []
    lines = [
        f"## {title}",
        f"- receiver_lora_enabled: {summary[f'{prefix}_receiver_lora_enabled']}",
        f"- receiver_lora_rank: {summary.get(f'{prefix}_receiver_lora_rank', 0)}",
        f"- receiver_lora_target_layers: {summary.get(f'{prefix}_receiver_lora_target_layers', [])}",
        f"- receiver_lora_target_modules: {summary.get(f'{prefix}_receiver_lora_target_modules', [])}",
        f"- receiver_lora_trainable_params: {summary.get(f'{prefix}_receiver_lora_trainable_params', 0)}",
        f"- best_adapt_task_score: {summary[f'{prefix}_best_adapt_task_score']:.4f}",
        f"- best_adapt_macro_f1: {summary[f'{prefix}_best_adapt_macro_f1']:.4f}",
        f"- dominant_label_collapse_onset_step: {summary[f'{prefix}_dominant_label_collapse_onset_step']}",
        f"- selection_passed: {summary[f'{prefix}_selection_passed']}",
        f"- screen248_test_gate_passed: {summary[f'{prefix}_screen248_test_gate_passed']}",
        f"- train_grad_norm_reader_steps_1_4_median: {summary[f'{prefix}_train_grad_norm_reader_steps_1_4_median']:.6f}",
        f"- train_grad_norm_fuser_steps_1_4_median: {summary[f'{prefix}_train_grad_norm_fuser_steps_1_4_median']:.6f}",
    ]
    if f"{prefix}_train_grad_norm_receiver_lora_steps_1_4_median" in summary:
        lines.append(
            f"- train_grad_norm_receiver_lora_steps_1_4_median: "
            f"{summary[f'{prefix}_train_grad_norm_receiver_lora_steps_1_4_median']:.6f}"
        )
    if f"{prefix}_reader_grad_boost" in summary:
        lines.extend(
            [
                f"- reader_grad_boost: {summary[f'{prefix}_reader_grad_boost']:.4f}",
                f"- fuser_grad_boost: {summary[f'{prefix}_fuser_grad_boost']:.4f}",
                f"- task_score_delta: {summary[f'{prefix}_task_score_delta']:.4f}",
                f"- collapse_delayed: {summary[f'{prefix}_collapse_delayed']}",
                f"- diagnostic_success: {summary[f'{prefix}_diagnostic_success']}",
                f"- partial_evidence: {summary[f'{prefix}_partial_evidence']}",
                f"- medium_success: {summary[f'{prefix}_medium_success']}",
                f"- strong_success: {summary[f'{prefix}_strong_success']}",
            ]
        )
    return lines + [""]


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_tl_micro_lora_runs(
        control_metrics_json=args.control_metrics_json,
        control_run_summary_json=args.control_run_summary_json,
        late3_metrics_json=args.late3_metrics_json,
        late3_run_summary_json=args.late3_run_summary_json,
        control_train_events_json=args.control_train_events_json,
        late3_train_events_json=args.late3_train_events_json,
        all5_metrics_json=args.all5_metrics_json,
        all5_run_summary_json=args.all5_run_summary_json,
        all5_train_events_json=args.all5_train_events_json,
        rank4_metrics_json=args.rank4_metrics_json,
        rank4_run_summary_json=args.rank4_run_summary_json,
        rank4_train_events_json=args.rank4_train_events_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# TL Micro-LoRA V2 Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- primary_interpretation: {summary['primary_interpretation']}",
        f"- recommended_arm: {summary['recommended_arm']}",
        f"- continue_to_l2: {summary['continue_to_l2']}",
        f"- continue_to_l3: {summary['continue_to_l3']}",
        f"- move_to_v3: {summary['move_to_v3']}",
        f"- move_to_v4: {summary['move_to_v4']}",
        f"- failure_reason: {summary['failure_reason']}",
        "",
    ]
    report_lines.extend(_arm_section(summary, "control", "Control"))
    report_lines.extend(_arm_section(summary, "late3", "Late3 R2"))
    report_lines.extend(_arm_section(summary, "all5", "All5 R2"))
    report_lines.extend(_arm_section(summary, "rank4", "Late3 R4"))
    Path(args.output_report).write_text("\n".join(report_lines).rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
