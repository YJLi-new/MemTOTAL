#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_m5_objective_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize M5.2 objective rewrite runs.")
    parser.add_argument("--canonical_summary_json", required=True)
    parser.add_argument("--anchor_only_summary_json", required=True)
    parser.add_argument("--task_only_control_summary_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_m5_objective_runs(
        canonical_summary_json=args.canonical_summary_json,
        anchor_only_summary_json=args.anchor_only_summary_json,
        task_only_control_summary_json=args.task_only_control_summary_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# M5.2 Objective Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- failure_reason: {summary['failure_reason']}",
        f"- objective_rewrite_supported: {summary['objective_rewrite_supported']}",
        f"- teacher_margin_increment_supported: {summary['teacher_margin_increment_supported']}",
        f"- canonical_selection_passed: {summary['canonical_selection_passed']}",
        f"- canonical_selected_step: {summary['canonical_selected_step']}",
        f"- canonical_primary_gate_passed: {summary['canonical_primary_gate_passed']}",
        f"- canonical_support_bank_brittle: {summary['canonical_support_bank_brittle']}",
        f"- canonical_fixed64_report_generated: {summary['canonical_fixed64_report_generated']}",
        f"- canonical_fixed64_gate_passed: {summary['canonical_fixed64_gate_passed']}",
        f"- anchor_only_selection_passed: {summary['anchor_only_selection_passed']}",
        f"- anchor_only_selected_step: {summary['anchor_only_selected_step']}",
        f"- anchor_only_primary_gate_passed: {summary['anchor_only_primary_gate_passed']}",
        f"- task_only_control_selection_passed: {summary['task_only_control_selection_passed']}",
        f"- task_only_control_selected_step: {summary['task_only_control_selected_step']}",
        f"- task_only_control_primary_gate_passed: {summary['task_only_control_primary_gate_passed']}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
