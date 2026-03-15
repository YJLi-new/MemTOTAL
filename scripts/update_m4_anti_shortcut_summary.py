#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_m4_anti_shortcut_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize M4 anti-shortcut run A vs run B.")
    parser.add_argument("--run_a_summary_json", required=True)
    parser.add_argument("--run_b_summary_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_m4_anti_shortcut_runs(
        run_a_summary_json=args.run_a_summary_json,
        run_b_summary_json=args.run_b_summary_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# M4 Anti-Shortcut Comparison",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- run_a_selection_passed: {summary['run_a_selection_passed']}",
        f"- run_b_selection_passed: {summary['run_b_selection_passed']}",
        f"- run_a_primary_gate_passed: {summary['run_a_primary_gate_passed']}",
        f"- run_b_primary_gate_passed: {summary['run_b_primary_gate_passed']}",
        f"- run_a_support_bank_brittle: {summary['run_a_support_bank_brittle']}",
        f"- run_b_support_bank_brittle: {summary['run_b_support_bank_brittle']}",
        f"- run_a_cap_saturation_onset_step: {summary['run_a_cap_saturation_onset_step']}",
        f"- run_b_cap_saturation_onset_step: {summary['run_b_cap_saturation_onset_step']}",
        f"- run_a_dominant_label_collapse_onset_step: {summary['run_a_dominant_label_collapse_onset_step']}",
        f"- run_b_dominant_label_collapse_onset_step: {summary['run_b_dominant_label_collapse_onset_step']}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
