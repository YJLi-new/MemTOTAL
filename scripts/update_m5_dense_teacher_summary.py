#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_m5_dense_teacher_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize M5.3 dense-teacher runs.")
    parser.add_argument("--canonical_summary_json", required=True)
    parser.add_argument("--control_summary_json", required=True)
    parser.add_argument("--hinge_off_audit_summary_json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_m5_dense_teacher_runs(
        canonical_summary_json=args.canonical_summary_json,
        control_summary_json=args.control_summary_json,
        hinge_off_audit_summary_json=args.hinge_off_audit_summary_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# M5.3 Dense Teacher Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- failure_reason: {summary['failure_reason']}",
        f"- informative_success: {summary['informative_success']}",
        f"- canonical_selection_passed: {summary['canonical_selection_passed']}",
        f"- canonical_selected_step: {summary['canonical_selected_step']}",
        f"- canonical_primary_gate_passed: {summary['canonical_primary_gate_passed']}",
        f"- canonical_support_bank_brittle: {summary['canonical_support_bank_brittle']}",
        f"- canonical_fixed64_report_generated: {summary['canonical_fixed64_report_generated']}",
        f"- canonical_fixed64_gate_passed: {summary['canonical_fixed64_gate_passed']}",
        f"- canonical_cap_saturation_onset_step: {summary['canonical_cap_saturation_onset_step']}",
        f"- canonical_dominant_label_collapse_onset_step: {summary['canonical_dominant_label_collapse_onset_step']}",
        f"- control_selection_passed: {summary['control_selection_passed']}",
        f"- control_selected_step: {summary['control_selected_step']}",
        f"- control_primary_gate_passed: {summary['control_primary_gate_passed']}",
        f"- control_support_bank_brittle: {summary['control_support_bank_brittle']}",
        f"- control_cap_saturation_onset_step: {summary['control_cap_saturation_onset_step']}",
        f"- control_dominant_label_collapse_onset_step: {summary['control_dominant_label_collapse_onset_step']}",
        f"- canonical_less_collapsed: {summary['canonical_less_collapsed']}",
        f"- canonical_less_saturated: {summary['canonical_less_saturated']}",
        f"- hinge_off_audit_present: {summary['hinge_off_audit_present']}",
        f"- hinge_off_audit_selection_passed: {summary['hinge_off_audit_selection_passed']}",
        f"- hinge_off_audit_primary_gate_passed: {summary['hinge_off_audit_primary_gate_passed']}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
