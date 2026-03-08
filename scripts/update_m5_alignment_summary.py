#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_m5_alignment_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize M5.1 canonical vs ablations alignment runs.")
    parser.add_argument("--canonical_summary_json", required=True)
    parser.add_argument("--freeze_writer_summary_json", required=True)
    parser.add_argument("--pooled_block_summary_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_m5_alignment_runs(
        canonical_summary_json=args.canonical_summary_json,
        freeze_writer_summary_json=args.freeze_writer_summary_json,
        pooled_block_summary_json=args.pooled_block_summary_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# M5.1 Alignment Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- failure_reason: {summary['failure_reason']}",
        f"- alignment_claim_supported: {summary['alignment_claim_supported']}",
        f"- canonical_selection_passed: {summary['canonical_selection_passed']}",
        f"- canonical_selected_step: {summary['canonical_selected_step']}",
        f"- canonical_primary_gate_passed: {summary['canonical_primary_gate_passed']}",
        f"- canonical_support_bank_brittle: {summary['canonical_support_bank_brittle']}",
        f"- canonical_fixed64_report_generated: {summary['canonical_fixed64_report_generated']}",
        f"- canonical_fixed64_gate_passed: {summary['canonical_fixed64_gate_passed']}",
        f"- freeze_writer_selection_passed: {summary['freeze_writer_selection_passed']}",
        f"- freeze_writer_selected_step: {summary['freeze_writer_selected_step']}",
        f"- freeze_writer_primary_gate_passed: {summary['freeze_writer_primary_gate_passed']}",
        f"- pooled_block_selection_passed: {summary['pooled_block_selection_passed']}",
        f"- pooled_block_selected_step: {summary['pooled_block_selected_step']}",
        f"- pooled_block_primary_gate_passed: {summary['pooled_block_primary_gate_passed']}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
