#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_tl_poc_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Workstream B TL-PoC runs.")
    parser.add_argument("--sl8_summary_json", required=True)
    parser.add_argument("--tl_h4_k8_summary_json", required=True)
    parser.add_argument("--tl_h4_k4_summary_json", required=True)
    parser.add_argument("--tl_h1_k4_summary_json", required=True)
    parser.add_argument("--tl_h4_k8_reader_query_csv")
    parser.add_argument("--tl_h4_k4_reader_query_csv")
    parser.add_argument("--tl_h1_k4_reader_query_csv")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_tl_poc_runs(
        sl8_summary_json=args.sl8_summary_json,
        tl_h4_k8_summary_json=args.tl_h4_k8_summary_json,
        tl_h4_k4_summary_json=args.tl_h4_k4_summary_json,
        tl_h1_k4_summary_json=args.tl_h1_k4_summary_json,
        tl_h4_k8_reader_query_csv=args.tl_h4_k8_reader_query_csv,
        tl_h4_k4_reader_query_csv=args.tl_h4_k4_reader_query_csv,
        tl_h1_k4_reader_query_csv=args.tl_h1_k4_reader_query_csv,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# TL-PoC Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- failure_reason: {summary['failure_reason']}",
        f"- bridge_supported: {summary['bridge_supported']}",
        f"- bottleneck_supported: {summary['bottleneck_supported']}",
        f"- specialization_supported: {summary['specialization_supported']}",
        f"- sl8_primary_gate_passed: {summary['sl8_primary_gate_passed']}",
        f"- tl_h4_k8_primary_gate_passed: {summary['tl_h4_k8_primary_gate_passed']}",
        f"- tl_h4_k4_primary_gate_passed: {summary['tl_h4_k4_primary_gate_passed']}",
        f"- tl_h1_k4_primary_gate_passed: {summary['tl_h1_k4_primary_gate_passed']}",
        f"- tl_h4_k8_reader_query_argmax_unique_mean: {summary['tl_h4_k8_reader_query_argmax_unique_mean']:.4f}",
        f"- tl_h4_k4_reader_query_argmax_unique_mean: {summary['tl_h4_k4_reader_query_argmax_unique_mean']:.4f}",
        f"- tl_h1_k4_reader_query_argmax_unique_mean: {summary['tl_h1_k4_reader_query_argmax_unique_mean']:.4f}",
        f"- tl_h4_k8_reader_query_entropy_mean: {summary['tl_h4_k8_reader_query_entropy_mean']:.4f}",
        f"- tl_h4_k4_reader_query_entropy_mean: {summary['tl_h4_k4_reader_query_entropy_mean']:.4f}",
        f"- tl_h1_k4_reader_query_entropy_mean: {summary['tl_h1_k4_reader_query_entropy_mean']:.4f}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
