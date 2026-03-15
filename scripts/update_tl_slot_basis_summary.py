#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_tl_slot_basis_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize TL slot-basis rescue follow-up runs.")
    parser.add_argument("--sl8_summary_json", required=True)
    parser.add_argument("--tl_h4_k8_summary_json", required=True)
    parser.add_argument("--tl_bridge_rescue_summary_json", required=True)
    parser.add_argument("--tl_slot_basis_summary_json", required=True)
    parser.add_argument("--tl_h4_k8_reader_query_csv")
    parser.add_argument("--tl_bridge_rescue_reader_query_csv")
    parser.add_argument("--tl_slot_basis_reader_query_csv")
    parser.add_argument("--tl_h4_k8_train_events_json")
    parser.add_argument("--tl_bridge_rescue_train_events_json")
    parser.add_argument("--tl_slot_basis_train_events_json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_tl_slot_basis_runs(
        sl8_summary_json=args.sl8_summary_json,
        tl_h4_k8_summary_json=args.tl_h4_k8_summary_json,
        tl_bridge_rescue_summary_json=args.tl_bridge_rescue_summary_json,
        tl_slot_basis_summary_json=args.tl_slot_basis_summary_json,
        tl_h4_k8_reader_query_csv=args.tl_h4_k8_reader_query_csv,
        tl_bridge_rescue_reader_query_csv=args.tl_bridge_rescue_reader_query_csv,
        tl_slot_basis_reader_query_csv=args.tl_slot_basis_reader_query_csv,
        tl_h4_k8_train_events_json=args.tl_h4_k8_train_events_json,
        tl_bridge_rescue_train_events_json=args.tl_bridge_rescue_train_events_json,
        tl_slot_basis_train_events_json=args.tl_slot_basis_train_events_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# TL Slot-Basis Rescue Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- failure_reason: {summary['failure_reason']}",
        f"- tl_slot_basis_selection_passed: {summary['tl_slot_basis_selection_passed']}",
        f"- tl_slot_basis_primary_gate_passed: {summary['tl_slot_basis_primary_gate_passed']}",
        f"- basis_selection_improved: {summary['basis_selection_improved']}",
        f"- basis_primary_gate_improved: {summary['basis_primary_gate_improved']}",
        f"- basis_collapse_delayed: {summary['basis_collapse_delayed']}",
        f"- basis_reader_specialization_improved: {summary['basis_reader_specialization_improved']}",
        f"- basis_geometry_improved: {summary['basis_geometry_improved']}",
        f"- basis_bridge_supported_vs_sl8: {summary['basis_bridge_supported_vs_sl8']}",
        f"- tl_h4_k8_reader_query_argmax_unique_mean: {summary['tl_h4_k8_reader_query_argmax_unique_mean']:.4f}",
        f"- tl_bridge_rescue_reader_query_argmax_unique_mean: {summary['tl_bridge_rescue_reader_query_argmax_unique_mean']:.4f}",
        f"- tl_slot_basis_reader_query_argmax_unique_mean: {summary['tl_slot_basis_reader_query_argmax_unique_mean']:.4f}",
        f"- tl_h4_k8_final_memory_long_effective_rank: {summary['tl_h4_k8_final_memory_long_effective_rank']:.4f}",
        f"- tl_bridge_rescue_final_memory_long_effective_rank: {summary['tl_bridge_rescue_final_memory_long_effective_rank']:.4f}",
        f"- tl_slot_basis_final_memory_long_effective_rank: {summary['tl_slot_basis_final_memory_long_effective_rank']:.4f}",
        f"- tl_slot_basis_final_writer_slot_basis_pairwise_cosine_mean: {summary['tl_slot_basis_final_writer_slot_basis_pairwise_cosine_mean']:.4f}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
