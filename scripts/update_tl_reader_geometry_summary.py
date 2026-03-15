#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_tl_reader_geometry_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RG-1 TL reader geometry probe runs.")
    parser.add_argument("--baseline_summary_json", required=True)
    parser.add_argument("--rg1a_summary_json", required=True)
    parser.add_argument("--rg1b_summary_json", required=True)
    parser.add_argument("--rg1c_summary_json", required=True)
    parser.add_argument("--baseline_reader_query_csv")
    parser.add_argument("--rg1a_reader_query_csv")
    parser.add_argument("--rg1b_reader_query_csv")
    parser.add_argument("--rg1c_reader_query_csv")
    parser.add_argument("--baseline_train_events_json")
    parser.add_argument("--rg1a_train_events_json")
    parser.add_argument("--rg1b_train_events_json")
    parser.add_argument("--rg1c_train_events_json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_tl_reader_geometry_runs(
        baseline_summary_json=args.baseline_summary_json,
        rg1a_summary_json=args.rg1a_summary_json,
        rg1b_summary_json=args.rg1b_summary_json,
        rg1c_summary_json=args.rg1c_summary_json,
        baseline_reader_query_csv=args.baseline_reader_query_csv,
        rg1a_reader_query_csv=args.rg1a_reader_query_csv,
        rg1b_reader_query_csv=args.rg1b_reader_query_csv,
        rg1c_reader_query_csv=args.rg1c_reader_query_csv,
        baseline_train_events_json=args.baseline_train_events_json,
        rg1a_train_events_json=args.rg1a_train_events_json,
        rg1b_train_events_json=args.rg1b_train_events_json,
        rg1c_train_events_json=args.rg1c_train_events_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# TL Reader Geometry RG-1 Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- failure_reason: {summary['failure_reason']}",
        f"- primary_interpretation: {summary['primary_interpretation']}",
        f"- recommended_control_arm: {summary['recommended_control_arm']}",
        f"- move_to_rg2: {summary['move_to_rg2']}",
        f"- context_overwrite_supported: {summary['context_overwrite_supported']}",
        f"- k_eq_h_supported: {summary['k_eq_h_supported']}",
        f"- linear_fuser_supported: {summary['linear_fuser_supported']}",
        "",
        "## Baseline",
        f"- reader_query_entropy_mean: {summary['baseline_reader_query_entropy_mean']:.4f}",
        f"- final_reader_attention_pairwise_cosine_mean: {summary['baseline_final_reader_attention_pairwise_cosine_mean']:.4f}",
        f"- final_memory_short_effective_rank: {summary['baseline_final_memory_short_effective_rank']:.4f}",
        "",
        "## RG-1A CTX-OFF H4-K8",
        f"- meaningful_movement: {summary['rg1a_meaningful_movement']}",
        f"- entropy_delta: {summary['rg1a_entropy_delta']:.4f}",
        f"- pairwise_delta: {summary['rg1a_pairwise_delta']:.4f}",
        f"- short_rank_delta: {summary['rg1a_short_rank_delta']:.4f}",
        f"- final_reader_context_overwrite_ratio: {summary['rg1a_final_reader_context_overwrite_ratio']:.4f}",
        "",
        "## RG-1B CTX-OFF H4-K4",
        f"- meaningful_movement: {summary['rg1b_meaningful_movement']}",
        f"- entropy_delta: {summary['rg1b_entropy_delta']:.4f}",
        f"- pairwise_delta: {summary['rg1b_pairwise_delta']:.4f}",
        f"- short_rank_delta: {summary['rg1b_short_rank_delta']:.4f}",
        f"- final_reader_context_overwrite_ratio: {summary['rg1b_final_reader_context_overwrite_ratio']:.4f}",
        "",
        "## RG-1C CTX-OFF H4-K4 Linear",
        f"- meaningful_movement: {summary['rg1c_meaningful_movement']}",
        f"- entropy_delta: {summary['rg1c_entropy_delta']:.4f}",
        f"- pairwise_delta: {summary['rg1c_pairwise_delta']:.4f}",
        f"- short_rank_delta: {summary['rg1c_short_rank_delta']:.4f}",
        f"- final_reader_context_overwrite_ratio: {summary['rg1c_final_reader_context_overwrite_ratio']:.4f}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
