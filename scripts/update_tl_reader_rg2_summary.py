#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_tl_reader_rg2_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RG-2 TL reader rescue runs.")
    parser.add_argument("--control_summary_json", required=True)
    parser.add_argument("--competitive_summary_json", required=True)
    parser.add_argument("--partition_summary_json", required=True)
    parser.add_argument("--control_train_events_json")
    parser.add_argument("--competitive_train_events_json")
    parser.add_argument("--partition_train_events_json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_tl_reader_rg2_runs(
        control_summary_json=args.control_summary_json,
        competitive_summary_json=args.competitive_summary_json,
        partition_summary_json=args.partition_summary_json,
        control_train_events_json=args.control_train_events_json,
        competitive_train_events_json=args.competitive_train_events_json,
        partition_train_events_json=args.partition_train_events_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# TL Reader Geometry RG-2 Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- failure_reason: {summary['failure_reason']}",
        f"- primary_interpretation: {summary['primary_interpretation']}",
        f"- recommended_arm: {summary['recommended_arm']}",
        f"- move_to_rg3: {summary['move_to_rg3']}",
        f"- geometry_alive: {summary['geometry_alive']}",
        f"- competitive_reader_supported: {summary['competitive_reader_supported']}",
        f"- partition_reader_supported: {summary['partition_reader_supported']}",
        f"- bridge_failure_submode: {summary['bridge_failure_submode']}",
        "",
        "## Control",
        f"- geometry_alive: {summary['control_geometry_alive']}",
        f"- best_reader_attention_pairwise_cosine_mean: {summary['control_best_reader_attention_pairwise_cosine_mean']:.4f}",
        f"- best_reader_attention_entropy_mean: {summary['control_best_reader_attention_entropy_mean']:.4f}",
        f"- best_memory_short_effective_rank: {summary['control_best_memory_short_effective_rank']:.4f}",
        "",
        "## Competitive",
        f"- geometry_alive: {summary['competitive_geometry_alive']}",
        f"- geometry_alive_step: {summary['competitive_geometry_alive_step']}",
        f"- entropy_delta: {summary['competitive_entropy_delta']:.4f}",
        f"- pairwise_delta: {summary['competitive_pairwise_delta']:.4f}",
        f"- short_rank_delta: {summary['competitive_short_rank_delta']:.4f}",
        f"- best_memory_short_effective_rank: {summary['competitive_best_memory_short_effective_rank']:.4f}",
        "",
        "## Partition",
        f"- geometry_alive: {summary['partition_geometry_alive']}",
        f"- geometry_alive_step: {summary['partition_geometry_alive_step']}",
        f"- entropy_delta: {summary['partition_entropy_delta']:.4f}",
        f"- pairwise_delta: {summary['partition_pairwise_delta']:.4f}",
        f"- short_rank_delta: {summary['partition_short_rank_delta']:.4f}",
        f"- best_memory_short_effective_rank: {summary['partition_best_memory_short_effective_rank']:.4f}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
