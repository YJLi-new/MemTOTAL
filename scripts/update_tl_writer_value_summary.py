#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_tl_writer_value_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv3 V1 writer-value FEVER runs.")
    parser.add_argument("--control_metrics_json", required=True)
    parser.add_argument("--shared_scaled_metrics_json", required=True)
    parser.add_argument("--slot_query_only_metrics_json", required=True)
    parser.add_argument("--control_train_events_json")
    parser.add_argument("--shared_scaled_train_events_json")
    parser.add_argument("--slot_query_only_train_events_json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_tl_writer_value_runs(
        control_metrics_json=args.control_metrics_json,
        shared_scaled_metrics_json=args.shared_scaled_metrics_json,
        slot_query_only_metrics_json=args.slot_query_only_metrics_json,
        control_train_events_json=args.control_train_events_json,
        shared_scaled_train_events_json=args.shared_scaled_train_events_json,
        slot_query_only_train_events_json=args.slot_query_only_train_events_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# TL Writer Value V1 Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- primary_interpretation: {summary['primary_interpretation']}",
        f"- recommended_arm: {summary['recommended_arm']}",
        f"- move_to_v1_penalties: {summary['move_to_v1_penalties']}",
        f"- move_to_v2: {summary['move_to_v2']}",
        f"- stop_after_v1_architecture: {summary['stop_after_v1_architecture']}",
        f"- failure_reason: {summary['failure_reason']}",
        "",
        "## Control",
        f"- writer_mode: {summary['control_writer_mode']}",
        f"- top1_top2_ratio: {summary['control_memory_long_top1_top2_ratio']:.4f}",
        f"- common_mode_energy_ratio: {summary['control_memory_long_common_mode_energy_ratio']:.4f}",
        f"- reader_readout_effective_rank: {summary['control_reader_readout_effective_rank']:.4f}",
        f"- reader_readout_pairwise_cosine_mean: {summary['control_reader_readout_pairwise_cosine_mean']:.4f}",
        "",
        "## Shared Add Scaled",
        f"- writer_mode: {summary['shared_scaled_writer_mode']}",
        f"- shared_state_scale: {summary['shared_scaled_shared_state_scale']:.4f}",
        f"- top1_top2_ratio: {summary['shared_scaled_memory_long_top1_top2_ratio']:.4f}",
        f"- top1_top2_reduction_factor: {summary['shared_scaled_top1_top2_reduction_factor']:.4f}",
        f"- readout_effective_rank: {summary['shared_scaled_reader_readout_effective_rank']:.4f}",
        f"- readout_pairwise_cosine_mean: {summary['shared_scaled_reader_readout_pairwise_cosine_mean']:.4f}",
        f"- collapse_delayed: {summary['shared_scaled_collapse_delayed']}",
        f"- medium_success: {summary['shared_scaled_medium_success']}",
        "",
        "## Slot Query Only",
        f"- writer_mode: {summary['slot_query_only_writer_mode']}",
        f"- top1_top2_ratio: {summary['slot_query_only_memory_long_top1_top2_ratio']:.4f}",
        f"- top1_top2_reduction_factor: {summary['slot_query_only_top1_top2_reduction_factor']:.4f}",
        f"- readout_effective_rank: {summary['slot_query_only_reader_readout_effective_rank']:.4f}",
        f"- readout_pairwise_cosine_mean: {summary['slot_query_only_reader_readout_pairwise_cosine_mean']:.4f}",
        f"- collapse_delayed: {summary['slot_query_only_collapse_delayed']}",
        f"- medium_success: {summary['slot_query_only_medium_success']}",
        f"- strong_success: {summary['slot_query_only_strong_success']}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
