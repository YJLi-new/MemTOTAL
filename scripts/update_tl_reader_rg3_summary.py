#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import compare_tl_reader_rg3_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RG-3 TL reader local bootstrap runs.")
    parser.add_argument("--control_summary_json", required=True)
    parser.add_argument("--bootstrap_summary_json", required=True)
    parser.add_argument("--bootstrap_reconstruction_summary_json", required=True)
    parser.add_argument("--control_train_events_json")
    parser.add_argument("--bootstrap_train_events_json")
    parser.add_argument("--bootstrap_reconstruction_train_events_json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = compare_tl_reader_rg3_runs(
        control_summary_json=args.control_summary_json,
        bootstrap_summary_json=args.bootstrap_summary_json,
        bootstrap_reconstruction_summary_json=args.bootstrap_reconstruction_summary_json,
        control_train_events_json=args.control_train_events_json,
        bootstrap_train_events_json=args.bootstrap_train_events_json,
        bootstrap_reconstruction_train_events_json=args.bootstrap_reconstruction_train_events_json,
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# TL Reader Geometry RG-3 Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- primary_interpretation: {summary['primary_interpretation']}",
        f"- recommended_arm: {summary['recommended_arm']}",
        f"- geometry_alive: {summary['geometry_alive']}",
        f"- move_to_rg4: {summary['move_to_rg4']}",
        f"- final_classification: {summary['final_classification']}",
        f"- stop_after_rg3: {summary['stop_after_rg3']}",
        f"- failure_reason: {summary['failure_reason']}",
        "",
        "## Control",
        f"- geometry_alive: {summary['control_geometry_alive']}",
        f"- dominant_label_collapse_onset_step: {summary['control_dominant_label_collapse_onset_step']}",
        f"- best_memory_short_effective_rank: {summary['control_best_memory_short_effective_rank']:.4f}",
        f"- best_reader_readout_effective_rank: {summary['control_best_reader_readout_effective_rank']:.4f}",
        "",
        "## Bootstrap",
        f"- geometry_alive: {summary['bootstrap_geometry_alive']}",
        f"- dominant_label_collapse_onset_step: {summary['bootstrap_dominant_label_collapse_onset_step']}",
        f"- collapse_delta: {summary['bootstrap_collapse_delta']:.4f}",
        f"- readout_rank_delta: {summary['bootstrap_readout_rank_delta']:.4f}",
        f"- short_rank_delta: {summary['bootstrap_short_rank_delta']:.4f}",
        "",
        "## Bootstrap + Reconstruction",
        f"- geometry_alive: {summary['bootstrap_reconstruction_geometry_alive']}",
        f"- dominant_label_collapse_onset_step: {summary['bootstrap_reconstruction_dominant_label_collapse_onset_step']}",
        f"- collapse_delta: {summary['bootstrap_reconstruction_collapse_delta']:.4f}",
        f"- readout_rank_delta: {summary['bootstrap_reconstruction_readout_rank_delta']:.4f}",
        f"- short_rank_delta: {summary['bootstrap_reconstruction_short_rank_delta']:.4f}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
