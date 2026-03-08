#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.analysis.m4_shared_injection import summarize_m4_support_bank_run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize an M4 support-bank run after canonical/heldout gates.")
    parser.add_argument("--selection_json", required=True)
    parser.add_argument("--run_metrics_json", required=True)
    parser.add_argument("--dynamics_summary_csv", required=True)
    parser.add_argument("--prefix_norm_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--screen248_test_metrics")
    parser.add_argument("--heldout_a_metrics")
    parser.add_argument("--heldout_b_metrics")
    parser.add_argument("--fixed64_metrics")
    parser.add_argument("--overwrite-selection", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    heldout_metrics = {}
    if args.heldout_a_metrics:
        heldout_metrics["heldout_a"] = args.heldout_a_metrics
    if args.heldout_b_metrics:
        heldout_metrics["heldout_b"] = args.heldout_b_metrics
    summary = summarize_m4_support_bank_run(
        selection_json=args.selection_json,
        run_metrics_json=args.run_metrics_json,
        dynamics_summary_csv=args.dynamics_summary_csv,
        prefix_norm_csv=args.prefix_norm_csv,
        screen248_test_metrics_json=args.screen248_test_metrics,
        heldout_metrics_by_name=heldout_metrics,
        fixed64_metrics_json=args.fixed64_metrics,
    )
    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# M4 Run Summary",
        "",
        f"- selection_passed: {summary['selection_passed']}",
        f"- selected_step: {summary['selected_step']}",
        f"- screen248_test_gate_passed: {summary['screen248_test_gate_passed']}",
        f"- heldout_sane_bank_count: {summary['heldout_sane_bank_count']}",
        f"- support_bank_brittle: {summary['support_bank_brittle']}",
        f"- fixed64_report_generated: {summary['fixed64_report_generated']}",
        f"- fixed64_gate_passed: {summary['fixed64_gate_passed']}",
        f"- milestone_gate_passed: {summary['milestone_gate_passed']}",
        f"- cap_saturation_onset_step: {summary['cap_saturation_onset_step']}",
        f"- dominant_label_collapse_onset_step: {summary['dominant_label_collapse_onset_step']}",
    ]
    if summary["heldout_results"]:
        report_lines.extend(["", "## Heldout Banks"])
        for name, result in sorted(summary["heldout_results"].items()):
            report_lines.append(
                f"- {name}: sane={result['sane']}, gate_passed={result['gate_passed']}, "
                f"regressions_vs_base={result['regressions_vs_base']}, "
                f"flip_gain_vs_shuffle={result['flip_gain_vs_shuffle']}, "
                f"flip_gain_vs_zero={result['flip_gain_vs_zero']}"
            )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    if args.overwrite_selection:
        selection_path = Path(args.selection_json)
        selection = json.loads(selection_path.read_text())
        for key in (
            "screen248_test_gate_passed",
            "support_bank_brittle",
            "heldout_sane_bank_count",
            "fixed64_report_generated",
            "fixed64_gate_passed",
            "milestone_gate_passed",
            "cap_saturation_onset_step",
            "dominant_label_collapse_onset_step",
        ):
            selection[key] = summary[key]
        selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
