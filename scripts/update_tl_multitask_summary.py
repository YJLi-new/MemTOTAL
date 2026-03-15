#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _task_summary(control_metrics: dict[str, Any], bridge_metrics: dict[str, Any]) -> dict[str, Any]:
    control_score = float(control_metrics.get("best_adapt_task_score", 0.0))
    bridge_score = float(bridge_metrics.get("best_adapt_task_score", 0.0))
    control_exact_match = float(control_metrics.get("best_adapt_exact_match", control_score))
    bridge_exact_match = float(bridge_metrics.get("best_adapt_exact_match", bridge_score))
    score_delta = bridge_score - control_score
    exact_match_delta = bridge_exact_match - control_exact_match
    positive_signal = score_delta > 0.0 or exact_match_delta > 0.0
    return {
        "task_name": str(bridge_metrics.get("task_name", control_metrics.get("task_name", ""))),
        "benchmark_id": str(bridge_metrics.get("benchmark_id", control_metrics.get("benchmark_id", ""))),
        "metric_name": str(bridge_metrics.get("task_metric_name", control_metrics.get("task_metric_name", "accuracy"))),
        "control_task_score": control_score,
        "bridge_task_score": bridge_score,
        "task_score_delta": score_delta,
        "control_exact_match": control_exact_match,
        "bridge_exact_match": bridge_exact_match,
        "exact_match_delta": exact_match_delta,
        "control_dominant_label_fraction": float(control_metrics.get("dominant_label_fraction", 0.0)),
        "bridge_dominant_label_fraction": float(bridge_metrics.get("dominant_label_fraction", 0.0)),
        "bridge_receiver_lora_enabled": bool(bridge_metrics.get("pilot_receiver_lora_enabled", False)),
        "bridge_receiver_lora_trainable_params": int(bridge_metrics.get("pilot_receiver_lora_trainable_params", 0)),
        "bridge_train_grad_norm_reader_steps_1_4_median": float(
            bridge_metrics.get("train_grad_norm_reader_steps_1_4_median", 0.0)
        ),
        "bridge_train_grad_norm_fuser_steps_1_4_median": float(
            bridge_metrics.get("train_grad_norm_fuser_steps_1_4_median", 0.0)
        ),
        "bridge_train_grad_norm_receiver_lora_steps_1_4_median": float(
            bridge_metrics.get("train_grad_norm_receiver_lora_steps_1_4_median", 0.0)
        ),
        "bridge_memory_long_top1_top2_ratio": float(bridge_metrics.get("memory_long_top1_top2_ratio", 0.0)),
        "bridge_reader_readout_effective_rank": float(bridge_metrics.get("reader_readout_effective_rank", 0.0)),
        "bridge_reader_readout_pairwise_cosine_mean": float(
            bridge_metrics.get("reader_readout_pairwise_cosine_mean", 0.0)
        ),
        "positive_signal": bool(positive_signal),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv3 V4 multitask bridge runs.")
    parser.add_argument("--fever_summary_json", required=True)
    parser.add_argument("--narrativeqa_control_metrics_json", required=True)
    parser.add_argument("--narrativeqa_bridge_metrics_json", required=True)
    parser.add_argument("--gsm8k_control_metrics_json", required=True)
    parser.add_argument("--gsm8k_bridge_metrics_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    fever_summary = _load_json(args.fever_summary_json)
    narrativeqa = _task_summary(
        _load_json(args.narrativeqa_control_metrics_json),
        _load_json(args.narrativeqa_bridge_metrics_json),
    )
    gsm8k = _task_summary(
        _load_json(args.gsm8k_control_metrics_json),
        _load_json(args.gsm8k_bridge_metrics_json),
    )

    fever_improved = bool(fever_summary.get("move_to_v3", False))
    any_nonfever_positive = bool(narrativeqa["positive_signal"] or gsm8k["positive_signal"])
    both_nonfever_positive = bool(narrativeqa["positive_signal"] and gsm8k["positive_signal"])
    weak_success = bool(fever_improved and any_nonfever_positive)
    medium_success = bool(fever_improved and both_nonfever_positive)
    strong_success = False

    if strong_success:
        comparison_conclusion = "strong_success"
        primary_interpretation = "multitask_bridge_alive"
    elif medium_success:
        comparison_conclusion = "medium_success"
        primary_interpretation = "multitask_directional_evidence"
    elif weak_success:
        comparison_conclusion = "weak_success"
        primary_interpretation = "fever_plus_single_task_signal"
    elif any_nonfever_positive:
        comparison_conclusion = "nonfever_only_signal"
        primary_interpretation = "task_sensitive_nonfever_signal"
    else:
        comparison_conclusion = "failure"
        primary_interpretation = "no_multitask_signal"

    summary = {
        "comparison_conclusion": comparison_conclusion,
        "primary_interpretation": primary_interpretation,
        "recommended_family": "micro_lora_r2_late3",
        "fever_reference_conclusion": str(fever_summary.get("comparison_conclusion", "")),
        "fever_reference_move_to_v4": bool(fever_summary.get("move_to_v4", False)),
        "fever_reference_task_score": float(fever_summary.get("late3_best_adapt_task_score", 0.0)),
        "narrativeqa": narrativeqa,
        "gsm8k": gsm8k,
        "any_nonfever_positive": any_nonfever_positive,
        "both_nonfever_positive": both_nonfever_positive,
        "weak_success": weak_success,
        "medium_success": medium_success,
        "strong_success": strong_success,
        "move_to_v5": medium_success,
        "move_to_v6": False,
        "stop_after_v4": not medium_success,
    }

    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# TL Bridge Multitask V4 Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- primary_interpretation: {summary['primary_interpretation']}",
        f"- recommended_family: {summary['recommended_family']}",
        f"- weak_success: {summary['weak_success']}",
        f"- medium_success: {summary['medium_success']}",
        f"- strong_success: {summary['strong_success']}",
        f"- move_to_v5: {summary['move_to_v5']}",
        f"- stop_after_v4: {summary['stop_after_v4']}",
        "",
        "## NarrativeQA",
        f"- control_task_score: {narrativeqa['control_task_score']:.4f}",
        f"- bridge_task_score: {narrativeqa['bridge_task_score']:.4f}",
        f"- task_score_delta: {narrativeqa['task_score_delta']:.4f}",
        f"- control_exact_match: {narrativeqa['control_exact_match']:.4f}",
        f"- bridge_exact_match: {narrativeqa['bridge_exact_match']:.4f}",
        f"- exact_match_delta: {narrativeqa['exact_match_delta']:.4f}",
        f"- positive_signal: {narrativeqa['positive_signal']}",
        "",
        "## GSM8K",
        f"- control_task_score: {gsm8k['control_task_score']:.4f}",
        f"- bridge_task_score: {gsm8k['bridge_task_score']:.4f}",
        f"- task_score_delta: {gsm8k['task_score_delta']:.4f}",
        f"- control_exact_match: {gsm8k['control_exact_match']:.4f}",
        f"- bridge_exact_match: {gsm8k['bridge_exact_match']:.4f}",
        f"- exact_match_delta: {gsm8k['exact_match_delta']:.4f}",
        f"- positive_signal: {gsm8k['positive_signal']}",
    ]
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
