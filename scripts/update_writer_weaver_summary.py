#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _as_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _task_summary(
    *,
    control_metrics: dict[str, Any],
    support_only_metrics: dict[str, Any],
    support_context_metrics: dict[str, Any],
) -> dict[str, Any]:
    control_score = _as_float(control_metrics, "best_adapt_task_score")
    support_only_score = _as_float(support_only_metrics, "best_adapt_task_score")
    support_context_score = _as_float(support_context_metrics, "best_adapt_task_score")
    control_exact_match = _as_float(control_metrics, "best_adapt_exact_match", control_score)
    support_only_exact_match = _as_float(
        support_only_metrics,
        "best_adapt_exact_match",
        support_only_score,
    )
    support_context_exact_match = _as_float(
        support_context_metrics,
        "best_adapt_exact_match",
        support_context_score,
    )
    support_only_delta_answer_logprob = _as_float(support_only_metrics, "delta_answer_logprob")
    support_context_delta_answer_logprob = _as_float(support_context_metrics, "delta_answer_logprob")
    common_mode_delta = (
        _as_float(support_context_metrics, "memory_long_common_mode_energy_ratio")
        - _as_float(support_only_metrics, "memory_long_common_mode_energy_ratio")
    )
    centered_rank_delta = (
        _as_float(support_context_metrics, "memory_long_centered_effective_rank")
        - _as_float(support_only_metrics, "memory_long_centered_effective_rank")
    )
    top1_top2_delta = (
        _as_float(support_context_metrics, "memory_long_top1_top2_ratio")
        - _as_float(support_only_metrics, "memory_long_top1_top2_ratio")
    )
    geometry_move_detected = bool(
        abs(common_mode_delta) >= 0.01
        or abs(centered_rank_delta) >= 0.1
        or abs(top1_top2_delta) >= 1.0
    )
    return {
        "task_name": str(
            support_context_metrics.get(
                "task_name",
                support_only_metrics.get("task_name", control_metrics.get("task_name", "")),
            )
        ),
        "benchmark_id": str(
            support_context_metrics.get(
                "benchmark_id",
                support_only_metrics.get("benchmark_id", control_metrics.get("benchmark_id", "")),
            )
        ),
        "metric_name": str(
            support_context_metrics.get(
                "task_metric_name",
                support_only_metrics.get("task_metric_name", control_metrics.get("task_metric_name", "accuracy")),
            )
        ),
        "control_task_score": control_score,
        "support_only_task_score": support_only_score,
        "support_context_task_score": support_context_score,
        "support_only_task_score_delta_vs_control": support_only_score - control_score,
        "support_context_task_score_delta_vs_control": support_context_score - control_score,
        "control_exact_match": control_exact_match,
        "support_only_exact_match": support_only_exact_match,
        "support_context_exact_match": support_context_exact_match,
        "support_only_exact_match_delta_vs_control": support_only_exact_match - control_exact_match,
        "support_context_exact_match_delta_vs_control": support_context_exact_match - control_exact_match,
        "support_only_delta_answer_logprob": support_only_delta_answer_logprob,
        "support_context_delta_answer_logprob": support_context_delta_answer_logprob,
        "delta_answer_logprob_gain_over_support_only": (
            support_context_delta_answer_logprob - support_only_delta_answer_logprob
        ),
        "support_only_context_free": _as_float(support_only_metrics, "writer_context_token_count") <= 0.0,
        "support_context_wired": _as_float(support_context_metrics, "writer_context_token_count") > 0.0,
        "delta_answer_logprob_finite": bool(
            _finite(support_only_delta_answer_logprob) and _finite(support_context_delta_answer_logprob)
        ),
        "support_only_common_mode_energy_ratio": _as_float(
            support_only_metrics,
            "memory_long_common_mode_energy_ratio",
        ),
        "support_context_common_mode_energy_ratio": _as_float(
            support_context_metrics,
            "memory_long_common_mode_energy_ratio",
        ),
        "support_only_centered_effective_rank": _as_float(
            support_only_metrics,
            "memory_long_centered_effective_rank",
        ),
        "support_context_centered_effective_rank": _as_float(
            support_context_metrics,
            "memory_long_centered_effective_rank",
        ),
        "support_only_top1_top2_ratio": _as_float(
            support_only_metrics,
            "memory_long_top1_top2_ratio",
        ),
        "support_context_top1_top2_ratio": _as_float(
            support_context_metrics,
            "memory_long_top1_top2_ratio",
        ),
        "support_context_projected_memory_effective_rank": _as_float(
            support_context_metrics,
            "projected_memory_effective_rank",
        ),
        "support_context_writer_token_pairwise_cosine_mean": _as_float(
            support_context_metrics,
            "memory_long_pairwise_cosine_mean",
        ),
        "common_mode_delta_context_vs_support_only": common_mode_delta,
        "centered_rank_delta_context_vs_support_only": centered_rank_delta,
        "top1_top2_delta_context_vs_support_only": top1_top2_delta,
        "geometry_move_detected": geometry_move_detected,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv4 W0 writer-weaver smoke runs.")
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        parser.add_argument(f"--{task_name}_control_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_support_only_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_support_context_metrics_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    task_summaries = {
        task_name: _task_summary(
            control_metrics=_load_json(getattr(args, f"{task_name}_control_metrics_json")),
            support_only_metrics=_load_json(getattr(args, f"{task_name}_support_only_metrics_json")),
            support_context_metrics=_load_json(getattr(args, f"{task_name}_support_context_metrics_json")),
        )
        for task_name in ("gsm8k", "narrativeqa", "fever")
    }
    ordered_tasks = [task_summaries["gsm8k"], task_summaries["narrativeqa"], task_summaries["fever"]]
    nonfever_tasks = [task for task in ordered_tasks if task["benchmark_id"] != "fever"]
    all_context_wired = all(bool(task["support_context_wired"]) for task in ordered_tasks)
    all_support_only_context_free = all(bool(task["support_only_context_free"]) for task in ordered_tasks)
    all_delta_finite = all(bool(task["delta_answer_logprob_finite"]) for task in ordered_tasks)
    geometry_move_any = any(bool(task["geometry_move_detected"]) for task in nonfever_tasks)
    support_context_beats_support_only_any = any(
        float(task["delta_answer_logprob_gain_over_support_only"]) > 0.0 for task in nonfever_tasks
    )
    nonfever_positive_delta_any = any(
        float(task["support_context_delta_answer_logprob"]) > 0.0 for task in nonfever_tasks
    )
    move_to_w1 = bool(
        all_context_wired
        and all_support_only_context_free
        and all_delta_finite
        and geometry_move_any
        and support_context_beats_support_only_any
        and nonfever_positive_delta_any
    )
    if move_to_w1:
        comparison_conclusion = "move_to_w1"
        primary_interpretation = "support_context_directional_signal"
    elif all_context_wired and all_support_only_context_free and all_delta_finite:
        comparison_conclusion = "plumbing_only"
        primary_interpretation = "writer_direct_wired_but_no_directional_signal"
    else:
        comparison_conclusion = "failure"
        primary_interpretation = "writer_direct_scaffold_incomplete_or_unstable"
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "primary_interpretation": primary_interpretation,
        "all_tasks_completed": True,
        "all_context_wired": all_context_wired,
        "all_support_context_wired": all_context_wired,
        "all_support_only_context_free": all_support_only_context_free,
        "all_delta_finite": all_delta_finite,
        "geometry_move_any": geometry_move_any,
        "support_context_beats_support_only_any": support_context_beats_support_only_any,
        "nonfever_positive_delta_any": nonfever_positive_delta_any,
        "move_to_w1": move_to_w1,
        "stop_after_w0": not move_to_w1,
        "gsm8k": task_summaries["gsm8k"],
        "narrativeqa": task_summaries["narrativeqa"],
        "fever": task_summaries["fever"],
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Writer-Weaver W0 Summary",
        "",
        f"- comparison_conclusion: {comparison_conclusion}",
        f"- primary_interpretation: {primary_interpretation}",
        f"- all_context_wired: {all_context_wired}",
        f"- all_support_only_context_free: {all_support_only_context_free}",
        f"- all_delta_finite: {all_delta_finite}",
        f"- geometry_move_any: {geometry_move_any}",
        f"- support_context_beats_support_only_any: {support_context_beats_support_only_any}",
        f"- nonfever_positive_delta_any: {nonfever_positive_delta_any}",
        f"- move_to_w1: {move_to_w1}",
    ]
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        task = task_summaries[task_name]
        report_lines.extend(
            [
                "",
                f"## {task_name}",
                f"- control_task_score: {task['control_task_score']:.4f}",
                f"- support_only_task_score: {task['support_only_task_score']:.4f}",
                f"- support_context_task_score: {task['support_context_task_score']:.4f}",
                f"- support_only_delta_answer_logprob: {task['support_only_delta_answer_logprob']:.4f}",
                f"- support_context_delta_answer_logprob: {task['support_context_delta_answer_logprob']:.4f}",
                f"- delta_gain_over_support_only: {task['delta_answer_logprob_gain_over_support_only']:.4f}",
                f"- support_context_wired: {task['support_context_wired']}",
                f"- geometry_move_detected: {task['geometry_move_detected']}",
            ]
        )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
