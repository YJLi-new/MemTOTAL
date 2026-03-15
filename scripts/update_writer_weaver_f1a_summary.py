#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _task_summary(
    *,
    control_metrics: dict[str, Any],
    w0_metrics: dict[str, Any],
    f1a_metrics: dict[str, Any],
) -> dict[str, Any]:
    control_score = _as_float(control_metrics, "best_adapt_task_score")
    w0_score = _as_float(w0_metrics, "best_adapt_task_score")
    f1a_score = _as_float(f1a_metrics, "best_adapt_task_score")
    control_exact_match = _as_float(control_metrics, "best_adapt_exact_match", control_score)
    w0_exact_match = _as_float(w0_metrics, "best_adapt_exact_match", w0_score)
    f1a_exact_match = _as_float(f1a_metrics, "best_adapt_exact_match", f1a_score)
    w0_delta = _as_float(w0_metrics, "delta_answer_logprob")
    f1a_delta = _as_float(f1a_metrics, "delta_answer_logprob")
    w0_common_mode = _as_float(w0_metrics, "memory_long_common_mode_energy_ratio")
    f1a_common_mode = _as_float(f1a_metrics, "memory_long_common_mode_energy_ratio")
    w0_top1_top2 = _as_float(w0_metrics, "memory_long_top1_top2_ratio")
    f1a_top1_top2 = _as_float(f1a_metrics, "memory_long_top1_top2_ratio")
    w0_centered_rank = _as_float(w0_metrics, "memory_long_centered_effective_rank")
    f1a_centered_rank = _as_float(f1a_metrics, "memory_long_centered_effective_rank")
    beneficial_geometry_directions = sum(
        [
            f1a_common_mode <= (w0_common_mode - 1e-4),
            f1a_top1_top2 <= (w0_top1_top2 - 5.0),
            f1a_centered_rank >= (w0_centered_rank + 0.5),
        ]
    )
    harmful_geometry_directions = sum(
        [
            f1a_common_mode >= (w0_common_mode + 1e-4),
            f1a_top1_top2 >= (w0_top1_top2 + 5.0),
            f1a_centered_rank <= (w0_centered_rank - 0.5),
        ]
    )
    geometry_improved_vs_w0 = bool(beneficial_geometry_directions >= 2)
    geometry_regressed_vs_w0 = bool(harmful_geometry_directions >= 2)
    meets_weak_geometry_thresholds = bool(
        f1a_common_mode <= 0.97 and f1a_top1_top2 <= 30.0 and f1a_centered_rank >= 3.0
    )
    meets_medium_geometry_thresholds = bool(
        f1a_common_mode <= 0.93 and f1a_top1_top2 <= 15.0 and f1a_centered_rank >= 4.0
    )
    positive_task_gain_vs_control = bool(
        (f1a_score > control_score) or (f1a_exact_match > control_exact_match)
    )
    return {
        "task_name": str(
            f1a_metrics.get(
                "task_name",
                w0_metrics.get("task_name", control_metrics.get("task_name", "")),
            )
        ),
        "benchmark_id": str(
            f1a_metrics.get(
                "benchmark_id",
                w0_metrics.get("benchmark_id", control_metrics.get("benchmark_id", "")),
            )
        ),
        "metric_name": str(
            f1a_metrics.get(
                "task_metric_name",
                w0_metrics.get("task_metric_name", control_metrics.get("task_metric_name", "accuracy")),
            )
        ),
        "control_task_score": control_score,
        "w0_task_score": w0_score,
        "f1a_task_score": f1a_score,
        "f1a_task_score_delta_vs_control": f1a_score - control_score,
        "f1a_task_score_delta_vs_w0": f1a_score - w0_score,
        "control_exact_match": control_exact_match,
        "w0_exact_match": w0_exact_match,
        "f1a_exact_match": f1a_exact_match,
        "f1a_exact_match_delta_vs_control": f1a_exact_match - control_exact_match,
        "f1a_exact_match_delta_vs_w0": f1a_exact_match - w0_exact_match,
        "w0_delta_answer_logprob": w0_delta,
        "f1a_delta_answer_logprob": f1a_delta,
        "delta_answer_logprob_gain_vs_w0": f1a_delta - w0_delta,
        "w0_common_mode_energy_ratio": w0_common_mode,
        "f1a_common_mode_energy_ratio": f1a_common_mode,
        "w0_top1_top2_ratio": w0_top1_top2,
        "f1a_top1_top2_ratio": f1a_top1_top2,
        "w0_centered_effective_rank": w0_centered_rank,
        "f1a_centered_effective_rank": f1a_centered_rank,
        "geometry_improved_vs_w0": geometry_improved_vs_w0,
        "meets_weak_geometry_thresholds": meets_weak_geometry_thresholds,
        "meets_medium_geometry_thresholds": meets_medium_geometry_thresholds,
        "positive_delta_answer_logprob": bool(f1a_delta > 0.0),
        "positive_task_gain_vs_control": positive_task_gain_vs_control,
        "f1a_writer_conditioning_layers": int(f1a_metrics.get("pilot_writer_conditioning_layers", 1)),
        "f1a_projected_memory_effective_rank": _as_float(f1a_metrics, "projected_memory_effective_rank"),
        "f1a_writer_token_pairwise_cosine_mean": _as_float(f1a_metrics, "memory_long_pairwise_cosine_mean"),
        "beneficial_geometry_directions": beneficial_geometry_directions,
        "harmful_geometry_directions": harmful_geometry_directions,
        "geometry_regressed_vs_w0": geometry_regressed_vs_w0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv4 F1a writer-capacity runs.")
    parser.add_argument("--w0_summary_json", required=True)
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        parser.add_argument(f"--{task_name}_control_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_w0_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_f1a_metrics_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    w0_summary = _load_json(args.w0_summary_json)
    task_summaries = {
        task_name: _task_summary(
            control_metrics=_load_json(getattr(args, f"{task_name}_control_metrics_json")),
            w0_metrics=_load_json(getattr(args, f"{task_name}_w0_metrics_json")),
            f1a_metrics=_load_json(getattr(args, f"{task_name}_f1a_metrics_json")),
        )
        for task_name in ("gsm8k", "narrativeqa", "fever")
    }
    nonfever_tasks = [task_summaries["gsm8k"], task_summaries["narrativeqa"]]
    weak_success_any = any(
        bool(task["meets_weak_geometry_thresholds"] and task["positive_delta_answer_logprob"])
        for task in nonfever_tasks
    )
    medium_success_any = any(
        bool(task["meets_medium_geometry_thresholds"] and task["positive_task_gain_vs_control"])
        for task in nonfever_tasks
    )
    geometry_improvement_any = any(bool(task["geometry_improved_vs_w0"]) for task in nonfever_tasks)
    geometry_regression_any = any(bool(task["geometry_regressed_vs_w0"]) for task in nonfever_tasks)
    if medium_success_any:
        comparison_conclusion = "move_to_w2"
        primary_interpretation = "writer_capacity_medium_success"
    elif weak_success_any:
        comparison_conclusion = "move_to_w2"
        primary_interpretation = "writer_capacity_weak_success"
    elif geometry_improvement_any:
        comparison_conclusion = "move_to_f1b"
        primary_interpretation = "writer_capacity_changed_geometry_without_usefulness"
    else:
        comparison_conclusion = "move_to_f1b"
        primary_interpretation = (
            "bigger_writer_regressed_geometry"
            if geometry_regression_any
            else "bigger_writer_still_flat"
        )
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "primary_interpretation": primary_interpretation,
        "w0_reference_conclusion": str(w0_summary.get("comparison_conclusion", "")),
        "weak_success_any": weak_success_any,
        "medium_success_any": medium_success_any,
        "geometry_improvement_any": geometry_improvement_any,
        "geometry_regression_any": geometry_regression_any,
        "move_to_w2": bool(weak_success_any or medium_success_any),
        "move_to_f1b": bool(not (weak_success_any or medium_success_any)),
        "stop_after_f1a": False,
        "gsm8k": task_summaries["gsm8k"],
        "narrativeqa": task_summaries["narrativeqa"],
        "fever": task_summaries["fever"],
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Writer-Weaver F1a Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- primary_interpretation: {summary['primary_interpretation']}",
        f"- w0_reference_conclusion: {summary['w0_reference_conclusion']}",
        f"- weak_success_any: {summary['weak_success_any']}",
        f"- medium_success_any: {summary['medium_success_any']}",
        f"- geometry_improvement_any: {summary['geometry_improvement_any']}",
        f"- geometry_regression_any: {summary['geometry_regression_any']}",
        f"- move_to_w2: {summary['move_to_w2']}",
        f"- move_to_f1b: {summary['move_to_f1b']}",
    ]
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        task = task_summaries[task_name]
        report_lines.extend(
            [
                "",
                f"## {task_name}",
                f"- w0_task_score: {task['w0_task_score']:.4f}",
                f"- f1a_task_score: {task['f1a_task_score']:.4f}",
                f"- f1a_delta_answer_logprob: {task['f1a_delta_answer_logprob']:.4f}",
                f"- f1a_common_mode_energy_ratio: {task['f1a_common_mode_energy_ratio']:.6f}",
                f"- f1a_top1_top2_ratio: {task['f1a_top1_top2_ratio']:.4f}",
                f"- f1a_centered_effective_rank: {task['f1a_centered_effective_rank']:.4f}",
                f"- geometry_improved_vs_w0: {task['geometry_improved_vs_w0']}",
                f"- geometry_regressed_vs_w0: {task['geometry_regressed_vs_w0']}",
                f"- meets_weak_geometry_thresholds: {task['meets_weak_geometry_thresholds']}",
                f"- positive_delta_answer_logprob: {task['positive_delta_answer_logprob']}",
            ]
        )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
