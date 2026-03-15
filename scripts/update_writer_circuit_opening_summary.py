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


def _layer_metric(payload: dict[str, Any], key: str) -> dict[str, float]:
    raw = payload.get(key, {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(layer_index): float(value)
        for layer_index, value in raw.items()
    }


def _run_summary(
    *,
    metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    expect_receiver_lora: bool,
) -> dict[str, Any]:
    task_score = _as_float(metrics, "best_adapt_task_score")
    exact_match = _as_float(metrics, "best_adapt_exact_match", task_score)
    control_score = _as_float(control_metrics, "best_adapt_task_score")
    control_exact_match = _as_float(control_metrics, "best_adapt_exact_match", control_score)
    delta_answer_logprob = _as_float(metrics, "delta_answer_logprob")
    source_grad = _as_float(metrics, "train_grad_norm_source_stub_steps_1_4_median")
    receiver_grad = _as_float(metrics, "train_grad_norm_receiver_lora_steps_1_4_median")
    prefix_attention_by_layer = _layer_metric(metrics, "prefix_attention_mass_mean_by_layer")
    nontrivial_layer_count = int(
        sum(1 for value in prefix_attention_by_layer.values() if float(value) > 1e-3)
    )
    receiver_lora_enabled = bool(metrics.get("pilot_receiver_lora_enabled", False))
    circuit_open = bool(
        source_grad > 1e-6
        and nontrivial_layer_count > 0
        and abs(delta_answer_logprob) > 1e-9
        and (not expect_receiver_lora or receiver_grad > 0.0)
    )
    positive_signal = bool(delta_answer_logprob > 0.0)
    non_regressive_task = bool(
        task_score >= (control_score - 1e-6)
        or exact_match >= (control_exact_match - 1e-6)
    )
    return {
        "task_name": str(metrics.get("task_name", control_metrics.get("task_name", ""))),
        "benchmark_id": str(metrics.get("benchmark_id", control_metrics.get("benchmark_id", ""))),
        "task_metric_name": str(metrics.get("task_metric_name", control_metrics.get("task_metric_name", "accuracy"))),
        "task_score": task_score,
        "exact_match": exact_match,
        "task_score_delta_vs_control": task_score - control_score,
        "exact_match_delta_vs_control": exact_match - control_exact_match,
        "delta_answer_logprob": delta_answer_logprob,
        "source_grad_norm_steps_1_4_median": source_grad,
        "receiver_lora_enabled": receiver_lora_enabled,
        "receiver_lora_grad_norm_steps_1_4_median": receiver_grad,
        "prefix_attention_mass_mean": _as_float(metrics, "prefix_attention_mass_mean"),
        "prefix_to_content_attention_ratio_mean": _as_float(
            metrics,
            "prefix_to_content_attention_ratio_mean",
        ),
        "prefix_attention_mass_mean_by_layer": prefix_attention_by_layer,
        "prefix_attention_nontrivial_layer_count": nontrivial_layer_count,
        "projected_memory_effective_rank": _as_float(metrics, "projected_memory_effective_rank"),
        "memory_long_common_mode_energy_ratio": _as_float(
            metrics,
            "memory_long_common_mode_energy_ratio",
        ),
        "circuit_open": circuit_open,
        "positive_signal": positive_signal,
        "non_regressive_task": non_regressive_task,
        "weak_success": bool(circuit_open and positive_signal and non_regressive_task),
    }


def _task_summary(
    *,
    control_metrics: dict[str, Any],
    p1a_metrics: dict[str, Any],
    p2a_metrics: dict[str, Any],
) -> dict[str, Any]:
    control = {
        "task_name": str(
            control_metrics.get("task_name", p1a_metrics.get("task_name", p2a_metrics.get("task_name", "")))
        ),
        "benchmark_id": str(
            control_metrics.get("benchmark_id", p1a_metrics.get("benchmark_id", p2a_metrics.get("benchmark_id", "")))
        ),
        "task_metric_name": str(
            control_metrics.get(
                "task_metric_name",
                p1a_metrics.get("task_metric_name", p2a_metrics.get("task_metric_name", "accuracy")),
            )
        ),
        "task_score": _as_float(control_metrics, "best_adapt_task_score"),
        "exact_match": _as_float(
            control_metrics,
            "best_adapt_exact_match",
            _as_float(control_metrics, "best_adapt_task_score"),
        ),
    }
    p1a = _run_summary(
        metrics=p1a_metrics,
        control_metrics=control_metrics,
        expect_receiver_lora=False,
    )
    p2a = _run_summary(
        metrics=p2a_metrics,
        control_metrics=control_metrics,
        expect_receiver_lora=True,
    )
    return {
        "control": control,
        "p1a": p1a,
        "p2a": p2a,
        "p1a_preferred_over_p2a": bool(
            p1a["weak_success"]
            and (
                not p2a["weak_success"]
                or p1a["delta_answer_logprob"] >= p2a["delta_answer_logprob"]
            )
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv5 P1a/P2a writer-circuit opening runs.")
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        parser.add_argument(f"--{task_name}_control_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_p1a_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_p2a_metrics_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    task_summaries = {
        task_name: _task_summary(
            control_metrics=_load_json(getattr(args, f"{task_name}_control_metrics_json")),
            p1a_metrics=_load_json(getattr(args, f"{task_name}_p1a_metrics_json")),
            p2a_metrics=_load_json(getattr(args, f"{task_name}_p2a_metrics_json")),
        )
        for task_name in ("gsm8k", "narrativeqa", "fever")
    }
    nonfever_tasks = [task_summaries["gsm8k"], task_summaries["narrativeqa"]]
    p1a_circuit_open_any_nonfever = any(bool(task["p1a"]["circuit_open"]) for task in nonfever_tasks)
    p1a_positive_any_nonfever = any(bool(task["p1a"]["positive_signal"]) for task in nonfever_tasks)
    p2a_circuit_open_any_nonfever = any(bool(task["p2a"]["circuit_open"]) for task in nonfever_tasks)
    p2a_positive_any_nonfever = any(bool(task["p2a"]["positive_signal"]) for task in nonfever_tasks)
    p1a_weak_success_any_nonfever = any(bool(task["p1a"]["weak_success"]) for task in nonfever_tasks)
    p2a_weak_success_any_nonfever = any(bool(task["p2a"]["weak_success"]) for task in nonfever_tasks)
    if p1a_circuit_open_any_nonfever and p1a_positive_any_nonfever:
        comparison_conclusion = "move_to_p1b"
        primary_interpretation = "source_stub_deep_prefix_opened_circuit_without_receiver_lora"
        recommended_substrate = "p1a_source_stub_no_lora"
        move_to_p1b = True
        move_to_p2b = False
    elif p2a_circuit_open_any_nonfever and p2a_positive_any_nonfever:
        comparison_conclusion = "move_to_p2b"
        primary_interpretation = "source_stub_deep_prefix_needed_receiver_lora_to_open_circuit"
        recommended_substrate = "p2a_source_stub_receiver_lora_r2"
        move_to_p1b = False
        move_to_p2b = True
    else:
        comparison_conclusion = "stop_after_p2a"
        primary_interpretation = "no_nonfever_circuit_open"
        recommended_substrate = "none"
        move_to_p1b = False
        move_to_p2b = False
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "primary_interpretation": primary_interpretation,
        "recommended_substrate": recommended_substrate,
        "p1a_circuit_open_any_nonfever": p1a_circuit_open_any_nonfever,
        "p1a_positive_any_nonfever": p1a_positive_any_nonfever,
        "p1a_weak_success_any_nonfever": p1a_weak_success_any_nonfever,
        "p2a_circuit_open_any_nonfever": p2a_circuit_open_any_nonfever,
        "p2a_positive_any_nonfever": p2a_positive_any_nonfever,
        "p2a_weak_success_any_nonfever": p2a_weak_success_any_nonfever,
        "move_to_p1b": move_to_p1b,
        "move_to_p2b": move_to_p2b,
        "stop_after_p2a": not (move_to_p1b or move_to_p2b),
        "gsm8k": task_summaries["gsm8k"],
        "narrativeqa": task_summaries["narrativeqa"],
        "fever": task_summaries["fever"],
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Writer Circuit Opening Summary",
        "",
        f"- comparison_conclusion: {comparison_conclusion}",
        f"- primary_interpretation: {primary_interpretation}",
        f"- recommended_substrate: {recommended_substrate}",
        f"- p1a_circuit_open_any_nonfever: {p1a_circuit_open_any_nonfever}",
        f"- p2a_circuit_open_any_nonfever: {p2a_circuit_open_any_nonfever}",
        f"- move_to_p1b: {move_to_p1b}",
        f"- move_to_p2b: {move_to_p2b}",
    ]
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        task = task_summaries[task_name]
        report_lines.extend(
            [
                "",
                f"## {task_name}",
                f"- control_task_score: {task['control']['task_score']:.4f}",
                f"- p1a_delta_answer_logprob: {task['p1a']['delta_answer_logprob']:.4f}",
                f"- p1a_source_grad_norm_steps_1_4_median: {task['p1a']['source_grad_norm_steps_1_4_median']:.6f}",
                f"- p1a_prefix_attention_nontrivial_layer_count: {task['p1a']['prefix_attention_nontrivial_layer_count']}",
                f"- p1a_circuit_open: {task['p1a']['circuit_open']}",
                f"- p2a_delta_answer_logprob: {task['p2a']['delta_answer_logprob']:.4f}",
                f"- p2a_source_grad_norm_steps_1_4_median: {task['p2a']['source_grad_norm_steps_1_4_median']:.6f}",
                f"- p2a_receiver_lora_grad_norm_steps_1_4_median: {task['p2a']['receiver_lora_grad_norm_steps_1_4_median']:.6f}",
                f"- p2a_prefix_attention_nontrivial_layer_count: {task['p2a']['prefix_attention_nontrivial_layer_count']}",
                f"- p2a_circuit_open: {task['p2a']['circuit_open']}",
            ]
        )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
