#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_train_events(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        events = payload.get("events", [])
        if isinstance(events, list):
            return events
    return []


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
    return {str(layer_index): float(value) for layer_index, value in raw.items()}


def _median_event_metric(
    events: list[dict[str, Any]],
    *,
    key: str,
    step_start: int,
    step_end: int,
) -> float:
    values = [
        float(event.get(key, 0.0))
        for event in events
        if step_start <= int(event.get("step", 0)) <= step_end
    ]
    if not values:
        return 0.0
    values.sort()
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _run_summary(
    *,
    metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    train_events: list[dict[str, Any]],
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
    route_live = bool(
        source_grad > 1e-6
        and nontrivial_layer_count > 0
        and (not expect_receiver_lora or receiver_grad > 0.0)
    )
    usefulness_positive = bool(delta_answer_logprob > 0.0)
    non_regressive_task = bool(
        task_score >= (control_score - 1e-6)
        or exact_match >= (control_exact_match - 1e-6)
    )
    loss_head = _median_event_metric(train_events, key="loss", step_start=1, step_end=8)
    loss_tail = _median_event_metric(train_events, key="loss", step_start=9, step_end=16)
    stable_training = bool(loss_tail > 0.0 and loss_tail <= loss_head)
    route_strength_score = float(
        (10.0 * nontrivial_layer_count)
        + min(10.0, source_grad * 10.0)
        + min(10.0, _as_float(metrics, "prefix_attention_mass_mean") * 1000.0)
        + (5.0 if expect_receiver_lora and receiver_grad > 0.0 else 0.0)
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
        "receiver_lora_grad_norm_steps_1_4_median": receiver_grad,
        "prefix_attention_mass_mean": _as_float(metrics, "prefix_attention_mass_mean"),
        "prefix_attention_mass_mean_by_layer": prefix_attention_by_layer,
        "prefix_attention_nontrivial_layer_count": nontrivial_layer_count,
        "projected_memory_effective_rank": _as_float(metrics, "projected_memory_effective_rank"),
        "memory_long_common_mode_energy_ratio": _as_float(metrics, "memory_long_common_mode_energy_ratio"),
        "route_live": route_live,
        "stable_training": stable_training,
        "usefulness_positive": usefulness_positive,
        "non_regressive_task": non_regressive_task,
        "loss_steps_1_8_median": loss_head,
        "loss_steps_9_16_median": loss_tail,
        "route_strength_score": route_strength_score,
    }


def _task_summary(
    *,
    control_metrics: dict[str, Any],
    p1a_metrics: dict[str, Any],
    p1a_events: list[dict[str, Any]],
    p2a_metrics: dict[str, Any],
    p2a_events: list[dict[str, Any]],
) -> dict[str, Any]:
    p1a = _run_summary(
        metrics=p1a_metrics,
        control_metrics=control_metrics,
        train_events=p1a_events,
        expect_receiver_lora=False,
    )
    p2a = _run_summary(
        metrics=p2a_metrics,
        control_metrics=control_metrics,
        train_events=p2a_events,
        expect_receiver_lora=True,
    )
    return {
        "control": {
            "task_name": str(control_metrics.get("task_name", "")),
            "benchmark_id": str(control_metrics.get("benchmark_id", "")),
            "task_metric_name": str(control_metrics.get("task_metric_name", "accuracy")),
            "task_score": _as_float(control_metrics, "best_adapt_task_score"),
            "exact_match": _as_float(
                control_metrics,
                "best_adapt_exact_match",
                _as_float(control_metrics, "best_adapt_task_score"),
            ),
        },
        "p1a": p1a,
        "p2a": p2a,
        "preferred_route": (
            "p2a"
            if p2a["route_live"] and p2a["route_strength_score"] > p1a["route_strength_score"]
            else ("p1a" if p1a["route_live"] else ("p2a" if p2a["route_live"] else "none"))
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the corrected PLANv5 source-stub interpretation.")
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        parser.add_argument(f"--{task_name}_control_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_p1a_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_p1a_train_events_json", required=True)
        parser.add_argument(f"--{task_name}_p2a_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_p2a_train_events_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    task_summaries = {
        task_name: _task_summary(
            control_metrics=_load_json(getattr(args, f"{task_name}_control_metrics_json")),
            p1a_metrics=_load_json(getattr(args, f"{task_name}_p1a_metrics_json")),
            p1a_events=_load_train_events(getattr(args, f"{task_name}_p1a_train_events_json")),
            p2a_metrics=_load_json(getattr(args, f"{task_name}_p2a_metrics_json")),
            p2a_events=_load_train_events(getattr(args, f"{task_name}_p2a_train_events_json")),
        )
        for task_name in ("gsm8k", "narrativeqa", "fever")
    }
    nonfever_tasks = [task_summaries["gsm8k"], task_summaries["narrativeqa"]]
    p1a_route_live_any_nonfever = any(bool(task["p1a"]["route_live"]) for task in nonfever_tasks)
    p2a_route_live_any_nonfever = any(bool(task["p2a"]["route_live"]) for task in nonfever_tasks)
    p1a_usefulness_positive_any_nonfever = any(
        bool(task["p1a"]["usefulness_positive"]) for task in nonfever_tasks
    )
    p2a_usefulness_positive_any_nonfever = any(
        bool(task["p2a"]["usefulness_positive"]) for task in nonfever_tasks
    )
    p1a_route_strength = sum(float(task["p1a"]["route_strength_score"]) for task in nonfever_tasks)
    p2a_route_strength = sum(float(task["p2a"]["route_strength_score"]) for task in nonfever_tasks)
    if p2a_route_live_any_nonfever and p2a_route_strength > p1a_route_strength:
        recommended_substrate = "p2a_source_stub_receiver_lora_r2"
    elif p1a_route_live_any_nonfever:
        recommended_substrate = "p1a_source_stub_no_lora"
    elif p2a_route_live_any_nonfever:
        recommended_substrate = "p2a_source_stub_receiver_lora_r2"
    else:
        recommended_substrate = "none"
    comparison_conclusion = (
        "move_to_writer_direct_validation"
        if recommended_substrate != "none"
        else "fix_source_stub_route"
    )
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "recommended_substrate": recommended_substrate,
        "primary_interpretation": (
            "source_stub_route_live_but_not_usefulness_testable"
            if recommended_substrate != "none"
            else "source_stub_route_not_live"
        ),
        "p1a_route_live_any_nonfever": p1a_route_live_any_nonfever,
        "p2a_route_live_any_nonfever": p2a_route_live_any_nonfever,
        "p1a_usefulness_positive_any_nonfever": p1a_usefulness_positive_any_nonfever,
        "p2a_usefulness_positive_any_nonfever": p2a_usefulness_positive_any_nonfever,
        "move_to_writer_direct_validation": bool(recommended_substrate != "none"),
        "stop_after_p2a": False,
        "gsm8k": task_summaries["gsm8k"],
        "narrativeqa": task_summaries["narrativeqa"],
        "fever": task_summaries["fever"],
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Writer Circuit Opening Addendum Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- primary_interpretation: {summary['primary_interpretation']}",
        f"- recommended_substrate: {summary['recommended_substrate']}",
        f"- p1a_route_live_any_nonfever: {summary['p1a_route_live_any_nonfever']}",
        f"- p2a_route_live_any_nonfever: {summary['p2a_route_live_any_nonfever']}",
    ]
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        task = task_summaries[task_name]
        report_lines.extend(
            [
                "",
                f"## {task_name}",
                f"- p1a_route_live: {task['p1a']['route_live']}",
                f"- p1a_usefulness_positive: {task['p1a']['usefulness_positive']}",
                f"- p1a_source_grad_norm_steps_1_4_median: {task['p1a']['source_grad_norm_steps_1_4_median']:.6f}",
                f"- p1a_prefix_attention_nontrivial_layer_count: {task['p1a']['prefix_attention_nontrivial_layer_count']}",
                f"- p2a_route_live: {task['p2a']['route_live']}",
                f"- p2a_usefulness_positive: {task['p2a']['usefulness_positive']}",
                f"- p2a_source_grad_norm_steps_1_4_median: {task['p2a']['source_grad_norm_steps_1_4_median']:.6f}",
                f"- p2a_receiver_lora_grad_norm_steps_1_4_median: {task['p2a']['receiver_lora_grad_norm_steps_1_4_median']:.6f}",
                f"- p2a_prefix_attention_nontrivial_layer_count: {task['p2a']['prefix_attention_nontrivial_layer_count']}",
                f"- preferred_route: {task['preferred_route']}",
            ]
        )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
