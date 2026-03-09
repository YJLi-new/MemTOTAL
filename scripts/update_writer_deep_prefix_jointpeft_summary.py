#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def _load_case_rows_from_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    raw_path = str(metrics.get("task_case_dump_path", "")).strip()
    if not raw_path:
        return []
    path = Path(raw_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


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


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _source_stub_health_summary(
    *,
    metrics: dict[str, Any],
    train_events: list[dict[str, Any]],
) -> dict[str, Any]:
    prefix_attention_by_layer = _layer_metric(metrics, "prefix_attention_mass_mean_by_layer")
    nontrivial_layer_count = int(
        sum(1 for value in prefix_attention_by_layer.values() if float(value) > 1e-3)
    )
    loss_head = _as_float(metrics, "train_loss_steps_1_50_median")
    loss_tail = _as_float(metrics, "train_loss_tail_50_steps_median")
    if train_events and (len(train_events) < 50 or loss_head <= 0.0 or loss_tail <= 0.0):
        first_half_end = max(1, len(train_events) // 2)
        head_values = [float(event.get("loss", 0.0)) for event in train_events[:first_half_end]]
        tail_values = [float(event.get("loss", 0.0)) for event in train_events[first_half_end:]]
        if head_values:
            loss_head = sum(head_values) / len(head_values)
        if tail_values:
            loss_tail = sum(tail_values) / len(tail_values)
    route_live = bool(
        _as_float(metrics, "train_grad_norm_source_stub_steps_1_4_median") > 1e-6
        and nontrivial_layer_count > 0
    )
    stable_recipe = bool(loss_tail > 0.0 and loss_tail <= loss_head)
    return {
        "route_live": route_live,
        "stable_recipe": stable_recipe,
        "loss_steps_1_50_median": loss_head,
        "loss_tail_50_steps_median": loss_tail,
        "delta_answer_logprob": _as_float(metrics, "delta_answer_logprob"),
        "prefix_attention_mass_mean": _as_float(metrics, "prefix_attention_mass_mean"),
        "prefix_attention_nontrivial_layer_count": nontrivial_layer_count,
        "source_grad_norm_steps_1_4_median": _as_float(metrics, "train_grad_norm_source_stub_steps_1_4_median"),
        "receiver_lora_grad_norm_steps_1_4_median": _as_float(
            metrics,
            "train_grad_norm_receiver_lora_steps_1_4_median",
        ),
    }


def _task_summary(
    *,
    control_metrics: dict[str, Any],
    writer_metrics: dict[str, Any],
) -> dict[str, Any]:
    task_score = _as_float(writer_metrics, "best_adapt_task_score")
    exact_match = _as_float(writer_metrics, "best_adapt_exact_match", task_score)
    control_score = _as_float(control_metrics, "best_adapt_task_score")
    control_exact_match = _as_float(control_metrics, "best_adapt_exact_match", control_score)
    delta_answer_logprob = _as_float(writer_metrics, "delta_answer_logprob")
    prefix_attention_by_layer = _layer_metric(writer_metrics, "prefix_attention_mass_mean_by_layer")
    nontrivial_layer_count = int(
        sum(1 for value in prefix_attention_by_layer.values() if float(value) > 1e-3)
    )
    loss_head = _as_float(writer_metrics, "train_loss_steps_1_50_median")
    loss_tail = _as_float(writer_metrics, "train_loss_steps_451_500_median")
    writer_grad_head = _as_float(writer_metrics, "train_grad_norm_writer_steps_1_50_median")
    writer_grad_tail = _as_float(writer_metrics, "train_grad_norm_writer_steps_451_500_median")
    projector_grad_head = _as_float(writer_metrics, "train_grad_norm_projector_steps_1_50_median")
    projector_grad_tail = _as_float(writer_metrics, "train_grad_norm_projector_steps_451_500_median")
    receiver_grad_head = _as_float(writer_metrics, "train_grad_norm_receiver_lora_steps_1_50_median")
    receiver_grad_tail = _as_float(writer_metrics, "train_grad_norm_receiver_lora_steps_451_500_median")
    case_rows = _load_case_rows_from_metrics(writer_metrics)
    nonzero_delta_case_count = int(
        sum(1 for row in case_rows if abs(float(row.get("delta_answer_logprob", 0.0))) > 1e-9)
    )
    route_live = bool(
        writer_grad_head > 1e-4
        and projector_grad_head > 1e-3
        and nontrivial_layer_count > 0
        and _all_finite(
            [
                loss_head,
                loss_tail,
                writer_grad_head,
                projector_grad_head,
                receiver_grad_head,
                _as_float(writer_metrics, "prefix_attention_mass_mean"),
            ]
        )
    )
    stable_training = bool(
        route_live
        and loss_tail > 0.0
        and loss_tail < loss_head
        and nonzero_delta_case_count > 0
    )
    non_regressive_task = bool(
        task_score >= (control_score - 1e-6)
        or exact_match >= (control_exact_match - 1e-6)
    )
    usefulness_positive = bool(
        stable_training
        and delta_answer_logprob > 0.0
        and non_regressive_task
    )
    weak_prefix_attention = bool(
        _as_float(writer_metrics, "prefix_attention_mass_mean") < 0.01
        or nontrivial_layer_count < 2
    )
    return {
        "task_name": str(writer_metrics.get("task_name", control_metrics.get("task_name", ""))),
        "benchmark_id": str(writer_metrics.get("benchmark_id", control_metrics.get("benchmark_id", ""))),
        "task_metric_name": str(
            writer_metrics.get("task_metric_name", control_metrics.get("task_metric_name", "accuracy"))
        ),
        "task_score": task_score,
        "exact_match": exact_match,
        "task_score_delta_vs_control": task_score - control_score,
        "exact_match_delta_vs_control": exact_match - control_exact_match,
        "delta_answer_logprob": delta_answer_logprob,
        "prefix_attention_mass_mean": _as_float(writer_metrics, "prefix_attention_mass_mean"),
        "prefix_attention_mass_mean_by_layer": prefix_attention_by_layer,
        "prefix_attention_nontrivial_layer_count": nontrivial_layer_count,
        "projected_memory_effective_rank": _as_float(writer_metrics, "projected_memory_effective_rank"),
        "memory_long_common_mode_energy_ratio": _as_float(
            writer_metrics,
            "memory_long_common_mode_energy_ratio",
        ),
        "loss_steps_1_50_median": loss_head,
        "loss_steps_451_500_median": loss_tail,
        "writer_grad_norm_steps_1_50_median": writer_grad_head,
        "writer_grad_norm_steps_451_500_median": writer_grad_tail,
        "projector_grad_norm_steps_1_50_median": projector_grad_head,
        "projector_grad_norm_steps_451_500_median": projector_grad_tail,
        "receiver_lora_grad_norm_steps_1_50_median": receiver_grad_head,
        "receiver_lora_grad_norm_steps_451_500_median": receiver_grad_tail,
        "nonzero_delta_case_count": nonzero_delta_case_count,
        "route_live": route_live,
        "stable_training": stable_training,
        "usefulness_positive": usefulness_positive,
        "non_regressive_task": non_regressive_task,
        "weak_prefix_attention": weak_prefix_attention,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the stable writer-direct deep-prefix joint-PEFT branch.")
    parser.add_argument("--source_stub_health_metrics_json", required=True)
    parser.add_argument("--source_stub_health_train_events_json", required=True)
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        parser.add_argument(f"--{task_name}_control_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_writer_metrics_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_stub_health = _source_stub_health_summary(
        metrics=_load_json(args.source_stub_health_metrics_json),
        train_events=_load_train_events(args.source_stub_health_train_events_json),
    )
    task_summaries = {
        task_name: _task_summary(
            control_metrics=_load_json(getattr(args, f"{task_name}_control_metrics_json")),
            writer_metrics=_load_json(getattr(args, f"{task_name}_writer_metrics_json")),
        )
        for task_name in ("gsm8k", "narrativeqa", "fever")
    }
    nonfever_tasks = [task_summaries["gsm8k"], task_summaries["narrativeqa"]]
    any_nonfever_route_live = any(bool(task["route_live"]) for task in nonfever_tasks)
    any_nonfever_stable_training = any(bool(task["stable_training"]) for task in nonfever_tasks)
    any_nonfever_usefulness_positive = any(bool(task["usefulness_positive"]) for task in nonfever_tasks)
    all_nonfever_weak_prefix_attention = all(bool(task["weak_prefix_attention"]) for task in nonfever_tasks)
    if any_nonfever_usefulness_positive:
        comparison_conclusion = "move_to_writer_usefulness_branch"
        recommended_next_step = "open_writer_usefulness_branch"
    elif any_nonfever_stable_training and all_nonfever_weak_prefix_attention:
        comparison_conclusion = "move_to_layer_expansion_comparator"
        recommended_next_step = "run_deeper_layer_comparator"
    elif any_nonfever_route_live:
        comparison_conclusion = "iterate_recipe_same_architecture"
        recommended_next_step = "stabilize_same_architecture"
    else:
        comparison_conclusion = "fix_route_before_next_run"
        recommended_next_step = "debug_route_or_recipe"
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "move_to_writer_usefulness_branch": comparison_conclusion == "move_to_writer_usefulness_branch",
        "move_to_layer_expansion_comparator": comparison_conclusion == "move_to_layer_expansion_comparator",
        "iterate_recipe_same_architecture": comparison_conclusion == "iterate_recipe_same_architecture",
        "fix_route_before_next_run": comparison_conclusion == "fix_route_before_next_run",
        "source_stub_health": source_stub_health,
        "any_nonfever_route_live": any_nonfever_route_live,
        "any_nonfever_stable_training": any_nonfever_stable_training,
        "any_nonfever_usefulness_positive": any_nonfever_usefulness_positive,
        "all_nonfever_weak_prefix_attention": all_nonfever_weak_prefix_attention,
        "gsm8k": task_summaries["gsm8k"],
        "narrativeqa": task_summaries["narrativeqa"],
        "fever": task_summaries["fever"],
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Writer Deep-Prefix Joint-PEFT Summary",
        "",
        f"- comparison_conclusion: {comparison_conclusion}",
        f"- recommended_next_step: {recommended_next_step}",
        f"- any_nonfever_route_live: {any_nonfever_route_live}",
        f"- any_nonfever_stable_training: {any_nonfever_stable_training}",
        f"- any_nonfever_usefulness_positive: {any_nonfever_usefulness_positive}",
        "",
        "## source_stub_health",
        f"- route_live: {source_stub_health['route_live']}",
        f"- stable_recipe: {source_stub_health['stable_recipe']}",
        f"- loss_steps_1_50_median: {source_stub_health['loss_steps_1_50_median']:.6f}",
        f"- loss_tail_50_steps_median: {source_stub_health['loss_tail_50_steps_median']:.6f}",
    ]
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        task = task_summaries[task_name]
        report_lines.extend(
            [
                "",
                f"## {task_name}",
                f"- route_live: {task['route_live']}",
                f"- stable_training: {task['stable_training']}",
                f"- usefulness_positive: {task['usefulness_positive']}",
                f"- loss_steps_1_50_median: {task['loss_steps_1_50_median']:.6f}",
                f"- loss_steps_451_500_median: {task['loss_steps_451_500_median']:.6f}",
                f"- writer_grad_norm_steps_1_50_median: {task['writer_grad_norm_steps_1_50_median']:.6f}",
                f"- projector_grad_norm_steps_1_50_median: {task['projector_grad_norm_steps_1_50_median']:.6f}",
                f"- receiver_lora_grad_norm_steps_1_50_median: {task['receiver_lora_grad_norm_steps_1_50_median']:.6f}",
                f"- delta_answer_logprob: {task['delta_answer_logprob']:.6f}",
                f"- prefix_attention_mass_mean: {task['prefix_attention_mass_mean']:.6f}",
                f"- prefix_attention_nontrivial_layer_count: {task['prefix_attention_nontrivial_layer_count']}",
            ]
        )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
