#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_train_events(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
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


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def _window_values(
    train_events: list[dict[str, Any]],
    *,
    key: str,
    start_step: int,
    end_step: int,
) -> list[float]:
    return [
        float(event.get(key, 0.0))
        for event in train_events
        if start_step <= int(event.get("step", 0)) <= end_step
    ]


def _window_median(
    train_events: list[dict[str, Any]],
    *,
    key: str,
    start_step: int,
    end_step: int,
) -> float:
    return _median(_window_values(train_events, key=key, start_step=start_step, end_step=end_step))


def _window_fraction(
    train_events: list[dict[str, Any]],
    *,
    key: str,
    start_step: int,
    end_step: int,
) -> float:
    values = [
        bool(event.get(key, False))
        for event in train_events
        if start_step <= int(event.get("step", 0)) <= end_step
    ]
    if not values:
        return 0.0
    return float(sum(1 for value in values if value) / len(values))


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _task_mode(task_metric_name: str) -> str:
    normalized = str(task_metric_name).strip().lower()
    if normalized in {"accuracy", "macro_f1"}:
        return "classification"
    return "generation"


def _find_post_unfreeze_start(
    *,
    train_events: list[dict[str, Any]],
    writer_metrics: dict[str, Any],
) -> int:
    first_unfrozen_step = next(
        (
            int(event.get("step", 0))
            for event in train_events
            if not bool(event.get("writer_frozen", False))
        ),
        0,
    )
    if first_unfrozen_step > 0:
        return first_unfrozen_step
    warmup_steps = int(
        writer_metrics.get(
            "train_writer_post_unfreeze_start_step",
            writer_metrics.get("pilot_projector_warmup_steps", 50) + 1,
        )
    )
    return max(1, warmup_steps)


def _nontrivial_layer_count(prefix_attention_by_layer: dict[str, float]) -> int:
    return int(sum(1 for value in prefix_attention_by_layer.values() if float(value) > 1e-3))


def _snapshot_prefix_growth_ratio(metrics: dict[str, Any]) -> float:
    snapshots = metrics.get("snapshot_metrics", [])
    if not isinstance(snapshots, list):
        return 0.0
    prefix_l2_values = [
        float(snapshot.get("prefix_l2", 0.0))
        for snapshot in snapshots
        if float(snapshot.get("prefix_l2", 0.0)) > 0.0
    ]
    if len(prefix_l2_values) < 2:
        return 0.0
    return float(prefix_l2_values[-1] / max(prefix_l2_values[0], 1e-8))


def _case_row_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", "")).strip()
        if example_id:
            indexed[example_id] = row
    return indexed


def _classification_usefulness_metrics(
    *,
    control_rows: list[dict[str, Any]],
    writer_rows: list[dict[str, Any]],
) -> dict[str, float]:
    indexed_control = _case_row_index(control_rows)
    indexed_writer = _case_row_index(writer_rows)
    shared_ids = sorted(set(indexed_control) & set(indexed_writer))
    if not shared_ids:
        return {
            "margin_delta_mean": 0.0,
            "margin_delta_median": 0.0,
            "positive_margin_shift_fraction": 0.0,
            "correct_flip_count": 0.0,
            "answer_switch_rate": 0.0,
        }
    margin_deltas: list[float] = []
    correct_flip_count = 0
    answer_switch_count = 0
    for example_id in shared_ids:
        control_row = indexed_control[example_id]
        writer_row = indexed_writer[example_id]
        control_margin = float(control_row.get("final_margin", 0.0))
        writer_margin = float(writer_row.get("final_margin", 0.0))
        margin_deltas.append(writer_margin - control_margin)
        if str(control_row.get("predicted_label", "")) != str(writer_row.get("predicted_label", "")):
            answer_switch_count += 1
        if (
            not bool(control_row.get("predicted_correct", False))
            and bool(writer_row.get("predicted_correct", False))
        ):
            correct_flip_count += 1
    return {
        "margin_delta_mean": float(sum(margin_deltas) / len(margin_deltas)),
        "margin_delta_median": _median(margin_deltas),
        "positive_margin_shift_fraction": float(
            sum(1 for value in margin_deltas if value > 0.0) / len(margin_deltas)
        ),
        "correct_flip_count": float(correct_flip_count),
        "answer_switch_rate": float(answer_switch_count / len(shared_ids)),
    }


def _generation_usefulness_metrics(
    *,
    control_rows: list[dict[str, Any]],
    writer_rows: list[dict[str, Any]],
) -> dict[str, float]:
    deltas = [float(row.get("delta_answer_logprob", 0.0)) for row in writer_rows]
    indexed_control = _case_row_index(control_rows)
    indexed_writer = _case_row_index(writer_rows)
    shared_ids = sorted(set(indexed_control) & set(indexed_writer))
    answer_switch_count = 0
    for example_id in shared_ids:
        control_prediction = str(indexed_control[example_id].get("predicted_text", ""))
        writer_prediction = str(indexed_writer[example_id].get("predicted_text", ""))
        if control_prediction != writer_prediction:
            answer_switch_count += 1
    return {
        "delta_answer_logprob_mean": float(sum(deltas) / max(1, len(deltas))),
        "delta_answer_logprob_median": _median(deltas),
        "positive_delta_fraction": float(
            sum(1 for value in deltas if value > 0.0) / max(1, len(deltas))
        ),
        "nonzero_delta_case_count": float(
            sum(1 for value in deltas if abs(value) > 1e-9)
        ),
        "answer_switch_rate": float(answer_switch_count / max(1, len(shared_ids))),
    }


def _source_stub_health_summary(
    *,
    metrics: dict[str, Any],
    train_events: list[dict[str, Any]],
    head_window: int,
    tail_window: int,
) -> dict[str, Any]:
    prefix_attention_by_layer = _layer_metric(metrics, "prefix_attention_mass_mean_by_layer")
    nontrivial_layer_count = _nontrivial_layer_count(prefix_attention_by_layer)
    if train_events:
        head_end = min(len(train_events), head_window)
        tail_start = max(1, len(train_events) - tail_window + 1)
        loss_head = _window_median(train_events, key="loss", start_step=1, end_step=head_end)
        loss_tail = _window_median(
            train_events,
            key="loss",
            start_step=tail_start,
            end_step=len(train_events),
        )
    else:
        loss_head = _as_float(metrics, "train_loss_steps_1_50_median")
        loss_tail = _as_float(metrics, "train_loss_tail_50_steps_median")
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
    writer_train_events: list[dict[str, Any]],
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    task_score = _as_float(writer_metrics, "best_adapt_task_score")
    exact_match = _as_float(writer_metrics, "best_adapt_exact_match", task_score)
    control_score = _as_float(control_metrics, "best_adapt_task_score")
    control_exact_match = _as_float(control_metrics, "best_adapt_exact_match", control_score)
    prefix_attention_by_layer = _layer_metric(writer_metrics, "prefix_attention_mass_mean_by_layer")
    nontrivial_layer_count = _nontrivial_layer_count(prefix_attention_by_layer)
    task_metric_name = str(
        writer_metrics.get("task_metric_name", control_metrics.get("task_metric_name", "accuracy"))
    )
    mode = _task_mode(task_metric_name)
    train_steps = len(writer_train_events) if writer_train_events else int(writer_metrics.get("pilot_train_steps", 0))
    head_end = min(train_steps, head_window) if train_steps > 0 else head_window
    tail_start = max(1, train_steps - tail_window + 1) if train_steps > 0 else max(1, 451)
    post_unfreeze_start = _find_post_unfreeze_start(
        train_events=writer_train_events,
        writer_metrics=writer_metrics,
    )
    post_unfreeze_end = (
        min(train_steps, post_unfreeze_start + post_unfreeze_window - 1)
        if train_steps > 0
        else (post_unfreeze_start + post_unfreeze_window - 1)
    )

    def event_or_metric(
        event_key: str,
        metric_key: str,
        *,
        start_step: int,
        end_step: int,
        default: float = 0.0,
    ) -> float:
        if writer_train_events:
            value = _window_median(
                writer_train_events,
                key=event_key,
                start_step=start_step,
                end_step=end_step,
            )
            if value != 0.0:
                return value
        return _as_float(writer_metrics, metric_key, default)

    loss_head = event_or_metric(
        "loss",
        "train_loss_steps_1_50_median",
        start_step=1,
        end_step=head_end,
    )
    loss_tail = event_or_metric(
        "loss",
        "train_loss_steps_451_500_median",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    writer_grad_post_unfreeze = event_or_metric(
        "grad_norm_writer",
        "train_grad_norm_writer_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    projector_grad_post_unfreeze = event_or_metric(
        "grad_norm_projector",
        "train_grad_norm_projector_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    receiver_grad_post_unfreeze = event_or_metric(
        "grad_norm_receiver_lora",
        "train_grad_norm_receiver_lora_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_grad_tail = event_or_metric(
        "grad_norm_writer",
        "train_grad_norm_writer_steps_451_500_median",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    projector_grad_tail = event_or_metric(
        "grad_norm_projector",
        "train_grad_norm_projector_steps_451_500_median",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    receiver_grad_tail = event_or_metric(
        "grad_norm_receiver_lora",
        "train_grad_norm_receiver_lora_steps_451_500_median",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    writer_task_only_grad = event_or_metric(
        "grad_probe_writer_task_only_norm",
        "train_grad_probe_writer_task_only_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_aux_only_grad = event_or_metric(
        "grad_probe_writer_aux_only_norm",
        "train_grad_probe_writer_aux_only_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_total_grad = event_or_metric(
        "grad_probe_writer_total_norm",
        "train_grad_probe_writer_total_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_task_aux_cosine = event_or_metric(
        "grad_probe_writer_task_aux_cosine",
        "train_grad_probe_writer_task_aux_cosine_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_task_total_cosine = event_or_metric(
        "grad_probe_writer_task_total_cosine",
        "train_grad_probe_writer_task_total_cosine_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_aux_total_cosine = event_or_metric(
        "grad_probe_writer_aux_total_cosine",
        "train_grad_probe_writer_aux_total_cosine_post_unfreeze_median",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_clip_fraction_tail = (
        _window_fraction(
            writer_train_events,
            key="was_grad_clipped_writer",
            start_step=tail_start,
            end_step=max(tail_start, train_steps),
        )
        if writer_train_events
        else _as_float(writer_metrics, "train_writer_clip_fraction_tail_50")
    )
    projector_clip_fraction_tail = (
        _window_fraction(
            writer_train_events,
            key="was_grad_clipped_projector",
            start_step=tail_start,
            end_step=max(tail_start, train_steps),
        )
        if writer_train_events
        else _as_float(writer_metrics, "train_projector_clip_fraction_tail_50")
    )
    receiver_clip_fraction_tail = (
        _window_fraction(
            writer_train_events,
            key="was_grad_clipped_receiver_lora",
            start_step=tail_start,
            end_step=max(tail_start, train_steps),
        )
        if writer_train_events
        else _as_float(writer_metrics, "train_receiver_lora_clip_fraction_tail_50")
    )
    loss_step_delta_tail = (
        _median(
            [
                abs(curr - prev)
                for prev, curr in zip(
                    _window_values(
                        writer_train_events,
                        key="loss",
                        start_step=tail_start,
                        end_step=max(tail_start, train_steps),
                    )[:-1],
                    _window_values(
                        writer_train_events,
                        key="loss",
                        start_step=tail_start,
                        end_step=max(tail_start, train_steps),
                    )[1:],
                    strict=False,
                )
            ]
        )
        if writer_train_events
        else _as_float(writer_metrics, "train_loss_step_delta_tail_50_mean")
    )
    control_rows = _load_case_rows_from_metrics(control_metrics)
    writer_rows = _load_case_rows_from_metrics(writer_metrics)
    delta_answer_logprob = _as_float(writer_metrics, "delta_answer_logprob")
    non_regressive_task = bool(
        task_score >= (control_score - 1e-6)
        or exact_match >= (control_exact_match - 1e-6)
    )
    route_live_post_unfreeze = bool(
        writer_grad_post_unfreeze > 1e-4
        and projector_grad_post_unfreeze > 1e-3
        and receiver_grad_post_unfreeze > 1e-4
        and nontrivial_layer_count >= 2
        and _all_finite(
            [
                loss_head,
                loss_tail,
                writer_grad_post_unfreeze,
                projector_grad_post_unfreeze,
                receiver_grad_post_unfreeze,
                _as_float(writer_metrics, "prefix_attention_mass_mean"),
            ]
        )
    )
    route_live_post_unfreeze_medium = bool(
        route_live_post_unfreeze
        and writer_grad_tail > 1e-4
        and projector_grad_tail > 1e-3
        and receiver_grad_tail > 1e-4
    )
    writer_task_ratio = float(writer_task_only_grad / max(writer_total_grad, 1e-8))
    writer_task_supervision_live = bool(
        writer_task_ratio >= 0.10
        and writer_task_total_cosine > 0.0
    )
    writer_task_supervision_live_medium = bool(
        writer_task_ratio >= 0.20
        and writer_task_total_cosine >= 0.30
        and writer_task_aux_cosine > -0.20
    )
    support_state_effective_rank = _as_float(writer_metrics, "train_final_support_state_effective_rank")
    writer_memory_slot_effective_rank = _as_float(
        writer_metrics,
        "train_final_memory_long_effective_rank",
        _as_float(writer_metrics, "memory_long_effective_rank"),
    )
    common_mode_ratio = _as_float(writer_metrics, "memory_long_common_mode_energy_ratio", 1.0)
    slot_pairwise_cosine_present = "train_final_writer_slot_basis_pairwise_cosine_mean" in writer_metrics
    slot_pairwise_cosine = _as_float(
        writer_metrics,
        "train_final_writer_slot_basis_pairwise_cosine_mean",
    )
    source_not_collapsed = bool(
        support_state_effective_rank > 1.2
        or writer_memory_slot_effective_rank > 1.5
        or common_mode_ratio < 0.999
        or (
            slot_pairwise_cosine_present
            and slot_pairwise_cosine > 0.0
            and slot_pairwise_cosine < 0.95
        )
    )
    prefix_growth_ratio = _snapshot_prefix_growth_ratio(writer_metrics)
    stable_training_v6 = bool(
        route_live_post_unfreeze
        and loss_tail > 0.0
        and loss_tail < loss_head
        and _all_finite([loss_head, loss_tail, loss_step_delta_tail, prefix_growth_ratio])
        and not (
            writer_clip_fraction_tail >= 0.95
            and projector_clip_fraction_tail >= 0.95
            and receiver_clip_fraction_tail >= 0.95
        )
        and (prefix_growth_ratio == 0.0 or prefix_growth_ratio <= 16.0)
    )
    if mode == "classification":
        usefulness_metrics = _classification_usefulness_metrics(
            control_rows=control_rows,
            writer_rows=writer_rows,
        )
        usefulness_positive_v6 = bool(
            non_regressive_task
            and (
                usefulness_metrics["margin_delta_mean"] > 0.0
                or usefulness_metrics["positive_margin_shift_fraction"] > 0.5
                or usefulness_metrics["correct_flip_count"] > 0.0
            )
        )
    else:
        usefulness_metrics = _generation_usefulness_metrics(
            control_rows=control_rows,
            writer_rows=writer_rows,
        )
        usefulness_positive_v6 = bool(
            non_regressive_task
            and (
                usefulness_metrics["delta_answer_logprob_median"] > 0.0
                or usefulness_metrics["positive_delta_fraction"] > 0.5
            )
        )
    return {
        "task_name": str(writer_metrics.get("task_name", control_metrics.get("task_name", ""))),
        "benchmark_id": str(writer_metrics.get("benchmark_id", control_metrics.get("benchmark_id", ""))),
        "task_metric_name": task_metric_name,
        "task_mode": mode,
        "task_score": task_score,
        "exact_match": exact_match,
        "task_score_delta_vs_control": task_score - control_score,
        "exact_match_delta_vs_control": exact_match - control_exact_match,
        "delta_answer_logprob": delta_answer_logprob,
        "prefix_attention_mass_mean": _as_float(writer_metrics, "prefix_attention_mass_mean"),
        "prefix_attention_mass_mean_by_layer": prefix_attention_by_layer,
        "prefix_attention_nontrivial_layer_count": nontrivial_layer_count,
        "projected_memory_effective_rank": _as_float(writer_metrics, "projected_memory_effective_rank"),
        "memory_long_common_mode_energy_ratio": common_mode_ratio,
        "support_state_effective_rank": support_state_effective_rank,
        "writer_memory_slot_effective_rank": writer_memory_slot_effective_rank,
        "loss_steps_1_50_median": loss_head,
        "loss_steps_451_500_median": loss_tail,
        "writer_grad_norm_post_unfreeze_median": writer_grad_post_unfreeze,
        "writer_grad_norm_steps_451_500_median": writer_grad_tail,
        "projector_grad_norm_post_unfreeze_median": projector_grad_post_unfreeze,
        "projector_grad_norm_steps_451_500_median": projector_grad_tail,
        "receiver_lora_grad_norm_post_unfreeze_median": receiver_grad_post_unfreeze,
        "receiver_lora_grad_norm_steps_451_500_median": receiver_grad_tail,
        "writer_task_only_grad_norm_post_unfreeze_median": writer_task_only_grad,
        "writer_aux_only_grad_norm_post_unfreeze_median": writer_aux_only_grad,
        "writer_total_grad_norm_post_unfreeze_median": writer_total_grad,
        "writer_task_to_total_grad_ratio_post_unfreeze": writer_task_ratio,
        "writer_task_aux_cosine_post_unfreeze_median": writer_task_aux_cosine,
        "writer_task_total_cosine_post_unfreeze_median": writer_task_total_cosine,
        "writer_aux_total_cosine_post_unfreeze_median": writer_aux_total_cosine,
        "writer_clip_fraction_tail_50": writer_clip_fraction_tail,
        "projector_clip_fraction_tail_50": projector_clip_fraction_tail,
        "receiver_lora_clip_fraction_tail_50": receiver_clip_fraction_tail,
        "loss_step_delta_tail_50_mean": loss_step_delta_tail,
        "prefix_l2_growth_ratio": prefix_growth_ratio,
        "non_regressive_task": non_regressive_task,
        "route_live_post_unfreeze": route_live_post_unfreeze,
        "route_live_post_unfreeze_medium": route_live_post_unfreeze_medium,
        "writer_task_supervision_live": writer_task_supervision_live,
        "writer_task_supervision_live_medium": writer_task_supervision_live_medium,
        "source_not_collapsed": source_not_collapsed,
        "stable_training_v6": stable_training_v6,
        "usefulness_positive_v6": usefulness_positive_v6,
        "route_live": route_live_post_unfreeze,
        "stable_training": stable_training_v6,
        "usefulness_positive": usefulness_positive_v6,
        "post_unfreeze_start_step": post_unfreeze_start,
        "post_unfreeze_end_step": post_unfreeze_end,
        **usefulness_metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the Writer-direct deep-prefix joint-PEFT branch under PLANv6 gate semantics."
    )
    parser.add_argument("--source_stub_health_metrics_json", required=True)
    parser.add_argument("--source_stub_health_train_events_json", required=True)
    for task_name in ("gsm8k", "narrativeqa", "fever"):
        parser.add_argument(f"--{task_name}_control_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_writer_metrics_json", required=True)
        parser.add_argument(f"--{task_name}_writer_train_events_json")
    parser.add_argument("--head_window", type=int, default=50)
    parser.add_argument("--post_unfreeze_window", type=int, default=50)
    parser.add_argument("--tail_window", type=int, default=50)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_stub_health = _source_stub_health_summary(
        metrics=_load_json(args.source_stub_health_metrics_json),
        train_events=_load_train_events(args.source_stub_health_train_events_json),
        head_window=args.head_window,
        tail_window=args.tail_window,
    )
    task_summaries = {
        task_name: _task_summary(
            control_metrics=_load_json(getattr(args, f"{task_name}_control_metrics_json")),
            writer_metrics=_load_json(getattr(args, f"{task_name}_writer_metrics_json")),
            writer_train_events=_load_train_events(
                getattr(args, f"{task_name}_writer_train_events_json", None)
            ),
            head_window=args.head_window,
            post_unfreeze_window=args.post_unfreeze_window,
            tail_window=args.tail_window,
        )
        for task_name in ("gsm8k", "narrativeqa", "fever")
    }
    nonfever_tasks = [task_summaries["gsm8k"], task_summaries["narrativeqa"]]
    any_nonfever_route_live = any(bool(task["route_live_post_unfreeze"]) for task in nonfever_tasks)
    any_nonfever_task_supervision_live = any(
        bool(task["writer_task_supervision_live"]) for task in nonfever_tasks
    )
    any_nonfever_source_not_collapsed = any(
        bool(task["source_not_collapsed"]) for task in nonfever_tasks
    )
    any_nonfever_stable_training = any(bool(task["stable_training_v6"]) for task in nonfever_tasks)
    any_nonfever_usefulness_positive = any(
        bool(task["usefulness_positive_v6"]) for task in nonfever_tasks
    )
    if any_nonfever_usefulness_positive and any_nonfever_task_supervision_live:
        comparison_conclusion = "move_to_writer_usefulness_branch"
        recommended_next_step = "open_writer_usefulness_branch"
    elif any_nonfever_route_live and not any_nonfever_task_supervision_live:
        comparison_conclusion = "move_to_v6_baseline_rerun"
        recommended_next_step = "run_clean_v6_baseline"
    elif any_nonfever_route_live and any_nonfever_task_supervision_live and not any_nonfever_source_not_collapsed:
        comparison_conclusion = "move_to_support_screening"
        recommended_next_step = "run_support_interface_screen"
    elif any_nonfever_route_live and any_nonfever_task_supervision_live and not any_nonfever_stable_training:
        comparison_conclusion = "stabilize_same_architecture"
        recommended_next_step = "run_recipe_stabilization"
    elif any_nonfever_route_live:
        comparison_conclusion = "iterate_same_architecture"
        recommended_next_step = "run_mixed_matrix"
    else:
        comparison_conclusion = "fix_route_before_next_run"
        recommended_next_step = "debug_route_or_recipe"
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "move_to_writer_usefulness_branch": comparison_conclusion == "move_to_writer_usefulness_branch",
        "move_to_v6_baseline_rerun": comparison_conclusion == "move_to_v6_baseline_rerun",
        "move_to_support_screening": comparison_conclusion == "move_to_support_screening",
        "stabilize_same_architecture": comparison_conclusion == "stabilize_same_architecture",
        "iterate_same_architecture": comparison_conclusion == "iterate_same_architecture",
        "fix_route_before_next_run": comparison_conclusion == "fix_route_before_next_run",
        "source_stub_health": source_stub_health,
        "any_nonfever_route_live": any_nonfever_route_live,
        "any_nonfever_task_supervision_live": any_nonfever_task_supervision_live,
        "any_nonfever_source_not_collapsed": any_nonfever_source_not_collapsed,
        "any_nonfever_stable_training": any_nonfever_stable_training,
        "any_nonfever_usefulness_positive": any_nonfever_usefulness_positive,
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
        f"- any_nonfever_task_supervision_live: {any_nonfever_task_supervision_live}",
        f"- any_nonfever_source_not_collapsed: {any_nonfever_source_not_collapsed}",
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
                f"- route_live_post_unfreeze: {task['route_live_post_unfreeze']}",
                f"- writer_task_supervision_live: {task['writer_task_supervision_live']}",
                f"- source_not_collapsed: {task['source_not_collapsed']}",
                f"- stable_training_v6: {task['stable_training_v6']}",
                f"- usefulness_positive_v6: {task['usefulness_positive_v6']}",
                f"- loss_steps_1_50_median: {task['loss_steps_1_50_median']:.6f}",
                f"- loss_steps_451_500_median: {task['loss_steps_451_500_median']:.6f}",
                f"- writer_grad_norm_post_unfreeze_median: {task['writer_grad_norm_post_unfreeze_median']:.6f}",
                f"- writer_task_to_total_grad_ratio_post_unfreeze: {task['writer_task_to_total_grad_ratio_post_unfreeze']:.6f}",
                f"- delta_answer_logprob: {task['delta_answer_logprob']:.6f}",
                f"- prefix_attention_mass_mean: {task['prefix_attention_mass_mean']:.6f}",
                f"- prefix_attention_nontrivial_layer_count: {task['prefix_attention_nontrivial_layer_count']}",
            ]
        )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
