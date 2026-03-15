#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
REPLAY_ARMS = ("c_add", "c_early", "c_mid")
ORACLE_ARMS = ("o_ctx_early", "o_ctx_mid", "o_sup_early", "o_sup_mid")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_train_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
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
    active_only_key: str | None = None,
) -> list[float]:
    return [
        float(event.get(key, 0.0))
        for event in train_events
        if start_step <= int(event.get("step", 0)) <= end_step
        and (active_only_key is None or bool(event.get(active_only_key, False)))
    ]


def _window_median(
    train_events: list[dict[str, Any]],
    *,
    key: str,
    start_step: int,
    end_step: int,
    active_only_key: str | None = None,
) -> float:
    return _median(
        _window_values(
            train_events,
            key=key,
            start_step=start_step,
            end_step=end_step,
            active_only_key=active_only_key,
        )
    )


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
            writer_metrics.get("pilot_projector_warmup_steps", 0) + 1,
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
    indexed_control = _case_row_index(control_rows)
    indexed_writer = _case_row_index(writer_rows)
    shared_ids = sorted(set(indexed_control) & set(indexed_writer))
    if not shared_ids:
        return {
            "delta_answer_logprob_mean": 0.0,
            "delta_answer_logprob_median": 0.0,
            "positive_delta_fraction": 0.0,
            "answer_switch_rate": 0.0,
        }
    delta_values: list[float] = []
    answer_switch_count = 0
    for example_id in shared_ids:
        control_row = indexed_control[example_id]
        writer_row = indexed_writer[example_id]
        delta_values.append(
            float(writer_row.get("answer_logprob_with_memory", 0.0))
            - float(control_row.get("answer_logprob_with_memory", 0.0))
        )
        if str(control_row.get("prediction", "")) != str(writer_row.get("prediction", "")):
            answer_switch_count += 1
    return {
        "delta_answer_logprob_mean": float(sum(delta_values) / len(delta_values)),
        "delta_answer_logprob_median": _median(delta_values),
        "positive_delta_fraction": float(sum(1 for value in delta_values if value > 0.0) / len(delta_values)),
        "answer_switch_rate": float(answer_switch_count / len(shared_ids)),
    }


def _writer_family_from_metrics(metrics: dict[str, Any]) -> str:
    slots = max(1, int(round(_as_float(metrics, "writer_memory_slots", 8.0))))
    if slots <= 16:
        return "w0_w1"
    if slots <= 32:
        return "w2"
    return "w3_w4"


def _strict_thresholds(metrics: dict[str, Any]) -> dict[str, float]:
    family = _writer_family_from_metrics(metrics)
    if family == "w2":
        return {
            "rank_floor": 4.0,
            "rank_fraction_floor": 0.125,
            "common_mode_ceiling": 0.990,
            "pairwise_cosine_ceiling": 0.85,
        }
    if family == "w3_w4":
        return {
            "rank_floor": 6.0,
            "rank_fraction_floor": 0.125,
            "common_mode_ceiling": 0.985,
            "pairwise_cosine_ceiling": 0.80,
        }
    return {
        "rank_floor": 2.0,
        "rank_fraction_floor": 0.125,
        "common_mode_ceiling": 0.995,
        "pairwise_cosine_ceiling": 0.90,
    }


def _old_permissive_collapse_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    support_state_effective_rank = _as_float(metrics, "train_final_support_state_effective_rank")
    writer_memory_slot_effective_rank = _as_float(
        metrics,
        "train_final_memory_long_effective_rank",
        _as_float(metrics, "memory_long_effective_rank"),
    )
    common_mode_ratio = _as_float(metrics, "memory_long_common_mode_energy_ratio", 1.0)
    slot_pairwise_cosine_present = "train_final_writer_slot_basis_pairwise_cosine_mean" in metrics
    slot_pairwise_cosine = _as_float(metrics, "train_final_writer_slot_basis_pairwise_cosine_mean")
    return {
        "support_state_effective_rank": support_state_effective_rank,
        "writer_memory_slot_effective_rank": writer_memory_slot_effective_rank,
        "memory_long_common_mode_energy_ratio": common_mode_ratio,
        "slot_pairwise_cosine_present": slot_pairwise_cosine_present,
        "slot_pairwise_cosine": slot_pairwise_cosine,
        "source_not_collapsed_old": bool(
            support_state_effective_rank > 1.2
            or writer_memory_slot_effective_rank > 1.5
            or common_mode_ratio < 0.999
            or (
                slot_pairwise_cosine_present
                and slot_pairwise_cosine > 0.0
                and slot_pairwise_cosine < 0.95
            )
        ),
    }


def _strict_writer_memory_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    old_gate = _old_permissive_collapse_gate(metrics)
    writer_slots = max(1, int(round(_as_float(metrics, "writer_memory_slots", 8.0))))
    writer_rank = float(old_gate["writer_memory_slot_effective_rank"])
    projected_rank = _as_float(metrics, "projected_memory_effective_rank")
    common_mode_ratio = float(old_gate["memory_long_common_mode_energy_ratio"])
    slot_pairwise_present = bool(old_gate["slot_pairwise_cosine_present"])
    slot_pairwise_cosine = float(old_gate["slot_pairwise_cosine"])
    slot_norm_std = _as_float(
        metrics,
        "memory_long_slot_norm_std",
        _as_float(metrics, "writer_slot_norm_std"),
    )
    slot_norm_mean = _as_float(
        metrics,
        "memory_long_slot_norm_mean",
        _as_float(metrics, "writer_slot_norm_mean"),
    )
    thresholds = _strict_thresholds(metrics)
    writer_rank_fraction = float(writer_rank / max(float(writer_slots), 1.0))
    projector_rank_gain_factor = float(projected_rank / max(writer_rank, 1e-6))
    strict_metrics_present = bool(
        slot_pairwise_present
        and _all_finite(
            [
                writer_rank,
                projected_rank,
                common_mode_ratio,
                slot_pairwise_cosine,
                slot_norm_std,
                slot_norm_mean,
                writer_rank_fraction,
                projector_rank_gain_factor,
            ]
        )
    )
    writer_memory_not_collapsed_strict = bool(
        strict_metrics_present
        and writer_rank >= thresholds["rank_floor"]
        and writer_rank_fraction >= thresholds["rank_fraction_floor"]
        and common_mode_ratio <= thresholds["common_mode_ceiling"]
        and slot_pairwise_cosine <= thresholds["pairwise_cosine_ceiling"]
        and slot_norm_std > 1e-6
        and slot_norm_mean > 1e-6
    )
    return {
        **old_gate,
        "writer_family": _writer_family_from_metrics(metrics),
        "writer_memory_slots": writer_slots,
        "writer_rank_fraction": writer_rank_fraction,
        "projector_rank_gain_factor": projector_rank_gain_factor,
        "strict_rank_floor": thresholds["rank_floor"],
        "strict_rank_fraction_floor": thresholds["rank_fraction_floor"],
        "strict_common_mode_ceiling": thresholds["common_mode_ceiling"],
        "strict_pairwise_cosine_ceiling": thresholds["pairwise_cosine_ceiling"],
        "strict_slot_norm_std": slot_norm_std,
        "strict_slot_norm_mean": slot_norm_mean,
        "strict_metrics_present": strict_metrics_present,
        "writer_memory_not_collapsed_strict": writer_memory_not_collapsed_strict,
        "projector_manufactured_diversity": bool(
            projected_rank >= thresholds["rank_floor"]
            and not writer_memory_not_collapsed_strict
            and projector_rank_gain_factor > 4.0
        ),
    }


def _support_interface_alive(metrics: dict[str, Any]) -> bool:
    support_rank = _as_float(metrics, "train_final_support_state_effective_rank")
    if support_rank >= 1.75:
        return True
    attention_entropy = _as_float(metrics, "writer_support_attention_entropy_mean")
    coverage_entropy = _as_float(metrics, "writer_support_attention_item_coverage_entropy_mean")
    distinct_top_items = _as_float(metrics, "writer_support_attention_distinct_top_items_mean")
    return bool(
        attention_entropy >= 0.20
        and coverage_entropy >= 0.20
        and distinct_top_items >= 1.50
    )


def _replay_task_summary(
    *,
    control_metrics: dict[str, Any],
    branch_metrics: dict[str, Any],
    branch_train_events: list[dict[str, Any]],
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    task_score = _as_float(branch_metrics, "best_adapt_task_score")
    exact_match = _as_float(branch_metrics, "best_adapt_exact_match")
    control_score = _as_float(control_metrics, "best_adapt_task_score")
    control_exact_match = _as_float(control_metrics, "best_adapt_exact_match")
    task_metric_name = str(branch_metrics.get("task_metric_name", "accuracy"))
    mode = _task_mode(task_metric_name)
    prefix_attention_by_layer = _layer_metric(branch_metrics, "prefix_attention_mass_mean_by_layer")
    nontrivial_layer_count = _nontrivial_layer_count(prefix_attention_by_layer)
    post_unfreeze_start = _find_post_unfreeze_start(
        train_events=branch_train_events,
        writer_metrics=branch_metrics,
    )
    train_steps = int(_as_float(branch_metrics, "pilot_train_steps"))
    post_unfreeze_end = max(post_unfreeze_start, post_unfreeze_start + post_unfreeze_window - 1)
    tail_start = max(1, train_steps - tail_window + 1)
    loss_head = _as_float(branch_metrics, "train_loss_steps_1_50_median")
    loss_tail = _as_float(branch_metrics, "train_loss_steps_451_500_median")
    if not branch_train_events:
        loss_head = _as_float(branch_metrics, "train_loss_steps_1_50_median")
        loss_tail = _as_float(branch_metrics, "train_loss_tail_50_steps_median")
    writer_grad_post_unfreeze = _window_median(
        branch_train_events,
        key="grad_norm_writer",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    projector_grad_post_unfreeze = _window_median(
        branch_train_events,
        key="grad_norm_projector",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    receiver_grad_post_unfreeze = _window_median(
        branch_train_events,
        key="grad_norm_receiver_lora",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_grad_tail = _window_median(
        branch_train_events,
        key="grad_norm_writer",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    projector_grad_tail = _window_median(
        branch_train_events,
        key="grad_norm_projector",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    receiver_grad_tail = _window_median(
        branch_train_events,
        key="grad_norm_receiver_lora",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    writer_task_only_grad = _window_median(
        branch_train_events,
        key="grad_probe_writer_task_only_norm",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_aux_only_grad = _window_median(
        branch_train_events,
        key="grad_probe_writer_aux_only_norm",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_total_grad = _window_median(
        branch_train_events,
        key="grad_probe_writer_total_norm",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_task_aux_cosine = _window_median(
        branch_train_events,
        key="grad_probe_writer_task_aux_cosine",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_task_total_cosine = _window_median(
        branch_train_events,
        key="grad_probe_writer_task_total_cosine",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_aux_total_cosine = _window_median(
        branch_train_events,
        key="grad_probe_writer_aux_total_cosine",
        start_step=post_unfreeze_start,
        end_step=post_unfreeze_end,
    )
    writer_clip_fraction_tail = _window_fraction(
        branch_train_events,
        key="was_grad_clipped_writer",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    projector_clip_fraction_tail = _window_fraction(
        branch_train_events,
        key="was_grad_clipped_projector",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    receiver_clip_fraction_tail = _window_fraction(
        branch_train_events,
        key="was_grad_clipped_receiver_lora",
        start_step=tail_start,
        end_step=max(tail_start, train_steps),
    )
    loss_step_delta_tail = _median(
        [
            event_b - event_a
            for event_a, event_b in zip(
                _window_values(
                    branch_train_events,
                    key="loss",
                    start_step=tail_start,
                    end_step=max(tail_start, train_steps),
                )[:-1],
                _window_values(
                    branch_train_events,
                    key="loss",
                    start_step=tail_start,
                    end_step=max(tail_start, train_steps),
                )[1:],
                strict=False,
            )
        ]
    )
    control_rows = _load_case_rows_from_metrics(control_metrics)
    branch_rows = _load_case_rows_from_metrics(branch_metrics)
    delta_answer_logprob = _as_float(branch_metrics, "delta_answer_logprob")
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
                _as_float(branch_metrics, "prefix_attention_mass_mean"),
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
    writer_task_supervision_live = bool(writer_task_ratio >= 0.10 and writer_task_total_cosine > 0.0)
    writer_task_supervision_live_medium = bool(
        writer_task_ratio >= 0.20
        and writer_task_total_cosine >= 0.30
        and writer_task_aux_cosine > -0.20
    )
    strict_metrics = _strict_writer_memory_metrics(branch_metrics)
    support_interface_alive = _support_interface_alive(branch_metrics)
    prefix_growth_ratio = _snapshot_prefix_growth_ratio(branch_metrics)
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
            writer_rows=branch_rows,
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
            writer_rows=branch_rows,
        )
        usefulness_positive_v6 = bool(
            non_regressive_task
            and (
                usefulness_metrics["delta_answer_logprob_median"] > 0.0
                or usefulness_metrics["positive_delta_fraction"] > 0.5
            )
        )
    primary_task_improved = bool(
        str(branch_metrics.get("benchmark_id", "")).strip().lower() in PRIMARY_TASKS
        and (task_score - control_score) > 0.0
    )
    return {
        "task_name": str(branch_metrics.get("task_name", control_metrics.get("task_name", ""))),
        "benchmark_id": str(branch_metrics.get("benchmark_id", control_metrics.get("benchmark_id", ""))),
        "task_metric_name": task_metric_name,
        "task_mode": mode,
        "task_score": task_score,
        "exact_match": exact_match,
        "task_score_delta_vs_control": task_score - control_score,
        "exact_match_delta_vs_control": exact_match - control_exact_match,
        "delta_answer_logprob": delta_answer_logprob,
        "prefix_attention_mass_mean": _as_float(branch_metrics, "prefix_attention_mass_mean"),
        "prefix_attention_mass_mean_by_layer": prefix_attention_by_layer,
        "prefix_attention_nontrivial_layer_count": nontrivial_layer_count,
        "projected_memory_effective_rank": _as_float(branch_metrics, "projected_memory_effective_rank"),
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
        "support_interface_alive": support_interface_alive,
        "stable_training_v6": stable_training_v6,
        "usefulness_positive_v6": usefulness_positive_v6,
        "primary_task_improved": primary_task_improved,
        **strict_metrics,
        **usefulness_metrics,
    }


def _oracle_task_summary(
    *,
    control_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
) -> dict[str, Any]:
    task_score = _as_float(oracle_metrics, "best_adapt_task_score")
    exact_match = _as_float(oracle_metrics, "best_adapt_exact_match")
    control_score = _as_float(control_metrics, "best_adapt_task_score")
    control_exact_match = _as_float(control_metrics, "best_adapt_exact_match")
    strict_metrics = _strict_writer_memory_metrics(oracle_metrics)
    return {
        "task_name": str(oracle_metrics.get("task_name", control_metrics.get("task_name", ""))),
        "benchmark_id": str(oracle_metrics.get("benchmark_id", control_metrics.get("benchmark_id", ""))),
        "task_metric_name": str(oracle_metrics.get("task_metric_name", control_metrics.get("task_metric_name", ""))),
        "task_score": task_score,
        "exact_match": exact_match,
        "task_score_delta_vs_control": task_score - control_score,
        "exact_match_delta_vs_control": exact_match - control_exact_match,
        "delta_answer_logprob": _as_float(oracle_metrics, "delta_answer_logprob"),
        "prefix_attention_mass_mean": _as_float(oracle_metrics, "prefix_attention_mass_mean"),
        "prefix_attention_mass_mean_by_layer": _layer_metric(oracle_metrics, "prefix_attention_mass_mean_by_layer"),
        "prefix_attention_nontrivial_layer_count": _nontrivial_layer_count(
            _layer_metric(oracle_metrics, "prefix_attention_mass_mean_by_layer")
        ),
        "primary_task_improved": bool(
            str(oracle_metrics.get("benchmark_id", "")).strip().lower() in PRIMARY_TASKS
            and (task_score - control_score) > 0.0
        ),
        **strict_metrics,
    }


def _arm_primary_score_delta_sum(task_summaries: dict[str, dict[str, Any]]) -> float:
    return float(
        sum(
            float(summary["task_score_delta_vs_control"])
            for task_name, summary in task_summaries.items()
            if task_name in PRIMARY_TASKS
        )
    )


def _replay_arm_summary(task_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary_tasks = [summary for task_name, summary in task_summaries.items() if task_name in PRIMARY_TASKS]
    primary_usefulness_positive = any(bool(summary["primary_task_improved"]) for summary in primary_tasks)
    primary_branch_success = any(
        bool(summary["route_live_post_unfreeze"])
        and bool(summary["stable_training_v6"])
        and bool(summary["writer_task_supervision_live_medium"])
        and bool(summary["writer_memory_not_collapsed_strict"])
        and bool(summary["primary_task_improved"])
        for summary in primary_tasks
    )
    return {
        "primary_task_score_delta_sum": _arm_primary_score_delta_sum(task_summaries),
        "old_any_source_not_collapsed": any(
            bool(summary["source_not_collapsed_old"]) for summary in task_summaries.values()
        ),
        "strict_any_writer_memory_not_collapsed": any(
            bool(summary["writer_memory_not_collapsed_strict"]) for summary in task_summaries.values()
        ),
        "any_projector_manufactured_diversity": any(
            bool(summary["projector_manufactured_diversity"]) for summary in task_summaries.values()
        ),
        "any_route_live_post_unfreeze": any(
            bool(summary["route_live_post_unfreeze"]) for summary in task_summaries.values()
        ),
        "any_stable_training_v6": any(
            bool(summary["stable_training_v6"]) for summary in task_summaries.values()
        ),
        "any_writer_task_supervision_live_medium": any(
            bool(summary["writer_task_supervision_live_medium"]) for summary in task_summaries.values()
        ),
        "primary_usefulness_positive": primary_usefulness_positive,
        "primary_branch_success": primary_branch_success,
    }


def _oracle_arm_summary(task_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "primary_task_score_delta_sum": _arm_primary_score_delta_sum(task_summaries),
        "any_primary_task_improved": any(
            bool(summary["primary_task_improved"]) for summary in task_summaries.values()
        ),
        "strict_any_memory_not_collapsed": any(
            bool(summary["writer_memory_not_collapsed_strict"]) for summary in task_summaries.values()
        ),
    }


def _compare_depth(
    *,
    early_summary: dict[str, Any],
    mid_summary: dict[str, Any],
) -> dict[str, Any]:
    early_total = float(early_summary["primary_task_score_delta_sum"])
    mid_total = float(mid_summary["primary_task_score_delta_sum"])
    return {
        "early_primary_task_score_delta_sum": early_total,
        "mid_primary_task_score_delta_sum": mid_total,
        "mid_minus_early_primary_task_score_delta_sum": mid_total - early_total,
        "mid_beats_early": bool(mid_total > early_total + 1e-12),
    }


def _load_metrics_tree(result_root: Path) -> dict[str, dict[str, Any]]:
    metrics_tree: dict[str, dict[str, Any]] = {}
    for arm_name in ("control", *REPLAY_ARMS, *ORACLE_ARMS):
        arm_root = result_root / arm_name
        task_payloads: dict[str, Any] = {}
        if not arm_root.exists():
            continue
        for task_dir in sorted(path for path in arm_root.iterdir() if path.is_dir()):
            task_payloads[task_dir.name] = {
                "metrics": _load_json(task_dir / "metrics.json"),
                "train_events": _load_train_events(task_dir / "train_events.json"),
            }
        metrics_tree[arm_name] = task_payloads
    return metrics_tree


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize PLANv7 V7-0 continuity replay and oracle metrics."
    )
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--head_window", type=int, default=50)
    parser.add_argument("--post_unfreeze_window", type=int, default=50)
    parser.add_argument("--tail_window", type=int, default=50)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result_root = Path(args.result_root).resolve()
    metrics_tree = _load_metrics_tree(result_root)
    control_metrics = metrics_tree["control"]
    replay_summaries: dict[str, dict[str, Any]] = {}
    for arm_name in REPLAY_ARMS:
        task_summaries: dict[str, dict[str, Any]] = {}
        for task_name in ("gsm8k", "triviaqa", "fever"):
            task_summaries[task_name] = _replay_task_summary(
                control_metrics=control_metrics[task_name]["metrics"],
                branch_metrics=metrics_tree[arm_name][task_name]["metrics"],
                branch_train_events=metrics_tree[arm_name][task_name]["train_events"],
                head_window=args.head_window,
                post_unfreeze_window=args.post_unfreeze_window,
                tail_window=args.tail_window,
            )
        replay_summaries[arm_name] = {
            **_replay_arm_summary(task_summaries),
            "tasks": task_summaries,
        }
    oracle_summaries: dict[str, dict[str, Any]] = {}
    for arm_name in ORACLE_ARMS:
        task_summaries: dict[str, dict[str, Any]] = {}
        for task_name in PRIMARY_TASKS:
            task_summaries[task_name] = _oracle_task_summary(
                control_metrics=control_metrics[task_name]["metrics"],
                oracle_metrics=metrics_tree[arm_name][task_name]["metrics"],
            )
        oracle_summaries[arm_name] = {
            **_oracle_arm_summary(task_summaries),
            "tasks": task_summaries,
        }
    early_vs_mid_baseline = _compare_depth(
        early_summary=replay_summaries["c_early"],
        mid_summary=replay_summaries["c_mid"],
    )
    early_vs_mid_oracle = {
        "context_echo": _compare_depth(
            early_summary=oracle_summaries["o_ctx_early"],
            mid_summary=oracle_summaries["o_ctx_mid"],
        ),
        "support_echo": _compare_depth(
            early_summary=oracle_summaries["o_sup_early"],
            mid_summary=oracle_summaries["o_sup_mid"],
        ),
    }
    additive_continuity = {
        "primary_task_score_delta_sum": replay_summaries["c_add"]["primary_task_score_delta_sum"],
        "beats_early": bool(
            replay_summaries["c_add"]["primary_task_score_delta_sum"]
            > replay_summaries["c_early"]["primary_task_score_delta_sum"] + 1e-12
        ),
        "beats_mid": bool(
            replay_summaries["c_add"]["primary_task_score_delta_sum"]
            > replay_summaries["c_mid"]["primary_task_score_delta_sum"] + 1e-12
        ),
    }
    owner_metadata = {
        "owner_locked_projector_lr": _as_float(
            control_metrics["gsm8k"]["metrics"], "owner_locked_projector_lr", 7.5e-6
        ),
        "repo_confirmed_v65_projector_lr_reference": _as_float(
            control_metrics["gsm8k"]["metrics"], "repo_confirmed_v65_projector_lr_reference", 7.5e-5
        ),
        "owner_override_note": bool(
            control_metrics["gsm8k"]["metrics"].get("owner_override_note", True)
        ),
    }
    any_mid_oracle_beats_early = bool(
        early_vs_mid_oracle["context_echo"]["mid_beats_early"]
        or early_vs_mid_oracle["support_echo"]["mid_beats_early"]
    )
    all_oracles_flat = all(
        abs(float(task_summary["task_score_delta_vs_control"])) <= 1e-12
        for arm_summary in oracle_summaries.values()
        for task_summary in arm_summary["tasks"].values()
    )
    if any_mid_oracle_beats_early:
        comparison_conclusion = "prefer_mid4_mainline"
        recommended_next_step = "open_v7_1_width_depth_scout_mid4"
        preferred_depth = "mid4"
    elif all_oracles_flat:
        comparison_conclusion = "oracle_flat_direct_injection_high_risk"
        recommended_next_step = "open_v7_1_width_depth_scout_keep_bridge_ready"
        preferred_depth = "indeterminate"
    else:
        comparison_conclusion = "carry_depth_signal_into_v7_1"
        recommended_next_step = "open_v7_1_width_depth_scout"
        preferred_depth = (
            "early4"
            if replay_summaries["c_early"]["primary_task_score_delta_sum"]
            > replay_summaries["c_mid"]["primary_task_score_delta_sum"] + 1e-12
            else "indeterminate"
        )
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "preferred_depth": preferred_depth,
        "owner_lr_discrepancy_metadata": owner_metadata,
        "baseline_replay": replay_summaries,
        "oracle_arms": oracle_summaries,
        "early_vs_mid_baseline": early_vs_mid_baseline,
        "early_vs_mid_oracle": early_vs_mid_oracle,
        "additive_continuity": additive_continuity,
        "all_oracles_flat_on_primary_tasks": all_oracles_flat,
        "any_mid_oracle_beats_early": any_mid_oracle_beats_early,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# PLANv7 V7-0 Metrics And Oracle Summary",
        "",
        f"- comparison_conclusion: {comparison_conclusion}",
        f"- recommended_next_step: {recommended_next_step}",
        f"- preferred_depth: {preferred_depth}",
        f"- any_mid_oracle_beats_early: {any_mid_oracle_beats_early}",
        f"- all_oracles_flat_on_primary_tasks: {all_oracles_flat}",
        "",
        "## Owner LR Metadata",
        f"- owner_locked_projector_lr: {owner_metadata['owner_locked_projector_lr']}",
        f"- repo_confirmed_v65_projector_lr_reference: {owner_metadata['repo_confirmed_v65_projector_lr_reference']}",
        f"- owner_override_note: {owner_metadata['owner_override_note']}",
        "",
        "## Early Vs Mid Baseline",
        f"- early_primary_task_score_delta_sum: {early_vs_mid_baseline['early_primary_task_score_delta_sum']:.6f}",
        f"- mid_primary_task_score_delta_sum: {early_vs_mid_baseline['mid_primary_task_score_delta_sum']:.6f}",
        f"- mid_beats_early: {early_vs_mid_baseline['mid_beats_early']}",
        "",
        "## Early Vs Mid Oracle",
        f"- context_echo_mid_beats_early: {early_vs_mid_oracle['context_echo']['mid_beats_early']}",
        f"- support_echo_mid_beats_early: {early_vs_mid_oracle['support_echo']['mid_beats_early']}",
        "",
        "## Additive Continuity",
        f"- primary_task_score_delta_sum: {additive_continuity['primary_task_score_delta_sum']:.6f}",
        f"- beats_early: {additive_continuity['beats_early']}",
        f"- beats_mid: {additive_continuity['beats_mid']}",
    ]
    for arm_name in REPLAY_ARMS:
        arm_summary = replay_summaries[arm_name]
        report_lines.extend(
            [
                "",
                f"## {arm_name}",
                f"- primary_task_score_delta_sum: {arm_summary['primary_task_score_delta_sum']:.6f}",
                f"- old_any_source_not_collapsed: {arm_summary['old_any_source_not_collapsed']}",
                f"- strict_any_writer_memory_not_collapsed: {arm_summary['strict_any_writer_memory_not_collapsed']}",
                f"- any_projector_manufactured_diversity: {arm_summary['any_projector_manufactured_diversity']}",
                f"- primary_usefulness_positive: {arm_summary['primary_usefulness_positive']}",
                f"- primary_branch_success: {arm_summary['primary_branch_success']}",
            ]
        )
        for task_name in ("gsm8k", "triviaqa", "fever"):
            task_summary = arm_summary["tasks"][task_name]
            report_lines.extend(
                [
                    f"- {task_name}.task_score_delta_vs_control: {task_summary['task_score_delta_vs_control']:.6f}",
                    f"- {task_name}.source_not_collapsed_old: {task_summary['source_not_collapsed_old']}",
                    f"- {task_name}.writer_memory_not_collapsed_strict: {task_summary['writer_memory_not_collapsed_strict']}",
                    f"- {task_name}.projector_manufactured_diversity: {task_summary['projector_manufactured_diversity']}",
                ]
            )
    for arm_name in ORACLE_ARMS:
        arm_summary = oracle_summaries[arm_name]
        report_lines.extend(
            [
                "",
                f"## {arm_name}",
                f"- primary_task_score_delta_sum: {arm_summary['primary_task_score_delta_sum']:.6f}",
                f"- any_primary_task_improved: {arm_summary['any_primary_task_improved']}",
            ]
        )
        for task_name in PRIMARY_TASKS:
            task_summary = arm_summary["tasks"][task_name]
            report_lines.extend(
                [
                    f"- {task_name}.task_score_delta_vs_control: {task_summary['task_score_delta_vs_control']:.6f}",
                    f"- {task_name}.delta_answer_logprob: {task_summary['delta_answer_logprob']:.6f}",
                    f"- {task_name}.prefix_attention_mass_mean: {task_summary['prefix_attention_mass_mean']:.6f}",
                ]
            )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
