#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
ARM_ORDER = ("s00", "s01", "s10", "s11")
ARM_METADATA = {
    "s00": {"writer_family": "W0", "depth_family": "D0", "projector_family": "P0"},
    "s01": {"writer_family": "W0", "depth_family": "D1", "projector_family": "P0"},
    "s10": {"writer_family": "W1", "depth_family": "D0", "projector_family": "P1"},
    "s11": {"writer_family": "W1", "depth_family": "D1", "projector_family": "P1"},
}
DEPTH_LABELS = {"d0": "early4", "d1": "mid4"}


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
        if line.strip():
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
        "answer_switch_rate": float(answer_switch_count / max(1, len(shared_ids))),
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
    writer_task_ratio = float(writer_task_only_grad / max(writer_total_grad, 1e-8))
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
        primary_usefulness_positive = bool(
            (task_score - control_score) > 0.0
            and non_regressive_task
            and (
                usefulness_metrics["margin_delta_mean"] > 0.0
                or usefulness_metrics["positive_margin_shift_fraction"] > 0.5
                or usefulness_metrics["correct_flip_count"] > 0.0
            )
        )
        helpfulness_score = float(
            usefulness_metrics["margin_delta_mean"]
            + usefulness_metrics["positive_margin_shift_fraction"]
            + usefulness_metrics["correct_flip_count"]
        )
    else:
        usefulness_metrics = _generation_usefulness_metrics(
            control_rows=control_rows,
            writer_rows=branch_rows,
        )
        primary_usefulness_positive = bool(
            (task_score - control_score) > 0.0
            and non_regressive_task
            and (
                usefulness_metrics["delta_answer_logprob_median"] > 0.0
                or usefulness_metrics["positive_delta_fraction"] > 0.5
            )
        )
        helpfulness_score = float(
            usefulness_metrics["delta_answer_logprob_mean"]
            + usefulness_metrics["positive_delta_fraction"]
        )
    primary_branch_success = bool(
        route_live_post_unfreeze
        and stable_training_v6
        and writer_task_supervision_live_medium
        and strict_metrics["writer_memory_not_collapsed_strict"]
        and primary_usefulness_positive
    )
    return {
        "task_name": str(branch_metrics.get("task_name", control_metrics.get("task_name", ""))),
        "benchmark_id": str(branch_metrics.get("benchmark_id", control_metrics.get("benchmark_id", ""))),
        "task_metric_name": task_metric_name,
        "task_mode": mode,
        "pilot_bridge_mode": str(branch_metrics.get("pilot_bridge_mode", "writer_direct")),
        "pilot_memory_path_variant": str(branch_metrics.get("pilot_memory_path_variant", "single_level")),
        "pilot_deep_prefix_layers": branch_metrics.get("pilot_deep_prefix_layers", []),
        "pilot_receiver_lora_target_layers": branch_metrics.get("pilot_receiver_lora_target_layers", []),
        "pilot_deep_prefix_rank": int(round(_as_float(branch_metrics, "pilot_deep_prefix_rank", 0.0))),
        "pilot_writer_conditioning_layers": int(
            round(_as_float(branch_metrics, "pilot_writer_conditioning_layers", 0.0))
        ),
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
        "receiver_lora_grad_norm_post_unfreeze_median": receiver_grad_post_unfreeze,
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
        "writer_task_supervision_live_medium": writer_task_supervision_live_medium,
        "support_interface_alive": support_interface_alive,
        "stable_training_v6": stable_training_v6,
        "primary_usefulness_positive": primary_usefulness_positive,
        "primary_branch_success": primary_branch_success,
        "first_answer_token_or_switch_helpfulness": helpfulness_score,
        **strict_metrics,
        **usefulness_metrics,
    }


def _load_metrics_tree(result_root: Path) -> dict[str, dict[str, Any]]:
    metrics_tree: dict[str, dict[str, Any]] = {}
    for arm_name in ("control", *ARM_ORDER):
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


def _arm_summary(
    *,
    arm_id: str,
    task_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gsm8k = task_summaries["gsm8k"]
    triviaqa = task_summaries["triviaqa"]
    strict_count = int(sum(bool(task["writer_memory_not_collapsed_strict"]) for task in task_summaries.values()))
    primary_success_count = int(sum(bool(task["primary_branch_success"]) for task in task_summaries.values()))
    stability_count = int(sum(bool(task["stable_training_v6"]) for task in task_summaries.values()))
    route_count = int(sum(bool(task["route_live_post_unfreeze"]) for task in task_summaries.values()))
    helpfulness_score = float(
        sum(float(task["first_answer_token_or_switch_helpfulness"]) for task in task_summaries.values())
    )
    strict_rank_fraction_mean = float(
        sum(float(task["writer_rank_fraction"]) for task in task_summaries.values()) / len(task_summaries)
    )
    projector_rank_gain_factor_mean = float(
        sum(float(task["projector_rank_gain_factor"]) for task in task_summaries.values()) / len(task_summaries)
    )
    penalties = {
        "projector_manufactured_diversity": any(
            bool(task["projector_manufactured_diversity"]) for task in task_summaries.values()
        ),
        "both_primary_flat": all(
            abs(float(task["task_score_delta_vs_control"])) <= 1e-12 for task in task_summaries.values()
        ),
        "writer_rank_near_one": any(
            float(task["writer_memory_slot_effective_rank"]) < 1.2 for task in task_summaries.values()
        ),
        "common_mode_near_one": any(
            float(task["memory_long_common_mode_energy_ratio"]) > 0.999 for task in task_summaries.values()
        ),
        "task_aux_conflict": any(
            float(task["writer_task_aux_cosine_post_unfreeze_median"]) < -0.20 for task in task_summaries.values()
        ),
    }
    penalty_count = int(sum(int(value) for value in penalties.values()))
    metadata = ARM_METADATA[arm_id]
    metrics_ref = task_summaries["gsm8k"]
    return {
        "arm_id": arm_id,
        "writer_family": metadata["writer_family"],
        "depth_family": metadata["depth_family"],
        "projector_family": metadata["projector_family"],
        "projector_mode": "shared_low_rank",
        "bridge_mode": str(metrics_ref.get("pilot_bridge_mode", "writer_direct")),
        "memory_path_variant": str(metrics_ref.get("pilot_memory_path_variant", "single_level")),
        "active_depth_layers": metrics_ref.get("pilot_deep_prefix_layers", []),
        "receiver_lora_layers": metrics_ref.get("pilot_receiver_lora_target_layers", []),
        "projector_rank": int(round(float(metrics_ref.get("pilot_deep_prefix_rank", 0.0)))),
        "writer_memory_slots": int(round(float(metrics_ref.get("writer_memory_slots", 0.0)))),
        "writer_conditioning_layers": int(round(float(metrics_ref.get("pilot_writer_conditioning_layers", 0.0)))),
        "gsm8k_task_score_delta_vs_control": float(gsm8k["task_score_delta_vs_control"]),
        "triviaqa_task_score_delta_vs_control": float(triviaqa["task_score_delta_vs_control"]),
        "primary_task_score_delta_sum": float(
            gsm8k["task_score_delta_vs_control"] + triviaqa["task_score_delta_vs_control"]
        ),
        "strict_writer_memory_task_count": strict_count,
        "strict_rank_fraction_mean": strict_rank_fraction_mean,
        "projector_rank_gain_factor_mean": projector_rank_gain_factor_mean,
        "route_live_task_count": route_count,
        "stable_training_task_count": stability_count,
        "primary_branch_success_task_count": primary_success_count,
        "any_primary_usefulness_positive": any(
            bool(task["primary_usefulness_positive"]) for task in task_summaries.values()
        ),
        "answer_switch_helpfulness_score": helpfulness_score,
        "ranking_penalties": penalties,
        "ranking_penalty_count": penalty_count,
        "ranking_key": [
            float(gsm8k["task_score_delta_vs_control"]),
            float(triviaqa["task_score_delta_vs_control"]),
            float(strict_count),
            strict_rank_fraction_mean,
            helpfulness_score,
            float(stability_count),
            float(-penalty_count),
        ],
        "tasks": task_summaries,
    }


def _depth_summary(
    *,
    depth_family: str,
    arm_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    depth_arm_ids = [arm_id for arm_id, payload in arm_summaries.items() if payload["depth_family"] == depth_family]
    depth_arms = [arm_summaries[arm_id] for arm_id in depth_arm_ids]
    gsm_deltas = [float(arm["gsm8k_task_score_delta_vs_control"]) for arm in depth_arms]
    trivia_deltas = [float(arm["triviaqa_task_score_delta_vs_control"]) for arm in depth_arms]
    strict_counts = [int(arm["strict_writer_memory_task_count"]) for arm in depth_arms]
    helpfulness_scores = [float(arm["answer_switch_helpfulness_score"]) for arm in depth_arms]
    stability_counts = [int(arm["stable_training_task_count"]) for arm in depth_arms]
    penalty_counts = [int(arm["ranking_penalty_count"]) for arm in depth_arms]
    strict_rank_fraction_means = [float(arm["strict_rank_fraction_mean"]) for arm in depth_arms]
    return {
        "depth_family": depth_family,
        "depth_label": DEPTH_LABELS[depth_family.lower()],
        "arm_ids": depth_arm_ids,
        "gsm8k_task_score_delta_mean": float(sum(gsm_deltas) / max(1, len(gsm_deltas))),
        "triviaqa_task_score_delta_mean": float(sum(trivia_deltas) / max(1, len(trivia_deltas))),
        "primary_task_score_delta_sum": float(sum(gsm_deltas) + sum(trivia_deltas)),
        "strict_writer_memory_task_count_total": int(sum(strict_counts)),
        "strict_rank_fraction_mean": float(sum(strict_rank_fraction_means) / max(1, len(strict_rank_fraction_means))),
        "answer_switch_helpfulness_score_mean": float(sum(helpfulness_scores) / max(1, len(helpfulness_scores))),
        "stable_training_task_count_total": int(sum(stability_counts)),
        "ranking_penalty_count_total": int(sum(penalty_counts)),
        "ranking_key": [
            float(sum(gsm_deltas) / max(1, len(gsm_deltas))),
            float(sum(trivia_deltas) / max(1, len(trivia_deltas))),
            float(sum(strict_counts)),
            float(sum(strict_rank_fraction_means) / max(1, len(strict_rank_fraction_means))),
            float(sum(helpfulness_scores) / max(1, len(helpfulness_scores))),
            float(sum(stability_counts)),
            float(-sum(penalty_counts)),
        ],
    }


def _select_winning_depth(
    *,
    d0: dict[str, Any],
    d1: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    d0_gsm = float(d0["gsm8k_task_score_delta_mean"])
    d1_gsm = float(d1["gsm8k_task_score_delta_mean"])
    d0_trivia = float(d0["triviaqa_task_score_delta_mean"])
    d1_trivia = float(d1["triviaqa_task_score_delta_mean"])
    d0_strict = int(d0["strict_writer_memory_task_count_total"])
    d1_strict = int(d1["strict_writer_memory_task_count_total"])
    d0_rank_frac = float(d0["strict_rank_fraction_mean"])
    d1_rank_frac = float(d1["strict_rank_fraction_mean"])
    split_by_task = bool((d0_gsm - d1_gsm) * (d0_trivia - d1_trivia) < 0.0)
    if split_by_task:
        trivia_winner = "D0" if d0_trivia > d1_trivia + 1e-12 else "D1"
        if trivia_winner == "D0":
            trivia_strict_better = bool(
                d0_strict > d1_strict or d0_rank_frac > d1_rank_frac + 0.02
            )
            winner = "D0" if trivia_strict_better else ("D0" if d0_gsm > d1_gsm + 1e-12 else "D1")
        else:
            trivia_strict_better = bool(
                d1_strict > d0_strict or d1_rank_frac > d0_rank_frac + 0.02
            )
            winner = "D1" if trivia_strict_better else ("D1" if d1_gsm > d0_gsm + 1e-12 else "D0")
    elif d1_gsm >= d0_gsm - 1e-12 and d1_trivia >= d0_trivia - 1e-12 and (
        d1_strict > d0_strict or d1_rank_frac > d0_rank_frac + 1e-6
    ):
        winner = "D1"
    elif d0_gsm > d1_gsm + 1e-12 and d0_trivia > d1_trivia + 1e-12 and (
        d0_strict > d1_strict or d0_rank_frac > d1_rank_frac + 1e-6
    ):
        winner = "D0"
    else:
        winner = "D1" if tuple(d1["ranking_key"]) > tuple(d0["ranking_key"]) else "D0"
    comparison = {
        "split_by_task": split_by_task,
        "d0_ranking_key": d0["ranking_key"],
        "d1_ranking_key": d1["ranking_key"],
        "d1_tied_or_better_on_primary": bool(d1_gsm >= d0_gsm - 1e-12 and d1_trivia >= d0_trivia - 1e-12),
        "d1_clearly_better_writer_metrics": bool(d1_strict > d0_strict or d1_rank_frac > d0_rank_frac + 0.02),
    }
    return winner, comparison


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv7 V7-1 width-depth scout.")
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
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        task_summaries: dict[str, dict[str, Any]] = {}
        for task_name in PRIMARY_TASKS:
            task_summaries[task_name] = _replay_task_summary(
                control_metrics=control_metrics[task_name]["metrics"],
                branch_metrics=metrics_tree[arm_id][task_name]["metrics"],
                branch_train_events=metrics_tree[arm_id][task_name]["train_events"],
                head_window=args.head_window,
                post_unfreeze_window=args.post_unfreeze_window,
                tail_window=args.tail_window,
            )
        arm_summaries[arm_id] = _arm_summary(arm_id=arm_id, task_summaries=task_summaries)

    depth_summaries = {
        "D0": _depth_summary(depth_family="D0", arm_summaries=arm_summaries),
        "D1": _depth_summary(depth_family="D1", arm_summaries=arm_summaries),
    }
    winning_depth, depth_comparison = _select_winning_depth(
        d0=depth_summaries["D0"],
        d1=depth_summaries["D1"],
    )
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
    winning_depth_label = DEPTH_LABELS[winning_depth.lower()]
    comparison_conclusion = (
        "select_mid4_for_v7_2" if winning_depth == "D1" else "select_early4_for_v7_2"
    )
    recommended_next_step = (
        "open_v7_2_direct_bandwidth_mid4" if winning_depth == "D1" else "open_v7_2_direct_bandwidth_early4"
    )
    arm_ranking = sorted(
        (
            {
                "arm_id": arm_id,
                "writer_family": summary["writer_family"],
                "depth_family": summary["depth_family"],
                "projector_family": summary["projector_family"],
                "ranking_key": summary["ranking_key"],
            }
            for arm_id, summary in arm_summaries.items()
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "winning_depth": winning_depth,
        "winning_depth_label": winning_depth_label,
        "owner_lr_discrepancy_metadata": owner_metadata,
        "arms": arm_summaries,
        "arm_ranking": arm_ranking,
        "depth_ranking": depth_summaries,
        "depth_comparison": depth_comparison,
        "acceptance": {
            "all_four_scout_arms_complete": len(arm_summaries) == 4,
            "strict_gate_ranking_computed": True,
            "single_winning_depth_selected": winning_depth in {"D0", "D1"},
            "fever_not_used_to_override_primary": True,
        },
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# PLANv7 V7-1 Width Depth Scout Summary",
        "",
        f"- comparison_conclusion: {comparison_conclusion}",
        f"- recommended_next_step: {recommended_next_step}",
        f"- winning_depth: {winning_depth}",
        f"- winning_depth_label: {winning_depth_label}",
        f"- fever_not_used_to_override_primary: True",
        "",
        "## Owner LR Metadata",
        f"- owner_locked_projector_lr: {owner_metadata['owner_locked_projector_lr']}",
        f"- repo_confirmed_v65_projector_lr_reference: {owner_metadata['repo_confirmed_v65_projector_lr_reference']}",
        f"- owner_override_note: {owner_metadata['owner_override_note']}",
        "",
        "## Depth Comparison",
        f"- split_by_task: {depth_comparison['split_by_task']}",
        f"- d0_ranking_key: {depth_comparison['d0_ranking_key']}",
        f"- d1_ranking_key: {depth_comparison['d1_ranking_key']}",
        f"- d1_tied_or_better_on_primary: {depth_comparison['d1_tied_or_better_on_primary']}",
        f"- d1_clearly_better_writer_metrics: {depth_comparison['d1_clearly_better_writer_metrics']}",
    ]
    for depth_id in ("D0", "D1"):
        depth_summary = depth_summaries[depth_id]
        report_lines.extend(
            [
                "",
                f"## {depth_id}",
                f"- depth_label: {depth_summary['depth_label']}",
                f"- arm_ids: {', '.join(depth_summary['arm_ids'])}",
                f"- gsm8k_task_score_delta_mean: {depth_summary['gsm8k_task_score_delta_mean']:.6f}",
                f"- triviaqa_task_score_delta_mean: {depth_summary['triviaqa_task_score_delta_mean']:.6f}",
                f"- primary_task_score_delta_sum: {depth_summary['primary_task_score_delta_sum']:.6f}",
                f"- strict_writer_memory_task_count_total: {depth_summary['strict_writer_memory_task_count_total']}",
                f"- strict_rank_fraction_mean: {depth_summary['strict_rank_fraction_mean']:.6f}",
                f"- answer_switch_helpfulness_score_mean: {depth_summary['answer_switch_helpfulness_score_mean']:.6f}",
                f"- stable_training_task_count_total: {depth_summary['stable_training_task_count_total']}",
                f"- ranking_penalty_count_total: {depth_summary['ranking_penalty_count_total']}",
            ]
        )
    for arm_id in ARM_ORDER:
        arm_summary = arm_summaries[arm_id]
        report_lines.extend(
            [
                "",
                f"## {arm_id}",
                f"- writer_family: {arm_summary['writer_family']}",
                f"- depth_family: {arm_summary['depth_family']}",
                f"- projector_family: {arm_summary['projector_family']}",
                f"- projector_mode: {arm_summary['projector_mode']}",
                f"- bridge_mode: {arm_summary['bridge_mode']}",
                f"- memory_path_variant: {arm_summary['memory_path_variant']}",
                f"- active_depth_layers: {arm_summary['active_depth_layers']}",
                f"- projector_rank: {arm_summary['projector_rank']}",
                f"- writer_memory_slots: {arm_summary['writer_memory_slots']}",
                f"- writer_conditioning_layers: {arm_summary['writer_conditioning_layers']}",
                f"- gsm8k_task_score_delta_vs_control: {arm_summary['gsm8k_task_score_delta_vs_control']:.6f}",
                f"- triviaqa_task_score_delta_vs_control: {arm_summary['triviaqa_task_score_delta_vs_control']:.6f}",
                f"- primary_task_score_delta_sum: {arm_summary['primary_task_score_delta_sum']:.6f}",
                f"- strict_writer_memory_task_count: {arm_summary['strict_writer_memory_task_count']}",
                f"- strict_rank_fraction_mean: {arm_summary['strict_rank_fraction_mean']:.6f}",
                f"- answer_switch_helpfulness_score: {arm_summary['answer_switch_helpfulness_score']:.6f}",
                f"- stable_training_task_count: {arm_summary['stable_training_task_count']}",
                f"- ranking_penalty_count: {arm_summary['ranking_penalty_count']}",
            ]
        )
        for task_name in PRIMARY_TASKS:
            task_summary = arm_summary["tasks"][task_name]
            report_lines.extend(
                [
                    f"- {task_name}.task_score_delta_vs_control: {task_summary['task_score_delta_vs_control']:.6f}",
                    f"- {task_name}.writer_memory_not_collapsed_strict: {task_summary['writer_memory_not_collapsed_strict']}",
                    f"- {task_name}.primary_usefulness_positive: {task_summary['primary_usefulness_positive']}",
                    f"- {task_name}.primary_branch_success: {task_summary['primary_branch_success']}",
                    f"- {task_name}.route_live_post_unfreeze: {task_summary['route_live_post_unfreeze']}",
                    f"- {task_name}.stable_training_v6: {task_summary['stable_training_v6']}",
                    f"- {task_name}.projector_manufactured_diversity: {task_summary['projector_manufactured_diversity']}",
                ]
            )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
