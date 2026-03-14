#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
ALL_TASKS = ("gsm8k", "triviaqa", "fever")
CONTROL_ARM_ID = "a0_none"
ARM_ORDER = (
    "a0_none",
    "a1_barlow",
    "a2_recon_bow",
    "a3_writer_opd_ans",
    "a4_writer_opd_ansctx",
    "a5_writer_opd_plus_recon",
)
AUX_ARMS = ARM_ORDER[1:]
ARM_METADATA = {
    "a0_none": {"auxiliary_family": "none"},
    "a1_barlow": {"auxiliary_family": "barlow_lite"},
    "a2_recon_bow": {"auxiliary_family": "reconstruction_lite"},
    "a3_writer_opd_ans": {"auxiliary_family": "writer_opd_answer_only"},
    "a4_writer_opd_ansctx": {"auxiliary_family": "writer_opd_answer_plus_context"},
    "a5_writer_opd_plus_recon": {"auxiliary_family": "writer_opd_plus_reconstruction"},
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _load_train_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        events = payload.get("events", [])
        if isinstance(events, list):
            return [dict(row) for row in events]
    return []


def _load_case_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _task_score(metrics: dict[str, Any]) -> float:
    for key in ("best_adapt_task_score", "task_score", "accuracy", "exact_match"):
        if key in metrics:
            return _safe_float(metrics.get(key))
    return 0.0


def _macro_f1(metrics: dict[str, Any]) -> float:
    for key in ("best_adapt_macro_f1", "macro_f1", "exact_match"):
        if key in metrics:
            return _safe_float(metrics.get(key))
    return 0.0


def _mean_row_value(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        _safe_float(row.get(key))
        for row in rows
        if math.isfinite(_safe_float(row.get(key), float("nan")))
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _row_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", "")).strip()
        if example_id:
            indexed[example_id] = row
    return indexed


def _shared_row_delta(
    control_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
    *,
    key: str,
) -> tuple[float, float]:
    indexed_control = _row_index(control_rows)
    indexed_branch = _row_index(branch_rows)
    shared_ids = sorted(set(indexed_control) & set(indexed_branch))
    if not shared_ids:
        return 0.0, 0.0
    deltas: list[float] = []
    positive_count = 0
    for example_id in shared_ids:
        delta = _safe_float(indexed_branch[example_id].get(key)) - _safe_float(
            indexed_control[example_id].get(key)
        )
        if math.isfinite(delta):
            deltas.append(delta)
            if delta > 0.0:
                positive_count += 1
    if not deltas:
        return 0.0, 0.0
    return float(sum(deltas) / len(deltas)), float(positive_count / len(deltas))


def _selected_prompt_modes(selected_prompt_modes_path: Path | None) -> dict[str, str]:
    if selected_prompt_modes_path is None or not selected_prompt_modes_path.exists():
        return {}
    payload = _load_json(selected_prompt_modes_path)
    selected: dict[str, str] = {}
    for task_name in ALL_TASKS:
        task_payload = payload.get(task_name, {})
        if isinstance(task_payload, dict):
            selected[task_name] = str(task_payload.get("selected_prompt_variant", "")).strip()
    return selected


def _v85_reference(v85_summary_path: Path | None) -> dict[str, Any]:
    if v85_summary_path is None or not v85_summary_path.exists():
        return {}
    return _load_json(v85_summary_path)


def _strict_metric_delta(control_task: dict[str, Any], branch_task: dict[str, Any]) -> dict[str, Any]:
    rank_fraction_delta = float(branch_task["writer_rank_fraction"] - control_task["writer_rank_fraction"])
    slot_rank_delta = float(
        branch_task["writer_memory_slot_effective_rank"] - control_task["writer_memory_slot_effective_rank"]
    )
    common_mode_gain = float(
        control_task["memory_long_common_mode_energy_ratio"] - branch_task["memory_long_common_mode_energy_ratio"]
    )
    pairwise_cosine_gain = float(
        abs(control_task["memory_long_pairwise_cosine_mean"])
        - abs(branch_task["memory_long_pairwise_cosine_mean"])
    )
    qualifies = bool(
        rank_fraction_delta >= 0.02
        or slot_rank_delta >= 1.0
        or common_mode_gain >= 0.01
        or pairwise_cosine_gain >= 0.01
    )
    return {
        "qualifies": qualifies,
        "rank_fraction_delta": rank_fraction_delta,
        "slot_rank_delta": slot_rank_delta,
        "common_mode_gain": common_mode_gain,
        "pairwise_cosine_gain": pairwise_cosine_gain,
    }


def _arm_task_payload(result_root: Path, arm_id: str, task_name: str) -> dict[str, Any]:
    task_root = result_root / arm_id / task_name
    metrics = _load_json(task_root / "metrics.json")
    train_events = _load_train_events(task_root / "train_events.json")
    case_rows = _load_case_rows(task_root / "task_case_dump.jsonl")
    prefix_stats = metrics.get("prefix_artifact_stats", {})
    if not isinstance(prefix_stats, dict):
        prefix_stats = {}

    task_score = _task_score(metrics)
    macro_f1 = _macro_f1(metrics)
    prompt_variant = str(metrics.get("pilot_prompt_variant") or metrics.get("prompt_variant") or "")
    memory_path_variant = str(metrics.get("pilot_memory_path_variant", "single_level"))
    writer_slots = _safe_int(metrics.get("writer_memory_slots", metrics.get("pilot_writer_memory_slots", 0)))
    reader_queries = _safe_int(metrics.get("pilot_reader_num_queries", 0))
    short_slots = _safe_int(metrics.get("pilot_fuser_short_slots", 0))
    memory_tokens_count = _safe_int(
        metrics.get("memory_tokens_count", prefix_stats.get("memory_tokens_count", 0))
    )
    prefix_attention_nontrivial_layer_count = _safe_int(
        metrics.get("prefix_attention_nontrivial_layer_count", 0)
    )
    prefix_attention_mass_mean = _safe_float(
        metrics.get("prefix_attention_mass_mean", _mean_row_value(case_rows, "prefix_attention_mass_mean"))
    )
    cross_attn_gate_open_fraction = _safe_float(
        metrics.get("cross_attn_gate_open_fraction", prefix_stats.get("cross_attn_gate_open_fraction", 0.0))
    )
    memory_token_attention_mass_mean = _safe_float(
        metrics.get(
            "memory_token_attention_mass_mean",
            prefix_stats.get("memory_token_attention_mass_mean", 0.0),
        )
    )
    reader_readout_effective_rank = _safe_float(metrics.get("reader_readout_effective_rank", 0.0))
    writer_grad_norms = [
        max(
            _safe_float(event.get("grad_norm_writer", 0.0)),
            _safe_float(event.get("grad_norm_support_encoder", 0.0)),
            _safe_float(event.get("grad_norm_projector", event.get("grad_norm_prefix_projector", 0.0))),
        )
        for event in train_events
    ]
    reader_grad_norms = [_safe_float(event.get("grad_norm_reader", 0.0)) for event in train_events]
    fuser_grad_norms = [_safe_float(event.get("grad_norm_fuser", 0.0)) for event in train_events]
    loss_values = [_safe_float(event.get("loss", 0.0)) for event in train_events]
    clipped_fraction = 0.0
    if train_events:
        clipped_fraction = float(
            sum(1 for event in train_events if bool(event.get("was_grad_clipped", False))) / len(train_events)
        )
    writer_clip_fraction = 0.0
    if train_events:
        writer_clip_fraction = float(
            sum(1 for event in train_events if bool(event.get("was_grad_clipped_writer", False))) / len(train_events)
        )
    stable_training = True
    if train_events:
        stable_training = _all_finite(loss_values + writer_grad_norms + reader_grad_norms + fuser_grad_norms) and (
            clipped_fraction < 0.95 and writer_clip_fraction < 0.95
        )
    writer_memory_slot_effective_rank = _safe_float(
        metrics.get(
            "train_final_memory_slot_effective_rank",
            metrics.get("writer_memory_slot_effective_rank", metrics.get("memory_slot_effective_rank", 0.0)),
        )
    )
    writer_rank_fraction = (
        float(writer_memory_slot_effective_rank / writer_slots) if writer_slots > 0 else 0.0
    )
    memory_long_common_mode_energy_ratio = _safe_float(
        metrics.get("memory_long_common_mode_energy_ratio", 0.0)
    )
    memory_long_pairwise_cosine_mean = _safe_float(
        metrics.get("memory_long_pairwise_cosine_mean", metrics.get("slot_pairwise_cosine", 0.0))
    )
    writer_task_only_grad = _safe_float(
        metrics.get(
            "train_grad_probe_writer_task_only_post_unfreeze_median",
            _median([_safe_float(event.get("grad_probe_writer_task_only_norm", 0.0)) for event in train_events]),
        )
    )
    writer_aux_only_grad = _safe_float(
        metrics.get(
            "train_grad_probe_writer_aux_only_post_unfreeze_median",
            _median([_safe_float(event.get("grad_probe_writer_aux_only_norm", 0.0)) for event in train_events]),
        )
    )
    writer_total_grad = _safe_float(
        metrics.get(
            "train_grad_probe_writer_total_post_unfreeze_median",
            _median([_safe_float(event.get("grad_probe_writer_total_norm", 0.0)) for event in train_events]),
        )
    )
    writer_task_aux_cosine = _safe_float(
        metrics.get(
            "train_grad_probe_writer_task_aux_cosine_post_unfreeze_median",
            _median([_safe_float(event.get("grad_probe_writer_task_aux_cosine", 0.0)) for event in train_events]),
        )
    )
    writer_aux_total_cosine = _safe_float(
        metrics.get(
            "train_grad_probe_writer_aux_total_cosine_post_unfreeze_median",
            _median([_safe_float(event.get("grad_probe_writer_aux_total_cosine", 0.0)) for event in train_events]),
        )
    )
    writer_task_ratio = float(writer_task_only_grad / max(writer_total_grad, 1.0e-8))
    writer_aux_signal_live = bool(
        writer_aux_only_grad > 0.0
        and writer_total_grad > 0.0
        and writer_task_aux_cosine > -0.20
        and writer_aux_total_cosine > -0.20
    )
    route_live = bool(
        (
            memory_path_variant == "two_level"
            and short_slots > 0
            and (
                reader_readout_effective_rank > 1.0
                or _median(reader_grad_norms) > 0.0
                or _median(fuser_grad_norms) > 0.0
            )
        )
        or (
            memory_path_variant == "single_level"
            and memory_tokens_count > 0
            and (
                prefix_attention_nontrivial_layer_count > 0
                or prefix_attention_mass_mean > 0.01
                or cross_attn_gate_open_fraction > 0.05
                or memory_token_attention_mass_mean > 0.01
            )
        )
    )
    return {
        "arm_id": arm_id,
        "task_name": task_name,
        "prompt_variant": prompt_variant,
        "task_score": task_score,
        "macro_f1": macro_f1,
        "answer_logprob_with_memory_mean": _mean_row_value(case_rows, "answer_logprob_with_memory"),
        "memory_path_variant": memory_path_variant,
        "writer_slots": writer_slots,
        "reader_queries": reader_queries,
        "short_slots": short_slots,
        "memory_tokens_count": memory_tokens_count,
        "prefix_attention_nontrivial_layer_count": prefix_attention_nontrivial_layer_count,
        "prefix_attention_mass_mean": prefix_attention_mass_mean,
        "cross_attn_gate_open_fraction": cross_attn_gate_open_fraction,
        "memory_token_attention_mass_mean": memory_token_attention_mass_mean,
        "reader_readout_effective_rank": reader_readout_effective_rank,
        "writer_grad_norm_median": _median(writer_grad_norms),
        "reader_grad_norm_median": _median(reader_grad_norms),
        "fuser_grad_norm_median": _median(fuser_grad_norms),
        "stable_training": stable_training,
        "route_live": route_live,
        "writer_aux_signal_live": writer_aux_signal_live,
        "writer_task_ratio": writer_task_ratio,
        "writer_task_only_grad": writer_task_only_grad,
        "writer_aux_only_grad": writer_aux_only_grad,
        "writer_total_grad": writer_total_grad,
        "writer_task_aux_cosine": writer_task_aux_cosine,
        "writer_aux_total_cosine": writer_aux_total_cosine,
        "writer_memory_slot_effective_rank": writer_memory_slot_effective_rank,
        "writer_rank_fraction": writer_rank_fraction,
        "memory_long_common_mode_energy_ratio": memory_long_common_mode_energy_ratio,
        "memory_long_pairwise_cosine_mean": memory_long_pairwise_cosine_mean,
        "pilot_active_aux_family": str(
            metrics.get("pilot_active_aux_family", ARM_METADATA[arm_id]["auxiliary_family"])
        ),
        "pilot_alignment_aux_mode": str(metrics.get("pilot_alignment_aux_mode", "")),
        "pilot_reconstruction_aux_mode": str(metrics.get("pilot_reconstruction_aux_mode", "")),
        "train_final_opd_mean_advantage": _safe_float(metrics.get("train_final_opd_mean_advantage", 0.0)),
        "train_final_opd_positive_token_fraction": _safe_float(
            metrics.get("train_final_opd_positive_token_fraction", 0.0)
        ),
        "train_final_opd_target_context_available": bool(
            metrics.get("train_final_opd_target_context_available", False)
        ),
        "train_barlow_aux_loss_post_unfreeze_median": _safe_float(
            metrics.get("train_barlow_aux_loss_post_unfreeze_median", 0.0)
        ),
        "train_reconstruction_aux_loss_post_unfreeze_median": _safe_float(
            metrics.get("train_reconstruction_aux_loss_post_unfreeze_median", 0.0)
        ),
        "case_rows": case_rows,
    }


def build_summary(
    *,
    result_root: Path,
    selected_prompt_modes_path: Path | None = None,
    v85_summary_path: Path | None = None,
) -> dict[str, Any]:
    selected_prompt_modes = _selected_prompt_modes(selected_prompt_modes_path)
    v85_reference = _v85_reference(v85_summary_path)
    base_arm_id = str(
        v85_reference.get("base_for_v8_6_arm_id") or v85_reference.get("best_arm_id") or ""
    ).strip()
    selected_interface_family = str(v85_reference.get("selected_interface_family_for_v8_6") or "").strip()
    selected_bridge_family = str(v85_reference.get("selected_bridge_family_for_v8_6") or "").strip()

    control_by_task = {
        task_name: _arm_task_payload(result_root, CONTROL_ARM_ID, task_name)
        for task_name in ALL_TASKS
    }
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        per_task: dict[str, Any] = {}
        strict_gain_count = 0
        strict_rank_fraction_delta_sum = 0.0
        strict_slot_rank_delta_sum = 0.0
        common_mode_gain_sum = 0.0
        pairwise_cosine_gain_sum = 0.0
        for task_name in ALL_TASKS:
            branch = _arm_task_payload(result_root, arm_id, task_name)
            control = control_by_task[task_name]
            answer_logprob_delta_vs_control_mean, positive_shift_fraction = _shared_row_delta(
                control["case_rows"],
                branch["case_rows"],
                key="answer_logprob_with_memory",
            )
            strict_delta = _strict_metric_delta(control, branch)
            strict_gain_count += int(strict_delta["qualifies"])
            strict_rank_fraction_delta_sum += float(strict_delta["rank_fraction_delta"])
            strict_slot_rank_delta_sum += float(strict_delta["slot_rank_delta"])
            common_mode_gain_sum += float(strict_delta["common_mode_gain"])
            pairwise_cosine_gain_sum += float(strict_delta["pairwise_cosine_gain"])
            per_task[task_name] = {
                **{k: v for k, v in branch.items() if k != "case_rows"},
                "control_task_score": float(control["task_score"]),
                "task_score_delta_vs_control": float(branch["task_score"] - control["task_score"]),
                "answer_logprob_delta_vs_control_mean": answer_logprob_delta_vs_control_mean,
                "positive_answer_logprob_shift_fraction": positive_shift_fraction,
                "strict_metric_delta": strict_delta,
            }
        gsm8k = per_task["gsm8k"]
        triviaqa = per_task["triviaqa"]
        fever = per_task["fever"]
        primary_task_score_delta_sum = float(
            gsm8k["task_score_delta_vs_control"] + triviaqa["task_score_delta_vs_control"]
        )
        primary_answer_logprob_delta_sum = float(
            gsm8k["answer_logprob_delta_vs_control_mean"]
            + triviaqa["answer_logprob_delta_vs_control_mean"]
        )
        positive_primary_score_task_count = int(
            sum(
                1
                for task_payload in (gsm8k, triviaqa)
                if float(task_payload["task_score_delta_vs_control"]) > 0.0
            )
        )
        nonnegative_primary_score_task_count = int(
            sum(
                1
                for task_payload in (gsm8k, triviaqa)
                if float(task_payload["task_score_delta_vs_control"]) >= 0.0
            )
        )
        route_live_primary_task_count = int(
            sum(1 for task_payload in (gsm8k, triviaqa) if bool(task_payload["route_live"]))
        )
        writer_aux_signal_primary_task_count = int(
            sum(1 for task_payload in (gsm8k, triviaqa) if bool(task_payload["writer_aux_signal_live"]))
        )
        stable_primary_task_count = int(
            sum(1 for task_payload in (gsm8k, triviaqa) if bool(task_payload["stable_training"]))
        )
        stable_all_tasks = bool(all(bool(task_payload["stable_training"]) for task_payload in per_task.values()))
        primary_gain_qualified = bool(
            arm_id != CONTROL_ARM_ID
            and stable_all_tasks
            and stable_primary_task_count == len(PRIMARY_TASKS)
            and route_live_primary_task_count >= 1
            and writer_aux_signal_primary_task_count >= 1
            and positive_primary_score_task_count >= 1
            and nonnegative_primary_score_task_count == len(PRIMARY_TASKS)
        )
        writer_health_qualified = bool(
            arm_id != CONTROL_ARM_ID
            and stable_all_tasks
            and stable_primary_task_count == len(PRIMARY_TASKS)
            and route_live_primary_task_count >= 1
            and writer_aux_signal_primary_task_count >= 1
            and strict_gain_count >= 1
            and nonnegative_primary_score_task_count == len(PRIMARY_TASKS)
        )
        acceptance_qualified = bool(primary_gain_qualified or writer_health_qualified)
        arm_summaries[arm_id] = {
            "arm_id": arm_id,
            "auxiliary_family": str(gsm8k["pilot_active_aux_family"] or ARM_METADATA[arm_id]["auxiliary_family"]),
            "memory_path_variant": str(gsm8k["memory_path_variant"]),
            "bridge_family": selected_bridge_family,
            "writer_slots": max(
                _safe_int(gsm8k["writer_slots"], 0),
                _safe_int(triviaqa["writer_slots"], 0),
                _safe_int(fever["writer_slots"], 0),
            ),
            "reader_queries": max(
                _safe_int(gsm8k["reader_queries"], 0),
                _safe_int(triviaqa["reader_queries"], 0),
                _safe_int(fever["reader_queries"], 0),
            ),
            "short_slots": max(
                _safe_int(gsm8k["short_slots"], 0),
                _safe_int(triviaqa["short_slots"], 0),
                _safe_int(fever["short_slots"], 0),
            ),
            "primary_task_score_delta_sum": primary_task_score_delta_sum,
            "primary_answer_logprob_delta_sum": primary_answer_logprob_delta_sum,
            "positive_primary_score_task_count": positive_primary_score_task_count,
            "nonnegative_primary_score_task_count": nonnegative_primary_score_task_count,
            "route_live_primary_task_count": route_live_primary_task_count,
            "writer_aux_signal_primary_task_count": writer_aux_signal_primary_task_count,
            "stable_primary_task_count": stable_primary_task_count,
            "stable_all_tasks": stable_all_tasks,
            "strict_writer_metric_gain_task_count": strict_gain_count,
            "strict_rank_fraction_delta_sum": float(strict_rank_fraction_delta_sum),
            "strict_slot_rank_delta_sum": float(strict_slot_rank_delta_sum),
            "common_mode_gain_sum": float(common_mode_gain_sum),
            "pairwise_cosine_gain_sum": float(pairwise_cosine_gain_sum),
            "primary_gain_qualified": primary_gain_qualified,
            "writer_health_qualified": writer_health_qualified,
            "acceptance_qualified": acceptance_qualified,
            "fever_task_score_delta_vs_control": float(fever["task_score_delta_vs_control"]),
            "ranking_key": [
                float(primary_gain_qualified),
                float(positive_primary_score_task_count),
                primary_task_score_delta_sum,
                float(writer_health_qualified),
                float(strict_gain_count),
                float(strict_rank_fraction_delta_sum),
                float(common_mode_gain_sum),
                primary_answer_logprob_delta_sum,
                float(route_live_primary_task_count),
                float(stable_primary_task_count),
                float(fever["task_score_delta_vs_control"]),
            ],
            "tasks": per_task,
        }

    aux_arm_ranking = sorted(
        (
            {
                "arm_id": arm_id,
                "auxiliary_family": summary["auxiliary_family"],
                "ranking_key": summary["ranking_key"],
                "acceptance_qualified": summary["acceptance_qualified"],
            }
            for arm_id, summary in arm_summaries.items()
            if arm_id in AUX_ARMS
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )
    best_aux_arm_id = aux_arm_ranking[0]["arm_id"]
    best_aux_arm = arm_summaries[best_aux_arm_id]
    control_arm = arm_summaries[CONTROL_ARM_ID]

    if best_aux_arm["primary_gain_qualified"]:
        comparison_conclusion = "writer_aux_primary_gain_open_v8_7_comparators"
        best_arm_id = best_aux_arm_id
    elif best_aux_arm["writer_health_qualified"]:
        comparison_conclusion = "writer_aux_writer_health_gain_open_v8_7_comparators"
        best_arm_id = best_aux_arm_id
    else:
        comparison_conclusion = "writer_aux_flat_keep_base_route_open_v8_7_comparators"
        best_arm_id = CONTROL_ARM_ID

    return {
        "phase": "V8-6",
        "selected_prompt_modes_by_task": selected_prompt_modes,
        "control_arm_id": CONTROL_ARM_ID,
        "control_scores_by_task": {
            task_name: float(control_by_task[task_name]["task_score"])
            for task_name in ALL_TASKS
        },
        "v85_best_arm_id": str(v85_reference.get("best_arm_id", "")).strip(),
        "v85_base_for_v8_6_arm_id": base_arm_id,
        "v85_selected_interface_family_for_v8_6": selected_interface_family,
        "v85_selected_bridge_family_for_v8_6": selected_bridge_family,
        "v85_recommended_next_step": str(v85_reference.get("recommended_next_step", "")).strip(),
        "arm_summaries": arm_summaries,
        "aux_arm_ranking": aux_arm_ranking,
        "best_aux_arm_id": best_aux_arm_id,
        "best_arm_id": best_arm_id,
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": "open_v8_7_comparators",
        "base_for_v8_7_arm_id": best_arm_id,
        "selected_interface_family_for_v8_7": selected_interface_family,
        "selected_bridge_family_for_v8_7": selected_bridge_family,
        "selected_aux_family_for_v8_7": arm_summaries[best_arm_id]["auxiliary_family"],
        "best_arm_acceptance_qualified": bool(best_arm_id != CONTROL_ARM_ID),
        "control_route_summary": {
            "arm_id": CONTROL_ARM_ID,
            "primary_task_score_delta_sum": control_arm["primary_task_score_delta_sum"],
            "primary_answer_logprob_delta_sum": control_arm["primary_answer_logprob_delta_sum"],
        },
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv8 V8-6 Writer Auxiliary Revisit",
        "",
        "## Decision",
        f"- `comparison_conclusion = {summary['comparison_conclusion']}`",
        f"- `recommended_next_step = {summary['recommended_next_step']}`",
        f"- `best_arm_id = {summary['best_arm_id']}`",
        f"- `best_aux_arm_id = {summary['best_aux_arm_id']}`",
        "",
        "## Auxiliary Ranking",
    ]
    for payload in summary["aux_arm_ranking"]:
        lines.append(
            "- "
            f"`{payload['arm_id']}` "
            f"family={payload['auxiliary_family']} "
            f"acceptance={payload['acceptance_qualified']}"
        )
    lines.extend(
        [
            "",
            "## Control Scores",
            *[
                f"- `{task_name}` control score = {summary['control_scores_by_task'][task_name]:.6f}"
                for task_name in ALL_TASKS
            ],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--selected_prompt_modes", type=Path, default=None)
    parser.add_argument("--v85_summary", type=Path, default=None)
    args = parser.parse_args()

    selected_prompt_modes_path = args.selected_prompt_modes
    if selected_prompt_modes_path is None:
        candidate = args.result_root / "selected-prompt-modes.json"
        if candidate.exists():
            selected_prompt_modes_path = candidate

    v85_summary_path = args.v85_summary
    if v85_summary_path is None:
        candidate = args.result_root / "v8-5-summary.reference.json"
        if candidate.exists():
            v85_summary_path = candidate

    summary = build_summary(
        result_root=args.result_root,
        selected_prompt_modes_path=selected_prompt_modes_path,
        v85_summary_path=v85_summary_path,
    )
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(_render_report(summary))


if __name__ == "__main__":
    main()
