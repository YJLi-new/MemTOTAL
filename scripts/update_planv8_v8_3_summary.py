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
ARM_ORDER = (
    "p0_ce_only",
    "p1_teacher_choice_kl",
    "p2_opd_ansonly_w01",
    "p3_opd_ansonly_w03",
    "p4_opd_ansplusctx_w03",
    "p5_opd_ansplusctx_centered",
)
ARM_METADATA = {
    "p0_ce_only": {
        "alignment_aux_mode": "off",
        "hint_strength": "none",
        "opd_weight_max": 0.0,
    },
    "p1_teacher_choice_kl": {
        "alignment_aux_mode": "teacher_choice_kl",
        "hint_strength": "task_default",
        "opd_weight_max": 0.1,
    },
    "p2_opd_ansonly_w01": {
        "alignment_aux_mode": "opd_token_ce",
        "hint_strength": "answer_only",
        "opd_weight_max": 0.1,
    },
    "p3_opd_ansonly_w03": {
        "alignment_aux_mode": "opd_token_ce",
        "hint_strength": "answer_only",
        "opd_weight_max": 0.3,
    },
    "p4_opd_ansplusctx_w03": {
        "alignment_aux_mode": "opd_token_ce",
        "hint_strength": "answer_plus_context",
        "opd_weight_max": 0.3,
    },
    "p5_opd_ansplusctx_centered": {
        "alignment_aux_mode": "opd_token_ce_centered",
        "hint_strength": "answer_plus_context",
        "opd_weight_max": 0.3,
    },
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


def _v82_reference(v82_summary_path: Path | None) -> dict[str, Any]:
    if v82_summary_path is None or not v82_summary_path.exists():
        return {}
    return _load_json(v82_summary_path)


def _arm_task_payload(result_root: Path, arm_id: str, task_name: str) -> dict[str, Any]:
    task_root = result_root / arm_id / task_name
    metrics = _load_json(task_root / "metrics.json")
    train_events = _load_train_events(task_root / "train_events.json")
    case_rows = _load_case_rows(task_root / "task_case_dump.jsonl")
    prefix_stats = metrics.get("prefix_artifact_stats", {})
    if not isinstance(prefix_stats, dict):
        prefix_stats = {}
    prompt_variant = str(metrics.get("pilot_prompt_variant") or metrics.get("prompt_variant") or "")
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
    reader_grad_norms = [
        max(
            _safe_float(event.get("grad_norm_prefix_projector", event.get("grad_norm_projector", 0.0))),
            _safe_float(event.get("grad_norm_receiver_lora", 0.0)),
            _safe_float(event.get("grad_norm_reader_cross_attn", 0.0)),
        )
        for event in train_events
    ]
    loss_values = [_safe_float(event.get("loss", 0.0)) for event in train_events]
    clipped_fraction = 0.0
    if train_events:
        clipped_fraction = float(
            sum(1 for event in train_events if bool(event.get("was_grad_clipped", False))) / len(train_events)
        )
    stable_training = True
    if train_events:
        stable_training = _all_finite(loss_values + reader_grad_norms) and clipped_fraction < 0.95
    task_score = _task_score(metrics)
    macro_f1 = _macro_f1(metrics)
    answer_logprob_with_memory_mean = _mean_row_value(case_rows, "answer_logprob_with_memory")
    within_arm_delta_answer_logprob_mean = _mean_row_value(case_rows, "delta_answer_logprob")
    activation_score = float(
        prefix_attention_mass_mean
        + cross_attn_gate_open_fraction
        + memory_token_attention_mass_mean
        + (0.01 * float(prefix_attention_nontrivial_layer_count))
    )
    route_live = bool(
        memory_tokens_count > 0
        and (
            prefix_attention_nontrivial_layer_count > 0
            or prefix_attention_mass_mean > 0.01
            or cross_attn_gate_open_fraction > 0.05
            or memory_token_attention_mass_mean > 0.01
        )
        and (not train_events or sum(1 for value in reader_grad_norms if value > 0.0) > 0)
    )
    return {
        "arm_id": arm_id,
        "task_name": task_name,
        "prompt_variant": prompt_variant,
        "task_score": task_score,
        "macro_f1": macro_f1,
        "answer_logprob_with_memory_mean": answer_logprob_with_memory_mean,
        "within_arm_delta_answer_logprob_mean": within_arm_delta_answer_logprob_mean,
        "memory_tokens_count": memory_tokens_count,
        "prefix_attention_nontrivial_layer_count": prefix_attention_nontrivial_layer_count,
        "prefix_attention_mass_mean": prefix_attention_mass_mean,
        "cross_attn_gate_open_fraction": cross_attn_gate_open_fraction,
        "memory_token_attention_mass_mean": memory_token_attention_mass_mean,
        "activation_score": activation_score,
        "stable_training": stable_training,
        "route_live": route_live,
        "reader_grad_norm_median": _median(reader_grad_norms),
        "reader_grad_nonzero_steps": int(sum(1 for value in reader_grad_norms if value > 0.0)),
        "train_steps": _safe_int(metrics.get("pilot_train_steps", len(train_events))),
        "clipped_fraction": clipped_fraction,
        "alignment_aux_mode": str(metrics.get("pilot_alignment_aux_mode", "")),
        "opd_hint_mode_gsm8k": str(metrics.get("pilot_opd_hint_mode_gsm8k", "")),
        "opd_hint_mode_triviaqa": str(metrics.get("pilot_opd_hint_mode_triviaqa", "")),
        "opd_hint_mode_fever": str(metrics.get("pilot_opd_hint_mode_fever", "")),
        "train_final_opd_mean_advantage": _safe_float(metrics.get("train_final_opd_mean_advantage")),
        "train_final_opd_positive_token_fraction": _safe_float(
            metrics.get("train_final_opd_positive_token_fraction")
        ),
        "train_final_opd_target_context_available": bool(
            metrics.get("train_final_opd_target_context_available", False)
        ),
        "train_final_opd_hint_mode_effective": str(
            metrics.get("train_final_opd_hint_mode_effective", "")
        ),
        "case_rows": case_rows,
    }


def build_summary(
    *,
    result_root: Path,
    selected_prompt_modes_path: Path | None = None,
    v82_summary_path: Path | None = None,
) -> dict[str, Any]:
    selected_prompt_modes = _selected_prompt_modes(selected_prompt_modes_path)
    v82_reference = _v82_reference(v82_summary_path)
    base_arm_id = str(
        v82_reference.get("base_for_v8_3_arm_id") or v82_reference.get("best_arm_id") or ""
    ).strip()
    selected_interface_family = str(
        v82_reference.get("selected_interface_family_for_v8_3")
        or v82_reference.get("best_interface_family")
        or ""
    ).strip()

    control_by_task = {
        task_name: _arm_task_payload(result_root, "p0_ce_only", task_name)
        for task_name in ALL_TASKS
    }
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        per_task: dict[str, Any] = {}
        for task_name in ALL_TASKS:
            branch = _arm_task_payload(result_root, arm_id, task_name)
            control = control_by_task[task_name]
            answer_logprob_delta_vs_control_mean, positive_shift_fraction = _shared_row_delta(
                control["case_rows"],
                branch["case_rows"],
                key="answer_logprob_with_memory",
            )
            per_task[task_name] = {
                **{k: v for k, v in branch.items() if k != "case_rows"},
                "control_task_score": float(control["task_score"]),
                "task_score_delta_vs_control": float(branch["task_score"] - control["task_score"]),
                "answer_logprob_delta_vs_control_mean": answer_logprob_delta_vs_control_mean,
                "positive_answer_logprob_shift_fraction": positive_shift_fraction,
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
        stable_primary_task_count = int(
            sum(1 for task_payload in (gsm8k, triviaqa) if bool(task_payload["stable_training"]))
        )
        stable_all_tasks = bool(all(bool(task_payload["stable_training"]) for task_payload in per_task.values()))
        primary_activation_score_mean = float(
            (float(gsm8k["activation_score"]) + float(triviaqa["activation_score"])) / 2.0
        )
        primary_opd_positive_fraction_mean = float(
            (
                float(gsm8k["train_final_opd_positive_token_fraction"])
                + float(triviaqa["train_final_opd_positive_token_fraction"])
            )
            / 2.0
        )
        primary_opd_advantage_mean = float(
            (
                float(gsm8k["train_final_opd_mean_advantage"])
                + float(triviaqa["train_final_opd_mean_advantage"])
            )
            / 2.0
        )
        acceptance_qualified = bool(
            stable_all_tasks
            and stable_primary_task_count == len(PRIMARY_TASKS)
            and route_live_primary_task_count >= 1
            and positive_primary_score_task_count >= 1
            and nonnegative_primary_score_task_count == len(PRIMARY_TASKS)
        )
        metadata = ARM_METADATA[arm_id]
        arm_summaries[arm_id] = {
            "arm_id": arm_id,
            "interface_family": selected_interface_family,
            "alignment_aux_mode": metadata["alignment_aux_mode"],
            "hint_strength": metadata["hint_strength"],
            "opd_weight_max": float(metadata["opd_weight_max"]),
            "primary_task_score_delta_sum": primary_task_score_delta_sum,
            "primary_answer_logprob_delta_sum": primary_answer_logprob_delta_sum,
            "positive_primary_score_task_count": positive_primary_score_task_count,
            "nonnegative_primary_score_task_count": nonnegative_primary_score_task_count,
            "route_live_primary_task_count": route_live_primary_task_count,
            "stable_primary_task_count": stable_primary_task_count,
            "stable_all_tasks": stable_all_tasks,
            "primary_activation_score_mean": primary_activation_score_mean,
            "primary_opd_positive_fraction_mean": primary_opd_positive_fraction_mean,
            "primary_opd_advantage_mean": primary_opd_advantage_mean,
            "fever_task_score_delta_vs_control": float(fever["task_score_delta_vs_control"]),
            "acceptance_qualified": acceptance_qualified,
            "ranking_key": [
                float(gsm8k["task_score_delta_vs_control"]),
                float(triviaqa["task_score_delta_vs_control"]),
                primary_task_score_delta_sum,
                float(positive_primary_score_task_count),
                float(nonnegative_primary_score_task_count),
                primary_answer_logprob_delta_sum,
                primary_activation_score_mean,
                primary_opd_positive_fraction_mean,
                primary_opd_advantage_mean,
                float(route_live_primary_task_count),
                float(stable_primary_task_count),
                float(fever["task_score_delta_vs_control"]),
            ],
            "tasks": per_task,
        }

    arm_ranking = sorted(
        (
            {
                "arm_id": arm_id,
                "interface_family": summary["interface_family"],
                "alignment_aux_mode": summary["alignment_aux_mode"],
                "hint_strength": summary["hint_strength"],
                "opd_weight_max": summary["opd_weight_max"],
                "primary_task_score_delta_sum": summary["primary_task_score_delta_sum"],
                "primary_answer_logprob_delta_sum": summary["primary_answer_logprob_delta_sum"],
                "primary_activation_score_mean": summary["primary_activation_score_mean"],
                "primary_opd_positive_fraction_mean": summary["primary_opd_positive_fraction_mean"],
                "positive_primary_score_task_count": summary["positive_primary_score_task_count"],
                "acceptance_qualified": summary["acceptance_qualified"],
                "ranking_key": summary["ranking_key"],
            }
            for arm_id, summary in arm_summaries.items()
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )
    best_arm_id = arm_ranking[0]["arm_id"]
    best_arm = arm_summaries[best_arm_id]

    if best_arm["acceptance_qualified"]:
        comparison_conclusion = "reader_opd_score_signal_open_v8_4_external_writer"
        recommended_next_step = "open_v8_4_external_writer"
    else:
        comparison_conclusion = "reader_opd_flat_open_v8_7_comparators"
        recommended_next_step = "open_v8_7_comparators"

    return {
        "phase": "V8-3",
        "selected_prompt_modes_by_task": selected_prompt_modes,
        "control_scores_by_task": {
            task_name: float(control_by_task[task_name]["task_score"])
            for task_name in ALL_TASKS
        },
        "v82_best_arm_id": str(v82_reference.get("best_arm_id", "")).strip(),
        "v82_base_for_v8_3_arm_id": base_arm_id,
        "v82_selected_interface_family_for_v8_3": selected_interface_family,
        "v82_recommended_next_step": str(v82_reference.get("recommended_next_step", "")).strip(),
        "arm_summaries": arm_summaries,
        "arm_ranking": arm_ranking,
        "best_arm_id": best_arm_id,
        "best_interface_family": best_arm["interface_family"],
        "best_alignment_aux_mode": best_arm["alignment_aux_mode"],
        "best_hint_strength": best_arm["hint_strength"],
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "base_for_v8_4_arm_id": best_arm_id,
        "selected_interface_family_for_v8_4": best_arm["interface_family"],
        "best_arm_acceptance_qualified": bool(best_arm["acceptance_qualified"]),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv8 V8-3 Reader OPD",
        "",
        "## Decision",
        f"- `comparison_conclusion = {summary['comparison_conclusion']}`",
        f"- `recommended_next_step = {summary['recommended_next_step']}`",
        f"- `best_arm_id = {summary['best_arm_id']}`",
        f"- `best_alignment_aux_mode = {summary['best_alignment_aux_mode']}`",
        f"- `base_for_v8_4_arm_id = {summary['base_for_v8_4_arm_id']}`",
        "",
        "## Ranking",
    ]
    for payload in summary["arm_ranking"]:
        lines.append(
            "- "
            f"`{payload['arm_id']}` "
            f"mode={payload['alignment_aux_mode']} "
            f"score_sum={payload['primary_task_score_delta_sum']:.6f} "
            f"logprob_sum={payload['primary_answer_logprob_delta_sum']:.6f} "
            f"activation={payload['primary_activation_score_mean']:.6f} "
            f"opd_pos_frac={payload['primary_opd_positive_fraction_mean']:.6f} "
            f"acceptance={payload['acceptance_qualified']}"
        )
    lines.extend(
        [
            "",
            "## Control",
            *[
                f"- `{task_name}` control score = {summary['control_scores_by_task'][task_name]:.6f}"
                for task_name in ALL_TASKS
            ],
            "",
            "## Notes",
            "- TriviaQA answer-plus-context arms can legitimately fall back to answer-only when the selected source lacks evidence sentences.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--selected_prompt_modes", type=Path, default=None)
    parser.add_argument("--v82_summary", type=Path, default=None)
    args = parser.parse_args()

    selected_prompt_modes_path = args.selected_prompt_modes
    if selected_prompt_modes_path is None:
        candidate = args.result_root / "selected-prompt-modes.json"
        if candidate.exists():
            selected_prompt_modes_path = candidate

    v82_summary_path = args.v82_summary
    if v82_summary_path is None:
        candidate = args.result_root / "v8-2-summary.reference.json"
        if candidate.exists():
            v82_summary_path = candidate

    summary = build_summary(
        result_root=args.result_root,
        selected_prompt_modes_path=selected_prompt_modes_path,
        v82_summary_path=v82_summary_path,
    )
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(_render_report(summary))


if __name__ == "__main__":
    main()
