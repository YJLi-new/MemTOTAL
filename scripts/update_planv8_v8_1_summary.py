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
    "i0_prefix_legacy_r2",
    "i1_seq16_r32_mid8",
    "i2_seq16_r64_mid8",
    "i3_seq32_r64_mid8",
    "i4_xattn16_mid4_r32",
    "i5_xattn16_mid4_r64",
)
ARM_METADATA = {
    "i0_prefix_legacy_r2": {
        "interface_family": "ri0_legacy_prefix",
        "memory_slots": 16,
        "reader_layers_label": "mid4",
        "reader_layers": [16, 17, 18, 19],
        "rank_label": "r2",
    },
    "i1_seq16_r32_mid8": {
        "interface_family": "ri1_prepend_block",
        "memory_slots": 16,
        "reader_layers_label": "mid8",
        "reader_layers": [12, 13, 14, 15, 16, 17, 18, 19],
        "rank_label": "r32",
    },
    "i2_seq16_r64_mid8": {
        "interface_family": "ri1_prepend_block",
        "memory_slots": 16,
        "reader_layers_label": "mid8",
        "reader_layers": [12, 13, 14, 15, 16, 17, 18, 19],
        "rank_label": "r64",
    },
    "i3_seq32_r64_mid8": {
        "interface_family": "ri1_prepend_block",
        "memory_slots": 32,
        "reader_layers_label": "mid8",
        "reader_layers": [12, 13, 14, 15, 16, 17, 18, 19],
        "rank_label": "r64",
    },
    "i4_xattn16_mid4_r32": {
        "interface_family": "ri2_cross_attn",
        "memory_slots": 16,
        "reader_layers_label": "mid4",
        "reader_layers": [16, 17, 18, 19],
        "rank_label": "r32_proxy",
    },
    "i5_xattn16_mid4_r64": {
        "interface_family": "ri2_cross_attn",
        "memory_slots": 16,
        "reader_layers_label": "mid4",
        "reader_layers": [16, 17, 18, 19],
        "rank_label": "r64_proxy",
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


def _v80_reference(v80_summary_path: Path | None) -> dict[str, Any]:
    if v80_summary_path is None or not v80_summary_path.exists():
        return {}
    return _load_json(v80_summary_path)


def _selected_primary_baseline_scores(v80_reference: dict[str, Any]) -> dict[str, float]:
    for key in (
        "selected_primary_baseline_scores",
        "selected_qwen3_baseline_scores",
        "selected_qwen34_baseline_scores",
    ):
        payload = v80_reference.get(key, {})
        if isinstance(payload, dict) and payload:
            return dict(payload)
    return {}


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
        "case_rows": case_rows,
    }


def build_summary(
    *,
    result_root: Path,
    selected_prompt_modes_path: Path | None = None,
    v80_summary_path: Path | None = None,
) -> dict[str, Any]:
    selected_prompt_modes = _selected_prompt_modes(selected_prompt_modes_path)
    v80_reference = _v80_reference(v80_summary_path)

    control_by_task = {
        task_name: _arm_task_payload(result_root, "control", task_name)
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
        metadata = ARM_METADATA[arm_id]
        gsm8k = per_task["gsm8k"]
        triviaqa = per_task["triviaqa"]
        fever = per_task["fever"]
        primary_score_delta_sum = float(
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
        positive_primary_answer_logprob_task_count = int(
            sum(
                1
                for task_payload in (gsm8k, triviaqa)
                if float(task_payload["answer_logprob_delta_vs_control_mean"]) > 0.0
            )
        )
        route_live_primary_task_count = int(
            sum(1 for task_payload in (gsm8k, triviaqa) if bool(task_payload["route_live"]))
        )
        stable_primary_task_count = int(
            sum(1 for task_payload in (gsm8k, triviaqa) if bool(task_payload["stable_training"]))
        )
        primary_activation_score_mean = float(
            (
                float(gsm8k["activation_score"])
                + float(triviaqa["activation_score"])
            )
            / 2.0
        )
        fever_non_regression = bool(float(fever["task_score_delta_vs_control"]) >= -0.05)
        acceptance_qualified = bool(
            stable_primary_task_count == len(PRIMARY_TASKS)
            and route_live_primary_task_count >= 1
            and fever_non_regression
            and (
                positive_primary_score_task_count > 0
                or (
                    positive_primary_answer_logprob_task_count == len(PRIMARY_TASKS)
                    and primary_activation_score_mean >= 0.05
                )
            )
        )
        arm_summaries[arm_id] = {
            "arm_id": arm_id,
            "interface_family": metadata["interface_family"],
            "memory_slots": metadata["memory_slots"],
            "reader_layers_label": metadata["reader_layers_label"],
            "reader_layers": list(metadata["reader_layers"]),
            "rank_label": metadata["rank_label"],
            "primary_task_score_delta_sum": primary_score_delta_sum,
            "primary_answer_logprob_delta_sum": primary_answer_logprob_delta_sum,
            "positive_primary_score_task_count": positive_primary_score_task_count,
            "positive_primary_answer_logprob_task_count": positive_primary_answer_logprob_task_count,
            "route_live_primary_task_count": route_live_primary_task_count,
            "stable_primary_task_count": stable_primary_task_count,
            "primary_activation_score_mean": primary_activation_score_mean,
            "fever_task_score_delta_vs_control": float(fever["task_score_delta_vs_control"]),
            "fever_non_regression": fever_non_regression,
            "acceptance_qualified": acceptance_qualified,
            "ranking_key": [
                float(gsm8k["task_score_delta_vs_control"]),
                float(triviaqa["task_score_delta_vs_control"]),
                primary_score_delta_sum,
                primary_answer_logprob_delta_sum,
                primary_activation_score_mean,
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
                "memory_slots": summary["memory_slots"],
                "reader_layers_label": summary["reader_layers_label"],
                "rank_label": summary["rank_label"],
                "primary_task_score_delta_sum": summary["primary_task_score_delta_sum"],
                "primary_answer_logprob_delta_sum": summary["primary_answer_logprob_delta_sum"],
                "primary_activation_score_mean": summary["primary_activation_score_mean"],
                "positive_primary_score_task_count": summary["positive_primary_score_task_count"],
                "positive_primary_answer_logprob_task_count": summary[
                    "positive_primary_answer_logprob_task_count"
                ],
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
    legacy_arm = arm_summaries["i0_prefix_legacy_r2"]
    best_nonlegacy_id = max(
        (arm_id for arm_id in ARM_ORDER if arm_id != "i0_prefix_legacy_r2"),
        key=lambda arm_id: tuple(arm_summaries[arm_id]["ranking_key"]),
    )
    best_nonlegacy_arm = arm_summaries[best_nonlegacy_id]
    nonlegacy_beats_legacy_on_primary = bool(
        float(best_nonlegacy_arm["primary_task_score_delta_sum"])
        > float(legacy_arm["primary_task_score_delta_sum"]) + 1e-9
        or float(best_nonlegacy_arm["primary_answer_logprob_delta_sum"])
        > float(legacy_arm["primary_answer_logprob_delta_sum"]) + 1e-6
    )
    if best_arm["acceptance_qualified"] and best_arm["positive_primary_score_task_count"] > 0:
        comparison_conclusion = "reader_interface_score_signal_open_v8_2"
        recommended_next_step = "open_v8_2_reader_sweep"
    elif best_arm["acceptance_qualified"]:
        comparison_conclusion = "reader_interface_diagnostic_signal_open_v8_2"
        recommended_next_step = "open_v8_2_reader_sweep"
    elif nonlegacy_beats_legacy_on_primary:
        comparison_conclusion = "reader_interface_family_preference_open_v8_2"
        recommended_next_step = "open_v8_2_reader_sweep"
    else:
        comparison_conclusion = "reader_interface_flat_open_v8_2_last_chance"
        recommended_next_step = "open_v8_2_reader_sweep_last_chance"

    v80_selected_baseline_scores = _selected_primary_baseline_scores(v80_reference)
    control_match_v80 = {
        task_name: (
            abs(float(control_by_task[task_name]["task_score"]) - _safe_float(v80_selected_baseline_scores.get(task_name)))
            <= 1e-9
            if task_name in v80_selected_baseline_scores
            else None
        )
        for task_name in ALL_TASKS
    }

    return {
        "phase": "V8-1",
        "selected_prompt_modes_by_task": selected_prompt_modes,
        "control_scores_by_task": {
            task_name: float(control_by_task[task_name]["task_score"])
            for task_name in ALL_TASKS
        },
        "v80_selected_baseline_scores": v80_selected_baseline_scores,
        "control_matches_v80_selected_baseline_by_task": control_match_v80,
        "arm_summaries": arm_summaries,
        "arm_ranking": arm_ranking,
        "best_arm_id": best_arm_id,
        "best_interface_family": best_arm["interface_family"],
        "best_nonlegacy_arm_id": best_nonlegacy_id,
        "nonlegacy_beats_legacy_on_primary": nonlegacy_beats_legacy_on_primary,
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "base_for_v8_2_arm_id": best_arm_id,
        "selected_interface_family_for_v8_2": best_arm["interface_family"],
        "best_arm_acceptance_qualified": bool(best_arm["acceptance_qualified"]),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv8 V8-1 Reader Interface Scout",
        "",
        "## Decision",
        f"- `comparison_conclusion = {summary['comparison_conclusion']}`",
        f"- `recommended_next_step = {summary['recommended_next_step']}`",
        f"- `best_arm_id = {summary['best_arm_id']}`",
        f"- `best_interface_family = {summary['best_interface_family']}`",
        f"- `base_for_v8_2_arm_id = {summary['base_for_v8_2_arm_id']}`",
        "",
        "## Ranking",
    ]
    for payload in summary["arm_ranking"]:
        lines.append(
            "- "
            f"`{payload['arm_id']}` "
            f"score_sum={payload['primary_task_score_delta_sum']:.6f} "
            f"logprob_sum={payload['primary_answer_logprob_delta_sum']:.6f} "
            f"activation={payload['primary_activation_score_mean']:.6f} "
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
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--selected_prompt_modes", type=Path, default=None)
    parser.add_argument("--v80_summary", type=Path, default=None)
    args = parser.parse_args()

    selected_prompt_modes_path = args.selected_prompt_modes
    if selected_prompt_modes_path is None:
        candidate = args.result_root / "selected-prompt-modes.json"
        if candidate.exists():
            selected_prompt_modes_path = candidate
    summary = build_summary(
        result_root=args.result_root,
        selected_prompt_modes_path=selected_prompt_modes_path,
        v80_summary_path=args.v80_summary,
    )
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(_render_report(summary))


if __name__ == "__main__":
    main()
