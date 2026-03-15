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
CONTROL_ARM_ID = "w0_oracle64"
ARM_ORDER = (
    "w0_oracle64",
    "w1_ext2layer64_lr2e5",
    "w2_ext3layer64_lr2e5",
    "w3_ext3layer96_lr1e5",
    "w4_ext3layer64_lr5e5",
)
TRAINABLE_ARMS = ARM_ORDER[1:]
ARM_METADATA = {
    "w0_oracle64": {"writer_family": "EW0", "memory_slots": 64, "transformer_layers": 2, "writer_lr": 0.0},
    "w1_ext2layer64_lr2e5": {"writer_family": "EW1", "memory_slots": 64, "transformer_layers": 2, "writer_lr": 2.0e-5},
    "w2_ext3layer64_lr2e5": {"writer_family": "EW2", "memory_slots": 64, "transformer_layers": 3, "writer_lr": 2.0e-5},
    "w3_ext3layer96_lr1e5": {"writer_family": "EW3", "memory_slots": 96, "transformer_layers": 3, "writer_lr": 1.0e-5},
    "w4_ext3layer64_lr5e5": {"writer_family": "EW2", "memory_slots": 64, "transformer_layers": 3, "writer_lr": 5.0e-5},
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


def _v83_reference(v83_summary_path: Path | None) -> dict[str, Any]:
    if v83_summary_path is None or not v83_summary_path.exists():
        return {}
    return _load_json(v83_summary_path)


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
    writer_grad_norms = [
        max(
            _safe_float(event.get("grad_norm_writer", 0.0)),
            _safe_float(event.get("grad_norm_support_encoder", 0.0)),
            _safe_float(event.get("grad_norm_projector", event.get("grad_norm_prefix_projector", 0.0))),
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
        stable_training = _all_finite(loss_values + writer_grad_norms) and clipped_fraction < 0.95
    task_score = _task_score(metrics)
    macro_f1 = _macro_f1(metrics)
    answer_logprob_with_memory_mean = _mean_row_value(case_rows, "answer_logprob_with_memory")
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
    )
    return {
        "arm_id": arm_id,
        "task_name": task_name,
        "prompt_variant": prompt_variant,
        "task_score": task_score,
        "macro_f1": macro_f1,
        "answer_logprob_with_memory_mean": answer_logprob_with_memory_mean,
        "memory_tokens_count": memory_tokens_count,
        "prefix_attention_nontrivial_layer_count": prefix_attention_nontrivial_layer_count,
        "prefix_attention_mass_mean": prefix_attention_mass_mean,
        "cross_attn_gate_open_fraction": cross_attn_gate_open_fraction,
        "memory_token_attention_mass_mean": memory_token_attention_mass_mean,
        "activation_score": activation_score,
        "stable_training": stable_training,
        "route_live": route_live,
        "writer_grad_norm_median": _median(writer_grad_norms),
        "writer_grad_nonzero_steps": int(sum(1 for value in writer_grad_norms if value > 0.0)),
        "train_steps": _safe_int(metrics.get("pilot_train_steps", len(train_events))),
        "clipped_fraction": clipped_fraction,
        "train_phase_final": str(metrics.get("train_final_phase", "")),
        "pilot_trainable_variant": str(metrics.get("pilot_trainable_variant", "")),
        "pilot_writer_family": str(metrics.get("pilot_writer_family", "")),
        "case_rows": case_rows,
    }


def build_summary(
    *,
    result_root: Path,
    selected_prompt_modes_path: Path | None = None,
    v83_summary_path: Path | None = None,
) -> dict[str, Any]:
    selected_prompt_modes = _selected_prompt_modes(selected_prompt_modes_path)
    v83_reference = _v83_reference(v83_summary_path)
    base_arm_id = str(
        v83_reference.get("base_for_v8_4_arm_id") or v83_reference.get("best_arm_id") or ""
    ).strip()
    selected_interface_family = str(
        v83_reference.get("selected_interface_family_for_v8_4")
        or v83_reference.get("best_interface_family")
        or ""
    ).strip()

    control_by_task = {
        task_name: _arm_task_payload(result_root, CONTROL_ARM_ID, task_name)
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
        acceptance_qualified = bool(
            arm_id != CONTROL_ARM_ID
            and stable_all_tasks
            and stable_primary_task_count == len(PRIMARY_TASKS)
            and route_live_primary_task_count >= 1
            and positive_primary_score_task_count >= 1
            and nonnegative_primary_score_task_count == len(PRIMARY_TASKS)
        )
        metadata = ARM_METADATA[arm_id]
        arm_summaries[arm_id] = {
            "arm_id": arm_id,
            "interface_family": selected_interface_family,
            "writer_family": metadata["writer_family"],
            "memory_slots": metadata["memory_slots"],
            "transformer_layers": metadata["transformer_layers"],
            "writer_learning_rate": float(metadata["writer_lr"]),
            "primary_task_score_delta_sum": primary_task_score_delta_sum,
            "primary_answer_logprob_delta_sum": primary_answer_logprob_delta_sum,
            "positive_primary_score_task_count": positive_primary_score_task_count,
            "nonnegative_primary_score_task_count": nonnegative_primary_score_task_count,
            "route_live_primary_task_count": route_live_primary_task_count,
            "stable_primary_task_count": stable_primary_task_count,
            "stable_all_tasks": stable_all_tasks,
            "primary_activation_score_mean": primary_activation_score_mean,
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
                float(route_live_primary_task_count),
                float(stable_primary_task_count),
                float(fever["task_score_delta_vs_control"]),
            ],
            "tasks": per_task,
        }

    trainable_arm_ranking = sorted(
        (
            {
                "arm_id": arm_id,
                "interface_family": summary["interface_family"],
                "writer_family": summary["writer_family"],
                "memory_slots": summary["memory_slots"],
                "transformer_layers": summary["transformer_layers"],
                "writer_learning_rate": summary["writer_learning_rate"],
                "primary_task_score_delta_sum": summary["primary_task_score_delta_sum"],
                "primary_answer_logprob_delta_sum": summary["primary_answer_logprob_delta_sum"],
                "primary_activation_score_mean": summary["primary_activation_score_mean"],
                "acceptance_qualified": summary["acceptance_qualified"],
                "ranking_key": summary["ranking_key"],
            }
            for arm_id, summary in arm_summaries.items()
            if arm_id in TRAINABLE_ARMS
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )
    best_trainable_arm_id = trainable_arm_ranking[0]["arm_id"]
    best_trainable_arm = arm_summaries[best_trainable_arm_id]
    oracle_control = arm_summaries[CONTROL_ARM_ID]

    if best_trainable_arm["acceptance_qualified"]:
        comparison_conclusion = "external_writer_beats_oracle_open_v8_5_bridge"
        recommended_next_step = "open_v8_5_bridge"
        best_arm_id = best_trainable_arm_id
        best_arm_acceptance = True
    else:
        comparison_conclusion = "external_writer_oracle_still_best_open_v8_7_comparators"
        recommended_next_step = "open_v8_7_comparators"
        best_arm_id = CONTROL_ARM_ID
        best_arm_acceptance = False

    return {
        "phase": "V8-4",
        "selected_prompt_modes_by_task": selected_prompt_modes,
        "control_arm_id": CONTROL_ARM_ID,
        "control_scores_by_task": {
            task_name: float(control_by_task[task_name]["task_score"])
            for task_name in ALL_TASKS
        },
        "v83_best_arm_id": str(v83_reference.get("best_arm_id", "")).strip(),
        "v83_base_for_v8_4_arm_id": base_arm_id,
        "v83_selected_interface_family_for_v8_4": selected_interface_family,
        "v83_recommended_next_step": str(v83_reference.get("recommended_next_step", "")).strip(),
        "arm_summaries": arm_summaries,
        "trainable_arm_ranking": trainable_arm_ranking,
        "best_trainable_arm_id": best_trainable_arm_id,
        "best_arm_id": best_arm_id,
        "best_interface_family": selected_interface_family,
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "base_for_v8_5_arm_id": best_arm_id,
        "selected_interface_family_for_v8_5": selected_interface_family,
        "best_arm_acceptance_qualified": bool(best_arm_acceptance),
        "oracle_control_summary": {
            "arm_id": CONTROL_ARM_ID,
            "primary_task_score_delta_sum": oracle_control["primary_task_score_delta_sum"],
            "primary_answer_logprob_delta_sum": oracle_control["primary_answer_logprob_delta_sum"],
        },
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv8 V8-4 External Writer",
        "",
        "## Decision",
        f"- `comparison_conclusion = {summary['comparison_conclusion']}`",
        f"- `recommended_next_step = {summary['recommended_next_step']}`",
        f"- `best_arm_id = {summary['best_arm_id']}`",
        f"- `best_trainable_arm_id = {summary['best_trainable_arm_id']}`",
        "",
        "## Trainable Ranking",
    ]
    for payload in summary["trainable_arm_ranking"]:
        lines.append(
            "- "
            f"`{payload['arm_id']}` "
            f"family={payload['writer_family']} "
            f"slots={payload['memory_slots']} "
            f"blocks={payload['transformer_layers']} "
            f"score_sum={payload['primary_task_score_delta_sum']:.6f} "
            f"logprob_sum={payload['primary_answer_logprob_delta_sum']:.6f} "
            f"acceptance={payload['acceptance_qualified']}"
        )
    lines.extend(
        [
            "",
            "## Oracle Control",
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
    parser.add_argument("--v83_summary", type=Path, default=None)
    args = parser.parse_args()

    selected_prompt_modes_path = args.selected_prompt_modes
    if selected_prompt_modes_path is None:
        candidate = args.result_root / "selected-prompt-modes.json"
        if candidate.exists():
            selected_prompt_modes_path = candidate

    v83_summary_path = args.v83_summary
    if v83_summary_path is None:
        candidate = args.result_root / "v8-3-summary.reference.json"
        if candidate.exists():
            v83_summary_path = candidate

    summary = build_summary(
        result_root=args.result_root,
        selected_prompt_modes_path=selected_prompt_modes_path,
        v83_summary_path=v83_summary_path,
    )
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(_render_report(summary))


if __name__ == "__main__":
    main()
