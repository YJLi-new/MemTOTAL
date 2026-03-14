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
CONTROL_ARM_ID = "b0_no_bridge"
ARM_ORDER = (
    "b0_no_bridge",
    "b1_q16_s16",
    "b2_q32_s16",
    "b3_q32_s8",
    "b4_q48_s16_x96",
)
BRIDGE_ARMS = ARM_ORDER[1:]
ARM_METADATA = {
    "b0_no_bridge": {"bridge_family": "BR0"},
    "b1_q16_s16": {"bridge_family": "BR1"},
    "b2_q32_s16": {"bridge_family": "BR2"},
    "b3_q32_s8": {"bridge_family": "BR3"},
    "b4_q48_s16_x96": {"bridge_family": "BR2"},
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


def _v84_reference(v84_summary_path: Path | None) -> dict[str, Any]:
    if v84_summary_path is None or not v84_summary_path.exists():
        return {}
    return _load_json(v84_summary_path)


def _arm_task_payload(result_root: Path, arm_id: str, task_name: str) -> dict[str, Any]:
    task_root = result_root / arm_id / task_name
    metrics = _load_json(task_root / "metrics.json")
    train_events = _load_train_events(task_root / "train_events.json")
    case_rows = _load_case_rows(task_root / "task_case_dump.jsonl")
    task_score = _task_score(metrics)
    macro_f1 = _macro_f1(metrics)
    writer_slots = _safe_int(
        metrics.get("writer_memory_slots", metrics.get("pilot_bridge_expected_input_slots", 0))
    )
    reader_queries = _safe_int(metrics.get("pilot_reader_num_queries", 0))
    short_slots = _safe_int(metrics.get("pilot_fuser_short_slots", 0))
    compression_ratio = 1.0
    if writer_slots > 0 and short_slots > 0:
        compression_ratio = float(short_slots / writer_slots)
    compute_reduction_fraction = 0.0
    if writer_slots > 0 and short_slots > 0:
        compute_reduction_fraction = float(1.0 - compression_ratio)
    reader_grad_norms = [_safe_float(event.get("grad_norm_reader", 0.0)) for event in train_events]
    fuser_grad_norms = [_safe_float(event.get("grad_norm_fuser", 0.0)) for event in train_events]
    loss_values = [_safe_float(event.get("loss", 0.0)) for event in train_events]
    clipped_fraction = 0.0
    if train_events:
        clipped_fraction = float(
            sum(1 for event in train_events if bool(event.get("was_grad_clipped", False))) / len(train_events)
        )
    stable_training = True
    if train_events:
        stable_training = _all_finite(loss_values + reader_grad_norms + fuser_grad_norms) and clipped_fraction < 0.95
    reader_readout_effective_rank = _safe_float(metrics.get("reader_readout_effective_rank", 0.0))
    reader_readout_pairwise_cosine_mean = _safe_float(
        metrics.get("reader_readout_pairwise_cosine_mean", 0.0)
    )
    prompt_variant = str(metrics.get("pilot_prompt_variant") or metrics.get("prompt_variant") or "")
    memory_path_variant = str(metrics.get("pilot_memory_path_variant", "single_level"))
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
            and _safe_int(metrics.get("memory_tokens_count", 0)) > 0
        )
    )
    return {
        "arm_id": arm_id,
        "task_name": task_name,
        "prompt_variant": prompt_variant,
        "task_score": task_score,
        "macro_f1": macro_f1,
        "writer_slots": writer_slots,
        "reader_queries": reader_queries,
        "short_slots": short_slots,
        "compression_ratio": compression_ratio,
        "compute_reduction_fraction": compute_reduction_fraction,
        "reader_readout_effective_rank": reader_readout_effective_rank,
        "reader_readout_pairwise_cosine_mean": reader_readout_pairwise_cosine_mean,
        "reader_grad_norm_median": _median(reader_grad_norms),
        "fuser_grad_norm_median": _median(fuser_grad_norms),
        "stable_training": stable_training,
        "route_live": route_live,
        "answer_logprob_with_memory_mean": _mean_row_value(case_rows, "answer_logprob_with_memory"),
        "memory_path_variant": memory_path_variant,
        "bridge_family": str(metrics.get("pilot_active_bridge_family", ARM_METADATA[arm_id]["bridge_family"])),
        "case_rows": case_rows,
    }


def build_summary(
    *,
    result_root: Path,
    selected_prompt_modes_path: Path | None = None,
    v84_summary_path: Path | None = None,
) -> dict[str, Any]:
    selected_prompt_modes = _selected_prompt_modes(selected_prompt_modes_path)
    v84_reference = _v84_reference(v84_summary_path)
    base_arm_id = str(
        v84_reference.get("base_for_v8_5_arm_id") or v84_reference.get("best_arm_id") or ""
    ).strip()
    selected_interface_family = str(
        v84_reference.get("selected_interface_family_for_v8_5")
        or v84_reference.get("best_interface_family")
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
        bridge_family = str(gsm8k["bridge_family"] or ARM_METADATA[arm_id]["bridge_family"])
        input_slots = max(
            _safe_int(gsm8k["writer_slots"], 0),
            _safe_int(triviaqa["writer_slots"], 0),
            _safe_int(fever["writer_slots"], 0),
        )
        short_slots = max(
            _safe_int(gsm8k["short_slots"], 0),
            _safe_int(triviaqa["short_slots"], 0),
            _safe_int(fever["short_slots"], 0),
        )
        compression_ratio = 1.0
        if input_slots > 0 and short_slots > 0:
            compression_ratio = float(short_slots / input_slots)
        compute_reduction_fraction = 0.0
        if input_slots > 0 and short_slots > 0:
            compute_reduction_fraction = float(1.0 - compression_ratio)
        material_compression = bool(
            arm_id != CONTROL_ARM_ID
            and short_slots > 0
            and input_slots > short_slots
            and compute_reduction_fraction >= 0.25
        )
        acceptance_qualified = bool(
            arm_id != CONTROL_ARM_ID
            and stable_all_tasks
            and stable_primary_task_count == len(PRIMARY_TASKS)
            and route_live_primary_task_count >= 1
            and material_compression
            and positive_primary_score_task_count >= 1
            and nonnegative_primary_score_task_count == len(PRIMARY_TASKS)
        )
        arm_summaries[arm_id] = {
            "arm_id": arm_id,
            "bridge_family": bridge_family,
            "memory_path_variant": str(gsm8k["memory_path_variant"]),
            "input_slots": input_slots,
            "reader_queries": max(
                _safe_int(gsm8k["reader_queries"], 0),
                _safe_int(triviaqa["reader_queries"], 0),
                _safe_int(fever["reader_queries"], 0),
            ),
            "short_slots": short_slots,
            "compression_ratio": compression_ratio,
            "compute_reduction_fraction": compute_reduction_fraction,
            "material_compression": material_compression,
            "primary_task_score_delta_sum": primary_task_score_delta_sum,
            "primary_answer_logprob_delta_sum": primary_answer_logprob_delta_sum,
            "positive_primary_score_task_count": positive_primary_score_task_count,
            "nonnegative_primary_score_task_count": nonnegative_primary_score_task_count,
            "route_live_primary_task_count": route_live_primary_task_count,
            "stable_primary_task_count": stable_primary_task_count,
            "stable_all_tasks": stable_all_tasks,
            "fever_task_score_delta_vs_control": float(fever["task_score_delta_vs_control"]),
            "acceptance_qualified": acceptance_qualified,
            "ranking_key": [
                float(positive_primary_score_task_count),
                float(nonnegative_primary_score_task_count),
                primary_task_score_delta_sum,
                primary_answer_logprob_delta_sum,
                float(route_live_primary_task_count),
                float(stable_primary_task_count),
                compute_reduction_fraction,
                -compression_ratio,
                float(fever["task_score_delta_vs_control"]),
            ],
            "tasks": per_task,
        }

    bridge_arm_ranking = sorted(
        (
            {
                "arm_id": arm_id,
                "bridge_family": summary["bridge_family"],
                "input_slots": summary["input_slots"],
                "reader_queries": summary["reader_queries"],
                "short_slots": summary["short_slots"],
                "compression_ratio": summary["compression_ratio"],
                "compute_reduction_fraction": summary["compute_reduction_fraction"],
                "primary_task_score_delta_sum": summary["primary_task_score_delta_sum"],
                "primary_answer_logprob_delta_sum": summary["primary_answer_logprob_delta_sum"],
                "acceptance_qualified": summary["acceptance_qualified"],
                "ranking_key": summary["ranking_key"],
            }
            for arm_id, summary in arm_summaries.items()
            if arm_id in BRIDGE_ARMS
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )
    best_bridge_arm_id = bridge_arm_ranking[0]["arm_id"]
    best_bridge_arm = arm_summaries[best_bridge_arm_id]
    no_bridge_control = arm_summaries[CONTROL_ARM_ID]

    if best_bridge_arm["acceptance_qualified"]:
        comparison_conclusion = "bridge_compression_preserves_gains_open_v8_6_writer_aux"
        recommended_next_step = "open_v8_6_writer_aux"
        best_arm_id = best_bridge_arm_id
        best_arm_acceptance = True
        selected_bridge_family_for_v8_6 = str(best_bridge_arm["bridge_family"])
    else:
        comparison_conclusion = "bridge_compression_hurts_keep_full_route_open_v8_6_writer_aux_full_route"
        recommended_next_step = "open_v8_6_writer_aux_full_route"
        best_arm_id = CONTROL_ARM_ID
        best_arm_acceptance = False
        selected_bridge_family_for_v8_6 = "BR0"

    return {
        "phase": "V8-5",
        "selected_prompt_modes_by_task": selected_prompt_modes,
        "control_arm_id": CONTROL_ARM_ID,
        "control_scores_by_task": {
            task_name: float(control_by_task[task_name]["task_score"])
            for task_name in ALL_TASKS
        },
        "v84_best_arm_id": str(v84_reference.get("best_arm_id", "")).strip(),
        "v84_base_for_v8_5_arm_id": base_arm_id,
        "v84_selected_interface_family_for_v8_5": selected_interface_family,
        "v84_recommended_next_step": str(v84_reference.get("recommended_next_step", "")).strip(),
        "arm_summaries": arm_summaries,
        "bridge_arm_ranking": bridge_arm_ranking,
        "best_bridge_arm_id": best_bridge_arm_id,
        "best_arm_id": best_arm_id,
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "base_for_v8_6_arm_id": best_arm_id,
        "selected_interface_family_for_v8_6": selected_interface_family,
        "selected_bridge_family_for_v8_6": selected_bridge_family_for_v8_6,
        "best_arm_acceptance_qualified": bool(best_arm_acceptance),
        "no_bridge_control_summary": {
            "arm_id": CONTROL_ARM_ID,
            "primary_task_score_delta_sum": no_bridge_control["primary_task_score_delta_sum"],
            "primary_answer_logprob_delta_sum": no_bridge_control["primary_answer_logprob_delta_sum"],
        },
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv8 V8-5 Bridge Revisit",
        "",
        "## Decision",
        f"- `comparison_conclusion = {summary['comparison_conclusion']}`",
        f"- `recommended_next_step = {summary['recommended_next_step']}`",
        f"- `best_arm_id = {summary['best_arm_id']}`",
        f"- `best_bridge_arm_id = {summary['best_bridge_arm_id']}`",
        "",
        "## Bridge Ranking",
    ]
    for payload in summary["bridge_arm_ranking"]:
        lines.append(
            "- "
            f"`{payload['arm_id']}` "
            f"family={payload['bridge_family']} "
            f"input_slots={payload['input_slots']} "
            f"queries={payload['reader_queries']} "
            f"short_slots={payload['short_slots']} "
            f"score_sum={payload['primary_task_score_delta_sum']:.6f} "
            f"compression_ratio={payload['compression_ratio']:.4f} "
            f"acceptance={payload['acceptance_qualified']}"
        )
    lines.extend(
        [
            "",
            "## No-Bridge Control",
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
    parser.add_argument("--v84_summary", type=Path, default=None)
    args = parser.parse_args()

    selected_prompt_modes_path = args.selected_prompt_modes
    if selected_prompt_modes_path is None:
        candidate = args.result_root / "selected-prompt-modes.json"
        if candidate.exists():
            selected_prompt_modes_path = candidate

    v84_summary_path = args.v84_summary
    if v84_summary_path is None:
        candidate = args.result_root / "v8-4-summary.reference.json"
        if candidate.exists():
            v84_summary_path = candidate

    summary = build_summary(
        result_root=args.result_root,
        selected_prompt_modes_path=selected_prompt_modes_path,
        v84_summary_path=v84_summary_path,
    )
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(_render_report(summary))


if __name__ == "__main__":
    main()
