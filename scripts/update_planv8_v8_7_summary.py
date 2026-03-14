#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
ALL_TASKS = ("gsm8k", "triviaqa", "fever")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _task_score_from_metrics(metrics: dict[str, Any]) -> float:
    for key in ("task_score", "mean_score", "accuracy", "exact_match", "compute_reward"):
        if key in metrics:
            return _safe_float(metrics.get(key))
    return 0.0


def _load_optional_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _load_v80_reference(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    scores = payload.get("selected_qwen34_baseline_scores", {})
    prompts = payload.get("selected_prompt_modes_by_task", {})
    return {
        "comparator_id": "m0_nomemory_qwen34",
        "scope": "planv8_selected_baseline_reference",
        "task_scores": {
            task_name: _safe_float(scores.get(task_name))
            for task_name in ALL_TASKS
            if task_name in scores
        },
        "prompt_variants": {
            task_name: str(prompts.get(task_name, "")).strip()
            for task_name in ALL_TASKS
            if str(prompts.get(task_name, "")).strip()
        },
        "recommended_next_step": str(payload.get("recommended_next_step", "")).strip(),
    }


def _load_v76_reference(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    best_variant_id = str(payload.get("best_confirmed_variant_id", "")).strip()
    branch = payload.get("branches", {}).get(best_variant_id, {})
    task_scores: dict[str, float] = {}
    for task_name in ALL_TASKS:
        task_payload = branch.get("tasks", {}).get(task_name, {})
        seed_rows = task_payload.get("seed_rows", [])
        values = [_safe_float(row.get("task_score")) for row in seed_rows if "task_score" in row]
        if values:
            task_scores[task_name] = float(sum(values) / len(values))
    return {
        "comparator_id": "m3_legacy_planv7_qwen25",
        "scope": "historical_multiseed_reference",
        "best_confirmed_variant_id": best_variant_id,
        "best_confirmed_promoted_arm_id": str(payload.get("best_confirmed_promoted_arm_id", "")).strip(),
        "task_scores": task_scores,
        "recommended_next_step": str(payload.get("recommended_next_step", "")).strip(),
    }


def _load_v86_reference(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    best_arm_id = str(
        payload.get("base_for_v8_7_arm_id")
        or payload.get("best_arm_id")
        or ""
    ).strip()
    arm_payload = payload.get("arm_summaries", {}).get(best_arm_id, {})
    task_scores: dict[str, float] = {}
    route_live: dict[str, bool] = {}
    prompt_variants: dict[str, str] = {}
    for task_name in ALL_TASKS:
        task_payload = arm_payload.get("tasks", {}).get(task_name, {})
        if task_payload:
            task_scores[task_name] = _safe_float(task_payload.get("task_score"))
            route_live[task_name] = bool(task_payload.get("route_live", False))
            prompt_variant = str(task_payload.get("prompt_variant", "")).strip()
            if prompt_variant:
                prompt_variants[task_name] = prompt_variant
    return {
        "comparator_id": "m4_best_v8_qwen34",
        "scope": "planv8_best_route_reference",
        "best_arm_id": best_arm_id,
        "selected_interface_family": str(payload.get("selected_interface_family_for_v8_7", "")).strip(),
        "selected_bridge_family": str(payload.get("selected_bridge_family_for_v8_7", "")).strip(),
        "selected_aux_family": str(payload.get("selected_aux_family_for_v8_7", "")).strip(),
        "task_scores": task_scores,
        "route_live_by_task": route_live,
        "prompt_variants": prompt_variants,
        "recommended_next_step": str(payload.get("recommended_next_step", "")).strip(),
    }


def _load_rag_reference(result_root: Path) -> dict[str, Any]:
    task_scores: dict[str, float] = {}
    metric_names: dict[str, str] = {}
    available_tasks: list[str] = []
    for task_name in ALL_TASKS:
        metrics = _load_optional_metrics(result_root / "m1_text_rag" / task_name / "metrics.json")
        if not metrics:
            continue
        task_scores[task_name] = _task_score_from_metrics(metrics)
        metric_names[task_name] = str(metrics.get("metric_name", metrics.get("benchmark_id", ""))).strip()
        available_tasks.append(task_name)
    return {
        "comparator_id": "m1_text_rag_qwen34",
        "scope": "planv8_split_live_text_rag",
        "task_scores": task_scores,
        "metric_names": metric_names,
        "available_tasks": available_tasks,
    }


def _load_memgen_reference(result_root: Path) -> dict[str, Any]:
    task_scores: dict[str, float] = {}
    num_predictions: dict[str, int] = {}
    available_tasks: list[str] = []
    for task_name in PRIMARY_TASKS:
        metrics = _load_optional_metrics(result_root / "m2_memgen" / task_name / "metrics.json")
        if not metrics:
            continue
        task_scores[task_name] = _task_score_from_metrics(metrics)
        num_predictions[task_name] = int(metrics.get("num_predictions", 0) or 0)
        available_tasks.append(task_name)
    return {
        "comparator_id": "m2_memgen_qwen34",
        "scope": "benchmark_native_memgen_mini_eval",
        "task_scores": task_scores,
        "num_predictions": num_predictions,
        "available_tasks": available_tasks,
        "non_isomorphic_scope": True,
    }


def _pairwise_delta(best: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, float] = {}
    positive_primary_task_count = 0
    non_regressive_primary_task_count = 0
    for task_name in ALL_TASKS:
        if task_name not in best["task_scores"] or task_name not in other["task_scores"]:
            continue
        delta = float(best["task_scores"][task_name] - other["task_scores"][task_name])
        deltas[task_name] = delta
        if task_name in PRIMARY_TASKS:
            if delta > 0.0:
                positive_primary_task_count += 1
            if delta >= -1e-6:
                non_regressive_primary_task_count += 1
    aggregate_primary_delta = float(sum(deltas.get(task_name, 0.0) for task_name in PRIMARY_TASKS))
    return {
        "task_deltas": deltas,
        "aggregate_primary_delta": aggregate_primary_delta,
        "positive_primary_task_count": positive_primary_task_count,
        "non_regressive_primary_task_count": non_regressive_primary_task_count,
    }


def build_summary(
    *,
    result_root: Path,
    v80_summary_path: Path,
    v76_summary_path: Path,
    v86_summary_path: Path,
) -> dict[str, Any]:
    floor = _load_v80_reference(v80_summary_path)
    rag = _load_rag_reference(result_root)
    memgen = _load_memgen_reference(result_root)
    legacy = _load_v76_reference(v76_summary_path)
    best_v8 = _load_v86_reference(v86_summary_path)

    best_vs_floor = _pairwise_delta(best_v8, floor)
    best_vs_rag = _pairwise_delta(best_v8, rag)
    best_vs_legacy = _pairwise_delta(best_v8, legacy)
    memgen_context = {
        "available_primary_tasks": list(memgen["available_tasks"]),
        "task_deltas": {
            task_name: float(best_v8["task_scores"][task_name] - memgen["task_scores"][task_name])
            for task_name in PRIMARY_TASKS
            if task_name in best_v8["task_scores"] and task_name in memgen["task_scores"]
        },
        "non_isomorphic_scope": True,
    }
    route_live_primary_task_count = sum(
        1 for task_name in PRIMARY_TASKS if bool(best_v8["route_live_by_task"].get(task_name, False))
    )
    beats_floor = (
        best_vs_floor["positive_primary_task_count"] >= 1
        and best_vs_floor["non_regressive_primary_task_count"] == len(PRIMARY_TASKS)
    )
    beats_rag = (
        best_vs_rag["positive_primary_task_count"] >= 1
        and best_vs_rag["non_regressive_primary_task_count"] == len(PRIMARY_TASKS)
    )
    route_real = route_live_primary_task_count >= 1

    if beats_floor and beats_rag and route_real:
        comparison_conclusion = "comparators_support_open_v8_8_multiseed_confirmation"
        recommended_next_step = "open_v8_8_multiseed_confirmation"
    elif beats_floor and route_real:
        comparison_conclusion = "comparators_floor_cleared_but_text_rag_blocks_v8_8"
        recommended_next_step = "hold_v8_8_rag_gap"
    else:
        comparison_conclusion = "comparators_do_not_support_v8_8"
        recommended_next_step = "hold_v8_8_comparator_review"

    return {
        "phase": "V8-7",
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "base_for_v8_8_arm_id": best_v8["best_arm_id"],
        "selected_interface_family_for_v8_8": best_v8["selected_interface_family"],
        "selected_bridge_family_for_v8_8": best_v8["selected_bridge_family"],
        "selected_aux_family_for_v8_8": best_v8["selected_aux_family"],
        "route_live_primary_task_count": route_live_primary_task_count,
        "best_v8_beats_floor": beats_floor,
        "best_v8_beats_text_rag": beats_rag,
        "comparators": {
            floor["comparator_id"]: floor,
            rag["comparator_id"]: rag,
            memgen["comparator_id"]: memgen,
            legacy["comparator_id"]: legacy,
            best_v8["comparator_id"]: best_v8,
        },
        "best_v8_vs_nomemory": best_vs_floor,
        "best_v8_vs_text_rag": best_vs_rag,
        "best_v8_vs_memgen_context": memgen_context,
        "best_v8_vs_legacy": best_vs_legacy,
    }


def write_report(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# PLANv8 V8-7 Comparator Summary",
        "",
        f"- Comparison conclusion: `{summary['comparison_conclusion']}`",
        f"- Recommended next step: `{summary['recommended_next_step']}`",
        f"- Base arm for V8-8: `{summary['base_for_v8_8_arm_id']}`",
        f"- Route-live primary task count: `{summary['route_live_primary_task_count']}`",
        "",
        "## Comparator Deltas",
        "",
        f"- Vs floor aggregate primary delta: `{summary['best_v8_vs_nomemory']['aggregate_primary_delta']:.6f}`",
        f"- Vs text RAG aggregate primary delta: `{summary['best_v8_vs_text_rag']['aggregate_primary_delta']:.6f}`",
    ]
    if summary["best_v8_vs_memgen_context"]["task_deltas"]:
        lines.extend(
            [
                "",
                "## MemGen Context",
                "",
                f"- Non-isomorphic scope: `{summary['best_v8_vs_memgen_context']['non_isomorphic_scope']}`",
                f"- Available primary tasks: `{', '.join(summary['best_v8_vs_memgen_context']['available_primary_tasks'])}`",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PLANv8 V8-7 comparator summary.")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--v80_summary", required=True)
    parser.add_argument("--v76_summary", required=True)
    parser.add_argument("--v86_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    args = parser.parse_args()

    summary = build_summary(
        result_root=Path(args.result_root),
        v80_summary_path=Path(args.v80_summary),
        v76_summary_path=Path(args.v76_summary),
        v86_summary_path=Path(args.v86_summary),
    )
    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary, Path(args.output_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
