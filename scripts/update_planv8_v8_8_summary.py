#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
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


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


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


def _task_score(metrics: dict[str, Any]) -> float:
    for key in ("best_adapt_task_score", "task_score", "exact_match", "accuracy", "compute_reward"):
        if key in metrics:
            return _safe_float(metrics.get(key))
    return 0.0


def _route_live(metrics: dict[str, Any]) -> bool:
    prefix_stats = metrics.get("prefix_artifact_stats", {})
    if not isinstance(prefix_stats, dict):
        prefix_stats = {}
    memory_path_variant = str(metrics.get("pilot_memory_path_variant", "single_level"))
    prefix_attention_nontrivial_layer_count = int(metrics.get("prefix_attention_nontrivial_layer_count", 0) or 0)
    prefix_attention_mass_mean = _safe_float(metrics.get("prefix_attention_mass_mean", 0.0))
    cross_attn_gate_open_fraction = _safe_float(
        metrics.get("cross_attn_gate_open_fraction", prefix_stats.get("cross_attn_gate_open_fraction", 0.0))
    )
    reader_readout_effective_rank = _safe_float(metrics.get("reader_readout_effective_rank", 0.0))
    memory_token_attention_mass_mean = _safe_float(
        metrics.get("memory_token_attention_mass_mean", prefix_stats.get("memory_token_attention_mass_mean", 0.0))
    )
    short_slots = int(metrics.get("pilot_fuser_short_slots", 0) or 0)
    if memory_path_variant == "two_level":
        return short_slots > 0 and (
            reader_readout_effective_rank > 1.0
            or cross_attn_gate_open_fraction > 0.05
            or memory_token_attention_mass_mean > 0.01
        )
    return (
        prefix_attention_nontrivial_layer_count > 0
        or prefix_attention_mass_mean > 0.01
        or cross_attn_gate_open_fraction > 0.05
        or memory_token_attention_mass_mean > 0.01
    )


def _stable_training(events: list[dict[str, Any]]) -> bool:
    if not events:
        return True
    numeric_values: list[float] = []
    for event in events:
        for key in (
            "loss",
            "grad_norm_writer",
            "grad_norm_reader",
            "grad_norm_fuser",
            "grad_norm_projector",
            "grad_norm_prefix_projector",
            "grad_norm_receiver_lora",
        ):
            if key in event:
                numeric_values.append(_safe_float(event.get(key), float("nan")))
    clipped_fraction = float(
        sum(1 for event in events if bool(event.get("was_grad_clipped", False))) / len(events)
    )
    return all(math.isfinite(value) for value in numeric_values) and clipped_fraction < 0.95


def _load_floor_scores(path: Path) -> dict[str, float]:
    payload = _load_json(path)
    scores = payload.get("selected_qwen34_baseline_scores", {})
    return {
        task_name: _safe_float(scores.get(task_name))
        for task_name in ALL_TASKS
        if task_name in scores
    }


def _seed_task_summary(task_root: Path) -> dict[str, Any]:
    metrics = _load_json(task_root / "metrics.json")
    train_events = _load_train_events(task_root / "train_events.json")
    return {
        "task_score": _task_score(metrics),
        "route_live": _route_live(metrics),
        "stable_training": _stable_training(train_events),
        "task_metric_name": str(metrics.get("task_metric_name") or metrics.get("metric_name") or "").strip(),
    }


def _aggregate_task(seed_rows: list[dict[str, Any]], floor_score: float) -> dict[str, Any]:
    scores = [float(row["task_score"]) for row in seed_rows]
    deltas = [float(score - floor_score) for score in scores]
    return {
        "seed_rows": seed_rows,
        "task_score_mean": _mean(scores),
        "mean_delta_vs_floor": _mean(deltas),
        "positive_seed_count_vs_floor": int(sum(delta > 1.0e-12 for delta in deltas)),
        "non_regressive_seed_count_vs_floor": int(sum(delta >= -1.0e-12 for delta in deltas)),
        "route_live_seed_count": int(sum(bool(row["route_live"]) for row in seed_rows)),
        "stable_seed_count": int(sum(bool(row["stable_training"]) for row in seed_rows)),
    }


def build_summary(
    *,
    result_root: Path,
    selection_manifest_path: Path,
    v80_summary_path: Path,
    v87_summary_path: Path,
) -> dict[str, Any]:
    selection_manifest = _load_json(selection_manifest_path)
    v87_summary = _load_json(v87_summary_path)
    floor_scores = _load_floor_scores(v80_summary_path)
    seeds = [f"seed_{int(seed)}" for seed in selection_manifest.get("seeds", [61109, 61110, 61111])]
    branches: dict[str, Any] = {}

    for variant in selection_manifest.get("promoted_variants", []):
        variant_id = str(variant.get("variant_id", "")).strip()
        if not variant_id:
            continue
        tasks: dict[str, Any] = {}
        for task_name in ALL_TASKS:
            seed_rows: list[dict[str, Any]] = []
            for seed_name in seeds:
                task_root = result_root / variant_id / seed_name / task_name
                if not (task_root / "metrics.json").exists():
                    raise ValueError(
                        f"Missing PLANv8 V8-8 payload for variant={variant_id} seed={seed_name} task={task_name}."
                    )
                seed_summary = _seed_task_summary(task_root)
                seed_summary["seed"] = seed_name
                seed_rows.append(seed_summary)
            tasks[task_name] = _aggregate_task(seed_rows, floor_scores.get(task_name, 0.0))

        positive_primary_task_count = int(
            sum(tasks[task_name]["mean_delta_vs_floor"] > 1.0e-12 for task_name in PRIMARY_TASKS)
        )
        non_regressive_primary_task_count = int(
            sum(tasks[task_name]["mean_delta_vs_floor"] >= -1.0e-12 for task_name in PRIMARY_TASKS)
        )
        route_live_primary_task_count = int(
            sum(tasks[task_name]["route_live_seed_count"] >= 2 for task_name in PRIMARY_TASKS)
        )
        stable_primary_task_count = int(
            sum(tasks[task_name]["stable_seed_count"] >= len(seeds) for task_name in PRIMARY_TASKS)
        )
        primary_score_delta_sum = float(sum(tasks[task_name]["mean_delta_vs_floor"] for task_name in PRIMARY_TASKS))
        fever_delta_vs_floor = float(tasks["fever"]["mean_delta_vs_floor"])
        confirmation_success = bool(
            positive_primary_task_count >= 1
            and non_regressive_primary_task_count == len(PRIMARY_TASKS)
            and fever_delta_vs_floor >= -0.02
            and route_live_primary_task_count >= 1
        )
        branches[variant_id] = {
            "variant_id": variant_id,
            "source_phase": str(variant.get("source_phase", "")).strip(),
            "arm_id": str(variant.get("arm_id", "")).strip(),
            "interface_family": str(variant.get("interface_family", "")).strip(),
            "bridge_family": str(variant.get("bridge_family", "")).strip(),
            "auxiliary_family": str(variant.get("auxiliary_family", "")).strip(),
            "positive_primary_task_count": positive_primary_task_count,
            "non_regressive_primary_task_count": non_regressive_primary_task_count,
            "route_live_primary_task_count": route_live_primary_task_count,
            "stable_primary_task_count": stable_primary_task_count,
            "primary_score_delta_sum_vs_floor": primary_score_delta_sum,
            "fever_delta_vs_floor": fever_delta_vs_floor,
            "confirmation_success": confirmation_success,
            "ranking_key": [
                float(confirmation_success),
                float(positive_primary_task_count),
                float(non_regressive_primary_task_count),
                primary_score_delta_sum,
                float(route_live_primary_task_count),
                float(stable_primary_task_count),
                fever_delta_vs_floor,
            ],
            "tasks": tasks,
        }

    ranked_variants = sorted(
        (
            {
                "variant_id": variant_id,
                "source_phase": payload["source_phase"],
                "arm_id": payload["arm_id"],
                "confirmation_success": payload["confirmation_success"],
                "ranking_key": payload["ranking_key"],
            }
            for variant_id, payload in branches.items()
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )
    best_ranked_variant_id = ranked_variants[0]["variant_id"] if ranked_variants else ""
    confirmed_variants = [payload for payload in ranked_variants if bool(payload["confirmation_success"])]
    best_confirmed_variant_id = confirmed_variants[0]["variant_id"] if confirmed_variants else ""

    if best_confirmed_variant_id:
        best_confirmed = branches[best_confirmed_variant_id]
        comparison_conclusion = "multiseed_confirmation_success_open_v8_9_cdmi"
        recommended_next_step = "open_v8_9_cdmi"
    else:
        best_confirmed = branches.get(best_ranked_variant_id, {})
        comparison_conclusion = "multiseed_confirmation_failed_hold_v8_9"
        recommended_next_step = "hold_v8_9_confirmation_review"

    return {
        "phase": "V8-8",
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "selection_manifest_path": str(selection_manifest_path.resolve()),
        "v87_comparison_conclusion": str(v87_summary.get("comparison_conclusion", "")).strip(),
        "v87_recommended_next_step": str(v87_summary.get("recommended_next_step", "")).strip(),
        "floor_scores_by_task": floor_scores,
        "ranked_variants": ranked_variants,
        "best_ranked_variant_id": best_ranked_variant_id,
        "best_confirmed_variant_id": best_confirmed_variant_id,
        "best_confirmed_source_phase": str(best_confirmed.get("source_phase", "")).strip(),
        "best_confirmed_arm_id": str(best_confirmed.get("arm_id", "")).strip(),
        "best_confirmed_interface_family": str(best_confirmed.get("interface_family", "")).strip(),
        "best_confirmed_bridge_family": str(best_confirmed.get("bridge_family", "")).strip(),
        "best_confirmed_auxiliary_family": str(best_confirmed.get("auxiliary_family", "")).strip(),
        "branches": branches,
    }


def write_report(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# PLANv8 V8-8 Multi-Seed Confirmation",
        "",
        f"- Comparison conclusion: `{summary['comparison_conclusion']}`",
        f"- Recommended next step: `{summary['recommended_next_step']}`",
        f"- Best ranked variant: `{summary['best_ranked_variant_id']}`",
        f"- Best confirmed variant: `{summary['best_confirmed_variant_id']}`",
        "",
        "## Ranked Variants",
    ]
    for payload in summary["ranked_variants"]:
        lines.append(
            "- "
            f"`{payload['variant_id']}` source={payload['source_phase']} arm={payload['arm_id']} "
            f"confirmed={payload['confirmation_success']}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PLANv8 V8-8 multiseed confirmation summary.")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--v80_summary", required=True)
    parser.add_argument("--v87_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    args = parser.parse_args()

    summary = build_summary(
        result_root=Path(args.result_root),
        selection_manifest_path=Path(args.selection_manifest),
        v80_summary_path=Path(args.v80_summary),
        v87_summary_path=Path(args.v87_summary),
    )
    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary, Path(args.output_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
