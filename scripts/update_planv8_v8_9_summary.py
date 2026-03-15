#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

CONDITION_SPECS = {
    "c0_math_self": {"task_name": "gsm8k", "label": "math_self"},
    "c1_trivia_self": {"task_name": "triviaqa", "label": "trivia_self"},
    "c2_joint_math": {"task_name": "gsm8k", "label": "joint_math"},
    "c3_joint_trivia": {"task_name": "triviaqa", "label": "joint_trivia"},
    "c4_math_support_on_trivia": {"task_name": "triviaqa", "label": "math_support_on_trivia"},
    "c5_trivia_support_on_math": {"task_name": "gsm8k", "label": "trivia_support_on_math"},
}
PRIMARY_TASKS = ("gsm8k", "triviaqa")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
        for task_name in PRIMARY_TASKS
        if task_name in scores
    }


def _condition_summary(condition_root: Path) -> dict[str, Any]:
    metrics = _load_json(condition_root / "metrics.json")
    train_events = _load_train_events(condition_root / "train_events.json")
    prefix_stats = metrics.get("prefix_artifact_stats", {})
    if not isinstance(prefix_stats, dict):
        prefix_stats = {}
    return {
        "task_name": str(metrics.get("benchmark_id", "")).strip(),
        "task_score": _task_score(metrics),
        "route_live": _route_live(metrics),
        "stable_training": _stable_training(train_events),
        "metric_name": str(metrics.get("task_metric_name") or metrics.get("metric_name") or "").strip(),
        "memory_path_variant": str(metrics.get("pilot_memory_path_variant", "")).strip(),
        "memory_consumer_mode": str(metrics.get("pilot_memory_consumer_mode", "")).strip(),
        "short_slots": int(metrics.get("pilot_fuser_short_slots", 0) or 0),
        "reader_readout_effective_rank": _safe_float(metrics.get("reader_readout_effective_rank", 0.0)),
        "prefix_attention_mass_mean": _safe_float(metrics.get("prefix_attention_mass_mean", 0.0)),
        "memory_token_attention_mass_mean": _safe_float(
            metrics.get("memory_token_attention_mass_mean", prefix_stats.get("memory_token_attention_mass_mean", 0.0))
        ),
        "cross_attn_gate_open_fraction": _safe_float(
            metrics.get("cross_attn_gate_open_fraction", prefix_stats.get("cross_attn_gate_open_fraction", 0.0))
        ),
        "prefix_attention_nontrivial_layer_count": int(
            metrics.get("prefix_attention_nontrivial_layer_count", 0) or 0
        ),
    }


def build_summary(
    *,
    result_root: Path,
    manifest_path: Path,
    v80_summary_path: Path,
    v88_summary_path: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    v88_summary = _load_json(v88_summary_path)
    floor_scores = _load_floor_scores(v80_summary_path)
    conditions = {
        condition_id: _condition_summary(result_root / condition_id)
        for condition_id in CONDITION_SPECS
    }

    joint_math_delta_vs_self = float(
        conditions["c2_joint_math"]["task_score"] - conditions["c0_math_self"]["task_score"]
    )
    joint_trivia_delta_vs_self = float(
        conditions["c3_joint_trivia"]["task_score"] - conditions["c1_trivia_self"]["task_score"]
    )
    math_support_penalty_on_trivia = float(
        conditions["c4_math_support_on_trivia"]["task_score"] - conditions["c3_joint_trivia"]["task_score"]
    )
    trivia_support_penalty_on_math = float(
        conditions["c5_trivia_support_on_math"]["task_score"] - conditions["c2_joint_math"]["task_score"]
    )
    negative_transfer_rate = float(
        mean(
            [
                float(joint_math_delta_vs_self < -1.0e-12),
                float(joint_trivia_delta_vs_self < -1.0e-12),
            ]
        )
    )
    cross_domain_leakage_rate = float(
        mean(
            [
                float(math_support_penalty_on_trivia < -1.0e-12),
                float(trivia_support_penalty_on_math < -1.0e-12),
            ]
        )
    )
    route_live_joint_task_count = int(
        sum(
            bool(conditions[condition_id]["route_live"])
            for condition_id in ("c2_joint_math", "c3_joint_trivia")
        )
    )
    domain_conditioned_gate_shift = abs(
        float(conditions["c2_joint_math"]["cross_attn_gate_open_fraction"])
        - float(conditions["c3_joint_trivia"]["cross_attn_gate_open_fraction"])
    )
    domain_conditioned_attention_shift = abs(
        float(conditions["c2_joint_math"]["prefix_attention_mass_mean"])
        - float(conditions["c3_joint_trivia"]["prefix_attention_mass_mean"])
    )
    domain_conditioned_memory_token_shift = abs(
        float(conditions["c2_joint_math"]["memory_token_attention_mass_mean"])
        - float(conditions["c3_joint_trivia"]["memory_token_attention_mass_mean"])
    )
    joint_vs_floor = {
        "gsm8k": float(conditions["c2_joint_math"]["task_score"] - floor_scores.get("gsm8k", 0.0)),
        "triviaqa": float(conditions["c3_joint_trivia"]["task_score"] - floor_scores.get("triviaqa", 0.0)),
    }
    compression_leakage_risk_flag = bool(
        str(manifest.get("source_bridge_family", "")).strip() not in {"", "BR0"}
        and cross_domain_leakage_rate > 0.0
    )

    if negative_transfer_rate == 0.0 and cross_domain_leakage_rate == 0.0 and route_live_joint_task_count >= 1:
        comparison_conclusion = "cdmi_profile_complete_paper_closeout_ready"
        recommended_next_step = "assemble_paper_closeout"
    elif route_live_joint_task_count >= 1:
        comparison_conclusion = "cdmi_profile_complete_with_negative_transfer"
        recommended_next_step = "assemble_paper_closeout_with_cdmi_risk_memo"
    else:
        comparison_conclusion = "cdmi_profile_complete_but_route_not_live"
        recommended_next_step = "assemble_paper_closeout_with_route_caveat"

    return {
        "phase": "V8-9",
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "best_confirmed_variant_id": str(manifest.get("best_confirmed_variant_id", "")).strip(),
        "source_phase": str(manifest.get("source_phase", "")).strip(),
        "source_run_root": str(manifest.get("source_run_root", "")).strip(),
        "source_arm_id": str(manifest.get("source_arm_id", "")).strip(),
        "source_interface_family": str(manifest.get("source_interface_family", "")).strip(),
        "source_bridge_family": str(manifest.get("source_bridge_family", "")).strip(),
        "source_auxiliary_family": str(manifest.get("source_auxiliary_family", "")).strip(),
        "source_prompt_variants": dict(manifest.get("source_prompt_variants", {})),
        "v88_comparison_conclusion": str(v88_summary.get("comparison_conclusion", "")).strip(),
        "v88_recommended_next_step": str(v88_summary.get("recommended_next_step", "")).strip(),
        "floor_scores_by_task": floor_scores,
        "joint_delta_vs_self": {
            "gsm8k": joint_math_delta_vs_self,
            "triviaqa": joint_trivia_delta_vs_self,
        },
        "joint_delta_vs_floor": joint_vs_floor,
        "cross_support_penalties": {
            "math_support_on_trivia": math_support_penalty_on_trivia,
            "trivia_support_on_math": trivia_support_penalty_on_math,
        },
        "negative_transfer_rate": negative_transfer_rate,
        "cross_domain_leakage_rate": cross_domain_leakage_rate,
        "cross_domain_leakage_detected": bool(cross_domain_leakage_rate > 0.0),
        "domain_conditioned_gate_shift": domain_conditioned_gate_shift,
        "domain_conditioned_attention_shift": domain_conditioned_attention_shift,
        "domain_conditioned_memory_token_shift": domain_conditioned_memory_token_shift,
        "route_live_joint_task_count": route_live_joint_task_count,
        "compression_leakage_risk_flag": compression_leakage_risk_flag,
        "conditions": conditions,
    }


def write_report(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# PLANv8 V8-9 CDMI Summary",
        "",
        f"- Comparison conclusion: `{summary['comparison_conclusion']}`",
        f"- Recommended next step: `{summary['recommended_next_step']}`",
        f"- Best confirmed variant: `{summary['best_confirmed_variant_id']}`",
        f"- Source arm: `{summary['source_arm_id']}`",
        f"- Source families: interface=`{summary['source_interface_family']}` bridge=`{summary['source_bridge_family']}` aux=`{summary['source_auxiliary_family']}`",
        f"- Negative transfer rate: `{summary['negative_transfer_rate']:.3f}`",
        f"- Cross-domain leakage rate: `{summary['cross_domain_leakage_rate']:.3f}`",
        f"- Compression leakage risk flag: `{summary['compression_leakage_risk_flag']}`",
        "",
        "## Primary Task Deltas",
        "",
        f"- Joint math vs self: `{summary['joint_delta_vs_self']['gsm8k']:.6f}`",
        f"- Joint trivia vs self: `{summary['joint_delta_vs_self']['triviaqa']:.6f}`",
        f"- Joint math vs floor: `{summary['joint_delta_vs_floor']['gsm8k']:.6f}`",
        f"- Joint trivia vs floor: `{summary['joint_delta_vs_floor']['triviaqa']:.6f}`",
        f"- Math support on trivia penalty: `{summary['cross_support_penalties']['math_support_on_trivia']:.6f}`",
        f"- Trivia support on math penalty: `{summary['cross_support_penalties']['trivia_support_on_math']:.6f}`",
        "",
        "## Domain Shifts",
        "",
        f"- Domain-conditioned gate shift: `{summary['domain_conditioned_gate_shift']:.6f}`",
        f"- Domain-conditioned attention shift: `{summary['domain_conditioned_attention_shift']:.6f}`",
        f"- Domain-conditioned memory-token shift: `{summary['domain_conditioned_memory_token_shift']:.6f}`",
        f"- Route-live joint task count: `{summary['route_live_joint_task_count']}`",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PLANv8 V8-9 CDMI runs.")
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v80_summary", type=Path, required=True)
    parser.add_argument("--v88_summary", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(
        result_root=args.result_root,
        manifest_path=args.manifest,
        v80_summary_path=args.v80_summary,
        v88_summary_path=args.v88_summary,
    )
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary, args.output_report)


if __name__ == "__main__":
    main()
