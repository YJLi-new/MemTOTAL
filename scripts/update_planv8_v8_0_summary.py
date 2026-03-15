#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")


def _baseline_arm_ids(primary_arm_prefix: str) -> tuple[str, ...]:
    return (
        f"b0_{primary_arm_prefix}_gsm8k_nonthink",
        f"b1_{primary_arm_prefix}_gsm8k_think_boxed",
        f"b2_{primary_arm_prefix}_trivia_nonthink",
        f"b3_{primary_arm_prefix}_trivia_think",
        f"b4_{primary_arm_prefix}_fever_nonthink",
    )


def _oracle_arm_ids(primary_arm_prefix: str) -> tuple[str, ...]:
    return (
        "o0_q25_prefix_replay_gsm8k",
        "o0_q25_prefix_replay_triviaqa",
        f"o1_{primary_arm_prefix}_prefix_oracle_mid4_gsm8k",
        f"o1_{primary_arm_prefix}_prefix_oracle_mid4_triviaqa",
        f"o2_{primary_arm_prefix}_seq_oracle16_gsm8k",
        f"o2_{primary_arm_prefix}_seq_oracle16_triviaqa",
        f"o3_{primary_arm_prefix}_seq_oracle32_gsm8k",
        f"o3_{primary_arm_prefix}_seq_oracle32_triviaqa",
        f"o4_{primary_arm_prefix}_xattn_oracle_smoke_gsm8k",
        f"o4_{primary_arm_prefix}_xattn_oracle_smoke_triviaqa",
    )


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


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


def _arm_payload(arm_id: str, run_root: Path) -> dict[str, Any]:
    arm_dir = run_root / arm_id
    metrics = _load_json(arm_dir / "metrics.json")
    train_events = _load_train_events(arm_dir / "train_events.json")
    prefix_stats = metrics.get("prefix_artifact_stats", {})
    if not isinstance(prefix_stats, dict):
        prefix_stats = {}
    task_name = str(metrics.get("benchmark_id") or metrics.get("task_name") or "").strip().lower()
    if task_name.endswith("_real_smoke"):
        task_name = task_name.removesuffix("_real_smoke")
    prompt_variant = str(
        metrics.get("prompt_variant")
        or metrics.get("pilot_prompt_variant")
        or ""
    )
    interface_family_label = str(
        metrics.get("memory_consumer_mode")
        or metrics.get("pilot_memory_consumer_mode")
        or prefix_stats.get("pilot_memory_consumer_mode")
        or "legacy_prefix"
    )
    writer_family_label = str(
        metrics.get("prefix_source_mode")
        or metrics.get("pilot_prefix_source_mode")
        or prefix_stats.get("pilot_prefix_source_mode")
        or ""
    )
    memory_tokens_count = int(
        metrics.get("memory_tokens_count")
        or prefix_stats.get("memory_tokens_count")
        or 0
    )
    cross_attn_gate_open_fraction = _safe_float(
        metrics.get("cross_attn_gate_open_fraction", prefix_stats.get("cross_attn_gate_open_fraction", 0.0))
    )
    memory_token_attention_mass_mean = _safe_float(
        metrics.get(
            "memory_token_attention_mass_mean",
            prefix_stats.get(
                "memory_token_attention_mass_mean",
                prefix_stats.get("memory_token_attention_top_mass_mean", 0.0),
            ),
        )
    )
    prefix_attention_mass_mean_by_layer = metrics.get("prefix_attention_mass_mean_by_layer", {})
    if not isinstance(prefix_attention_mass_mean_by_layer, dict):
        prefix_attention_mass_mean_by_layer = {}
    prefix_attention_nontrivial_layer_count = int(
        metrics.get(
            "prefix_attention_nontrivial_layer_count",
            sum(
                1
                for value in prefix_attention_mass_mean_by_layer.values()
                if _safe_float(value) > 1e-3
            ),
        )
    )
    reader_cross_attn_grad_norms = [
        _safe_float(event.get("grad_norm_reader_cross_attn", 0.0))
        for event in train_events
    ]
    return {
        "arm_id": arm_id,
        "task_name": task_name,
        "task_score": _task_score(metrics),
        "macro_f1": _macro_f1(metrics),
        "backbone": str(metrics.get("backbone", "")),
        "prompt_variant": prompt_variant,
        "writer_family_label": writer_family_label,
        "interface_family_label": interface_family_label,
        "train_steps": int(metrics.get("pilot_train_steps", 0)),
        "memory_tokens_count": memory_tokens_count,
        "cross_attn_gate_open_fraction": cross_attn_gate_open_fraction,
        "memory_token_attention_mass_mean": memory_token_attention_mass_mean,
        "prefix_attention_nontrivial_layer_count": prefix_attention_nontrivial_layer_count,
        "reader_cross_attn_grad_norm_median": _median(reader_cross_attn_grad_norms),
        "reader_cross_attn_grad_norm_nonzero_steps": int(
            sum(1 for value in reader_cross_attn_grad_norms if value > 0.0)
        ),
        "train_events_count": len(train_events),
        "all_finite": _all_finite(
            [
                _task_score(metrics),
                _macro_f1(metrics),
                cross_attn_gate_open_fraction,
                memory_token_attention_mass_mean,
            ]
        ),
    }


def _load_selected_prompts(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return _load_json(path)


def _load_qwen25_reference(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    baseline_replay = payload.get("baseline_replay", {})
    selected = baseline_replay.get("c_add", {}).get("tasks", {})
    reference_scores: dict[str, float] = {}
    for task_name in PRIMARY_TASKS:
        task_payload = selected.get(task_name, {})
        reference_scores[task_name] = _safe_float(task_payload.get("task_score", 0.0))
    return reference_scores


def build_summary(
    *,
    run_root: Path,
    qwen25_reference_summary: Path | None,
    selected_prompt_modes_path: Path | None,
    primary_backbone_key: str = "qwen3",
    primary_backbone_label: str = "Qwen3-8B",
    primary_arm_prefix: str = "q3",
) -> dict[str, Any]:
    baseline_arm_ids = _baseline_arm_ids(primary_arm_prefix)
    oracle_arm_ids = _oracle_arm_ids(primary_arm_prefix)
    baseline_arms = {arm_id: _arm_payload(arm_id, run_root) for arm_id in baseline_arm_ids}
    oracle_arms = {arm_id: _arm_payload(arm_id, run_root) for arm_id in oracle_arm_ids}
    selected_prompt_modes = _load_selected_prompts(selected_prompt_modes_path)

    gsm8k_default_arm = f"b0_{primary_arm_prefix}_gsm8k_nonthink"
    triviaqa_default_arm = f"b2_{primary_arm_prefix}_trivia_nonthink"
    fever_default_arm = f"b4_{primary_arm_prefix}_fever_nonthink"
    legacy_gsm8k_arm = f"o1_{primary_arm_prefix}_prefix_oracle_mid4_gsm8k"
    legacy_triviaqa_arm = f"o1_{primary_arm_prefix}_prefix_oracle_mid4_triviaqa"
    ri1_gsm8k_arm = f"o2_{primary_arm_prefix}_seq_oracle16_gsm8k"
    ri1_triviaqa_arm = f"o2_{primary_arm_prefix}_seq_oracle16_triviaqa"
    ri1_wide_gsm8k_arm = f"o3_{primary_arm_prefix}_seq_oracle32_gsm8k"
    ri1_wide_triviaqa_arm = f"o3_{primary_arm_prefix}_seq_oracle32_triviaqa"
    ri2_gsm8k_arm = f"o4_{primary_arm_prefix}_xattn_oracle_smoke_gsm8k"
    ri2_triviaqa_arm = f"o4_{primary_arm_prefix}_xattn_oracle_smoke_triviaqa"

    gsm8k_selected = selected_prompt_modes.get("gsm8k", {})
    triviaqa_selected = selected_prompt_modes.get("triviaqa", {})
    selected_baseline_by_task = {
        "gsm8k": str(gsm8k_selected.get("selected_arm_id", gsm8k_default_arm)),
        "triviaqa": str(triviaqa_selected.get("selected_arm_id", triviaqa_default_arm)),
        "fever": fever_default_arm,
    }
    selected_prompt_modes_by_task = {
        "gsm8k": str(
            gsm8k_selected.get(
                "selected_prompt_variant",
                baseline_arms[gsm8k_default_arm]["prompt_variant"],
            )
        ),
        "triviaqa": str(
            triviaqa_selected.get(
                "selected_prompt_variant",
                baseline_arms[triviaqa_default_arm]["prompt_variant"],
            )
        ),
        "fever": "answer_slot_labels",
    }

    selected_baseline_scores = {
        task_name: baseline_arms[selected_arm_id]["task_score"]
        for task_name, selected_arm_id in selected_baseline_by_task.items()
        if task_name in {"gsm8k", "triviaqa", "fever"}
    }

    qwen25_replay_scores = {
        "gsm8k": oracle_arms["o0_q25_prefix_replay_gsm8k"]["task_score"],
        "triviaqa": oracle_arms["o0_q25_prefix_replay_triviaqa"]["task_score"],
    }
    qwen25_reference_scores = _load_qwen25_reference(qwen25_reference_summary)

    legacy_prefix_primary = {
        "gsm8k": oracle_arms[legacy_gsm8k_arm]["task_score"],
        "triviaqa": oracle_arms[legacy_triviaqa_arm]["task_score"],
    }
    ri1_smoke = {
        ri1_gsm8k_arm: oracle_arms[ri1_gsm8k_arm],
        ri1_triviaqa_arm: oracle_arms[ri1_triviaqa_arm],
        ri1_wide_gsm8k_arm: oracle_arms[ri1_wide_gsm8k_arm],
        ri1_wide_triviaqa_arm: oracle_arms[ri1_wide_triviaqa_arm],
    }
    ri2_smoke = {
        ri2_gsm8k_arm: oracle_arms[ri2_gsm8k_arm],
        ri2_triviaqa_arm: oracle_arms[ri2_triviaqa_arm],
    }

    ri1_passed_basic_smoke = all(
        payload["all_finite"]
        and payload["memory_tokens_count"] > 0
        and payload["prefix_attention_nontrivial_layer_count"] > 0
        for payload in ri1_smoke.values()
    )
    ri2_passed_basic_smoke = all(
        payload["all_finite"]
        and payload["train_events_count"] > 0
        and payload["reader_cross_attn_grad_norm_nonzero_steps"] > 0
        for payload in ri2_smoke.values()
    )
    primary_beats_q25_replay_on_any_task = any(
        selected_baseline_scores[task_name] > qwen25_replay_scores[task_name]
        for task_name in PRIMARY_TASKS
    )
    primary_beats_historical_q25_on_any_task = any(
        selected_baseline_scores[task_name] > qwen25_reference_scores.get(task_name, 0.0)
        for task_name in PRIMARY_TASKS
    ) if qwen25_reference_scores else False
    legacy_prefix_oracle_reproduced_or_bounded = all(
        legacy_prefix_primary[task_name] <= (selected_baseline_scores[task_name] + 1e-6)
        for task_name in PRIMARY_TASKS
    )

    if not ri1_passed_basic_smoke or not ri2_passed_basic_smoke:
        comparison_conclusion = f"repair_{primary_backbone_key}_interface_before_v8_1"
        recommended_next_step = "repair_v8_0_interface_path"
    elif not primary_beats_q25_replay_on_any_task and not primary_beats_historical_q25_on_any_task:
        comparison_conclusion = f"{primary_backbone_key}_baseline_under_calibrated_repair_before_v8_1"
        recommended_next_step = f"repair_{primary_backbone_key}_prompt_or_harness"
    else:
        comparison_conclusion = f"{primary_backbone_key}_calibrated_interfaces_alive_open_v8_1"
        recommended_next_step = "open_v8_1_reader_interface_scout"

    summary = {
        "phase": "V8-0",
        "primary_backbone_key": primary_backbone_key,
        "primary_backbone_label": primary_backbone_label,
        "primary_arm_prefix": primary_arm_prefix,
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "selected_baseline_arms_by_task": selected_baseline_by_task,
        "selected_prompt_modes_by_task": selected_prompt_modes_by_task,
        "selected_primary_baseline_scores": selected_baseline_scores,
        "qwen25_replay_scores": qwen25_replay_scores,
        "historical_qwen25_reference_scores": qwen25_reference_scores,
        "primary_beats_q25_replay_on_any_task": primary_beats_q25_replay_on_any_task,
        "primary_beats_historical_q25_on_any_task": primary_beats_historical_q25_on_any_task,
        "legacy_prefix_oracle_reproduced_or_bounded": legacy_prefix_oracle_reproduced_or_bounded,
        "ri1_passed_basic_smoke": ri1_passed_basic_smoke,
        "ri2_passed_basic_smoke": ri2_passed_basic_smoke,
        "gain_before_writer_training": True,
        "gain_depended_on_compression": False,
        "fever_not_overruling_primary_tasks": True,
        "reader_activation_metrics": {
            arm_id: {
                "cross_attn_gate_open_fraction": payload["cross_attn_gate_open_fraction"],
                "memory_token_attention_mass_mean": payload["memory_token_attention_mass_mean"],
                "reader_cross_attn_grad_norm_median": payload["reader_cross_attn_grad_norm_median"],
            }
            for arm_id, payload in ri2_smoke.items()
        },
        "baseline_arms": baseline_arms,
        "oracle_arms": oracle_arms,
    }
    summary[f"selected_{primary_backbone_key}_baseline_scores"] = selected_baseline_scores
    summary[f"{primary_backbone_key}_primary_beats_q25_replay_on_any_task"] = (
        primary_beats_q25_replay_on_any_task
    )
    summary[f"{primary_backbone_key}_primary_beats_historical_q25_on_any_task"] = (
        primary_beats_historical_q25_on_any_task
    )
    return summary


def _render_markdown(summary: dict[str, Any]) -> str:
    primary_label = str(summary.get("primary_backbone_label", "Primary"))
    lines = [
        f"# PLANv8 V8-0 Summary ({primary_label})",
        "",
        f"- `comparison_conclusion = {summary['comparison_conclusion']}`",
        f"- `recommended_next_step = {summary['recommended_next_step']}`",
        f"- `ri1_passed_basic_smoke = {summary['ri1_passed_basic_smoke']}`",
        f"- `ri2_passed_basic_smoke = {summary['ri2_passed_basic_smoke']}`",
        f"- `legacy_prefix_oracle_reproduced_or_bounded = {summary['legacy_prefix_oracle_reproduced_or_bounded']}`",
        "",
        "## Selected Prompt Modes",
        "",
    ]
    for task_name in ("gsm8k", "triviaqa", "fever"):
        lines.append(
            f"- `{task_name}`: `{summary['selected_prompt_modes_by_task'][task_name]}` "
            f"via `{summary['selected_baseline_arms_by_task'][task_name]}`"
        )
    lines.extend(
        [
            "",
            "## Primary Baselines",
            "",
            f"- `gsm8k`: {primary_label}=`{summary['selected_primary_baseline_scores']['gsm8k']:.6f}`, "
            f"qwen2.5 replay=`{summary['qwen25_replay_scores']['gsm8k']:.6f}`",
            f"- `triviaqa`: {primary_label}=`{summary['selected_primary_baseline_scores']['triviaqa']:.6f}`, "
            f"qwen2.5 replay=`{summary['qwen25_replay_scores']['triviaqa']:.6f}`",
            "",
            "## Reader Activation",
            "",
        ]
    )
    for arm_id, payload in summary["reader_activation_metrics"].items():
        lines.append(
            f"- `{arm_id}`: gate=`{payload['cross_attn_gate_open_fraction']:.6f}`, "
            f"attention_mass=`{payload['memory_token_attention_mass_mean']:.6f}`, "
            f"grad_median=`{payload['reader_cross_attn_grad_norm_median']:.6f}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PLANv8 V8-0 baseline/oracle runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qwen25-reference-summary", default="")
    parser.add_argument("--selected-prompt-modes", default="")
    parser.add_argument("--primary-backbone-key", default="qwen3")
    parser.add_argument("--primary-backbone-label", default="Qwen3-8B")
    parser.add_argument("--primary-arm-prefix", default="q3")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        run_root=run_root,
        qwen25_reference_summary=Path(args.qwen25_reference_summary).resolve()
        if str(args.qwen25_reference_summary).strip()
        else None,
        selected_prompt_modes_path=Path(args.selected_prompt_modes).resolve()
        if str(args.selected_prompt_modes).strip()
        else None,
        primary_backbone_key=str(args.primary_backbone_key).strip(),
        primary_backbone_label=str(args.primary_backbone_label).strip(),
        primary_arm_prefix=str(args.primary_arm_prefix).strip(),
    )
    summary_path = output_root / "v8-0-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_root / "v8-0-summary.md").write_text(_render_markdown(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
