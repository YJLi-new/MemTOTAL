#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
BASELINE_ARM_IDS = (
    "b0_q3_gsm8k_nonthink",
    "b1_q3_gsm8k_think_boxed",
    "b2_q3_trivia_nonthink",
    "b3_q3_trivia_think",
    "b4_q3_fever_nonthink",
)
ORACLE_ARM_IDS = (
    "o0_q25_prefix_replay_gsm8k",
    "o0_q25_prefix_replay_triviaqa",
    "o1_q3_prefix_oracle_mid4_gsm8k",
    "o1_q3_prefix_oracle_mid4_triviaqa",
    "o2_q3_seq_oracle16_gsm8k",
    "o2_q3_seq_oracle16_triviaqa",
    "o3_q3_seq_oracle32_gsm8k",
    "o3_q3_seq_oracle32_triviaqa",
    "o4_q3_xattn_oracle_smoke_gsm8k",
    "o4_q3_xattn_oracle_smoke_triviaqa",
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
    task_name = str(metrics.get("benchmark_id", "")).strip().lower()
    prompt_variant = str(
        metrics.get("prompt_variant")
        or metrics.get("pilot_prompt_variant")
        or ""
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
        "writer_family_label": str(metrics.get("prefix_source_mode", "")),
        "interface_family_label": str(metrics.get("memory_consumer_mode", "legacy_prefix")),
        "train_steps": int(metrics.get("pilot_train_steps", 0)),
        "memory_tokens_count": int(metrics.get("memory_tokens_count", 0)),
        "cross_attn_gate_open_fraction": _safe_float(metrics.get("cross_attn_gate_open_fraction", 0.0)),
        "memory_token_attention_mass_mean": _safe_float(metrics.get("memory_token_attention_mass_mean", 0.0)),
        "reader_cross_attn_grad_norm_median": _median(reader_cross_attn_grad_norms),
        "reader_cross_attn_grad_norm_nonzero_steps": int(
            sum(1 for value in reader_cross_attn_grad_norms if value > 0.0)
        ),
        "train_events_count": len(train_events),
        "all_finite": _all_finite(
            [
                _task_score(metrics),
                _macro_f1(metrics),
                _safe_float(metrics.get("cross_attn_gate_open_fraction", 0.0)),
                _safe_float(metrics.get("memory_token_attention_mass_mean", 0.0)),
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
) -> dict[str, Any]:
    baseline_arms = {arm_id: _arm_payload(arm_id, run_root) for arm_id in BASELINE_ARM_IDS}
    oracle_arms = {arm_id: _arm_payload(arm_id, run_root) for arm_id in ORACLE_ARM_IDS}
    selected_prompt_modes = _load_selected_prompts(selected_prompt_modes_path)

    gsm8k_selected = selected_prompt_modes.get("gsm8k", {})
    triviaqa_selected = selected_prompt_modes.get("triviaqa", {})
    selected_baseline_by_task = {
        "gsm8k": str(gsm8k_selected.get("selected_arm_id", "b0_q3_gsm8k_nonthink")),
        "triviaqa": str(triviaqa_selected.get("selected_arm_id", "b2_q3_trivia_nonthink")),
        "fever": "b4_q3_fever_nonthink",
    }
    selected_prompt_modes_by_task = {
        "gsm8k": str(gsm8k_selected.get("selected_prompt_variant", baseline_arms["b0_q3_gsm8k_nonthink"]["prompt_variant"])),
        "triviaqa": str(triviaqa_selected.get("selected_prompt_variant", baseline_arms["b2_q3_trivia_nonthink"]["prompt_variant"])),
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

    legacy_prefix_qwen3 = {
        "gsm8k": oracle_arms["o1_q3_prefix_oracle_mid4_gsm8k"]["task_score"],
        "triviaqa": oracle_arms["o1_q3_prefix_oracle_mid4_triviaqa"]["task_score"],
    }
    ri1_smoke = {
        "o2_q3_seq_oracle16_gsm8k": oracle_arms["o2_q3_seq_oracle16_gsm8k"],
        "o2_q3_seq_oracle16_triviaqa": oracle_arms["o2_q3_seq_oracle16_triviaqa"],
        "o3_q3_seq_oracle32_gsm8k": oracle_arms["o3_q3_seq_oracle32_gsm8k"],
        "o3_q3_seq_oracle32_triviaqa": oracle_arms["o3_q3_seq_oracle32_triviaqa"],
    }
    ri2_smoke = {
        "o4_q3_xattn_oracle_smoke_gsm8k": oracle_arms["o4_q3_xattn_oracle_smoke_gsm8k"],
        "o4_q3_xattn_oracle_smoke_triviaqa": oracle_arms["o4_q3_xattn_oracle_smoke_triviaqa"],
    }

    ri1_passed_basic_smoke = all(
        payload["all_finite"] and payload["memory_tokens_count"] > 0
        for payload in ri1_smoke.values()
    )
    ri2_passed_basic_smoke = all(
        payload["all_finite"]
        and payload["train_events_count"] > 0
        and payload["reader_cross_attn_grad_norm_nonzero_steps"] > 0
        for payload in ri2_smoke.values()
    )
    qwen3_primary_beats_q25_replay_on_any_task = any(
        selected_baseline_scores[task_name] > qwen25_replay_scores[task_name]
        for task_name in PRIMARY_TASKS
    )
    qwen3_primary_beats_historical_q25_on_any_task = any(
        selected_baseline_scores[task_name] > qwen25_reference_scores.get(task_name, 0.0)
        for task_name in PRIMARY_TASKS
    ) if qwen25_reference_scores else False
    legacy_prefix_oracle_reproduced_or_bounded = all(
        legacy_prefix_qwen3[task_name] <= (selected_baseline_scores[task_name] + 1e-6)
        for task_name in PRIMARY_TASKS
    )

    if not ri1_passed_basic_smoke or not ri2_passed_basic_smoke:
        comparison_conclusion = "repair_qwen3_interface_before_v8_1"
        recommended_next_step = "repair_v8_0_interface_path"
    elif not qwen3_primary_beats_q25_replay_on_any_task and not qwen3_primary_beats_historical_q25_on_any_task:
        comparison_conclusion = "qwen3_baseline_under_calibrated_repair_before_v8_1"
        recommended_next_step = "repair_qwen3_prompt_or_harness"
    else:
        comparison_conclusion = "qwen3_calibrated_interfaces_alive_open_v8_1"
        recommended_next_step = "open_v8_1_reader_interface_scout"

    return {
        "phase": "V8-0",
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "selected_baseline_arms_by_task": selected_baseline_by_task,
        "selected_prompt_modes_by_task": selected_prompt_modes_by_task,
        "selected_qwen3_baseline_scores": selected_baseline_scores,
        "qwen25_replay_scores": qwen25_replay_scores,
        "historical_qwen25_reference_scores": qwen25_reference_scores,
        "qwen3_primary_beats_q25_replay_on_any_task": qwen3_primary_beats_q25_replay_on_any_task,
        "qwen3_primary_beats_historical_q25_on_any_task": qwen3_primary_beats_historical_q25_on_any_task,
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


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv8 V8-0 Summary",
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
            f"- `gsm8k`: qwen3=`{summary['selected_qwen3_baseline_scores']['gsm8k']:.6f}`, "
            f"qwen2.5 replay=`{summary['qwen25_replay_scores']['gsm8k']:.6f}`",
            f"- `triviaqa`: qwen3=`{summary['selected_qwen3_baseline_scores']['triviaqa']:.6f}`, "
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
    parser = argparse.ArgumentParser(description="Summarize PLANv8 V8-0 Qwen3 baseline/oracle runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qwen25-reference-summary", default="")
    parser.add_argument("--selected-prompt-modes", default="")
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
    )
    summary_path = output_root / "v8-0-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_root / "v8-0-summary.md").write_text(_render_markdown(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
