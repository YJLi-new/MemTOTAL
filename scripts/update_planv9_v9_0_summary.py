#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVAL_EXAMPLES = 64
STEP = 1.0 / EVAL_EXAMPLES
NEAR_TOLERANCE = 2 * STEP
STRONG_POSITIVE_DELTA = 2 * STEP
COLLAPSE_DELTA = 8 * STEP

ARM_ORDER = (
    "a0_nomemory_control",
    "a1_legacy_prefix_oracle",
    "a2_precache_latent_oracle",
    "a3_sequence_replay_oracle",
)
ARM_LABELS = {
    "a0_nomemory_control": "A0 no-memory control",
    "a1_legacy_prefix_oracle": "A1 legacy-prefix oracle",
    "a2_precache_latent_oracle": "A2 precache-latent oracle",
    "a3_sequence_replay_oracle": "A3 sequence replay oracle",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _task_score(metrics: dict[str, Any]) -> float:
    for key in ("best_adapt_task_score", "task_score", "accuracy", "exact_match"):
        if key in metrics:
            return _safe_float(metrics.get(key))
    return 0.0


def _reference_gsm8k_score(v80_summary_path: Path) -> float:
    payload = _load_json(v80_summary_path)
    score_payload = payload.get("selected_qwen34_baseline_scores") or payload.get("selected_primary_baseline_scores") or {}
    if not isinstance(score_payload, dict):
        return 0.0
    return _safe_float(score_payload.get("gsm8k", 0.0))


def _selected_prompt_variant(selected_prompt_modes_path: Path) -> str:
    payload = _load_json(selected_prompt_modes_path)
    gsm8k = payload.get("gsm8k", {})
    if not isinstance(gsm8k, dict):
        return ""
    return str(gsm8k.get("selected_prompt_variant", ""))


def _arm_payload(arm_id: str, run_root: Path) -> dict[str, Any]:
    arm_dir = run_root / arm_id
    metrics = _load_json(arm_dir / "metrics.json")
    profiling = _load_json(arm_dir / "profiling.json") if (arm_dir / "profiling.json").exists() else {}
    case_rows = _load_jsonl(arm_dir / "task_case_dump.jsonl")
    prefix_stats = metrics.get("prefix_artifact_stats", {})
    if not isinstance(prefix_stats, dict):
        prefix_stats = {}
    predicted_rows = [row for row in case_rows if "predicted_text" in row]
    generation_lengths_words = [
        float(len(str(row.get("predicted_text", "")).split()))
        for row in predicted_rows
    ]
    generation_lengths_chars = [
        float(len(str(row.get("predicted_text", ""))))
        for row in predicted_rows
    ]
    malformed_rate = 0.0
    empty_prediction_rate = 0.0
    if predicted_rows:
        malformed_rate = float(
            sum(
                1
                for row in predicted_rows
                if not str(row.get("normalized_prediction", "")).strip()
            )
            / len(predicted_rows)
        )
        empty_prediction_rate = float(
            sum(1 for row in predicted_rows if not str(row.get("predicted_text", "")).strip())
            / len(predicted_rows)
        )
    attention_mass = _safe_float(
        metrics.get(
            "prefix_attention_mass_mean",
            metrics.get("memory_token_attention_mass_mean", 0.0),
        )
    )
    return {
        "arm_id": arm_id,
        "label": ARM_LABELS[arm_id],
        "task_score": _task_score(metrics),
        "answer_logprob_with_memory": _safe_float(metrics.get("answer_logprob_with_memory", 0.0)),
        "answer_logprob_without_memory": _safe_float(metrics.get("answer_logprob_without_memory", 0.0)),
        "delta_answer_logprob": _safe_float(metrics.get("delta_answer_logprob", 0.0)),
        "generation_length_words_mean": _mean(generation_lengths_words),
        "generation_length_chars_mean": _mean(generation_lengths_chars),
        "malformed_answer_rate": malformed_rate,
        "empty_prediction_rate": empty_prediction_rate,
        "memory_tokens_count": int(
            metrics.get("cache_growth_tokens")
            or prefix_stats.get("memory_tokens_count")
            or 0
        ),
        "memory_tokens_l2": _safe_float(prefix_stats.get("memory_tokens_l2", 0.0)),
        "memory_tokens_slot_norm_mean": _safe_float(prefix_stats.get("memory_tokens_slot_norm_mean", 0.0)),
        "memory_tokens_slot_norm_std": _safe_float(prefix_stats.get("memory_tokens_slot_norm_std", 0.0)),
        "memory_tokens_slot_norm_max": _safe_float(prefix_stats.get("memory_tokens_slot_norm_max", 0.0)),
        "prefix_attention_mass_mean": attention_mass,
        "prefix_attention_nontrivial_layer_count": int(metrics.get("prefix_attention_nontrivial_layer_count", 0)),
        "peak_device_memory_mib": _safe_float(metrics.get("peak_device_memory_mib", 0.0)),
        "wall_time_sec": _safe_float(profiling.get("wall_time_sec", 0.0)),
        "prompt_variant": str(metrics.get("prompt_variant") or metrics.get("pilot_prompt_variant") or ""),
    }


def _delta_examples(delta: float) -> int:
    return int(round(delta / STEP))


def _is_near(lhs: float, rhs: float, *, tolerance: float = NEAR_TOLERANCE) -> bool:
    return abs(float(lhs) - float(rhs)) <= tolerance


def _classify_outcome(a0: float, a1: float, a2: float, a3: float) -> tuple[str, str, str]:
    if a2 >= (a0 + STRONG_POSITIVE_DELTA):
        return (
            "O1",
            "flashmem_precache_beats_nomemory_control",
            "open_v9_1_longhorizon_benchmark_hardening_with_c1_mainline",
        )
    if _is_near(a2, a1) and a2 >= (a0 - NEAR_TOLERANCE) and a1 >= (a0 - NEAR_TOLERANCE):
        return (
            "O0",
            "flashmem_precache_matches_safe_prefix_near_baseline",
            "open_v9_1_longhorizon_benchmark_hardening",
        )
    if _is_near(a2, a3) and a2 <= (a0 - COLLAPSE_DELTA) and a3 <= (a0 - COLLAPSE_DELTA):
        return (
            "O2",
            "flashmem_precache_collapse_matches_sequence_replay",
            "hard_fail_a2_shift_mainline_consumer_to_c0_or_c2",
        )
    return (
        "O3",
        "flashmem_precache_partially_reduces_but_does_not_remove_damage",
        "open_v9_1_longhorizon_benchmark_hardening_with_c1_guardrails",
    )


def build_summary(
    *,
    run_root: Path,
    v80_summary_path: Path,
    selected_prompt_modes_path: Path,
) -> dict[str, Any]:
    arm_summaries = {
        arm_id: _arm_payload(arm_id, run_root)
        for arm_id in ARM_ORDER
    }
    a0_score = arm_summaries["a0_nomemory_control"]["task_score"]
    a1_score = arm_summaries["a1_legacy_prefix_oracle"]["task_score"]
    a2_score = arm_summaries["a2_precache_latent_oracle"]["task_score"]
    a3_score = arm_summaries["a3_sequence_replay_oracle"]["task_score"]
    outcome_id, comparison_conclusion, recommended_next_step = _classify_outcome(
        a0_score,
        a1_score,
        a2_score,
        a3_score,
    )
    consumer_candidate = {
        "O0": "C1_flash_style_soft_append_to_cache",
        "O1": "C1_flash_style_soft_append_to_cache",
        "O2": "C0_or_C2_only",
        "O3": "C1_guarded_with_safety_regularization",
    }[outcome_id]
    v80_reference_score = _reference_gsm8k_score(v80_summary_path)
    selected_prompt_variant = _selected_prompt_variant(selected_prompt_modes_path)
    summary = {
        "phase": "V9-0",
        "primary_backbone_key": "qwen34",
        "primary_backbone_label": "Qwen3-4B",
        "benchmark_id": "gsm8k",
        "eval_examples": EVAL_EXAMPLES,
        "selected_prompt_variant": selected_prompt_variant,
        "v8_0_reference_gsm8k_score": v80_reference_score,
        "arm_order": list(ARM_ORDER),
        "arm_labels": dict(ARM_LABELS),
        "arm_summaries": arm_summaries,
        "a1_vs_a0_delta": float(a1_score - a0_score),
        "a2_vs_a0_delta": float(a2_score - a0_score),
        "a2_vs_a1_delta": float(a2_score - a1_score),
        "a2_vs_a3_delta": float(a2_score - a3_score),
        "a1_vs_a0_examples": _delta_examples(a1_score - a0_score),
        "a2_vs_a0_examples": _delta_examples(a2_score - a0_score),
        "a2_vs_a1_examples": _delta_examples(a2_score - a1_score),
        "a2_vs_a3_examples": _delta_examples(a2_score - a3_score),
        "outcome_id": outcome_id,
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "mainline_consumer_candidate": consumer_candidate,
        "flashmem_precache_safe_enough": bool(outcome_id in {"O0", "O1", "O3"}),
        "hard_fail_a2": bool(outcome_id == "O2"),
    }
    return summary


def _write_outputs(summary: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PLANv9 V9-0 FlashMem Discrimination Summary",
        "",
        f"- `outcome_id = {summary['outcome_id']}`",
        f"- `comparison_conclusion = {summary['comparison_conclusion']}`",
        f"- `recommended_next_step = {summary['recommended_next_step']}`",
        f"- `mainline_consumer_candidate = {summary['mainline_consumer_candidate']}`",
        f"- `selected_prompt_variant = {summary['selected_prompt_variant']}`",
        f"- `v8_0_reference_gsm8k_score = {summary['v8_0_reference_gsm8k_score']:.4f}`",
        "",
        "| Arm | Score | Delta vs A0 | Delta answer logprob | Gen len words | Malformed | Attention mass | Nontrivial layers | Cache growth | Peak VRAM MiB | Wall time s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    a0_score = float(summary["arm_summaries"]["a0_nomemory_control"]["task_score"])
    for arm_id in ARM_ORDER:
        arm = summary["arm_summaries"][arm_id]
        lines.append(
            "| {label} | {score:.4f} | {delta:+.4f} | {logprob:+.4f} | {gen:.2f} | {bad:.4f} | {attn:.4f} | {layers} | {cache} | {vram:.2f} | {wall:.2f} |".format(
                label=arm["label"],
                score=float(arm["task_score"]),
                delta=float(arm["task_score"]) - a0_score,
                logprob=float(arm["delta_answer_logprob"]),
                gen=float(arm["generation_length_words_mean"]),
                bad=float(arm["malformed_answer_rate"]),
                attn=float(arm["prefix_attention_mass_mean"]),
                layers=int(arm["prefix_attention_nontrivial_layer_count"]),
                cache=int(arm["memory_tokens_count"]),
                vram=float(arm["peak_device_memory_mib"]),
                wall=float(arm["wall_time_sec"]),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            f"- `A2 vs A0 = {summary['a2_vs_a0_delta']:+.4f}` ({summary['a2_vs_a0_examples']:+d} / 64 examples)",
            f"- `A2 vs A1 = {summary['a2_vs_a1_delta']:+.4f}` ({summary['a2_vs_a1_examples']:+d} / 64 examples)",
            f"- `A2 vs A3 = {summary['a2_vs_a3_delta']:+.4f}` ({summary['a2_vs_a3_examples']:+d} / 64 examples)",
            "",
            "This phase resolves whether a FlashMem-style cache-prefill route is non-destructive on the qwen34 GSM8K continuity split before opening broader long-horizon engineering.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PLANv9 V9-0 summary.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--v80_summary_path", required=True)
    parser.add_argument("--selected_prompt_modes_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    summary = build_summary(
        run_root=Path(args.run_root),
        v80_summary_path=Path(args.v80_summary_path),
        selected_prompt_modes_path=Path(args.selected_prompt_modes_path),
    )
    _write_outputs(
        summary,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
