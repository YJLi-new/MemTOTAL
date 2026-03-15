#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BENCHMARK_ORDER = ("memoryagentbench", "longmemeval", "alfworld")
BASELINE_ORDER = ("b0_short_window", "b1_full_history", "b2_text_summary", "b3_text_rag")
BASELINE_LABELS = {
    "b0_short_window": "B0 short-window baseline",
    "b1_full_history": "B1 full-history baseline",
    "b2_text_summary": "B2 text-summary baseline",
    "b3_text_rag": "B3 text-RAG baseline",
}
BENCHMARK_LABELS = {
    "memoryagentbench": "MemoryAgentBench",
    "longmemeval": "LongMemEval",
    "alfworld": "ALFWorld",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _baseline_summary(run_root: Path, benchmark_name: str, baseline_id: str) -> dict[str, Any]:
    metrics_path = run_root / benchmark_name / baseline_id / "metrics.json"
    if not metrics_path.exists():
        return {
            "baseline_id": baseline_id,
            "label": BASELINE_LABELS[baseline_id],
            "present": False,
        }
    metrics = _load_json(metrics_path)
    return {
        "baseline_id": baseline_id,
        "label": BASELINE_LABELS[baseline_id],
        "present": True,
        "task_score": _safe_float(metrics.get("task_score", metrics.get("success_rate", 0.0))),
        "examples_evaluated": int(metrics.get("examples_evaluated", 0)),
        "metric_name": str(metrics.get("metric_name", metrics.get("baseline_mode", ""))),
        "success_rate": _safe_float(metrics.get("success_rate", 0.0)),
        "mean_steps_executed": _safe_float(metrics.get("mean_steps_executed", 0.0)),
        "mean_invalid_resolution_count": _safe_float(metrics.get("mean_invalid_resolution_count", 0.0)),
        "support_pool_size_mean": _safe_float(metrics.get("support_pool_size_mean", 0.0)),
        "mean_support_retrieval_score": _safe_float(metrics.get("mean_support_retrieval_score", 0.0)),
    }


def build_summary(*, run_root: Path, v90_summary_path: Path) -> dict[str, Any]:
    v90_summary = _load_json(v90_summary_path)
    benchmark_manifests = {
        "memoryagentbench": _load_json(run_root / "materialized-manifests" / "memoryagentbench-pilot.json"),
        "longmemeval": _load_json(run_root / "materialized-manifests" / "longmemeval-pilot.json"),
        "alfworld": _load_json(run_root / "materialized-manifests" / "alfworld-pilot.json"),
    }
    benchmark_summaries: dict[str, Any] = {}
    acceptance_checks: list[bool] = []
    for benchmark_name in BENCHMARK_ORDER:
        baselines = {
            baseline_id: _baseline_summary(run_root, benchmark_name, baseline_id)
            for baseline_id in BASELINE_ORDER
        }
        all_present = all(baselines[baseline_id]["present"] for baseline_id in BASELINE_ORDER)
        benchmark_summaries[benchmark_name] = {
            "benchmark_name": BENCHMARK_LABELS[benchmark_name],
            "manifest": benchmark_manifests[benchmark_name],
            "baselines": baselines,
            "all_required_baselines_present": all_present,
        }
        acceptance_checks.append(all_present)

    benchmark_hardening_complete = all(acceptance_checks)
    recommended_next_step = (
        "open_v9_2_withinsession_sharedkv_scout_c0_c2"
        if benchmark_hardening_complete
        else "hold_v9_1_repair_and_rerun"
    )
    return {
        "phase": "V9-1",
        "primary_backbone_key": "qwen34",
        "primary_backbone_label": "Qwen3-4B",
        "source_v9_0_outcome": str(v90_summary.get("outcome_id", "")),
        "source_v9_0_next_step": str(v90_summary.get("recommended_next_step", "")),
        "mainline_consumer_candidate": "C0_or_C2_only",
        "benchmark_summaries": benchmark_summaries,
        "benchmark_hardening_complete": benchmark_hardening_complete,
        "recommended_next_step": recommended_next_step,
        "acceptance": {
            "memoryagentbench_ready": benchmark_summaries["memoryagentbench"]["all_required_baselines_present"],
            "longmemeval_ready": benchmark_summaries["longmemeval"]["all_required_baselines_present"],
            "alfworld_ready": benchmark_summaries["alfworld"]["all_required_baselines_present"],
        },
    }


def _write_outputs(summary: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# PLANv9 V9-1 Long-Horizon Baseline Summary",
        "",
        f"- `benchmark_hardening_complete = {summary['benchmark_hardening_complete']}`",
        f"- `recommended_next_step = {summary['recommended_next_step']}`",
        f"- `mainline_consumer_candidate = {summary['mainline_consumer_candidate']}`",
        f"- `source_v9_0_outcome = {summary['source_v9_0_outcome']}`",
        "",
        "| Benchmark | Baseline | Present | Task score | Examples | Success rate | Mean steps | Mean support score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark_name in BENCHMARK_ORDER:
        benchmark = summary["benchmark_summaries"][benchmark_name]
        for baseline_id in BASELINE_ORDER:
            baseline = benchmark["baselines"][baseline_id]
            lines.append(
                "| {benchmark} | {baseline_label} | {present} | {task_score:.4f} | {examples} | {success_rate:.4f} | {mean_steps:.2f} | {support_score:.4f} |".format(
                    benchmark=benchmark["benchmark_name"],
                    baseline_label=baseline["label"],
                    present="yes" if baseline["present"] else "no",
                    task_score=float(baseline.get("task_score", 0.0)),
                    examples=int(baseline.get("examples_evaluated", 0)),
                    success_rate=float(baseline.get("success_rate", 0.0)),
                    mean_steps=float(baseline.get("mean_steps_executed", 0.0)),
                    support_score=float(baseline.get("mean_support_retrieval_score", 0.0)),
                )
            )
    output_md.write_text("\n".join(lines) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv9 V9-1 long-horizon baselines.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--v90_summary_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    summary = build_summary(
        run_root=Path(args.run_root),
        v90_summary_path=Path(args.v90_summary_path),
    )
    _write_outputs(summary, Path(args.output_json), Path(args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
