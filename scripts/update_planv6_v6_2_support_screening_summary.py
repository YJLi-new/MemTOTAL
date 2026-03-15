#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from update_writer_deep_prefix_jointpeft_summary import (  # noqa: E402
    _load_json,
    _load_train_events,
    _task_summary,
)


MODE_DEFINITIONS: dict[str, dict[str, str]] = {
    "s0_pooled_block_legacy": {
        "label": "S0 pooled_block_legacy",
        "support_mode": "pooled_block",
        "stimulus_mix": "C1 support_and_context_legacy",
        "balance_mode": "off",
    },
    "s1_pooled_block_gated": {
        "label": "S1 pooled_block_gated",
        "support_mode": "pooled_block",
        "stimulus_mix": "C2 support_and_context_gated",
        "balance_mode": "layernorm_learned_scalar",
    },
    "s2_structured_support_set": {
        "label": "S2 structured_support_set",
        "support_mode": "structured_support_set",
        "stimulus_mix": "C1 support_and_context_legacy",
        "balance_mode": "off",
    },
    "s3_multi_item_cross_attn_raw": {
        "label": "S3 multi_item_cross_attn_raw",
        "support_mode": "multi_item_cross_attn_raw",
        "stimulus_mix": "C0 support_only",
        "balance_mode": "off",
    },
    "s4_multi_item_cross_attn_encoded": {
        "label": "S4 multi_item_cross_attn_encoded",
        "support_mode": "multi_item_cross_attn_encoded",
        "stimulus_mix": "C1 support_and_context_legacy",
        "balance_mode": "off",
    },
    "s5_hybrid_pooled_plus_items": {
        "label": "S5 hybrid_pooled_plus_items",
        "support_mode": "hybrid_pooled_plus_items",
        "stimulus_mix": "C2 support_and_context_gated",
        "balance_mode": "layernorm_learned_scalar",
    },
}
TASK_ORDER = ("fever", "gsm8k", "narrativeqa")


def _mode_selection_tuple(mode_summary: dict[str, Any]) -> tuple[float, ...]:
    tasks = mode_summary["tasks"]
    fever = tasks.get("fever", {})
    diagnostics = [tasks.get(task_name, {}) for task_name in TASK_ORDER if task_name != "fever"]
    fever_usefulness = float(bool(fever.get("usefulness_positive_v6", False)))
    fever_source = float(bool(fever.get("source_not_collapsed", False)))
    fever_task_supervision = float(bool(fever.get("writer_task_supervision_live", False)))
    fever_route = float(bool(fever.get("route_live_post_unfreeze", False)))
    fever_stable = float(bool(fever.get("stable_training_v6", False)))
    any_diag_usefulness = float(any(bool(task.get("usefulness_positive_v6", False)) for task in diagnostics))
    any_diag_source = float(any(bool(task.get("source_not_collapsed", False)) for task in diagnostics))
    any_diag_route = float(any(bool(task.get("route_live_post_unfreeze", False)) for task in diagnostics))
    fever_score_delta = float(fever.get("task_score_delta_vs_control", 0.0))
    fever_signal_delta = float(
        fever.get("margin_delta_mean", fever.get("delta_answer_logprob_median", 0.0))
    )
    diagnostic_pass_count = float(
        sum(
            int(bool(task.get("writer_task_supervision_live", False)))
            + int(bool(task.get("source_not_collapsed", False)))
            + int(bool(task.get("usefulness_positive_v6", False)))
            for task in diagnostics
        )
    )
    return (
        fever_usefulness,
        fever_source,
        fever_task_supervision,
        fever_route,
        fever_stable,
        any_diag_usefulness,
        any_diag_source,
        any_diag_route,
        fever_score_delta,
        fever_signal_delta,
        diagnostic_pass_count,
    )


def _mode_summary(result_root: Path, mode_id: str, *, head_window: int, post_unfreeze_window: int, tail_window: int) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_name in TASK_ORDER:
        control_dir = result_root / task_name / "control"
        writer_dir = result_root / task_name / mode_id
        if not control_dir.exists() or not writer_dir.exists():
            continue
        control_metrics = _load_json(str(control_dir / "metrics.json"))
        writer_metrics = _load_json(str(writer_dir / "metrics.json"))
        writer_train_events = _load_train_events(str(writer_dir / "train_events.json"))
        tasks[task_name] = _task_summary(
            control_metrics=control_metrics,
            writer_metrics=writer_metrics,
            writer_train_events=writer_train_events,
            head_window=head_window,
            post_unfreeze_window=post_unfreeze_window,
            tail_window=tail_window,
        )
    selection_tuple = _mode_selection_tuple({"tasks": tasks})
    mode_definition = MODE_DEFINITIONS[mode_id]
    return {
        "mode_id": mode_id,
        "label": mode_definition["label"],
        "support_mode": mode_definition["support_mode"],
        "stimulus_mix": mode_definition["stimulus_mix"],
        "balance_mode": mode_definition["balance_mode"],
        "tasks": tasks,
        "selection_tuple": list(selection_tuple),
        "fever_primary_task": tasks.get("fever", {}).get("task_name", "fever"),
        "fever_usefulness_positive": bool(tasks.get("fever", {}).get("usefulness_positive_v6", False)),
        "fever_source_not_collapsed": bool(tasks.get("fever", {}).get("source_not_collapsed", False)),
        "any_task_route_live": any(bool(task.get("route_live_post_unfreeze", False)) for task in tasks.values()),
        "any_task_source_not_collapsed": any(bool(task.get("source_not_collapsed", False)) for task in tasks.values()),
        "any_task_usefulness_positive": any(bool(task.get("usefulness_positive_v6", False)) for task in tasks.values()),
    }


def build_summary(
    *,
    result_root: Path,
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    mode_summaries = [
        _mode_summary(
            result_root,
            mode_id,
            head_window=head_window,
            post_unfreeze_window=post_unfreeze_window,
            tail_window=tail_window,
        )
        for mode_id in MODE_DEFINITIONS
        if (result_root / "fever" / mode_id).exists()
    ]
    ranked_modes = sorted(
        mode_summaries,
        key=lambda payload: tuple(payload["selection_tuple"]),
        reverse=True,
    )
    top_two = [payload["mode_id"] for payload in ranked_modes[:2]]
    if len(top_two) >= 2:
        comparison_conclusion = "select_top_two_support_modes"
        recommended_next_step = "open_v6_3_anti_homogenization_screen"
    elif ranked_modes:
        comparison_conclusion = "need_more_support_screening_coverage"
        recommended_next_step = "complete_support_matrix"
    else:
        comparison_conclusion = "missing_support_screening_artifacts"
        recommended_next_step = "run_v6_2_support_screening"
    return {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "top_two_support_modes": top_two,
        "mode_rank_order": [payload["mode_id"] for payload in ranked_modes],
        "mode_summaries": mode_summaries,
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv6 V6-2 Support Screening Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        f"- top_two_support_modes: {', '.join(summary['top_two_support_modes']) if summary['top_two_support_modes'] else 'none'}",
        "",
    ]
    ranked = sorted(
        summary["mode_summaries"],
        key=lambda payload: tuple(payload["selection_tuple"]),
        reverse=True,
    )
    for index, mode_summary in enumerate(ranked, start=1):
        lines.extend(
            [
                f"## {index}. {mode_summary['label']}",
                f"- support_mode: {mode_summary['support_mode']}",
                f"- stimulus_mix: {mode_summary['stimulus_mix']}",
                f"- balance_mode: {mode_summary['balance_mode']}",
                f"- selection_tuple: {mode_summary['selection_tuple']}",
            ]
        )
        fever = mode_summary["tasks"].get("fever")
        if fever:
            lines.extend(
                [
                    "- FEVER:",
                    f"  route_live_post_unfreeze={fever['route_live_post_unfreeze']}, "
                    f"writer_task_supervision_live={fever['writer_task_supervision_live']}, "
                    f"source_not_collapsed={fever['source_not_collapsed']}, "
                    f"stable_training_v6={fever['stable_training_v6']}, "
                    f"usefulness_positive_v6={fever['usefulness_positive_v6']}, "
                    f"task_score_delta_vs_control={fever['task_score_delta_vs_control']:.6f}",
                ]
            )
        for task_name in ("gsm8k", "narrativeqa"):
            task = mode_summary["tasks"].get(task_name)
            if task:
                lines.append(
                    f"- {task_name}: route_live_post_unfreeze={task['route_live_post_unfreeze']}, "
                    f"source_not_collapsed={task['source_not_collapsed']}, "
                    f"usefulness_positive_v6={task['usefulness_positive_v6']}, "
                    f"task_score_delta_vs_control={task['task_score_delta_vs_control']:.6f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PLANv6 V6-2 support screening runs.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--head-window", type=int, default=50)
    parser.add_argument("--post-unfreeze-window", type=int, default=50)
    parser.add_argument("--tail-window", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result_root = Path(args.result_root).resolve()
    summary = build_summary(
        result_root=result_root,
        head_window=int(args.head_window),
        post_unfreeze_window=int(args.post_unfreeze_window),
        tail_window=int(args.tail_window),
    )
    output_json = Path(args.output_json).resolve()
    output_report = Path(args.output_report).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output_report.write_text(_render_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
