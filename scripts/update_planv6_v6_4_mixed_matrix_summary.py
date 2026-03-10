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


SUPPORT_MODE_DEFINITIONS: dict[str, dict[str, str]] = {
    "s3_multi_item_cross_attn_raw": {
        "label": "S3 multi_item_cross_attn_raw",
        "support_mode": "multi_item_cross_attn_raw",
    },
    "s5_hybrid_pooled_plus_items": {
        "label": "S5 hybrid_pooled_plus_items",
        "support_mode": "hybrid_pooled_plus_items",
    },
}

STIMULUS_DEFINITIONS: dict[str, dict[str, str]] = {
    "c0_support_only": {
        "label": "C0 support_only",
        "stimulus_mode": "support_only",
        "balance_mode": "off",
    },
    "c2_support_and_context_gated": {
        "label": "C2 support_and_context_gated",
        "stimulus_mode": "support_and_context",
        "balance_mode": "layernorm_learned_scalar",
    },
}

LOSS_DEFINITIONS: dict[str, dict[str, str]] = {
    "l2_contrastive": {
        "label": "L2 contrastive",
        "family": "contrastive",
    },
    "l3_vicreg": {
        "label": "L3 VICReg / VCReg",
        "family": "vicreg",
    },
    "l5_orthogonality_coverage": {
        "label": "L5 orthogonality + coverage",
        "family": "orthogonality_coverage",
    },
}

TASK_ORDER = ("fever", "gsm8k", "narrativeqa")


def _task_bool(task: dict[str, Any], key: str) -> float:
    return float(bool(task.get(key, False)))


def _combo_selection_tuple(combo_summary: dict[str, Any]) -> tuple[float, ...]:
    tasks = combo_summary["tasks"]
    fever = tasks.get("fever", {})
    gsm8k = tasks.get("gsm8k", {})
    narrativeqa = tasks.get("narrativeqa", {})
    nonfever_tasks = [gsm8k, narrativeqa]
    route_count = float(sum(int(bool(task.get("route_live_post_unfreeze", False))) for task in nonfever_tasks))
    stable_count = float(sum(int(bool(task.get("stable_training_v6", False))) for task in nonfever_tasks))
    usefulness_count = float(sum(int(bool(task.get("usefulness_positive_v6", False))) for task in nonfever_tasks))
    writer_task_ratios = [
        float(task.get("writer_task_to_total_grad_ratio_post_unfreeze", 0.0))
        for task in tasks.values()
        if task
    ]
    writer_task_ratio_mean = float(sum(writer_task_ratios) / max(1, len(writer_task_ratios)))
    fever_signal_delta = float(
        fever.get("margin_delta_mean", fever.get("delta_answer_logprob_median", 0.0))
    )
    narrative_score_delta = float(narrativeqa.get("task_score_delta_vs_control", 0.0))
    return (
        _task_bool(fever, "usefulness_positive_v6"),
        _task_bool(fever, "source_not_collapsed"),
        _task_bool(fever, "writer_task_supervision_live"),
        _task_bool(fever, "route_live_post_unfreeze"),
        _task_bool(fever, "stable_training_v6"),
        _task_bool(narrativeqa, "usefulness_positive_v6"),
        _task_bool(narrativeqa, "stable_training_v6"),
        _task_bool(narrativeqa, "route_live_post_unfreeze"),
        _task_bool(gsm8k, "usefulness_positive_v6"),
        _task_bool(gsm8k, "stable_training_v6"),
        _task_bool(gsm8k, "route_live_post_unfreeze"),
        usefulness_count,
        stable_count,
        route_count,
        writer_task_ratio_mean,
        float(fever.get("task_score_delta_vs_control", 0.0)),
        narrative_score_delta,
        fever_signal_delta,
    )


def _combo_summary(
    result_root: Path,
    support_mode_id: str,
    stimulus_id: str,
    loss_id: str,
    *,
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    combo_id = f"{support_mode_id}__{stimulus_id}__{loss_id}"
    tasks: dict[str, Any] = {}
    for task_name in TASK_ORDER:
        control_dir = result_root / task_name / "control"
        combo_dir = result_root / task_name / combo_id
        if not control_dir.exists() or not combo_dir.exists():
            continue
        control_metrics = _load_json(str(control_dir / "metrics.json"))
        combo_metrics = _load_json(str(combo_dir / "metrics.json"))
        combo_train_events = _load_train_events(str(combo_dir / "train_events.json"))
        tasks[task_name] = _task_summary(
            control_metrics=control_metrics,
            writer_metrics=combo_metrics,
            writer_train_events=combo_train_events,
            head_window=head_window,
            post_unfreeze_window=post_unfreeze_window,
            tail_window=tail_window,
        )
    support_definition = SUPPORT_MODE_DEFINITIONS[support_mode_id]
    stimulus_definition = STIMULUS_DEFINITIONS[stimulus_id]
    loss_definition = LOSS_DEFINITIONS[loss_id]
    selection_tuple = _combo_selection_tuple({"tasks": tasks})
    return {
        "combo_id": combo_id,
        "support_mode_id": support_mode_id,
        "support_label": support_definition["label"],
        "support_mode": support_definition["support_mode"],
        "stimulus_id": stimulus_id,
        "stimulus_label": stimulus_definition["label"],
        "stimulus_mode": stimulus_definition["stimulus_mode"],
        "balance_mode": stimulus_definition["balance_mode"],
        "loss_id": loss_id,
        "loss_label": loss_definition["label"],
        "loss_family": loss_definition["family"],
        "tasks": tasks,
        "selection_tuple": list(selection_tuple),
        "any_task_route_live": any(bool(task.get("route_live_post_unfreeze", False)) for task in tasks.values()),
        "any_task_source_not_collapsed": any(
            bool(task.get("source_not_collapsed", False)) for task in tasks.values()
        ),
        "any_task_usefulness_positive": any(
            bool(task.get("usefulness_positive_v6", False)) for task in tasks.values()
        ),
    }


def build_summary(
    *,
    result_root: Path,
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    combo_summaries = [
        _combo_summary(
            result_root,
            support_mode_id,
            stimulus_id,
            loss_id,
            head_window=head_window,
            post_unfreeze_window=post_unfreeze_window,
            tail_window=tail_window,
        )
        for support_mode_id in SUPPORT_MODE_DEFINITIONS
        for stimulus_id in STIMULUS_DEFINITIONS
        for loss_id in LOSS_DEFINITIONS
        if (result_root / "fever" / f"{support_mode_id}__{stimulus_id}__{loss_id}").exists()
    ]
    ranked_combos = sorted(
        combo_summaries,
        key=lambda payload: tuple(payload["selection_tuple"]),
        reverse=True,
    )
    finalists = [payload["combo_id"] for payload in ranked_combos[:3]]

    support_summaries: list[dict[str, Any]] = []
    for support_mode_id, support_definition in SUPPORT_MODE_DEFINITIONS.items():
        support_combos = [payload for payload in combo_summaries if payload["support_mode_id"] == support_mode_id]
        if not support_combos:
            continue
        best_combo = max(support_combos, key=lambda payload: tuple(payload["selection_tuple"]))
        support_summaries.append(
            {
                "support_mode_id": support_mode_id,
                "support_label": support_definition["label"],
                "best_combo_id": best_combo["combo_id"],
                "best_selection_tuple": list(best_combo["selection_tuple"]),
            }
        )

    stimulus_summaries: list[dict[str, Any]] = []
    for stimulus_id, stimulus_definition in STIMULUS_DEFINITIONS.items():
        stimulus_combos = [payload for payload in combo_summaries if payload["stimulus_id"] == stimulus_id]
        if not stimulus_combos:
            continue
        best_combo = max(stimulus_combos, key=lambda payload: tuple(payload["selection_tuple"]))
        stimulus_summaries.append(
            {
                "stimulus_id": stimulus_id,
                "stimulus_label": stimulus_definition["label"],
                "best_combo_id": best_combo["combo_id"],
                "best_selection_tuple": list(best_combo["selection_tuple"]),
            }
        )

    loss_summaries: list[dict[str, Any]] = []
    for loss_id, loss_definition in LOSS_DEFINITIONS.items():
        loss_combos = [payload for payload in combo_summaries if payload["loss_id"] == loss_id]
        if not loss_combos:
            continue
        best_combo = max(loss_combos, key=lambda payload: tuple(payload["selection_tuple"]))
        loss_summaries.append(
            {
                "loss_id": loss_id,
                "loss_label": loss_definition["label"],
                "best_combo_id": best_combo["combo_id"],
                "best_selection_tuple": list(best_combo["selection_tuple"]),
            }
        )

    if len(finalists) >= 3:
        comparison_conclusion = "select_finalists"
        recommended_next_step = "open_v6_5_recipe_stabilization"
    elif ranked_combos:
        comparison_conclusion = "need_more_mixed_matrix_coverage"
        recommended_next_step = "complete_v6_4_matrix"
    else:
        comparison_conclusion = "missing_mixed_matrix_artifacts"
        recommended_next_step = "run_v6_4_mixed_matrix"

    return {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "finalist_configs": finalists,
        "combo_rank_order": [payload["combo_id"] for payload in ranked_combos],
        "support_rank_order": [
            payload["support_mode_id"]
            for payload in sorted(support_summaries, key=lambda p: tuple(p["best_selection_tuple"]), reverse=True)
        ],
        "stimulus_rank_order": [
            payload["stimulus_id"]
            for payload in sorted(stimulus_summaries, key=lambda p: tuple(p["best_selection_tuple"]), reverse=True)
        ],
        "loss_rank_order": [
            payload["loss_id"]
            for payload in sorted(loss_summaries, key=lambda p: tuple(p["best_selection_tuple"]), reverse=True)
        ],
        "combo_summaries": combo_summaries,
        "support_summaries": support_summaries,
        "stimulus_summaries": stimulus_summaries,
        "loss_summaries": loss_summaries,
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv6 V6-4 Mixed Matrix Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        f"- finalist_configs: {', '.join(summary['finalist_configs']) if summary['finalist_configs'] else 'none'}",
        "",
        "## Ranked Combos",
        "",
    ]
    ranked_combos = sorted(
        summary["combo_summaries"],
        key=lambda payload: tuple(payload["selection_tuple"]),
        reverse=True,
    )
    for index, combo_summary in enumerate(ranked_combos, start=1):
        lines.extend(
            [
                f"### {index}. {combo_summary['support_label']} + {combo_summary['stimulus_label']} + {combo_summary['loss_label']}",
                f"- combo_id: {combo_summary['combo_id']}",
                f"- support_mode: {combo_summary['support_mode']}",
                f"- stimulus_mode: {combo_summary['stimulus_mode']}",
                f"- balance_mode: {combo_summary['balance_mode']}",
                f"- selection_tuple: {combo_summary['selection_tuple']}",
            ]
        )
        for task_name in TASK_ORDER:
            task = combo_summary["tasks"].get(task_name)
            if not task:
                continue
            lines.append(
                f"- {task_name}: "
                f"route_live_post_unfreeze={task['route_live_post_unfreeze']}, "
                f"writer_task_supervision_live={task['writer_task_supervision_live']}, "
                f"source_not_collapsed={task['source_not_collapsed']}, "
                f"stable_training_v6={task['stable_training_v6']}, "
                f"usefulness_positive_v6={task['usefulness_positive_v6']}, "
                f"task_score_delta_vs_control={task['task_score_delta_vs_control']:.6f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PLANv6 V6-4 mixed-matrix runs.")
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
