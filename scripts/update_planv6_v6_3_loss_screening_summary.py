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
        "stimulus_mix": "C0 support_only",
        "balance_mode": "off",
    },
    "s5_hybrid_pooled_plus_items": {
        "label": "S5 hybrid_pooled_plus_items",
        "support_mode": "hybrid_pooled_plus_items",
        "stimulus_mix": "C2 support_and_context_gated",
        "balance_mode": "layernorm_learned_scalar",
    },
}

LOSS_DEFINITIONS: dict[str, dict[str, str]] = {
    "l0_task_only": {
        "label": "L0 task-only",
        "family": "task_only",
    },
    "l1_legacy": {
        "label": "L1 legacy",
        "family": "legacy",
    },
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


def _combo_selection_tuple(combo_summary: dict[str, Any]) -> tuple[float, ...]:
    tasks = combo_summary["tasks"]
    fever = tasks.get("fever", {})
    diagnostics = [tasks.get(task_name, {}) for task_name in TASK_ORDER if task_name != "fever"]
    fever_signal_delta = float(
        fever.get("margin_delta_mean", fever.get("delta_answer_logprob_median", 0.0))
    )
    writer_task_ratio = max(
        [float(task.get("writer_task_to_total_grad_ratio_post_unfreeze", 0.0)) for task in tasks.values()],
        default=0.0,
    )
    diagnostic_support_count = float(
        sum(int(bool(task.get("source_not_collapsed", False))) for task in diagnostics)
    )
    diagnostic_usefulness_count = float(
        sum(int(bool(task.get("usefulness_positive_v6", False))) for task in diagnostics)
    )
    return (
        float(bool(fever.get("usefulness_positive_v6", False))),
        float(bool(fever.get("source_not_collapsed", False))),
        float(bool(fever.get("writer_task_supervision_live", False))),
        float(bool(fever.get("route_live_post_unfreeze", False))),
        float(bool(fever.get("stable_training_v6", False))),
        float(any(bool(task.get("usefulness_positive_v6", False)) for task in diagnostics)),
        float(any(bool(task.get("source_not_collapsed", False)) for task in diagnostics)),
        float(any(bool(task.get("route_live_post_unfreeze", False)) for task in diagnostics)),
        writer_task_ratio,
        diagnostic_usefulness_count,
        diagnostic_support_count,
        float(fever.get("task_score_delta_vs_control", 0.0)),
        fever_signal_delta,
    )


def _combo_summary(
    result_root: Path,
    support_mode_id: str,
    loss_id: str,
    *,
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    combo_id = f"{support_mode_id}__{loss_id}"
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
    loss_definition = LOSS_DEFINITIONS[loss_id]
    selection_tuple = _combo_selection_tuple({"tasks": tasks})
    return {
        "combo_id": combo_id,
        "support_mode_id": support_mode_id,
        "support_label": support_definition["label"],
        "support_mode": support_definition["support_mode"],
        "stimulus_mix": support_definition["stimulus_mix"],
        "balance_mode": support_definition["balance_mode"],
        "loss_id": loss_id,
        "loss_label": loss_definition["label"],
        "loss_family": loss_definition["family"],
        "tasks": tasks,
        "selection_tuple": list(selection_tuple),
        "any_task_route_live": any(bool(task.get("route_live_post_unfreeze", False)) for task in tasks.values()),
        "any_task_source_not_collapsed": any(bool(task.get("source_not_collapsed", False)) for task in tasks.values()),
        "any_task_usefulness_positive": any(bool(task.get("usefulness_positive_v6", False)) for task in tasks.values()),
        "legacy_misleading_movement": (
            loss_id == "l1_legacy"
            and any(bool(task.get("route_live_post_unfreeze", False)) for task in tasks.values())
            and not any(bool(task.get("usefulness_positive_v6", False)) for task in tasks.values())
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
            loss_id,
            head_window=head_window,
            post_unfreeze_window=post_unfreeze_window,
            tail_window=tail_window,
        )
        for support_mode_id in SUPPORT_MODE_DEFINITIONS
        for loss_id in LOSS_DEFINITIONS
        if (result_root / "fever" / f"{support_mode_id}__{loss_id}").exists()
    ]
    ranked_combos = sorted(
        combo_summaries,
        key=lambda payload: tuple(payload["selection_tuple"]),
        reverse=True,
    )
    loss_summaries: list[dict[str, Any]] = []
    for loss_id, loss_definition in LOSS_DEFINITIONS.items():
        family_combos = [payload for payload in combo_summaries if payload["loss_id"] == loss_id]
        if not family_combos:
            continue
        best_combo = max(family_combos, key=lambda payload: tuple(payload["selection_tuple"]))
        loss_summaries.append(
            {
                "loss_id": loss_id,
                "loss_label": loss_definition["label"],
                "loss_family": loss_definition["family"],
                "best_combo_id": best_combo["combo_id"],
                "best_support_mode_id": best_combo["support_mode_id"],
                "best_selection_tuple": list(best_combo["selection_tuple"]),
                "any_task_route_live": any(payload["any_task_route_live"] for payload in family_combos),
                "any_task_source_not_collapsed": any(
                    payload["any_task_source_not_collapsed"] for payload in family_combos
                ),
                "any_task_usefulness_positive": any(
                    payload["any_task_usefulness_positive"] for payload in family_combos
                ),
                "legacy_misleading_movement": any(
                    payload["legacy_misleading_movement"] for payload in family_combos
                ),
            }
        )
    ranked_loss_summaries = sorted(
        loss_summaries,
        key=lambda payload: tuple(payload["best_selection_tuple"]),
        reverse=True,
    )
    top_families = [payload["loss_id"] for payload in ranked_loss_summaries[:3]]
    if len(ranked_loss_summaries) >= 3:
        comparison_conclusion = "select_top_auxiliary_families"
        recommended_next_step = "open_v6_4_mixed_matrix"
    elif ranked_loss_summaries:
        comparison_conclusion = "need_more_loss_screening_coverage"
        recommended_next_step = "complete_v6_3_matrix"
    else:
        comparison_conclusion = "missing_loss_screening_artifacts"
        recommended_next_step = "run_v6_3_loss_screening"
    return {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "top_auxiliary_families": top_families,
        "combo_rank_order": [payload["combo_id"] for payload in ranked_combos],
        "loss_rank_order": [payload["loss_id"] for payload in ranked_loss_summaries],
        "combo_summaries": combo_summaries,
        "loss_summaries": loss_summaries,
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv6 V6-3 Anti-Homogenization Loss Screening Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        f"- top_auxiliary_families: {', '.join(summary['top_auxiliary_families']) if summary['top_auxiliary_families'] else 'none'}",
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
                f"### {index}. {combo_summary['support_label']} + {combo_summary['loss_label']}",
                f"- combo_id: {combo_summary['combo_id']}",
                f"- stimulus_mix: {combo_summary['stimulus_mix']}",
                f"- balance_mode: {combo_summary['balance_mode']}",
                f"- selection_tuple: {combo_summary['selection_tuple']}",
            ]
        )
        fever = combo_summary["tasks"].get("fever")
        if fever:
            lines.append(
                "- FEVER: "
                f"route_live_post_unfreeze={fever['route_live_post_unfreeze']}, "
                f"writer_task_supervision_live={fever['writer_task_supervision_live']}, "
                f"source_not_collapsed={fever['source_not_collapsed']}, "
                f"stable_training_v6={fever['stable_training_v6']}, "
                f"usefulness_positive_v6={fever['usefulness_positive_v6']}, "
                f"task_score_delta_vs_control={fever['task_score_delta_vs_control']:.6f}, "
                f"writer_task_to_total_grad_ratio_post_unfreeze={fever['writer_task_to_total_grad_ratio_post_unfreeze']:.6f}"
            )
        for task_name in ("gsm8k", "narrativeqa"):
            task = combo_summary["tasks"].get(task_name)
            if task:
                lines.append(
                    f"- {task_name}: route_live_post_unfreeze={task['route_live_post_unfreeze']}, "
                    f"source_not_collapsed={task['source_not_collapsed']}, "
                    f"usefulness_positive_v6={task['usefulness_positive_v6']}, "
                    f"task_score_delta_vs_control={task['task_score_delta_vs_control']:.6f}"
                )
        if combo_summary["legacy_misleading_movement"]:
            lines.append("- warning: legacy_misleading_movement=true")
        lines.append("")
    lines.append("## Ranked Families")
    lines.append("")
    ranked_families = sorted(
        summary["loss_summaries"],
        key=lambda payload: tuple(payload["best_selection_tuple"]),
        reverse=True,
    )
    for index, loss_summary in enumerate(ranked_families, start=1):
        lines.extend(
            [
                f"### {index}. {loss_summary['loss_label']}",
                f"- loss_id: {loss_summary['loss_id']}",
                f"- best_combo_id: {loss_summary['best_combo_id']}",
                f"- best_support_mode_id: {loss_summary['best_support_mode_id']}",
                f"- best_selection_tuple: {loss_summary['best_selection_tuple']}",
                f"- any_task_route_live: {loss_summary['any_task_route_live']}",
                f"- any_task_source_not_collapsed: {loss_summary['any_task_source_not_collapsed']}",
                f"- any_task_usefulness_positive: {loss_summary['any_task_usefulness_positive']}",
                f"- legacy_misleading_movement: {loss_summary['legacy_misleading_movement']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PLANv6 V6-3 loss screening runs.")
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
