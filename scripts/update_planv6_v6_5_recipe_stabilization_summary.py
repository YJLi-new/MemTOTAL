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


DEFAULT_FINALISTS = [
    "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage",
    "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg",
    "s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive",
]


def _screen_selection_tuple(recipe_summary: dict[str, Any]) -> tuple[float, ...]:
    fever = recipe_summary["task"]
    return (
        float(bool(fever.get("usefulness_positive_v6", False))),
        float(bool(fever.get("stable_training_v6", False))),
        float(bool(fever.get("source_not_collapsed", False))),
        float(bool(fever.get("writer_task_supervision_live", False))),
        float(bool(fever.get("route_live_post_unfreeze", False))),
        float(bool(fever.get("non_regressive_task", False))),
        float(fever.get("task_score_delta_vs_control", 0.0)),
        float(fever.get("writer_task_to_total_grad_ratio_post_unfreeze", 0.0)),
        float(fever.get("margin_delta_mean", fever.get("delta_answer_logprob_median", 0.0))),
    )


def _confirmation_selection_tuple(confirmation_summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(bool(confirmation_summary.get("stable_feverscore_improved_across_all_seeds", False))),
        float(bool(confirmation_summary.get("all_seeds_source_not_collapsed", False))),
        float(bool(confirmation_summary.get("all_seeds_writer_task_supervision_live", False))),
        float(bool(confirmation_summary.get("all_seeds_route_live_post_unfreeze", False))),
        float(bool(confirmation_summary.get("all_seeds_stable_training_v6", False))),
        float(confirmation_summary.get("seed_success_count", 0.0)),
        float(confirmation_summary.get("mean_task_score_delta_vs_control", 0.0)),
        float(confirmation_summary.get("min_task_score_delta_vs_control", 0.0)),
        float(confirmation_summary.get("mean_writer_task_to_total_grad_ratio_post_unfreeze", 0.0)),
        float(confirmation_summary.get("mean_margin_delta", 0.0)),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        finalists = [
            {
                "alias": f"F{index + 1}",
                "combo_id": combo_id,
            }
            for index, combo_id in enumerate(DEFAULT_FINALISTS)
        ]
        return {
            "finalists": finalists,
            "screen_recipes": [],
            "confirmation_recipe_ids": [],
        }
    return json.loads(path.read_text())


def _screen_recipe_summary(
    *,
    result_root: Path,
    recipe: dict[str, Any],
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any] | None:
    control_dir = result_root / "control"
    recipe_dir = result_root / "screen" / str(recipe["recipe_id"])
    if not control_dir.exists() or not recipe_dir.exists():
        return None
    control_metrics = _load_json(str(control_dir / "metrics.json"))
    recipe_metrics = _load_json(str(recipe_dir / "metrics.json"))
    recipe_train_events = _load_train_events(str(recipe_dir / "train_events.json"))
    task_summary = _task_summary(
        control_metrics=control_metrics,
        writer_metrics=recipe_metrics,
        writer_train_events=recipe_train_events,
        head_window=head_window,
        post_unfreeze_window=post_unfreeze_window,
        tail_window=tail_window,
    )
    summary = {
        **recipe,
        "task": task_summary,
    }
    summary["selection_tuple"] = list(_screen_selection_tuple(summary))
    return summary


def _confirmation_recipe_summary(
    *,
    result_root: Path,
    recipe: dict[str, Any],
    seeds: list[int],
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any] | None:
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = result_root / "confirm" / f"seed_{seed}"
        control_dir = seed_dir / "control"
        recipe_dir = seed_dir / str(recipe["recipe_id"])
        if not control_dir.exists() or not recipe_dir.exists():
            continue
        control_metrics = _load_json(str(control_dir / "metrics.json"))
        recipe_metrics = _load_json(str(recipe_dir / "metrics.json"))
        recipe_train_events = _load_train_events(str(recipe_dir / "train_events.json"))
        task_summary = _task_summary(
            control_metrics=control_metrics,
            writer_metrics=recipe_metrics,
            writer_train_events=recipe_train_events,
            head_window=head_window,
            post_unfreeze_window=post_unfreeze_window,
            tail_window=tail_window,
        )
        seed_summaries.append(
            {
                "seed": seed,
                "task": task_summary,
            }
        )
    if not seed_summaries:
        return None
    tasks = [payload["task"] for payload in seed_summaries]
    seed_success_count = int(
        sum(
            int(
                bool(task.get("usefulness_positive_v6", False))
                and bool(task.get("stable_training_v6", False))
                and bool(task.get("source_not_collapsed", False))
                and bool(task.get("writer_task_supervision_live", False))
                and float(task.get("task_score_delta_vs_control", 0.0)) > 0.0
            )
            for task in tasks
        )
    )
    mean_delta = float(
        sum(float(task.get("task_score_delta_vs_control", 0.0)) for task in tasks) / max(1, len(tasks))
    )
    min_delta = float(min(float(task.get("task_score_delta_vs_control", 0.0)) for task in tasks))
    mean_ratio = float(
        sum(float(task.get("writer_task_to_total_grad_ratio_post_unfreeze", 0.0)) for task in tasks)
        / max(1, len(tasks))
    )
    mean_margin_delta = float(
        sum(float(task.get("margin_delta_mean", task.get("delta_answer_logprob_median", 0.0))) for task in tasks)
        / max(1, len(tasks))
    )
    summary = {
        **recipe,
        "seed_summaries": seed_summaries,
        "seed_count": len(seed_summaries),
        "seed_success_count": seed_success_count,
        "mean_task_score_delta_vs_control": mean_delta,
        "min_task_score_delta_vs_control": min_delta,
        "mean_writer_task_to_total_grad_ratio_post_unfreeze": mean_ratio,
        "mean_margin_delta": mean_margin_delta,
        "all_seeds_source_not_collapsed": all(
            bool(task.get("source_not_collapsed", False)) for task in tasks
        ),
        "all_seeds_writer_task_supervision_live": all(
            bool(task.get("writer_task_supervision_live", False)) for task in tasks
        ),
        "all_seeds_route_live_post_unfreeze": all(
            bool(task.get("route_live_post_unfreeze", False)) for task in tasks
        ),
        "all_seeds_stable_training_v6": all(
            bool(task.get("stable_training_v6", False)) for task in tasks
        ),
        "all_seeds_usefulness_positive_v6": all(
            bool(task.get("usefulness_positive_v6", False)) for task in tasks
        ),
        "stable_feverscore_improved_across_all_seeds": bool(
            len(seed_summaries) >= 3
            and all(bool(task.get("usefulness_positive_v6", False)) for task in tasks)
            and all(bool(task.get("stable_training_v6", False)) for task in tasks)
            and all(bool(task.get("source_not_collapsed", False)) for task in tasks)
            and all(bool(task.get("writer_task_supervision_live", False)) for task in tasks)
            and min_delta > 0.0
        ),
    }
    summary["selection_tuple"] = list(_confirmation_selection_tuple(summary))
    return summary


def build_summary(
    *,
    result_root: Path,
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    manifest = _load_manifest(result_root / "screen-manifest.json")
    finalists = manifest.get("finalists", [])
    screen_recipes = manifest.get("screen_recipes", [])

    screen_summaries = [
        payload
        for payload in (
            _screen_recipe_summary(
                result_root=result_root,
                recipe=recipe,
                head_window=head_window,
                post_unfreeze_window=post_unfreeze_window,
                tail_window=tail_window,
            )
            for recipe in screen_recipes
        )
        if payload is not None
    ]
    ranked_screen = sorted(screen_summaries, key=lambda payload: tuple(payload["selection_tuple"]), reverse=True)
    top_two_recipe_ids = [payload["recipe_id"] for payload in ranked_screen[:2]]

    confirmation_recipe_ids = manifest.get("confirmation_recipe_ids") or top_two_recipe_ids
    confirmation_recipe_lookup = {
        recipe["recipe_id"]: recipe
        for recipe in screen_recipes
        if recipe["recipe_id"] in confirmation_recipe_ids
    }
    confirm_seed_values = [int(value) for value in manifest.get("confirmation_seeds", [])]
    confirmation_summaries = [
        payload
        for payload in (
            _confirmation_recipe_summary(
                result_root=result_root,
                recipe=recipe,
                seeds=confirm_seed_values,
                head_window=head_window,
                post_unfreeze_window=post_unfreeze_window,
                tail_window=tail_window,
            )
            for recipe in confirmation_recipe_lookup.values()
        )
        if payload is not None
    ]
    ranked_confirmation = sorted(
        confirmation_summaries,
        key=lambda payload: tuple(payload["selection_tuple"]),
        reverse=True,
    )
    stabilized_recipes = [
        payload["recipe_id"]
        for payload in ranked_confirmation
        if bool(payload.get("stable_feverscore_improved_across_all_seeds", False))
    ]

    if stabilized_recipes:
        comparison_conclusion = "select_stabilized_recipe"
        recommended_next_step = "open_v6_7_reader_reopening"
    elif ranked_confirmation:
        comparison_conclusion = "recipe_not_yet_stable"
        recommended_next_step = "inspect_v6_6_trigger"
    elif ranked_screen:
        comparison_conclusion = "screen_complete_confirmation_pending"
        recommended_next_step = "run_v6_5_confirmation_stage"
    else:
        comparison_conclusion = "missing_v6_5_artifacts"
        recommended_next_step = "run_v6_5_recipe_stabilization"

    return {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "finalists_from_v6_4": finalists,
        "screen_recipe_count": len(screen_summaries),
        "screen_rank_order": [payload["recipe_id"] for payload in ranked_screen],
        "screen_top_two_recipes": top_two_recipe_ids,
        "screen_recipe_summaries": ranked_screen,
        "confirmation_recipe_ids": confirmation_recipe_ids,
        "confirmation_rank_order": [payload["recipe_id"] for payload in ranked_confirmation],
        "confirmation_recipe_summaries": ranked_confirmation,
        "stabilized_recipes": stabilized_recipes,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PLANv6 V6-5 Recipe Stabilization Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        f"- screen_top_two_recipes: {', '.join(summary['screen_top_two_recipes']) or '(none)'}",
        f"- stabilized_recipes: {', '.join(summary['stabilized_recipes']) or '(none)'}",
        "",
        "## Screen Ranking",
        "",
    ]
    for index, recipe in enumerate(summary["screen_recipe_summaries"], start=1):
        task = recipe["task"]
        lines.extend(
            [
                f"### {index}. {recipe['recipe_id']}",
                f"- finalist_combo_id: {recipe['combo_id']}",
                f"- warmup_steps: {recipe['warmup_steps']}",
                f"- clipping_scheme: {recipe['clipping_scheme']}",
                f"- projector_learning_rate: {recipe['projector_learning_rate']}",
                f"- accumulation_steps: {recipe['accumulation_steps']}",
                f"- layer_variant: {recipe['layer_variant']}",
                (
                    f"- fever: route_live_post_unfreeze={task['route_live_post_unfreeze']}, "
                    f"writer_task_supervision_live={task['writer_task_supervision_live']}, "
                    f"source_not_collapsed={task['source_not_collapsed']}, "
                    f"stable_training_v6={task['stable_training_v6']}, "
                    f"usefulness_positive_v6={task['usefulness_positive_v6']}, "
                    f"task_score_delta_vs_control={task['task_score_delta_vs_control']:.6f}"
                ),
                "",
            ]
        )
    lines.extend(["## Confirmation", ""])
    for recipe in summary["confirmation_recipe_summaries"]:
        lines.extend(
            [
                f"### {recipe['recipe_id']}",
                f"- stable_feverscore_improved_across_all_seeds: {recipe['stable_feverscore_improved_across_all_seeds']}",
                f"- seed_success_count: {recipe['seed_success_count']}/{recipe['seed_count']}",
                f"- mean_task_score_delta_vs_control: {recipe['mean_task_score_delta_vs_control']:.6f}",
                f"- min_task_score_delta_vs_control: {recipe['min_task_score_delta_vs_control']:.6f}",
                f"- all_seeds_source_not_collapsed: {recipe['all_seeds_source_not_collapsed']}",
                f"- all_seeds_writer_task_supervision_live: {recipe['all_seeds_writer_task_supervision_live']}",
                f"- all_seeds_route_live_post_unfreeze: {recipe['all_seeds_route_live_post_unfreeze']}",
                f"- all_seeds_stable_training_v6: {recipe['all_seeds_stable_training_v6']}",
                f"- all_seeds_usefulness_positive_v6: {recipe['all_seeds_usefulness_positive_v6']}",
            ]
        )
        for seed_payload in recipe["seed_summaries"]:
            task = seed_payload["task"]
            lines.append(
                f"- seed {seed_payload['seed']}: delta={task['task_score_delta_vs_control']:.6f}, "
                f"route_live={task['route_live_post_unfreeze']}, "
                f"stable={task['stable_training_v6']}, "
                f"useful={task['usefulness_positive_v6']}"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--head-window", type=int, default=50)
    parser.add_argument("--post-unfreeze-window", type=int, default=50)
    parser.add_argument("--tail-window", type=int, default=50)
    args = parser.parse_args()

    summary = build_summary(
        result_root=Path(args.result_root),
        head_window=max(1, int(args.head_window)),
        post_unfreeze_window=max(1, int(args.post_unfreeze_window)),
        tail_window=max(1, int(args.tail_window)),
    )
    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_report(output_report, summary)


if __name__ == "__main__":
    main()
