#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
ALL_TASKS = ("gsm8k", "triviaqa", "fever")
FROZEN_CONTROL_ARM_ID = "c0_frozen_no_memory"
ADDITIVE_CONTROL_ARM_ID = "c1_additive_continuity"
DIRECT_CONTROL_ARM_ID = "c2_best_direct"


def _load_helper_script(filename: str, module_name: str) -> Any:
    helper_path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper script {helper_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V71 = _load_helper_script("update_planv7_v7_1_width_depth_summary.py", "_planv7_v71_helpers")
_V73 = _load_helper_script("update_planv7_v7_3_bridge_summary.py", "_planv7_v73_helpers")
_V75 = _load_helper_script("update_planv7_v7_5_targeted_aux_summary.py", "_planv7_v75_helpers")

_as_float = _V71._as_float
_load_json = _V71._load_json
_load_train_events = _V71._load_train_events
_bridge_task_summary = _V73._bridge_task_summary
_strict_metric_delta = _V75._strict_metric_delta
_V75_ARM_METADATA = getattr(_V75, "ARM_METADATA", {})


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _load_selection(result_root: Path, v75_summary: Path, v70_summary: Path) -> dict[str, Any]:
    selection_manifest = result_root / "selection-manifest.json"
    selection: dict[str, Any] = {}
    if selection_manifest.exists():
        selection = _load_json(selection_manifest)
    v75 = _load_json(v75_summary)
    v70 = _load_json(v70_summary)
    promoted_arms = list(selection.get("promoted_arms", []))
    if not promoted_arms:
        ranking = v75.get("aux_arm_ranking", [])
        promoted_arms = [str(row.get("arm_id", "")).strip() for row in ranking[:1] if str(row.get("arm_id", "")).strip()]
    seeds = selection.get("seeds", [61109, 61110, 61111])
    variants = selection.get("variants", [])
    if not variants:
        variants = [
            FROZEN_CONTROL_ARM_ID,
            ADDITIVE_CONTROL_ARM_ID,
            *([DIRECT_CONTROL_ARM_ID] if selection.get("winner_uses_bridge", False) else []),
            *[f"p{index + 1}_{arm_id}" for index, arm_id in enumerate(promoted_arms)],
        ]
    return {
        "selection_manifest_path": str(selection_manifest.resolve()) if selection_manifest.exists() else "",
        "seeds": [int(seed) for seed in seeds],
        "promoted_arms": promoted_arms,
        "variants": variants,
        "base_arm_id": str(selection.get("base_arm_id", v75.get("base_from_v7_4_arm_id", ""))).strip(),
        "base_source_phase": str(selection.get("base_source_phase", v75.get("base_from_v7_4_source_phase", "v7_3"))).strip() or "v7_3",
        "control_source_arm_id": str(selection.get("control_source_arm_id", v75.get("control_source_arm_id", ""))).strip(),
        "direct_control_arm_id": str(selection.get("direct_control_arm_id", v75.get("direct_control_arm_id", ""))).strip(),
        "winning_depth": str(selection.get("winning_depth", v75.get("winning_depth", "D1"))).strip() or "D1",
        "winning_depth_label": str(selection.get("winning_depth_label", v75.get("winning_depth_label", "mid4"))).strip() or "mid4",
        "winner_uses_bridge": bool(selection.get("winner_uses_bridge", v75.get("baseline", {}).get("memory_path_variant") == "two_level")),
        "v75_summary": v75,
        "v70_summary": v70,
    }


def _load_metrics_tree(result_root: Path, variants: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    metrics_tree: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in variants:
        variant_root = result_root / variant
        if not variant_root.exists():
            continue
        seed_payloads: dict[str, dict[str, Any]] = {}
        for seed_dir in sorted(path for path in variant_root.iterdir() if path.is_dir()):
            task_payloads: dict[str, Any] = {}
            for task_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
                metrics_path = task_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                task_payloads[task_dir.name] = {
                    "metrics": _load_json(metrics_path),
                    "train_events": _load_train_events(task_dir / "train_events.json"),
                }
            if task_payloads:
                seed_payloads[seed_dir.name] = task_payloads
        if seed_payloads:
            metrics_tree[variant] = seed_payloads
    return metrics_tree


def _control_task_summary(control_payload: dict[str, Any]) -> dict[str, Any]:
    return _bridge_task_summary(
        control_metrics=control_payload["metrics"],
        branch_metrics=control_payload["metrics"],
        branch_train_events=control_payload["train_events"],
        head_window=50,
        post_unfreeze_window=50,
        tail_window=50,
    )


def _branch_seed_task_summary(
    *,
    branch_payload: dict[str, Any],
    frozen_control_payload: dict[str, Any],
    additive_control_payload: dict[str, Any],
    direct_control_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    branch_vs_frozen = _bridge_task_summary(
        control_metrics=frozen_control_payload["metrics"],
        branch_metrics=branch_payload["metrics"],
        branch_train_events=branch_payload["train_events"],
        head_window=50,
        post_unfreeze_window=50,
        tail_window=50,
    )
    additive_control_summary = _control_task_summary(additive_control_payload)
    branch_vs_additive = _bridge_task_summary(
        control_metrics=additive_control_payload["metrics"],
        branch_metrics=branch_payload["metrics"],
        branch_train_events=branch_payload["train_events"],
        head_window=50,
        post_unfreeze_window=50,
        tail_window=50,
    )
    strict_vs_additive = _strict_metric_delta(additive_control_summary, branch_vs_additive)
    branch_vs_direct: dict[str, Any] | None = None
    strict_vs_direct: dict[str, Any] | None = None
    if direct_control_payload is not None:
        direct_control_summary = _control_task_summary(direct_control_payload)
        branch_vs_direct = _bridge_task_summary(
            control_metrics=direct_control_payload["metrics"],
            branch_metrics=branch_payload["metrics"],
            branch_train_events=branch_payload["train_events"],
            head_window=50,
            post_unfreeze_window=50,
            tail_window=50,
        )
        strict_vs_direct = _strict_metric_delta(direct_control_summary, branch_vs_direct)
    return {
        "task_score": float(branch_vs_additive["task_score"]),
        "task_metric_name": str(branch_payload["metrics"].get("task_metric_name", "")),
        "vs_frozen_control": branch_vs_frozen,
        "vs_additive_control": branch_vs_additive,
        "strict_vs_additive_control": strict_vs_additive,
        "vs_direct_control": branch_vs_direct,
        "strict_vs_direct_control": strict_vs_direct,
    }


def _aggregate_control_variant(
    variant_id: str,
    variant_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    for task_name in ALL_TASKS:
        task_rows: list[dict[str, Any]] = []
        for seed_name, seed_payload in sorted(variant_payloads.items()):
            payload = seed_payload.get(task_name)
            if payload is None:
                continue
            summary = _control_task_summary(payload)
            task_rows.append(
                {
                    "seed": seed_name,
                    "task_score": float(summary["task_score"]),
                    "writer_memory_not_collapsed_strict": bool(summary["writer_memory_not_collapsed_strict"]),
                    "writer_rank_fraction": float(summary["writer_rank_fraction"]),
                    "memory_long_common_mode_energy_ratio": float(summary["memory_long_common_mode_energy_ratio"]),
                }
            )
        tasks[task_name] = {
            "seed_rows": task_rows,
            "task_score_mean": _mean([float(row["task_score"]) for row in task_rows]),
            "strict_writer_seed_count": int(sum(bool(row["writer_memory_not_collapsed_strict"]) for row in task_rows)),
        }
    return {
        "arm_id": variant_id,
        "tasks": tasks,
    }


def _aggregate_branch_task(
    *,
    seed_rows: list[dict[str, Any]],
    has_direct_control: bool,
) -> dict[str, Any]:
    additive_deltas = [float(row["vs_additive_control"]["task_score_delta_vs_control"]) for row in seed_rows]
    direct_deltas = [
        float(row["vs_direct_control"]["task_score_delta_vs_control"])
        for row in seed_rows
        if row["vs_direct_control"] is not None
    ]
    route_live_count = int(sum(bool(row["vs_additive_control"]["route_live_post_unfreeze"]) for row in seed_rows))
    stable_count = int(sum(bool(row["vs_additive_control"]["stable_training_v6"]) for row in seed_rows))
    usefulness_count = int(sum(bool(row["vs_additive_control"]["primary_usefulness_positive"]) for row in seed_rows))
    strict_gain_count = int(sum(bool(row["strict_vs_additive_control"]["qualifies"]) for row in seed_rows))
    strict_writer_seed_count = int(
        sum(bool(row["vs_additive_control"]["writer_memory_not_collapsed_strict"]) for row in seed_rows)
    )
    projector_only_warning_count = int(
        sum(bool(row["vs_additive_control"]["projector_manufactured_diversity"]) for row in seed_rows)
    )
    positive_seed_count_vs_additive = int(sum(delta > 1e-12 for delta in additive_deltas))
    non_regressive_seed_count_vs_additive = int(sum(delta >= -1e-12 for delta in additive_deltas))
    positive_seed_count_vs_direct = int(sum(delta > 1e-12 for delta in direct_deltas))
    non_regressive_seed_count_vs_direct = int(sum(delta >= -1e-12 for delta in direct_deltas))
    mean_delta_vs_additive = _mean(additive_deltas)
    mean_delta_vs_direct = _mean(direct_deltas)
    mean_delta_vs_frozen = _mean(
        [float(row["vs_frozen_control"]["task_score_delta_vs_control"]) for row in seed_rows]
    )
    consistent_primary_gain = bool(
        positive_seed_count_vs_additive >= 2
        and non_regressive_seed_count_vs_additive == len(seed_rows)
        and mean_delta_vs_additive > 1e-12
    )
    passes_direct_non_regression = bool(
        not has_direct_control
        or not direct_deltas
        or (
            non_regressive_seed_count_vs_direct >= max(2, len(direct_deltas) - 1)
            and mean_delta_vs_direct >= -1e-12
        )
    )
    return {
        "seed_rows": seed_rows,
        "task_score_mean": _mean([float(row["task_score"]) for row in seed_rows]),
        "mean_delta_vs_frozen_control": mean_delta_vs_frozen,
        "mean_delta_vs_additive_control": mean_delta_vs_additive,
        "mean_delta_vs_direct_control": mean_delta_vs_direct if direct_deltas else None,
        "positive_seed_count_vs_additive_control": positive_seed_count_vs_additive,
        "non_regressive_seed_count_vs_additive_control": non_regressive_seed_count_vs_additive,
        "positive_seed_count_vs_direct_control": positive_seed_count_vs_direct if direct_deltas else None,
        "non_regressive_seed_count_vs_direct_control": non_regressive_seed_count_vs_direct if direct_deltas else None,
        "route_live_seed_count": route_live_count,
        "stable_training_seed_count": stable_count,
        "usefulness_positive_seed_count": usefulness_count,
        "strict_writer_metric_gain_seed_count": strict_gain_count,
        "strict_writer_seed_count": strict_writer_seed_count,
        "projector_only_warning_seed_count": projector_only_warning_count,
        "consistent_primary_gain": consistent_primary_gain,
        "passes_direct_control_non_regression": passes_direct_non_regression,
        "strict_writer_metric_improved": bool(strict_gain_count >= 2),
    }


def _aggregate_branch_variant(
    *,
    variant_id: str,
    promoted_arm_id: str,
    variant_payloads: dict[str, dict[str, Any]],
    frozen_payloads: dict[str, dict[str, Any]],
    additive_payloads: dict[str, dict[str, Any]],
    direct_payloads: dict[str, dict[str, Any]] | None,
    seed_names: list[str],
) -> dict[str, Any]:
    per_task_rows: dict[str, list[dict[str, Any]]] = {task_name: [] for task_name in ALL_TASKS}
    for seed_name in seed_names:
        variant_seed = variant_payloads.get(seed_name, {})
        frozen_seed = frozen_payloads.get(seed_name, {})
        additive_seed = additive_payloads.get(seed_name, {})
        direct_seed = direct_payloads.get(seed_name, {}) if direct_payloads is not None else {}
        for task_name in ALL_TASKS:
            branch_payload = variant_seed.get(task_name)
            frozen_control_payload = frozen_seed.get(task_name)
            additive_control_payload = additive_seed.get(task_name)
            if branch_payload is None or frozen_control_payload is None or additive_control_payload is None:
                raise ValueError(
                    f"Missing PLANv7 V7-6 payload for variant={variant_id} seed={seed_name} task={task_name}."
                )
            direct_control_payload = direct_seed.get(task_name) if direct_payloads is not None else None
            seed_summary = _branch_seed_task_summary(
                branch_payload=branch_payload,
                frozen_control_payload=frozen_control_payload,
                additive_control_payload=additive_control_payload,
                direct_control_payload=direct_control_payload,
            )
            seed_summary["seed"] = seed_name
            per_task_rows[task_name].append(seed_summary)

    task_summaries = {
        task_name: _aggregate_branch_task(
            seed_rows=rows,
            has_direct_control=direct_payloads is not None,
        )
        for task_name, rows in per_task_rows.items()
    }
    primary_success_task_count = int(
        sum(
            bool(task_summaries[task_name]["consistent_primary_gain"])
            and bool(task_summaries[task_name]["passes_direct_control_non_regression"])
            for task_name in PRIMARY_TASKS
        )
    )
    strict_writer_improved_task_count = int(
        sum(bool(task_summaries[task_name]["strict_writer_metric_improved"]) for task_name in PRIMARY_TASKS)
    )
    projector_only_primary_warning_count = int(
        sum(int(task_summaries[task_name]["projector_only_warning_seed_count"] >= 2) for task_name in PRIMARY_TASKS)
    )
    reproduced_partial_success = bool(primary_success_task_count > 0)
    strict_writer_metrics_improved = bool(strict_writer_improved_task_count > 0)
    projector_only_illusion_warning = bool(
        projector_only_primary_warning_count == len(PRIMARY_TASKS) and not strict_writer_metrics_improved
    )
    path_p_eligible = bool(
        reproduced_partial_success
        and strict_writer_metrics_improved
        and not projector_only_illusion_warning
    )
    arm_metadata = _V75_ARM_METADATA.get(promoted_arm_id, {})
    return {
        "variant_id": variant_id,
        "promoted_arm_id": promoted_arm_id,
        "aux_family": str(arm_metadata.get("family", promoted_arm_id)),
        "variant_label": str(arm_metadata.get("label", promoted_arm_id)),
        "includes_reconstruction": bool(arm_metadata.get("reconstruction", False)),
        "includes_vicreg": bool(arm_metadata.get("vicreg", False)),
        "includes_contrastive": bool(arm_metadata.get("contrastive", False)),
        "includes_barlow": bool(arm_metadata.get("barlow", False)),
        "primary_success_task_count": primary_success_task_count,
        "strict_writer_improved_task_count": strict_writer_improved_task_count,
        "projector_only_primary_warning_count": projector_only_primary_warning_count,
        "reproduced_partial_success": reproduced_partial_success,
        "strict_writer_metrics_improved": strict_writer_metrics_improved,
        "projector_only_illusion_warning": projector_only_illusion_warning,
        "path_p_eligible": path_p_eligible,
        "ranking_key": [
            float(path_p_eligible),
            float(primary_success_task_count),
            float(sum(float(task_summaries[task]["mean_delta_vs_additive_control"]) for task in PRIMARY_TASKS)),
            float(strict_writer_improved_task_count),
            float(-projector_only_primary_warning_count),
            float(task_summaries["fever"]["mean_delta_vs_additive_control"]),
        ],
        "tasks": task_summaries,
    }


def build_summary(*, result_root: Path, v75_summary: Path, v70_summary: Path) -> dict[str, Any]:
    selection = _load_selection(result_root, v75_summary, v70_summary)
    metrics_tree = _load_metrics_tree(result_root, selection["variants"])
    seed_names = [f"seed_{int(seed)}" for seed in selection["seeds"]]
    if FROZEN_CONTROL_ARM_ID not in metrics_tree or ADDITIVE_CONTROL_ARM_ID not in metrics_tree:
        raise ValueError(f"Missing required PLANv7 V7-6 controls under {result_root}.")
    direct_payloads = metrics_tree.get(DIRECT_CONTROL_ARM_ID)
    if selection["winner_uses_bridge"] and selection["direct_control_arm_id"] and direct_payloads is None:
        raise ValueError(f"Missing required PLANv7 V7-6 direct control under {result_root / DIRECT_CONTROL_ARM_ID}.")

    control_summaries = {
        FROZEN_CONTROL_ARM_ID: _aggregate_control_variant(FROZEN_CONTROL_ARM_ID, metrics_tree[FROZEN_CONTROL_ARM_ID]),
        ADDITIVE_CONTROL_ARM_ID: _aggregate_control_variant(ADDITIVE_CONTROL_ARM_ID, metrics_tree[ADDITIVE_CONTROL_ARM_ID]),
    }
    if direct_payloads is not None:
        control_summaries[DIRECT_CONTROL_ARM_ID] = _aggregate_control_variant(DIRECT_CONTROL_ARM_ID, direct_payloads)

    branch_summaries: dict[str, dict[str, Any]] = {}
    for index, promoted_arm_id in enumerate(selection["promoted_arms"]):
        variant_id = f"p{index + 1}_{promoted_arm_id}"
        if variant_id not in metrics_tree:
            raise ValueError(f"Missing PLANv7 V7-6 promoted branch metrics under {result_root / variant_id}.")
        branch_summaries[variant_id] = _aggregate_branch_variant(
            variant_id=variant_id,
            promoted_arm_id=promoted_arm_id,
            variant_payloads=metrics_tree[variant_id],
            frozen_payloads=metrics_tree[FROZEN_CONTROL_ARM_ID],
            additive_payloads=metrics_tree[ADDITIVE_CONTROL_ARM_ID],
            direct_payloads=direct_payloads,
            seed_names=seed_names,
        )

    branch_ranking = sorted(
        (
            {
                "variant_id": variant_id,
                "promoted_arm_id": summary["promoted_arm_id"],
                "aux_family": summary["aux_family"],
                "variant_label": summary["variant_label"],
                "ranking_key": summary["ranking_key"],
            }
            for variant_id, summary in branch_summaries.items()
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )
    best_variant_id = branch_ranking[0]["variant_id"]
    best_branch = branch_summaries[best_variant_id]
    oracle_gate_weak = bool(selection["v70_summary"].get("all_oracles_flat_on_primary_tasks", False))
    any_real_primary_gain = any(bool(summary["reproduced_partial_success"]) for summary in branch_summaries.values())
    any_strict_writer_improvement = any(
        bool(summary["strict_writer_metrics_improved"]) for summary in branch_summaries.values()
    )

    if best_branch["path_p_eligible"]:
        comparison_conclusion = "path_p_external_writer_survives_main_thesis"
        recommended_next_step = "stabilize_best_branch_for_paper_facing_runs"
    elif any_strict_writer_improvement:
        comparison_conclusion = "path_q_external_writer_unresolved_not_dead"
        recommended_next_step = "open_stronger_integrated_writer_or_true_highdim_branch"
    elif oracle_gate_weak and not any_real_primary_gain:
        comparison_conclusion = "path_r_architecture_pivot_required"
        recommended_next_step = "prepare_backbone_native_writer_pivot"
    else:
        comparison_conclusion = "path_q_external_writer_unresolved_not_dead"
        recommended_next_step = "open_stronger_integrated_writer_or_true_highdim_branch"

    return {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "selection_manifest_path": selection["selection_manifest_path"],
        "v75_summary_path": str(v75_summary.resolve()),
        "v70_summary_path": str(v70_summary.resolve()),
        "base_arm_id": selection["base_arm_id"],
        "base_source_phase": selection["base_source_phase"],
        "control_source_arm_id": selection["control_source_arm_id"],
        "direct_control_arm_id": selection["direct_control_arm_id"],
        "winner_uses_bridge": selection["winner_uses_bridge"],
        "winning_depth": selection["winning_depth"],
        "winning_depth_label": selection["winning_depth_label"],
        "seeds": selection["seeds"],
        "controls": control_summaries,
        "branches": branch_summaries,
        "branch_ranking": branch_ranking,
        "best_confirmed_variant_id": best_variant_id,
        "best_confirmed_promoted_arm_id": best_branch["promoted_arm_id"],
        "evidence": {
            "oracle_gate_weak": oracle_gate_weak,
            "any_real_primary_gain_across_three_seeds": any_real_primary_gain,
            "any_strict_writer_metric_improvement_across_three_seeds": any_strict_writer_improvement,
            "best_branch_not_projector_only": not best_branch["projector_only_illusion_warning"],
            "optional_paper_facing_comparator_included": False,
        },
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv7 V7-6 Multi-seed Confirmation Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        f"- best_confirmed_variant_id: {summary['best_confirmed_variant_id']}",
        f"- best_confirmed_promoted_arm_id: {summary['best_confirmed_promoted_arm_id']}",
        f"- winner_uses_bridge: {summary['winner_uses_bridge']}",
        f"- winning_depth: {summary['winning_depth']}",
        f"- winning_depth_label: {summary['winning_depth_label']}",
        f"- seeds: {summary['seeds']}",
        "",
        "## Evidence",
    ]
    for key, value in summary["evidence"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Controls",
        ]
    )
    for control_id, control_summary in summary["controls"].items():
        lines.append(f"- {control_id}:")
        for task_name in ALL_TASKS:
            task_summary = control_summary["tasks"][task_name]
            lines.append(f"  {task_name}.task_score_mean = {task_summary['task_score_mean']:.6f}")
            lines.append(f"  {task_name}.strict_writer_seed_count = {task_summary['strict_writer_seed_count']}")
    for variant_id, branch_summary in summary["branches"].items():
        lines.extend(
            [
                "",
                f"## {variant_id}",
                f"- promoted_arm_id: {branch_summary['promoted_arm_id']}",
                f"- aux_family: {branch_summary['aux_family']}",
                f"- variant_label: {branch_summary['variant_label']}",
                f"- reproduced_partial_success: {branch_summary['reproduced_partial_success']}",
                f"- strict_writer_metrics_improved: {branch_summary['strict_writer_metrics_improved']}",
                f"- projector_only_illusion_warning: {branch_summary['projector_only_illusion_warning']}",
                f"- path_p_eligible: {branch_summary['path_p_eligible']}",
            ]
        )
        for task_name in ALL_TASKS:
            task_summary = branch_summary["tasks"][task_name]
            lines.extend(
                [
                    f"- {task_name}.task_score_mean: {task_summary['task_score_mean']:.6f}",
                    f"- {task_name}.mean_delta_vs_additive_control: {task_summary['mean_delta_vs_additive_control']:.6f}",
                    f"- {task_name}.mean_delta_vs_frozen_control: {task_summary['mean_delta_vs_frozen_control']:.6f}",
                    f"- {task_name}.strict_writer_metric_gain_seed_count: {task_summary['strict_writer_metric_gain_seed_count']}",
                    f"- {task_name}.route_live_seed_count: {task_summary['route_live_seed_count']}",
                    f"- {task_name}.stable_training_seed_count: {task_summary['stable_training_seed_count']}",
                    f"- {task_name}.consistent_primary_gain: {task_summary['consistent_primary_gain']}",
                ]
            )
            if task_summary["mean_delta_vs_direct_control"] is not None:
                lines.append(
                    f"- {task_name}.mean_delta_vs_direct_control: {task_summary['mean_delta_vs_direct_control']:.6f}"
                )
                lines.append(
                    f"- {task_name}.passes_direct_control_non_regression: {task_summary['passes_direct_control_non_regression']}"
                )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv7 V7-6 multi-seed confirmation.")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--v75_summary", required=True)
    parser.add_argument("--v70_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_summary(
        result_root=Path(args.result_root).resolve(),
        v75_summary=Path(args.v75_summary).resolve(),
        v70_summary=Path(args.v70_summary).resolve(),
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.output_report).write_text(_render_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
