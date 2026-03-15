#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
BASELINE_ARM_ID = "a0_baseline"
ARM_ORDER = (
    "a1_reconstruction",
    "a2_vicreg",
    "a3_contrastive",
    "a4_reconstruction_vicreg",
    "a5_barlow",
)
ARM_METADATA = {
    "a0_baseline": {
        "family": "A0",
        "label": "l5_baseline",
        "reconstruction": False,
        "vicreg": False,
        "contrastive": False,
        "barlow": False,
    },
    "a1_reconstruction": {
        "family": "A1",
        "label": "l5_plus_reconstruction",
        "reconstruction": True,
        "vicreg": False,
        "contrastive": False,
        "barlow": False,
    },
    "a2_vicreg": {
        "family": "A2",
        "label": "l5_plus_vicreg",
        "reconstruction": False,
        "vicreg": True,
        "contrastive": False,
        "barlow": False,
    },
    "a3_contrastive": {
        "family": "A3",
        "label": "l5_plus_contrastive",
        "reconstruction": False,
        "vicreg": False,
        "contrastive": True,
        "barlow": False,
    },
    "a4_reconstruction_vicreg": {
        "family": "A4",
        "label": "l5_plus_reconstruction_plus_vicreg",
        "reconstruction": True,
        "vicreg": True,
        "contrastive": False,
        "barlow": False,
    },
    "a5_barlow": {
        "family": "A5",
        "label": "l5_plus_barlow",
        "reconstruction": False,
        "vicreg": False,
        "contrastive": False,
        "barlow": True,
    },
}


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

_as_float = _V71._as_float
_load_json = _V71._load_json
_load_train_events = _V71._load_train_events
_bridge_task_summary = _V73._bridge_task_summary


def _load_v74_selection(summary_path: Path) -> dict[str, Any]:
    payload = _load_json(summary_path)
    base_arm_id = str(payload.get("base_for_v7_5_arm_id", "")).strip()
    if not base_arm_id:
        raise ValueError(f"Missing base_for_v7_5_arm_id in {summary_path}.")
    return {
        "base_arm_id": base_arm_id,
        "base_source_phase": str(payload.get("base_for_v7_5_source_phase", "v7_3")).strip() or "v7_3",
        "control_source_arm_id": str(payload.get("control_source_arm_id", base_arm_id)).strip() or base_arm_id,
        "direct_control_arm_id": str(payload.get("direct_control_arm_id", "")).strip(),
        "winning_depth": str(payload.get("winning_depth", "D1")).strip() or "D1",
        "winning_depth_label": str(payload.get("winning_depth_label", "mid4")).strip() or "mid4",
    }


def _load_metrics_tree(result_root: Path) -> dict[str, dict[str, Any]]:
    metrics_tree: dict[str, dict[str, Any]] = {}
    for arm_name in (BASELINE_ARM_ID, *ARM_ORDER):
        arm_root = result_root / arm_name
        if not arm_root.exists():
            continue
        task_payloads: dict[str, Any] = {}
        for task_dir in sorted(path for path in arm_root.iterdir() if path.is_dir()):
            metrics_path = task_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            task_payloads[task_dir.name] = {
                "metrics": _load_json(metrics_path),
                "train_events": _load_train_events(task_dir / "train_events.json"),
            }
        metrics_tree[arm_name] = task_payloads
    return metrics_tree


def _baseline_summary(control_payloads: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    metrics_ref = control_payloads["gsm8k"]["metrics"]
    return {
        "base_arm_id": selection["base_arm_id"],
        "base_source_phase": selection["base_source_phase"],
        "control_source_arm_id": selection["control_source_arm_id"],
        "direct_control_arm_id": selection["direct_control_arm_id"],
        "winning_depth": selection["winning_depth"],
        "winning_depth_label": selection["winning_depth_label"],
        "writer_family": str(metrics_ref.get("pilot_active_writer_family", "")),
        "bridge_family": str(metrics_ref.get("pilot_active_bridge_family", "")),
        "projector_family": str(metrics_ref.get("pilot_active_projector_family", "")),
        "memory_path_variant": str(metrics_ref.get("pilot_memory_path_variant", "")),
        "projector_token_source": str(metrics_ref.get("pilot_projector_token_source", "")),
        "active_depth_layers": metrics_ref.get("pilot_deep_prefix_layers", []),
        "writer_memory_slots": int(round(_as_float(metrics_ref, "writer_memory_slots", 0.0))),
        "runtime_reader_queries": int(round(_as_float(metrics_ref, "pilot_reader_num_queries", 0.0))),
        "runtime_fuser_short_slots": int(round(_as_float(metrics_ref, "pilot_fuser_short_slots", 0.0))),
        "tasks": {
            task_name: {
                "task_score": _as_float(task_payload["metrics"], "best_adapt_task_score"),
                "exact_match": _as_float(task_payload["metrics"], "best_adapt_exact_match"),
                "writer_memory_not_collapsed_strict": bool(
                    task_payload["metrics"].get("writer_memory_not_collapsed_strict", False)
                ),
                "writer_rank_fraction": _as_float(task_payload["metrics"], "writer_rank_fraction"),
                "writer_memory_slot_effective_rank": _as_float(
                    task_payload["metrics"], "writer_memory_slot_effective_rank"
                ),
                "memory_long_common_mode_energy_ratio": _as_float(
                    task_payload["metrics"], "memory_long_common_mode_energy_ratio"
                ),
            }
            for task_name, task_payload in control_payloads.items()
        },
    }


def _strict_metric_delta(control_task: dict[str, Any], branch_task: dict[str, Any]) -> dict[str, Any]:
    rank_fraction_delta = float(branch_task["writer_rank_fraction"] - control_task["writer_rank_fraction"])
    slot_rank_delta = float(
        branch_task["writer_memory_slot_effective_rank"] - control_task["writer_memory_slot_effective_rank"]
    )
    common_mode_gain = float(
        control_task["memory_long_common_mode_energy_ratio"] - branch_task["memory_long_common_mode_energy_ratio"]
    )
    pairwise_cosine_gain = 0.0
    if control_task["slot_pairwise_cosine_present"] and branch_task["slot_pairwise_cosine_present"]:
        pairwise_cosine_gain = float(
            abs(control_task["slot_pairwise_cosine"]) - abs(branch_task["slot_pairwise_cosine"])
        )
    qualifies = bool(
        (
            branch_task["writer_memory_not_collapsed_strict"]
            and not control_task["writer_memory_not_collapsed_strict"]
        )
        or rank_fraction_delta >= 0.02
        or slot_rank_delta >= 1.0
        or common_mode_gain >= 0.01
        or pairwise_cosine_gain >= 0.01
    )
    return {
        "qualifies": qualifies,
        "rank_fraction_delta": rank_fraction_delta,
        "slot_rank_delta": slot_rank_delta,
        "common_mode_gain": common_mode_gain,
        "pairwise_cosine_gain": pairwise_cosine_gain,
    }


def _aggregate_arm_summary(
    *,
    arm_id: str,
    metadata: dict[str, Any],
    control_tasks: dict[str, dict[str, Any]],
    task_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    strict_deltas: dict[str, dict[str, Any]] = {}
    actual_improvement_count = 0
    regression_count = 0
    strict_gain_count = 0
    strict_rank_fraction_delta_sum = 0.0
    strict_slot_rank_delta_sum = 0.0
    common_mode_gain_sum = 0.0
    helpfulness_score = 0.0
    route_live_count = 0
    stable_training_count = 0
    usefulness_positive_count = 0
    task_score_delta_sum = 0.0
    for task_name, task_summary in task_summaries.items():
        task_delta = float(task_summary["task_score_delta_vs_control"])
        task_score_delta_sum += task_delta
        helpfulness_score += float(task_summary["first_answer_token_or_switch_helpfulness"])
        route_live_count += int(bool(task_summary["route_live_post_unfreeze"]))
        stable_training_count += int(bool(task_summary["stable_training_v6"]))
        usefulness_positive_count += int(bool(task_summary["primary_usefulness_positive"]))
        actual_improvement_count += int(task_delta > 1e-12)
        regression_count += int(task_delta < -1e-12)
        strict_delta = _strict_metric_delta(control_tasks[task_name], task_summary)
        strict_deltas[task_name] = strict_delta
        strict_gain_count += int(strict_delta["qualifies"])
        strict_rank_fraction_delta_sum += float(strict_delta["rank_fraction_delta"])
        strict_slot_rank_delta_sum += float(strict_delta["slot_rank_delta"])
        common_mode_gain_sum += float(strict_delta["common_mode_gain"])
        task_summary["strict_metric_delta"] = strict_delta
    non_regressive_all_tasks = bool(regression_count == 0)
    acceptance_qualified = bool(
        non_regressive_all_tasks and (actual_improvement_count > 0 or strict_gain_count > 0)
    )
    return {
        "arm_id": arm_id,
        "aux_family": metadata["family"],
        "variant_label": metadata["label"],
        "includes_reconstruction": bool(metadata["reconstruction"]),
        "includes_vicreg": bool(metadata["vicreg"]),
        "includes_contrastive": bool(metadata["contrastive"]),
        "includes_barlow": bool(metadata["barlow"]),
        "task_score_delta_sum": float(task_score_delta_sum),
        "actual_primary_improvement_task_count": int(actual_improvement_count),
        "regressed_primary_task_count": int(regression_count),
        "non_regressive_all_tasks": non_regressive_all_tasks,
        "strict_writer_metric_gain_task_count": int(strict_gain_count),
        "strict_rank_fraction_delta_sum": float(strict_rank_fraction_delta_sum),
        "strict_slot_rank_delta_sum": float(strict_slot_rank_delta_sum),
        "common_mode_gain_sum": float(common_mode_gain_sum),
        "route_live_task_count": int(route_live_count),
        "stable_training_task_count": int(stable_training_count),
        "usefulness_positive_task_count": int(usefulness_positive_count),
        "answer_switch_helpfulness_score": float(helpfulness_score),
        "acceptance_qualified": acceptance_qualified,
        "ranking_key": [
            float(actual_improvement_count),
            float(task_score_delta_sum),
            float(strict_gain_count),
            float(strict_rank_fraction_delta_sum),
            float(common_mode_gain_sum),
            float(helpfulness_score),
            float(stable_training_count),
            float(route_live_count),
            float(-regression_count),
        ],
        "tasks": task_summaries,
    }


def _rank_arms(arm_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "arm_id": arm_id,
                "aux_family": summary["aux_family"],
                "variant_label": summary["variant_label"],
                "ranking_key": summary["ranking_key"],
            }
            for arm_id, summary in arm_summaries.items()
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )


def build_summary(*, result_root: Path, v74_summary: Path) -> dict[str, Any]:
    selection = _load_v74_selection(v74_summary)
    metrics_tree = _load_metrics_tree(result_root)
    baseline_payloads = metrics_tree.get(BASELINE_ARM_ID, {})
    missing_control = [task for task in PRIMARY_TASKS if task not in baseline_payloads]
    if missing_control:
        raise ValueError(
            f"Missing V7-5 baseline metrics for tasks {missing_control} under {result_root / BASELINE_ARM_ID}."
        )

    baseline_summary = _baseline_summary(baseline_payloads, selection)
    control_tasks: dict[str, dict[str, Any]] = {}
    for task_name in PRIMARY_TASKS:
        control_tasks[task_name] = _bridge_task_summary(
            control_metrics=baseline_payloads[task_name]["metrics"],
            branch_metrics=baseline_payloads[task_name]["metrics"],
            branch_train_events=baseline_payloads[task_name]["train_events"],
            head_window=50,
            post_unfreeze_window=50,
            tail_window=50,
        )

    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        task_summaries: dict[str, dict[str, Any]] = {}
        for task_name in PRIMARY_TASKS:
            branch_payload = metrics_tree.get(arm_id, {}).get(task_name)
            if branch_payload is None:
                raise ValueError(f"Missing metrics for {arm_id}/{task_name} under {result_root}.")
            task_summaries[task_name] = _bridge_task_summary(
                control_metrics=baseline_payloads[task_name]["metrics"],
                branch_metrics=branch_payload["metrics"],
                branch_train_events=branch_payload["train_events"],
                head_window=50,
                post_unfreeze_window=50,
                tail_window=50,
            )
        arm_summaries[arm_id] = _aggregate_arm_summary(
            arm_id=arm_id,
            metadata=ARM_METADATA[arm_id],
            control_tasks=control_tasks,
            task_summaries=task_summaries,
        )

    ranking = _rank_arms(arm_summaries)
    qualified = [arm for arm in ranking if arm_summaries[arm["arm_id"]]["acceptance_qualified"]]
    promoted_arm_id = qualified[0]["arm_id"] if qualified else ranking[0]["arm_id"]
    any_primary_improvement = any(
        summary["actual_primary_improvement_task_count"] > 0 and summary["non_regressive_all_tasks"]
        for summary in arm_summaries.values()
    )
    any_strict_gain = any(
        summary["strict_writer_metric_gain_task_count"] > 0 and summary["non_regressive_all_tasks"]
        for summary in arm_summaries.values()
    )
    reconstruction_branch_qualified = any(
        summary["acceptance_qualified"] and summary["includes_reconstruction"]
        for summary in arm_summaries.values()
    )
    if any_primary_improvement:
        comparison_conclusion = "aux_revisit_finds_primary_gain_open_v7_6"
        recommended_next_step = "open_v7_6_multiseed_confirmation"
    elif any_strict_gain:
        comparison_conclusion = "aux_revisit_improves_strict_writer_metrics_open_v7_6"
        recommended_next_step = "open_v7_6_multiseed_confirmation"
    else:
        comparison_conclusion = "aux_revisit_flat_best_branch_for_decision_point"
        recommended_next_step = "prepare_v7_6_decision_point"

    return {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "base_from_v7_4_arm_id": selection["base_arm_id"],
        "base_from_v7_4_source_phase": selection["base_source_phase"],
        "control_source_arm_id": selection["control_source_arm_id"],
        "direct_control_arm_id": selection["direct_control_arm_id"],
        "winning_depth": selection["winning_depth"],
        "winning_depth_label": selection["winning_depth_label"],
        "baseline_arm_id": BASELINE_ARM_ID,
        "baseline": baseline_summary,
        "arms": arm_summaries,
        "aux_arm_ranking": ranking,
        "base_for_v7_6_arm_id": promoted_arm_id,
        "optional_barlow_supported": True,
        "evidence": {
            "any_aux_actual_primary_score_improvement": any_primary_improvement,
            "any_aux_non_regressive_strict_metric_gain": any_strict_gain,
            "reconstruction_branch_acceptance_qualified": reconstruction_branch_qualified,
        },
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv7 V7-5 Targeted Auxiliary Revisit Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        f"- base_from_v7_4_arm_id: {summary['base_from_v7_4_arm_id']}",
        f"- base_from_v7_4_source_phase: {summary['base_from_v7_4_source_phase']}",
        f"- winning_depth: {summary['winning_depth']}",
        f"- winning_depth_label: {summary['winning_depth_label']}",
        f"- base_for_v7_6_arm_id: {summary['base_for_v7_6_arm_id']}",
        f"- optional_barlow_supported: {summary['optional_barlow_supported']}",
        "",
        "## Evidence",
    ]
    for key, value in summary["evidence"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Baseline",
            f"- writer_family: {summary['baseline']['writer_family']}",
            f"- bridge_family: {summary['baseline']['bridge_family']}",
            f"- projector_family: {summary['baseline']['projector_family']}",
            f"- memory_path_variant: {summary['baseline']['memory_path_variant']}",
            f"- projector_token_source: {summary['baseline']['projector_token_source']}",
            f"- active_depth_layers: {summary['baseline']['active_depth_layers']}",
            f"- writer_memory_slots: {summary['baseline']['writer_memory_slots']}",
        ]
    )
    for task_name in PRIMARY_TASKS:
        task_summary = summary["baseline"]["tasks"][task_name]
        lines.extend(
            [
                f"- baseline.{task_name}.task_score: {task_summary['task_score']:.6f}",
                (
                    f"- baseline.{task_name}.writer_memory_not_collapsed_strict: "
                    f"{task_summary['writer_memory_not_collapsed_strict']}"
                ),
                f"- baseline.{task_name}.writer_rank_fraction: {task_summary['writer_rank_fraction']:.6f}",
                (
                    f"- baseline.{task_name}.memory_long_common_mode_energy_ratio: "
                    f"{task_summary['memory_long_common_mode_energy_ratio']:.6f}"
                ),
            ]
        )
    for arm_id in ARM_ORDER:
        arm_summary = summary["arms"][arm_id]
        lines.extend(
            [
                "",
                f"## {arm_id}",
                f"- aux_family: {arm_summary['aux_family']}",
                f"- variant_label: {arm_summary['variant_label']}",
                f"- includes_reconstruction: {arm_summary['includes_reconstruction']}",
                f"- includes_vicreg: {arm_summary['includes_vicreg']}",
                f"- includes_contrastive: {arm_summary['includes_contrastive']}",
                f"- includes_barlow: {arm_summary['includes_barlow']}",
                f"- acceptance_qualified: {arm_summary['acceptance_qualified']}",
                f"- non_regressive_all_tasks: {arm_summary['non_regressive_all_tasks']}",
                f"- actual_primary_improvement_task_count: {arm_summary['actual_primary_improvement_task_count']}",
                f"- strict_writer_metric_gain_task_count: {arm_summary['strict_writer_metric_gain_task_count']}",
                f"- task_score_delta_sum: {arm_summary['task_score_delta_sum']:.6f}",
                f"- strict_rank_fraction_delta_sum: {arm_summary['strict_rank_fraction_delta_sum']:.6f}",
                f"- common_mode_gain_sum: {arm_summary['common_mode_gain_sum']:.6f}",
                f"- answer_switch_helpfulness_score: {arm_summary['answer_switch_helpfulness_score']:.6f}",
            ]
        )
        for task_name in PRIMARY_TASKS:
            task_summary = arm_summary["tasks"][task_name]
            strict_delta = task_summary["strict_metric_delta"]
            lines.extend(
                [
                    f"- {task_name}.task_score_delta_vs_control: {task_summary['task_score_delta_vs_control']:.6f}",
                    (
                        f"- {task_name}.writer_memory_not_collapsed_strict: "
                        f"{task_summary['writer_memory_not_collapsed_strict']}"
                    ),
                    f"- {task_name}.writer_rank_fraction: {task_summary['writer_rank_fraction']:.6f}",
                    (
                        f"- {task_name}.memory_long_common_mode_energy_ratio: "
                        f"{task_summary['memory_long_common_mode_energy_ratio']:.6f}"
                    ),
                    f"- {task_name}.route_live_post_unfreeze: {task_summary['route_live_post_unfreeze']}",
                    f"- {task_name}.stable_training_v6: {task_summary['stable_training_v6']}",
                    f"- {task_name}.primary_usefulness_positive: {task_summary['primary_usefulness_positive']}",
                    f"- {task_name}.strict_metric_gain: {strict_delta['qualifies']}",
                    f"- {task_name}.strict_rank_fraction_delta: {strict_delta['rank_fraction_delta']:.6f}",
                    f"- {task_name}.common_mode_gain: {strict_delta['common_mode_gain']:.6f}",
                ]
            )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv7 V7-5 targeted auxiliary revisit.")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--v74_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_summary(
        result_root=Path(args.result_root).resolve(),
        v74_summary=Path(args.v74_summary).resolve(),
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.output_report).write_text(_render_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
