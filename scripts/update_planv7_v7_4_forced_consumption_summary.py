#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
ARM_ORDER = ("f1_num_mask", "f2_rx_only", "f3_anneal", "f4_dyn_budget")
ARM_METADATA = {
    "f1_num_mask": {
        "family": "F1",
        "label": "num_mask",
        "tasks": ("gsm8k",),
    },
    "f2_rx_only": {
        "family": "F2",
        "label": "receiver_then_joint",
        "tasks": ("gsm8k", "triviaqa"),
    },
    "f3_anneal": {
        "family": "F3",
        "label": "starvation_anneal",
        "tasks": ("gsm8k",),
    },
    "f4_dyn_budget": {
        "family": "F4",
        "label": "dynamic_budget",
        "tasks": ("gsm8k", "triviaqa"),
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


def _load_v73_selection(summary_path: Path) -> dict[str, Any]:
    payload = _load_json(summary_path)
    bridge_ranking = payload.get("bridge_arm_ranking", [])
    control_source_arm_id = ""
    if isinstance(bridge_ranking, list) and bridge_ranking:
        control_source_arm_id = str(bridge_ranking[0].get("arm_id", "")).strip()
    if not control_source_arm_id:
        control_source_arm_id = str(payload.get("direct_control_arm_id", "")).strip()
    if not control_source_arm_id:
        raise ValueError(f"Unable to resolve V7-4 control source arm from {summary_path}.")
    return {
        "control_source_arm_id": control_source_arm_id,
        "direct_control_arm_id": str(payload.get("direct_control_arm_id", "")).strip(),
        "winning_depth": str(payload.get("winning_depth", "D1")).strip() or "D1",
        "winning_depth_label": str(payload.get("winning_depth_label", "mid4")).strip() or "mid4",
    }


def _load_metrics_tree(result_root: Path) -> dict[str, dict[str, Any]]:
    metrics_tree: dict[str, dict[str, Any]] = {}
    for arm_name in ("control", *ARM_ORDER):
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


def _control_summary(control_payloads: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    metrics_ref = control_payloads["gsm8k"]["metrics"]
    control_source_arm_id = selection["control_source_arm_id"]
    inferred_metadata: dict[str, Any] = {}
    if control_source_arm_id in getattr(_V73, "BRIDGE_ARM_METADATA", {}):
        inferred_metadata = dict(_V73.BRIDGE_ARM_METADATA[control_source_arm_id])
    elif control_source_arm_id in getattr(_V73, "DIRECT_CONTROL_METADATA", {}):
        inferred_metadata = dict(_V73.DIRECT_CONTROL_METADATA[control_source_arm_id])
    return {
        "control_source_arm_id": control_source_arm_id,
        "direct_control_arm_id": selection["direct_control_arm_id"],
        "winning_depth": selection["winning_depth"],
        "winning_depth_label": selection["winning_depth_label"],
        "writer_family": str(
            metrics_ref.get(
                "pilot_active_writer_family",
                inferred_metadata.get("writer_family", ""),
            )
        ),
        "bridge_family": str(
            metrics_ref.get(
                "pilot_active_bridge_family",
                inferred_metadata.get("bridge_family", ""),
            )
        ),
        "projector_family": str(
            metrics_ref.get(
                "pilot_active_projector_family",
                inferred_metadata.get("projector_family", ""),
            )
        ),
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
            }
            for task_name, task_payload in control_payloads.items()
        },
    }


def _aggregate_arm_summary(
    *,
    arm_id: str,
    metadata: dict[str, Any],
    task_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_score_delta_sum = float(
        sum(float(task["task_score_delta_vs_control"]) for task in task_summaries.values())
    )
    actual_improvement_count = int(
        sum(float(task["task_score_delta_vs_control"]) > 1e-12 for task in task_summaries.values())
    )
    primary_success_count = int(sum(bool(task["primary_branch_success"]) for task in task_summaries.values()))
    usefulness_positive_count = int(
        sum(bool(task["primary_usefulness_positive"]) for task in task_summaries.values())
    )
    route_live_count = int(sum(bool(task["route_live_post_unfreeze"]) for task in task_summaries.values()))
    stable_training_count = int(sum(bool(task["stable_training_v6"]) for task in task_summaries.values()))
    helpfulness_score = float(
        sum(float(task["first_answer_token_or_switch_helpfulness"]) for task in task_summaries.values())
    )
    diagnostic_only = bool(
        actual_improvement_count == 0
        and any(
            float(task["delta_answer_logprob_mean"]) > 1e-12
            or float(task["first_answer_token_or_switch_helpfulness"]) > 1e-12
            for task in task_summaries.values()
        )
    )
    gsm8k_delta = float(task_summaries.get("gsm8k", {}).get("task_score_delta_vs_control", 0.0))
    triviaqa_delta = float(task_summaries.get("triviaqa", {}).get("task_score_delta_vs_control", 0.0))
    return {
        "arm_id": arm_id,
        "forced_consumption_family": metadata["family"],
        "variant_label": metadata["label"],
        "covered_tasks": list(metadata["tasks"]),
        "task_score_delta_sum": task_score_delta_sum,
        "gsm8k_task_score_delta_vs_control": gsm8k_delta,
        "triviaqa_task_score_delta_vs_control": triviaqa_delta,
        "actual_primary_improvement_task_count": actual_improvement_count,
        "primary_branch_success_task_count": primary_success_count,
        "usefulness_positive_task_count": usefulness_positive_count,
        "route_live_task_count": route_live_count,
        "stable_training_task_count": stable_training_count,
        "answer_switch_helpfulness_score": helpfulness_score,
        "acceptance_qualified": bool(actual_improvement_count > 0),
        "diagnostic_only": diagnostic_only,
        "ranking_key": [
            task_score_delta_sum,
            float(actual_improvement_count),
            gsm8k_delta,
            triviaqa_delta,
            float(primary_success_count),
            helpfulness_score,
            float(stable_training_count),
            float(route_live_count),
        ],
        "tasks": task_summaries,
    }


def _rank_arms(arm_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "arm_id": arm_id,
                "forced_consumption_family": summary["forced_consumption_family"],
                "variant_label": summary["variant_label"],
                "ranking_key": summary["ranking_key"],
            }
            for arm_id, summary in arm_summaries.items()
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )


def build_summary(*, result_root: Path, v73_summary: Path) -> dict[str, Any]:
    selection = _load_v73_selection(v73_summary)
    metrics_tree = _load_metrics_tree(result_root)
    control_payloads = metrics_tree.get("control", {})
    missing_control = [task for task in PRIMARY_TASKS if task not in control_payloads]
    if missing_control:
        raise ValueError(
            f"Missing V7-4 control metrics for tasks {missing_control} under {result_root / 'control'}."
        )

    control_summary = _control_summary(control_payloads, selection)
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        metadata = ARM_METADATA[arm_id]
        task_summaries: dict[str, dict[str, Any]] = {}
        for task_name in metadata["tasks"]:
            branch_payload = metrics_tree.get(arm_id, {}).get(task_name)
            if branch_payload is None:
                raise ValueError(f"Missing metrics for {arm_id}/{task_name} under {result_root}.")
            task_summaries[task_name] = _bridge_task_summary(
                control_metrics=control_payloads[task_name]["metrics"],
                branch_metrics=branch_payload["metrics"],
                branch_train_events=branch_payload["train_events"],
                head_window=50,
                post_unfreeze_window=50,
                tail_window=50,
            )
        arm_summaries[arm_id] = _aggregate_arm_summary(
            arm_id=arm_id,
            metadata=metadata,
            task_summaries=task_summaries,
        )

    ranking = _rank_arms(arm_summaries)
    qualified = [arm for arm in ranking if arm_summaries[arm["arm_id"]]["acceptance_qualified"]]
    promoted_arm_id = qualified[0]["arm_id"] if qualified else selection["control_source_arm_id"]
    promoted_source_phase = "v7_4" if qualified else "v7_3"
    any_primary_improvement = any(
        summary["acceptance_qualified"] for summary in arm_summaries.values()
    )
    any_diagnostic_only = any(summary["diagnostic_only"] for summary in arm_summaries.values())
    any_route_live = any(
        summary["route_live_task_count"] > 0 for summary in arm_summaries.values()
    )
    any_stable = any(
        summary["stable_training_task_count"] > 0 for summary in arm_summaries.values()
    )
    if any_primary_improvement:
        comparison_conclusion = "forced_consumption_changes_primary_scores_move_to_v7_5"
    elif any_diagnostic_only or any_route_live or any_stable:
        comparison_conclusion = "forced_consumption_diagnostic_only_move_to_v7_5"
    else:
        comparison_conclusion = "forced_consumption_flat_move_to_v7_5"

    return {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": "open_v7_5_targeted_aux_revisit",
        "control_source_arm_id": selection["control_source_arm_id"],
        "direct_control_arm_id": selection["direct_control_arm_id"],
        "winning_depth": selection["winning_depth"],
        "winning_depth_label": selection["winning_depth_label"],
        "base_for_v7_5_arm_id": promoted_arm_id,
        "base_for_v7_5_source_phase": promoted_source_phase,
        "control": control_summary,
        "arms": arm_summaries,
        "forced_consumption_arm_ranking": ranking,
        "evidence": {
            "any_forced_consumption_primary_score_improvement": any_primary_improvement,
            "any_forced_consumption_diagnostic_only": any_diagnostic_only,
            "any_forced_consumption_route_live": any_route_live,
            "any_forced_consumption_stable_training": any_stable,
        },
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PLANv7 V7-4 Forced Consumption Summary",
        "",
        f"- comparison_conclusion: {summary['comparison_conclusion']}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        f"- control_source_arm_id: {summary['control_source_arm_id']}",
        f"- direct_control_arm_id: {summary['direct_control_arm_id']}",
        f"- winning_depth: {summary['winning_depth']}",
        f"- winning_depth_label: {summary['winning_depth_label']}",
        f"- base_for_v7_5_arm_id: {summary['base_for_v7_5_arm_id']}",
        f"- base_for_v7_5_source_phase: {summary['base_for_v7_5_source_phase']}",
        "",
        "## Evidence",
    ]
    for key, value in summary["evidence"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Control",
            f"- writer_family: {summary['control']['writer_family']}",
            f"- bridge_family: {summary['control']['bridge_family']}",
            f"- projector_family: {summary['control']['projector_family']}",
            f"- memory_path_variant: {summary['control']['memory_path_variant']}",
            f"- projector_token_source: {summary['control']['projector_token_source']}",
            f"- active_depth_layers: {summary['control']['active_depth_layers']}",
            f"- writer_memory_slots: {summary['control']['writer_memory_slots']}",
        ]
    )
    for task_name, task_summary in summary["control"]["tasks"].items():
        lines.append(f"- control.{task_name}.task_score: {task_summary['task_score']:.6f}")
    for arm_id in ("f1_num_mask", "f2_rx_only", "f3_anneal", "f4_dyn_budget"):
        arm = summary["arms"][arm_id]
        lines.extend(
            [
                "",
                f"## {arm_id}",
                f"- forced_consumption_family: {arm['forced_consumption_family']}",
                f"- variant_label: {arm['variant_label']}",
                f"- covered_tasks: {arm['covered_tasks']}",
                f"- task_score_delta_sum: {arm['task_score_delta_sum']:.6f}",
                f"- actual_primary_improvement_task_count: {arm['actual_primary_improvement_task_count']}",
                f"- primary_branch_success_task_count: {arm['primary_branch_success_task_count']}",
                f"- usefulness_positive_task_count: {arm['usefulness_positive_task_count']}",
                f"- route_live_task_count: {arm['route_live_task_count']}",
                f"- stable_training_task_count: {arm['stable_training_task_count']}",
                f"- answer_switch_helpfulness_score: {arm['answer_switch_helpfulness_score']:.6f}",
                f"- acceptance_qualified: {arm['acceptance_qualified']}",
                f"- diagnostic_only: {arm['diagnostic_only']}",
            ]
        )
        for task_name, task_summary in arm["tasks"].items():
            lines.extend(
                [
                    f"- {task_name}.task_score_delta_vs_control: {task_summary['task_score_delta_vs_control']:.6f}",
                    f"- {task_name}.primary_branch_success: {task_summary['primary_branch_success']}",
                    f"- {task_name}.primary_usefulness_positive: {task_summary['primary_usefulness_positive']}",
                    f"- {task_name}.delta_answer_logprob_mean: {task_summary['delta_answer_logprob_mean']:.6f}",
                    f"- {task_name}.route_live_post_unfreeze: {task_summary['route_live_post_unfreeze']}",
                    f"- {task_name}.stable_training_v6: {task_summary['stable_training_v6']}",
                    f"- {task_name}.tail_window_source: {task_summary['tail_window_source']}",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--v73_summary", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(
        result_root=args.result_root.resolve(),
        v73_summary=args.v73_summary.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_render_report(summary))


if __name__ == "__main__":
    main()
