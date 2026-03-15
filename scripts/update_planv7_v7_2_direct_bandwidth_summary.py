#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
FEVER_TASK = "fever"
ARM_ORDER = ("d_w1_shared", "d_w2_shared", "d_w2_perlayer")
ARM_METADATA = {
    "d_w1_shared": {
        "writer_family": "W1",
        "depth_family": "D1",
        "projector_family": "P1_shared_rank64",
        "projector_mode": "shared_low_rank",
    },
    "d_w2_shared": {
        "writer_family": "W2",
        "depth_family": "D1",
        "projector_family": "shared_rank64_control",
        "projector_mode": "shared_low_rank",
    },
    "d_w2_perlayer": {
        "writer_family": "W2",
        "depth_family": "D1",
        "projector_family": "P2_per_layer_rank128",
        "projector_mode": "per_layer_low_rank",
    },
}
DEPTH_LABELS = {"d1": "mid4"}


def _load_v71_helpers() -> Any:
    helper_path = Path(__file__).resolve().with_name("update_planv7_v7_1_width_depth_summary.py")
    spec = importlib.util.spec_from_file_location("_planv7_v71_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load PLANv7 V7-1 helpers from {helper_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V71 = _load_v71_helpers()
_as_float = _V71._as_float
_load_json = _V71._load_json
_load_train_events = _V71._load_train_events
_replay_task_summary = _V71._replay_task_summary


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


def _arm_summary(
    *,
    arm_id: str,
    task_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gsm8k = task_summaries["gsm8k"]
    triviaqa = task_summaries["triviaqa"]
    strict_count = int(sum(bool(task["writer_memory_not_collapsed_strict"]) for task in task_summaries.values()))
    primary_success_count = int(sum(bool(task["primary_branch_success"]) for task in task_summaries.values()))
    stability_count = int(sum(bool(task["stable_training_v6"]) for task in task_summaries.values()))
    route_count = int(sum(bool(task["route_live_post_unfreeze"]) for task in task_summaries.values()))
    helpfulness_score = float(
        sum(float(task["first_answer_token_or_switch_helpfulness"]) for task in task_summaries.values())
    )
    strict_rank_fraction_mean = float(
        sum(float(task["writer_rank_fraction"]) for task in task_summaries.values()) / len(task_summaries)
    )
    projector_rank_gain_factor_mean = float(
        sum(float(task["projector_rank_gain_factor"]) for task in task_summaries.values()) / len(task_summaries)
    )
    penalties = {
        "projector_manufactured_diversity": any(
            bool(task["projector_manufactured_diversity"]) for task in task_summaries.values()
        ),
        "both_primary_flat": all(
            abs(float(task["task_score_delta_vs_control"])) <= 1e-12 for task in task_summaries.values()
        ),
        "writer_rank_near_one": any(
            float(task["writer_memory_slot_effective_rank"]) < 1.2 for task in task_summaries.values()
        ),
        "common_mode_near_one": any(
            float(task["memory_long_common_mode_energy_ratio"]) > 0.999 for task in task_summaries.values()
        ),
        "task_aux_conflict": any(
            float(task["writer_task_aux_cosine_post_unfreeze_median"]) < -0.20 for task in task_summaries.values()
        ),
    }
    penalty_count = int(sum(int(value) for value in penalties.values()))
    metadata = ARM_METADATA[arm_id]
    metrics_ref = task_summaries["gsm8k"]
    projector_mode = str(
        metrics_ref.get("pilot_deep_prefix_projector_mode", metadata["projector_mode"])
    )
    return {
        "arm_id": arm_id,
        "writer_family": metadata["writer_family"],
        "depth_family": metadata["depth_family"],
        "depth_label": DEPTH_LABELS[metadata["depth_family"].lower()],
        "projector_family": metadata["projector_family"],
        "projector_mode": projector_mode,
        "bridge_mode": str(metrics_ref.get("pilot_bridge_mode", "writer_direct")),
        "memory_path_variant": str(metrics_ref.get("pilot_memory_path_variant", "single_level")),
        "active_depth_layers": metrics_ref.get("pilot_deep_prefix_layers", []),
        "receiver_lora_layers": metrics_ref.get("pilot_receiver_lora_target_layers", []),
        "projector_rank": int(round(float(metrics_ref.get("pilot_deep_prefix_rank", 0.0)))),
        "writer_memory_slots": int(round(float(metrics_ref.get("writer_memory_slots", 0.0)))),
        "writer_conditioning_layers": int(round(float(metrics_ref.get("pilot_writer_conditioning_layers", 0.0)))),
        "gsm8k_task_score_delta_vs_control": float(gsm8k["task_score_delta_vs_control"]),
        "triviaqa_task_score_delta_vs_control": float(triviaqa["task_score_delta_vs_control"]),
        "primary_task_score_delta_sum": float(
            gsm8k["task_score_delta_vs_control"] + triviaqa["task_score_delta_vs_control"]
        ),
        "strict_writer_memory_task_count": strict_count,
        "strict_rank_fraction_mean": strict_rank_fraction_mean,
        "projector_rank_gain_factor_mean": projector_rank_gain_factor_mean,
        "route_live_task_count": route_count,
        "stable_training_task_count": stability_count,
        "primary_branch_success_task_count": primary_success_count,
        "any_primary_usefulness_positive": any(
            bool(task["primary_usefulness_positive"]) for task in task_summaries.values()
        ),
        "answer_switch_helpfulness_score": helpfulness_score,
        "ranking_penalties": penalties,
        "ranking_penalty_count": penalty_count,
        "ranking_key": [
            float(gsm8k["task_score_delta_vs_control"]),
            float(triviaqa["task_score_delta_vs_control"]),
            float(strict_count),
            strict_rank_fraction_mean,
            helpfulness_score,
            float(stability_count),
            float(-penalty_count),
        ],
        "tasks": task_summaries,
    }


def _rank_arms(arm_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "arm_id": arm_id,
                "writer_family": summary["writer_family"],
                "depth_family": summary["depth_family"],
                "projector_family": summary["projector_family"],
                "projector_mode": summary["projector_mode"],
                "ranking_key": summary["ranking_key"],
            }
            for arm_id, summary in arm_summaries.items()
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )


def _fever_guardrail_summary(
    *,
    metrics_tree: dict[str, dict[str, Any]],
    promoted_arms: list[str],
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    control_payload = metrics_tree.get("control", {}).get(FEVER_TASK)
    if control_payload is None:
        return {
            "complete": False,
            "evaluated_arms": promoted_arms,
            "task": FEVER_TASK,
            "fever_not_used_to_override_primary": True,
            "missing_reason": "fever_control_missing",
            "branches": {},
        }
    branches: dict[str, Any] = {}
    missing_arms: list[str] = []
    for arm_id in promoted_arms:
        branch_payload = metrics_tree.get(arm_id, {}).get(FEVER_TASK)
        if branch_payload is None:
            missing_arms.append(arm_id)
            continue
        branches[arm_id] = _replay_task_summary(
            control_metrics=control_payload["metrics"],
            branch_metrics=branch_payload["metrics"],
            branch_train_events=branch_payload["train_events"],
            head_window=head_window,
            post_unfreeze_window=post_unfreeze_window,
            tail_window=tail_window,
        )
    return {
        "complete": len(missing_arms) == 0,
        "evaluated_arms": promoted_arms,
        "task": FEVER_TASK,
        "fever_not_used_to_override_primary": True,
        "missing_reason": "" if not missing_arms else f"missing_fever_for:{','.join(missing_arms)}",
        "branches": branches,
    }


def _bandwidth_evidence(
    *,
    arm_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    w1 = arm_summaries["d_w1_shared"]
    w2_shared = arm_summaries["d_w2_shared"]
    w2_perlayer = arm_summaries["d_w2_perlayer"]
    best_w2_strict_count = max(
        int(w2_shared["strict_writer_memory_task_count"]),
        int(w2_perlayer["strict_writer_memory_task_count"]),
    )
    best_w2_rank_fraction = max(
        float(w2_shared["strict_rank_fraction_mean"]),
        float(w2_perlayer["strict_rank_fraction_mean"]),
    )
    bandwidth_strict_improvement_observed = bool(
        best_w2_strict_count > int(w1["strict_writer_memory_task_count"])
        or best_w2_rank_fraction > float(w1["strict_rank_fraction_mean"]) + 0.02
    )
    projector_scaling_usefulness_change_observed = bool(
        abs(
            float(w2_perlayer["primary_task_score_delta_sum"])
            - float(w2_shared["primary_task_score_delta_sum"])
        )
        > 1e-12
        or abs(
            float(w2_perlayer["answer_switch_helpfulness_score"])
            - float(w2_shared["answer_switch_helpfulness_score"])
        )
        > 1e-6
        or bool(w2_perlayer["any_primary_usefulness_positive"])
        != bool(w2_shared["any_primary_usefulness_positive"])
    )
    return {
        "bandwidth_strict_improvement_observed": bandwidth_strict_improvement_observed,
        "projector_scaling_usefulness_change_observed": projector_scaling_usefulness_change_observed,
        "w1_shared_strict_writer_memory_task_count": int(w1["strict_writer_memory_task_count"]),
        "w2_shared_strict_writer_memory_task_count": int(w2_shared["strict_writer_memory_task_count"]),
        "w2_perlayer_strict_writer_memory_task_count": int(w2_perlayer["strict_writer_memory_task_count"]),
        "w1_shared_strict_rank_fraction_mean": float(w1["strict_rank_fraction_mean"]),
        "w2_shared_strict_rank_fraction_mean": float(w2_shared["strict_rank_fraction_mean"]),
        "w2_perlayer_strict_rank_fraction_mean": float(w2_perlayer["strict_rank_fraction_mean"]),
        "w2_shared_primary_task_score_delta_sum": float(w2_shared["primary_task_score_delta_sum"]),
        "w2_perlayer_primary_task_score_delta_sum": float(w2_perlayer["primary_task_score_delta_sum"]),
    }


def _select_conclusion(
    *,
    arm_summaries: dict[str, dict[str, Any]],
    primary_arm_ranking: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> tuple[str, str]:
    top_arm_id = primary_arm_ranking[0]["arm_id"]
    w2_shared = arm_summaries["d_w2_shared"]
    w2_perlayer = arm_summaries["d_w2_perlayer"]
    perlayer_primary_gain = any(
        float(w2_perlayer["tasks"][task_name]["task_score_delta_vs_control"]) > 0.0
        for task_name in PRIMARY_TASKS
    )
    all_w2_primary_flat = all(
        abs(float(summary["tasks"][task_name]["task_score_delta_vs_control"])) <= 1e-12
        for summary in (w2_shared, w2_perlayer)
        for task_name in PRIMARY_TASKS
    )
    direct_path_noisy = bool(
        int(w2_perlayer["stable_training_task_count"]) == 0
        or int(w2_perlayer["route_live_task_count"]) == 0
        or int(w2_perlayer["ranking_penalty_count"]) >= 3
    )
    if (
        top_arm_id == "d_w2_perlayer"
        and int(w2_perlayer["strict_writer_memory_task_count"]) > 0
        and perlayer_primary_gain
    ):
        return (
            "promote_w2_perlayer_direct_and_bridge_control",
            "open_v7_3_bridge_phase_with_w2_perlayer_direct_control",
        )
    if bool(evidence["bandwidth_strict_improvement_observed"]) and all_w2_primary_flat:
        return (
            "w2_metrics_up_tasks_flat_move_to_v7_3_v7_4",
            "open_v7_3_bridge_first_wide_writer_and_v7_4",
        )
    if direct_path_noisy:
        return (
            "direct_32_slot_noisy_move_to_v7_3_bridge_first",
            "open_v7_3_bridge_first_wide_writer",
        )
    return (
        "direct_bandwidth_inconclusive_keep_bridge_ready",
        "open_v7_3_bridge_first_wide_writer_keep_direct_control_ready",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv7 V7-2 direct-bandwidth ladder.")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--head_window", type=int, default=50)
    parser.add_argument("--post_unfreeze_window", type=int, default=50)
    parser.add_argument("--tail_window", type=int, default=50)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result_root = Path(args.result_root).resolve()
    metrics_tree = _load_metrics_tree(result_root)
    control_metrics = metrics_tree.get("control", {})
    missing_primary = [
        task_name for task_name in PRIMARY_TASKS if task_name not in control_metrics
    ]
    if missing_primary:
        raise FileNotFoundError(
            f"Missing control metrics for primary tasks: {', '.join(missing_primary)}."
        )
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        task_payloads = metrics_tree.get(arm_id, {})
        missing_tasks = [task_name for task_name in PRIMARY_TASKS if task_name not in task_payloads]
        if missing_tasks:
            raise FileNotFoundError(
                f"Missing primary-task metrics for {arm_id}: {', '.join(missing_tasks)}."
            )
        task_summaries: dict[str, dict[str, Any]] = {}
        for task_name in PRIMARY_TASKS:
            task_summaries[task_name] = _replay_task_summary(
                control_metrics=control_metrics[task_name]["metrics"],
                branch_metrics=task_payloads[task_name]["metrics"],
                branch_train_events=task_payloads[task_name]["train_events"],
                head_window=args.head_window,
                post_unfreeze_window=args.post_unfreeze_window,
                tail_window=args.tail_window,
            )
        arm_summaries[arm_id] = _arm_summary(arm_id=arm_id, task_summaries=task_summaries)

    primary_arm_ranking = _rank_arms(arm_summaries)
    promoted_arms = [payload["arm_id"] for payload in primary_arm_ranking[:2]]
    fever_guardrail = _fever_guardrail_summary(
        metrics_tree=metrics_tree,
        promoted_arms=promoted_arms,
        head_window=args.head_window,
        post_unfreeze_window=args.post_unfreeze_window,
        tail_window=args.tail_window,
    )
    evidence = _bandwidth_evidence(arm_summaries=arm_summaries)
    comparison_conclusion, recommended_next_step = _select_conclusion(
        arm_summaries=arm_summaries,
        primary_arm_ranking=primary_arm_ranking,
        evidence=evidence,
    )
    owner_metadata = {
        "owner_locked_projector_lr": _as_float(
            control_metrics["gsm8k"]["metrics"], "owner_locked_projector_lr", 7.5e-6
        ),
        "repo_confirmed_v65_projector_lr_reference": _as_float(
            control_metrics["gsm8k"]["metrics"], "repo_confirmed_v65_projector_lr_reference", 7.5e-5
        ),
        "owner_override_note": bool(
            control_metrics["gsm8k"]["metrics"].get("owner_override_note", True)
        ),
    }
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "winning_depth": "D1",
        "winning_depth_label": "mid4",
        "promoted_arms": promoted_arms,
        "owner_lr_discrepancy_metadata": owner_metadata,
        "arms": arm_summaries,
        "primary_arm_ranking": primary_arm_ranking,
        "fever_guardrail": fever_guardrail,
        "evidence": evidence,
        "acceptance": {
            "main_primary_matrix_complete": len(arm_summaries) == len(ARM_ORDER),
            "fever_guardrail_complete": bool(fever_guardrail["complete"]),
            "repo_can_answer_bandwidth_question": True,
            "repo_can_answer_projector_scaling_question": True,
        },
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# PLANv7 V7-2 Direct Bandwidth Summary",
        "",
        f"- comparison_conclusion: {comparison_conclusion}",
        f"- recommended_next_step: {recommended_next_step}",
        f"- winning_depth: D1",
        f"- winning_depth_label: mid4",
        f"- promoted_arms: {', '.join(promoted_arms)}",
        "",
        "## Owner LR Metadata",
        f"- owner_locked_projector_lr: {owner_metadata['owner_locked_projector_lr']}",
        f"- repo_confirmed_v65_projector_lr_reference: {owner_metadata['repo_confirmed_v65_projector_lr_reference']}",
        f"- owner_override_note: {owner_metadata['owner_override_note']}",
        "",
        "## Evidence",
        f"- bandwidth_strict_improvement_observed: {evidence['bandwidth_strict_improvement_observed']}",
        f"- projector_scaling_usefulness_change_observed: {evidence['projector_scaling_usefulness_change_observed']}",
        f"- w1_shared_strict_writer_memory_task_count: {evidence['w1_shared_strict_writer_memory_task_count']}",
        f"- w2_shared_strict_writer_memory_task_count: {evidence['w2_shared_strict_writer_memory_task_count']}",
        f"- w2_perlayer_strict_writer_memory_task_count: {evidence['w2_perlayer_strict_writer_memory_task_count']}",
        f"- w1_shared_strict_rank_fraction_mean: {evidence['w1_shared_strict_rank_fraction_mean']:.6f}",
        f"- w2_shared_strict_rank_fraction_mean: {evidence['w2_shared_strict_rank_fraction_mean']:.6f}",
        f"- w2_perlayer_strict_rank_fraction_mean: {evidence['w2_perlayer_strict_rank_fraction_mean']:.6f}",
        f"- w2_shared_primary_task_score_delta_sum: {evidence['w2_shared_primary_task_score_delta_sum']:.6f}",
        f"- w2_perlayer_primary_task_score_delta_sum: {evidence['w2_perlayer_primary_task_score_delta_sum']:.6f}",
        "",
        "## FEVER Guardrail",
        f"- complete: {fever_guardrail['complete']}",
        f"- fever_not_used_to_override_primary: {fever_guardrail['fever_not_used_to_override_primary']}",
        f"- evaluated_arms: {', '.join(fever_guardrail['evaluated_arms'])}",
        f"- missing_reason: {fever_guardrail['missing_reason']}",
    ]
    for arm_id in ARM_ORDER:
        arm_summary = arm_summaries[arm_id]
        report_lines.extend(
            [
                "",
                f"## {arm_id}",
                f"- writer_family: {arm_summary['writer_family']}",
                f"- depth_family: {arm_summary['depth_family']}",
                f"- projector_family: {arm_summary['projector_family']}",
                f"- projector_mode: {arm_summary['projector_mode']}",
                f"- bridge_mode: {arm_summary['bridge_mode']}",
                f"- memory_path_variant: {arm_summary['memory_path_variant']}",
                f"- active_depth_layers: {arm_summary['active_depth_layers']}",
                f"- projector_rank: {arm_summary['projector_rank']}",
                f"- writer_memory_slots: {arm_summary['writer_memory_slots']}",
                f"- writer_conditioning_layers: {arm_summary['writer_conditioning_layers']}",
                f"- gsm8k_task_score_delta_vs_control: {arm_summary['gsm8k_task_score_delta_vs_control']:.6f}",
                f"- triviaqa_task_score_delta_vs_control: {arm_summary['triviaqa_task_score_delta_vs_control']:.6f}",
                f"- primary_task_score_delta_sum: {arm_summary['primary_task_score_delta_sum']:.6f}",
                f"- strict_writer_memory_task_count: {arm_summary['strict_writer_memory_task_count']}",
                f"- strict_rank_fraction_mean: {arm_summary['strict_rank_fraction_mean']:.6f}",
                f"- projector_rank_gain_factor_mean: {arm_summary['projector_rank_gain_factor_mean']:.6f}",
                f"- route_live_task_count: {arm_summary['route_live_task_count']}",
                f"- stable_training_task_count: {arm_summary['stable_training_task_count']}",
                f"- primary_branch_success_task_count: {arm_summary['primary_branch_success_task_count']}",
                f"- any_primary_usefulness_positive: {arm_summary['any_primary_usefulness_positive']}",
                f"- answer_switch_helpfulness_score: {arm_summary['answer_switch_helpfulness_score']:.6f}",
                f"- ranking_penalty_count: {arm_summary['ranking_penalty_count']}",
            ]
        )
        for task_name in PRIMARY_TASKS:
            task_summary = arm_summary["tasks"][task_name]
            report_lines.extend(
                [
                    f"- {task_name}.task_score_delta_vs_control: {task_summary['task_score_delta_vs_control']:.6f}",
                    f"- {task_name}.writer_memory_not_collapsed_strict: {task_summary['writer_memory_not_collapsed_strict']}",
                    f"- {task_name}.primary_usefulness_positive: {task_summary['primary_usefulness_positive']}",
                    f"- {task_name}.primary_branch_success: {task_summary['primary_branch_success']}",
                    f"- {task_name}.route_live_post_unfreeze: {task_summary['route_live_post_unfreeze']}",
                    f"- {task_name}.stable_training_v6: {task_summary['stable_training_v6']}",
                    f"- {task_name}.projector_manufactured_diversity: {task_summary['projector_manufactured_diversity']}",
                ]
            )
        fever_task = fever_guardrail["branches"].get(arm_id)
        if fever_task is not None:
            report_lines.extend(
                [
                    f"- fever.task_score_delta_vs_control: {float(fever_task['task_score_delta_vs_control']):.6f}",
                    f"- fever.writer_memory_not_collapsed_strict: {fever_task['writer_memory_not_collapsed_strict']}",
                    f"- fever.primary_usefulness_positive: {fever_task['primary_usefulness_positive']}",
                ]
            )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
