#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

PRIMARY_TASKS = ("gsm8k", "triviaqa")
ARM_ORDER = ("b_w3_q8", "b_w3_q16", "b_w3_q16_s8", "b_w4_q16")
BRIDGE_ARM_METADATA = {
    "b_w3_q8": {
        "writer_family": "W3",
        "bridge_family": "B1",
        "projector_family": "P2",
        "projector_mode": "per_layer_low_rank",
        "bridge_queries": 8,
        "short_slots": 8,
    },
    "b_w3_q16": {
        "writer_family": "W3",
        "bridge_family": "B2",
        "projector_family": "P2",
        "projector_mode": "per_layer_low_rank",
        "bridge_queries": 16,
        "short_slots": 16,
    },
    "b_w3_q16_s8": {
        "writer_family": "W3",
        "bridge_family": "B3",
        "projector_family": "P2",
        "projector_mode": "per_layer_low_rank",
        "bridge_queries": 16,
        "short_slots": 8,
    },
    "b_w4_q16": {
        "writer_family": "W4",
        "bridge_family": "B2",
        "projector_family": "P3",
        "projector_mode": "per_layer_low_rank",
        "bridge_queries": 16,
        "short_slots": 16,
    },
}
DIRECT_CONTROL_METADATA = {
    "d_w1_shared": {
        "writer_family": "W1",
        "bridge_family": "B0",
        "projector_family": "P1_shared_rank64",
        "projector_mode": "shared_low_rank",
        "bridge_queries": 0,
        "short_slots": 0,
    },
    "d_w2_shared": {
        "writer_family": "W2",
        "bridge_family": "B0",
        "projector_family": "shared_rank64_control",
        "projector_mode": "shared_low_rank",
        "bridge_queries": 0,
        "short_slots": 0,
    },
    "d_w2_perlayer": {
        "writer_family": "W2",
        "bridge_family": "B0",
        "projector_family": "P2_per_layer_rank128",
        "projector_mode": "per_layer_low_rank",
        "bridge_queries": 0,
        "short_slots": 0,
    },
}
DEPTH_LABELS = {"D0": "early4", "D1": "mid4", "D2": "wider_mid", "D3": "hybrid"}


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


def _load_v72_selection(summary_path: Path) -> tuple[str, str, str]:
    payload = _load_json(summary_path)
    ranking = payload.get("primary_arm_ranking", [])
    if not isinstance(ranking, list) or not ranking:
        raise ValueError(f"Missing primary_arm_ranking in {summary_path}.")
    control_arm_id = str(ranking[0].get("arm_id", "")).strip()
    if control_arm_id not in DIRECT_CONTROL_METADATA:
        raise ValueError(
            f"Unsupported V7-2 direct control arm {control_arm_id!r}; "
            f"expected one of {sorted(DIRECT_CONTROL_METADATA)}."
        )
    winning_depth = str(payload.get("winning_depth", "D1")).strip() or "D1"
    winning_depth_label = str(payload.get("winning_depth_label", DEPTH_LABELS.get(winning_depth, "mid4"))).strip()
    if not winning_depth_label:
        winning_depth_label = DEPTH_LABELS.get(winning_depth, "mid4")
    return control_arm_id, winning_depth, winning_depth_label


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


def _patched_short_run_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    patched = dict(metrics)
    train_steps = int(_as_float(metrics, "pilot_train_steps"))
    if train_steps < 451:
        patched["train_loss_steps_451_500_median"] = _as_float(
            metrics,
            "train_loss_tail_50_steps_median",
            _as_float(metrics, "train_loss_steps_451_500_median"),
        )
    return patched


def _bridge_task_summary(
    *,
    control_metrics: dict[str, Any],
    branch_metrics: dict[str, Any],
    branch_train_events: list[dict[str, Any]],
    head_window: int,
    post_unfreeze_window: int,
    tail_window: int,
) -> dict[str, Any]:
    patched_branch_metrics = _patched_short_run_metrics(branch_metrics)
    summary = _replay_task_summary(
        control_metrics=control_metrics,
        branch_metrics=patched_branch_metrics,
        branch_train_events=branch_train_events,
        head_window=head_window,
        post_unfreeze_window=post_unfreeze_window,
        tail_window=tail_window,
    )
    summary["loss_tail_50_steps_median"] = _as_float(
        branch_metrics,
        "train_loss_tail_50_steps_median",
        summary["loss_steps_451_500_median"],
    )
    summary["tail_window_source"] = (
        "train_loss_tail_50_steps_median"
        if int(_as_float(branch_metrics, "pilot_train_steps")) < 451
        else "train_loss_steps_451_500_median"
    )
    summary["pilot_reader_context_mode"] = str(branch_metrics.get("pilot_reader_context_mode", "prompt_summary"))
    summary["pilot_projector_token_source"] = str(branch_metrics.get("pilot_projector_token_source", "writer_slots"))
    summary["pilot_reader_num_queries"] = int(round(_as_float(branch_metrics, "pilot_reader_num_queries", 0.0)))
    summary["pilot_fuser_short_slots"] = int(round(_as_float(branch_metrics, "pilot_fuser_short_slots", 0.0)))
    summary["pilot_deep_prefix_projector_mode"] = str(
        branch_metrics.get("pilot_deep_prefix_projector_mode", "shared_low_rank")
    )
    return summary


def _aggregate_arm_summary(
    *,
    arm_id: str,
    metadata: dict[str, Any],
    task_summaries: dict[str, dict[str, Any]],
    depth_family: str,
    depth_label: str,
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
        "projector_manufactured_diversity": any(
            bool(task["projector_manufactured_diversity"]) for task in task_summaries.values()
        ),
    }
    penalty_count = int(sum(int(value) for value in penalties.values()))
    metrics_ref = task_summaries["gsm8k"]
    writer_slots = int(round(float(metrics_ref.get("writer_memory_slots", 0.0))))
    return {
        "arm_id": arm_id,
        "writer_family": metadata["writer_family"],
        "depth_family": depth_family,
        "depth_label": depth_label,
        "bridge_family": metadata["bridge_family"],
        "projector_family": metadata["projector_family"],
        "projector_mode": str(metrics_ref.get("pilot_deep_prefix_projector_mode", metadata["projector_mode"])),
        "bridge_mode": str(metrics_ref.get("pilot_bridge_mode", "writer_direct")),
        "memory_path_variant": str(metrics_ref.get("pilot_memory_path_variant", "single_level")),
        "projector_token_source": str(metrics_ref.get("pilot_projector_token_source", "writer_slots")),
        "reader_context_mode": str(metrics_ref.get("pilot_reader_context_mode", "prompt_summary")),
        "bridge_queries": int(metadata["bridge_queries"]),
        "runtime_reader_queries": int(metrics_ref.get("pilot_reader_num_queries", 0)),
        "bridge_short_slots": int(metadata["short_slots"]),
        "runtime_fuser_short_slots": int(metrics_ref.get("pilot_fuser_short_slots", 0)),
        "active_depth_layers": metrics_ref.get("pilot_deep_prefix_layers", []),
        "receiver_lora_layers": metrics_ref.get("pilot_receiver_lora_target_layers", []),
        "projector_rank": int(round(float(metrics_ref.get("pilot_deep_prefix_rank", 0.0)))),
        "writer_memory_slots": writer_slots,
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
            float(primary_success_count),
            float(strict_count),
            float(stability_count),
            float(route_count),
            strict_rank_fraction_mean,
            helpfulness_score,
            float(writer_slots),
            float(-penalty_count),
        ],
        "tasks": task_summaries,
    }


def _rank_bridge_arms(arm_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "arm_id": arm_id,
                "writer_family": summary["writer_family"],
                "bridge_family": summary["bridge_family"],
                "projector_family": summary["projector_family"],
                "projector_mode": summary["projector_mode"],
                "ranking_key": summary["ranking_key"],
            }
            for arm_id, summary in arm_summaries.items()
        ),
        key=lambda payload: tuple(payload["ranking_key"]),
        reverse=True,
    )


def _bridge_evidence(
    *,
    direct_control: dict[str, Any],
    arm_summaries: dict[str, dict[str, Any]],
    bridge_arm_ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    control_slots = int(direct_control["writer_memory_slots"])
    control_strict_count = int(direct_control["strict_writer_memory_task_count"])
    control_rank_fraction = float(direct_control["strict_rank_fraction_mean"])
    top_bridge_arm_id = bridge_arm_ranking[0]["arm_id"]
    any_primary_score_improvement = any(
        any(float(summary["tasks"][task_name]["task_score_delta_vs_control"]) > 0.0 for task_name in PRIMARY_TASKS)
        for summary in arm_summaries.values()
    )
    any_strict_metric_gain = any(
        int(summary["strict_writer_memory_task_count"]) > control_strict_count
        or float(summary["strict_rank_fraction_mean"]) > control_rank_fraction + 0.02
        for summary in arm_summaries.values()
    )
    any_wide_writer_stable_non_regressive = any(
        int(summary["writer_memory_slots"]) > control_slots
        and int(summary["stable_training_task_count"]) == len(PRIMARY_TASKS)
        and int(summary["route_live_task_count"]) == len(PRIMARY_TASKS)
        and float(summary["primary_task_score_delta_sum"]) >= -1e-12
        for summary in arm_summaries.values()
    )
    bridge_stabilizes_wide_writer = bool(
        not any_primary_score_improvement
        and (any_wide_writer_stable_non_regressive or any_strict_metric_gain)
    )
    return {
        "top_bridge_arm_id": top_bridge_arm_id,
        "any_bridge_primary_score_improvement": any_primary_score_improvement,
        "any_bridge_strict_metric_gain": any_strict_metric_gain,
        "any_bridge_wide_writer_stable_non_regressive": any_wide_writer_stable_non_regressive,
        "bridge_stabilizes_wide_writer_without_primary_gain": bridge_stabilizes_wide_writer,
        "direct_control_writer_slots": control_slots,
        "direct_control_strict_writer_memory_task_count": control_strict_count,
        "direct_control_strict_rank_fraction_mean": control_rank_fraction,
    }


def _select_conclusion(
    *,
    evidence: dict[str, Any],
) -> tuple[str, str]:
    if bool(evidence["any_bridge_primary_score_improvement"]):
        return (
            "bridge_beats_direct_control_promote_bridge_winner",
            "open_v7_4_forced_consumption_from_bridge_winner",
        )
    if bool(evidence["bridge_stabilizes_wide_writer_without_primary_gain"]):
        return (
            "bridge_stabilizes_wide_writer_tasks_flat_move_to_v7_4",
            "open_v7_4_forced_consumption",
        )
    return (
        "bridge_flat_open_v7_5_reconstruction_aux",
        "open_v7_5_reconstruction_aux",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PLANv7 V7-3 bounded bridge sweep.")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--v72_summary", required=True)
    parser.add_argument("--head_window", type=int, default=50)
    parser.add_argument("--post_unfreeze_window", type=int, default=50)
    parser.add_argument("--tail_window", type=int, default=50)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result_root = Path(args.result_root).resolve()
    v72_summary_path = Path(args.v72_summary).resolve()
    direct_control_arm_id, winning_depth, winning_depth_label = _load_v72_selection(v72_summary_path)
    metrics_tree = _load_metrics_tree(result_root)
    control_payloads = metrics_tree.get("control", {})
    missing_control_tasks = [task_name for task_name in PRIMARY_TASKS if task_name not in control_payloads]
    if missing_control_tasks:
        raise FileNotFoundError(
            f"Missing direct control metrics for primary tasks: {', '.join(missing_control_tasks)}."
        )

    direct_control_tasks: dict[str, dict[str, Any]] = {}
    for task_name in PRIMARY_TASKS:
        direct_control_tasks[task_name] = _bridge_task_summary(
            control_metrics=control_payloads[task_name]["metrics"],
            branch_metrics=control_payloads[task_name]["metrics"],
            branch_train_events=control_payloads[task_name]["train_events"],
            head_window=args.head_window,
            post_unfreeze_window=args.post_unfreeze_window,
            tail_window=args.tail_window,
        )
    direct_control = _aggregate_arm_summary(
        arm_id=direct_control_arm_id,
        metadata=DIRECT_CONTROL_METADATA[direct_control_arm_id],
        task_summaries=direct_control_tasks,
        depth_family=winning_depth,
        depth_label=winning_depth_label,
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
            task_summaries[task_name] = _bridge_task_summary(
                control_metrics=control_payloads[task_name]["metrics"],
                branch_metrics=task_payloads[task_name]["metrics"],
                branch_train_events=task_payloads[task_name]["train_events"],
                head_window=args.head_window,
                post_unfreeze_window=args.post_unfreeze_window,
                tail_window=args.tail_window,
            )
        arm_summaries[arm_id] = _aggregate_arm_summary(
            arm_id=arm_id,
            metadata=BRIDGE_ARM_METADATA[arm_id],
            task_summaries=task_summaries,
            depth_family=winning_depth,
            depth_label=winning_depth_label,
        )

    bridge_arm_ranking = _rank_bridge_arms(arm_summaries)
    evidence = _bridge_evidence(
        direct_control=direct_control,
        arm_summaries=arm_summaries,
        bridge_arm_ranking=bridge_arm_ranking,
    )
    comparison_conclusion, recommended_next_step = _select_conclusion(evidence=evidence)
    owner_metadata = {
        "owner_locked_projector_lr": _as_float(
            control_payloads["gsm8k"]["metrics"], "owner_locked_projector_lr", 7.5e-6
        ),
        "repo_confirmed_v65_projector_lr_reference": _as_float(
            control_payloads["gsm8k"]["metrics"], "repo_confirmed_v65_projector_lr_reference", 7.5e-5
        ),
        "owner_override_note": bool(
            control_payloads["gsm8k"]["metrics"].get("owner_override_note", True)
        ),
    }
    summary = {
        "comparison_conclusion": comparison_conclusion,
        "recommended_next_step": recommended_next_step,
        "winning_depth": winning_depth,
        "winning_depth_label": winning_depth_label,
        "direct_control_arm_id": direct_control_arm_id,
        "direct_control": direct_control,
        "arms": arm_summaries,
        "bridge_arm_ranking": bridge_arm_ranking,
        "owner_lr_discrepancy_metadata": owner_metadata,
        "evidence": evidence,
        "acceptance": {
            "direct_control_complete": True,
            "main_bridge_matrix_complete": len(arm_summaries) == len(ARM_ORDER),
            "repo_can_answer_bridge_vs_direct_question": True,
            "repo_can_answer_bounded_compression_question": True,
        },
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# PLANv7 V7-3 Bridge Summary",
        "",
        f"- comparison_conclusion: {comparison_conclusion}",
        f"- recommended_next_step: {recommended_next_step}",
        f"- direct_control_arm_id: {direct_control_arm_id}",
        f"- winning_depth: {winning_depth}",
        f"- winning_depth_label: {winning_depth_label}",
        f"- top_bridge_arm_id: {evidence['top_bridge_arm_id']}",
        "",
        "## Owner LR Metadata",
        f"- owner_locked_projector_lr: {owner_metadata['owner_locked_projector_lr']}",
        f"- repo_confirmed_v65_projector_lr_reference: {owner_metadata['repo_confirmed_v65_projector_lr_reference']}",
        f"- owner_override_note: {owner_metadata['owner_override_note']}",
        "",
        "## Evidence",
        f"- any_bridge_primary_score_improvement: {evidence['any_bridge_primary_score_improvement']}",
        f"- any_bridge_strict_metric_gain: {evidence['any_bridge_strict_metric_gain']}",
        f"- any_bridge_wide_writer_stable_non_regressive: {evidence['any_bridge_wide_writer_stable_non_regressive']}",
        (
            f"- bridge_stabilizes_wide_writer_without_primary_gain: "
            f"{evidence['bridge_stabilizes_wide_writer_without_primary_gain']}"
        ),
        f"- direct_control_writer_slots: {evidence['direct_control_writer_slots']}",
        (
            f"- direct_control_strict_writer_memory_task_count: "
            f"{evidence['direct_control_strict_writer_memory_task_count']}"
        ),
        (
            f"- direct_control_strict_rank_fraction_mean: "
            f"{evidence['direct_control_strict_rank_fraction_mean']:.6f}"
        ),
        "",
        "## Direct Control",
        f"- writer_family: {direct_control['writer_family']}",
        f"- bridge_family: {direct_control['bridge_family']}",
        f"- projector_family: {direct_control['projector_family']}",
        f"- projector_mode: {direct_control['projector_mode']}",
        f"- memory_path_variant: {direct_control['memory_path_variant']}",
        f"- projector_token_source: {direct_control['projector_token_source']}",
        f"- active_depth_layers: {direct_control['active_depth_layers']}",
        f"- writer_memory_slots: {direct_control['writer_memory_slots']}",
    ]
    for task_name in PRIMARY_TASKS:
        task_summary = direct_control["tasks"][task_name]
        report_lines.extend(
            [
                f"- control.{task_name}.task_score: {task_summary['task_score']:.6f}",
                f"- control.{task_name}.route_live_post_unfreeze: {task_summary['route_live_post_unfreeze']}",
                f"- control.{task_name}.stable_training_v6: {task_summary['stable_training_v6']}",
            ]
        )
    for arm_id in ARM_ORDER:
        arm_summary = arm_summaries[arm_id]
        report_lines.extend(
            [
                "",
                f"## {arm_id}",
                f"- writer_family: {arm_summary['writer_family']}",
                f"- bridge_family: {arm_summary['bridge_family']}",
                f"- projector_family: {arm_summary['projector_family']}",
                f"- projector_mode: {arm_summary['projector_mode']}",
                f"- bridge_mode: {arm_summary['bridge_mode']}",
                f"- memory_path_variant: {arm_summary['memory_path_variant']}",
                f"- projector_token_source: {arm_summary['projector_token_source']}",
                f"- reader_context_mode: {arm_summary['reader_context_mode']}",
                f"- bridge_queries: {arm_summary['bridge_queries']}",
                f"- bridge_short_slots: {arm_summary['bridge_short_slots']}",
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
                    f"- {task_name}.tail_window_source: {task_summary['tail_window_source']}",
                ]
            )
    Path(args.output_report).write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
