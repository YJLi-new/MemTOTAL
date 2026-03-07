from __future__ import annotations

import csv
import json
from math import ceil
from pathlib import Path

from memtotal.utils.io import write_json


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _load_backbone(run_dir: Path, metrics: dict[str, object]) -> str:
    run_info_path = run_dir / "run_info.json"
    if run_info_path.exists():
        run_info = _read_json(run_info_path)
        backbone = str(run_info.get("backbone", "")).strip()
        if backbone:
            return backbone
    backbone = str(metrics.get("backbone", "")).strip()
    return backbone or "unknown"


def _load_seed(run_dir: Path) -> int | None:
    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        return None
    run_info = _read_json(run_info_path)
    seed = run_info.get("seed")
    if seed is None:
        return None
    return int(seed)


def _load_adapt_cost(run_dir: Path) -> dict[str, object]:
    adapt_cost_path = run_dir / "adapt_cost.json"
    if not adapt_cost_path.exists():
        return {}
    return _read_json(adapt_cost_path)


def collect_stage_c_seed_sweep_rows(input_root: str | Path) -> list[dict[str, object]]:
    root = Path(input_root).resolve()
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = _read_json(metrics_path)
        if metrics.get("training_stage") != "stage_c":
            continue
        if str(metrics.get("adaptation_target", "")) != "q_only":
            continue
        run_dir = metrics_path.parent
        adapt_cost = _load_adapt_cost(run_dir)
        zero_shot_task_score = float(metrics.get("zero_shot_task_score") or 0.0)
        best_adapt_task_score = float(metrics.get("best_adapt_task_score") or 0.0)
        zero_shot_task_proxy_score = float(metrics.get("zero_shot_task_proxy_score") or 0.0)
        best_adapt_task_proxy_score = float(metrics.get("best_adapt_task_proxy_score") or 0.0)
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "backbone": _load_backbone(run_dir, metrics),
                "seed": _load_seed(run_dir),
                "query_learning_mode": metrics.get("query_learning_mode"),
                "query_objective": metrics.get("query_objective"),
                "adaptation_target": metrics.get("adaptation_target"),
                "trainable_module": metrics.get("trainable_module"),
                "trainable_parameter_count": metrics.get("trainable_parameter_count"),
                "adapt_learning_rate": metrics.get("adapt_learning_rate"),
                "adapt_steps": metrics.get("adapt_steps"),
                "target_eval_repeats": metrics.get("target_eval_repeats"),
                "target_episode_repeats": metrics.get("target_episode_repeats"),
                "target_episode_policy": metrics.get("target_episode_policy"),
                "target_support_weighting": metrics.get("target_support_weighting"),
                "target_split_policy": metrics.get("target_split_policy"),
                "target_support_bank_size": metrics.get("target_support_bank_size"),
                "support_updates": metrics.get("support_updates", adapt_cost.get("support_updates")),
                "support_examples_touched": metrics.get(
                    "support_examples_touched",
                    adapt_cost.get("support_examples_touched"),
                ),
                "zero_shot_task_score": zero_shot_task_score,
                "best_adapt_task_score": best_adapt_task_score,
                "task_gain": best_adapt_task_score - zero_shot_task_score,
                "task_metric_name": metrics.get("task_metric_name"),
                "zero_shot_task_proxy_score": zero_shot_task_proxy_score,
                "best_adapt_task_proxy_score": best_adapt_task_proxy_score,
                "proxy_gain": best_adapt_task_proxy_score - zero_shot_task_proxy_score,
                "task_proxy_name": metrics.get("task_proxy_name"),
                "best_adapt_task_margin": metrics.get("best_adapt_task_margin"),
                "zero_shot_query_loss": metrics.get("zero_shot_query_loss"),
                "best_adapt_query_loss": metrics.get("best_adapt_query_loss"),
                "best_adapt_shot": metrics.get("best_adapt_shot"),
                "best_adapt_step": metrics.get("best_adapt_step"),
                "adaptation_effective": metrics.get("adaptation_effective"),
            }
        )
    rows.sort(key=lambda row: (str(row["backbone"]), int(row["seed"] or -1), str(row["run_name"])))
    return rows


def write_stage_c_seed_sweep_csv(output_path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "backbone",
        "seed",
        "query_learning_mode",
        "query_objective",
        "adaptation_target",
        "trainable_module",
        "trainable_parameter_count",
        "adapt_learning_rate",
        "adapt_steps",
        "target_eval_repeats",
        "target_episode_repeats",
        "target_episode_policy",
        "target_support_weighting",
        "target_split_policy",
        "target_support_bank_size",
        "support_updates",
        "support_examples_touched",
        "zero_shot_task_score",
        "best_adapt_task_score",
        "task_gain",
        "task_metric_name",
        "zero_shot_task_proxy_score",
        "best_adapt_task_proxy_score",
        "proxy_gain",
        "task_proxy_name",
        "best_adapt_task_margin",
        "zero_shot_query_loss",
        "best_adapt_query_loss",
        "best_adapt_shot",
        "best_adapt_step",
        "adaptation_effective",
        "run_dir",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_stage_c_seed_sweep_svg(output_path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='560' height='120'>"
            "<text x='24' y='64' font-size='18' font-family='monospace'>No Stage C seed sweep rows</text></svg>"
        )
        return

    values = [float(row.get("task_gain") or 0.0) for row in rows]
    max_abs = max(max(abs(value) for value in values), 1e-6)
    width = 980
    height = 132 + 54 * len(rows)
    center_x = 520
    half_bar = 180
    palette = {
        "Qwen2.5-1.5B-Instruct": "#2f5aa8",
        "Qwen3-8B": "#b5651d",
    }
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='34' font-size='20' font-family='monospace'>M3 Stage C q_only seed sweep</text>",
        "<text x='24' y='56' font-size='13' font-family='monospace'>task_gain = best_adapt_task_score - zero_shot_task_score</text>",
        f"<line x1='{center_x}' y1='82' x2='{center_x}' y2='{height - 24}' stroke='#6b5f47' stroke-width='2' />",
    ]
    for index, row in enumerate(rows):
        top = 92 + index * 46
        value = float(row.get("task_gain") or 0.0)
        scaled = ceil((abs(value) / max_abs) * half_bar)
        bar_width = max(scaled, 2 if value != 0 else 0)
        bar_x = center_x if value >= 0 else center_x - bar_width
        backbone = str(row["backbone"])
        parts.append(
            f"<text x='24' y='{top + 16}' font-size='12' font-family='monospace'>{backbone} {row.get('target_split_policy') or 'random'} {row.get('target_support_weighting') or 'uniform'} seed={row['seed']}</text>"
        )
        parts.append(
            f"<rect x='{center_x - half_bar}' y='{top}' width='{half_bar * 2}' height='24' fill='#efe6d0' rx='4' />"
        )
        if bar_width > 0:
            parts.append(
                f"<rect x='{bar_x}' y='{top}' width='{bar_width}' height='24' fill='{palette.get(backbone, '#4a6fa5')}' rx='4' />"
            )
        score_label = (
            f"{float(row.get('zero_shot_task_score') or 0.0):.3f}->{float(row.get('best_adapt_task_score') or 0.0):.3f}"
        )
        proxy_label = (
            f"{float(row.get('zero_shot_task_proxy_score') or 0.0):.3f}->{float(row.get('best_adapt_task_proxy_score') or 0.0):.3f}"
        )
        parts.append(
            f"<text x='{center_x + half_bar + 16}' y='{top + 16}' font-size='12' font-family='monospace'>{value:.3f} score={score_label}</text>"
        )
        parts.append(
            f"<text x='{center_x + half_bar + 16}' y='{top + 30}' font-size='11' font-family='monospace'>proxy={proxy_label} episodes={row.get('target_episode_repeats') or 1} queries={row.get('target_eval_repeats') or 1} split={row.get('target_split_policy') or 'random'}</text>"
        )
    parts.append("</svg>")
    destination.write_text("".join(parts))


def run_m3_stage_c_seed_sweep_summary(
    *,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    rows = collect_stage_c_seed_sweep_rows(input_root)
    if dry_run:
        rows = rows[: max(1, min(4, len(rows)))]

    summary_csv = output_dir / "seed_sweep.csv"
    summary_svg = output_dir / "seed_sweep.svg"
    write_stage_c_seed_sweep_csv(summary_csv, rows)
    write_stage_c_seed_sweep_svg(summary_svg, rows)

    def _summarize(group_rows: list[dict[str, object]]) -> dict[str, object]:
        positive = [row for row in group_rows if float(row["task_gain"]) > 0.0]
        non_negative = [row for row in group_rows if float(row["task_gain"]) >= 0.0]
        best_row = max(group_rows, key=lambda row: (float(row["task_gain"]), float(row["proxy_gain"])))
        worst_row = min(group_rows, key=lambda row: (float(row["task_gain"]), float(row["proxy_gain"])))
        return {
            "seed_count": len(group_rows),
            "positive_gain_count": len(positive),
            "positive_gain_rate": len(positive) / len(group_rows),
            "non_negative_gain_rate": len(non_negative) / len(group_rows),
            "mean_task_gain": sum(float(row["task_gain"]) for row in group_rows) / len(group_rows),
            "mean_proxy_gain": sum(float(row["proxy_gain"]) for row in group_rows) / len(group_rows),
            "mean_support_updates": sum(float(row.get("support_updates") or 0.0) for row in group_rows)
            / len(group_rows),
            "mean_support_examples_touched": sum(
                float(row.get("support_examples_touched") or 0.0) for row in group_rows
            )
            / len(group_rows),
            "target_eval_repeats": sorted({int(row.get("target_eval_repeats") or 1) for row in group_rows}),
            "target_episode_repeats": sorted(
                {int(row.get("target_episode_repeats") or 1) for row in group_rows}
            ),
            "target_episode_policies": sorted(
                {str(row.get("target_episode_policy") or "independent") for row in group_rows}
            ),
            "target_support_weightings": sorted(
                {str(row.get("target_support_weighting") or "uniform") for row in group_rows}
            ),
            "target_split_policies": sorted(
                {str(row.get("target_split_policy") or "random") for row in group_rows}
            ),
            "target_support_bank_sizes": sorted(
                {str(row.get("target_support_bank_size") or "auto") for row in group_rows}
            ),
            "best_seed": best_row["seed"],
            "best_task_gain": best_row["task_gain"],
            "worst_seed": worst_row["seed"],
            "worst_task_gain": worst_row["task_gain"],
        }

    by_backbone: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        backbone_rows = [row for row in rows if str(row["backbone"]) == backbone]
        by_backbone[backbone] = _summarize(backbone_rows)

    by_backbone_policy: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        for policy in sorted({str(row.get("target_episode_policy") or "independent") for row in rows if str(row["backbone"]) == backbone}):
            grouped_rows = [
                row
                for row in rows
                if str(row["backbone"]) == backbone
                and str(row.get("target_episode_policy") or "independent") == policy
            ]
            if not grouped_rows:
                continue
            by_backbone_policy[f"{backbone}::{policy}"] = _summarize(grouped_rows)

    by_backbone_episode_budget: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        for repeats in sorted({int(row.get("target_episode_repeats") or 1) for row in rows if str(row["backbone"]) == backbone}):
            grouped_rows = [
                row
                for row in rows
                if str(row["backbone"]) == backbone
                and int(row.get("target_episode_repeats") or 1) == repeats
            ]
            if not grouped_rows:
                continue
            by_backbone_episode_budget[f"{backbone}::episodes={repeats}"] = _summarize(grouped_rows)

    by_backbone_support_weighting: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        for weighting in sorted(
            {str(row.get("target_support_weighting") or "uniform") for row in rows if str(row["backbone"]) == backbone}
        ):
            grouped_rows = [
                row
                for row in rows
                if str(row["backbone"]) == backbone
                and str(row.get("target_support_weighting") or "uniform") == weighting
            ]
            if not grouped_rows:
                continue
            by_backbone_support_weighting[f"{backbone}::weight={weighting}"] = _summarize(grouped_rows)

    by_backbone_target_split: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        for split_policy in sorted(
            {str(row.get("target_split_policy") or "random") for row in rows if str(row["backbone"]) == backbone}
        ):
            grouped_rows = [
                row
                for row in rows
                if str(row["backbone"]) == backbone
                and str(row.get("target_split_policy") or "random") == split_policy
            ]
            if not grouped_rows:
                continue
            by_backbone_target_split[f"{backbone}::split={split_policy}"] = _summarize(grouped_rows)

    by_backbone_support_bank_size: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        for bank_size in sorted(
            {str(row.get("target_support_bank_size") or "auto") for row in rows if str(row["backbone"]) == backbone}
        ):
            grouped_rows = [
                row
                for row in rows
                if str(row["backbone"]) == backbone
                and str(row.get("target_support_bank_size") or "auto") == bank_size
            ]
            if not grouped_rows:
                continue
            by_backbone_support_bank_size[f"{backbone}::bank={bank_size}"] = _summarize(grouped_rows)

    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_seed_sweep_summary",
        "rows_collected": len(rows),
        "input_root": str(Path(input_root).resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "summary_plot": str(summary_svg.resolve()),
        "by_backbone": by_backbone,
        "by_backbone_policy": by_backbone_policy,
        "by_backbone_episode_budget": by_backbone_episode_budget,
        "by_backbone_support_weighting": by_backbone_support_weighting,
        "by_backbone_target_split": by_backbone_target_split,
        "by_backbone_support_bank_size": by_backbone_support_bank_size,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
