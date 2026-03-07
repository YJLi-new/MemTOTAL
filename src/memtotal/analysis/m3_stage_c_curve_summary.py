from __future__ import annotations

import csv
import json
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


def collect_stage_c_curve_rows(input_root: str | Path) -> list[dict[str, object]]:
    root = Path(input_root).resolve()
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = _read_json(metrics_path)
        if metrics.get("training_stage") != "stage_c":
            continue
        run_dir = metrics_path.parent
        curve_path = run_dir / "adapt_curve.csv"
        if not curve_path.exists():
            continue
        backbone = _load_backbone(run_dir, metrics)
        seed = _load_seed(run_dir)
        with curve_path.open() as handle:
            reader = csv.DictReader(handle)
            curve_rows = list(reader)
        if not curve_rows:
            continue
        zero_shot_score = float(metrics.get("zero_shot_task_score") or curve_rows[0].get("task_score") or 0.0)
        zero_shot_proxy = float(
            metrics.get("zero_shot_task_proxy_score") or curve_rows[0].get("task_proxy_score") or 0.0
        )
        for row in curve_rows:
            task_score = float(row.get("task_score") or 0.0)
            task_proxy_score = float(row.get("task_proxy_score") or 0.0)
            rows.append(
                {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "backbone": backbone,
                    "seed": seed,
                    "query_learning_mode": row.get("query_learning_mode"),
                    "query_objective": row.get("query_objective"),
                    "adaptation_target": row.get("adaptation_target"),
                    "trainable_module": row.get("trainable_module"),
                    "trainable_parameter_count": int(float(row.get("trainable_parameter_count") or 0.0)),
                    "shot": int(float(row.get("shot") or 0.0)),
                    "step": int(float(row.get("step") or 0.0)),
                    "target_eval_repeats": int(float(row.get("target_eval_repeats") or 1.0)),
                    "target_episode_repeats": int(float(row.get("target_episode_repeats") or 1.0)),
                    "target_episode_policy": str(row.get("target_episode_policy") or "independent"),
                    "target_support_weighting": str(row.get("target_support_weighting") or "uniform"),
                    "target_split_policy": str(row.get("target_split_policy") or "random"),
                    "evaluated_target_episodes": int(float(row.get("evaluated_target_episodes") or 0.0)),
                    "evaluated_query_examples": int(float(row.get("evaluated_query_examples") or 0.0)),
                    "query_candidate_pool_size": int(float(row.get("query_candidate_pool_size") or 0.0)),
                    "support_candidate_pool_size": int(float(row.get("support_candidate_pool_size") or 0.0)),
                    "objective_loss": float(row.get("objective_loss") or 0.0),
                    "objective_accuracy": float(row.get("objective_accuracy") or 0.0),
                    "task_score": task_score,
                    "task_gain": task_score - zero_shot_score,
                    "task_metric_name": str(row.get("task_metric_name") or "none"),
                    "task_proxy_score": task_proxy_score,
                    "task_proxy_gain": task_proxy_score - zero_shot_proxy,
                    "task_proxy_name": str(row.get("task_proxy_name") or "none"),
                    "task_margin": float(row.get("task_margin") or 0.0),
                    "preceding_support_grad_norm": float(row.get("preceding_support_grad_norm") or 0.0),
                    "preceding_support_update_max_abs": float(
                        row.get("preceding_support_update_max_abs") or 0.0
                    ),
                    "preceding_support_update_l2": float(row.get("preceding_support_update_l2") or 0.0),
                    "query_loss": float(row.get("query_loss") or 0.0),
                    "query_accuracy": float(row.get("query_accuracy") or 0.0),
                }
            )
    rows.sort(key=lambda row: (str(row["backbone"]), int(row["seed"] or -1), int(row["shot"]), int(row["step"])))
    return rows


def _write_csv(output_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_rows(
    rows: list[dict[str, object]],
    *,
    group_key: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["backbone"]), int(row[group_key])), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for (backbone, x_value), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append(
            {
                "backbone": backbone,
                group_key: x_value,
                "seed_count": len(group_rows),
                "mean_task_score": _mean([float(row["task_score"]) for row in group_rows]),
                "mean_task_gain": _mean([float(row["task_gain"]) for row in group_rows]),
                "mean_task_proxy_score": _mean([float(row["task_proxy_score"]) for row in group_rows]),
                "mean_task_proxy_gain": _mean([float(row["task_proxy_gain"]) for row in group_rows]),
                "mean_objective_loss": _mean([float(row["objective_loss"]) for row in group_rows]),
                "mean_query_loss": _mean([float(row["query_loss"]) for row in group_rows]),
                "positive_gain_rate": _mean(
                    [1.0 if float(row["task_gain"]) > 0.0 else 0.0 for row in group_rows]
                ),
                "task_metric_name": str(group_rows[0]["task_metric_name"]),
                "task_proxy_name": str(group_rows[0]["task_proxy_name"]),
                "target_episode_policy": str(group_rows[0]["target_episode_policy"]),
                "target_support_weighting": str(group_rows[0]["target_support_weighting"]),
                "target_split_policy": str(group_rows[0]["target_split_policy"]),
            }
        )
    return summary_rows


def _select_final_shot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_rows: dict[tuple[str, int, int], dict[str, object]] = {}
    for row in rows:
        key = (str(row["run_dir"]), int(row["seed"] or -1), int(row["shot"]))
        current = best_rows.get(key)
        if current is None or int(row["step"]) > int(current["step"]):
            best_rows[key] = row
    return list(best_rows.values())


def _select_max_shot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    max_shot_by_run: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["run_dir"]), int(row["seed"] or -1))
        max_shot_by_run[key] = max(max_shot_by_run.get(key, 0), int(row["shot"]))
    return [
        row
        for row in rows
        if int(row["shot"]) == max_shot_by_run[(str(row["run_dir"]), int(row["seed"] or -1))]
    ]


def _write_curve_svg(
    output_path: Path,
    rows: list[dict[str, object]],
    *,
    x_key: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='560' height='120'>"
            "<text x='24' y='64' font-size='18' font-family='monospace'>No Stage C curve rows</text></svg>"
        )
        return
    width = 900
    height = 360
    margin_left = 72
    margin_right = 28
    margin_top = 56
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    palette = {
        "Qwen2.5-1.5B-Instruct": "#2f5aa8",
        "Qwen3-8B": "#b5651d",
    }
    x_values = sorted({int(row[x_key]) for row in rows})
    y_values = [float(row["mean_task_score"]) for row in rows]
    y_min = min(y_values)
    y_max = max(y_values)
    if abs(y_max - y_min) < 1.0e-6:
        y_min -= 0.1
        y_max += 0.1

    def x_pos(value: int) -> float:
        if len(x_values) == 1:
            return margin_left + plot_width / 2
        index = x_values.index(value)
        return margin_left + (plot_width * index / (len(x_values) - 1))

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1.0 - ((value - y_min) / (y_max - y_min)))

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        f"<text x='24' y='32' font-size='20' font-family='monospace'>{title}</text>",
        "<line x1='72' y1='300' x2='872' y2='300' stroke='#6b5f47' stroke-width='2' />",
        "<line x1='72' y1='56' x2='72' y2='300' stroke='#6b5f47' stroke-width='2' />",
    ]
    for value in x_values:
        xpos = x_pos(value)
        parts.append(
            f"<text x='{xpos - 8}' y='326' font-size='12' font-family='monospace'>{value}</text>"
        )
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        series = [row for row in rows if str(row["backbone"]) == backbone]
        series.sort(key=lambda row: int(row[x_key]))
        points = " ".join(
            f"{x_pos(int(row[x_key])):.1f},{y_pos(float(row['mean_task_score'])):.1f}" for row in series
        )
        parts.append(
            f"<polyline fill='none' stroke='{palette.get(backbone, '#4a6fa5')}' stroke-width='3' points='{points}' />"
        )
        for row in series:
            xpos = x_pos(int(row[x_key]))
            ypos = y_pos(float(row["mean_task_score"]))
            parts.append(
                f"<circle cx='{xpos:.1f}' cy='{ypos:.1f}' r='4' fill='{palette.get(backbone, '#4a6fa5')}' />"
            )
            parts.append(
                f"<text x='{xpos + 6:.1f}' y='{ypos - 6:.1f}' font-size='10' font-family='monospace'>{float(row['mean_task_score']):.3f}</text>"
            )
    legend_y = 42
    for index, backbone in enumerate(sorted({str(row["backbone"]) for row in rows})):
        x = 540 + 160 * index
        color = palette.get(backbone, "#4a6fa5")
        parts.append(f"<rect x='{x}' y='{legend_y - 10}' width='18' height='8' fill='{color}' />")
        parts.append(f"<text x='{x + 24}' y='{legend_y - 2}' font-size='11' font-family='monospace'>{backbone}</text>")
    parts.append("</svg>")
    output_path.write_text("".join(parts))


def run_m3_stage_c_curve_summary(
    *,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    rows = collect_stage_c_curve_rows(input_root)
    if dry_run:
        rows = rows[: max(1, min(8, len(rows)))]

    curve_rows_csv = output_dir / "curve_rows.csv"
    _write_csv(
        curve_rows_csv,
        rows,
        fieldnames=[
            "run_name",
            "backbone",
            "seed",
            "query_learning_mode",
            "query_objective",
            "adaptation_target",
            "trainable_module",
            "trainable_parameter_count",
            "shot",
            "step",
            "target_eval_repeats",
            "target_episode_repeats",
            "target_episode_policy",
            "target_support_weighting",
            "target_split_policy",
            "evaluated_target_episodes",
            "evaluated_query_examples",
            "query_candidate_pool_size",
            "support_candidate_pool_size",
            "objective_loss",
            "objective_accuracy",
            "task_score",
            "task_gain",
            "task_metric_name",
            "task_proxy_score",
            "task_proxy_gain",
            "task_proxy_name",
            "task_margin",
            "preceding_support_grad_norm",
            "preceding_support_update_max_abs",
            "preceding_support_update_l2",
            "query_loss",
            "query_accuracy",
            "run_dir",
        ],
    )

    shot_rows = _aggregate_rows(_select_final_shot_rows(rows), group_key="shot")
    step_rows = _aggregate_rows(_select_max_shot_rows(rows), group_key="step")
    shot_curve_csv = output_dir / "shot_curve.csv"
    step_curve_csv = output_dir / "step_curve.csv"
    common_fieldnames = [
        "backbone",
        "seed_count",
        "mean_task_score",
        "mean_task_gain",
        "mean_task_proxy_score",
        "mean_task_proxy_gain",
        "mean_objective_loss",
        "mean_query_loss",
        "positive_gain_rate",
        "task_metric_name",
        "task_proxy_name",
        "target_episode_policy",
        "target_support_weighting",
        "target_split_policy",
    ]
    _write_csv(shot_curve_csv, shot_rows, fieldnames=["backbone", "shot", *common_fieldnames[1:]])
    _write_csv(step_curve_csv, step_rows, fieldnames=["backbone", "step", *common_fieldnames[1:]])

    shot_curve_svg = output_dir / "shot_curve.svg"
    step_curve_svg = output_dir / "step_curve.svg"
    _write_curve_svg(shot_curve_svg, shot_rows, x_key="shot", title="M3 Stage C shot curve")
    _write_curve_svg(step_curve_svg, step_rows, x_key="step", title="M3 Stage C step curve")

    best_final_by_backbone: dict[str, dict[str, object]] = {}
    final_rows = _select_final_shot_rows(rows)
    for backbone in sorted({str(row["backbone"]) for row in final_rows}):
        backbone_rows = [row for row in final_rows if str(row["backbone"]) == backbone]
        best_row = max(backbone_rows, key=lambda row: (float(row["task_score"]), float(row["task_proxy_score"])))
        best_final_by_backbone[backbone] = {
            "shot": int(best_row["shot"]),
            "step": int(best_row["step"]),
            "task_score": float(best_row["task_score"]),
            "task_gain": float(best_row["task_gain"]),
            "task_proxy_score": float(best_row["task_proxy_score"]),
            "target_split_policy": str(best_row["target_split_policy"]),
        }

    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_curve_summary",
        "rows_collected": len(rows),
        "shot_curve_rows": len(shot_rows),
        "step_curve_rows": len(step_rows),
        "input_root": str(Path(input_root).resolve()),
        "curve_rows_csv": str(curve_rows_csv.resolve()),
        "shot_curve_csv": str(shot_curve_csv.resolve()),
        "step_curve_csv": str(step_curve_csv.resolve()),
        "shot_curve_svg": str(shot_curve_svg.resolve()),
        "step_curve_svg": str(step_curve_svg.resolve()),
        "best_final_by_backbone": best_final_by_backbone,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
