from __future__ import annotations

import csv
from pathlib import Path

from memtotal.analysis.m3_stage_c_curve_summary import collect_stage_c_curve_rows
from memtotal.utils.io import write_json


def _write_csv(output_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_negative_seed_curve_rows(input_root: str | Path) -> list[dict[str, object]]:
    curve_rows = collect_stage_c_curve_rows(input_root)
    negative_runs: set[tuple[str, int]] = set()
    for row in curve_rows:
        if int(row["shot"]) == 0 and int(row["step"]) == 0 and float(row["task_margin"]) < 0.0:
            negative_runs.add((str(row["run_dir"]), int(row["seed"] or -1)))
    filtered = [
        row
        for row in curve_rows
        if (str(row["run_dir"]), int(row["seed"] or -1)) in negative_runs
    ]
    filtered.sort(key=lambda row: (str(row["backbone"]), int(row["seed"] or -1), int(row["shot"]), int(row["step"])))
    return filtered


def _aggregate_rows(rows: list[dict[str, object]], *, group_key: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["backbone"]), int(row[group_key])), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for (backbone, x_value), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append(
            {
                "backbone": backbone,
                group_key: x_value,
                "seed_count": len({int(row["seed"] or -1) for row in group_rows}),
                "row_count": len(group_rows),
                "mean_task_score": sum(float(row["task_score"]) for row in group_rows) / len(group_rows),
                "mean_task_margin": sum(float(row["task_margin"]) for row in group_rows) / len(group_rows),
                "mean_margin_gap_to_flip": sum(max(0.0, -float(row["task_margin"])) for row in group_rows)
                / len(group_rows),
                "mean_task_proxy_score": sum(float(row["task_proxy_score"]) for row in group_rows)
                / len(group_rows),
                "mean_objective_loss": sum(float(row["objective_loss"]) for row in group_rows)
                / len(group_rows),
            }
        )
    return summary_rows


def _write_svg(output_path: Path, rows: list[dict[str, object]], *, x_key: str, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='560' height='120'>"
            "<text x='24' y='64' font-size='18' font-family='monospace'>No negative-seed curve rows</text></svg>"
        )
        return
    width = 920
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
    y_values = [float(row["mean_margin_gap_to_flip"]) for row in rows]
    y_min = 0.0
    y_max = max(y_values) if y_values else 1.0
    if abs(y_max - y_min) < 1.0e-9:
        y_max = y_min + 0.1

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
        "<text x='24' y='50' font-size='12' font-family='monospace'>Mean gap-to-flip on zero-shot negative seeds</text>",
        "<line x1='72' y1='300' x2='892' y2='300' stroke='#6b5f47' stroke-width='2' />",
        "<line x1='72' y1='56' x2='72' y2='300' stroke='#6b5f47' stroke-width='2' />",
    ]
    for value in x_values:
        xpos = x_pos(value)
        parts.append(f"<text x='{xpos - 8}' y='326' font-size='12' font-family='monospace'>{value}</text>")
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        series = [row for row in rows if str(row["backbone"]) == backbone]
        series.sort(key=lambda row: int(row[x_key]))
        points = " ".join(
            f"{x_pos(int(row[x_key])):.1f},{y_pos(float(row['mean_margin_gap_to_flip'])):.1f}" for row in series
        )
        color = palette.get(backbone, "#4a6fa5")
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{points}' />")
        for row in series:
            xpos = x_pos(int(row[x_key]))
            ypos = y_pos(float(row["mean_margin_gap_to_flip"]))
            parts.append(f"<circle cx='{xpos:.1f}' cy='{ypos:.1f}' r='4' fill='{color}' />")
            parts.append(
                f"<text x='{xpos + 6:.1f}' y='{ypos - 6:.1f}' font-size='10' font-family='monospace'>{float(row['mean_margin_gap_to_flip']):.4f}</text>"
            )
    parts.append("</svg>")
    output_path.write_text("".join(parts))


def run_m3_stage_c_negative_seed_curve_audit(
    *,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    rows = collect_negative_seed_curve_rows(input_root)
    if dry_run:
        rows = rows[: max(1, min(8, len(rows)))]
    negative_curve_csv = output_dir / "negative_seed_curve_rows.csv"
    _write_csv(
        negative_curve_csv,
        rows,
        fieldnames=[
            "run_name",
            "backbone",
            "seed",
            "shot",
            "step",
            "task_score",
            "task_margin",
            "task_proxy_score",
            "objective_loss",
            "run_dir",
        ],
    )
    negative_max_shot_rows = [row for row in rows if int(row["shot"]) == max(int(item["shot"]) for item in rows)] if rows else []
    negative_step_rows = _aggregate_rows(negative_max_shot_rows, group_key="step") if negative_max_shot_rows else []
    negative_step_csv = output_dir / "negative_seed_step_curve.csv"
    _write_csv(
        negative_step_csv,
        negative_step_rows,
        fieldnames=[
            "backbone",
            "step",
            "seed_count",
            "row_count",
            "mean_task_score",
            "mean_task_margin",
            "mean_margin_gap_to_flip",
            "mean_task_proxy_score",
            "mean_objective_loss",
        ],
    )
    negative_step_svg = output_dir / "negative_seed_step_curve.svg"
    _write_svg(
        negative_step_svg,
        negative_step_rows,
        x_key="step",
        title="M3 Stage C negative-seed step curve",
    )

    negative_step0_rows = [row for row in rows if int(row["step"]) == 0]
    negative_shot_rows = _aggregate_rows(negative_step0_rows, group_key="shot") if negative_step0_rows else []
    negative_shot_csv = output_dir / "negative_seed_shot_curve.csv"
    _write_csv(
        negative_shot_csv,
        negative_shot_rows,
        fieldnames=[
            "backbone",
            "shot",
            "seed_count",
            "row_count",
            "mean_task_score",
            "mean_task_margin",
            "mean_margin_gap_to_flip",
            "mean_task_proxy_score",
            "mean_objective_loss",
        ],
    )
    negative_shot_svg = output_dir / "negative_seed_shot_curve.svg"
    _write_svg(
        negative_shot_svg,
        negative_shot_rows,
        x_key="shot",
        title="M3 Stage C negative-seed shot curve",
    )

    by_backbone: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        backbone_rows = [row for row in rows if str(row["backbone"]) == backbone]
        by_backbone[backbone] = {
            "negative_seed_count": len({int(row["seed"] or -1) for row in backbone_rows}),
            "max_shot": max(int(row["shot"]) for row in backbone_rows),
            "max_step": max(int(row["step"]) for row in backbone_rows),
            "mean_zero_shot_gap_to_flip": sum(
                max(0.0, -float(row["task_margin"]))
                for row in backbone_rows
                if int(row["shot"]) == 0 and int(row["step"]) == 0
            )
            / max(1, sum(1 for row in backbone_rows if int(row["shot"]) == 0 and int(row["step"]) == 0)),
            "mean_max_shot_step0_gap_to_flip": sum(
                max(0.0, -float(row["task_margin"]))
                for row in backbone_rows
                if int(row["shot"]) == max(int(item["shot"]) for item in backbone_rows) and int(row["step"]) == 0
            )
            / max(
                1,
                sum(
                    1
                    for row in backbone_rows
                    if int(row["shot"]) == max(int(item["shot"]) for item in backbone_rows) and int(row["step"]) == 0
                ),
            ),
            "mean_max_shot_final_gap_to_flip": sum(
                max(0.0, -float(row["task_margin"]))
                for row in backbone_rows
                if int(row["shot"]) == max(int(item["shot"]) for item in backbone_rows)
                and int(row["step"]) == max(int(item["step"]) for item in backbone_rows if int(item["shot"]) == int(row["shot"]))
            )
            / max(
                1,
                sum(
                    1
                    for row in backbone_rows
                    if int(row["shot"]) == max(int(item["shot"]) for item in backbone_rows)
                    and int(row["step"]) == max(int(item["step"]) for item in backbone_rows if int(item["shot"]) == int(row["shot"]))
                ),
            ),
        }

    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_negative_seed_curve_audit",
        "rows_collected": len(rows),
        "input_root": str(Path(input_root).resolve()),
        "negative_seed_curve_csv": str(negative_curve_csv.resolve()),
        "negative_seed_step_curve_csv": str(negative_step_csv.resolve()),
        "negative_seed_step_curve_svg": str(negative_step_svg.resolve()),
        "negative_seed_shot_curve_csv": str(negative_shot_csv.resolve()),
        "negative_seed_shot_curve_svg": str(negative_shot_svg.resolve()),
        "by_backbone": by_backbone,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
