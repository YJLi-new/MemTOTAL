from __future__ import annotations

import csv
from pathlib import Path

from memtotal.analysis.m3_stage_c_curve_summary import collect_stage_c_curve_rows
from memtotal.utils.io import write_json


def _write_csv(output_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_margin_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "seed_count": len(rows),
        "mean_zero_shot_task_margin": sum(float(row["zero_shot_task_margin"]) for row in rows) / len(rows),
        "mean_final_task_margin": sum(float(row["final_task_margin"]) for row in rows) / len(rows),
        "mean_total_margin_gain": sum(float(row["total_margin_gain"]) for row in rows) / len(rows),
        "mean_zero_shot_margin_gap_to_flip": sum(
            float(row["zero_shot_margin_gap_to_flip"]) for row in rows
        )
        / len(rows),
        "mean_final_margin_gap_to_flip": sum(
            float(row["final_margin_gap_to_flip"]) for row in rows
        )
        / len(rows),
        "mean_margin_gap_closed": sum(float(row["margin_gap_closed"]) for row in rows) / len(rows),
        "margin_improves_rate": sum(1.0 if row["margin_improves"] else 0.0 for row in rows) / len(rows),
        "cross_zero_margin_rate": sum(1.0 if row["crosses_zero_margin"] else 0.0 for row in rows)
        / len(rows),
        "target_split_policies": sorted({str(row["target_split_policy"]) for row in rows}),
    }


def collect_stage_c_margin_rows(input_root: str | Path) -> list[dict[str, object]]:
    curve_rows = collect_stage_c_curve_rows(input_root)
    runs: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in curve_rows:
        runs.setdefault((str(row["run_dir"]), int(row["seed"] or -1)), []).append(row)

    audit_rows: list[dict[str, object]] = []
    for (_, _), rows in runs.items():
        rows.sort(key=lambda row: (int(row["shot"]), int(row["step"])))
        zero_row = next(row for row in rows if int(row["shot"]) == 0 and int(row["step"]) == 0)
        max_shot = max(int(row["shot"]) for row in rows)
        shot_rows = [row for row in rows if int(row["shot"]) == max_shot]
        step0_row = next(row for row in shot_rows if int(row["step"]) == 0)
        final_row = max(shot_rows, key=lambda row: int(row["step"]))
        zero_margin = float(zero_row["task_margin"])
        step0_margin = float(step0_row["task_margin"])
        final_margin = float(final_row["task_margin"])
        zero_gap = max(0.0, -zero_margin)
        final_gap = max(0.0, -final_margin)
        audit_rows.append(
            {
                "run_name": str(final_row["run_name"]),
                "run_dir": str(final_row["run_dir"]),
                "backbone": str(final_row["backbone"]),
                "seed": int(final_row["seed"] or -1),
                "target_split_policy": str(final_row["target_split_policy"]),
                "target_episode_policy": str(final_row["target_episode_policy"]),
                "target_support_weighting": str(final_row["target_support_weighting"]),
                "max_shot": max_shot,
                "final_step": int(final_row["step"]),
                "zero_shot_task_margin": zero_margin,
                "step0_task_margin": step0_margin,
                "final_task_margin": final_margin,
                "zero_to_step0_margin_gain": step0_margin - zero_margin,
                "step0_to_final_margin_gain": final_margin - step0_margin,
                "total_margin_gain": final_margin - zero_margin,
                "zero_shot_margin_gap_to_flip": zero_gap,
                "final_margin_gap_to_flip": final_gap,
                "margin_gap_closed": zero_gap - final_gap,
                "margin_improves": final_margin > zero_margin,
                "crosses_zero_margin": zero_margin < 0.0 <= final_margin,
            }
        )
    audit_rows.sort(key=lambda row: (str(row["backbone"]), int(row["seed"])))
    return audit_rows


def write_stage_c_margin_svg(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='560' height='120'>"
            "<text x='24' y='64' font-size='18' font-family='monospace'>No Stage C margin rows</text></svg>"
        )
        return
    width = 980
    height = 132 + 54 * len(rows)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='32' font-size='20' font-family='monospace'>M3 Stage C margin audit</text>",
        "<text x='24' y='54' font-size='12' font-family='monospace'>Track mean gold-vs-best-other margin and remaining gap to rank flip</text>",
    ]
    for index, row in enumerate(rows):
        top = 76 + index * 44
        parts.append(
            f"<text x='24' y='{top + 16}' font-size='12' font-family='monospace'>{row['backbone']} seed={row['seed']} margin={float(row['zero_shot_task_margin']):.4f}->{float(row['final_task_margin']):.4f}</text>"
        )
        parts.append(
            f"<text x='470' y='{top + 16}' font-size='12' font-family='monospace'>gap={float(row['zero_shot_margin_gap_to_flip']):.4f}->{float(row['final_margin_gap_to_flip']):.4f} closed={float(row['margin_gap_closed']):.4f}</text>"
        )
    parts.append("</svg>")
    output_path.write_text("".join(parts))


def run_m3_stage_c_margin_audit(
    *,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    rows = collect_stage_c_margin_rows(input_root)
    if dry_run:
        rows = rows[: max(1, min(4, len(rows)))]
    audit_csv = output_dir / "audit_rows.csv"
    audit_svg = output_dir / "audit_summary.svg"
    _write_csv(
        audit_csv,
        rows,
        fieldnames=[
            "run_name",
            "backbone",
            "seed",
            "target_split_policy",
            "target_episode_policy",
            "target_support_weighting",
            "max_shot",
            "final_step",
            "zero_shot_task_margin",
            "step0_task_margin",
            "final_task_margin",
            "zero_to_step0_margin_gain",
            "step0_to_final_margin_gain",
            "total_margin_gain",
            "zero_shot_margin_gap_to_flip",
            "final_margin_gap_to_flip",
            "margin_gap_closed",
            "margin_improves",
            "crosses_zero_margin",
            "run_dir",
        ],
    )
    write_stage_c_margin_svg(audit_svg, rows)

    by_backbone: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        backbone_rows = [row for row in rows if str(row["backbone"]) == backbone]
        by_backbone[backbone] = _summarize_margin_rows(backbone_rows)

    by_backbone_negative_only: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        backbone_rows = [row for row in rows if str(row["backbone"]) == backbone]
        negative_rows = [row for row in backbone_rows if float(row["zero_shot_task_margin"]) < 0.0]
        if not negative_rows:
            continue
        by_backbone_negative_only[backbone] = _summarize_margin_rows(negative_rows)

    by_backbone_non_negative_only: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        backbone_rows = [row for row in rows if str(row["backbone"]) == backbone]
        non_negative_rows = [row for row in backbone_rows if float(row["zero_shot_task_margin"]) >= 0.0]
        if not non_negative_rows:
            continue
        by_backbone_non_negative_only[backbone] = _summarize_margin_rows(non_negative_rows)

    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_margin_audit",
        "rows_collected": len(rows),
        "input_root": str(Path(input_root).resolve()),
        "audit_csv": str(audit_csv.resolve()),
        "audit_svg": str(audit_svg.resolve()),
        "by_backbone": by_backbone,
        "by_backbone_negative_only": by_backbone_negative_only,
        "by_backbone_non_negative_only": by_backbone_non_negative_only,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
