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


def collect_stage_c_step_saturation_rows(input_root: str | Path) -> list[dict[str, object]]:
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
        zero_to_step0_gain = float(step0_row["task_score"]) - float(zero_row["task_score"])
        step0_to_final_gain = float(final_row["task_score"]) - float(step0_row["task_score"])
        total_gain = float(final_row["task_score"]) - float(zero_row["task_score"])
        step_contribution_ratio = step0_to_final_gain / total_gain if abs(total_gain) > 1.0e-9 else 0.0
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
                "zero_shot_task_score": float(zero_row["task_score"]),
                "step0_task_score": float(step0_row["task_score"]),
                "final_task_score": float(final_row["task_score"]),
                "zero_to_step0_task_gain": zero_to_step0_gain,
                "step0_to_final_task_gain": step0_to_final_gain,
                "total_task_gain": total_gain,
                "step_contribution_ratio": step_contribution_ratio,
                "zero_shot_task_proxy_score": float(zero_row["task_proxy_score"]),
                "step0_task_proxy_score": float(step0_row["task_proxy_score"]),
                "final_task_proxy_score": float(final_row["task_proxy_score"]),
                "zero_to_step0_proxy_gain": float(step0_row["task_proxy_score"]) - float(zero_row["task_proxy_score"]),
                "step0_to_final_proxy_gain": float(final_row["task_proxy_score"]) - float(step0_row["task_proxy_score"]),
                "shot_dominates": zero_to_step0_gain >= (0.8 * total_gain) if total_gain > 0.0 else False,
            }
        )
    audit_rows.sort(key=lambda row: (str(row["backbone"]), int(row["seed"])))
    return audit_rows


def write_stage_c_step_saturation_svg(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='560' height='120'>"
            "<text x='24' y='64' font-size='18' font-family='monospace'>No Stage C step audit rows</text></svg>"
        )
        return
    width = 960
    height = 132 + 54 * len(rows)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='32' font-size='20' font-family='monospace'>M3 Stage C step saturation audit</text>",
        "<text x='24' y='54' font-size='12' font-family='monospace'>Compare zero->step0 gain vs step0->final gain</text>",
    ]
    for index, row in enumerate(rows):
        top = 76 + index * 44
        parts.append(
            f"<text x='24' y='{top + 16}' font-size='12' font-family='monospace'>{row['backbone']} seed={row['seed']} total={float(row['total_task_gain']):.3f}</text>"
        )
        parts.append(
            f"<text x='420' y='{top + 16}' font-size='12' font-family='monospace'>0->s0={float(row['zero_to_step0_task_gain']):.3f} s0->final={float(row['step0_to_final_task_gain']):.3f} ratio={float(row['step_contribution_ratio']):.3f}</text>"
        )
    parts.append("</svg>")
    output_path.write_text("".join(parts))


def run_m3_stage_c_step_saturation_audit(
    *,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    rows = collect_stage_c_step_saturation_rows(input_root)
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
            "zero_shot_task_score",
            "step0_task_score",
            "final_task_score",
            "zero_to_step0_task_gain",
            "step0_to_final_task_gain",
            "total_task_gain",
            "step_contribution_ratio",
            "zero_shot_task_proxy_score",
            "step0_task_proxy_score",
            "final_task_proxy_score",
            "zero_to_step0_proxy_gain",
            "step0_to_final_proxy_gain",
            "shot_dominates",
            "run_dir",
        ],
    )
    write_stage_c_step_saturation_svg(audit_svg, rows)

    by_backbone: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        backbone_rows = [row for row in rows if str(row["backbone"]) == backbone]
        by_backbone[backbone] = {
            "seed_count": len(backbone_rows),
            "mean_zero_to_step0_task_gain": sum(float(row["zero_to_step0_task_gain"]) for row in backbone_rows)
            / len(backbone_rows),
            "mean_step0_to_final_task_gain": sum(float(row["step0_to_final_task_gain"]) for row in backbone_rows)
            / len(backbone_rows),
            "mean_total_task_gain": sum(float(row["total_task_gain"]) for row in backbone_rows)
            / len(backbone_rows),
            "mean_step_contribution_ratio": sum(float(row["step_contribution_ratio"]) for row in backbone_rows)
            / len(backbone_rows),
            "shot_dominates_rate": sum(1.0 if row["shot_dominates"] else 0.0 for row in backbone_rows)
            / len(backbone_rows),
            "target_split_policies": sorted({str(row["target_split_policy"]) for row in backbone_rows}),
        }

    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_step_saturation_audit",
        "rows_collected": len(rows),
        "input_root": str(Path(input_root).resolve()),
        "audit_csv": str(audit_csv.resolve()),
        "audit_svg": str(audit_svg.resolve()),
        "by_backbone": by_backbone,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
