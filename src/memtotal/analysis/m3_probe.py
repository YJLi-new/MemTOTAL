from __future__ import annotations

import csv
import json
from math import ceil
from pathlib import Path

from memtotal.utils.io import write_json
from memtotal.utils.config import load_config
import yaml


def collect_stage_b_probe_rows(input_root: str | Path) -> list[dict[str, object]]:
    root = Path(input_root).resolve()
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("training_stage") != "stage_b":
            continue
        run_dir = metrics_path.parent
        run_info_path = run_dir / "run_info.json"
        run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
        row = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "backbone": run_info.get("backbone", metrics.get("backbone", "unknown")),
            "query_learning_mode": metrics.get("query_learning_mode"),
            "query_objective": metrics.get("query_objective"),
            "stage_b_trainable_target": metrics.get("stage_b_trainable_target"),
            "trainable_module": metrics.get("trainable_module"),
            "retrieval_negative_count": metrics.get("retrieval_negative_count"),
            "meta_episodes": metrics.get("meta_episodes"),
            "inner_steps": metrics.get("inner_steps"),
            "inner_learning_rate": metrics.get("inner_learning_rate"),
            "meta_learning_rate": metrics.get("meta_learning_rate"),
            "mean_adaptation_gain": metrics.get("mean_adaptation_gain"),
            "mean_zero_shot_query_loss": metrics.get("mean_zero_shot_query_loss"),
            "mean_adapted_query_loss": metrics.get("mean_adapted_query_loss"),
            "source_eval_task_score": metrics.get("source_eval_task_score"),
            "source_eval_metric_name": metrics.get("source_eval_metric_name"),
            "source_eval_query_loss": metrics.get("source_eval_query_loss"),
            "source_eval_query_accuracy": metrics.get("source_eval_query_accuracy"),
        }
        rows.append(row)
    return rows


def probe_run_matches_config(run_dir: str | Path, config_path: str | Path, seed: int) -> bool:
    run_path = Path(run_dir)
    snapshot_path = run_path / "config.snapshot.yaml"
    run_info_path = run_path / "run_info.json"
    if not snapshot_path.exists() or not run_info_path.exists():
        return False
    try:
        expected_config = load_config(config_path)
        actual_snapshot = yaml.safe_load(snapshot_path.read_text())
        run_info = json.loads(run_info_path.read_text())
    except Exception:
        return False
    return actual_snapshot == expected_config and int(run_info.get("seed", -1)) == int(seed)


def write_stage_b_probe_csv(output_path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "backbone",
        "query_learning_mode",
        "query_objective",
        "stage_b_trainable_target",
        "trainable_module",
        "retrieval_negative_count",
        "meta_episodes",
        "inner_steps",
        "inner_learning_rate",
        "meta_learning_rate",
        "mean_adaptation_gain",
        "mean_zero_shot_query_loss",
        "mean_adapted_query_loss",
        "source_eval_task_score",
        "source_eval_metric_name",
        "source_eval_query_loss",
        "source_eval_query_accuracy",
        "run_dir",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_stage_b_probe_svg(output_path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='560' height='120'>"
            "<text x='24' y='64' font-size='18' font-family='monospace'>No Stage B probe rows</text></svg>"
        )
        return

    values = [float(row.get("mean_adaptation_gain") or 0.0) for row in rows]
    max_abs = max(max(abs(value) for value in values), 1e-6)
    width = 860
    height = 120 + 54 * len(rows)
    label_width = 250
    center_x = 520
    half_bar = 180
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='34' font-size='20' font-family='monospace'>M3 Stage B probe summary</text>",
        "<text x='24' y='56' font-size='13' font-family='monospace'>mean_adaptation_gain centered at 0</text>",
        f"<line x1='{center_x}' y1='78' x2='{center_x}' y2='{height - 24}' stroke='#6b5f47' stroke-width='2' />",
    ]
    for index, row in enumerate(rows):
        top = 84 + index * 50
        value = float(row.get("mean_adaptation_gain") or 0.0)
        scaled = ceil((abs(value) / max_abs) * half_bar)
        bar_width = max(scaled, 2 if value != 0 else 0)
        if value >= 0:
            bar_x = center_x
            fill = "#2b8a3e"
        else:
            bar_x = center_x - bar_width
            fill = "#b5651d"
        label = f"{row['backbone']} {row['run_name']}"
        parts.append(
            f"<text x='24' y='{top + 18}' font-size='12' font-family='monospace'>{label[:label_width // 7]}</text>"
        )
        parts.append(
            f"<rect x='{center_x - half_bar}' y='{top}' width='{half_bar * 2}' height='24' fill='#efe6d0' rx='4' />"
        )
        if bar_width > 0:
            parts.append(
                f"<rect x='{bar_x}' y='{top}' width='{bar_width}' height='24' fill='{fill}' rx='4' />"
            )
        parts.append(
            f"<text x='{center_x + half_bar + 16}' y='{top + 18}' font-size='12' font-family='monospace'>{value:.6f}</text>"
        )
    parts.append("</svg>")
    destination.write_text("".join(parts))


def run_m3_stage_b_probe_summary(
    *,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    rows = collect_stage_b_probe_rows(input_root)
    if dry_run:
        rows = rows[: max(1, min(2, len(rows)))]

    summary_csv = output_dir / "probe_summary.csv"
    summary_svg = output_dir / "probe_summary.svg"
    write_stage_b_probe_csv(summary_csv, rows)
    write_stage_b_probe_svg(summary_svg, rows)

    best_by_backbone: dict[str, dict[str, object]] = {}
    for row in rows:
        backbone = str(row["backbone"])
        current = best_by_backbone.get(backbone)
        current_gain = float(current["mean_adaptation_gain"]) if current is not None else float("-inf")
        row_gain = float(row.get("mean_adaptation_gain") or 0.0)
        if current is None or row_gain > current_gain:
            best_by_backbone[backbone] = {
                "run_name": row["run_name"],
                "mean_adaptation_gain": row_gain,
                "source_eval_task_score": row.get("source_eval_task_score"),
                "meta_episodes": row.get("meta_episodes"),
                "meta_learning_rate": row.get("meta_learning_rate"),
            }

    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_b_probe_summary",
        "rows_collected": len(rows),
        "input_root": str(Path(input_root).resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "summary_plot": str(summary_svg.resolve()),
        "best_by_backbone": best_by_backbone,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
