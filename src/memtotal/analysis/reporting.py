from __future__ import annotations

import csv
import json
from math import ceil
from pathlib import Path


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_primary_metric(row: dict[str, object]) -> tuple[str, float]:
    metric_preferences = [
        ("memoryagent_score", _coerce_float(row.get("memoryagent_score"))),
        ("mean_score", _coerce_float(row.get("mean_score"))),
        ("accuracy", _coerce_float(row.get("accuracy"))),
        ("compute_reward", _coerce_float(row.get("compute_reward"))),
        ("checks_pass_rate", _coerce_float(row.get("checks_pass_rate"))),
        ("best_adapt_query_accuracy", _coerce_float(row.get("best_adapt_query_accuracy"))),
        ("zero_shot_query_accuracy", _coerce_float(row.get("zero_shot_query_accuracy"))),
        ("mean_adaptation_gain", _coerce_float(row.get("mean_adaptation_gain"))),
        ("source_eval_query_accuracy", _coerce_float(row.get("source_eval_query_accuracy"))),
        ("mean_similarity", _coerce_float(row.get("mean_similarity"))),
    ]
    for metric_name, metric_value in metric_preferences:
        if metric_value is not None:
            return metric_name, metric_value

    best_adapt_query_loss = _coerce_float(row.get("best_adapt_query_loss"))
    if best_adapt_query_loss is not None:
        return "inv_best_adapt_query_loss", 1.0 / (1.0 + max(best_adapt_query_loss, 0.0))

    zero_shot_query_loss = _coerce_float(row.get("zero_shot_query_loss"))
    if zero_shot_query_loss is not None:
        return "inv_zero_shot_query_loss", 1.0 / (1.0 + max(zero_shot_query_loss, 0.0))

    source_eval_query_loss = _coerce_float(row.get("source_eval_query_loss"))
    if source_eval_query_loss is not None:
        return "inv_source_eval_query_loss", 1.0 / (1.0 + max(source_eval_query_loss, 0.0))

    mean_loss = _coerce_float(row.get("mean_loss"))
    if mean_loss is not None:
        return "inv_mean_loss", 1.0 / (1.0 + max(mean_loss, 0.0))

    return "none", 0.0


def collect_metrics(input_root: str | Path) -> list[dict[str, object]]:
    root = Path(input_root).resolve()
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        run_dir = metrics_path.parent
        run_info_path = run_dir / "run_info.json"
        run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
        row = {
            "run_dir": str(run_dir),
            "backbone": run_info.get("backbone", metrics.get("backbone", "unknown")),
            "task_name": run_info.get("task_name", "unknown"),
            "benchmark_id": run_info.get("benchmark_id", metrics.get("benchmark_id")),
            "task_domain": run_info.get("task_domain", metrics.get("task_domain")),
            "smoke_subset": run_info.get("smoke_subset", metrics.get("smoke_subset")),
            "mode": metrics.get("mode", "unknown"),
        }
        for key, value in metrics.items():
            row[key] = value
        capability_scores = metrics.get("capability_scores")
        capability_metric_names = metrics.get("capability_metric_names", {})
        if isinstance(capability_scores, dict):
            for capability, score in capability_scores.items():
                row[f"capability_{capability}_score"] = score
                if isinstance(capability_metric_names, dict) and capability in capability_metric_names:
                    row[f"capability_{capability}_metric"] = capability_metric_names[capability]
        primary_metric, primary_score = resolve_primary_metric(row)
        row["primary_metric"] = primary_metric
        row["primary_score"] = primary_score
        rows.append(row)
    return rows


def write_summary_csv(output_path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_sanity_plot(output_path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='480' height='120'>"
            "<text x='24' y='64' font-size='18'>No metrics collected</text></svg>"
        )
        return

    score_rows = []
    for row in rows:
        score_value = _coerce_float(row.get("primary_score"))
        if score_value is None:
            score_value = 0.0
        score_rows.append(
            (
                f"{row['mode']}:{Path(str(row['run_dir'])).name} [{row.get('primary_metric', 'none')}]",
                score_value,
            )
        )

    max_value = max(score for _, score in score_rows) or 1.0
    chart_width = 640
    chart_height = 120 + 60 * len(score_rows)
    bar_left = 180
    bar_width = 380
    bar_height = 28
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{chart_width}' height='{chart_height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='36' font-size='20' font-family='monospace'>MemTOTAL sanity plot</text>",
    ]
    for index, (label, value) in enumerate(score_rows):
        top = 70 + index * 54
        scaled_width = 0 if max_value == 0 else ceil((value / max_value) * bar_width)
        parts.append(
            f"<text x='24' y='{top + 20}' font-size='14' font-family='monospace'>{label}</text>"
        )
        parts.append(
            f"<rect x='{bar_left}' y='{top}' width='{bar_width}' height='{bar_height}' fill='#efe6d0' rx='4' />"
        )
        parts.append(
            f"<rect x='{bar_left}' y='{top}' width='{scaled_width}' height='{bar_height}' fill='#4a6fa5' rx='4' />"
        )
        parts.append(
            f"<text x='{bar_left + bar_width + 12}' y='{top + 20}' font-size='14' font-family='monospace'>{value:.4f}</text>"
        )
    parts.append("</svg>")
    destination.write_text("".join(parts))
