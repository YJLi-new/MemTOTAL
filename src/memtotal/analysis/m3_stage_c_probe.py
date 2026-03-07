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


def _stage_c_probe_row_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row.get("best_adapt_task_score") or 0.0),
        float(row.get("best_adapt_task_proxy_score") or row.get("best_adapt_task_score") or 0.0),
        float(row.get("task_gain") or 0.0),
        -float(row.get("best_adapt_query_loss") or 0.0),
        -float(row.get("trainable_parameter_count") or 0.0),
    )


def collect_stage_c_probe_rows(
    input_root: str | Path,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    root = Path(input_root).resolve()
    rows: list[dict[str, object]] = []
    gradient_audits: dict[tuple[str, str], dict[str, object]] = {}

    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = _read_json(metrics_path)
        run_dir = metrics_path.parent
        backbone = _load_backbone(run_dir, metrics)

        if metrics.get("analysis_mode") == "m3_stage_c_gradient_audit":
            adaptation_target = str(metrics.get("adaptation_target", "unknown"))
            gradient_audits[(backbone, adaptation_target)] = {
                "gradient_run_dir": str(run_dir),
                "queries_grad_norm": metrics.get("queries_grad_norm"),
                "reader_non_query_grad_norm": metrics.get("reader_non_query_grad_norm"),
                "fuser_grad_norm": metrics.get("fuser_grad_norm"),
                "writer_grad_norm": metrics.get("writer_grad_norm"),
                "query_to_fuser_grad_ratio": metrics.get("query_to_fuser_grad_ratio"),
                "query_to_writer_grad_ratio": metrics.get("query_to_writer_grad_ratio"),
            }
            continue

        if metrics.get("training_stage") != "stage_c":
            continue

        zero_shot_task_score = float(metrics.get("zero_shot_task_score") or 0.0)
        best_adapt_task_score = float(metrics.get("best_adapt_task_score") or 0.0)
        adaptation_target = str(metrics.get("adaptation_target", "unknown"))
        row = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "backbone": backbone,
            "seed": _load_seed(run_dir),
            "query_learning_mode": metrics.get("query_learning_mode"),
            "query_objective": metrics.get("query_objective"),
            "adaptation_target": adaptation_target,
            "trainable_module": metrics.get("trainable_module"),
            "trainable_parameter_count": metrics.get("trainable_parameter_count"),
            "adapt_learning_rate": metrics.get("adapt_learning_rate"),
            "adapt_steps": metrics.get("adapt_steps"),
            "target_eval_repeats": metrics.get("target_eval_repeats"),
            "target_episode_repeats": metrics.get("target_episode_repeats"),
            "target_episode_policy": metrics.get("target_episode_policy"),
            "target_support_weighting": metrics.get("target_support_weighting"),
            "zero_shot_task_score": zero_shot_task_score,
            "best_adapt_task_score": best_adapt_task_score,
            "task_gain": best_adapt_task_score - zero_shot_task_score,
            "task_metric_name": metrics.get("task_metric_name"),
            "zero_shot_task_proxy_score": metrics.get("zero_shot_task_proxy_score"),
            "best_adapt_task_proxy_score": metrics.get("best_adapt_task_proxy_score"),
            "task_proxy_name": metrics.get("task_proxy_name"),
            "best_adapt_task_margin": metrics.get("best_adapt_task_margin"),
            "zero_shot_query_loss": metrics.get("zero_shot_query_loss"),
            "best_adapt_query_loss": metrics.get("best_adapt_query_loss"),
            "best_adapt_shot": metrics.get("best_adapt_shot"),
            "best_adapt_step": metrics.get("best_adapt_step"),
            "mean_support_grad_norm": metrics.get("mean_support_grad_norm"),
            "max_support_update_max_abs": metrics.get("max_support_update_max_abs"),
            "adaptation_effective": metrics.get("adaptation_effective"),
        }
        row.update(
            gradient_audits.get(
                (backbone, adaptation_target),
                {
                    "gradient_run_dir": None,
                    "queries_grad_norm": None,
                    "reader_non_query_grad_norm": None,
                    "fuser_grad_norm": None,
                    "writer_grad_norm": None,
                    "query_to_fuser_grad_ratio": None,
                    "query_to_writer_grad_ratio": None,
                },
            )
        )
        rows.append(row)

    for row in rows:
        audit = gradient_audits.get((str(row["backbone"]), str(row["adaptation_target"])))
        if audit is not None:
            row.update(audit)
    return rows, gradient_audits


def write_stage_c_probe_csv(output_path: str | Path, rows: list[dict[str, object]]) -> None:
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
        "zero_shot_task_score",
        "best_adapt_task_score",
        "task_gain",
        "task_metric_name",
        "zero_shot_task_proxy_score",
        "best_adapt_task_proxy_score",
        "task_proxy_name",
        "best_adapt_task_margin",
        "zero_shot_query_loss",
        "best_adapt_query_loss",
        "best_adapt_shot",
        "best_adapt_step",
        "mean_support_grad_norm",
        "max_support_update_max_abs",
        "adaptation_effective",
        "queries_grad_norm",
        "reader_non_query_grad_norm",
        "fuser_grad_norm",
        "writer_grad_norm",
        "query_to_fuser_grad_ratio",
        "query_to_writer_grad_ratio",
        "gradient_run_dir",
        "run_dir",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_stage_c_probe_svg(output_path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='560' height='120'>"
            "<text x='24' y='64' font-size='18' font-family='monospace'>No Stage C probe rows</text></svg>"
        )
        return

    values = [float(row.get("task_gain") or 0.0) for row in rows]
    max_abs = max(max(abs(value) for value in values), 1e-6)
    width = 980
    height = 132 + 62 * len(rows)
    center_x = 560
    half_bar = 180
    palette = {
        "q_only": "#2f5aa8",
        "w_only": "#b5651d",
        "w_plus_q": "#2b8a3e",
    }
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='34' font-size='20' font-family='monospace'>M3 Stage C target probe</text>",
        "<text x='24' y='56' font-size='13' font-family='monospace'>task_gain = best_adapt_task_score - zero_shot_task_score</text>",
        f"<line x1='{center_x}' y1='82' x2='{center_x}' y2='{height - 24}' stroke='#6b5f47' stroke-width='2' />",
    ]
    for index, row in enumerate(rows):
        top = 92 + index * 54
        value = float(row.get("task_gain") or 0.0)
        scaled = ceil((abs(value) / max_abs) * half_bar)
        bar_width = max(scaled, 2 if value != 0 else 0)
        if value >= 0:
            bar_x = center_x
        else:
            bar_x = center_x - bar_width
        adaptation_target = str(row["adaptation_target"])
        parts.append(
            f"<text x='24' y='{top + 16}' font-size='12' font-family='monospace'>{row['backbone']} {adaptation_target}</text>"
        )
        parts.append(
            f"<rect x='{center_x - half_bar}' y='{top}' width='{half_bar * 2}' height='24' fill='#efe6d0' rx='4' />"
        )
        if bar_width > 0:
            parts.append(
                f"<rect x='{bar_x}' y='{top}' width='{bar_width}' height='24' fill='{palette.get(adaptation_target, '#4a6fa5')}' rx='4' />"
            )
        effective = "yes" if row.get("adaptation_effective") else "no"
        score_label = (
            f"{float(row.get('zero_shot_task_score') or 0.0):.3f}->{float(row.get('best_adapt_task_score') or 0.0):.3f}"
        )
        proxy_label = (
            f"{float(row.get('zero_shot_task_proxy_score') or 0.0):.3f}->{float(row.get('best_adapt_task_proxy_score') or 0.0):.3f}"
        )
        parts.append(
            f"<text x='{center_x + half_bar + 16}' y='{top + 16}' font-size='12' font-family='monospace'>{value:.3f} eff={effective} score={score_label}</text>"
        )
        parts.append(
            f"<text x='{center_x + half_bar + 16}' y='{top + 30}' font-size='11' font-family='monospace'>proxy={proxy_label} [{row.get('task_proxy_name') or 'none'}]</text>"
        )
        if row.get("target_episode_policy"):
            parts.append(
                f"<text x='{center_x + half_bar + 16}' y='{top + 44}' font-size='11' font-family='monospace'>episode_policy={row.get('target_episode_policy')} repeats={row.get('target_episode_repeats') or 1} weight={row.get('target_support_weighting') or 'uniform'}</text>"
            )
        ratio = row.get("query_to_writer_grad_ratio")
        if ratio is not None:
            parts.append(
                f"<text x='{center_x + half_bar + 16}' y='{top + 58}' font-size='11' font-family='monospace'>q/w grad ratio={float(ratio):.3e}</text>"
            )
    parts.append("</svg>")
    destination.write_text("".join(parts))


def run_m3_stage_c_probe_summary(
    *,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    rows, gradient_audits = collect_stage_c_probe_rows(input_root)
    if dry_run:
        rows = rows[: max(1, min(3, len(rows)))]

    summary_csv = output_dir / "probe_summary.csv"
    summary_svg = output_dir / "probe_summary.svg"
    write_stage_c_probe_csv(summary_csv, rows)
    write_stage_c_probe_svg(summary_svg, rows)

    best_by_backbone: dict[str, dict[str, object]] = {}
    for row in rows:
        backbone = str(row["backbone"])
        current = best_by_backbone.get(backbone)
        if current is None or _stage_c_probe_row_key(row) > _stage_c_probe_row_key(current):
            best_by_backbone[backbone] = {
                "run_name": row["run_name"],
                "adaptation_target": row["adaptation_target"],
                "seed": row["seed"],
                "task_gain": float(row.get("task_gain") or 0.0),
                "best_adapt_task_score": row["best_adapt_task_score"],
                "best_adapt_task_proxy_score": row.get("best_adapt_task_proxy_score"),
                "task_proxy_name": row.get("task_proxy_name"),
                "best_adapt_query_loss": row["best_adapt_query_loss"],
                "adapt_learning_rate": row["adapt_learning_rate"],
                "adapt_steps": row["adapt_steps"],
                "target_eval_repeats": row.get("target_eval_repeats"),
                "target_episode_repeats": row.get("target_episode_repeats"),
                "target_episode_policy": row.get("target_episode_policy"),
                "target_support_weighting": row.get("target_support_weighting"),
                "trainable_parameter_count": row["trainable_parameter_count"],
            }

    q_only_by_backbone: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["adaptation_target"] != "q_only":
            continue
        backbone = str(row["backbone"])
        current = q_only_by_backbone.get(backbone)
        candidate = {
            "run_name": row["run_name"],
            "seed": row["seed"],
            "task_gain": row["task_gain"],
            "adapt_learning_rate": row["adapt_learning_rate"],
            "adapt_steps": row["adapt_steps"],
            "target_eval_repeats": row.get("target_eval_repeats"),
            "target_episode_repeats": row.get("target_episode_repeats"),
            "target_episode_policy": row.get("target_episode_policy"),
            "target_support_weighting": row.get("target_support_weighting"),
            "adaptation_effective": row["adaptation_effective"],
            "best_adapt_task_score": row["best_adapt_task_score"],
            "best_adapt_task_proxy_score": row.get("best_adapt_task_proxy_score"),
            "task_proxy_name": row.get("task_proxy_name"),
            "best_adapt_query_loss": row["best_adapt_query_loss"],
            "trainable_parameter_count": row["trainable_parameter_count"],
            "query_to_writer_grad_ratio": row["query_to_writer_grad_ratio"],
            "query_to_fuser_grad_ratio": row["query_to_fuser_grad_ratio"],
        }
        if current is None or _stage_c_probe_row_key(candidate) > _stage_c_probe_row_key(current):
            q_only_by_backbone[backbone] = candidate

    seed_consistent_by_backbone: dict[str, bool] = {}
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        seeds = {row["seed"] for row in rows if str(row["backbone"]) == backbone}
        seed_consistent_by_backbone[backbone] = len(seeds) <= 1

    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_probe_summary",
        "rows_collected": len(rows),
        "gradient_audits_collected": len(gradient_audits),
        "input_root": str(Path(input_root).resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "summary_plot": str(summary_svg.resolve()),
        "best_by_backbone": best_by_backbone,
        "q_only_by_backbone": q_only_by_backbone,
        "seed_consistent_by_backbone": seed_consistent_by_backbone,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
