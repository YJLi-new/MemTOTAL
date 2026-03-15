from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from memtotal.utils.io import write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _extract_story_context(segment: str) -> str:
    marker = "|| Candidate endings:"
    if marker in segment:
        return segment.split(marker, 1)[0].strip()
    return segment.strip()


def _normalized_overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _majority_value(values: list[str]) -> str:
    if not values:
        return ""
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _stringify_ids(values: set[str]) -> str:
    return "|".join(sorted(value for value in values if value))


def _aggregate_case_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    predicted_labels = [str(row.get("predicted_label", "")) for row in rows]
    competitor_labels = [str(row.get("top_competitor_label", "")) for row in rows]
    support_ids: set[str] = set()
    eval_query_set_ids: set[str] = set()
    for row in rows:
        support_ids.update(str(value) for value in row.get("support_ids", []))
        eval_query_set_ids.update(str(value) for value in row.get("eval_query_set_ids", []))
    representative = rows[0]
    correct_flags = [1.0 if bool(row.get("predicted_correct")) else 0.0 for row in rows]
    task_scores = [float(row.get("task_score", 0.0)) for row in rows]
    task_proxy_scores = [float(row.get("task_proxy_score", 0.0)) for row in rows]
    task_margins = [float(row.get("task_margin", 0.0)) for row in rows]
    gold_probabilities = [float(row.get("gold_probability", 0.0)) for row in rows]
    competitor_probabilities = [float(row.get("top_competitor_probability", 0.0)) for row in rows]
    return {
        "example_id": str(representative.get("example_id", "")),
        "benchmark_id": str(representative.get("benchmark_id", "")),
        "domain": str(representative.get("domain", "")),
        "task_name": str(representative.get("task_name", "")),
        "segment": str(representative.get("segment", "")),
        "gold_label": str(representative.get("gold_label", "")),
        "gold_text": str(representative.get("gold_text", "")),
        "predicted_label_majority": _majority_value(predicted_labels),
        "predicted_label_variants": _stringify_ids(set(predicted_labels)),
        "top_competitor_label_majority": _majority_value(competitor_labels),
        "top_competitor_text": str(representative.get("top_competitor_text", "")),
        "choices_json": json.dumps(representative.get("choices", []), ensure_ascii=True, sort_keys=True),
        "task_score": sum(task_scores) / len(task_scores),
        "task_proxy_score": sum(task_proxy_scores) / len(task_proxy_scores),
        "task_margin": sum(task_margins) / len(task_margins),
        "gold_probability": sum(gold_probabilities) / len(gold_probabilities),
        "top_competitor_probability": sum(competitor_probabilities) / len(competitor_probabilities),
        "correct_rate": sum(correct_flags) / len(correct_flags),
        "predicted_correct_majority": (_majority_value(["1" if flag else "0" for flag in correct_flags]) == "1"),
        "support_ids": _stringify_ids(support_ids),
        "eval_query_set_ids": _stringify_ids(eval_query_set_ids),
        "row_count": len(rows),
    }


def _build_heuristic_tags(
    *,
    zero_margin: float,
    final_margin: float,
    zero_proxy: float,
    final_proxy: float,
    context_gold_overlap: float,
    context_competitor_overlap: float,
    gold_competitor_jaccard: float,
    near_threshold_margin: float,
) -> list[str]:
    tags: list[str] = []
    if zero_margin < 0.0 and final_margin < 0.0 and final_margin > zero_margin:
        tags.append("improving_but_unflipped")
    if final_margin < 0.0 and final_margin >= -near_threshold_margin:
        tags.append("near_rank_flip")
    if final_proxy > zero_proxy and final_margin < 0.0:
        tags.append("proxy_up_without_flip")
    if gold_competitor_jaccard >= 0.35:
        tags.append("high_choice_overlap")
    overlap_gap = context_gold_overlap - context_competitor_overlap
    if overlap_gap <= -0.05:
        tags.append("story_context_favors_competitor")
    elif abs(overlap_gap) < 0.05:
        tags.append("story_context_ambiguous")
    return tags


def collect_stage_c_error_attribution_rows(
    input_root: str | Path,
    *,
    near_threshold_margin: float,
) -> list[dict[str, object]]:
    root = Path(input_root).resolve()
    pair_rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = _read_json(metrics_path)
        if metrics.get("training_stage") != "stage_c":
            continue
        if str(metrics.get("adaptation_target", "")) != "q_only":
            continue
        run_dir = metrics_path.parent
        case_dump_path = run_dir / "task_case_dump.jsonl"
        if not case_dump_path.exists():
            continue
        case_rows = _read_jsonl(case_dump_path)
        grouped: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
        for row in case_rows:
            grouped[(int(row.get("shot", 0)), int(row.get("step", 0)), str(row.get("example_id", "")))].append(row)
        best_shot = int(metrics.get("best_adapt_shot", 0))
        best_step = int(metrics.get("best_adapt_step", 0))
        zero_rows = {
            example_id: _aggregate_case_rows(rows)
            for (shot, step, example_id), rows in grouped.items()
            if shot == 0 and step == 0
        }
        final_rows = {
            example_id: _aggregate_case_rows(rows)
            for (shot, step, example_id), rows in grouped.items()
            if shot == best_shot and step == best_step
        }
        common_example_ids = sorted(set(zero_rows) & set(final_rows))
        for example_id in common_example_ids:
            zero_row = zero_rows[example_id]
            final_row = final_rows[example_id]
            story_context = _extract_story_context(str(final_row["segment"]))
            context_tokens = _tokenize(story_context)
            gold_tokens = _tokenize(str(final_row["gold_text"]))
            competitor_tokens = _tokenize(str(final_row["top_competitor_text"]))
            context_gold_overlap = _normalized_overlap(gold_tokens, context_tokens)
            context_competitor_overlap = _normalized_overlap(competitor_tokens, context_tokens)
            gold_competitor_jaccard = _jaccard(gold_tokens, competitor_tokens)
            heuristic_tags = _build_heuristic_tags(
                zero_margin=float(zero_row["task_margin"]),
                final_margin=float(final_row["task_margin"]),
                zero_proxy=float(zero_row["task_proxy_score"]),
                final_proxy=float(final_row["task_proxy_score"]),
                context_gold_overlap=context_gold_overlap,
                context_competitor_overlap=context_competitor_overlap,
                gold_competitor_jaccard=gold_competitor_jaccard,
                near_threshold_margin=near_threshold_margin,
            )
            pair_rows.append(
                {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "backbone": str(metrics.get("backbone", _read_json(run_dir / "run_info.json").get("backbone", ""))),
                    "seed": int(_read_json(run_dir / "run_info.json").get("seed", -1)),
                    "example_id": example_id,
                    "benchmark_id": str(final_row["benchmark_id"]),
                    "domain": str(final_row["domain"]),
                    "task_name": str(final_row["task_name"]),
                    "best_adapt_shot": best_shot,
                    "best_adapt_step": best_step,
                    "zero_task_score": float(zero_row["task_score"]),
                    "final_task_score": float(final_row["task_score"]),
                    "task_gain": float(final_row["task_score"]) - float(zero_row["task_score"]),
                    "zero_task_proxy_score": float(zero_row["task_proxy_score"]),
                    "final_task_proxy_score": float(final_row["task_proxy_score"]),
                    "proxy_gain": float(final_row["task_proxy_score"]) - float(zero_row["task_proxy_score"]),
                    "zero_task_margin": float(zero_row["task_margin"]),
                    "final_task_margin": float(final_row["task_margin"]),
                    "margin_gain": float(final_row["task_margin"]) - float(zero_row["task_margin"]),
                    "zero_correct_rate": float(zero_row["correct_rate"]),
                    "final_correct_rate": float(final_row["correct_rate"]),
                    "still_wrong": float(final_row["correct_rate"]) < 0.5,
                    "crosses_zero_margin": float(zero_row["task_margin"]) < 0.0 <= float(final_row["task_margin"]),
                    "near_threshold_unflipped": (
                        float(final_row["task_margin"]) < 0.0
                        and float(final_row["task_margin"]) >= -near_threshold_margin
                    ),
                    "gold_label": str(final_row["gold_label"]),
                    "gold_text": str(final_row["gold_text"]),
                    "predicted_label_majority": str(final_row["predicted_label_majority"]),
                    "predicted_label_variants": str(final_row["predicted_label_variants"]),
                    "top_competitor_label_majority": str(final_row["top_competitor_label_majority"]),
                    "top_competitor_text": str(final_row["top_competitor_text"]),
                    "story_context": story_context,
                    "context_gold_overlap": context_gold_overlap,
                    "context_competitor_overlap": context_competitor_overlap,
                    "context_overlap_gap": context_gold_overlap - context_competitor_overlap,
                    "gold_competitor_jaccard": gold_competitor_jaccard,
                    "support_ids": str(final_row["support_ids"]),
                    "eval_query_set_ids": str(final_row["eval_query_set_ids"]),
                    "heuristic_tags": "|".join(heuristic_tags),
                    "segment": str(final_row["segment"]),
                    "choices_json": str(final_row["choices_json"]),
                }
            )
    pair_rows.sort(
        key=lambda row: (
            str(row["backbone"]),
            float(abs(float(row["final_task_margin"]))),
            int(row["seed"]),
            str(row["example_id"]),
        )
    )
    return pair_rows


def _summarize_case_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    tag_counter: Counter[str] = Counter()
    for row in rows:
        for tag in str(row["heuristic_tags"]).split("|"):
            if tag:
                tag_counter[tag] += 1
    near_threshold_rows = [row for row in rows if bool(row["near_threshold_unflipped"])]
    improving_rows = [
        row
        for row in rows
        if float(row["zero_task_margin"]) < 0.0
        and float(row["final_task_margin"]) < 0.0
        and float(row["margin_gain"]) > 0.0
    ]
    return {
        "paired_case_count": len(rows),
        "near_threshold_bad_case_count": len(near_threshold_rows),
        "improving_but_unflipped_count": len(improving_rows),
        "cross_zero_margin_count": sum(1 for row in rows if bool(row["crosses_zero_margin"])),
        "mean_margin_gain": sum(float(row["margin_gain"]) for row in rows) / len(rows) if rows else 0.0,
        "mean_proxy_gain": sum(float(row["proxy_gain"]) for row in rows) / len(rows) if rows else 0.0,
        "mean_final_margin_near_threshold": (
            sum(float(row["final_task_margin"]) for row in near_threshold_rows) / len(near_threshold_rows)
            if near_threshold_rows
            else 0.0
        ),
        "tag_counts": dict(sorted(tag_counter.items())),
    }


def _write_report(
    output_path: Path,
    *,
    near_threshold_rows: list[dict[str, object]],
    by_backbone: dict[str, dict[str, object]],
    near_threshold_margin: float,
) -> None:
    lines = [
        "# Stage C Error Attribution",
        "",
        f"- near-threshold margin window: [-{near_threshold_margin}, 0)",
        "",
        "## Backbone Summary",
        "",
    ]
    for backbone, summary in by_backbone.items():
        lines.append(
            f"- {backbone}: paired={summary['paired_case_count']}, near_threshold={summary['near_threshold_bad_case_count']}, "
            f"improving_unflipped={summary['improving_but_unflipped_count']}, cross_zero={summary['cross_zero_margin_count']}"
        )
    lines.extend(["", "## Top Near-Threshold Cases", ""])
    if not near_threshold_rows:
        lines.append("- No near-threshold bad cases found.")
    else:
        for row in near_threshold_rows[:10]:
            lines.append(
                f"- {row['backbone']} seed={row['seed']} example={row['example_id']} "
                f"margin={float(row['zero_task_margin']):.4f}->{float(row['final_task_margin']):.4f} "
                f"proxy={float(row['zero_task_proxy_score']):.4f}->{float(row['final_task_proxy_score']):.4f} "
                f"tags={row['heuristic_tags'] or 'none'}"
            )
            lines.append(f"  gold: {row['gold_label']} {row['gold_text']}")
            lines.append(
                f"  competitor: {row['top_competitor_label_majority']} {row['top_competitor_text']}"
            )
            lines.append(f"  support_ids: {row['support_ids']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _write_svg(output_path: Path, by_backbone: dict[str, dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not by_backbone:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='560' height='120'>"
            "<text x='24' y='64' font-size='18' font-family='monospace'>No Stage C error-attribution rows</text></svg>"
        )
        return
    width = 920
    height = 150 + 110 * len(by_backbone)
    max_cases = max(int(summary["paired_case_count"]) for summary in by_backbone.values())
    bar_width = 280
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='32' font-size='20' font-family='monospace'>M3 Stage C error attribution</text>",
        "<text x='24' y='54' font-size='12' font-family='monospace'>paired / near-threshold / improving-but-unflipped</text>",
    ]
    palette = ["#2f5aa8", "#b5651d", "#2b8a3e"]
    for index, (backbone, summary) in enumerate(sorted(by_backbone.items())):
        top = 92 + index * 104
        parts.append(f"<text x='24' y='{top}' font-size='13' font-family='monospace'>{backbone}</text>")
        labels = [
            ("paired", int(summary["paired_case_count"]), palette[0]),
            ("near-threshold", int(summary["near_threshold_bad_case_count"]), palette[1]),
            ("improving-unflipped", int(summary["improving_but_unflipped_count"]), palette[2]),
        ]
        for offset, (label, value, color) in enumerate(labels):
            row_top = top + 12 + offset * 26
            scaled = 0 if max_cases == 0 else max(4, round((value / max_cases) * bar_width)) if value > 0 else 0
            parts.append(
                f"<text x='36' y='{row_top + 14}' font-size='11' font-family='monospace'>{label}</text>"
            )
            parts.append(
                f"<rect x='220' y='{row_top}' width='{bar_width}' height='18' fill='#efe6d0' rx='4' />"
            )
            if scaled > 0:
                parts.append(
                    f"<rect x='220' y='{row_top}' width='{scaled}' height='18' fill='{color}' rx='4' />"
                )
            parts.append(
                f"<text x='516' y='{row_top + 14}' font-size='11' font-family='monospace'>{value}</text>"
            )
        parts.append(
            f"<text x='600' y='{top + 26}' font-size='11' font-family='monospace'>mean_margin_gain={float(summary['mean_margin_gain']):.4e}</text>"
        )
        parts.append(
            f"<text x='600' y='{top + 44}' font-size='11' font-family='monospace'>mean_proxy_gain={float(summary['mean_proxy_gain']):.4e}</text>"
        )
    parts.append("</svg>")
    output_path.write_text("".join(parts))


def run_m3_stage_c_error_attribution(
    *,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
    near_threshold_margin: float,
) -> dict[str, object]:
    pair_rows = collect_stage_c_error_attribution_rows(
        input_root,
        near_threshold_margin=near_threshold_margin,
    )
    if dry_run:
        pair_rows = pair_rows[: max(1, min(8, len(pair_rows)))]
    near_threshold_rows = [
        row
        for row in pair_rows
        if bool(row["near_threshold_unflipped"]) and bool(row["still_wrong"])
    ]
    stubborn_rows = [
        row
        for row in pair_rows
        if bool(row["still_wrong"]) and float(row["margin_gain"]) <= 0.0
    ]
    by_backbone: dict[str, dict[str, object]] = {}
    for backbone in sorted({str(row["backbone"]) for row in pair_rows}):
        backbone_rows = [row for row in pair_rows if str(row["backbone"]) == backbone]
        by_backbone[backbone] = _summarize_case_rows(backbone_rows)

    case_pairs_csv = output_dir / "case_pairs.csv"
    near_threshold_csv = output_dir / "near_threshold_bad_cases.csv"
    stubborn_csv = output_dir / "stubborn_wrong_cases.csv"
    report_path = output_dir / "report.md"
    summary_svg = output_dir / "summary.svg"
    fieldnames = [
        "run_name",
        "backbone",
        "seed",
        "example_id",
        "benchmark_id",
        "domain",
        "task_name",
        "best_adapt_shot",
        "best_adapt_step",
        "zero_task_score",
        "final_task_score",
        "task_gain",
        "zero_task_proxy_score",
        "final_task_proxy_score",
        "proxy_gain",
        "zero_task_margin",
        "final_task_margin",
        "margin_gain",
        "zero_correct_rate",
        "final_correct_rate",
        "still_wrong",
        "crosses_zero_margin",
        "near_threshold_unflipped",
        "gold_label",
        "gold_text",
        "predicted_label_majority",
        "predicted_label_variants",
        "top_competitor_label_majority",
        "top_competitor_text",
        "question_gold_overlap",
        "context_gold_overlap",
        "context_competitor_overlap",
        "context_overlap_gap",
        "gold_competitor_jaccard",
        "support_ids",
        "eval_query_set_ids",
        "heuristic_tags",
        "story_context",
        "segment",
        "choices_json",
        "run_dir",
    ]
    _write_csv(case_pairs_csv, pair_rows, fieldnames)
    _write_csv(near_threshold_csv, near_threshold_rows, fieldnames)
    _write_csv(stubborn_csv, stubborn_rows, fieldnames)
    _write_report(
        report_path,
        near_threshold_rows=near_threshold_rows,
        by_backbone=by_backbone,
        near_threshold_margin=near_threshold_margin,
    )
    _write_svg(summary_svg, by_backbone)
    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_error_attribution",
        "input_root": str(Path(input_root).resolve()),
        "near_threshold_margin": near_threshold_margin,
        "rows_collected": len(pair_rows),
        "near_threshold_bad_case_count": len(near_threshold_rows),
        "stubborn_wrong_case_count": len(stubborn_rows),
        "case_pairs_csv": str(case_pairs_csv.resolve()),
        "near_threshold_bad_cases_csv": str(near_threshold_csv.resolve()),
        "stubborn_wrong_cases_csv": str(stubborn_csv.resolve()),
        "report_path": str(report_path.resolve()),
        "summary_svg": str(summary_svg.resolve()),
        "by_backbone": by_backbone,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
