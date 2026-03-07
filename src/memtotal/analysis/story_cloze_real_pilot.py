from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from memtotal.analysis.reporting import write_sanity_plot, write_summary_csv
from memtotal.data import load_jsonl_dataset
from memtotal.utils.io import write_json, write_jsonl


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _normalized_overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _assign_shuffled_memory_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = [dict(row) for row in sorted(rows, key=lambda item: str(item["id"]))]
    shuffled_ids = [str(row["id"]) for row in ordered[1:]] + [str(ordered[0]["id"])]
    assigned = []
    for row, shuffled_id in zip(ordered, shuffled_ids, strict=True):
        row["shuffled_memory_example_id"] = shuffled_id
        assigned.append(row)
    return assigned


def _select_label_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(dict(row))
    rng = random.Random(seed)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
    ordered_labels = sorted(by_label)
    selected: list[dict[str, Any]] = []
    while ordered_labels and len(selected) < count:
        next_labels: list[str] = []
        for label in ordered_labels:
            label_rows = by_label[label]
            if label_rows and len(selected) < count:
                selected.append(label_rows.pop())
            if label_rows:
                next_labels.append(label)
        ordered_labels = next_labels
    return selected


def _rows_by_example(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["example_id"]): row for row in rows}


def _read_selected_case_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    selected_step = int(metrics.get("best_adapt_step", 0))
    case_rows = [
        json.loads(line)
        for line in (run_dir / "task_case_dump.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return metrics, [row for row in case_rows if int(row["step"]) == selected_step]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


_REAL_PILOT_ARM_KEYS = {
    "A": "base_only|real|continuation_retrieval",
    "B": "shared_summary_late_fusion|real|continuation_retrieval",
    "C": "candidate_conditioned_late_fusion|real|continuation_retrieval",
    "D": "candidate_conditioned_late_fusion|shuffled|continuation_retrieval",
    "E": "candidate_conditioned_late_fusion|real|choice_ce_plus_margin",
    "F": "shared_plus_candidate_delta_late_fusion|real|continuation_retrieval",
    "G": "shared_plus_candidate_delta_late_fusion|shuffled|continuation_retrieval",
}
_DEFAULT_COMPARE_ALIASES = ["A", "B", "C", "D", "E"]
_DEFAULT_COMPARE_PAIRS = ["A->B", "A->C", "C->D", "C->E"]
_DEFAULT_ORACLE_ALPHA_GRID = [
    0.0,
    0.25,
    -0.25,
    0.5,
    -0.5,
    1.0,
    -1.0,
    2.0,
    -2.0,
    4.0,
    -4.0,
    8.0,
    -8.0,
    16.0,
    -16.0,
    32.0,
    -32.0,
    64.0,
    -64.0,
    128.0,
    -128.0,
    256.0,
    -256.0,
    512.0,
    -512.0,
    1024.0,
    -1024.0,
    2048.0,
    -2048.0,
    4096.0,
    -4096.0,
]


def _run_real_pilot_split_impl(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> None:
    source_dataset_path = Path(config["task"]["dataset_path"]).resolve()
    source_rows = load_jsonl_dataset(source_dataset_path)
    max_examples = 16 if dry_run else int(config["runtime"].get("screen_source_examples", 256))
    support_examples = int(config["runtime"].get("pilot_support_examples", 8))
    selected_rows = source_rows[: min(len(source_rows), max_examples)]
    support_rows = _select_label_balanced_rows(
        selected_rows,
        count=min(support_examples, len(selected_rows)),
        seed=int(config["runtime"].get("support_selection_seed", config["runtime"].get("seed_offset", 0))),
    )
    support_ids = {str(row["id"]) for row in support_rows}
    eval_rows = [dict(row) for row in selected_rows if str(row["id"]) not in support_ids]
    support_rows = _assign_shuffled_memory_ids(support_rows)
    eval_rows = _assign_shuffled_memory_ids(eval_rows)
    for row in support_rows:
        row["screening_split"] = "support"
    for row in eval_rows:
        row["screening_split"] = "screen_eval"

    pilot_root = Path(
        config["runtime"].get(
            "pilot_output_root",
            f"data/benchmarks/pilots/{config.get('task', {}).get('benchmark_id', 'story_cloze')}",
        )
    ).resolve()
    pilot_root.mkdir(parents=True, exist_ok=True)
    support_path = pilot_root / f"pilot-support{len(support_rows)}.jsonl"
    eval_path = pilot_root / f"screen-eval{len(eval_rows)}.jsonl"
    manifest_path = pilot_root / "screen-split-manifest.json"
    write_jsonl(support_path, support_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "source_dataset_path": str(source_dataset_path),
        "support_dataset_path": str(support_path.resolve()),
        "eval_dataset_path": str(eval_path.resolve()),
        "support_examples": len(support_rows),
        "eval_examples": len(eval_rows),
        "support_ids": [str(row["id"]) for row in support_rows],
        "support_labels": [str(row["label"]) for row in support_rows],
    }
    write_json(manifest_path, manifest)
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": str(config["runtime"].get("analysis_mode", "real_pilot_split")),
            "source_dataset_path": str(source_dataset_path),
            "support_dataset_path": str(support_path.resolve()),
            "eval_dataset_path": str(eval_path.resolve()),
            "support_examples": len(support_rows),
            "eval_examples": len(eval_rows),
            "manifest_path": str(manifest_path.resolve()),
        },
    )


def _collect_stage_c_real_pilot_runs(root: Path) -> dict[str, list[tuple[dict[str, Any], Path]]]:
    runs_by_key: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("training_stage") != "stage_c_real_pilot":
            continue
        key = "|".join(
            [
                str(metrics.get("decision_mode")),
                str(metrics.get("memory_control_mode")),
                str(metrics.get("choice_objective")),
            ]
        )
        runs_by_key.setdefault(key, []).append((metrics, metrics_path.parent))
    return runs_by_key


def _pick_real_pilot_run(
    candidates: list[tuple[dict[str, Any], Path]],
    *,
    pilot_split_name: str,
    dataset_filename: str,
) -> tuple[dict[str, Any], Path] | None:
    matching_candidates = [
        (metrics, run_dir)
        for metrics, run_dir in candidates
        if str(metrics.get("pilot_split")) == pilot_split_name
        or Path(str(metrics.get("eval_dataset_path", ""))).name == dataset_filename
    ]
    if len(matching_candidates) == 1:
        return matching_candidates[0]
    if len(matching_candidates) > 1:
        matching_candidates.sort(key=lambda item: str(item[1]))
        return matching_candidates[-1]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_import_arm_runs(config: dict[str, Any] | None) -> dict[str, Path]:
    raw_imports = None if config is None else config.get("runtime", {}).get("import_arm_runs")
    if raw_imports is None:
        return {}
    if not isinstance(raw_imports, dict):
        raise ValueError("runtime.import_arm_runs must be a mapping from alias to run directory.")
    resolved: dict[str, Path] = {}
    for alias, run_dir in raw_imports.items():
        alias_name = str(alias).strip()
        if alias_name not in _REAL_PILOT_ARM_KEYS:
            raise ValueError(f"Unsupported imported arm alias: {alias_name}")
        resolved[alias_name] = Path(str(run_dir)).resolve()
    return resolved


def _resolve_compare_aliases(config: dict[str, Any] | None) -> list[str]:
    raw_aliases = None if config is None else config.get("runtime", {}).get("required_arm_aliases")
    if raw_aliases is None:
        return list(_DEFAULT_COMPARE_ALIASES)
    if isinstance(raw_aliases, str):
        aliases = [part.strip() for part in raw_aliases.split(",") if part.strip()]
    else:
        aliases = [str(part).strip() for part in raw_aliases if str(part).strip()]
    if not aliases:
        raise ValueError("runtime.required_arm_aliases must not be empty.")
    unknown = [alias for alias in aliases if alias not in _REAL_PILOT_ARM_KEYS]
    if unknown:
        raise ValueError(f"Unsupported arm aliases: {unknown}")
    return aliases


def _resolve_compare_pairs(
    config: dict[str, Any] | None,
    *,
    aliases: list[str],
) -> list[tuple[str, str]]:
    raw_pairs = None if config is None else config.get("runtime", {}).get("pair_specs")
    if raw_pairs is None:
        requested_pairs = list(_DEFAULT_COMPARE_PAIRS)
    elif isinstance(raw_pairs, str):
        requested_pairs = [part.strip() for part in raw_pairs.split(",") if part.strip()]
    else:
        requested_pairs = [str(part).strip() for part in raw_pairs if str(part).strip()]
    resolved: list[tuple[str, str]] = []
    alias_set = set(aliases)
    for spec in requested_pairs:
        left_alias, right_alias = [part.strip() for part in spec.split("->", maxsplit=1)]
        if left_alias not in alias_set or right_alias not in alias_set:
            continue
        resolved.append((left_alias, right_alias))
    return resolved


def _resolve_flip_compare_aliases(config: dict[str, Any] | None) -> tuple[str, str]:
    raw_spec = None if config is None else config.get("runtime", {}).get("flip_compare_aliases")
    if raw_spec is None:
        return ("A", "C")
    if isinstance(raw_spec, str):
        parts = [part.strip() for part in raw_spec.split("->") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw_spec]
    if len(parts) != 2 or any(part not in _REAL_PILOT_ARM_KEYS for part in parts):
        raise ValueError("runtime.flip_compare_aliases must specify exactly two supported aliases.")
    return parts[0], parts[1]


def _resolve_memory_control_pair(config: dict[str, Any] | None) -> tuple[str, str]:
    raw_spec = None if config is None else config.get("runtime", {}).get("memory_control_pair")
    if raw_spec is None:
        return ("C", "D")
    if isinstance(raw_spec, str):
        parts = [part.strip() for part in raw_spec.split("->") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw_spec]
    if len(parts) != 2 or any(part not in _REAL_PILOT_ARM_KEYS for part in parts):
        raise ValueError("runtime.memory_control_pair must specify exactly two supported aliases.")
    return parts[0], parts[1]


def _resolve_pilot_split_name(config: dict[str, Any] | None, *, default: str) -> str:
    if config is None:
        return default
    return str(config.get("runtime", {}).get("pilot_split_name", default))


def _resolve_context_preview(row: dict[str, Any]) -> str:
    for field in ("story", "claim", "segment", "question", "context"):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def _build_arm_summary_row(alias: str, metrics: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "alias": alias,
        "run_dir": str(run_dir.resolve()),
        "mode": "analysis",
        "decision_mode": str(metrics["decision_mode"]),
        "memory_control_mode": str(metrics["memory_control_mode"]),
        "choice_objective": str(metrics["choice_objective"]),
        "primary_metric": "best_adapt_task_score",
        "primary_score": float(metrics["best_adapt_task_score"]),
        "zero_shot_task_score": float(metrics["zero_shot_task_score"]),
        "best_adapt_task_score": float(metrics["best_adapt_task_score"]),
        "best_adapt_task_proxy_score": float(metrics["best_adapt_task_proxy_score"]),
        "best_adapt_task_margin": float(metrics["best_adapt_task_margin"]),
        "best_adapt_step": int(metrics["best_adapt_step"]),
        "task_case_dump_path": str(metrics["task_case_dump_path"]),
    }


def _load_selected_rows_by_alias(
    *,
    input_root: Path,
    config: dict[str, Any] | None,
    default_pilot_split: str,
    default_dataset_filename: str,
) -> tuple[dict[str, tuple[dict[str, Any], Path]], dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    pilot_split_name = _resolve_pilot_split_name(config, default=default_pilot_split)
    dataset_filename = default_dataset_filename
    if config is not None and "task" in config and "dataset_path" in config["task"]:
        dataset_filename = Path(str(config["task"]["dataset_path"])).name
    aliases = _resolve_compare_aliases(config)
    imported_runs = _resolve_import_arm_runs(config)
    runs_by_key = _collect_stage_c_real_pilot_runs(input_root)
    runs: dict[str, tuple[dict[str, Any], Path]] = {}
    selected_rows_by_alias: dict[str, dict[str, dict[str, Any]]] = {}
    arm_summary_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for alias in aliases:
        if alias in imported_runs:
            run_dir = imported_runs[alias]
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                raise ValueError(f"Imported arm {alias} is missing metrics.json at {run_dir}")
            metrics = json.loads(metrics_path.read_text())
            runs[alias] = (metrics, run_dir)
            _selected_metrics, rows = _read_selected_case_rows(run_dir)
            selected_rows_by_alias[alias] = _rows_by_example(rows)
            arm_summary_rows.append(_build_arm_summary_row(alias, metrics, run_dir))
            continue
        key = _REAL_PILOT_ARM_KEYS[alias]
        picked = _pick_real_pilot_run(
            runs_by_key.get(key, []),
            pilot_split_name=pilot_split_name,
            dataset_filename=dataset_filename,
        )
        if picked is None:
            missing.append(alias)
            continue
        runs[alias] = picked
        metrics, run_dir = picked
        _selected_metrics, rows = _read_selected_case_rows(run_dir)
        selected_rows_by_alias[alias] = _rows_by_example(rows)
        arm_summary_rows.append(_build_arm_summary_row(alias, metrics, run_dir))
    if missing:
        raise ValueError(f"Missing required real pilot arms: {missing}")
    return runs, selected_rows_by_alias, arm_summary_rows


def run_story_cloze_real_pilot_split(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> None:
    _run_real_pilot_split_impl(config=config, output_dir=output_dir, dry_run=dry_run)


def run_fever_real_pilot_split(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> None:
    _run_real_pilot_split_impl(config=config, output_dir=output_dir, dry_run=dry_run)


def _resolve_story_context_favors_competitor(row: dict[str, Any]) -> bool:
    story_tokens = _tokenize(str(row.get("story", "")))
    choices = row.get("choices", [])
    gold_label = str(row["label"])
    gold_text = ""
    competitor_text = ""
    for choice in choices:
        if str(choice["label"]) == gold_label:
            gold_text = str(choice["text"])
        else:
            competitor_text = str(choice["text"])
    gold_overlap = _normalized_overlap(_tokenize(gold_text), story_tokens)
    competitor_overlap = _normalized_overlap(_tokenize(competitor_text), story_tokens)
    return (gold_overlap - competitor_overlap) <= -0.05


def _select_bucket_rows(
    rows: list[dict[str, Any]],
    *,
    bucket: str,
    quota: int,
) -> list[dict[str, Any]]:
    if bucket == "near_threshold_bad":
        ordered = sorted(rows, key=lambda row: abs(float(row["screening_base_margin"])))
        return ordered[:quota]
    if bucket == "improving_but_unflipped":
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["screening_margin_gain"]),
                abs(float(row["screening_shared_margin"])),
                str(row["id"]),
            ),
        )
        return ordered[:quota]
    if bucket == "base_correct_control":
        near_quota = min(max(1, quota // 2), quota)
        near = sorted(rows, key=lambda row: (float(row["screening_base_margin"]), str(row["id"])))[:near_quota]
        near_ids = {str(row["id"]) for row in near}
        stable = [
            row
            for row in sorted(rows, key=lambda row: (-float(row["screening_base_margin"]), str(row["id"])))
            if str(row["id"]) not in near_ids
        ][: max(0, quota - len(near))]
        return [*near, *stable]
    ordered = sorted(rows, key=lambda row: (abs(float(row["screening_base_margin"])), str(row["id"])))
    return ordered[:quota]


def _fill_remaining_rows(
    already_selected_ids: set[str],
    candidate_rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    fallback_wrong = [
        row
        for row in sorted(candidate_rows, key=lambda row: (abs(float(row["screening_base_margin"])), str(row["id"])))
        if str(row["id"]) not in already_selected_ids and not bool(row["screening_base_correct"])
    ]
    chosen = fallback_wrong[:count]
    chosen_ids = {str(row["id"]) for row in chosen}
    if len(chosen) < count:
        fallback_correct = [
            row
            for row in sorted(candidate_rows, key=lambda row: (float(row["screening_base_margin"]), str(row["id"])))
            if str(row["id"]) not in already_selected_ids and str(row["id"]) not in chosen_ids
        ]
        chosen.extend(fallback_correct[: max(0, count - len(chosen))])
    return chosen


def _select_calibration_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    bucket_rank = {
        "near_threshold_bad": 0,
        "improving_but_unflipped": 1,
        "stubborn_wrong_story_context": 2,
        "stubborn_wrong_other": 3,
        "base_correct_control": 4,
    }
    return sorted(
        rows,
        key=lambda row: (
            bucket_rank.get(str(row.get("screening_bucket", "")), 5),
            abs(float(row.get("screening_base_margin", 0.0)))
            if not bool(row.get("screening_base_correct", False))
            else float(row.get("screening_base_margin", 0.0)),
            str(row["id"]),
        ),
    )[:count]


def run_story_cloze_real_fixed_set_builder(
    *,
    config: dict[str, Any],
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> None:
    source_eval_path = Path(config["task"]["dataset_path"]).resolve()
    source_rows = load_jsonl_dataset(source_eval_path)
    source_by_id = {str(row["id"]): dict(row) for row in source_rows}
    root = Path(input_root).resolve()
    runs_by_key = _collect_stage_c_real_pilot_runs(root)
    screening_filename = Path(config["task"]["dataset_path"]).name
    picked_base = _pick_real_pilot_run(
        runs_by_key.get(_REAL_PILOT_ARM_KEYS["A"], []),
        pilot_split_name="screen_eval248",
        dataset_filename=screening_filename,
    )
    picked_shared = _pick_real_pilot_run(
        runs_by_key.get(_REAL_PILOT_ARM_KEYS["B"], []),
        pilot_split_name="screen_eval248",
        dataset_filename=screening_filename,
    )
    if picked_base is None or picked_shared is None:
        raise ValueError("Expected screening base/shared runs for fixed-set building, but they were not found.")
    _, base_run_dir = picked_base
    _, shared_run_dir = picked_shared
    _base_metrics, base_rows = _read_selected_case_rows(base_run_dir)
    _shared_metrics, shared_rows = _read_selected_case_rows(shared_run_dir)
    base_by_id = _rows_by_example(base_rows)
    shared_by_id = _rows_by_example(shared_rows)
    common_ids = sorted(set(base_by_id) & set(shared_by_id) & set(source_by_id))
    candidate_rows: list[dict[str, Any]] = []
    for example_id in common_ids:
        source_row = dict(source_by_id[example_id])
        base_row = base_by_id[example_id]
        shared_row = shared_by_id[example_id]
        base_margin = float(base_row["final_margin"])
        shared_margin = float(shared_row["final_margin"])
        margin_gain = shared_margin - base_margin
        base_correct = bool(base_row["predicted_correct"])
        shared_correct = bool(shared_row["predicted_correct"])
        story_context_favors_competitor = _resolve_story_context_favors_competitor(source_row)
        if not base_correct and base_margin >= -0.1:
            bucket = "near_threshold_bad"
        elif not base_correct and not shared_correct and margin_gain > 0.0 and base_margin < -0.1:
            bucket = "improving_but_unflipped"
        elif not base_correct and story_context_favors_competitor:
            bucket = "stubborn_wrong_story_context"
        elif not base_correct:
            bucket = "stubborn_wrong_other"
        else:
            bucket = "base_correct_control"
        source_row.update(
            {
                "screening_bucket": bucket,
                "screening_split": "fixed100",
                "screening_base_margin": base_margin,
                "screening_shared_margin": shared_margin,
                "screening_margin_gain": margin_gain,
                "screening_base_correct": base_correct,
                "screening_shared_correct": shared_correct,
            }
        )
        candidate_rows.append(source_row)

    quotas = {
        "near_threshold_bad": 20,
        "improving_but_unflipped": 20,
        "stubborn_wrong_story_context": 20,
        "stubborn_wrong_other": 20,
        "base_correct_control": 20,
    }
    calibration_quota = max(0, int(config["runtime"].get("pilot_calibration_examples", 32)))
    if dry_run:
        quotas = {key: min(2, value) for key, value in quotas.items()}
        calibration_quota = min(4, calibration_quota)
    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    bucket_counts: dict[str, int] = {}
    for bucket, quota in quotas.items():
        bucket_rows = [row for row in candidate_rows if str(row["screening_bucket"]) == bucket]
        chosen = _select_bucket_rows(bucket_rows, bucket=bucket, quota=quota)
        selected_rows.extend(chosen)
        selected_ids.update(str(row["id"]) for row in chosen)
        bucket_counts[bucket] = len(chosen)
    total_quota = sum(quotas.values())
    if len(selected_rows) < total_quota:
        selected_rows.extend(
            _fill_remaining_rows(
                selected_ids,
                candidate_rows,
                total_quota - len(selected_rows),
            )
        )
    selected_rows = _assign_shuffled_memory_ids(selected_rows[:total_quota])
    selected_ids = {str(row["id"]) for row in selected_rows}
    calibration_rows = _select_calibration_rows(
        [row for row in candidate_rows if str(row["id"]) not in selected_ids],
        count=calibration_quota,
    )
    for row in calibration_rows:
        row["screening_split"] = "calibration"
    calibration_rows = _assign_shuffled_memory_ids(calibration_rows)
    bucket_counts = {}
    for row in selected_rows:
        bucket = str(row["screening_bucket"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    pilot_root = Path(config["runtime"].get("pilot_output_root", "data/benchmarks/pilots/story_cloze")).resolve()
    pilot_root.mkdir(parents=True, exist_ok=True)
    fixed_path = pilot_root / f"fixed{len(selected_rows)}.jsonl"
    calibration_path = pilot_root / f"calibration-hard{len(calibration_rows)}.jsonl"
    manifest_path = pilot_root / "fixed100-manifest.json"
    write_jsonl(fixed_path, selected_rows)
    write_jsonl(calibration_path, calibration_rows)
    manifest = {
        "source_eval_dataset_path": str(source_eval_path),
        "fixed_eval_dataset_path": str(fixed_path.resolve()),
        "calibration_dataset_path": str(calibration_path.resolve()),
        "fixed_eval_examples": len(selected_rows),
        "calibration_examples": len(calibration_rows),
        "bucket_counts": bucket_counts,
        "selected_ids": [str(row["id"]) for row in selected_rows],
        "calibration_ids": [str(row["id"]) for row in calibration_rows],
        "base_run_dir": str(base_run_dir.resolve()),
        "shared_run_dir": str(shared_run_dir.resolve()),
    }
    write_json(manifest_path, manifest)
    summary_rows = []
    for bucket, quota in quotas.items():
        summary_rows.append(
            {
                "bucket": bucket,
                "requested_examples": quota,
                "selected_examples": sum(1 for row in selected_rows if str(row["screening_bucket"]) == bucket),
            }
        )
    _write_csv(output_dir / "bucket_summary.csv", summary_rows)
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "story_cloze_real_fixed_set_builder",
            "fixed_eval_dataset_path": str(fixed_path.resolve()),
            "fixed_eval_examples": len(selected_rows),
            "calibration_dataset_path": str(calibration_path.resolve()),
            "calibration_examples": len(calibration_rows),
            "manifest_path": str(manifest_path.resolve()),
            "bucket_counts": bucket_counts,
        },
    )


def run_fever_real_fixed_set_builder(
    *,
    config: dict[str, Any],
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> None:
    source_eval_path = Path(config["task"]["dataset_path"]).resolve()
    source_rows = load_jsonl_dataset(source_eval_path)
    source_by_id = {str(row["id"]): dict(row) for row in source_rows}
    root = Path(input_root).resolve()
    runs_by_key = _collect_stage_c_real_pilot_runs(root)
    screening_filename = Path(config["task"]["dataset_path"]).name
    picked_base = _pick_real_pilot_run(
        runs_by_key.get(_REAL_PILOT_ARM_KEYS["A"], []),
        pilot_split_name="screen_eval248",
        dataset_filename=screening_filename,
    )
    picked_shared = _pick_real_pilot_run(
        runs_by_key.get(_REAL_PILOT_ARM_KEYS["B"], []),
        pilot_split_name="screen_eval248",
        dataset_filename=screening_filename,
    )
    if picked_base is None or picked_shared is None:
        raise ValueError("Expected screening base/shared runs for FEVER fixed-set building, but they were not found.")
    _, base_run_dir = picked_base
    _, shared_run_dir = picked_shared
    _base_metrics, base_rows = _read_selected_case_rows(base_run_dir)
    _shared_metrics, shared_rows = _read_selected_case_rows(shared_run_dir)
    base_by_id = _rows_by_example(base_rows)
    shared_by_id = _rows_by_example(shared_rows)
    common_ids = sorted(set(base_by_id) & set(shared_by_id) & set(source_by_id))

    candidate_rows: list[dict[str, Any]] = []
    for example_id in common_ids:
        source_row = dict(source_by_id[example_id])
        base_row = base_by_id[example_id]
        shared_row = shared_by_id[example_id]
        base_margin = float(base_row["final_margin"])
        shared_margin = float(shared_row["final_margin"])
        margin_gain = shared_margin - base_margin
        base_correct = bool(base_row["predicted_correct"])
        shared_correct = bool(shared_row["predicted_correct"])
        if (not base_correct) and base_margin >= -0.1:
            bucket = "near_threshold_bad"
        elif (not base_correct) and (not shared_correct) and margin_gain > 0.0:
            bucket = "improving_but_unflipped"
        elif not base_correct:
            bucket = "stubborn_wrong"
        else:
            bucket = "base_correct_control"
        source_row.update(
            {
                "screening_bucket": bucket,
                "screening_split": "fixed64",
                "screening_base_margin": base_margin,
                "screening_shared_margin": shared_margin,
                "screening_margin_gain": margin_gain,
                "screening_base_correct": base_correct,
                "screening_shared_correct": shared_correct,
            }
        )
        candidate_rows.append(source_row)

    quotas = {
        "near_threshold_bad": 16,
        "improving_but_unflipped": 16,
        "stubborn_wrong": 16,
        "base_correct_control": 16,
    }
    if dry_run:
        quotas = {key: min(2, value) for key, value in quotas.items()}

    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for bucket in ("near_threshold_bad", "improving_but_unflipped", "stubborn_wrong", "base_correct_control"):
        if bucket == "stubborn_wrong":
            bucket_rows = [
                row
                for row in candidate_rows
                if (not bool(row["screening_base_correct"])) and str(row["id"]) not in selected_ids
            ]
        else:
            bucket_rows = [
                row
                for row in candidate_rows
                if str(row["screening_bucket"]) == bucket and str(row["id"]) not in selected_ids
            ]
        chosen = _select_bucket_rows(bucket_rows, bucket=bucket, quota=quotas[bucket])
        if bucket == "stubborn_wrong":
            chosen = [{**row, "screening_bucket": "stubborn_wrong"} for row in chosen]
        selected_rows.extend(chosen)
        selected_ids.update(str(row["id"]) for row in chosen)

    total_quota = sum(quotas.values())
    if len(selected_rows) < total_quota:
        selected_rows.extend(
            _fill_remaining_rows(
                selected_ids,
                candidate_rows,
                total_quota - len(selected_rows),
            )
        )
    selected_rows = _assign_shuffled_memory_ids(selected_rows[:total_quota])
    bucket_counts: dict[str, int] = {}
    for row in selected_rows:
        bucket = str(row["screening_bucket"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    pilot_root = Path(config["runtime"].get("pilot_output_root", "data/benchmarks/pilots/fever")).resolve()
    pilot_root.mkdir(parents=True, exist_ok=True)
    fixed_path = pilot_root / f"fixed{len(selected_rows)}.jsonl"
    manifest_path = pilot_root / "fixed64-manifest.json"
    write_jsonl(fixed_path, selected_rows)
    manifest = {
        "source_eval_dataset_path": str(source_eval_path),
        "fixed_eval_dataset_path": str(fixed_path.resolve()),
        "fixed_eval_examples": len(selected_rows),
        "bucket_counts": bucket_counts,
        "selected_ids": [str(row["id"]) for row in selected_rows],
        "base_run_dir": str(base_run_dir.resolve()),
        "shared_run_dir": str(shared_run_dir.resolve()),
    }
    write_json(manifest_path, manifest)
    summary_rows = [
        {
            "bucket": bucket,
            "requested_examples": quota,
            "selected_examples": sum(1 for row in selected_rows if str(row["screening_bucket"]) == bucket),
        }
        for bucket, quota in quotas.items()
    ]
    _write_csv(output_dir / "bucket_summary.csv", summary_rows)
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "fever_real_fixed_set_builder",
            "fixed_eval_dataset_path": str(fixed_path.resolve()),
            "fixed_eval_examples": len(selected_rows),
            "manifest_path": str(manifest_path.resolve()),
            "bucket_counts": bucket_counts,
        },
    )


def run_stage_c_real_pilot_compare(
    *,
    config: dict[str, Any] | None = None,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> None:
    root = Path(input_root).resolve()
    default_dataset_filename = "fixed100.jsonl"
    if config is not None:
        default_dataset_filename = Path(str(config["task"]["dataset_path"])).name
    runs, selected_rows_by_alias, arm_summary_rows = _load_selected_rows_by_alias(
        input_root=root,
        config=config,
        default_pilot_split="fixed100",
        default_dataset_filename=default_dataset_filename,
    )

    pair_rows: list[dict[str, Any]] = []
    pair_specs = _resolve_compare_pairs(config, aliases=sorted(runs))
    for left_alias, right_alias in pair_specs:
        left_rows = selected_rows_by_alias[left_alias]
        right_rows = selected_rows_by_alias[right_alias]
        common_ids = sorted(set(left_rows) & set(right_rows))
        if dry_run:
            common_ids = common_ids[: min(8, len(common_ids))]
        flip_count = 0
        regression_count = 0
        task_gain_total = 0.0
        proxy_gain_total = 0.0
        margin_gain_total = 0.0
        for example_id in common_ids:
            left_row = left_rows[example_id]
            right_row = right_rows[example_id]
            left_correct = bool(left_row["predicted_correct"])
            right_correct = bool(right_row["predicted_correct"])
            flip_count += int((not left_correct) and right_correct)
            regression_count += int(left_correct and (not right_correct))
            task_gain_total += float(right_row["task_score"]) - float(left_row["task_score"])
            proxy_gain_total += float(right_row["task_proxy_score"]) - float(left_row["task_proxy_score"])
            margin_gain_total += float(right_row["final_margin"]) - float(left_row["final_margin"])
        pair_rows.append(
            {
                "left_alias": left_alias,
                "right_alias": right_alias,
                "paired_examples": len(common_ids),
                "flip_count": flip_count,
                "regression_count": regression_count,
                "flip_count_delta": flip_count - regression_count,
                "mean_task_gain": task_gain_total / max(1, len(common_ids)),
                "mean_proxy_gain": proxy_gain_total / max(1, len(common_ids)),
                "mean_margin_gain": margin_gain_total / max(1, len(common_ids)),
            }
        )

    flip_rows: list[dict[str, Any]] = []
    flip_left_alias, flip_right_alias = _resolve_flip_compare_aliases(config)
    if flip_left_alias in selected_rows_by_alias and flip_right_alias in selected_rows_by_alias:
        common_flip_ids = sorted(set(selected_rows_by_alias[flip_left_alias]) & set(selected_rows_by_alias[flip_right_alias]))
    else:
        common_flip_ids = []
    for example_id in common_flip_ids:
        base_row = selected_rows_by_alias[flip_left_alias][example_id]
        candidate_row = selected_rows_by_alias[flip_right_alias][example_id]
        if (not bool(base_row["predicted_correct"])) and bool(candidate_row["predicted_correct"]):
            flip_rows.append(
                {
                    "example_id": example_id,
                    "left_alias": flip_left_alias,
                    "right_alias": flip_right_alias,
                    "bucket": str(candidate_row.get("screening_bucket", "")),
                    "base_margin": float(base_row["final_margin"]),
                    "candidate_margin": float(candidate_row["final_margin"]),
                    "margin_gain": float(candidate_row["final_margin"]) - float(base_row["final_margin"]),
                    "context_preview": _resolve_context_preview(candidate_row),
                    "gold_text": str(candidate_row.get("gold_text", "")),
                    "base_predicted_label": str(base_row.get("final_predicted_label", "")),
                    "candidate_predicted_label": str(candidate_row.get("final_predicted_label", "")),
                }
            )

    bucket_summary_rows: list[dict[str, Any]] = []
    bucket_keys = sorted(
        {
            str(row.get("screening_bucket", ""))
            for alias_rows in selected_rows_by_alias.values()
            for row in alias_rows.values()
        }
    )
    for alias, alias_rows in selected_rows_by_alias.items():
        for bucket in bucket_keys:
            bucket_rows = [row for row in alias_rows.values() if str(row.get("screening_bucket", "")) == bucket]
            if not bucket_rows:
                continue
            bucket_summary_rows.append(
                {
                    "alias": alias,
                    "bucket": bucket,
                    "examples": len(bucket_rows),
                    "task_score": sum(float(row["task_score"]) for row in bucket_rows) / len(bucket_rows),
                    "task_proxy_score": sum(float(row["task_proxy_score"]) for row in bucket_rows) / len(bucket_rows),
                    "task_margin": sum(float(row["final_margin"]) for row in bucket_rows) / len(bucket_rows),
                }
            )

    arm_summary_path = output_dir / "arm_summary.csv"
    pairwise_path = output_dir / "arm_pairwise_compare.csv"
    flip_cases_path = output_dir / "flip_cases.csv"
    bucket_summary_path = output_dir / "bucket_summary.csv"
    memory_control_gap_path = output_dir / "memory_control_gap.csv"
    write_summary_csv(arm_summary_path, arm_summary_rows)
    write_sanity_plot(output_dir / "summary.svg", arm_summary_rows)
    _write_csv(pairwise_path, pair_rows)
    _write_csv(flip_cases_path, flip_rows)
    _write_csv(bucket_summary_path, bucket_summary_rows)
    memory_left_alias, memory_right_alias = _resolve_memory_control_pair(config)
    _write_csv(
        memory_control_gap_path,
        [
            row
            for row in pair_rows
            if row["left_alias"] == memory_left_alias and row["right_alias"] == memory_right_alias
        ],
    )
    report_title = "Stage C Real Pilot Compare"
    if config is not None:
        report_title = str(config.get("runtime", {}).get("report_title", report_title))
    report_lines = [
        f"# {report_title}",
        "",
        "## Arm Summary",
    ]
    for row in arm_summary_rows:
        report_lines.append(
            f"- {row['alias']}: score={row['best_adapt_task_score']:.4f}, "
            f"proxy={row['best_adapt_task_proxy_score']:.4f}, margin={row['best_adapt_task_margin']:.4f}, "
            f"best_step={row['best_adapt_step']}"
        )
    report_lines.append("")
    report_lines.append("## Pairwise Deltas")
    for row in pair_rows:
        report_lines.append(
            f"- {row['left_alias']} -> {row['right_alias']}: "
            f"flip_count_delta={row['flip_count_delta']}, "
            f"mean_task_gain={row['mean_task_gain']:.4f}, "
            f"mean_margin_gain={row['mean_margin_gain']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## Bucket Summary")
    for row in bucket_summary_rows:
        report_lines.append(
            f"- {row['alias']} / {row['bucket']}: "
            f"task_score={row['task_score']:.4f}, "
            f"proxy={row['task_proxy_score']:.4f}, "
            f"margin={row['task_margin']:.4f}"
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "stage_c_real_pilot_compare",
            "arm_summary_csv": str(arm_summary_path.resolve()),
            "pairwise_compare_csv": str(pairwise_path.resolve()),
            "flip_cases_csv": str(flip_cases_path.resolve()),
            "bucket_summary_csv": str(bucket_summary_path.resolve()),
            "memory_control_gap_csv": str(memory_control_gap_path.resolve()),
            "pair_count": len(pair_rows),
            "flip_cases": len(flip_rows),
        },
    )


def _resolve_branch_fate_min_flip_gain(config: dict[str, Any] | None) -> int:
    if config is None:
        return 2
    return int(config.get("runtime", {}).get("branch_fate_min_flip_gain", 2))


def _resolve_branch_fate_min_alignment_rate(config: dict[str, Any] | None) -> float:
    if config is None:
        return 0.6
    return float(config.get("runtime", {}).get("branch_fate_min_alignment_rate", 0.6))


def _resolve_primary_control_task(config: dict[str, Any] | None) -> bool:
    if config is None:
        return False
    return bool(config.get("runtime", {}).get("primary_control_task", False))


def _content_audit_summary_row(
    *,
    alias: str,
    rows: list[dict[str, Any]],
    primary_metric: str = "task_score",
) -> dict[str, Any]:
    count = max(1, len(rows))
    task_score = sum(float(row["task_score"]) for row in rows) / count
    return {
        "alias": alias,
        "mode": "analysis",
        "run_dir": alias,
        "primary_metric": primary_metric,
        "primary_score": task_score,
        "task_score": task_score,
        "task_proxy_score": sum(float(row["task_proxy_score"]) for row in rows) / count,
        "task_margin": sum(float(row["final_margin"]) for row in rows) / count,
    }


def _append_content_alignment_rows(
    output_rows: list[dict[str, Any]],
    *,
    case_rows: list[dict[str, Any]],
    population: str,
) -> None:
    if population == "all":
        selected = list(case_rows)
    elif population == "base_wrong":
        selected = [row for row in case_rows if not bool(row["base_correct"])]
    elif population == "shared_wrong":
        selected = [row for row in case_rows if not bool(row["shared_correct"])]
    else:
        raise ValueError(f"Unsupported content audit population: {population}")
    if not selected:
        return
    bucket_values = ["ALL", *sorted({str(row["screening_bucket"]) for row in selected if str(row["screening_bucket"])})]
    shared_effect_values = ["ALL", *sorted({str(row["shared_effect_bucket"]) for row in selected})]
    margin_band_values = ["ALL", *sorted({str(row["shared_margin_band"]) for row in selected})]
    for bucket in bucket_values:
        for shared_effect_bucket in shared_effect_values:
            for margin_band in margin_band_values:
                if bucket == "ALL" and shared_effect_bucket == "ALL" and margin_band == "ALL":
                    subset = selected
                else:
                    subset = [
                        row
                        for row in selected
                        if (bucket == "ALL" or str(row["screening_bucket"]) == bucket)
                        and (shared_effect_bucket == "ALL" or str(row["shared_effect_bucket"]) == shared_effect_bucket)
                        and (margin_band == "ALL" or str(row["shared_margin_band"]) == margin_band)
                    ]
                if not subset:
                    continue
                alignment_rows = [row for row in subset if not bool(row["shared_correct"])]
                alignment_examples = len(alignment_rows)
                output_rows.append(
                    {
                        "population": population,
                        "bucket": bucket,
                        "shared_effect_bucket": shared_effect_bucket,
                        "margin_band": margin_band,
                        "examples": len(subset),
                        "alignment_examples": alignment_examples,
                        "content_alignment_rate": (
                            sum(int(row["content_alignment_sign"] > 0) for row in alignment_rows) / alignment_examples
                            if alignment_examples
                            else 0.0
                        ),
                        "weighted_content_alignment": (
                            sum(float(row["weighted_content_alignment"]) for row in alignment_rows) / alignment_examples
                            if alignment_examples
                            else 0.0
                        ),
                        "mean_delta_shared": sum(float(row["delta_shared"]) for row in subset) / len(subset),
                        "mean_delta_candidate_total": sum(float(row["delta_candidate_total"]) for row in subset)
                        / len(subset),
                        "mean_delta_content": sum(float(row["delta_content"]) for row in subset) / len(subset),
                        "mean_delta_branch": sum(float(row["delta_branch"]) for row in subset) / len(subset),
                    }
                )


def run_stage_c_real_pilot_content_audit(
    *,
    config: dict[str, Any] | None,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> None:
    root = Path(input_root).resolve()
    default_dataset_filename = "fixed100.jsonl"
    if config is not None:
        default_dataset_filename = Path(str(config["task"]["dataset_path"])).name
    audit_config = {
        "task": {} if config is None else dict(config.get("task", {})),
        "runtime": {} if config is None else dict(config.get("runtime", {})),
    }
    audit_config["runtime"]["required_arm_aliases"] = ["A", "B", "F", "G"]
    runs, selected_rows_by_alias, _arm_summary_rows = _load_selected_rows_by_alias(
        input_root=root,
        config=audit_config,
        default_pilot_split="fixed100",
        default_dataset_filename=default_dataset_filename,
    )
    common_ids = sorted(
        set(selected_rows_by_alias["A"])
        & set(selected_rows_by_alias["B"])
        & set(selected_rows_by_alias["F"])
        & set(selected_rows_by_alias["G"])
    )
    if dry_run:
        common_ids = common_ids[: min(16, len(common_ids))]
    alpha_grid = _resolve_oracle_alpha_grid(config)

    case_rows: list[dict[str, Any]] = []
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_id in common_ids:
        row_a = selected_rows_by_alias["A"][example_id]
        row_b = selected_rows_by_alias["B"][example_id]
        row_f = selected_rows_by_alias["F"][example_id]
        row_g = selected_rows_by_alias["G"][example_id]
        choices = row_b.get("choices", [])
        candidate_labels = [str(choice["label"]) for choice in choices]
        gold_label = str(row_b["gold_label"])

        scores_a = [float(value) for value in row_a["final_choice_scores"]]
        scores_b = [float(value) for value in row_b["final_choice_scores"]]
        scores_f = [float(value) for value in row_f["final_choice_scores"]]
        scores_g = [float(value) for value in row_g["final_choice_scores"]]
        shared_effect_scores = [right - left for left, right in zip(scores_a, scores_b, strict=True)]
        candidate_total_scores = [right - left for left, right in zip(scores_b, scores_f, strict=True)]
        content_effect_scores = [right - left for left, right in zip(scores_g, scores_f, strict=True)]
        branch_effect_scores = [right - left for left, right in zip(scores_b, scores_g, strict=True)]
        scores_b_plus_content = [
            base_score + content_score
            for base_score, content_score in zip(scores_b, content_effect_scores, strict=True)
        ]

        outcome_a = _compute_choice_outcome(
            choice_scores=scores_a,
            candidate_labels=candidate_labels,
            gold_label=gold_label,
        )
        outcome_b = _compute_choice_outcome(
            choice_scores=scores_b,
            candidate_labels=candidate_labels,
            gold_label=gold_label,
        )
        outcome_f = _compute_choice_outcome(
            choice_scores=scores_f,
            candidate_labels=candidate_labels,
            gold_label=gold_label,
        )
        outcome_g = _compute_choice_outcome(
            choice_scores=scores_g,
            candidate_labels=candidate_labels,
            gold_label=gold_label,
        )
        outcome_b_plus_content = _compute_choice_outcome(
            choice_scores=scores_b_plus_content,
            candidate_labels=candidate_labels,
            gold_label=gold_label,
        )
        oracle_alpha_content = _pick_oracle_alpha_from_anchor(
            anchor_scores=scores_b,
            delta_scores=content_effect_scores,
            candidate_labels=candidate_labels,
            gold_label=gold_label,
            alpha_grid=alpha_grid,
        )

        base_correct = bool(outcome_a["predicted_correct"])
        shared_correct = bool(outcome_b["predicted_correct"])
        f_correct = bool(outcome_f["predicted_correct"])
        g_correct = bool(outcome_g["predicted_correct"])
        b_plus_content_correct = bool(outcome_b_plus_content["predicted_correct"])
        best_of_bf_correct = f_correct or shared_correct
        best_of_b_plus_content_correct = b_plus_content_correct or shared_correct
        best_of_bf_source = "F" if f_correct and (not shared_correct) else "B"
        best_of_b_plus_content_source = (
            "B_plus_content"
            if b_plus_content_correct and (not shared_correct)
            else "B"
        )

        delta_shared = float(outcome_b["final_margin"] - outcome_a["final_margin"])
        delta_candidate_total = float(outcome_f["final_margin"] - outcome_b["final_margin"])
        delta_content = float(outcome_f["final_margin"] - outcome_g["final_margin"])
        delta_branch = float(outcome_g["final_margin"] - outcome_b["final_margin"])
        needed_fix = float(-outcome_b["final_margin"])
        content_push = float(outcome_b_plus_content["final_margin"] - outcome_b["final_margin"])
        if shared_correct:
            content_alignment_sign = 0
            weighted_content_alignment = 0.0
        else:
            if content_push > 0.0:
                content_alignment_sign = 1
            elif content_push < 0.0:
                content_alignment_sign = -1
            else:
                content_alignment_sign = 0
            weighted_content_alignment = _safe_weighted_alignment(needed_fix, content_push)

        case_payload = {
            "example_id": example_id,
            "benchmark_id": str(row_b.get("benchmark_id", "")),
            "screening_bucket": _resolve_first_nonempty_field(
                row_b,
                row_f,
                row_g,
                row_a,
                field="screening_bucket",
            ),
            "shared_effect_bucket": _resolve_shared_effect_bucket(base_correct, shared_correct),
            "shared_margin_band": _resolve_margin_band(float(outcome_b["final_margin"])),
            "base_correct": base_correct,
            "shared_correct": shared_correct,
            "F_correct": f_correct,
            "G_correct": g_correct,
            "b_plus_content_correct": b_plus_content_correct,
            "best_of_bf_correct": bool(best_of_bf_correct),
            "best_of_bf_source": best_of_bf_source,
            "best_of_b_plus_content_correct": bool(best_of_b_plus_content_correct),
            "best_of_b_plus_content_source": best_of_b_plus_content_source,
            "oracle_alpha_content": float(oracle_alpha_content["alpha"]),
            "oracle_alpha_content_correct": bool(oracle_alpha_content["predicted_correct"]),
            "oracle_alpha_content_selection_reason": str(oracle_alpha_content["selection_reason"]),
            "margin_A": float(outcome_a["final_margin"]),
            "margin_B": float(outcome_b["final_margin"]),
            "margin_F": float(outcome_f["final_margin"]),
            "margin_G": float(outcome_g["final_margin"]),
            "margin_B_plus_content": float(outcome_b_plus_content["final_margin"]),
            "delta_shared": delta_shared,
            "delta_candidate_total": delta_candidate_total,
            "delta_content": delta_content,
            "delta_branch": delta_branch,
            "needed_fix": needed_fix,
            "content_push_alpha1": content_push,
            "content_alignment_sign": int(content_alignment_sign),
            "weighted_content_alignment": float(weighted_content_alignment),
            "base_predicted_label": str(outcome_a["predicted_label"]),
            "shared_predicted_label": str(outcome_b["predicted_label"]),
            "F_predicted_label": str(outcome_f["predicted_label"]),
            "G_predicted_label": str(outcome_g["predicted_label"]),
            "b_plus_content_predicted_label": str(outcome_b_plus_content["predicted_label"]),
            "oracle_alpha_content_predicted_label": str(oracle_alpha_content["predicted_label"]),
            "base_choice_scores": json.dumps(scores_a),
            "shared_choice_scores": json.dumps(scores_b),
            "F_choice_scores": json.dumps(scores_f),
            "G_choice_scores": json.dumps(scores_g),
            "content_effect_scores": json.dumps(content_effect_scores),
            "branch_effect_scores": json.dumps(branch_effect_scores),
            "shared_effect_scores": json.dumps(shared_effect_scores),
            "candidate_total_scores": json.dumps(candidate_total_scores),
            "b_plus_content_choice_scores": json.dumps(scores_b_plus_content),
            "oracle_alpha_content_choice_scores": json.dumps(oracle_alpha_content["choice_scores"]),
            "context_preview": _resolve_context_preview(row_b),
            "gold_text": str(row_b.get("gold_text", "")),
        }
        case_rows.append(case_payload)
        bucket_rows[str(case_payload["screening_bucket"])].append(case_payload)

    count = max(1, len(case_rows))
    summary_rows = [
        _content_audit_summary_row(alias="A", rows=[{
            "task_score": float(row["base_correct"]),
            "task_proxy_score": float(selected_rows_by_alias["A"][row["example_id"]]["task_proxy_score"]),
            "final_margin": float(row["margin_A"]),
        } for row in case_rows]),
        _content_audit_summary_row(alias="B", rows=[{
            "task_score": float(row["shared_correct"]),
            "task_proxy_score": float(selected_rows_by_alias["B"][row["example_id"]]["task_proxy_score"]),
            "final_margin": float(row["margin_B"]),
        } for row in case_rows]),
        _content_audit_summary_row(alias="F", rows=[{
            "task_score": float(row["F_correct"]),
            "task_proxy_score": float(selected_rows_by_alias["F"][row["example_id"]]["task_proxy_score"]),
            "final_margin": float(row["margin_F"]),
        } for row in case_rows]),
        _content_audit_summary_row(alias="G", rows=[{
            "task_score": float(row["G_correct"]),
            "task_proxy_score": float(selected_rows_by_alias["G"][row["example_id"]]["task_proxy_score"]),
            "final_margin": float(row["margin_G"]),
        } for row in case_rows]),
        _content_audit_summary_row(alias="B_plus_content", rows=[{
            "task_score": float(row["b_plus_content_correct"]),
            "task_proxy_score": _compute_choice_outcome(
                choice_scores=json.loads(row["b_plus_content_choice_scores"]),
                candidate_labels=[str(choice["label"]) for choice in selected_rows_by_alias["B"][row["example_id"]]["choices"]],
                gold_label=str(selected_rows_by_alias["B"][row["example_id"]]["gold_label"]),
            )["task_proxy_score"],
            "final_margin": float(row["margin_B_plus_content"]),
        } for row in case_rows]),
        _content_audit_summary_row(alias="oracle_best_of_BF", rows=[{
            "task_score": float(row["best_of_bf_correct"]),
            "task_proxy_score": (
                float(selected_rows_by_alias["F"][row["example_id"]]["task_proxy_score"])
                if row["best_of_bf_source"] == "F"
                else float(selected_rows_by_alias["B"][row["example_id"]]["task_proxy_score"])
            ),
            "final_margin": (
                float(row["margin_F"])
                if row["best_of_bf_source"] == "F"
                else float(row["margin_B"])
            ),
        } for row in case_rows]),
        _content_audit_summary_row(alias="oracle_best_of_B_plus_content", rows=[{
            "task_score": float(row["best_of_b_plus_content_correct"]),
            "task_proxy_score": (
                _compute_choice_outcome(
                    choice_scores=json.loads(row["b_plus_content_choice_scores"]),
                    candidate_labels=[str(choice["label"]) for choice in selected_rows_by_alias["B"][row["example_id"]]["choices"]],
                    gold_label=str(selected_rows_by_alias["B"][row["example_id"]]["gold_label"]),
                )["task_proxy_score"]
                if row["best_of_b_plus_content_source"] == "B_plus_content"
                else float(selected_rows_by_alias["B"][row["example_id"]]["task_proxy_score"])
            ),
            "final_margin": (
                float(row["margin_B_plus_content"])
                if row["best_of_b_plus_content_source"] == "B_plus_content"
                else float(row["margin_B"])
            ),
        } for row in case_rows]),
        _content_audit_summary_row(alias="oracle_per_case_alpha_content", rows=[{
            "task_score": float(row["oracle_alpha_content_correct"]),
            "task_proxy_score": _compute_choice_outcome(
                choice_scores=json.loads(row["oracle_alpha_content_choice_scores"]),
                candidate_labels=[str(choice["label"]) for choice in selected_rows_by_alias["B"][row["example_id"]]["choices"]],
                gold_label=str(selected_rows_by_alias["B"][row["example_id"]]["gold_label"]),
            )["task_proxy_score"],
            "final_margin": float(
                _compute_choice_outcome(
                    choice_scores=json.loads(row["oracle_alpha_content_choice_scores"]),
                    candidate_labels=[str(choice["label"]) for choice in selected_rows_by_alias["B"][row["example_id"]]["choices"]],
                    gold_label=str(selected_rows_by_alias["B"][row["example_id"]]["gold_label"]),
                )["final_margin"]
            ),
        } for row in case_rows]),
    ]

    bucket_summary_rows: list[dict[str, Any]] = []
    for bucket in sorted(bucket_rows):
        rows = bucket_rows[bucket]
        bucket_count = max(1, len(rows))
        bucket_summary_rows.append(
            {
                "bucket": bucket,
                "examples": len(rows),
                "B_task_score": sum(float(row["shared_correct"]) for row in rows) / bucket_count,
                "F_task_score": sum(float(row["F_correct"]) for row in rows) / bucket_count,
                "G_task_score": sum(float(row["G_correct"]) for row in rows) / bucket_count,
                "B_plus_content_task_score": sum(float(row["b_plus_content_correct"]) for row in rows) / bucket_count,
                "oracle_best_of_BF_task_score": sum(float(row["best_of_bf_correct"]) for row in rows) / bucket_count,
                "oracle_best_of_B_plus_content_task_score": sum(
                    float(row["best_of_b_plus_content_correct"]) for row in rows
                )
                / bucket_count,
                "oracle_per_case_alpha_content_task_score": sum(
                    float(row["oracle_alpha_content_correct"]) for row in rows
                )
                / bucket_count,
                "best_of_BF_flip_gain": sum(
                    int((not bool(row["shared_correct"])) and bool(row["best_of_bf_correct"])) for row in rows
                ),
                "best_of_B_plus_content_flip_gain": sum(
                    int((not bool(row["shared_correct"])) and bool(row["best_of_b_plus_content_correct"]))
                    for row in rows
                ),
                "oracle_per_case_alpha_content_flip_gain": sum(
                    int((not bool(row["shared_correct"])) and bool(row["oracle_alpha_content_correct"]))
                    for row in rows
                ),
            }
        )

    alignment_summary_rows: list[dict[str, Any]] = []
    for population in ("all", "base_wrong", "shared_wrong"):
        _append_content_alignment_rows(
            alignment_summary_rows,
            case_rows=case_rows,
            population=population,
        )

    content_effect_path = output_dir / "content_effect_case_dump.csv"
    content_alignment_path = output_dir / "content_alignment_summary.csv"
    content_oracle_summary_path = output_dir / "content_oracle_summary.csv"
    content_oracle_by_bucket_path = output_dir / "content_oracle_by_bucket.csv"
    content_oracle_case_deltas_path = output_dir / "content_oracle_case_deltas.csv"
    write_summary_csv(content_oracle_summary_path, summary_rows)
    write_sanity_plot(output_dir / "summary.svg", summary_rows)
    _write_csv(content_effect_path, case_rows)
    _write_csv(content_alignment_path, alignment_summary_rows)
    _write_csv(content_oracle_by_bucket_path, bucket_summary_rows)
    _write_csv(content_oracle_case_deltas_path, case_rows)

    best_of_bf_flip_gain = sum(
        int((not bool(row["shared_correct"])) and bool(row["best_of_bf_correct"])) for row in case_rows
    )
    best_of_b_plus_content_flip_gain = sum(
        int((not bool(row["shared_correct"])) and bool(row["best_of_b_plus_content_correct"])) for row in case_rows
    )
    oracle_per_case_alpha_content_flip_gain = sum(
        int((not bool(row["shared_correct"])) and bool(row["oracle_alpha_content_correct"])) for row in case_rows
    )
    shared_wrong_rows = [row for row in case_rows if not bool(row["shared_correct"])]
    shared_wrong_count = len(shared_wrong_rows)
    content_alignment_rate_shared_wrong = (
        sum(int(row["content_alignment_sign"] > 0) for row in shared_wrong_rows) / shared_wrong_count
        if shared_wrong_count
        else 0.0
    )
    weighted_alignment_shared_wrong = (
        sum(float(row["weighted_content_alignment"]) for row in shared_wrong_rows) / shared_wrong_count
        if shared_wrong_count
        else 0.0
    )
    min_flip_gain = _resolve_branch_fate_min_flip_gain(config)
    min_alignment_rate = _resolve_branch_fate_min_alignment_rate(config)
    branch_fate_payload = {
        "analysis_mode": "stage_c_real_pilot_content_audit",
        "examples": len(case_rows),
        "primary_control_task": _resolve_primary_control_task(config),
        "summary_csv": str(content_oracle_summary_path.resolve()),
        "content_effect_case_dump_csv": str(content_effect_path.resolve()),
        "content_alignment_summary_csv": str(content_alignment_path.resolve()),
        "content_oracle_by_bucket_csv": str(content_oracle_by_bucket_path.resolve()),
        "content_oracle_case_deltas_csv": str(content_oracle_case_deltas_path.resolve()),
        "best_of_BF_flip_gain": best_of_bf_flip_gain,
        "best_of_B_plus_content_flip_gain": best_of_b_plus_content_flip_gain,
        "oracle_per_case_alpha_content_flip_gain": oracle_per_case_alpha_content_flip_gain,
        "content_alignment_rate_shared_wrong": content_alignment_rate_shared_wrong,
        "weighted_content_alignment_shared_wrong": weighted_alignment_shared_wrong,
        "branch_fate_min_flip_gain": min_flip_gain,
        "branch_fate_min_alignment_rate": min_alignment_rate,
        "continue_candidate_branch": bool(
            best_of_b_plus_content_flip_gain >= min_flip_gain
            and content_alignment_rate_shared_wrong >= min_alignment_rate
        ),
    }
    write_json(output_dir / "branch_fate_metrics.json", branch_fate_payload)
    report_title = "Stage C Real Pilot Content Audit"
    if config is not None:
        report_title = str(config.get("runtime", {}).get("report_title", report_title))
    report_lines = [
        f"# {report_title}",
        "",
        f"- examples={len(case_rows)}",
        f"- best_of_BF_flip_gain={best_of_bf_flip_gain}",
        f"- best_of_B_plus_content_flip_gain={best_of_b_plus_content_flip_gain}",
        f"- oracle_per_case_alpha_content_flip_gain={oracle_per_case_alpha_content_flip_gain}",
        f"- content_alignment_rate_shared_wrong={content_alignment_rate_shared_wrong:.4f}",
        f"- weighted_content_alignment_shared_wrong={weighted_alignment_shared_wrong:.4f}",
        f"- continue_candidate_branch={branch_fate_payload['continue_candidate_branch']}",
        "",
        "## Summary",
    ]
    for row in summary_rows:
        report_lines.append(
            f"- {row['alias']}: task_score={row['task_score']:.4f}, "
            f"proxy={row['task_proxy_score']:.4f}, margin={row['task_margin']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## By Bucket")
    for row in bucket_summary_rows:
        report_lines.append(
            f"- {row['bucket']}: B={row['B_task_score']:.4f}, "
            f"F={row['F_task_score']:.4f}, G={row['G_task_score']:.4f}, "
            f"B_plus_content={row['B_plus_content_task_score']:.4f}, "
            f"oracle_alpha={row['oracle_per_case_alpha_content_task_score']:.4f}"
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    write_json(output_dir / "metrics.json", branch_fate_payload)


def _resolve_oracle_alpha_grid(config: dict[str, Any] | None) -> list[float]:
    raw_grid = None if config is None else config.get("runtime", {}).get("stage_c_oracle_alpha_grid")
    if raw_grid is None:
        return list(_DEFAULT_ORACLE_ALPHA_GRID)
    if isinstance(raw_grid, str):
        values = [float(part.strip()) for part in raw_grid.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw_grid]
    ordered: list[float] = []
    seen: set[float] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _compute_choice_outcome(
    *,
    choice_scores: list[float],
    candidate_labels: list[str],
    gold_label: str,
) -> dict[str, Any]:
    if not candidate_labels:
        raise ValueError("candidate_labels must not be empty.")
    if len(choice_scores) != len(candidate_labels):
        raise ValueError("choice_scores and candidate_labels must have the same length.")
    gold_index = candidate_labels.index(gold_label)
    predicted_index = max(range(len(choice_scores)), key=lambda index: choice_scores[index])
    predicted_label = candidate_labels[predicted_index]
    score_tensor = torch.tensor(choice_scores, dtype=torch.float32)
    probabilities = torch.softmax(score_tensor, dim=0)
    return {
        "gold_index": gold_index,
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "predicted_correct": bool(predicted_label == gold_label),
        "task_score": float(predicted_label == gold_label),
        "task_proxy_score": float(probabilities[gold_index].item()),
        "final_margin": _score_margin(choice_scores, gold_index=gold_index),
        "choice_scores": [float(value) for value in choice_scores],
    }


def _compute_scores_from_alpha(
    *,
    base_scores: list[float],
    residual_scores: list[float],
    alpha: float,
) -> list[float]:
    return [
        float(base_score + (alpha * residual_score))
        for base_score, residual_score in zip(base_scores, residual_scores, strict=True)
    ]


def _score_margin(choice_scores: list[float], *, gold_index: int) -> float:
    if not choice_scores:
        return 0.0
    competitor_indices = [index for index in range(len(choice_scores)) if index != gold_index]
    if not competitor_indices:
        return float(choice_scores[gold_index])
    competitor_index = max(competitor_indices, key=lambda index: choice_scores[index])
    return float(choice_scores[gold_index] - choice_scores[competitor_index])


def _pick_oracle_alpha_case(
    candidate_row: dict[str, Any],
    *,
    alpha_grid: list[float],
) -> dict[str, Any]:
    base_scores = [float(value) for value in candidate_row["base_choice_scores"]]
    residual_scores = [float(value) for value in candidate_row["memory_residual_scores"]]
    choices = candidate_row.get("choices", [])
    candidate_labels = [str(choice["label"]) for choice in choices]
    gold_label = str(candidate_row["gold_label"])
    alpha_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        choice_scores = _compute_scores_from_alpha(
            base_scores=base_scores,
            residual_scores=residual_scores,
            alpha=float(alpha),
        )
        outcome = _compute_choice_outcome(
            choice_scores=choice_scores,
            candidate_labels=candidate_labels,
            gold_label=gold_label,
        )
        alpha_rows.append(
            {
                "alpha": float(alpha),
                **outcome,
            }
        )
    correct_rows = [row for row in alpha_rows if bool(row["predicted_correct"])]
    if correct_rows:
        best_row = min(
            correct_rows,
            key=lambda row: (
                abs(float(row["alpha"])),
                0 if float(row["alpha"]) >= 0.0 else 1,
                abs(float(row["alpha"]) - 1.0),
            ),
        )
        selection_reason = "correct_min_abs"
    else:
        best_row = max(
            alpha_rows,
            key=lambda row: (
                float(row["final_margin"]),
                -abs(float(row["alpha"])),
                1 if float(row["alpha"]) >= 0.0 else 0,
                -abs(float(row["alpha"]) - 1.0),
            ),
        )
        selection_reason = "best_margin_when_all_wrong"
    return {
        **best_row,
        "selection_reason": selection_reason,
        "candidate_labels": candidate_labels,
        "gold_label": gold_label,
    }


def _pick_oracle_alpha_from_anchor(
    *,
    anchor_scores: list[float],
    delta_scores: list[float],
    candidate_labels: list[str],
    gold_label: str,
    alpha_grid: list[float],
) -> dict[str, Any]:
    alpha_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        choice_scores = _compute_scores_from_alpha(
            base_scores=anchor_scores,
            residual_scores=delta_scores,
            alpha=float(alpha),
        )
        outcome = _compute_choice_outcome(
            choice_scores=choice_scores,
            candidate_labels=candidate_labels,
            gold_label=gold_label,
        )
        alpha_rows.append(
            {
                "alpha": float(alpha),
                **outcome,
            }
        )
    correct_rows = [row for row in alpha_rows if bool(row["predicted_correct"])]
    if correct_rows:
        best_row = min(
            correct_rows,
            key=lambda row: (
                abs(float(row["alpha"])),
                0 if float(row["alpha"]) >= 0.0 else 1,
                abs(float(row["alpha"]) - 1.0),
            ),
        )
        selection_reason = "correct_min_abs"
    else:
        best_row = max(
            alpha_rows,
            key=lambda row: (
                float(row["final_margin"]),
                -abs(float(row["alpha"])),
                1 if float(row["alpha"]) >= 0.0 else 0,
                -abs(float(row["alpha"]) - 1.0),
            ),
        )
        selection_reason = "best_margin_when_all_wrong"
    return {
        **best_row,
        "selection_reason": selection_reason,
        "candidate_labels": list(candidate_labels),
        "gold_label": gold_label,
    }


def _resolve_margin_band(margin_value: float) -> str:
    if margin_value < 0.0:
        return "near_threshold" if margin_value >= -0.1 else "far_negative"
    return "non_negative"


def _resolve_shared_effect_bucket(base_correct: bool, shared_correct: bool) -> str:
    if (not base_correct) and shared_correct:
        return "shared_helped"
    if base_correct and (not shared_correct):
        return "shared_hurt"
    return "shared_neutral"


def _safe_weighted_alignment(needed_fix: float, content_push: float) -> float:
    return float((content_push * needed_fix) / (abs(needed_fix) + 1e-8))


def _resolve_first_nonempty_field(*rows: dict[str, Any], field: str) -> str:
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def run_stage_c_real_pilot_oracle_audit(
    *,
    config: dict[str, Any] | None,
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> None:
    root = Path(input_root).resolve()
    default_dataset_filename = "fixed100.jsonl"
    if config is not None:
        default_dataset_filename = Path(str(config["task"]["dataset_path"])).name
    oracle_config = {
        "task": {} if config is None else dict(config.get("task", {})),
        "runtime": {} if config is None else dict(config.get("runtime", {})),
    }
    oracle_config["runtime"]["required_arm_aliases"] = ["A", "C"]
    runs, selected_rows_by_alias, _arm_summary_rows = _load_selected_rows_by_alias(
        input_root=root,
        config=oracle_config,
        default_pilot_split="fixed100",
        default_dataset_filename=default_dataset_filename,
    )
    if "A" not in selected_rows_by_alias or "C" not in selected_rows_by_alias:
        raise ValueError("Oracle audit requires at least A=base_only and C=candidate_conditioned arms.")
    alpha_grid = _resolve_oracle_alpha_grid(config)
    common_ids = sorted(set(selected_rows_by_alias["A"]) & set(selected_rows_by_alias["C"]))
    if dry_run:
        common_ids = common_ids[: min(16, len(common_ids))]

    case_rows: list[dict[str, Any]] = []
    summary_counts = {
        "A": 0.0,
        "C": 0.0,
        "oracle_best_of_two": 0.0,
        "oracle_per_case_alpha": 0.0,
    }
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_id in common_ids:
        base_row = selected_rows_by_alias["A"][example_id]
        candidate_row = selected_rows_by_alias["C"][example_id]
        best_of_two_correct = bool(candidate_row["predicted_correct"]) and (not bool(base_row["predicted_correct"]))
        oracle_best_correct = best_of_two_correct or bool(base_row["predicted_correct"])
        oracle_alpha = _pick_oracle_alpha_case(candidate_row, alpha_grid=alpha_grid)
        summary_counts["A"] += float(base_row["task_score"])
        summary_counts["C"] += float(candidate_row["task_score"])
        summary_counts["oracle_best_of_two"] += float(oracle_best_correct)
        summary_counts["oracle_per_case_alpha"] += float(oracle_alpha["predicted_correct"])
        case_payload = {
            "example_id": example_id,
            "screening_bucket": str(candidate_row.get("screening_bucket", "")),
            "base_correct": bool(base_row["predicted_correct"]),
            "candidate_correct": bool(candidate_row["predicted_correct"]),
            "oracle_best_of_two_correct": bool(oracle_best_correct),
            "oracle_best_of_two_source": "C"
            if best_of_two_correct
            else "A",
            "oracle_alpha": float(oracle_alpha["alpha"]),
            "oracle_alpha_correct": bool(oracle_alpha["predicted_correct"]),
            "oracle_alpha_selection_reason": str(oracle_alpha["selection_reason"]),
            "base_margin": float(base_row["final_margin"]),
            "candidate_margin": float(candidate_row["final_margin"]),
            "oracle_alpha_margin": float(oracle_alpha["final_margin"]),
            "base_predicted_label": str(base_row["final_predicted_label"]),
            "candidate_predicted_label": str(candidate_row["final_predicted_label"]),
            "oracle_alpha_predicted_label": str(oracle_alpha["predicted_label"]),
            "base_choice_scores": json.dumps(base_row["final_choice_scores"]),
            "candidate_residual_scores": json.dumps(candidate_row["memory_residual_scores"]),
            "oracle_alpha_choice_scores": json.dumps(oracle_alpha["choice_scores"]),
            "gold_label": str(candidate_row["gold_label"]),
            "context_preview": _resolve_context_preview(candidate_row),
            "gold_text": str(candidate_row.get("gold_text", "")),
        }
        case_rows.append(case_payload)
        bucket_rows[str(candidate_row.get("screening_bucket", ""))].append(case_payload)

    total_examples = max(1, len(common_ids))
    summary_rows = [
        {
            "alias": alias,
            "mode": "analysis",
            "run_dir": alias,
            "primary_metric": "task_score",
            "primary_score": summary_counts[alias] / total_examples,
            "task_score": summary_counts[alias] / total_examples,
        }
        for alias in ("A", "C", "oracle_best_of_two", "oracle_per_case_alpha")
    ]
    oracle_best_gain = int(summary_counts["oracle_best_of_two"] - summary_counts["A"])
    oracle_alpha_gain = int(summary_counts["oracle_per_case_alpha"] - summary_counts["A"])
    bucket_summary_rows: list[dict[str, Any]] = []
    for bucket in sorted(bucket_rows):
        rows = bucket_rows[bucket]
        count = max(1, len(rows))
        bucket_summary_rows.append(
            {
                "bucket": bucket,
                "examples": len(rows),
                "base_task_score": sum(float(row["base_correct"]) for row in rows) / count,
                "candidate_task_score": sum(float(row["candidate_correct"]) for row in rows) / count,
                "oracle_best_of_two_task_score": sum(float(row["oracle_best_of_two_correct"]) for row in rows) / count,
                "oracle_per_case_alpha_task_score": sum(float(row["oracle_alpha_correct"]) for row in rows) / count,
                "oracle_best_of_two_flip_gain": sum(
                    int((not bool(row["base_correct"])) and bool(row["oracle_best_of_two_correct"]))
                    for row in rows
                ),
                "oracle_per_case_alpha_flip_gain": sum(
                    int((not bool(row["base_correct"])) and bool(row["oracle_alpha_correct"]))
                    for row in rows
                ),
            }
        )

    oracle_summary_path = output_dir / "oracle_summary.csv"
    oracle_by_bucket_path = output_dir / "oracle_by_bucket.csv"
    oracle_case_deltas_path = output_dir / "oracle_case_deltas.csv"
    write_summary_csv(oracle_summary_path, summary_rows)
    write_sanity_plot(output_dir / "summary.svg", summary_rows)
    _write_csv(oracle_by_bucket_path, bucket_summary_rows)
    _write_csv(oracle_case_deltas_path, case_rows)
    report_lines = [
        "# Story Cloze Real Pilot Oracle Audit",
        "",
        f"- examples={len(common_ids)}",
        f"- oracle_best_of_two_flip_gain={oracle_best_gain}",
        f"- oracle_per_case_alpha_flip_gain={oracle_alpha_gain}",
        "",
        "## Summary",
    ]
    for row in summary_rows:
        report_lines.append(f"- {row['alias']}: task_score={row['task_score']:.4f}")
    report_lines.append("")
    report_lines.append("## By Bucket")
    for row in bucket_summary_rows:
        report_lines.append(
            f"- {row['bucket']}: base={row['base_task_score']:.4f}, "
            f"candidate={row['candidate_task_score']:.4f}, "
            f"oracle_best={row['oracle_best_of_two_task_score']:.4f}, "
            f"oracle_alpha={row['oracle_per_case_alpha_task_score']:.4f}"
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "stage_c_real_pilot_oracle_audit",
            "examples": len(common_ids),
            "oracle_summary_csv": str(oracle_summary_path.resolve()),
            "oracle_by_bucket_csv": str(oracle_by_bucket_path.resolve()),
            "oracle_case_deltas_csv": str(oracle_case_deltas_path.resolve()),
            "oracle_best_of_two_flip_gain": oracle_best_gain,
            "oracle_per_case_alpha_flip_gain": oracle_alpha_gain,
        },
    )
