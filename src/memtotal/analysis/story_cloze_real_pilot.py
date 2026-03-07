from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    gold_index = candidate_labels.index(gold_label)
    alpha_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        choice_scores = _compute_scores_from_alpha(
            base_scores=base_scores,
            residual_scores=residual_scores,
            alpha=float(alpha),
        )
        predicted_index = max(range(len(choice_scores)), key=lambda index: choice_scores[index])
        predicted_label = candidate_labels[predicted_index]
        alpha_rows.append(
            {
                "alpha": float(alpha),
                "choice_scores": choice_scores,
                "predicted_index": predicted_index,
                "predicted_label": predicted_label,
                "predicted_correct": bool(predicted_label == gold_label),
                "final_margin": _score_margin(choice_scores, gold_index=gold_index),
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
