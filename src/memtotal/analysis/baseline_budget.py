from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from memtotal.analysis.reporting import write_sanity_plot, write_summary_csv
from memtotal.baselines.budgeting import build_baseline_budget_fields
from memtotal.utils.io import write_json
from memtotal.utils.repro import SUPPORTED_BACKBONES


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_snapshot_config(run_dir: Path) -> dict[str, Any]:
    snapshot_path = run_dir / "config.snapshot.yaml"
    if not snapshot_path.exists():
        return {}
    return yaml.safe_load(snapshot_path.read_text()) or {}


def _infer_baseline_family(metrics: dict[str, Any], config: dict[str, Any]) -> str | None:
    if metrics.get("baseline_family"):
        return str(metrics["baseline_family"])
    if metrics.get("mode") == "memgen_adapter":
        return "memgen"
    baseline_cfg = config.get("baseline", {})
    if baseline_cfg.get("family"):
        return str(baseline_cfg["family"])
    if baseline_cfg.get("name") == "memgen":
        return "memgen"
    return None


def _infer_baseline_mode(metrics: dict[str, Any], config: dict[str, Any], family: str) -> str:
    if metrics.get("baseline_mode"):
        return str(metrics["baseline_mode"])
    baseline_cfg = config.get("baseline", {})
    if baseline_cfg.get("mode"):
        return str(baseline_cfg["mode"])
    if family == "memgen":
        return "external_eval"
    return "unknown"


def collect_baseline_budget_rows(
    input_root: str | Path,
    *,
    baseline_families: set[str] | None = None,
) -> list[dict[str, object]]:
    root = Path(input_root).resolve()
    rows: list[dict[str, object]] = []
    requested_families = baseline_families or {"prompting", "meta_prompting", "adapter", "rag"}
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        metrics = _load_json(metrics_path)
        config = _load_snapshot_config(run_dir)
        run_info_path = run_dir / "run_info.json"
        run_info = _load_json(run_info_path) if run_info_path.exists() else {}
        baseline_family = _infer_baseline_family(metrics, config)
        if baseline_family is None or baseline_family not in requested_families:
            continue
        baseline_mode = _infer_baseline_mode(metrics, config, baseline_family)
        budget_fields = build_baseline_budget_fields(
            config=config or {"baseline": {}, "runtime": {}},
            baseline_family=baseline_family,
            baseline_mode=baseline_mode,
            support_examples=metrics.get("support_examples"),
            train_steps=metrics.get("train_steps"),
            trainable_parameter_count=metrics.get("trainable_parameter_count"),
        )
        rows.append(
            {
                "run_dir": str(run_dir),
                "mode": metrics.get("mode", "unknown"),
                "backbone": run_info.get("backbone", metrics.get("backbone", "unknown")),
                "task_name": run_info.get("task_name", metrics.get("task_name", "unknown")),
                "benchmark_id": run_info.get("benchmark_id", metrics.get("benchmark_id")),
                "smoke_subset": run_info.get("smoke_subset", metrics.get("smoke_subset")),
                "baseline_family": baseline_family,
                "baseline_mode": baseline_mode,
                "accuracy": metrics.get("accuracy"),
                "compute_reward": metrics.get("compute_reward"),
                **budget_fields,
            }
        )
    return rows


def _row_issues(row: dict[str, object]) -> list[str]:
    issues: list[str] = []
    backbone = str(row.get("backbone", ""))
    if backbone not in SUPPORTED_BACKBONES:
        issues.append(f"unsupported_backbone:{backbone or 'missing'}")
    if not str(row.get("baseline_mode", "")):
        issues.append("missing_baseline_mode")
    support_examples = row.get("support_examples")
    train_steps = row.get("train_steps")
    trainable_parameter_count = row.get("trainable_parameter_count")
    if support_examples is None or int(support_examples) < 0:
        issues.append("invalid_support_examples")
    if train_steps is None or int(train_steps) < 0:
        issues.append("invalid_train_steps")
    if trainable_parameter_count is None or int(trainable_parameter_count) < 0:
        issues.append("invalid_trainable_parameter_count")
    family = str(row.get("baseline_family", ""))
    if family == "adapter":
        if int(support_examples or 0) <= 0:
            issues.append("adapter_requires_support_examples")
        if int(train_steps or 0) <= 0:
            issues.append("adapter_requires_train_steps")
        if int(trainable_parameter_count or 0) <= 0:
            issues.append("adapter_requires_trainable_parameters")
    if family in {"prompting", "meta_prompting", "rag"}:
        if int(train_steps or 0) != 0:
            issues.append("prompt_baselines_must_have_zero_train_steps")
        if int(trainable_parameter_count or 0) != 0:
            issues.append("prompt_baselines_must_have_zero_trainable_parameters")
    return issues


def audit_baseline_budget_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    coverage: dict[tuple[str, str, str, str, str], set[str]] = {}
    for row in rows:
        coverage_key = (
            str(row.get("mode", "")),
            str(row.get("baseline_family", "")),
            str(row.get("baseline_mode", "")),
            str(row.get("task_name", "")),
            str(row.get("smoke_subset", "")),
        )
        coverage.setdefault(coverage_key, set()).add(str(row.get("backbone", "")))

    issues: list[dict[str, object]] = []
    audited_rows: list[dict[str, object]] = []
    for row in rows:
        row_issues = _row_issues(row)
        coverage_key = (
            str(row.get("mode", "")),
            str(row.get("baseline_family", "")),
            str(row.get("baseline_mode", "")),
            str(row.get("task_name", "")),
            str(row.get("smoke_subset", "")),
        )
        missing_backbones = sorted(set(SUPPORTED_BACKBONES) - coverage.get(coverage_key, set()))
        if missing_backbones:
            row_issues.append(f"missing_backbone_pair:{','.join(missing_backbones)}")
        audited_row = dict(row)
        audited_row["budget_ok"] = 1.0 if not row_issues else 0.0
        audited_row["primary_metric"] = "budget_ok"
        audited_row["primary_score"] = audited_row["budget_ok"]
        audited_row["budget_issue_count"] = len(row_issues)
        audited_row["budget_issues"] = ";".join(row_issues)
        audited_rows.append(audited_row)
        if row_issues:
            issues.append(
                {
                    "run_dir": row["run_dir"],
                    "baseline_family": row["baseline_family"],
                    "baseline_mode": row["baseline_mode"],
                    "mode": row["mode"],
                    "backbone": row["backbone"],
                    "task_name": row["task_name"],
                    "smoke_subset": row["smoke_subset"],
                    "issues": row_issues,
                }
            )
    return audited_rows, issues


def run_baseline_budget_audit(
    *,
    config: dict[str, Any],
    output_dir: Path,
    input_root: str | Path,
    dry_run: bool,
) -> None:
    analysis_cfg = config.get("analysis", {})
    baseline_families = set(analysis_cfg.get("baseline_families", ["prompting", "meta_prompting", "adapter", "rag"]))
    rows = collect_baseline_budget_rows(input_root, baseline_families=baseline_families)
    if dry_run:
        rows = rows[: max(1, min(2, len(rows)))]
    audited_rows, issues = audit_baseline_budget_rows(rows)
    summary_path = output_dir / "summary.csv"
    plot_path = output_dir / "summary.svg"
    report_path = output_dir / "baseline_budget_report.json"
    write_summary_csv(summary_path, audited_rows)
    write_sanity_plot(plot_path, audited_rows)
    write_json(
        report_path,
        {
            "baseline_families": sorted(baseline_families),
            "rows_collected": len(audited_rows),
            "issues_found": len(issues),
            "checks_pass_rate": (
                sum(float(row["budget_ok"]) for row in audited_rows) / len(audited_rows)
                if audited_rows
                else 0.0
            ),
            "issues": issues,
        },
    )
    write_json(
        output_dir / "metrics.json",
        {
            "mode": "analysis",
            "analysis_mode": "baseline_budget_audit",
            "rows_collected": len(audited_rows),
            "issues_found": len(issues),
            "checks_pass_rate": (
                sum(float(row["budget_ok"]) for row in audited_rows) / len(audited_rows)
                if audited_rows
                else 0.0
            ),
            "input_root": str(Path(input_root).resolve()),
            "summary_csv": str(summary_path.resolve()),
            "summary_plot": str(plot_path.resolve()),
            "report_json": str(report_path.resolve()),
        },
    )
