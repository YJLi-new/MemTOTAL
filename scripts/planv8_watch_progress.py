#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _last_jsonl_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_row: dict[str, Any] | None = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            last_row = payload
    return last_row


def _summarize_arm(run_dir: Path) -> tuple[int, int]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        last_step = 0
        last_row = _last_jsonl_row(run_dir / "train_trace.live.jsonl")
        if last_row is not None:
            last_step = int(last_row.get("step", 0) or 0)
        return (1, last_step)
    snapshot_count = sum(
        1 for path in (run_dir / "snapshot_evals").glob("step_*/metrics.json") if path.is_file()
    )
    last_step = 0
    for candidate in (
        run_dir / "train_trace.live.jsonl",
        run_dir / "snapshot_metrics.live.jsonl",
    ):
        last_row = _last_jsonl_row(candidate)
        if last_row is not None:
            last_step = max(last_step, int(last_row.get("step", 0) or 0))
    return (snapshot_count, last_step)


def build_progress_summary(run_root: Path) -> dict[str, Any]:
    arm_dirs = sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir()
        and (
            (path / ".suite.lock").exists()
            or (path / "train_trace.live.jsonl").exists()
            or (path / "snapshot_evals").exists()
            or (path / "metrics.json").exists()
        )
    )
    final_metrics_count = 0
    snapshot_metrics_count = 0
    latest_arm_name = ""
    latest_step = 0
    active_arm_count = 0
    for arm_dir in arm_dirs:
        arm_final_metrics = 1 if (arm_dir / "metrics.json").exists() else 0
        final_metrics_count += arm_final_metrics
        arm_snapshot_metrics, arm_last_step = _summarize_arm(arm_dir)
        snapshot_metrics_count += arm_snapshot_metrics
        if arm_final_metrics == 0:
            active_arm_count += 1
        if arm_last_step >= latest_step:
            latest_step = arm_last_step
            latest_arm_name = arm_dir.name
    return {
        "run_root": str(run_root),
        "final_metrics_count": int(final_metrics_count),
        "snapshot_metrics_count": int(snapshot_metrics_count),
        "active_arm_count": int(active_arm_count),
        "latest_arm": latest_arm_name,
        "latest_step": int(latest_step),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_progress_summary(args.run_root), sort_keys=True))


if __name__ == "__main__":
    main()
