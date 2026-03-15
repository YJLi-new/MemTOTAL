#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARM_SPECS = (
    ("source_stub_health_gsm8k", "gsm8k-source-stub-health", "pilot-I-source-stub-health"),
    ("writer_gsm8k", "gsm8k-writer", "pilot-I-writer-direct"),
    ("writer_narrativeqa", "narrativeqa-writer", "pilot-I-writer-direct"),
    ("writer_fever", "fever-writer", "pilot-I-writer-direct"),
)

SUITE_DIRS = (
    "gsm8k-source-stub-health",
    "gsm8k-control",
    "gsm8k-writer",
    "narrativeqa-control",
    "narrativeqa-writer",
    "fever-control",
    "fever-writer",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _tail(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    return values[-count:]


def _tmux_session_alive(session_name: str) -> bool:
    if not session_name:
        return False
    try:
        probe = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False
    return probe.returncode == 0


def _log_tail(log_path: Path, *, max_lines: int = 40) -> list[str]:
    if not log_path.exists():
        return []
    lines = [line.rstrip() for line in log_path.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-max_lines:]


def _suite_complete(run_root: Path, suite_dir: str) -> bool:
    return (run_root / suite_dir / "suite_metrics.json").exists()


def _suite_completion_summary(run_root: Path) -> tuple[int, int]:
    completed = sum(1 for suite_dir in SUITE_DIRS if _suite_complete(run_root, suite_dir))
    return completed, len(SUITE_DIRS)


def _snapshot_rows(arm_dir: Path) -> list[dict[str, Any]]:
    live_path = arm_dir / "snapshot_metrics.live.jsonl"
    rows = _load_jsonl(live_path)
    if rows:
        return rows
    snapshot_root = arm_dir / "snapshot_evals"
    if snapshot_root.exists():
        discovered: list[dict[str, Any]] = []
        for metrics_path in sorted(snapshot_root.glob("step_*/metrics.json")):
            payload = _load_json(metrics_path)
            payload.setdefault("step", int(metrics_path.parent.name.split("_")[-1]))
            discovered.append(payload)
        if discovered:
            return discovered
    final_train_events_path = arm_dir / "train_events.json"
    if final_train_events_path.exists():
        payload = _load_json(final_train_events_path)
        snapshots = payload.get("snapshots", [])
        if isinstance(snapshots, list):
            return [dict(row) for row in snapshots if isinstance(row, dict)]
    return []


def _train_rows(arm_dir: Path) -> list[dict[str, Any]]:
    live_path = arm_dir / "train_trace.live.jsonl"
    rows = _load_jsonl(live_path)
    if rows:
        return rows
    final_train_events_path = arm_dir / "train_events.json"
    if final_train_events_path.exists():
        payload = _load_json(final_train_events_path)
        events = payload.get("events", [])
        if isinstance(events, list):
            return [dict(row) for row in events if isinstance(row, dict)]
    return []


def _arm_summary(run_root: Path, suite_dir: str, pilot_subdir: str) -> dict[str, Any]:
    arm_root = run_root / suite_dir
    arm_dir = arm_root / pilot_subdir
    train_rows = _train_rows(arm_dir)
    snapshot_rows = _snapshot_rows(arm_dir)
    latest_train = train_rows[-1] if train_rows else {}
    latest_snapshot = snapshot_rows[-1] if snapshot_rows else {}
    if _suite_complete(run_root, suite_dir):
        state = "completed"
    elif train_rows or snapshot_rows or arm_dir.exists():
        state = "running"
    elif arm_root.exists():
        state = "queued"
    else:
        state = "not_started"
    recent_rows = _tail(train_rows, 8)
    recent_losses = [_safe_float(row.get("loss")) for row in recent_rows]
    recent_writer_grads = [_safe_float(row.get("grad_norm_writer")) for row in recent_rows]
    recent_source_grads = [_safe_float(row.get("grad_norm_source_stub")) for row in recent_rows]
    recent_projector_grads = [_safe_float(row.get("grad_norm_projector")) for row in recent_rows]
    recent_receiver_grads = [_safe_float(row.get("grad_norm_receiver_lora")) for row in recent_rows]
    recent_deltas = [_safe_float(row.get("delta_answer_logprob")) for row in recent_rows]
    recent_clipped = [bool(row.get("was_grad_clipped", False)) for row in recent_rows]
    return {
        "state": state,
        "arm_dir": str(arm_dir),
        "steps_recorded": len(train_rows),
        "latest_step": int(latest_train.get("step", 0)) if latest_train else 0,
        "latest_loss": _safe_float(latest_train.get("loss")),
        "latest_delta_answer_logprob": _safe_float(latest_train.get("delta_answer_logprob")),
        "latest_writer_grad": _safe_float(latest_train.get("grad_norm_writer")),
        "latest_source_grad": _safe_float(latest_train.get("grad_norm_source_stub")),
        "latest_projector_grad": _safe_float(latest_train.get("grad_norm_projector")),
        "latest_receiver_grad": _safe_float(latest_train.get("grad_norm_receiver_lora")),
        "latest_total_grad_norm_pre_clip": _safe_float(latest_train.get("total_grad_norm_pre_clip")),
        "latest_was_grad_clipped": bool(latest_train.get("was_grad_clipped", False)),
        "latest_optimizer_lrs": dict(latest_train.get("optimizer_lr_by_group", {})),
        "recent_loss_median": _median(recent_losses),
        "recent_writer_grad_median": _median(recent_writer_grads),
        "recent_source_grad_median": _median(recent_source_grads),
        "recent_projector_grad_median": _median(recent_projector_grads),
        "recent_receiver_grad_median": _median(recent_receiver_grads),
        "recent_delta_answer_logprob_median": _median(recent_deltas),
        "recent_clipped_steps": int(sum(1 for value in recent_clipped if value)),
        "recent_loss_trace": [round(value, 6) for value in recent_losses],
        "recent_writer_grad_trace": [round(value, 6) for value in recent_writer_grads],
        "recent_source_grad_trace": [round(value, 6) for value in recent_source_grads],
        "recent_projector_grad_trace": [round(value, 6) for value in recent_projector_grads],
        "recent_receiver_grad_trace": [round(value, 6) for value in recent_receiver_grads],
        "snapshot_steps": [int(row.get("step", 0)) for row in snapshot_rows],
        "latest_snapshot_step": int(latest_snapshot.get("step", 0)) if latest_snapshot else 0,
        "latest_snapshot_accuracy": _safe_float(latest_snapshot.get("accuracy")),
        "latest_snapshot_macro_f1": _safe_float(latest_snapshot.get("macro_f1")),
        "latest_snapshot_margin": _safe_float(latest_snapshot.get("mean_margin")),
        "latest_snapshot_prefix_l2": _safe_float(latest_snapshot.get("prefix_l2")),
    }


def _render_report(
    *,
    run_root: Path,
    output_report: Path,
    session_name: str,
    log_path: Path,
    base_seed: int,
) -> str:
    session_alive = _tmux_session_alive(session_name)
    completed_suites, total_suites = _suite_completion_summary(run_root)
    lines = [
        "# Writer Deep-Prefix JointPEFT Live Monitor",
        "",
        f"- updated_at_utc: {_utc_now()}",
        f"- session_name: {session_name}",
        f"- session_alive: {session_alive}",
        f"- base_seed: {base_seed}",
        f"- run_root: {run_root}",
        f"- monitor_output: {output_report}",
        f"- log_path: {log_path}",
        f"- completed_suites: {completed_suites}/{total_suites}",
    ]
    for label, suite_dir, pilot_subdir in ARM_SPECS:
        summary = _arm_summary(run_root, suite_dir, pilot_subdir)
        lines.extend(
            [
                "",
                f"## {label}",
                f"- state: {summary['state']}",
                f"- arm_dir: {summary['arm_dir']}",
                f"- steps_recorded: {summary['steps_recorded']}",
                f"- latest_step: {summary['latest_step']}",
                f"- latest_loss: {summary['latest_loss']:.6f}",
                f"- latest_delta_answer_logprob: {summary['latest_delta_answer_logprob']:.6f}",
                f"- latest_source_grad: {summary['latest_source_grad']:.6f}",
                f"- latest_writer_grad: {summary['latest_writer_grad']:.6f}",
                f"- latest_projector_grad: {summary['latest_projector_grad']:.6f}",
                f"- latest_receiver_grad: {summary['latest_receiver_grad']:.6f}",
                f"- latest_total_grad_norm_pre_clip: {summary['latest_total_grad_norm_pre_clip']:.6f}",
                f"- latest_was_grad_clipped: {summary['latest_was_grad_clipped']}",
                f"- recent_loss_median: {summary['recent_loss_median']:.6f}",
                f"- recent_delta_answer_logprob_median: {summary['recent_delta_answer_logprob_median']:.6f}",
                f"- recent_source_grad_median: {summary['recent_source_grad_median']:.6f}",
                f"- recent_writer_grad_median: {summary['recent_writer_grad_median']:.6f}",
                f"- recent_projector_grad_median: {summary['recent_projector_grad_median']:.6f}",
                f"- recent_receiver_grad_median: {summary['recent_receiver_grad_median']:.6f}",
                f"- recent_clipped_steps: {summary['recent_clipped_steps']}",
                f"- recent_loss_trace: {summary['recent_loss_trace']}",
                f"- recent_source_grad_trace: {summary['recent_source_grad_trace']}",
                f"- recent_writer_grad_trace: {summary['recent_writer_grad_trace']}",
                f"- recent_projector_grad_trace: {summary['recent_projector_grad_trace']}",
                f"- recent_receiver_grad_trace: {summary['recent_receiver_grad_trace']}",
                f"- latest_optimizer_lrs: {summary['latest_optimizer_lrs']}",
                f"- snapshot_steps_seen: {summary['snapshot_steps']}",
                f"- latest_snapshot_step: {summary['latest_snapshot_step']}",
                f"- latest_snapshot_accuracy: {summary['latest_snapshot_accuracy']:.6f}",
                f"- latest_snapshot_macro_f1: {summary['latest_snapshot_macro_f1']:.6f}",
                f"- latest_snapshot_margin: {summary['latest_snapshot_margin']:.6f}",
                f"- latest_snapshot_prefix_l2: {summary['latest_snapshot_prefix_l2']:.6f}",
            ]
        )
    log_tail = _log_tail(log_path)
    lines.extend(["", "## Log Tail"])
    if log_tail:
        lines.append("```text")
        lines.extend(log_tail)
        lines.append("```")
    else:
        lines.append("- log file not created yet")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuously update a markdown monitor for the writer joint-PEFT tmux run.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--session_name", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--base_seed", type=int, required=True)
    parser.add_argument("--poll_seconds", type=int, default=120)
    parser.add_argument("--idle_polls_after_exit", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_root).resolve()
    output_report = Path(args.output_report).resolve()
    log_path = Path(args.log_path).resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    idle_polls = 0
    last_report = ""
    while True:
        report = _render_report(
            run_root=run_root,
            output_report=output_report,
            session_name=args.session_name,
            log_path=log_path,
            base_seed=int(args.base_seed),
        )
        if report != last_report:
            output_report.write_text(report)
            last_report = report
        if args.once:
            return 0
        if _tmux_session_alive(args.session_name):
            idle_polls = 0
        else:
            idle_polls += 1
            completed_suites, total_suites = _suite_completion_summary(run_root)
            if completed_suites >= total_suites or idle_polls >= int(args.idle_polls_after_exit):
                return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
