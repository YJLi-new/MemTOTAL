from __future__ import annotations

import argparse
import sys
from pathlib import Path

from memtotal.analysis.baseline_budget import run_baseline_budget_audit
from memtotal.analysis.failure_checks import run_m3_failure_checks
from memtotal.analysis.m3_gradient_audit import run_m3_stage_c_gradient_audit
from memtotal.analysis.m3_probe import run_m3_stage_b_probe_summary
from memtotal.analysis.reporting import collect_metrics, write_sanity_plot, write_summary_csv
from memtotal.utils.config import load_config
from memtotal.utils.io import initialize_run_artifacts, write_json
from memtotal.utils.profiling import ProfileTracker
from memtotal.utils.repro import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MemTOTAL bootstrap analysis entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_root", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    set_seed(args.seed)
    initialize_run_artifacts(
        output_dir=args.output_dir,
        config=config,
        seed=args.seed,
        argv=sys.argv if argv is None else ["analysis", *argv],
    )
    analysis_mode = str(config["runtime"].get("analysis_mode", "summary"))
    output_dir = Path(args.output_dir)
    if analysis_mode == "m3_failure_checks":
        run_m3_failure_checks(
            config=config,
            seed=args.seed,
            output_dir=output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        return 0
    if analysis_mode == "baseline_budget_audit":
        if not args.input_root:
            raise ValueError("--input_root is required for baseline_budget_audit mode.")
        run_baseline_budget_audit(
            config=config,
            output_dir=output_dir,
            input_root=args.input_root,
            dry_run=args.dry_run,
        )
        return 0
    if analysis_mode == "m3_stage_b_probe_summary":
        if not args.input_root:
            raise ValueError("--input_root is required for m3_stage_b_probe_summary mode.")
        run_m3_stage_b_probe_summary(
            output_dir=output_dir,
            input_root=args.input_root,
            dry_run=args.dry_run,
        )
        return 0
    if analysis_mode == "m3_stage_c_gradient_audit":
        run_m3_stage_c_gradient_audit(
            config=config,
            seed=args.seed,
            output_dir=output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        return 0
    if not args.input_root:
        raise ValueError("--input_root is required for summary analysis mode.")
    profiler = ProfileTracker(
        output_dir=output_dir,
        device="cpu",
        event_name="analysis",
    )
    rows = collect_metrics(args.input_root)
    if args.dry_run:
        rows = rows[: max(1, min(2, len(rows)))]
    profiler.add_example(len(rows))
    summary_path = output_dir / "summary.csv"
    plot_path = output_dir / "summary.svg"
    write_summary_csv(summary_path, rows)
    write_sanity_plot(plot_path, rows)
    profile_metrics = profiler.finalize()
    write_json(
        output_dir / "metrics.json",
        {
            "mode": "analysis",
            "rows_collected": len(rows),
            "input_root": str(Path(args.input_root).resolve()),
            "summary_csv": str(summary_path.resolve()),
            "summary_plot": str(plot_path.resolve()),
            **profile_metrics,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
