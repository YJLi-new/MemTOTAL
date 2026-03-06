from __future__ import annotations

import argparse
from pathlib import Path

from memtotal.tasks.sources import list_benchmark_sources, materialize_benchmark_source
from memtotal.utils.io import write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize benchmark data sources into repo-managed JSONL files.")
    parser.add_argument(
        "--benchmarks",
        default="gsm8k,math,gpqa,triviaqa,story_cloze,kodcode,rocstories,fever,alfworld,memoryagentbench",
        help="Comma-separated benchmark ids to materialize or document.",
    )
    parser.add_argument("--output_root", default="data/benchmarks/materialized")
    parser.add_argument("--manifest_root", default="data/benchmarks/manifests")
    parser.add_argument("--summary_path", default="data/benchmarks/source_summary.json")
    parser.add_argument("--max_examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=701)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selected = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    manifests = []
    for benchmark_id in selected:
        manifests.append(
            materialize_benchmark_source(
                benchmark_id=benchmark_id,
                output_root=Path(args.output_root),
                manifest_root=Path(args.manifest_root),
                max_examples=args.max_examples,
                seed=args.seed,
            )
        )
    write_json(args.summary_path, {"benchmarks": manifests})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
