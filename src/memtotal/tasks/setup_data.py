from __future__ import annotations

import argparse
import json
from pathlib import Path

from memtotal.tasks.sources import list_benchmark_sources, materialize_benchmark_source
from memtotal.utils.io import write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize benchmark data sources into repo-managed JSONL files.")
    parser.add_argument(
        "--benchmarks",
        default="gsm8k,math,gpqa,triviaqa,story_cloze,narrativeqa,kodcode,rocstories,fever,alfworld,memoryagentbench",
        help="Comma-separated benchmark ids to materialize or document.",
    )
    parser.add_argument("--output_root", default="data/benchmarks/materialized")
    parser.add_argument("--manifest_root", default="data/benchmarks/manifests")
    parser.add_argument("--summary_path", default="data/benchmarks/source_summary.json")
    parser.add_argument("--max_examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=701)
    return parser


def _load_existing_summary(summary_path: Path) -> dict[str, dict]:
    if not summary_path.exists():
        return {}
    raw = json.loads(summary_path.read_text())
    benchmarks = raw.get("benchmarks", raw if isinstance(raw, list) else [])
    existing: dict[str, dict] = {}
    for item in benchmarks:
        if not isinstance(item, dict):
            continue
        benchmark_id = item.get("benchmark_id")
        if benchmark_id:
            existing[str(benchmark_id)] = item
    return existing


def _merge_summary_manifests(existing: dict[str, dict], updates: list[dict]) -> list[dict]:
    merged = dict(existing)
    for manifest in updates:
        benchmark_id = str(manifest["benchmark_id"])
        merged[benchmark_id] = manifest
    ordered_ids = [spec.benchmark_id for spec in list_benchmark_sources() if spec.benchmark_id in merged]
    extra_ids = sorted(set(merged) - set(ordered_ids))
    return [merged[benchmark_id] for benchmark_id in [*ordered_ids, *extra_ids]]


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
    summary_path = Path(args.summary_path)
    existing = _load_existing_summary(summary_path)
    write_json(summary_path, {"benchmarks": _merge_summary_manifests(existing, manifests)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
