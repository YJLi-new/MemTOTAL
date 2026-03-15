from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from memtotal.data import load_jsonl_dataset
from memtotal.tasks.sources import materialize_benchmark_source
from memtotal.utils.io import write_json, write_jsonl


@dataclass(frozen=True)
class SplitPlan:
    source_examples: int
    support_examples: int
    train_examples: int
    eval_examples: int

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SplitPlan":
        return cls(
            source_examples=int(payload["source_examples"]),
            support_examples=int(payload["support_examples"]),
            train_examples=int(payload["train_examples"]),
            eval_examples=int(payload["eval_examples"]),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "source_examples": int(self.source_examples),
            "support_examples": int(self.support_examples),
            "train_examples": int(self.train_examples),
            "eval_examples": int(self.eval_examples),
        }


DEFAULT_SPLIT_PLANS: dict[str, SplitPlan] = {
    "gsm8k": SplitPlan(source_examples=128, support_examples=8, train_examples=80, eval_examples=40),
    "triviaqa": SplitPlan(source_examples=128, support_examples=8, train_examples=80, eval_examples=40),
    "narrativeqa": SplitPlan(source_examples=64, support_examples=8, train_examples=32, eval_examples=24),
    "fever": SplitPlan(source_examples=256, support_examples=8, train_examples=64, eval_examples=64),
}
DEFAULT_BENCHMARKS: tuple[str, ...] = ("gsm8k", "triviaqa", "fever")


def normalize_benchmarks(raw_benchmarks: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if raw_benchmarks is None:
        return DEFAULT_BENCHMARKS
    if isinstance(raw_benchmarks, str):
        benchmarks = [part.strip().lower() for part in raw_benchmarks.split(",") if part.strip()]
    else:
        benchmarks = [str(benchmark).strip().lower() for benchmark in raw_benchmarks if str(benchmark).strip()]
    if not benchmarks:
        raise ValueError("At least one benchmark must be selected for writer_jointpeft_data.")
    deduplicated: list[str] = []
    seen: set[str] = set()
    for benchmark_id in benchmarks:
        if benchmark_id in seen:
            continue
        seen.add(benchmark_id)
        deduplicated.append(benchmark_id)
    return tuple(deduplicated)


def load_split_plan_overrides(path: str | Path | None) -> dict[str, SplitPlan]:
    if path is None or not str(path).strip():
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("split_plan_json must contain a JSON object keyed by benchmark id.")
    overrides: dict[str, SplitPlan] = {}
    for benchmark_id, split_payload in payload.items():
        if not isinstance(split_payload, dict):
            raise ValueError(
                f"split_plan_json entry for {benchmark_id!r} must be an object with split counts."
            )
        overrides[str(benchmark_id).strip().lower()] = SplitPlan.from_mapping(split_payload)
    return overrides


def resolve_split_plans(
    benchmarks: tuple[str, ...],
    *,
    overrides: dict[str, SplitPlan] | None = None,
) -> dict[str, SplitPlan]:
    resolved: dict[str, SplitPlan] = {}
    active_overrides = overrides or {}
    for benchmark_id in benchmarks:
        if benchmark_id in active_overrides:
            resolved[benchmark_id] = active_overrides[benchmark_id]
            continue
        if benchmark_id not in DEFAULT_SPLIT_PLANS:
            raise ValueError(
                f"No default split plan is defined for benchmark_id={benchmark_id!r}. "
                "Provide --split_plan_json with an explicit plan."
            )
        resolved[benchmark_id] = DEFAULT_SPLIT_PLANS[benchmark_id]
    return resolved


def deterministic_split_rows(
    rows: list[dict[str, Any]],
    *,
    split_sizes: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    total_required = sum(int(size) for size in split_sizes.values())
    if len(rows) < total_required:
        raise ValueError(
            f"Requested {total_required} rows across {sorted(split_sizes)}, but only {len(rows)} rows are available."
        )
    shuffled_rows = list(rows)
    random.Random(int(seed)).shuffle(shuffled_rows)
    selected_rows = shuffled_rows[:total_required]
    partitions: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split_name, split_size in split_sizes.items():
        next_cursor = cursor + int(split_size)
        partitions[split_name] = selected_rows[cursor:next_cursor]
        cursor = next_cursor
    return partitions


def materialize_writer_jointpeft_bundle(
    *,
    output_root: str | Path,
    source_output_root: str | Path,
    manifest_root: str | Path,
    seed: int,
    benchmarks: str | list[str] | tuple[str, ...] | None = None,
    split_plans: dict[str, SplitPlan] | None = None,
    fever_support_path: str | Path | None = None,
    fever_eval_path: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    source_output_root = Path(source_output_root).resolve()
    manifest_root = Path(manifest_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_output_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    selected_benchmarks = normalize_benchmarks(benchmarks)
    resolved_split_plans = resolve_split_plans(selected_benchmarks, overrides=split_plans)

    manifest: dict[str, Any] = {
        "seed": int(seed),
        "benchmarks": list(selected_benchmarks),
        "tasks": {},
    }
    split_order = ("support", "train", "eval")
    for benchmark_index, benchmark_id in enumerate(selected_benchmarks):
        split_plan = resolved_split_plans[benchmark_id]
        if benchmark_id == "fever":
            if fever_support_path is None or fever_eval_path is None:
                raise ValueError(
                    "fever is selected, so fever_support_path and fever_eval_path must both be provided."
                )
            fever_support_rows = load_jsonl_dataset(fever_support_path)
            fever_support_split = deterministic_split_rows(
                fever_support_rows,
                split_sizes={"support": split_plan.support_examples},
                seed=int(seed) + 2026,
            )
            fever_eval_rows = load_jsonl_dataset(fever_eval_path)
            fever_train_eval_split = deterministic_split_rows(
                fever_eval_rows,
                split_sizes={
                    "train": split_plan.train_examples,
                    "eval": split_plan.eval_examples,
                },
                seed=int(seed) + 3039,
            )
            fever_root = output_root / benchmark_id
            fever_root.mkdir(parents=True, exist_ok=True)
            fever_manifest = {
                "source_materialized_path": str(Path(fever_eval_path).resolve()),
                "source_rows": len(fever_eval_rows),
                "support_source_path": str(Path(fever_support_path).resolve()),
                "support_source_rows": len(fever_support_rows),
                "split_plan": split_plan.as_dict(),
                "splits": {},
            }
            fever_partitions = {
                "support": fever_support_split["support"],
                "train": fever_train_eval_split["train"],
                "eval": fever_train_eval_split["eval"],
            }
            for split_name in split_order:
                split_path = fever_root / f"{split_name}.jsonl"
                split_payload = fever_partitions[split_name]
                write_jsonl(split_path, split_payload)
                fever_manifest["splits"][split_name] = {
                    "path": str(split_path.resolve()),
                    "rows": len(split_payload),
                    "ids": [str(row.get("id", "")) for row in split_payload],
                }
            manifest["tasks"][benchmark_id] = fever_manifest
            continue
        source_manifest = materialize_benchmark_source(
            benchmark_id=benchmark_id,
            output_root=source_output_root,
            manifest_root=manifest_root,
            max_examples=split_plan.source_examples,
            seed=int(seed) + (benchmark_index * 101),
        )
        source_rows = load_jsonl_dataset(str(source_manifest["materialized_path"]))
        split_rows = deterministic_split_rows(
            source_rows,
            split_sizes={
                "support": split_plan.support_examples,
                "train": split_plan.train_examples,
                "eval": split_plan.eval_examples,
            },
            seed=int(seed) + (benchmark_index * 997),
        )
        task_root = output_root / benchmark_id
        task_root.mkdir(parents=True, exist_ok=True)
        task_manifest = {
            "source_materialized_path": str(Path(str(source_manifest["materialized_path"])).resolve()),
            "source_rows": len(source_rows),
            "split_plan": split_plan.as_dict(),
            "splits": {},
        }
        for split_name in split_order:
            split_path = task_root / f"{split_name}.jsonl"
            split_payload = split_rows[split_name]
            write_jsonl(split_path, split_payload)
            task_manifest["splits"][split_name] = {
                "path": str(split_path.resolve()),
                "rows": len(split_payload),
                "ids": [str(row.get("id", "")) for row in split_payload],
            }
        manifest["tasks"][benchmark_id] = task_manifest

    manifest_path = output_root / "split-manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the medium-slice dataset bundle for writer joint-PEFT runs.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_output_root", required=True)
    parser.add_argument("--manifest_root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark ids to materialize.",
    )
    parser.add_argument(
        "--split_plan_json",
        default="",
        help="Optional JSON file mapping benchmark ids to explicit split plans.",
    )
    parser.add_argument(
        "--fever_support_path",
        default="data/benchmarks/pilots/fever/pilot-support8.jsonl",
    )
    parser.add_argument(
        "--fever_eval_path",
        default="data/benchmarks/materialized/fever/eval-real-smoke256.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    split_plan_overrides = load_split_plan_overrides(args.split_plan_json or None)
    materialize_writer_jointpeft_bundle(
        output_root=args.output_root,
        source_output_root=args.source_output_root,
        manifest_root=args.manifest_root,
        seed=args.seed,
        benchmarks=args.benchmarks,
        split_plans=split_plan_overrides,
        fever_support_path=args.fever_support_path,
        fever_eval_path=args.fever_eval_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
