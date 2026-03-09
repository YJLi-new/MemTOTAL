from __future__ import annotations

import argparse
from dataclasses import dataclass
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


DEFAULT_SPLIT_PLANS: dict[str, SplitPlan] = {
    "gsm8k": SplitPlan(source_examples=128, support_examples=8, train_examples=80, eval_examples=40),
    "narrativeqa": SplitPlan(source_examples=64, support_examples=8, train_examples=32, eval_examples=24),
    "fever": SplitPlan(source_examples=256, support_examples=8, train_examples=64, eval_examples=64),
}


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
    fever_support_path: str | Path,
    fever_eval_path: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    source_output_root = Path(source_output_root).resolve()
    manifest_root = Path(manifest_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_output_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "seed": int(seed),
        "tasks": {},
    }
    split_order = ("support", "train", "eval")
    for benchmark_index, benchmark_id in enumerate(("gsm8k", "narrativeqa")):
        split_plan = DEFAULT_SPLIT_PLANS[benchmark_id]
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

    fever_split_plan = DEFAULT_SPLIT_PLANS["fever"]
    fever_support_rows = load_jsonl_dataset(fever_support_path)
    fever_support_split = deterministic_split_rows(
        fever_support_rows,
        split_sizes={"support": fever_split_plan.support_examples},
        seed=int(seed) + 2026,
    )
    fever_eval_rows = load_jsonl_dataset(fever_eval_path)
    fever_train_eval_split = deterministic_split_rows(
        fever_eval_rows,
        split_sizes={
            "train": fever_split_plan.train_examples,
            "eval": fever_split_plan.eval_examples,
        },
        seed=int(seed) + 3039,
    )
    fever_root = output_root / "fever"
    fever_root.mkdir(parents=True, exist_ok=True)
    fever_manifest = {
        "source_materialized_path": str(Path(fever_eval_path).resolve()),
        "source_rows": len(fever_eval_rows),
        "support_source_path": str(Path(fever_support_path).resolve()),
        "support_source_rows": len(fever_support_rows),
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
    manifest["tasks"]["fever"] = fever_manifest

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
    materialize_writer_jointpeft_bundle(
        output_root=args.output_root,
        source_output_root=args.source_output_root,
        manifest_root=args.manifest_root,
        seed=args.seed,
        fever_support_path=args.fever_support_path,
        fever_eval_path=args.fever_eval_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
