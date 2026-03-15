#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.baselines.retrieval import RetrievalBaselineRuntime
from memtotal.tasks import build_task_evaluator, load_task_dataset
from memtotal.utils.config import load_config
from memtotal.utils.io import initialize_run_artifacts, write_json, write_jsonl
from memtotal.utils.repro import set_seed


def _load_support_dataset(config: dict) -> list[dict]:
    support_path = str(config.get("task", {}).get("support_dataset_path", "")).strip()
    if not support_path:
        return load_task_dataset(config)
    support_config = dict(config)
    support_task = dict(config["task"])
    support_task["dataset_path"] = support_path
    support_config["task"] = support_task
    return load_task_dataset(support_config)


def _candidate_payload(example: dict) -> tuple[list[str], list[str]]:
    choices = example.get("choices", [])
    if not choices:
        raise ValueError("multiple_choice retrieval evaluation requires per-example `choices`.")
    return [str(choice["label"]) for choice in choices], [str(choice["text"]) for choice in choices]


def _support_pool_for_example(support_dataset: list[dict], example: dict) -> list[dict]:
    retrieval_group = str(example.get("retrieval_group", "")).strip()
    if not retrieval_group:
        return support_dataset
    grouped = [
        row
        for row in support_dataset
        if str(row.get("retrieval_group", "")).strip() == retrieval_group
    ]
    return grouped or support_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PLANv9 V9-1 grouped retrieval baseline eval.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    initialize_run_artifacts(
        output_dir=output_dir,
        config=config,
        seed=args.seed,
        argv=sys.argv if argv is None else ["planv9_v9_1_retrieval_eval", *argv],
    )

    eval_dataset = load_task_dataset(config)
    support_dataset = _load_support_dataset(config)
    evaluator = build_task_evaluator(config)
    runtime = RetrievalBaselineRuntime(config=config, seed=args.seed)
    support_examples_requested = int(config.get("baseline", {}).get("support_examples", 0))
    max_examples = min(len(eval_dataset), int(config["runtime"].get("eval_examples", len(eval_dataset))))

    predictions: list[dict] = []
    correct = 0
    score_total = 0.0
    support_score_total = 0.0
    support_score_count = 0
    support_pool_sizes: list[int] = []
    retriever = str(config.get("baseline", {}).get("rag", {}).get("retriever", "lexical_overlap"))
    for example in eval_dataset[:max_examples]:
        support_pool = _support_pool_for_example(support_dataset, example)
        support_pool_sizes.append(len(support_pool))
        support_examples, support_scores, retriever = runtime.select_support_examples(
            support_pool,
            example,
            support_examples_requested,
        )
        if support_scores:
            support_score_total += float(sum(support_scores))
            support_score_count += len(support_scores)
        if evaluator.evaluator_type == "multiple_choice":
            candidate_labels, candidate_texts = _candidate_payload(example)
            output = runtime.predict_multiple_choice(
                example,
                candidate_labels=candidate_labels,
                candidate_texts=candidate_texts,
                support_examples=support_examples,
            )
            score_payload = evaluator.evaluate_prediction(
                {"label": output.predicted_label, "text": output.predicted_text},
                example,
            )
        else:
            output = runtime.generate_text(
                example,
                support_examples=support_examples,
            )
            score_payload = evaluator.evaluate_prediction({"text": output.predicted_text}, example)

        is_correct = bool(score_payload["correct"])
        score_value = float(score_payload["score"])
        correct += int(is_correct)
        score_total += score_value
        predictions.append(
            {
                "id": example["id"],
                "benchmark_id": example.get("benchmark_id"),
                "gold_label": example.get("label", ""),
                "gold_answer": example.get("gold_answer"),
                "predicted_label": output.predicted_label,
                "predicted_text": output.predicted_text,
                "correct": is_correct,
                "score": score_value,
                "normalized_prediction": score_payload["normalized_prediction"],
                "normalized_reference": score_payload["normalized_reference"],
                "baseline_prompt": output.prompt,
                "baseline_support_ids": [row["id"] for row in support_examples],
                "baseline_support_scores": support_scores,
                "baseline_retriever": retriever,
                "retrieval_group": example.get("retrieval_group"),
                "support_pool_size": len(support_pool),
            }
        )

    mean_score = score_total / max(1, max_examples)
    metrics = {
        "mode": "eval_baseline_rag_grouped_pool",
        "examples_evaluated": max_examples,
        "accuracy": correct / max(1, max_examples),
        "mean_score": mean_score,
        "task_score": mean_score,
        str(config["task"]["metric_name"]): mean_score,
        "benchmark_id": config["task"].get("benchmark_id"),
        "task_domain": config["task"].get("domain"),
        "evaluator_type": evaluator.evaluator_type,
        "backbone": config["backbone"]["name"],
        "metric_name": config["task"]["metric_name"],
        "baseline_family": "rag",
        "baseline_mode": str(config.get("baseline", {}).get("mode", "retrieval_augmented")),
        "baseline_retriever": retriever,
        "support_examples_requested": support_examples_requested,
        "support_pool_size_mean": (sum(support_pool_sizes) / len(support_pool_sizes)) if support_pool_sizes else 0.0,
        "mean_support_retrieval_score": (
            support_score_total / max(1, support_score_count) if support_score_count > 0 else 0.0
        ),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
