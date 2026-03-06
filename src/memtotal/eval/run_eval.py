from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from memtotal.pipeline import MemoryRuntime
from memtotal.tasks import build_task_evaluator, load_task_dataset
from memtotal.utils.config import load_config
from memtotal.utils.io import initialize_run_artifacts, write_json, write_jsonl
from memtotal.utils.profiling import ProfileTracker
from memtotal.utils.repro import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MemTOTAL bootstrap eval entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
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
        argv=sys.argv if argv is None else ["eval", *argv],
    )
    dataset = load_task_dataset(config)
    evaluator = build_task_evaluator(config)
    runtime = MemoryRuntime(config=config, seed=args.seed)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        runtime.load_state_dict(checkpoint["model_state"])

    max_examples = min(len(dataset), 2 if args.dry_run else int(config["runtime"]["eval_examples"]))
    predictions = []
    correct = 0
    similarities = []
    gate_means = []
    active_query_counts = []
    segment_gate_means: list[float] = []
    segment_active_query_counts: list[int] = []
    profiler = ProfileTracker(
        output_dir=Path(args.output_dir),
        device=str(config["runtime"].get("device", "cpu")),
        event_name="eval",
    )

    for example in dataset[:max_examples]:
        profiler.add_example()
        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
        if evaluator.evaluator_type in {"dataset_label_classification", "multiple_choice"}:
            if evaluator.evaluator_type == "multiple_choice":
                choices = example.get("choices", [])
                if not choices:
                    raise ValueError("multiple_choice evaluation requires per-example `choices`.")
                candidate_labels = [str(choice["label"]) for choice in choices]
                candidate_texts = [str(choice["text"]) for choice in choices]
            else:
                candidate_labels = [str(row["label"]) for row in dataset]
                candidate_texts = [str(row["continuation"]) for row in dataset]
            candidate_states = runtime.backbone.summarize_texts(candidate_texts)
            predicted_label, similarity, forward = runtime.predict_label(
                example,
                candidate_states=candidate_states,
                candidate_labels=candidate_labels,
            )
            predicted_text = ""
            if evaluator.evaluator_type == "multiple_choice":
                predicted_text = next(
                    choice["text"] for choice in example["choices"] if choice["label"] == predicted_label
                )
            score_payload = evaluator.evaluate_prediction(
                {"label": predicted_label, "text": predicted_text},
                example,
            )
        else:
            forward = runtime.forward_example(example)
            similarity = float(
                F.cosine_similarity(forward.predicted_state, forward.target_state, dim=-1).mean().item()
            )
            predicted_label = ""
            predicted_text = ""
            score_payload = evaluator.evaluate_prediction({"text": predicted_text}, example)
        profiler.add_tokens(runtime.backbone.count_tokens(forward.next_prompt))
        generated_text = runtime.backbone.generate(
            [forward.next_prompt],
            memory_tokens=forward.generation_memory,
        )[0]
        profiler.add_tokens(runtime.backbone.count_tokens(generated_text))
        if evaluator.evaluator_type == "exact_match":
            score_payload = evaluator.evaluate_prediction({"text": generated_text}, example)
            predicted_text = generated_text
        is_correct = bool(score_payload["correct"])
        correct += int(is_correct)
        similarities.append(similarity)
        gate_mean = float(forward.gating.mean().item())
        active_queries = int((forward.gating > 0.5).sum().item())
        gate_means.append(gate_mean)
        active_query_counts.append(active_queries)
        segment_gate_means.extend(float(item["mean_gate"]) for item in forward.segment_stats)
        segment_active_query_counts.extend(int(item["active_queries"]) for item in forward.segment_stats)
        predictions.append(
            {
                "id": example["id"],
                "domain": example["domain"],
                "benchmark_id": example.get("benchmark_id"),
                "gold_label": example["label"],
                "gold_answer": example.get("gold_answer"),
                "predicted_label": predicted_label,
                "predicted_text": predicted_text,
                "correct": is_correct,
                "score": float(score_payload["score"]),
                "normalized_prediction": score_payload["normalized_prediction"],
                "normalized_reference": score_payload["normalized_reference"],
                "evaluator_type": evaluator.evaluator_type,
                "similarity": similarity,
                "gating_mode": runtime.reader.gating_mode,
                "gates": [float(value) for value in forward.gating.squeeze(0).tolist()],
                "mean_gate": gate_mean,
                "active_queries": active_queries,
                "conditioning": forward.conditioning,
                "injection_position": runtime.injector.position,
                "injection_anchors": forward.injection_anchors,
                "segment_stats": forward.segment_stats,
                "generated_text": generated_text,
            }
        )

    profile_metrics = profiler.finalize()
    metrics = {
        "mode": "eval",
        "examples_evaluated": max_examples,
        "accuracy": correct / max_examples,
        evaluator.metric_name: correct / max_examples,
        "benchmark_id": config["task"].get("benchmark_id"),
        "task_domain": config["task"].get("domain"),
        "smoke_subset": config["task"].get("smoke_subset"),
        "evaluator_type": evaluator.evaluator_type,
        "mean_similarity": sum(similarities) / len(similarities),
        "gating_mode": runtime.reader.gating_mode,
        "mean_gate": sum(gate_means) / len(gate_means),
        "mean_active_queries": sum(active_query_counts) / len(active_query_counts),
        "mean_segment_gate": sum(segment_gate_means) / len(segment_gate_means),
        "mean_segment_active_queries": sum(segment_active_query_counts) / len(segment_active_query_counts),
        "injection_position": runtime.injector.position,
        "conditioning_schema": {
            "domain_name": runtime.conditioning_cfg["domain_key"],
            "task_name": "config.task.name" if runtime.conditioning_cfg["include_task_name"] else "disabled",
        },
        "backbone": config["backbone"]["name"],
        "metric_name": config["task"]["metric_name"],
        **profile_metrics,
    }
    write_json(Path(args.output_dir) / "metrics.json", metrics)
    write_jsonl(Path(args.output_dir) / "predictions.jsonl", predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
