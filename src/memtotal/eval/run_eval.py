from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from memtotal.baselines import (
    AdapterBaselineRuntime,
    LightThinkerBaselineRuntime,
    MemoryBankBaselineRuntime,
    PromptBaselineRuntime,
    RetrievalBaselineRuntime,
)
from memtotal.baselines.budgeting import build_baseline_budget_fields
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


def _select_support_examples(
    dataset: list[dict[str, object]],
    example: dict[str, object],
    support_examples: int,
) -> list[dict[str, object]]:
    if support_examples <= 0:
        return []
    candidates = [row for row in dataset if row["id"] != example["id"]]
    return candidates[: min(support_examples, len(candidates))]


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
    baseline_cfg = config.get("baseline", {})
    baseline_family = str(baseline_cfg.get("family", ""))
    use_baseline = baseline_family in {"prompting", "adapter", "meta_prompting", "rag", "lightthinker", "memory_bank"}
    if baseline_family in {"prompting", "meta_prompting"}:
        runtime = PromptBaselineRuntime(config=config, seed=args.seed)
    elif baseline_family == "rag":
        runtime = RetrievalBaselineRuntime(config=config, seed=args.seed)
    elif baseline_family == "lightthinker":
        runtime = LightThinkerBaselineRuntime(config=config, seed=args.seed)
    elif baseline_family == "memory_bank":
        runtime = MemoryBankBaselineRuntime(config=config, seed=args.seed)
    elif baseline_family == "adapter":
        runtime = AdapterBaselineRuntime(config=config, seed=args.seed)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            runtime.load_state_dict(checkpoint["model_state"])
    else:
        runtime = MemoryRuntime(config=config, seed=args.seed)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            runtime.load_state_dict(checkpoint["model_state"])
    baseline_budget_fields: dict[str, object] = {}
    if use_baseline:
        trainable_parameter_count = 0
        if baseline_family == "adapter":
            trainable_parameter_count = sum(parameter.numel() for parameter in runtime.parameters() if parameter.requires_grad)
        baseline_budget_fields = build_baseline_budget_fields(
            config=config,
            baseline_family=baseline_family,
            baseline_mode=str(baseline_cfg.get("mode", "vanilla")),
            trainable_parameter_count=trainable_parameter_count,
        )

    max_examples = min(len(dataset), 2 if args.dry_run else int(config["runtime"]["eval_examples"]))
    predictions = []
    correct = 0
    score_total = 0.0
    similarities = []
    gate_means = []
    active_query_counts = []
    retrieval_support_score_means: list[float] = []
    thought_sketch_token_counts: list[int] = []
    memory_bank_entry_counts: list[int] = []
    memory_bank_selection_score_means: list[float] = []
    segment_gate_means: list[float] = []
    segment_active_query_counts: list[int] = []
    capability_scores: dict[str, list[float]] = {}
    capability_metric_names: dict[str, str] = {}
    profiler = ProfileTracker(
        output_dir=Path(args.output_dir),
        device=str(config["runtime"].get("device", "cpu")),
        event_name="eval",
    )

    for example in dataset[:max_examples]:
        profiler.add_example()
        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
        uses_candidate_selection = evaluator.evaluator_type in {"dataset_label_classification", "multiple_choice"}
        baseline_support_examples: list[dict[str, object]] = []
        baseline_support_scores: list[float] = []
        baseline_retriever: str | None = None
        requested_support_examples = int(baseline_budget_fields.get("support_examples", 0)) if use_baseline else 0
        if use_baseline:
            if baseline_family == "rag":
                (
                    baseline_support_examples,
                    baseline_support_scores,
                    baseline_retriever,
                ) = runtime.select_support_examples(dataset, example, requested_support_examples)
            elif baseline_family == "memory_bank":
                (
                    baseline_support_examples,
                    baseline_support_scores,
                    baseline_retriever,
                ) = runtime.select_support_examples(dataset, example, requested_support_examples)
            else:
                baseline_support_examples = _select_support_examples(
                    dataset,
                    example,
                    requested_support_examples,
                )
        if use_baseline:
            baseline_output = None
            similarity = 0.0
            generated_text = ""
            predicted_label = ""
            predicted_text = ""
            if baseline_family == "adapter" and not uses_candidate_selection:
                raise ValueError("Adapter baselines currently support only candidate-selection tasks.")
            if uses_candidate_selection:
                if evaluator.evaluator_type == "multiple_choice":
                    choices = example.get("choices", [])
                    if not choices:
                        raise ValueError("multiple_choice evaluation requires per-example `choices`.")
                    candidate_labels = [str(choice["label"]) for choice in choices]
                    candidate_texts = [str(choice["text"]) for choice in choices]
                else:
                    candidate_labels = [str(row["label"]) for row in dataset]
                    candidate_texts = [str(row["continuation"]) for row in dataset]
                baseline_kwargs = (
                    {"support_examples": baseline_support_examples}
                    if baseline_family in {"prompting", "meta_prompting", "rag", "lightthinker"}
                    else {
                        "support_examples": baseline_support_examples,
                        "support_scores": baseline_support_scores,
                    }
                    if baseline_family == "memory_bank"
                    else {}
                )
                baseline_output = runtime.predict_multiple_choice(
                    example,
                    candidate_labels=candidate_labels,
                    candidate_texts=candidate_texts,
                    **baseline_kwargs,
                )
                predicted_label = baseline_output.predicted_label
                predicted_text = baseline_output.predicted_text
                similarity = float(baseline_output.similarity or 0.0)
                score_payload = evaluator.evaluate_prediction(
                    {"label": predicted_label, "text": predicted_text},
                    example,
                )
            else:
                baseline_kwargs = (
                    {"support_examples": baseline_support_examples}
                    if baseline_family in {"prompting", "meta_prompting", "rag", "lightthinker"}
                    else {
                        "support_examples": baseline_support_examples,
                        "support_scores": baseline_support_scores,
                    }
                    if baseline_family == "memory_bank"
                    else {}
                )
                baseline_output = runtime.generate_text(
                    example,
                    **baseline_kwargs,
                )
                generated_text = baseline_output.predicted_text
                predicted_text = generated_text
                score_payload = evaluator.evaluate_prediction({"text": generated_text}, example)
            profiler.add_tokens(runtime.backbone.count_tokens(baseline_output.prompt))
            if generated_text:
                profiler.add_tokens(runtime.backbone.count_tokens(generated_text))
            forward = None
            gate_mean = 0.0
            active_queries = 0
            benchmark_metadata = {
                key: example[key]
                for key in (
                    "narrativeqa_view",
                    "story_chunk_pool_size",
                    "story_segments_materialized",
                    "story_selected_indexes",
                    "story_selection_strategy",
                    "story_runtime_segment_budget",
                    "story_runtime_selector",
                    "story_query_token_count",
                    "story_truncated_for_smoke",
                )
                if key in example
            }
            candidate_scores = baseline_output.candidate_scores
            prompt_text = baseline_output.prompt
            if baseline_support_scores:
                if baseline_family == "rag":
                    retrieval_support_score_means.append(sum(baseline_support_scores) / len(baseline_support_scores))
                if baseline_family == "memory_bank":
                    memory_bank_selection_score_means.append(sum(baseline_support_scores) / len(baseline_support_scores))
            thought_sketch = getattr(baseline_output, "thought_sketch", "")
            if thought_sketch:
                thought_sketch_token_counts.append(runtime.backbone.count_tokens(thought_sketch))
            memory_bank_entries = getattr(baseline_output, "memory_bank_entries", [])
            if memory_bank_entries:
                memory_bank_entry_counts.append(len(memory_bank_entries))
        else:
            if uses_candidate_selection:
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
                score_payload = {}
            profiler.add_tokens(runtime.backbone.count_tokens(forward.next_prompt))
            generated_text = runtime.backbone.generate(
                [forward.next_prompt],
                memory_tokens=forward.generation_memory,
            )[0]
            profiler.add_tokens(runtime.backbone.count_tokens(generated_text))
            if not uses_candidate_selection:
                score_payload = evaluator.evaluate_prediction({"text": generated_text}, example)
                predicted_text = generated_text
            gate_mean = float(forward.gating.mean().item())
            active_queries = int((forward.gating > 0.5).sum().item())
            benchmark_metadata = {
                key: example[key]
                for key in (
                    "narrativeqa_view",
                    "story_chunk_pool_size",
                    "story_segments_materialized",
                    "story_selected_indexes",
                    "story_selection_strategy",
                    "story_runtime_segment_budget",
                    "story_runtime_selector",
                    "story_query_token_count",
                    "story_truncated_for_smoke",
                )
                if key in example
            }
            candidate_scores = None
            prompt_text = forward.next_prompt
        is_correct = bool(score_payload["correct"])
        score_value = float(score_payload["score"])
        correct += int(is_correct)
        score_total += score_value
        similarities.append(similarity)
        gate_means.append(gate_mean)
        active_query_counts.append(active_queries)
        if forward is not None:
            segment_gate_means.extend(float(item["mean_gate"]) for item in forward.segment_stats)
            segment_active_query_counts.extend(int(item["active_queries"]) for item in forward.segment_stats)
        capability_name = str(score_payload.get("capability", example.get("capability", "")))
        if capability_name:
            capability_scores.setdefault(capability_name, []).append(score_value)
            capability_metric_names[capability_name] = str(
                score_payload.get("capability_metric_name", example.get("capability_metric_name", evaluator.metric_name))
            )
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
                "score": score_value,
                "normalized_prediction": score_payload["normalized_prediction"],
                "normalized_reference": score_payload["normalized_reference"],
                "evaluator_type": evaluator.evaluator_type,
                "extra_metrics": score_payload.get("extra_metrics", {}),
                "capability": capability_name or None,
                "capability_metric_name": score_payload.get(
                    "capability_metric_name",
                    example.get("capability_metric_name"),
                ),
                "similarity": similarity,
                "baseline_family": baseline_cfg.get("family") if use_baseline else None,
                "baseline_mode": baseline_cfg.get("mode") if use_baseline else None,
                "baseline_prompt": prompt_text if use_baseline else None,
                "baseline_support_ids": [row["id"] for row in baseline_support_examples] if use_baseline else [],
                "baseline_support_scores": baseline_support_scores if use_baseline else [],
                "baseline_retriever": baseline_retriever if use_baseline else None,
                "baseline_memory_bank_entries": (
                    getattr(baseline_output, "memory_bank_entries", None) if use_baseline else None
                ),
                "baseline_memory_bank_selector": (
                    baseline_retriever if use_baseline and baseline_family == "memory_bank" else None
                ),
                "baseline_memory_bank_eviction_policy": (
                    getattr(runtime, "eviction_policy", None) if use_baseline and baseline_family == "memory_bank" else None
                ),
                "lightthinker_compression_prompt": (
                    getattr(baseline_output, "compression_prompt", None) if use_baseline else None
                ),
                "lightthinker_thought_sketch": getattr(baseline_output, "thought_sketch", None) if use_baseline else None,
                "candidate_scores": candidate_scores,
                "gating_mode": None if use_baseline else runtime.reader.gating_mode,
                "conditioning_mode": None if use_baseline else runtime.reader.conditioning_mode,
                "attention_mode": None if use_baseline else runtime.reader.attention_mode,
                "gated_add_scale": None if use_baseline else runtime.reader.gated_add_scale,
                "query_residual_scale": None if use_baseline else runtime.reader.query_residual_scale,
                "gates": None if use_baseline else [float(value) for value in forward.gating.squeeze(0).tolist()],
                "mean_gate": gate_mean,
                "active_queries": active_queries,
                "conditioning": None if use_baseline else forward.conditioning,
                "injection_position": None if use_baseline else runtime.injector.position,
                "injection_anchors": [] if use_baseline else forward.injection_anchors,
                "segment_stats": [] if use_baseline else forward.segment_stats,
                "generated_text": generated_text,
                "benchmark_metadata": benchmark_metadata or None,
            }
        )

    profile_metrics = profiler.finalize()
    mean_accuracy = correct / max_examples
    mean_score = score_total / max_examples
    metrics = {
        "mode": "eval_baseline" if use_baseline else "eval",
        "examples_evaluated": max_examples,
        "accuracy": mean_accuracy,
        "mean_score": mean_score,
        evaluator.metric_name: mean_score,
        "benchmark_id": config["task"].get("benchmark_id"),
        "task_domain": config["task"].get("domain"),
        "smoke_subset": config["task"].get("smoke_subset"),
        "evaluator_type": evaluator.evaluator_type,
        "mean_similarity": sum(similarities) / len(similarities),
        "gating_mode": "disabled" if use_baseline else runtime.reader.gating_mode,
        "conditioning_mode": "disabled" if use_baseline else runtime.reader.conditioning_mode,
        "attention_mode": "disabled" if use_baseline else runtime.reader.attention_mode,
        "gated_add_scale": None if use_baseline else runtime.reader.gated_add_scale,
        "query_residual_scale": None if use_baseline else runtime.reader.query_residual_scale,
        "mean_gate": sum(gate_means) / len(gate_means),
        "mean_active_queries": sum(active_query_counts) / len(active_query_counts),
        "mean_segment_gate": (
            sum(segment_gate_means) / len(segment_gate_means) if segment_gate_means else 0.0
        ),
        "mean_segment_active_queries": (
            sum(segment_active_query_counts) / len(segment_active_query_counts)
            if segment_active_query_counts
            else 0.0
        ),
        "injection_position": None if use_baseline else runtime.injector.position,
        "conditioning_schema": {
            "domain_name": "disabled" if use_baseline else runtime.conditioning_cfg["domain_key"],
            "task_name": (
                "disabled"
                if use_baseline
                else ("config.task.name" if runtime.conditioning_cfg["include_task_name"] else "disabled")
            ),
        },
        "backbone": config["backbone"]["name"],
        "metric_name": config["task"]["metric_name"],
        **profile_metrics,
    }
    if use_baseline:
        metrics.update(baseline_budget_fields)
        if baseline_retriever is not None:
            metrics["baseline_retriever"] = baseline_retriever
    if retrieval_support_score_means:
        metrics["mean_support_retrieval_score"] = (
            sum(retrieval_support_score_means) / len(retrieval_support_score_means)
        )
    if thought_sketch_token_counts:
        metrics["mean_thought_sketch_tokens"] = (
            sum(thought_sketch_token_counts) / len(thought_sketch_token_counts)
        )
    if memory_bank_entry_counts:
        metrics["mean_memory_bank_entry_count"] = (
            sum(memory_bank_entry_counts) / len(memory_bank_entry_counts)
        )
    if memory_bank_selection_score_means:
        metrics["mean_memory_bank_selection_score"] = (
            sum(memory_bank_selection_score_means) / len(memory_bank_selection_score_means)
        )
    if baseline_family == "memory_bank":
        metrics["memory_bank_selector"] = str(getattr(runtime, "selector", "overlap_then_recency"))
        metrics["memory_bank_eviction_policy"] = str(getattr(runtime, "eviction_policy", "topk"))
    narrativeqa_runtime_cfg = config["task"].get("narrativeqa_runtime")
    if isinstance(narrativeqa_runtime_cfg, dict):
        metrics["story_runtime_selector"] = str(narrativeqa_runtime_cfg.get("selector", "question_aware"))
        metrics["story_runtime_segment_budget"] = int(
            narrativeqa_runtime_cfg.get("segment_budget", 0)
        )
    if capability_scores:
        metrics["capability_scores"] = {
            capability: sum(values) / len(values) for capability, values in sorted(capability_scores.items())
        }
        metrics["capability_metric_names"] = {
            capability: capability_metric_names[capability] for capability in sorted(capability_metric_names)
        }
    write_json(Path(args.output_dir) / "metrics.json", metrics)
    write_jsonl(Path(args.output_dir) / "predictions.jsonl", predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
