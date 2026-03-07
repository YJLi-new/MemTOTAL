from __future__ import annotations

import copy
import csv
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from memtotal.data import (
    EpisodeSampler,
    build_meta_manifest,
    load_meta_grouped_examples,
    split_target_domain_examples,
    validate_meta_split,
)
from memtotal.pipeline import MemoryRuntime
from memtotal.tasks import TaskEvaluator, get_task_spec
from memtotal.utils.io import write_json
from memtotal.utils.profiling import ProfileTracker


def _load_meta_config(config: dict) -> dict:
    meta_cfg = config["task"]["meta"]
    return {
        "general_domains": list(meta_cfg["general_domains"]),
        "source_domains": list(meta_cfg["source_domains"]),
        "target_domain": str(meta_cfg["target_domain"]),
        "support_size": int(meta_cfg["support_size"]),
        "query_size": int(meta_cfg["query_size"]),
        "sampling_policy": str(meta_cfg.get("sampling_policy", "stratified_labels")),
    }


def _flatten_domains(grouped_examples: dict[str, list[dict[str, str]]], domains: list[str]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for domain in domains:
        flattened.extend(grouped_examples[domain])
    return flattened


def _build_label_prototypes(
    runtime: MemoryRuntime,
    examples: list[dict[str, str]],
) -> tuple[torch.Tensor, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in examples:
        grouped.setdefault(str(row["label"]), []).append(row["continuation"])
    labels = sorted(grouped)
    states = []
    for label in labels:
        states.append(runtime.backbone.summarize_texts(grouped[label]).mean(dim=0))
    return torch.stack(states, dim=0), labels


def _resolve_query_objective(config: dict) -> str:
    query_objective = str(config["runtime"].get("query_objective", "label_prototype"))
    if query_objective not in {"label_prototype", "continuation_retrieval"}:
        raise ValueError(
            f"Unsupported runtime.query_objective={query_objective}. "
            "Expected one of label_prototype, continuation_retrieval."
        )
    return query_objective


def _resolve_retrieval_negative_count(config: dict) -> int:
    return int(config["runtime"].get("retrieval_negative_count", 7))


def _resolve_target_eval_repeats(config: dict) -> int:
    return max(1, int(config["runtime"].get("target_eval_repeats", 1)))


def _resolve_target_episode_repeats(config: dict) -> int:
    return max(1, int(config["runtime"].get("target_episode_repeats", 1)))


def _resolve_target_episode_policy(config: dict) -> str:
    policy = str(config["runtime"].get("target_episode_policy", "independent"))
    if policy not in {"independent", "aggregate_support"}:
        raise ValueError(
            f"Unsupported runtime.target_episode_policy={policy}. "
            "Expected one of independent, aggregate_support."
        )
    return policy


def _resolve_target_support_weighting(config: dict) -> str:
    weighting = str(config["runtime"].get("target_support_weighting", "uniform"))
    if weighting not in {"uniform", "proxy_softmax", "proxy_top1"}:
        raise ValueError(
            f"Unsupported runtime.target_support_weighting={weighting}. "
            "Expected one of uniform, proxy_softmax, proxy_top1."
        )
    return weighting


def _resolve_target_split_policy(config: dict) -> str:
    policy = str(config["runtime"].get("target_split_policy", "random"))
    if policy not in {"random", "proxy_topk_support", "proxy_bottomk_support"}:
        raise ValueError(
            f"Unsupported runtime.target_split_policy={policy}. "
            "Expected one of random, proxy_topk_support, proxy_bottomk_support."
        )
    return policy


def _resolve_target_support_bank_size_spec(config: dict) -> str | int:
    spec = config["runtime"].get("target_support_bank_size", "auto")
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("runtime.target_support_bank_size must be positive when provided as an integer.")
        return spec
    if isinstance(spec, str) and spec.isdigit():
        return int(spec)
    spec_text = str(spec)
    if spec_text not in {"auto", "max_shot", "all_non_holdout"}:
        raise ValueError(
            f"Unsupported runtime.target_support_bank_size={spec_text}. "
            "Expected a positive integer or one of auto, max_shot, all_non_holdout."
        )
    return spec_text


def _resolve_target_support_negative_pool(config: dict) -> str:
    pool = str(config["runtime"].get("target_support_negative_pool", "support_bank"))
    if pool not in {"support_bank", "source_plus_support_bank"}:
        raise ValueError(
            f"Unsupported runtime.target_support_negative_pool={pool}. "
            "Expected one of support_bank, source_plus_support_bank."
        )
    return pool


def _resolve_target_support_negative_sampler(config: dict) -> str:
    sampler = str(config["runtime"].get("target_support_negative_sampler", "deterministic_id"))
    if sampler not in {"deterministic_id", "hard_by_continuation"}:
        raise ValueError(
            f"Unsupported runtime.target_support_negative_sampler={sampler}. "
            "Expected one of deterministic_id, hard_by_continuation."
        )
    return sampler


def _resolve_artifact_path(resume: str | None, expected_name: str) -> Path:
    if not resume:
        raise ValueError(f"This stage requires --resume pointing to '{expected_name}' or its parent run dir.")
    path = Path(resume).resolve()
    if path.is_file():
        return path
    candidate = path / expected_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve required artifact '{expected_name}' from --resume={resume}.")


def _resolve_stage_c_adaptation_target(config: dict) -> str:
    adaptation_target = str(config["runtime"].get("adaptation_target", "q_only"))
    if adaptation_target not in {"q_only", "w_only", "w_plus_q"}:
        raise ValueError(
            f"Unsupported Stage C adaptation_target={adaptation_target}. "
            "Expected one of q_only, w_only, w_plus_q."
        )
    return adaptation_target


def _resolve_stage_b_query_learning_mode(config: dict) -> str:
    query_learning_mode = str(config["runtime"].get("query_learning_mode", "meta_trained"))
    aliases = {
        "meta": "meta_trained",
        "meta_trained": "meta_trained",
        "non_meta": "non_meta_multitask",
        "non_meta_multitask": "non_meta_multitask",
        "random": "random",
    }
    normalized = aliases.get(query_learning_mode)
    if normalized is None:
        raise ValueError(
            f"Unsupported Stage B query_learning_mode={query_learning_mode}. "
            "Expected one of meta_trained, non_meta_multitask, random."
        )
    return normalized


def _resolve_stage_b_trainable_target(config: dict) -> str:
    trainable_target = str(config["runtime"].get("stage_b_trainable_target", "queries_plus_fuser"))
    aliases = {
        "queries_only": "queries_only",
        "q_only": "queries_only",
        "queries_plus_fuser": "queries_plus_fuser",
        "q_plus_fuser": "queries_plus_fuser",
    }
    normalized = aliases.get(trainable_target)
    if normalized is None:
        raise ValueError(
            f"Unsupported Stage B stage_b_trainable_target={trainable_target}. "
            "Expected one of queries_only, queries_plus_fuser."
        )
    return normalized


def _resolve_expected_stage_c_query_learning_mode(config: dict) -> str | None:
    expected = config["runtime"].get("expected_query_learning_mode")
    if expected is None:
        return None
    normalized = _resolve_stage_b_query_learning_mode({"runtime": {"query_learning_mode": expected}})
    return normalized


def _configure_stage_c_trainables(
    runtime: MemoryRuntime,
    adaptation_target: str,
) -> tuple[list[torch.nn.Parameter], str]:
    # Stage C follows the paper contract: default adaptation is query-only.
    runtime.writer.freeze()
    runtime.reader.freeze()
    runtime.fuser.freeze()

    if adaptation_target == "q_only":
        runtime.reader.queries.requires_grad_(True)
        return [runtime.reader.queries], "reader.queries"
    if adaptation_target == "w_only":
        runtime.writer.unfreeze()
        return list(runtime.writer.parameters()), "writer"

    runtime.writer.unfreeze()
    runtime.reader.queries.requires_grad_(True)
    return [*runtime.writer.parameters(), runtime.reader.queries], "writer+reader.queries"


def _count_unique_parameters(parameters: list[torch.nn.Parameter]) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        pointer = parameter.data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += int(parameter.numel())
    return total


def _parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum().item())
    return total ** 0.5


def _snapshot_parameter_tensors(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _parameter_update_stats(
    before: list[torch.Tensor],
    after: list[torch.nn.Parameter],
) -> tuple[float, float]:
    max_abs = 0.0
    total = 0.0
    for previous, current in zip(before, after):
        delta = current.detach() - previous
        if delta.numel() == 0:
            continue
        max_abs = max(max_abs, float(delta.abs().max().item()))
        total += float(delta.pow(2).sum().item())
    return max_abs, total ** 0.5


def _stage_b_trainable_parameters(
    runtime: MemoryRuntime,
    trainable_target: str,
) -> tuple[list[torch.nn.Parameter], str]:
    runtime.writer.freeze()
    runtime.reader.unfreeze()
    runtime.fuser.freeze()
    if trainable_target == "queries_only":
        runtime.reader.queries.requires_grad_(True)
        return [runtime.reader.queries], "reader.queries"
    if trainable_target == "queries_plus_fuser":
        runtime.fuser.unfreeze()
        return [runtime.reader.queries, *runtime.fuser.parameters()], "reader.queries+fuser"
    raise ValueError(
        f"Unsupported Stage B trainable_target={trainable_target}. "
        "Expected one of queries_only, queries_plus_fuser."
    )


def _example_loss(runtime: MemoryRuntime, example: dict[str, str]) -> torch.Tensor:
    forward = runtime.forward_example(example)
    return F.mse_loss(forward.predicted_state, forward.target_state)


def _classification_loss(
    runtime: MemoryRuntime,
    example: dict[str, str],
    candidate_states: torch.Tensor,
    candidate_labels: list[str],
) -> torch.Tensor:
    forward = runtime.forward_example(example)
    memory_summary = runtime.summarize_memory_short(forward.memory_short)
    scores = runtime.score_candidates(memory_summary, candidate_states)
    gold_index = candidate_labels.index(example["label"])
    return F.cross_entropy(scores.unsqueeze(0), torch.tensor([gold_index], dtype=torch.long))


def _mean_loss(runtime: MemoryRuntime, examples: list[dict[str, str]]) -> float:
    losses = [_example_loss(runtime, example) for example in examples]
    return float(torch.stack(losses).mean().item())


def _mean_classification_loss(
    runtime: MemoryRuntime,
    examples: list[dict[str, str]],
    candidate_states: torch.Tensor,
    candidate_labels: list[str],
) -> float:
    losses = [
        _classification_loss(runtime, example, candidate_states, candidate_labels)
        for example in examples
    ]
    return float(torch.stack(losses).mean().item())


def _mean_classification_loss_by_domain(
    runtime: MemoryRuntime,
    examples: list[dict[str, str]],
    candidate_bank: dict[str, tuple[torch.Tensor, list[str]]],
) -> float:
    losses = [
        _classification_loss(
            runtime,
            example,
            candidate_states=candidate_bank[str(example["domain"])][0],
            candidate_labels=candidate_bank[str(example["domain"])][1],
        )
        for example in examples
    ]
    return float(torch.stack(losses).mean().item())


def _compute_accuracy(
    runtime: MemoryRuntime,
    examples: list[dict[str, str]],
    candidate_states: torch.Tensor,
    candidate_labels: list[str],
) -> float:
    correct = 0
    for example in examples:
        forward = runtime.forward_example(example)
        memory_summary = runtime.summarize_memory_short(forward.memory_short)
        scores = runtime.score_candidates(memory_summary, candidate_states)
        predicted_label = candidate_labels[int(torch.argmax(scores).item())]
        correct += int(predicted_label == example["label"])
    return correct / len(examples)


def _compute_accuracy_by_domain(
    runtime: MemoryRuntime,
    examples: list[dict[str, str]],
    candidate_bank: dict[str, tuple[torch.Tensor, list[str]]],
) -> float:
    correct = 0
    for example in examples:
        candidate_states, candidate_labels = candidate_bank[str(example["domain"])]
        forward = runtime.forward_example(example)
        memory_summary = runtime.summarize_memory_short(forward.memory_short)
        scores = runtime.score_candidates(memory_summary, candidate_states)
        predicted_label = candidate_labels[int(torch.argmax(scores).item())]
        correct += int(predicted_label == example["label"])
    return correct / len(examples)


def _build_task_evaluator_for_example(example: dict[str, Any]) -> TaskEvaluator:
    benchmark_id = str(example["benchmark_id"])
    spec = get_task_spec(benchmark_id)
    return TaskEvaluator(
        evaluator_type=str(example.get("evaluator_type", spec.evaluator_type)),
        metric_name=str(example.get("metric_name", spec.metric_name)),
        normalizer=str(example.get("normalizer", spec.normalizer)),
        benchmark_id=benchmark_id,
    )


def _evaluate_task_example(
    runtime: MemoryRuntime,
    example: dict[str, Any],
) -> dict[str, object]:
    evaluator = _build_task_evaluator_for_example(example)
    if evaluator.evaluator_type == "multiple_choice":
        choices = example.get("choices", [])
        if not choices:
            raise ValueError("multiple_choice task examples require per-example choices.")
        candidate_labels = [str(choice["label"]) for choice in choices]
        candidate_texts = [str(choice["text"]) for choice in choices]
        candidate_states = runtime.backbone.summarize_texts(candidate_texts)
        forward = runtime.forward_example(example)
        memory_summary = runtime.summarize_memory_short(forward.memory_short)
        scores = runtime.score_candidates(memory_summary, candidate_states)
        probabilities = torch.softmax(scores, dim=0)
        predicted_index = int(torch.argmax(scores).item())
        predicted_label = candidate_labels[predicted_index]
        predicted_text = candidate_texts[predicted_index]
        score_payload = evaluator.evaluate_prediction(
            {"label": predicted_label, "text": predicted_text},
            example,
        )
        gold_label = str(example["label"])
        gold_index = candidate_labels.index(gold_label)
        if len(candidate_labels) > 1:
            best_other_score = max(
                float(score.item()) for index, score in enumerate(scores) if index != gold_index
            )
        else:
            best_other_score = float(scores[gold_index].item())
        return {
            "task_score": float(score_payload["score"]),
            "task_metric_name": evaluator.metric_name,
            "task_proxy_score": float(probabilities[gold_index].item()),
            "task_proxy_name": "gold_choice_probability",
            "task_margin": float(scores[gold_index].item()) - best_other_score,
        }

    forward = runtime.forward_example(example)
    generated_text = runtime.backbone.generate(
        [forward.next_prompt],
        memory_tokens=forward.generation_memory,
    )[0]
    score_payload = evaluator.evaluate_prediction({"text": generated_text}, example)
    score = float(score_payload["score"])
    return {
        "task_score": score,
        "task_metric_name": evaluator.metric_name,
        "task_proxy_score": score,
        "task_proxy_name": evaluator.metric_name,
        "task_margin": 0.0,
    }


def _mean_task_score(
    runtime: MemoryRuntime,
    examples: list[dict[str, Any]],
) -> dict[str, object]:
    task_scores: list[float] = []
    metric_names: list[str] = []
    proxy_scores: list[float] = []
    proxy_names: list[str] = []
    task_margins: list[float] = []
    for example in examples:
        metrics = _evaluate_task_example(runtime, example)
        task_scores.append(float(metrics["task_score"]))
        metric_names.append(str(metrics["task_metric_name"]))
        proxy_scores.append(float(metrics["task_proxy_score"]))
        proxy_names.append(str(metrics["task_proxy_name"]))
        task_margins.append(float(metrics["task_margin"]))
    if not task_scores:
        return {
            "task_score": 0.0,
            "task_metric_name": "none",
            "task_proxy_score": 0.0,
            "task_proxy_name": "none",
            "task_margin": 0.0,
        }
    unique_metric_names = sorted(set(metric_names))
    resolved_metric_name = unique_metric_names[0] if len(unique_metric_names) == 1 else "mean_score"
    unique_proxy_names = sorted(set(proxy_names))
    resolved_proxy_name = unique_proxy_names[0] if len(unique_proxy_names) == 1 else "mean_proxy_score"
    return {
        "task_score": sum(task_scores) / len(task_scores),
        "task_metric_name": resolved_metric_name,
        "task_proxy_score": sum(proxy_scores) / len(proxy_scores),
        "task_proxy_name": resolved_proxy_name,
        "task_margin": sum(task_margins) / len(task_margins),
    }


def _resolve_retrieval_candidates(
    example: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    *,
    negative_count: int,
    runtime: MemoryRuntime | None = None,
    negative_sampler: str = "deterministic_id",
) -> list[dict[str, Any]]:
    negatives = [
        row
        for row in sorted(candidate_pool, key=lambda item: str(item["id"]))
        if str(row["id"]) != str(example["id"])
    ]
    if negative_sampler == "hard_by_continuation" and negatives:
        if runtime is None:
            raise ValueError("runtime is required when negative_sampler='hard_by_continuation'.")
        positive_state = runtime.backbone.summarize_texts([str(example["continuation"])]).squeeze(0)
        candidate_states = runtime.backbone.summarize_texts(
            [str(row["continuation"]) for row in negatives]
        )
        positive_state = F.normalize(positive_state.unsqueeze(0), dim=1).squeeze(0)
        candidate_states = F.normalize(candidate_states, dim=1)
        scores = torch.mv(candidate_states, positive_state)
        ranked_negatives = sorted(
            zip(scores.tolist(), (str(row["id"]) for row in negatives), negatives, strict=False),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        selected_negatives = [item[2] for item in ranked_negatives[: min(negative_count, len(ranked_negatives))]]
    else:
        selected_negatives = negatives[: min(negative_count, len(negatives))]
    return [example, *selected_negatives]


def _exclude_support_from_query_pool(
    domain_examples: list[dict[str, Any]],
    support_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not support_examples:
        return list(domain_examples)
    support_ids = {str(row["id"]) for row in support_examples}
    filtered = [row for row in domain_examples if str(row["id"]) not in support_ids]
    return filtered or list(domain_examples)


def _sample_target_eval_query_sets(
    candidate_pool: list[dict[str, Any]],
    *,
    query_size: int,
    repeats: int,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    query_sets: list[list[dict[str, Any]]] = []
    for repeat_index in range(repeats):
        rng = random.Random(seed + (7919 * repeat_index))
        shuffled = list(candidate_pool)
        rng.shuffle(shuffled)
        query_sets.append(shuffled[: min(query_size, len(shuffled))])
    return query_sets, candidate_pool


def _sample_target_support_bank(
    domain_examples: list[dict[str, Any]],
    *,
    support_bank_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(domain_examples)
    rng.shuffle(shuffled)
    return shuffled[:support_bank_size]


def _compute_target_support_bank_size(
    *,
    support_bank_size_spec: str | int,
    max_shot: int,
    domain_size: int,
    query_size: int,
    retrieval_negative_count: int,
) -> int:
    max_available = max(1, domain_size - query_size)
    if isinstance(support_bank_size_spec, int):
        requested_size = support_bank_size_spec
    elif support_bank_size_spec == "max_shot":
        requested_size = max_shot
    elif support_bank_size_spec == "all_non_holdout":
        requested_size = max_available
    else:
        requested_size = max(max_shot, retrieval_negative_count + 1)
    return min(max_available, max(1, requested_size))


def _build_stage_c_episode_states(
    *,
    runtime: MemoryRuntime,
    grouped_examples: dict[str, list[dict[str, Any]]],
    manifest: dict[str, object],
    domain_examples: list[dict[str, Any]],
    shot: int,
    seed: int,
    max_shot: int,
    target_episode_repeats: int,
    target_eval_repeats: int,
    target_split_policy: str,
    target_support_bank_size_spec: str | int,
    retrieval_negative_count: int,
    target_support_negative_pool: str,
    target_support_negative_sampler: str,
) -> list[dict[str, object]]:
    episode_states: list[dict[str, object]] = []
    source_negative_examples = _flatten_domains(grouped_examples, list(manifest["source_domains"]))
    for episode_index in range(target_episode_repeats):
        episode_seed = seed + (episode_index * 16127)
        support_bank_size = _compute_target_support_bank_size(
            support_bank_size_spec=target_support_bank_size_spec,
            max_shot=max_shot,
            domain_size=len(domain_examples),
            query_size=int(manifest["query_size"]),
            retrieval_negative_count=retrieval_negative_count,
        )
        candidate_support_examples = _sample_target_support_bank(
            domain_examples,
            support_bank_size=support_bank_size,
            seed=episode_seed,
        )
        support_examples = _select_target_support_examples(
            runtime,
            candidate_support_examples,
            shot=shot,
            split_policy=target_split_policy,
        )
        eval_candidate_pool = [
            row for row in domain_examples if str(row["id"]) not in {str(item["id"]) for item in candidate_support_examples}
        ]
        if len(eval_candidate_pool) < int(manifest["query_size"]):
            eval_candidate_pool = list(domain_examples)
        eval_query_sets, query_candidate_pool = _sample_target_eval_query_sets(
            eval_candidate_pool,
            query_size=int(manifest["query_size"]),
            repeats=target_eval_repeats,
            seed=episode_seed + 7919,
        )
        support_candidate_pool = list(candidate_support_examples)
        if target_support_negative_pool == "source_plus_support_bank":
            support_candidate_pool.extend(source_negative_examples)
        episode_states.append(
            {
                "episode_seed": episode_seed,
                "support_examples": support_examples,
                "support_candidate_examples": candidate_support_examples,
                "eval_query_sets": eval_query_sets,
                "query_candidate_pool": query_candidate_pool,
                "support_candidate_pool": support_candidate_pool,
                "target_support_bank_size": support_bank_size,
                "target_support_negative_sampler": target_support_negative_sampler,
            }
        )
    return episode_states


def _resolve_episode_support_weights(
    episode_proxy_scores: list[float],
    *,
    weighting: str,
) -> list[float]:
    if not episode_proxy_scores:
        return []
    if weighting == "uniform":
        uniform = 1.0 / len(episode_proxy_scores)
        return [uniform for _ in episode_proxy_scores]
    if weighting == "proxy_top1":
        best_index = max(range(len(episode_proxy_scores)), key=lambda idx: episode_proxy_scores[idx])
        return [1.0 if idx == best_index else 0.0 for idx in range(len(episode_proxy_scores))]
    logits = torch.tensor(episode_proxy_scores, dtype=torch.float32)
    weights = torch.softmax(logits, dim=0)
    return [float(value.item()) for value in weights]


def _score_target_support_candidate(
    runtime: MemoryRuntime,
    example: dict[str, Any],
) -> float:
    metrics = _evaluate_task_example(runtime, example)
    return float(metrics["task_proxy_score"])


def _select_target_support_examples(
    runtime: MemoryRuntime,
    candidate_examples: list[dict[str, Any]],
    *,
    shot: int,
    split_policy: str,
) -> list[dict[str, Any]]:
    if shot <= 0:
        return []
    if split_policy == "random":
        return list(candidate_examples[:shot])
    scored_examples = [
        (
            _score_target_support_candidate(runtime, example),
            str(example["id"]),
            example,
        )
        for example in candidate_examples
    ]
    reverse = split_policy == "proxy_topk_support"
    scored_examples.sort(key=lambda item: (item[0], item[1]), reverse=reverse)
    return [item[2] for item in scored_examples[:shot]]


def _continuation_retrieval_loss(
    runtime: MemoryRuntime,
    example: dict[str, Any],
    *,
    candidate_pool: list[dict[str, Any]],
    negative_count: int,
    negative_sampler: str = "deterministic_id",
) -> tuple[torch.Tensor, float]:
    candidate_examples = _resolve_retrieval_candidates(
        example,
        candidate_pool,
        negative_count=negative_count,
        runtime=runtime,
        negative_sampler=negative_sampler,
    )
    forward = runtime.forward_example(example)
    memory_summary = runtime.summarize_memory_short(forward.memory_short)
    candidate_states = runtime.backbone.summarize_texts(
        [str(row["continuation"]) for row in candidate_examples]
    )
    scores = runtime.score_candidates(memory_summary, candidate_states)
    loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([0], dtype=torch.long))
    accuracy = float(int(torch.argmax(scores).item()) == 0)
    return loss, accuracy


def _mean_continuation_retrieval_metrics(
    runtime: MemoryRuntime,
    examples: list[dict[str, Any]],
    *,
    candidate_pool_resolver,
    negative_count: int,
) -> tuple[float, float]:
    losses: list[torch.Tensor] = []
    accuracies: list[float] = []
    for example in examples:
        loss, accuracy = _continuation_retrieval_loss(
            runtime,
            example,
            candidate_pool=candidate_pool_resolver(example),
            negative_count=negative_count,
        )
        losses.append(loss)
        accuracies.append(accuracy)
    return float(torch.stack(losses).mean().item()), sum(accuracies) / len(accuracies)


def _stage_c_row_key(row: dict[str, object]) -> tuple[float, float, int, int]:
    return (
        float(row.get("task_score", row.get("query_accuracy", 0.0))),
        float(row.get("task_proxy_score", row.get("task_score", row.get("query_accuracy", 0.0)))),
        -float(row.get("objective_loss", row.get("query_loss", 0.0))),
        int(row["shot"]),
        int(row["step"]),
    )


def _save_stage_b_state(
    *,
    output_dir: Path,
    runtime: MemoryRuntime,
    seed: int,
    source_domains: list[str],
    query_learning_mode: str,
    query_objective: str,
    stage_b_trainable_target: str,
    writer_checkpoint: Path,
) -> Path:
    queries_path = output_dir / "queries_meta_init.pt"
    torch.save(
        {
            "reader_state": runtime.reader.state_dict(),
            "fuser_state": runtime.fuser.state_dict(),
            "seed": seed,
            "writer_checkpoint": str(writer_checkpoint.resolve()),
            "source_domains": source_domains,
            "query_learning_mode": query_learning_mode,
            "query_objective": query_objective,
            "stage_b_trainable_target": stage_b_trainable_target,
        },
        queries_path,
    )
    return queries_path


def _build_meta_context(config: dict) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    meta_cfg = _load_meta_config(config)
    task_cfg = config["task"]
    grouped_examples = load_meta_grouped_examples(task_cfg)
    validate_meta_split(grouped_examples, **meta_cfg)
    manifest_kwargs = {
        "grouped_examples": grouped_examples,
        **meta_cfg,
    }
    dataset_sources = task_cfg.get("meta", {}).get("dataset_sources")
    if dataset_sources:
        manifest = build_meta_manifest(
            dataset_sources=dataset_sources,
            **manifest_kwargs,
        )
    else:
        manifest = build_meta_manifest(
            dataset_path=task_cfg["dataset_path"],
            **manifest_kwargs,
        )
    return grouped_examples, manifest


def run_stage_a(
    *,
    config: dict,
    seed: int,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, object]:
    grouped_examples, manifest = _build_meta_context(config)
    write_json(output_dir / "meta_data_manifest.json", manifest)

    runtime = MemoryRuntime(config=config, seed=seed)
    runtime.reader.freeze()
    runtime.fuser.freeze()
    optimizer = torch.optim.Adam(runtime.writer.parameters(), lr=float(config["runtime"]["learning_rate"]))
    general_examples = _flatten_domains(grouped_examples, manifest["general_domains"])
    steps = min(len(general_examples), 2 if dry_run else int(config["runtime"]["train_steps"]))
    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="train")
    events = []

    for step in range(steps):
        example = general_examples[step % len(general_examples)]
        profiler.add_example()
        optimizer.zero_grad()
        loss = _example_loss(runtime, example)
        loss.backward()
        optimizer.step()
        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
        events.append(
            {
                "step": step,
                "domain": example["domain"],
                "loss": float(loss.item()),
            }
        )

    writer_path = runtime.writer.save_to(output_dir / "writer.ckpt")
    metrics = {
        "mode": "train",
        "training_stage": "stage_a",
        "examples_seen": steps,
        "trainable_module": "writer",
        "final_loss": events[-1]["loss"],
        "mean_loss": sum(item["loss"] for item in events) / len(events),
        "general_domains": manifest["general_domains"],
        "dataset_sha256": manifest["dataset_sha256"],
        "writer_checkpoint": str(writer_path.resolve()),
        **profiler.finalize(),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "stage_a_events.json", {"events": events})
    return metrics


def run_stage_b(
    *,
    config: dict,
    seed: int,
    output_dir: Path,
    resume: str | None,
    dry_run: bool,
) -> dict[str, object]:
    grouped_examples, manifest = _build_meta_context(config)
    write_json(output_dir / "meta_data_manifest.json", manifest)

    writer_path = _resolve_artifact_path(resume, "writer.ckpt")
    runtime = MemoryRuntime(config=config, seed=seed)
    runtime.writer.load_from(writer_path)
    query_learning_mode = _resolve_stage_b_query_learning_mode(config)
    stage_b_trainable_target = _resolve_stage_b_trainable_target(config)
    query_objective = _resolve_query_objective(config)
    retrieval_negative_count = _resolve_retrieval_negative_count(config)
    trainable_parameters, trainable_module = _stage_b_trainable_parameters(
        runtime,
        stage_b_trainable_target,
    )
    shutil.copy2(writer_path, output_dir / "writer.ckpt")
    source_examples = _flatten_domains(grouped_examples, manifest["source_domains"])
    source_eval_label_space = "per_domain"
    source_eval_candidate_pool = "per_domain"
    domain_candidate_bank: dict[str, tuple[torch.Tensor, list[str]]] = {}
    if query_objective == "label_prototype":
        domain_candidate_bank = {
            domain: _build_label_prototypes(runtime, grouped_examples[domain])
            for domain in manifest["source_domains"]
        }

    def _source_pool_resolver(example: dict[str, Any]) -> list[dict[str, Any]]:
        if query_learning_mode == "meta_trained":
            return list(grouped_examples[str(example["domain"])])
        return source_examples

    if query_learning_mode == "random":
        profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="train")
        for example in source_examples:
            profiler.add_example()
            profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
            profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
        if query_objective == "label_prototype":
            source_zero_shot_query_loss = _mean_classification_loss_by_domain(
                runtime,
                source_examples,
                candidate_bank=domain_candidate_bank,
            )
            source_zero_shot_query_accuracy = _compute_accuracy_by_domain(
                runtime,
                source_examples,
                candidate_bank=domain_candidate_bank,
            )
            source_eval_task_score = source_zero_shot_query_accuracy
            source_eval_metric_name = "accuracy"
        else:
            source_eval_candidate_pool = "global_source"
            source_zero_shot_query_loss, source_zero_shot_query_accuracy = _mean_continuation_retrieval_metrics(
                runtime,
                source_examples,
                candidate_pool_resolver=lambda _: source_examples,
                negative_count=retrieval_negative_count,
            )
            source_eval_metrics = _mean_task_score(runtime, source_examples)
            source_eval_task_score = float(source_eval_metrics["task_score"])
            source_eval_metric_name = str(source_eval_metrics["task_metric_name"])
            source_eval_task_proxy_score = float(source_eval_metrics["task_proxy_score"])
            source_eval_task_proxy_name = str(source_eval_metrics["task_proxy_name"])
            source_eval_task_margin = float(source_eval_metrics["task_margin"])
        if query_objective == "label_prototype":
            source_eval_task_proxy_score = source_eval_task_score
            source_eval_task_proxy_name = source_eval_metric_name
            source_eval_task_margin = 0.0
        queries_path = _save_stage_b_state(
            output_dir=output_dir,
            runtime=runtime,
            seed=seed,
            source_domains=manifest["source_domains"],
            query_learning_mode=query_learning_mode,
            query_objective=query_objective,
            stage_b_trainable_target=stage_b_trainable_target,
            writer_checkpoint=output_dir / "writer.ckpt",
        )
        metrics = {
            "mode": "train",
            "training_stage": "stage_b",
            "query_learning_mode": query_learning_mode,
            "query_objective": query_objective,
            "retrieval_negative_count": retrieval_negative_count,
            "meta_episodes": int(config["runtime"].get("meta_episodes", 0)),
            "inner_steps": int(config["runtime"].get("inner_steps", 0)),
            "inner_learning_rate": float(config["runtime"].get("inner_learning_rate", 0.0)),
            "meta_learning_rate": float(config["runtime"].get("meta_learning_rate", 0.0)),
            "multitask_steps": int(config["runtime"].get("multitask_steps", config["runtime"].get("meta_episodes", 0))),
            "multitask_learning_rate": float(
                config["runtime"].get("multitask_learning_rate", config["runtime"].get("meta_learning_rate", 0.0))
            ),
            "query_candidate_pool_policy": (
                "exclude_support_for_query_eval" if query_objective == "continuation_retrieval" else "label_prototype"
            ),
            "support_candidate_pool_policy": (
                "support_only_for_inner_loop" if query_objective == "continuation_retrieval" else "label_prototype"
            ),
            "episodes_completed": 0,
            "steps_completed": 0,
            "examples_seen": 0,
            "trainable_module": trainable_module,
            "stage_b_trainable_target": stage_b_trainable_target,
            "source_eval_label_space": source_eval_label_space,
            "source_eval_candidate_pool": source_eval_candidate_pool,
            "source_eval_query_loss": source_zero_shot_query_loss,
            "source_eval_query_accuracy": source_zero_shot_query_accuracy,
            "source_eval_task_score": source_eval_task_score,
            "source_eval_metric_name": source_eval_metric_name,
            "source_eval_task_proxy_score": source_eval_task_proxy_score,
            "source_eval_task_proxy_name": source_eval_task_proxy_name,
            "source_eval_task_margin": source_eval_task_margin,
            "queries_meta_init": str(queries_path.resolve()),
            "writer_checkpoint": str((output_dir / "writer.ckpt").resolve()),
            "source_domains": manifest["source_domains"],
            "dataset_sha256": manifest["dataset_sha256"],
            **profiler.finalize(),
        }
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "stage_b_events.json", {"events": []})
        return metrics

    episodes = 2 if dry_run else int(config["runtime"]["meta_episodes"])
    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="train")
    events = []

    if query_learning_mode == "meta_trained":
        sampler = EpisodeSampler(
            grouped_examples,
            source_domains=manifest["source_domains"],
            support_size=int(manifest["support_size"]),
            query_size=int(manifest["query_size"]),
            seed=seed,
            sampling_policy=str(manifest["sampling_policy"]),
        )
        inner_lr = float(config["runtime"]["inner_learning_rate"])
        meta_lr = float(config["runtime"]["meta_learning_rate"])
        inner_steps = int(config["runtime"]["inner_steps"])

        for episode_index in range(episodes):
            profiler.add_example()
            episode = sampler.sample_episode()
            fast_runtime = copy.deepcopy(runtime)
            fast_trainable_parameters, _ = _stage_b_trainable_parameters(
                fast_runtime,
                stage_b_trainable_target,
            )
            inner_optimizer = torch.optim.SGD(fast_trainable_parameters, lr=inner_lr)
            domain_examples = list(grouped_examples[episode.domain])
            query_candidate_pool = _exclude_support_from_query_pool(
                domain_examples,
                episode.support_examples,
            )
            support_candidate_pool = list(episode.support_examples)
            if query_objective == "label_prototype":
                candidate_states, candidate_labels = _build_label_prototypes(runtime, domain_examples)
                zero_shot_query_loss = _mean_classification_loss(
                    runtime,
                    episode.query_examples,
                    candidate_states=candidate_states,
                    candidate_labels=candidate_labels,
                )
                zero_shot_query_accuracy = _compute_accuracy(
                    runtime,
                    episode.query_examples,
                    candidate_states=candidate_states,
                    candidate_labels=candidate_labels,
                )
            else:
                zero_shot_query_loss, zero_shot_query_accuracy = _mean_continuation_retrieval_metrics(
                    runtime,
                    episode.query_examples,
                    candidate_pool_resolver=lambda _: query_candidate_pool,
                    negative_count=retrieval_negative_count,
                )
            for _ in range(inner_steps):
                if query_objective == "label_prototype":
                    support_loss = torch.stack(
                        [
                            _classification_loss(
                                fast_runtime,
                                example,
                                candidate_states=candidate_states,
                                candidate_labels=candidate_labels,
                            )
                            for example in episode.support_examples
                        ]
                    ).mean()
                else:
                    support_loss = torch.stack(
                        [
                            _continuation_retrieval_loss(
                                fast_runtime,
                                example,
                                candidate_pool=support_candidate_pool,
                                negative_count=retrieval_negative_count,
                            )[0]
                            for example in episode.support_examples
                        ]
                    ).mean()
                inner_optimizer.zero_grad()
                support_loss.backward()
                inner_optimizer.step()
            if query_objective == "label_prototype":
                adapted_query_loss = _mean_classification_loss(
                    fast_runtime,
                    episode.query_examples,
                    candidate_states=candidate_states,
                    candidate_labels=candidate_labels,
                )
                adapted_query_accuracy = _compute_accuracy(
                    fast_runtime,
                    episode.query_examples,
                    candidate_states=candidate_states,
                    candidate_labels=candidate_labels,
                )
            else:
                adapted_query_loss, adapted_query_accuracy = _mean_continuation_retrieval_metrics(
                    fast_runtime,
                    episode.query_examples,
                    candidate_pool_resolver=lambda _: query_candidate_pool,
                    negative_count=retrieval_negative_count,
                )

            with torch.no_grad():
                runtime.reader.queries.add_(meta_lr * (fast_runtime.reader.queries - runtime.reader.queries))
                for parameter, fast_parameter in zip(runtime.fuser.parameters(), fast_runtime.fuser.parameters()):
                    parameter.add_(meta_lr * (fast_parameter - parameter))

            for example in episode.support_examples + episode.query_examples:
                profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
                profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
            events.append(
                {
                    "episode": episode_index,
                    "domain": episode.domain,
                    "query_objective": query_objective,
                    "query_candidate_pool_size": len(query_candidate_pool),
                    "support_candidate_pool_size": len(support_candidate_pool),
                    "zero_shot_query_loss": zero_shot_query_loss,
                    "zero_shot_query_accuracy": zero_shot_query_accuracy,
                    "adapted_query_loss": adapted_query_loss,
                    "adapted_query_accuracy": adapted_query_accuracy,
                    "adaptation_gain": zero_shot_query_loss - adapted_query_loss,
                }
            )
    else:
        source_eval_label_space = "global_multitask"
        source_eval_candidate_pool = "global_source"
        if query_objective == "label_prototype":
            global_candidate_states, global_candidate_labels = _build_label_prototypes(runtime, source_examples)
        multitask_steps = 2 if dry_run else int(config["runtime"].get("multitask_steps", episodes))
        multitask_learning_rate = float(
            config["runtime"].get("multitask_learning_rate", config["runtime"]["meta_learning_rate"])
        )
        optimizer = torch.optim.SGD(trainable_parameters, lr=multitask_learning_rate)

        for step in range(multitask_steps):
            profiler.add_example()
            example = source_examples[step % len(source_examples)]
            optimizer.zero_grad()
            if query_objective == "label_prototype":
                loss = _classification_loss(
                    runtime,
                    example,
                    candidate_states=global_candidate_states,
                    candidate_labels=global_candidate_labels,
                )
            else:
                loss, _ = _continuation_retrieval_loss(
                    runtime,
                    example,
                    candidate_pool=source_examples,
                    negative_count=retrieval_negative_count,
                )
            loss.backward()
            optimizer.step()
            profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
            profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
            events.append(
                {
                    "step": step,
                    "domain": example["domain"],
                    "query_objective": query_objective,
                    "query_loss": float(loss.item()),
                }
            )

    queries_path = _save_stage_b_state(
        output_dir=output_dir,
        runtime=runtime,
        seed=seed,
        source_domains=manifest["source_domains"],
        query_learning_mode=query_learning_mode,
        query_objective=query_objective,
        stage_b_trainable_target=stage_b_trainable_target,
        writer_checkpoint=output_dir / "writer.ckpt",
    )
    if query_objective == "label_prototype":
        source_zero_shot_query_loss = _mean_classification_loss_by_domain(
            runtime,
            source_examples,
            candidate_bank=domain_candidate_bank,
        )
        source_zero_shot_query_accuracy = _compute_accuracy_by_domain(
            runtime,
            source_examples,
            candidate_bank=domain_candidate_bank,
        )
        source_eval_task_score = source_zero_shot_query_accuracy
        source_eval_metric_name = "accuracy"
        source_eval_task_proxy_score = source_eval_task_score
        source_eval_task_proxy_name = source_eval_metric_name
        source_eval_task_margin = 0.0
    else:
        source_zero_shot_query_loss, source_zero_shot_query_accuracy = _mean_continuation_retrieval_metrics(
            runtime,
            source_examples,
            candidate_pool_resolver=_source_pool_resolver,
            negative_count=retrieval_negative_count,
        )
        source_eval_metrics = _mean_task_score(runtime, source_examples)
        source_eval_task_score = float(source_eval_metrics["task_score"])
        source_eval_metric_name = str(source_eval_metrics["task_metric_name"])
        source_eval_task_proxy_score = float(source_eval_metrics["task_proxy_score"])
        source_eval_task_proxy_name = str(source_eval_metrics["task_proxy_name"])
        source_eval_task_margin = float(source_eval_metrics["task_margin"])
    metrics = {
        "mode": "train",
        "training_stage": "stage_b",
        "query_learning_mode": query_learning_mode,
        "query_objective": query_objective,
        "retrieval_negative_count": retrieval_negative_count,
        "meta_episodes": int(config["runtime"].get("meta_episodes", 0)),
        "inner_steps": int(config["runtime"].get("inner_steps", 0)),
        "inner_learning_rate": float(config["runtime"].get("inner_learning_rate", 0.0)),
        "meta_learning_rate": float(config["runtime"].get("meta_learning_rate", 0.0)),
        "multitask_steps": int(config["runtime"].get("multitask_steps", config["runtime"].get("meta_episodes", 0))),
        "multitask_learning_rate": float(
            config["runtime"].get("multitask_learning_rate", config["runtime"].get("meta_learning_rate", 0.0))
        ),
        "query_candidate_pool_policy": (
            "exclude_support_for_query_eval" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "support_candidate_pool_policy": (
            "support_only_for_inner_loop" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "episodes_completed": episodes if query_learning_mode == "meta_trained" else 0,
        "steps_completed": len(events) if query_learning_mode == "non_meta_multitask" else 0,
        "examples_seen": len(events),
        "trainable_module": trainable_module,
        "stage_b_trainable_target": stage_b_trainable_target,
        "source_eval_label_space": source_eval_label_space,
        "source_eval_candidate_pool": source_eval_candidate_pool,
        "source_eval_query_loss": source_zero_shot_query_loss,
        "source_eval_query_accuracy": source_zero_shot_query_accuracy,
        "source_eval_task_score": source_eval_task_score,
        "source_eval_metric_name": source_eval_metric_name,
        "source_eval_task_proxy_score": source_eval_task_proxy_score,
        "source_eval_task_proxy_name": source_eval_task_proxy_name,
        "source_eval_task_margin": source_eval_task_margin,
        "queries_meta_init": str(queries_path.resolve()),
        "writer_checkpoint": str((output_dir / "writer.ckpt").resolve()),
        "source_domains": manifest["source_domains"],
        "dataset_sha256": manifest["dataset_sha256"],
        **profiler.finalize(),
    }
    if query_learning_mode == "meta_trained":
        metrics["mean_zero_shot_query_loss"] = sum(item["zero_shot_query_loss"] for item in events) / len(events)
        metrics["mean_adapted_query_loss"] = sum(item["adapted_query_loss"] for item in events) / len(events)
        metrics["mean_adaptation_gain"] = sum(item["adaptation_gain"] for item in events) / len(events)
    else:
        metrics["mean_query_loss"] = sum(item["query_loss"] for item in events) / len(events)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "stage_b_events.json", {"events": events})
    return metrics


def run_stage_c(
    *,
    config: dict,
    seed: int,
    output_dir: Path,
    resume: str | None,
    dry_run: bool,
) -> dict[str, object]:
    grouped_examples, manifest = _build_meta_context(config)
    write_json(output_dir / "meta_data_manifest.json", manifest)

    writer_path = _resolve_artifact_path(resume, "writer.ckpt")
    queries_path = _resolve_artifact_path(resume, "queries_meta_init.pt")
    adaptation_target = _resolve_stage_c_adaptation_target(config)
    expected_query_learning_mode = _resolve_expected_stage_c_query_learning_mode(config)
    query_objective = _resolve_query_objective(config)
    retrieval_negative_count = _resolve_retrieval_negative_count(config)
    target_eval_repeats = _resolve_target_eval_repeats(config)
    target_episode_repeats = _resolve_target_episode_repeats(config)
    target_episode_policy = _resolve_target_episode_policy(config)
    target_support_weighting = _resolve_target_support_weighting(config)
    target_split_policy = _resolve_target_split_policy(config)
    target_support_bank_size_spec = _resolve_target_support_bank_size_spec(config)
    target_support_negative_pool = _resolve_target_support_negative_pool(config)
    target_support_negative_sampler = _resolve_target_support_negative_sampler(config)
    runtime = MemoryRuntime(config=config, seed=seed)
    runtime.writer.load_from(writer_path)
    state = torch.load(queries_path, map_location="cpu")
    query_learning_mode = str(state.get("query_learning_mode", "meta_trained"))
    resumed_query_objective = str(state.get("query_objective", query_objective))
    if expected_query_learning_mode is not None and query_learning_mode != expected_query_learning_mode:
        raise ValueError(
            f"Stage C expected query_learning_mode={expected_query_learning_mode}, "
            f"but resume artifact provides {query_learning_mode}."
        )
    if resumed_query_objective != query_objective:
        raise ValueError(
            f"Stage C expected query_objective={query_objective}, "
            f"but resume artifact provides {resumed_query_objective}."
        )
    runtime.reader.load_state_dict(state["reader_state"])
    if "fuser_state" in state:
        runtime.fuser.load_state_dict(state["fuser_state"])
    adaptable_parameters, trainable_module = _configure_stage_c_trainables(runtime, adaptation_target)
    trainable_parameter_count = _count_unique_parameters(adaptable_parameters)

    domain_examples = list(grouped_examples[manifest["target_domain"]])
    candidate_states: torch.Tensor | None = None
    candidate_labels: list[str] | None = None
    if query_objective == "label_prototype":
        candidate_states, candidate_labels = _build_label_prototypes(runtime, domain_examples)
    shots_list = [int(shot) for shot in config["runtime"]["adapt_shots"]]
    max_steps = 1 if dry_run else int(config["runtime"]["adapt_steps"])
    adapt_lr = float(config["runtime"]["adapt_learning_rate"])
    adaptation_effective_threshold = float(config["runtime"].get("adaptation_effective_threshold", 1e-7))
    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="train")
    curve_rows: list[dict[str, object]] = []
    best_runtime = copy.deepcopy(runtime)
    best_adapt_row: dict[str, object] | None = None
    support_updates = 0
    support_examples_touched = 0
    query_examples_touched = 0
    support_grad_norms: list[float] = []
    support_update_max_abs: list[float] = []
    support_update_l2: list[float] = []
    latest_support_grad_norm = 0.0
    latest_support_update_max_abs = 0.0
    latest_support_update_l2 = 0.0
    checkpoint_target_episode_policy = (
        "shared_aggregate" if target_episode_policy == "aggregate_support" else "first_repeat"
    )

    for shot in shots_list:
        episode_states = _build_stage_c_episode_states(
            runtime=runtime,
            grouped_examples=grouped_examples,
            manifest=manifest,
            domain_examples=domain_examples,
            shot=shot,
            seed=seed,
            max_shot=max(shots_list),
            target_episode_repeats=target_episode_repeats,
            target_eval_repeats=target_eval_repeats,
            target_split_policy=target_split_policy,
            target_support_bank_size_spec=target_support_bank_size_spec,
            retrieval_negative_count=retrieval_negative_count,
            target_support_negative_pool=target_support_negative_pool,
            target_support_negative_sampler=target_support_negative_sampler,
        )
        if target_episode_policy == "aggregate_support":
            shared_runtime = copy.deepcopy(runtime)
            for episode_state in episode_states:
                episode_state["runtime"] = shared_runtime
        else:
            for episode_state in episode_states:
                episode_state["runtime"] = copy.deepcopy(runtime)
        max_recorded_steps = 0 if shot == 0 else max_steps
        for step in range(max_recorded_steps + 1):
            profiler.add_example()
            eval_objective_losses: list[float] = []
            eval_objective_accuracies: list[float] = []
            eval_task_scores: list[float] = []
            eval_task_metric_names: list[str] = []
            eval_task_proxy_scores: list[float] = []
            eval_task_proxy_names: list[str] = []
            eval_task_margins: list[float] = []
            episode_proxy_scores: list[float] = []
            evaluated_query_examples = 0
            for episode_state in episode_states:
                current_runtime = episode_state["runtime"]
                eval_query_sets = episode_state["eval_query_sets"]
                query_candidate_pool = episode_state["query_candidate_pool"]
                episode_task_proxy_scores: list[float] = []
                for query_examples in eval_query_sets:
                    for example in query_examples:
                        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
                        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
                        query_examples_touched += 1
                        evaluated_query_examples += 1
                    if query_objective == "label_prototype":
                        objective_loss = _mean_classification_loss(
                            current_runtime,
                            query_examples,
                            candidate_states=candidate_states,
                            candidate_labels=candidate_labels,
                        )
                        objective_accuracy = _compute_accuracy(
                            current_runtime,
                            query_examples,
                            candidate_states=candidate_states,
                            candidate_labels=candidate_labels,
                        )
                        task_score = objective_accuracy
                        task_metric_name = "accuracy"
                        task_proxy_score = task_score
                        task_proxy_name = task_metric_name
                        task_margin = 0.0
                    else:
                        objective_loss, objective_accuracy = _mean_continuation_retrieval_metrics(
                            current_runtime,
                            query_examples,
                            candidate_pool_resolver=lambda _: query_candidate_pool,
                            negative_count=retrieval_negative_count,
                        )
                        task_metrics = _mean_task_score(current_runtime, query_examples)
                        task_score = float(task_metrics["task_score"])
                        task_metric_name = str(task_metrics["task_metric_name"])
                        task_proxy_score = float(task_metrics["task_proxy_score"])
                        task_proxy_name = str(task_metrics["task_proxy_name"])
                        task_margin = float(task_metrics["task_margin"])
                    eval_objective_losses.append(objective_loss)
                    eval_objective_accuracies.append(objective_accuracy)
                    eval_task_scores.append(task_score)
                    eval_task_metric_names.append(task_metric_name)
                    eval_task_proxy_scores.append(task_proxy_score)
                    eval_task_proxy_names.append(task_proxy_name)
                    eval_task_margins.append(task_margin)
                    episode_task_proxy_scores.append(task_proxy_score)
                episode_proxy_scores.append(
                    sum(episode_task_proxy_scores) / len(episode_task_proxy_scores)
                    if episode_task_proxy_scores
                    else 0.0
                )
            objective_loss = sum(eval_objective_losses) / len(eval_objective_losses)
            objective_accuracy = sum(eval_objective_accuracies) / len(eval_objective_accuracies)
            task_score = sum(eval_task_scores) / len(eval_task_scores)
            task_metric_name = (
                sorted(set(eval_task_metric_names))[0]
                if len(set(eval_task_metric_names)) == 1
                else "mean_score"
            )
            task_proxy_score = sum(eval_task_proxy_scores) / len(eval_task_proxy_scores)
            task_proxy_name = (
                sorted(set(eval_task_proxy_names))[0]
                if len(set(eval_task_proxy_names)) == 1
                else "mean_proxy_score"
            )
            task_margin = sum(eval_task_margins) / len(eval_task_margins)
            curve_rows.append(
                {
                    "query_learning_mode": query_learning_mode,
                    "query_objective": query_objective,
                    "adaptation_target": adaptation_target,
                    "trainable_module": trainable_module,
                    "trainable_parameter_count": trainable_parameter_count,
                    "shot": shot,
                    "step": step,
                    "target_eval_repeats": target_eval_repeats,
                    "target_episode_repeats": target_episode_repeats,
                    "target_episode_policy": target_episode_policy,
                    "target_support_weighting": target_support_weighting,
                    "target_split_policy": target_split_policy,
                    "target_support_bank_size": (
                        sum(int(state["target_support_bank_size"]) for state in episode_states) / len(episode_states)
                    ),
                    "target_support_negative_pool": target_support_negative_pool,
                    "target_support_negative_sampler": target_support_negative_sampler,
                    "evaluated_target_episodes": len(episode_states),
                    "evaluated_query_examples": evaluated_query_examples,
                    "query_candidate_pool_size": (
                        sum(len(state["query_candidate_pool"]) for state in episode_states) / len(episode_states)
                    ),
                    "support_candidate_pool_size": (
                        sum(len(state["support_candidate_pool"]) for state in episode_states) / len(episode_states)
                    ),
                    "objective_loss": objective_loss,
                    "objective_accuracy": objective_accuracy,
                    "task_score": task_score,
                    "task_metric_name": task_metric_name,
                    "task_proxy_score": task_proxy_score,
                    "task_proxy_name": task_proxy_name,
                    "task_margin": task_margin,
                    "preceding_support_grad_norm": latest_support_grad_norm,
                    "preceding_support_update_max_abs": latest_support_update_max_abs,
                    "preceding_support_update_l2": latest_support_update_l2,
                    "query_loss": objective_loss,
                    "query_accuracy": objective_accuracy,
                }
            )
            if shot > 0:
                current_row = curve_rows[-1]
                if best_adapt_row is None or _stage_c_row_key(current_row) > _stage_c_row_key(best_adapt_row):
                    best_adapt_row = dict(current_row)
                    best_runtime = copy.deepcopy(episode_states[0]["runtime"])
            if step == max_recorded_steps or shot == 0:
                continue
            per_episode_grad_norms: list[float] = []
            per_episode_update_max_abs: list[float] = []
            per_episode_update_l2: list[float] = []
            if target_episode_policy == "aggregate_support":
                shared_runtime = episode_states[0]["runtime"]
                current_adapt_parameters, _ = _configure_stage_c_trainables(shared_runtime, adaptation_target)
                optimizer = torch.optim.SGD(current_adapt_parameters, lr=adapt_lr)
                episode_support_losses: list[torch.Tensor] = []
                episode_support_weights = _resolve_episode_support_weights(
                    episode_proxy_scores,
                    weighting=target_support_weighting,
                )
                for episode_index, episode_state in enumerate(episode_states):
                    support_examples = episode_state["support_examples"]
                    support_candidate_pool = episode_state["support_candidate_pool"]
                    if query_objective == "label_prototype":
                        episode_support_loss = torch.stack(
                            [
                                _classification_loss(
                                    shared_runtime,
                                    example,
                                    candidate_states=candidate_states,
                                    candidate_labels=candidate_labels,
                                )
                                for example in support_examples
                            ]
                        ).mean()
                    else:
                        episode_support_loss = torch.stack(
                            [
                                _continuation_retrieval_loss(
                                    shared_runtime,
                                    example,
                                    candidate_pool=support_candidate_pool,
                                    negative_count=retrieval_negative_count,
                                    negative_sampler=target_support_negative_sampler,
                                )[0]
                                for example in support_examples
                            ]
                        ).mean()
                    episode_support_losses.append(
                        episode_support_loss * float(episode_support_weights[episode_index])
                    )
                    for example in support_examples:
                        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
                        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
                        support_examples_touched += 1
                optimizer.zero_grad()
                before_update = _snapshot_parameter_tensors(current_adapt_parameters)
                support_loss = torch.stack(episode_support_losses).sum()
                support_loss.backward()
                episode_grad_norm = _parameter_grad_norm(current_adapt_parameters)
                optimizer.step()
                episode_update_max_abs, episode_update_l2 = _parameter_update_stats(
                    before_update,
                    current_adapt_parameters,
                )
                support_updates += 1
                per_episode_grad_norms.append(episode_grad_norm)
                per_episode_update_max_abs.append(episode_update_max_abs)
                per_episode_update_l2.append(episode_update_l2)
                support_grad_norms.append(episode_grad_norm)
                support_update_max_abs.append(episode_update_max_abs)
                support_update_l2.append(episode_update_l2)
            else:
                for episode_state in episode_states:
                    current_runtime = episode_state["runtime"]
                    support_examples = episode_state["support_examples"]
                    support_candidate_pool = episode_state["support_candidate_pool"]
                    current_adapt_parameters, _ = _configure_stage_c_trainables(current_runtime, adaptation_target)
                    optimizer = torch.optim.SGD(current_adapt_parameters, lr=adapt_lr)
                    if query_objective == "label_prototype":
                        support_loss = torch.stack(
                            [
                                _classification_loss(
                                    current_runtime,
                                    example,
                                    candidate_states=candidate_states,
                                    candidate_labels=candidate_labels,
                                )
                                for example in support_examples
                            ]
                        ).mean()
                    else:
                        support_loss = torch.stack(
                            [
                                _continuation_retrieval_loss(
                                    current_runtime,
                                    example,
                                    candidate_pool=support_candidate_pool,
                                    negative_count=retrieval_negative_count,
                                    negative_sampler=target_support_negative_sampler,
                                )[0]
                                for example in support_examples
                            ]
                        ).mean()
                    optimizer.zero_grad()
                    before_update = _snapshot_parameter_tensors(current_adapt_parameters)
                    support_loss.backward()
                    episode_grad_norm = _parameter_grad_norm(current_adapt_parameters)
                    optimizer.step()
                    episode_update_max_abs, episode_update_l2 = _parameter_update_stats(
                        before_update,
                        current_adapt_parameters,
                    )
                    support_updates += 1
                    per_episode_grad_norms.append(episode_grad_norm)
                    per_episode_update_max_abs.append(episode_update_max_abs)
                    per_episode_update_l2.append(episode_update_l2)
                    support_grad_norms.append(episode_grad_norm)
                    support_update_max_abs.append(episode_update_max_abs)
                    support_update_l2.append(episode_update_l2)
                    for example in support_examples:
                        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
                        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
                        support_examples_touched += 1
            latest_support_grad_norm = sum(per_episode_grad_norms) / len(per_episode_grad_norms)
            latest_support_update_max_abs = (
                sum(per_episode_update_max_abs) / len(per_episode_update_max_abs)
            )
            latest_support_update_l2 = sum(per_episode_update_l2) / len(per_episode_update_l2)

    adapt_curve_path = output_dir / "adapt_curve.csv"
    with adapt_curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_learning_mode",
                "query_objective",
                "adaptation_target",
                "trainable_module",
                "trainable_parameter_count",
                "shot",
                "step",
                "target_eval_repeats",
                "target_episode_repeats",
                "target_episode_policy",
                "target_support_weighting",
                "target_split_policy",
                "target_support_bank_size",
                "target_support_negative_pool",
                "target_support_negative_sampler",
                "evaluated_target_episodes",
                "evaluated_query_examples",
                "query_candidate_pool_size",
                "support_candidate_pool_size",
                "objective_loss",
                "objective_accuracy",
                "task_score",
                "task_metric_name",
                "task_proxy_score",
                "task_proxy_name",
                "task_margin",
                "preceding_support_grad_norm",
                "preceding_support_update_max_abs",
                "preceding_support_update_l2",
                "query_loss",
                "query_accuracy",
            ],
        )
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)
    adapted_writer_path = None
    if adaptation_target in {"w_only", "w_plus_q"}:
        adapted_writer_path = best_runtime.writer.save_to(output_dir / "writer_adapted.ckpt")
    torch.save(
        {
            "reader_state": best_runtime.reader.state_dict(),
            "fuser_state": best_runtime.fuser.state_dict(),
            "seed": seed,
            "target_domain": manifest["target_domain"],
            "query_learning_mode": query_learning_mode,
            "query_objective": query_objective,
            "target_episode_policy": target_episode_policy,
            "target_support_weighting": target_support_weighting,
            "target_split_policy": target_split_policy,
            "target_support_bank_size": target_support_bank_size_spec,
            "target_support_negative_pool": target_support_negative_pool,
            "target_support_negative_sampler": target_support_negative_sampler,
            "adaptation_target": adaptation_target,
            "trainable_module": trainable_module,
            "writer_checkpoint": str((adapted_writer_path or writer_path).resolve()),
        },
        output_dir / "queries_adapted.pt",
    )
    zero_shot_row = next(row for row in curve_rows if row["shot"] == 0)
    best_row = best_adapt_row or zero_shot_row
    profile_metrics = profiler.finalize()
    adapt_cost = {
        "query_learning_mode": query_learning_mode,
        "query_objective": query_objective,
        "query_candidate_pool_policy": (
            "fixed_target_holdout_for_query_eval" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "support_candidate_pool_policy": (
            "support_bank_for_inner_loop" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "adaptation_target": adaptation_target,
        "trainable_module": trainable_module,
        "trainable_parameter_count": trainable_parameter_count,
        "adapt_learning_rate": adapt_lr,
        "adapt_steps": max_steps,
        "adapt_shots": shots_list,
        "target_eval_repeats": target_eval_repeats,
        "target_episode_repeats": target_episode_repeats,
        "target_episode_policy": target_episode_policy,
        "target_support_weighting": target_support_weighting,
        "target_split_policy": target_split_policy,
        "target_support_bank_size": target_support_bank_size_spec,
        "target_support_negative_pool": target_support_negative_pool,
        "target_support_negative_sampler": target_support_negative_sampler,
        "support_updates": support_updates,
        "support_examples_touched": support_examples_touched,
        "query_examples_touched": query_examples_touched,
        "mean_support_grad_norm": (
            sum(support_grad_norms) / len(support_grad_norms) if support_grad_norms else 0.0
        ),
        "max_support_grad_norm": max(support_grad_norms) if support_grad_norms else 0.0,
        "mean_support_update_max_abs": (
            sum(support_update_max_abs) / len(support_update_max_abs) if support_update_max_abs else 0.0
        ),
        "max_support_update_max_abs": max(support_update_max_abs) if support_update_max_abs else 0.0,
        "mean_support_update_l2": (
            sum(support_update_l2) / len(support_update_l2) if support_update_l2 else 0.0
        ),
        "max_support_update_l2": max(support_update_l2) if support_update_l2 else 0.0,
        "adaptation_effective_threshold": adaptation_effective_threshold,
        "adaptation_effective": (
            max(support_update_max_abs) >= adaptation_effective_threshold if support_update_max_abs else False
        ),
        "target_domain": manifest["target_domain"],
        "checkpoint_target_episode_policy": checkpoint_target_episode_policy,
        **profile_metrics,
    }
    adapt_cost_path = output_dir / "adapt_cost.json"
    write_json(adapt_cost_path, adapt_cost)
    metrics = {
        "mode": "train",
        "training_stage": "stage_c",
        "target_domain": manifest["target_domain"],
        "query_learning_mode": query_learning_mode,
        "query_objective": query_objective,
        "retrieval_negative_count": retrieval_negative_count,
        "query_candidate_pool_policy": (
            "fixed_target_holdout_for_query_eval" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "support_candidate_pool_policy": (
            "support_bank_for_inner_loop" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "adaptation_target": adaptation_target,
        "trainable_module": trainable_module,
        "trainable_parameter_count": trainable_parameter_count,
        "adapt_learning_rate": adapt_lr,
        "adapt_steps": max_steps,
        "adapt_shots": shots_list,
        "target_eval_repeats": target_eval_repeats,
        "target_episode_repeats": target_episode_repeats,
        "target_episode_policy": target_episode_policy,
        "target_support_weighting": target_support_weighting,
        "target_split_policy": target_split_policy,
        "target_support_bank_size": target_support_bank_size_spec,
        "target_support_negative_pool": target_support_negative_pool,
        "target_support_negative_sampler": target_support_negative_sampler,
        "mean_support_grad_norm": adapt_cost["mean_support_grad_norm"],
        "max_support_grad_norm": adapt_cost["max_support_grad_norm"],
        "mean_support_update_max_abs": adapt_cost["mean_support_update_max_abs"],
        "max_support_update_max_abs": adapt_cost["max_support_update_max_abs"],
        "mean_support_update_l2": adapt_cost["mean_support_update_l2"],
        "max_support_update_l2": adapt_cost["max_support_update_l2"],
        "adaptation_effective_threshold": adaptation_effective_threshold,
        "adaptation_effective": adapt_cost["adaptation_effective"],
        "zero_shot_query_loss": zero_shot_row["query_loss"],
        "zero_shot_query_accuracy": zero_shot_row["query_accuracy"],
        "zero_shot_objective_loss": zero_shot_row["objective_loss"],
        "zero_shot_task_score": zero_shot_row["task_score"],
        "zero_shot_task_proxy_score": zero_shot_row["task_proxy_score"],
        "zero_shot_task_margin": zero_shot_row["task_margin"],
        "task_metric_name": best_row["task_metric_name"],
        "task_proxy_name": best_row["task_proxy_name"],
        "best_adapt_query_accuracy": best_row["query_accuracy"],
        "best_adapt_query_loss": best_row["query_loss"],
        "best_adapt_objective_loss": best_row["objective_loss"],
        "best_adapt_task_score": best_row["task_score"],
        "best_adapt_task_proxy_score": best_row["task_proxy_score"],
        "best_adapt_task_margin": best_row["task_margin"],
        "best_adapt_shot": best_row["shot"],
        "best_adapt_step": best_row["step"],
        "adapt_curve_path": str(adapt_curve_path.resolve()),
        "adapt_cost_path": str(adapt_cost_path.resolve()),
        "queries_meta_init": str(queries_path.resolve()),
        "writer_checkpoint": str(writer_path.resolve()),
        "adapted_queries_checkpoint": str((output_dir / "queries_adapted.pt").resolve()),
        "adapted_writer_checkpoint": (
            str(adapted_writer_path.resolve()) if adapted_writer_path is not None else None
        ),
        "checkpoint_target_episode_policy": checkpoint_target_episode_policy,
        "dataset_sha256": manifest["dataset_sha256"],
        **profile_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "adapt_curve.json", {"rows": curve_rows})
    return metrics
