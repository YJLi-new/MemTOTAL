from __future__ import annotations

import copy
import csv
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
) -> tuple[float, str]:
    evaluator = _build_task_evaluator_for_example(example)
    if evaluator.evaluator_type == "multiple_choice":
        choices = example.get("choices", [])
        if not choices:
            raise ValueError("multiple_choice task examples require per-example choices.")
        candidate_labels = [str(choice["label"]) for choice in choices]
        candidate_texts = [str(choice["text"]) for choice in choices]
        candidate_states = runtime.backbone.summarize_texts(candidate_texts)
        predicted_label, _, _ = runtime.predict_label(
            example,
            candidate_states=candidate_states,
            candidate_labels=candidate_labels,
        )
        predicted_text = next(
            str(choice["text"]) for choice in choices if str(choice["label"]) == predicted_label
        )
        score_payload = evaluator.evaluate_prediction(
            {"label": predicted_label, "text": predicted_text},
            example,
        )
        return float(score_payload["score"]), evaluator.metric_name

    forward = runtime.forward_example(example)
    generated_text = runtime.backbone.generate(
        [forward.next_prompt],
        memory_tokens=forward.generation_memory,
    )[0]
    score_payload = evaluator.evaluate_prediction({"text": generated_text}, example)
    return float(score_payload["score"]), evaluator.metric_name


def _mean_task_score(
    runtime: MemoryRuntime,
    examples: list[dict[str, Any]],
) -> tuple[float, str]:
    scores: list[float] = []
    metric_names: list[str] = []
    for example in examples:
        score, metric_name = _evaluate_task_example(runtime, example)
        scores.append(score)
        metric_names.append(metric_name)
    if not scores:
        return 0.0, "none"
    unique_metric_names = sorted(set(metric_names))
    resolved_metric_name = unique_metric_names[0] if len(unique_metric_names) == 1 else "mean_score"
    return sum(scores) / len(scores), resolved_metric_name


def _resolve_retrieval_candidates(
    example: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    *,
    negative_count: int,
) -> list[dict[str, Any]]:
    negatives = [
        row
        for row in sorted(candidate_pool, key=lambda item: str(item["id"]))
        if str(row["id"]) != str(example["id"])
    ]
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


def _continuation_retrieval_loss(
    runtime: MemoryRuntime,
    example: dict[str, Any],
    *,
    candidate_pool: list[dict[str, Any]],
    negative_count: int,
) -> tuple[torch.Tensor, float]:
    candidate_examples = _resolve_retrieval_candidates(
        example,
        candidate_pool,
        negative_count=negative_count,
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
            source_eval_task_score, source_eval_metric_name = _mean_task_score(runtime, source_examples)
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
    else:
        source_zero_shot_query_loss, source_zero_shot_query_accuracy = _mean_continuation_retrieval_metrics(
            runtime,
            source_examples,
            candidate_pool_resolver=_source_pool_resolver,
            negative_count=retrieval_negative_count,
        )
        source_eval_task_score, source_eval_metric_name = _mean_task_score(runtime, source_examples)
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

    target_episode = split_target_domain_examples(
        grouped_examples,
        target_domain=manifest["target_domain"],
        support_size=int(manifest["support_size"]),
        query_size=int(manifest["query_size"]),
        seed=seed,
        sampling_policy=str(manifest["sampling_policy"]),
    )
    domain_examples = list(grouped_examples[manifest["target_domain"]])
    candidate_states: torch.Tensor | None = None
    candidate_labels: list[str] | None = None
    if query_objective == "label_prototype":
        candidate_states, candidate_labels = _build_label_prototypes(runtime, domain_examples)
    shots_list = [int(shot) for shot in config["runtime"]["adapt_shots"]]
    max_steps = 1 if dry_run else int(config["runtime"]["adapt_steps"])
    adapt_lr = float(config["runtime"]["adapt_learning_rate"])
    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="train")
    curve_rows: list[dict[str, object]] = []
    best_runtime = copy.deepcopy(runtime)
    best_adapt_row: dict[str, object] | None = None
    support_updates = 0
    support_examples_touched = 0
    query_examples_touched = 0

    for shot in shots_list:
        current_runtime = copy.deepcopy(runtime)
        support_examples = target_episode.support_examples[:shot]
        query_candidate_pool = _exclude_support_from_query_pool(
            domain_examples,
            support_examples,
        )
        support_candidate_pool = list(support_examples) or list(domain_examples)
        max_recorded_steps = 0 if shot == 0 else max_steps
        for step in range(max_recorded_steps + 1):
            profiler.add_example()
            for example in target_episode.query_examples:
                profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
                profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
                query_examples_touched += 1
            if query_objective == "label_prototype":
                objective_loss = _mean_classification_loss(
                    current_runtime,
                    target_episode.query_examples,
                    candidate_states=candidate_states,
                    candidate_labels=candidate_labels,
                )
                objective_accuracy = _compute_accuracy(
                    current_runtime,
                    target_episode.query_examples,
                    candidate_states=candidate_states,
                    candidate_labels=candidate_labels,
                )
                task_score = objective_accuracy
                task_metric_name = "accuracy"
            else:
                objective_loss, objective_accuracy = _mean_continuation_retrieval_metrics(
                    current_runtime,
                    target_episode.query_examples,
                    candidate_pool_resolver=lambda _: query_candidate_pool,
                    negative_count=retrieval_negative_count,
                )
                task_score, task_metric_name = _mean_task_score(
                    current_runtime,
                    target_episode.query_examples,
                )
            curve_rows.append(
                {
                    "query_learning_mode": query_learning_mode,
                    "query_objective": query_objective,
                    "adaptation_target": adaptation_target,
                    "trainable_module": trainable_module,
                    "trainable_parameter_count": trainable_parameter_count,
                    "shot": shot,
                    "step": step,
                    "query_candidate_pool_size": len(query_candidate_pool),
                    "support_candidate_pool_size": len(support_candidate_pool),
                    "objective_loss": objective_loss,
                    "objective_accuracy": objective_accuracy,
                    "task_score": task_score,
                    "task_metric_name": task_metric_name,
                    "query_loss": objective_loss,
                    "query_accuracy": objective_accuracy,
                }
            )
            if shot > 0:
                current_row = curve_rows[-1]
                if best_adapt_row is None or _stage_c_row_key(current_row) > _stage_c_row_key(best_adapt_row):
                    best_adapt_row = dict(current_row)
                    best_runtime = copy.deepcopy(current_runtime)
            if step == max_recorded_steps or shot == 0:
                continue
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
                        )[0]
                        for example in support_examples
                    ]
                ).mean()
            optimizer.zero_grad()
            support_loss.backward()
            optimizer.step()
            support_updates += 1
            for example in support_examples:
                profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
                profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
                support_examples_touched += 1

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
                "query_candidate_pool_size",
                "support_candidate_pool_size",
                "objective_loss",
                "objective_accuracy",
                "task_score",
                "task_metric_name",
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
            "exclude_support_for_query_eval" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "support_candidate_pool_policy": (
            "support_only_for_inner_loop" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "adaptation_target": adaptation_target,
        "trainable_module": trainable_module,
        "trainable_parameter_count": trainable_parameter_count,
        "adapt_learning_rate": adapt_lr,
        "adapt_steps": max_steps,
        "adapt_shots": shots_list,
        "support_updates": support_updates,
        "support_examples_touched": support_examples_touched,
        "query_examples_touched": query_examples_touched,
        "target_domain": manifest["target_domain"],
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
            "exclude_support_for_query_eval" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "support_candidate_pool_policy": (
            "support_only_for_inner_loop" if query_objective == "continuation_retrieval" else "label_prototype"
        ),
        "adaptation_target": adaptation_target,
        "trainable_module": trainable_module,
        "trainable_parameter_count": trainable_parameter_count,
        "adapt_learning_rate": adapt_lr,
        "adapt_steps": max_steps,
        "adapt_shots": shots_list,
        "zero_shot_query_loss": zero_shot_row["query_loss"],
        "zero_shot_query_accuracy": zero_shot_row["query_accuracy"],
        "zero_shot_objective_loss": zero_shot_row["objective_loss"],
        "zero_shot_task_score": zero_shot_row["task_score"],
        "task_metric_name": best_row["task_metric_name"],
        "best_adapt_query_accuracy": best_row["query_accuracy"],
        "best_adapt_query_loss": best_row["query_loss"],
        "best_adapt_objective_loss": best_row["objective_loss"],
        "best_adapt_task_score": best_row["task_score"],
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
        "dataset_sha256": manifest["dataset_sha256"],
        **profile_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "adapt_curve.json", {"rows": curve_rows})
    return metrics
