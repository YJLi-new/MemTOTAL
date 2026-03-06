from __future__ import annotations

import copy
import csv
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from memtotal.data import (
    EpisodeSampler,
    build_meta_manifest,
    load_domain_dataset,
    split_target_domain_examples,
    validate_meta_split,
)
from memtotal.pipeline import MemoryRuntime
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


def _stage_b_trainable_parameters(runtime: MemoryRuntime) -> tuple[list[torch.nn.Parameter], str]:
    runtime.writer.freeze()
    runtime.fuser.unfreeze()
    runtime.reader.unfreeze()
    return [runtime.reader.queries, *runtime.fuser.parameters()], "reader.queries+fuser"


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
    memory_summary = forward.memory_short.mean(dim=1)
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
        memory_summary = forward.memory_short.mean(dim=1)
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
        memory_summary = forward.memory_short.mean(dim=1)
        scores = runtime.score_candidates(memory_summary, candidate_states)
        predicted_label = candidate_labels[int(torch.argmax(scores).item())]
        correct += int(predicted_label == example["label"])
    return correct / len(examples)


def _stage_c_row_key(row: dict[str, object]) -> tuple[float, float, int, int]:
    return (
        float(row["query_accuracy"]),
        -float(row["query_loss"]),
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
        },
        queries_path,
    )
    return queries_path


def _build_meta_context(config: dict) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    meta_cfg = _load_meta_config(config)
    grouped_examples = load_domain_dataset(config["task"]["dataset_path"])
    validate_meta_split(grouped_examples, **meta_cfg)
    manifest = build_meta_manifest(
        dataset_path=config["task"]["dataset_path"],
        grouped_examples=grouped_examples,
        **meta_cfg,
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
    trainable_parameters, trainable_module = _stage_b_trainable_parameters(runtime)
    shutil.copy2(writer_path, output_dir / "writer.ckpt")
    source_examples = _flatten_domains(grouped_examples, manifest["source_domains"])
    domain_candidate_bank = {
        domain: _build_label_prototypes(runtime, grouped_examples[domain])
        for domain in manifest["source_domains"]
    }
    source_eval_label_space = "per_domain"

    if query_learning_mode == "random":
        profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="train")
        for example in source_examples:
            profiler.add_example()
            profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
            profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
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
        queries_path = _save_stage_b_state(
            output_dir=output_dir,
            runtime=runtime,
            seed=seed,
            source_domains=manifest["source_domains"],
            query_learning_mode=query_learning_mode,
            writer_checkpoint=output_dir / "writer.ckpt",
        )
        metrics = {
            "mode": "train",
            "training_stage": "stage_b",
            "query_learning_mode": query_learning_mode,
            "episodes_completed": 0,
            "steps_completed": 0,
            "examples_seen": 0,
            "trainable_module": trainable_module,
            "source_eval_label_space": source_eval_label_space,
            "source_eval_query_loss": source_zero_shot_query_loss,
            "source_eval_query_accuracy": source_zero_shot_query_accuracy,
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
        )
        inner_lr = float(config["runtime"]["inner_learning_rate"])
        meta_lr = float(config["runtime"]["meta_learning_rate"])
        inner_steps = int(config["runtime"]["inner_steps"])

        for episode_index in range(episodes):
            profiler.add_example()
            episode = sampler.sample_episode()
            fast_runtime = copy.deepcopy(runtime)
            _stage_b_trainable_parameters(fast_runtime)
            inner_optimizer = torch.optim.SGD(
                [fast_runtime.reader.queries, *fast_runtime.fuser.parameters()],
                lr=inner_lr,
            )
            domain_examples = list(grouped_examples[episode.domain])
            candidate_states, candidate_labels = _build_label_prototypes(runtime, domain_examples)

            zero_shot_query_loss = _mean_classification_loss(
                runtime,
                episode.query_examples,
                candidate_states=candidate_states,
                candidate_labels=candidate_labels,
            )
            for _ in range(inner_steps):
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
                inner_optimizer.zero_grad()
                support_loss.backward()
                inner_optimizer.step()
            adapted_query_loss = _mean_classification_loss(
                fast_runtime,
                episode.query_examples,
                candidate_states=candidate_states,
                candidate_labels=candidate_labels,
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
                    "zero_shot_query_loss": zero_shot_query_loss,
                    "adapted_query_loss": adapted_query_loss,
                    "adaptation_gain": zero_shot_query_loss - adapted_query_loss,
                }
            )
    else:
        source_eval_label_space = "global_multitask"
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
            loss = _classification_loss(
                runtime,
                example,
                candidate_states=global_candidate_states,
                candidate_labels=global_candidate_labels,
            )
            loss.backward()
            optimizer.step()
            profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
            profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
            events.append(
                {
                    "step": step,
                    "domain": example["domain"],
                    "query_loss": float(loss.item()),
                }
            )

    queries_path = _save_stage_b_state(
        output_dir=output_dir,
        runtime=runtime,
        seed=seed,
        source_domains=manifest["source_domains"],
        query_learning_mode=query_learning_mode,
        writer_checkpoint=output_dir / "writer.ckpt",
    )
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
    metrics = {
        "mode": "train",
        "training_stage": "stage_b",
        "query_learning_mode": query_learning_mode,
        "episodes_completed": episodes if query_learning_mode == "meta_trained" else 0,
        "steps_completed": len(events) if query_learning_mode == "non_meta_multitask" else 0,
        "examples_seen": len(events),
        "trainable_module": trainable_module,
        "source_eval_label_space": source_eval_label_space,
        "source_eval_query_loss": source_zero_shot_query_loss,
        "source_eval_query_accuracy": source_zero_shot_query_accuracy,
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
    runtime = MemoryRuntime(config=config, seed=seed)
    runtime.writer.load_from(writer_path)
    state = torch.load(queries_path, map_location="cpu")
    query_learning_mode = str(state.get("query_learning_mode", "meta_trained"))
    if expected_query_learning_mode is not None and query_learning_mode != expected_query_learning_mode:
        raise ValueError(
            f"Stage C expected query_learning_mode={expected_query_learning_mode}, "
            f"but resume artifact provides {query_learning_mode}."
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
    )
    domain_examples = list(grouped_examples[manifest["target_domain"]])
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
        max_recorded_steps = 0 if shot == 0 else max_steps
        for step in range(max_recorded_steps + 1):
            profiler.add_example()
            for example in target_episode.query_examples:
                profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
                profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
                query_examples_touched += 1
            query_loss = _mean_classification_loss(
                current_runtime,
                target_episode.query_examples,
                candidate_states=candidate_states,
                candidate_labels=candidate_labels,
            )
            query_accuracy = _compute_accuracy(
                current_runtime,
                target_episode.query_examples,
                candidate_states=candidate_states,
                candidate_labels=candidate_labels,
            )
            curve_rows.append(
                {
                    "query_learning_mode": query_learning_mode,
                    "adaptation_target": adaptation_target,
                    "trainable_module": trainable_module,
                    "trainable_parameter_count": trainable_parameter_count,
                    "shot": shot,
                    "step": step,
                    "query_loss": query_loss,
                    "query_accuracy": query_accuracy,
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
                "adaptation_target",
                "trainable_module",
                "trainable_parameter_count",
                "shot",
                "step",
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
        "adaptation_target": adaptation_target,
        "trainable_module": trainable_module,
        "trainable_parameter_count": trainable_parameter_count,
        "zero_shot_query_loss": zero_shot_row["query_loss"],
        "zero_shot_query_accuracy": zero_shot_row["query_accuracy"],
        "best_adapt_query_accuracy": best_row["query_accuracy"],
        "best_adapt_query_loss": best_row["query_loss"],
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
