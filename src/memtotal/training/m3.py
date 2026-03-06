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
    runtime.writer.freeze()
    runtime.fuser.unfreeze()
    runtime.reader.unfreeze()
    shutil.copy2(writer_path, output_dir / "writer.ckpt")

    episodes = 2 if dry_run else int(config["runtime"]["meta_episodes"])
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
    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="train")
    events = []

    for episode_index in range(episodes):
        profiler.add_example()
        episode = sampler.sample_episode()
        fast_runtime = copy.deepcopy(runtime)
        fast_runtime.writer.freeze()
        fast_runtime.fuser.unfreeze()
        fast_runtime.reader.unfreeze()
        inner_optimizer = torch.optim.SGD(
            [fast_runtime.reader.queries, *fast_runtime.fuser.parameters()],
            lr=inner_lr,
        )
        domain_examples = list(grouped_examples[episode.domain])
        candidate_labels = [row["label"] for row in domain_examples]
        candidate_states = runtime.backbone.summarize_texts(
            row["continuation"] for row in domain_examples
        )

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

    queries_path = output_dir / "queries_meta_init.pt"
    torch.save(
        {
            "reader_state": runtime.reader.state_dict(),
            "fuser_state": runtime.fuser.state_dict(),
            "seed": seed,
            "writer_checkpoint": str((output_dir / "writer.ckpt").resolve()),
            "source_domains": manifest["source_domains"],
        },
        queries_path,
    )
    metrics = {
        "mode": "train",
        "training_stage": "stage_b",
        "episodes_completed": episodes,
        "trainable_module": "reader.queries+fuser",
        "mean_zero_shot_query_loss": sum(item["zero_shot_query_loss"] for item in events) / len(events),
        "mean_adapted_query_loss": sum(item["adapted_query_loss"] for item in events) / len(events),
        "mean_adaptation_gain": sum(item["adaptation_gain"] for item in events) / len(events),
        "queries_meta_init": str(queries_path.resolve()),
        "writer_checkpoint": str((output_dir / "writer.ckpt").resolve()),
        "source_domains": manifest["source_domains"],
        "dataset_sha256": manifest["dataset_sha256"],
        **profiler.finalize(),
    }
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
    runtime = MemoryRuntime(config=config, seed=seed)
    runtime.writer.load_from(writer_path)
    runtime.writer.freeze()
    state = torch.load(queries_path, map_location="cpu")
    runtime.reader.load_state_dict(state["reader_state"])
    runtime.reader.unfreeze()
    if "fuser_state" in state:
        runtime.fuser.load_state_dict(state["fuser_state"])
    runtime.fuser.unfreeze()

    target_episode = split_target_domain_examples(
        grouped_examples,
        target_domain=manifest["target_domain"],
        support_size=int(manifest["support_size"]),
        query_size=int(manifest["query_size"]),
        seed=seed,
    )
    domain_examples = list(grouped_examples[manifest["target_domain"]])
    candidate_labels = [row["label"] for row in domain_examples]
    candidate_states = runtime.backbone.summarize_texts(row["continuation"] for row in domain_examples)
    shots_list = [int(shot) for shot in config["runtime"]["adapt_shots"]]
    max_steps = 1 if dry_run else int(config["runtime"]["adapt_steps"])
    adapt_lr = float(config["runtime"]["adapt_learning_rate"])
    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="train")
    curve_rows: list[dict[str, object]] = []
    best_runtime = copy.deepcopy(runtime)

    for shot in shots_list:
        current_runtime = copy.deepcopy(runtime)
        support_examples = target_episode.support_examples[:shot]
        max_recorded_steps = 0 if shot == 0 else max_steps
        for step in range(max_recorded_steps + 1):
            profiler.add_example()
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
                    "shot": shot,
                    "step": step,
                    "query_loss": query_loss,
                    "query_accuracy": query_accuracy,
                }
            )
            if shot == max(shots_list) and step == max_recorded_steps:
                best_runtime = copy.deepcopy(current_runtime)
            if step == max_recorded_steps or shot == 0:
                continue
            adapt_parameters = [current_runtime.reader.queries, *current_runtime.fuser.parameters()]
            optimizer = torch.optim.SGD(adapt_parameters, lr=adapt_lr)
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
            for example in support_examples:
                profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
                profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))

    adapt_curve_path = output_dir / "adapt_curve.csv"
    with adapt_curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shot", "step", "query_loss", "query_accuracy"])
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)
    torch.save(
        {
            "reader_state": best_runtime.reader.state_dict(),
            "fuser_state": best_runtime.fuser.state_dict(),
            "seed": seed,
            "target_domain": manifest["target_domain"],
        },
        output_dir / "queries_adapted.pt",
    )
    zero_shot_row = next(row for row in curve_rows if row["shot"] == 0)
    best_row = max(curve_rows, key=lambda row: row["query_accuracy"])
    metrics = {
        "mode": "train",
        "training_stage": "stage_c",
        "target_domain": manifest["target_domain"],
        "zero_shot_query_loss": zero_shot_row["query_loss"],
        "zero_shot_query_accuracy": zero_shot_row["query_accuracy"],
        "best_adapt_query_accuracy": best_row["query_accuracy"],
        "best_adapt_query_loss": min(row["query_loss"] for row in curve_rows),
        "adapt_curve_path": str(adapt_curve_path.resolve()),
        "queries_meta_init": str(queries_path.resolve()),
        "writer_checkpoint": str(writer_path.resolve()),
        "adapted_queries_checkpoint": str((output_dir / "queries_adapted.pt").resolve()),
        "dataset_sha256": manifest["dataset_sha256"],
        **profiler.finalize(),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "adapt_curve.json", {"rows": curve_rows})
    return metrics
