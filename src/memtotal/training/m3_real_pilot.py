from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from memtotal.pipeline import MemoryRuntime
from memtotal.tasks import TaskEvaluator, get_task_spec, load_task_dataset
from memtotal.training.m3 import (
    _configure_stage_c_trainables,
    _pairwise_margin_loss,
    _resolve_artifact_path,
    _resolve_expected_stage_c_query_learning_mode,
)
from memtotal.utils.io import write_json, write_jsonl
from memtotal.utils.profiling import ProfileTracker


@dataclass(frozen=True)
class PilotExampleCache:
    example: dict[str, Any]
    story_state: torch.Tensor
    conditioning_state: torch.Tensor
    memory_long: torch.Tensor
    candidate_states: torch.Tensor
    continuation_state: torch.Tensor
    base_scores: torch.Tensor
    candidate_labels: list[str]
    candidate_texts: list[str]
    choice_scoring_prompt: str


@dataclass(frozen=True)
class RepairAnchor:
    choice_scores: torch.Tensor
    gold_index: int
    predicted_index: int
    competitor_index: int
    margin: float
    predicted_label: str
    competitor_label: str
    repair_weight: float
    repair_active: bool
    repair_bucket: str


def _load_task_dataset_from_path(config: dict[str, Any], dataset_path: str) -> list[dict[str, Any]]:
    config_copy = copy.deepcopy(config)
    config_copy["task"]["dataset_path"] = dataset_path
    return load_task_dataset(config_copy)


def _resolve_decision_mode(config: dict[str, Any]) -> str:
    decision_mode = str(config["runtime"].get("stage_c_decision_mode", "shared_summary_late_fusion"))
    if decision_mode not in {
        "base_only",
        "shared_summary_late_fusion",
        "candidate_conditioned_late_fusion",
        "shared_plus_candidate_delta_late_fusion",
        "shared_plus_candidate_conditioned_late_fusion",
    }:
        raise ValueError(
            f"Unsupported runtime.stage_c_decision_mode={decision_mode}. "
            "Expected one of base_only, shared_summary_late_fusion, "
            "candidate_conditioned_late_fusion, shared_plus_candidate_delta_late_fusion, "
            "shared_plus_candidate_conditioned_late_fusion."
        )
    return decision_mode


def _resolve_memory_control(config: dict[str, Any]) -> str:
    memory_control = str(config["runtime"].get("stage_c_memory_control", "real"))
    if memory_control not in {"real", "shuffled", "zero"}:
        raise ValueError(
            f"Unsupported runtime.stage_c_memory_control={memory_control}. "
            "Expected one of real, shuffled, zero."
        )
    return memory_control


def _resolve_candidate_memory_control(
    config: dict[str, Any],
    *,
    decision_mode: str,
    default_memory_control: str,
) -> str:
    raw_value = config["runtime"].get("stage_c_candidate_memory_control")
    if raw_value is None:
        if decision_mode == "shared_plus_candidate_conditioned_late_fusion":
            return "real"
        return default_memory_control
    memory_control = str(raw_value)
    if memory_control not in {"real", "shuffled", "zero"}:
        raise ValueError(
            f"Unsupported runtime.stage_c_candidate_memory_control={memory_control}. "
            "Expected one of real, shuffled, zero."
        )
    return memory_control


def _resolve_choice_objective(config: dict[str, Any]) -> str:
    choice_objective = str(config["runtime"].get("stage_c_choice_objective", "continuation_retrieval"))
    if choice_objective not in {
        "continuation_retrieval",
        "choice_ce_plus_margin",
        "choice_repair_ce_margin",
    }:
        raise ValueError(
            f"Unsupported runtime.stage_c_choice_objective={choice_objective}. "
            "Expected one of continuation_retrieval, choice_ce_plus_margin, choice_repair_ce_margin."
        )
    return choice_objective


def _resolve_residual_calibration_mode(config: dict[str, Any]) -> str:
    calibration_mode = str(config["runtime"].get("stage_c_residual_calibration_mode", "none"))
    if calibration_mode not in {"none", "support_grid_search"}:
        raise ValueError(
            f"Unsupported runtime.stage_c_residual_calibration_mode={calibration_mode}. "
            "Expected one of none, support_grid_search."
        )
    return calibration_mode


def _resolve_residual_calibration_alpha_grid(
    config: dict[str, Any],
    *,
    configured_residual_scale: float,
) -> list[float]:
    raw_grid = config["runtime"].get("stage_c_residual_calibration_alpha_grid")
    if raw_grid is None:
        raw_values: list[float] = [
            0.0,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            20.0,
            40.0,
            80.0,
            100.0,
            200.0,
            500.0,
            1000.0,
            2000.0,
            3200.0,
            4000.0,
            5000.0,
        ]
    elif isinstance(raw_grid, str):
        raw_values = [float(part.strip()) for part in raw_grid.split(",") if part.strip()]
    else:
        raw_values = [float(value) for value in raw_grid]
    ordered: list[float] = []
    seen: set[float] = set()
    for value in [float(configured_residual_scale), *raw_values]:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _resolve_candidate_delta_scale(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("stage_c_candidate_delta_scale", 1.0))


def _resolve_candidate_residual_scale(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("stage_c_candidate_residual_scale", 1.0))


def _resolve_candidate_delta_gate_tau(config: dict[str, Any]) -> float:
    gate_tau = float(config["runtime"].get("stage_c_candidate_delta_gate_tau", 0.01))
    if gate_tau <= 0.0:
        raise ValueError("runtime.stage_c_candidate_delta_gate_tau must be > 0.0")
    return gate_tau


def _resolve_repair_anchor_margin_threshold(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("stage_c_repair_anchor_margin_threshold", 0.01))


def _resolve_story_text(example: dict[str, Any]) -> str:
    story_text = str(example.get("story", "")).strip()
    if story_text:
        return story_text
    segment = str(example["segment"])
    marker = "|| Candidate endings:"
    if marker in segment:
        return segment.split(marker, maxsplit=1)[0].replace("Story:", "", 1).strip()
    return segment


def _resolve_choice_scoring_prompt(example: dict[str, Any]) -> str:
    prompt = str(example.get("choice_scoring_prompt", "")).strip()
    if prompt:
        return prompt
    story_text = _resolve_story_text(example)
    return f"Story: {story_text} || Candidate ending:"


def _build_task_evaluator(example: dict[str, Any]) -> TaskEvaluator:
    benchmark_id = str(example["benchmark_id"])
    spec = get_task_spec(benchmark_id)
    return TaskEvaluator(
        evaluator_type=str(example.get("evaluator_type", spec.evaluator_type)),
        metric_name=str(example.get("metric_name", spec.metric_name)),
        normalizer=str(example.get("normalizer", spec.normalizer)),
        benchmark_id=benchmark_id,
    )


def _build_example_caches(
    runtime: MemoryRuntime,
    examples: list[dict[str, Any]],
) -> dict[str, PilotExampleCache]:
    caches: dict[str, PilotExampleCache] = {}
    for example in examples:
        story_text = _resolve_story_text(example)
        story_state = runtime.backbone.summarize_texts([story_text])
        _, conditioning_state = runtime._resolve_conditioning(example)
        memory_long = runtime.writer.write(story_state)
        choices = example.get("choices", [])
        if not choices:
            raise ValueError("Stage C real pilot currently supports only multiple-choice examples.")
        candidate_labels = [str(choice["label"]) for choice in choices]
        candidate_texts = [str(choice["text"]) for choice in choices]
        candidate_states = runtime.backbone.summarize_texts(candidate_texts)
        choice_scoring_prompt = _resolve_choice_scoring_prompt(example)
        base_scores = runtime.backbone.score_continuations(choice_scoring_prompt, candidate_texts).detach()
        continuation_state = candidate_states[candidate_labels.index(str(example["label"])) : candidate_labels.index(str(example["label"])) + 1]
        caches[str(example["id"])] = PilotExampleCache(
            example=example,
            story_state=story_state.detach(),
            conditioning_state=conditioning_state.detach(),
            memory_long=memory_long.detach(),
            candidate_states=candidate_states.detach(),
            continuation_state=continuation_state.detach(),
            base_scores=base_scores.detach(),
            candidate_labels=candidate_labels,
            candidate_texts=candidate_texts,
            choice_scoring_prompt=choice_scoring_prompt,
        )
    return caches


def _resolve_memory_source_cache(
    example_cache: PilotExampleCache,
    cache_lookup: dict[str, PilotExampleCache],
    *,
    memory_control: str,
) -> PilotExampleCache | None:
    if memory_control == "real":
        return cache_lookup[str(example_cache.example["id"])]
    if memory_control == "zero":
        return None
    shuffled_id = str(example_cache.example.get("shuffled_memory_example_id", "")).strip()
    if not shuffled_id:
        raise ValueError(
            f"Example {example_cache.example['id']} is missing shuffled_memory_example_id "
            "required for runtime.stage_c_memory_control=shuffled."
        )
    if shuffled_id not in cache_lookup:
        raise ValueError(
            f"Example {example_cache.example['id']} requests shuffled_memory_example_id={shuffled_id}, "
            "but that example is missing from the current cache lookup."
        )
    return cache_lookup[shuffled_id]


def _resolve_memory_long(
    example_cache: PilotExampleCache,
    memory_source_cache: PilotExampleCache | None,
) -> torch.Tensor:
    if memory_source_cache is None:
        return torch.zeros_like(example_cache.memory_long)
    return memory_source_cache.memory_long


def _base_reader_context(example_cache: PilotExampleCache) -> torch.Tensor:
    return (example_cache.story_state + example_cache.conditioning_state) / 2.0


def _shared_memory_summary(
    runtime: MemoryRuntime,
    example_cache: PilotExampleCache,
    *,
    memory_source_cache: PilotExampleCache | None,
) -> torch.Tensor:
    memory_long = _resolve_memory_long(example_cache, memory_source_cache)
    reader_output = runtime.reader.read(memory_long, context=_base_reader_context(example_cache))
    memory_short = runtime.fuser.fuse(reader_output["readouts"])
    return runtime.summarize_memory_short(memory_short)


def _shared_residual_scores(
    runtime: MemoryRuntime,
    example_cache: PilotExampleCache,
    *,
    memory_source_cache: PilotExampleCache | None,
) -> torch.Tensor:
    memory_summary = _shared_memory_summary(
        runtime,
        example_cache,
        memory_source_cache=memory_source_cache,
    )
    return runtime.score_candidates(memory_summary, example_cache.candidate_states)


def _candidate_conditioned_residual_scores(
    runtime: MemoryRuntime,
    example_cache: PilotExampleCache,
    *,
    memory_source_cache: PilotExampleCache | None,
) -> torch.Tensor:
    memory_long = _resolve_memory_long(example_cache, memory_source_cache)
    base_context = _base_reader_context(example_cache)
    residual_scores: list[torch.Tensor] = []
    for candidate_index in range(len(example_cache.candidate_labels)):
        candidate_state = example_cache.candidate_states[candidate_index : candidate_index + 1]
        reader_output = runtime.reader.read(
            memory_long,
            context=base_context,
            candidate_state=candidate_state,
        )
        memory_short = runtime.fuser.fuse(reader_output["readouts"])
        conditioned_summary = runtime.summarize_memory_short(memory_short)
        residual_scores.append(runtime.score_candidates(conditioned_summary, candidate_state).squeeze(0))
    return torch.stack(residual_scores)


def _top2_gap(scores: torch.Tensor) -> float:
    if scores.numel() < 2:
        return 0.0
    topk = torch.topk(scores, k=2, dim=0).values
    return float((topk[0] - topk[1]).item())


def _compute_candidate_delta_gate(shared_residual_scores: torch.Tensor, *, gate_tau: float) -> float:
    if gate_tau <= 0.0:
        raise ValueError("gate_tau must be > 0.0")
    gap = _top2_gap(shared_residual_scores)
    return float(max(0.0, min(1.0, 1.0 - (gap / gate_tau))))


def _zero_mean_candidate_delta(raw_delta_scores: torch.Tensor) -> torch.Tensor:
    if raw_delta_scores.numel() == 0:
        return raw_delta_scores
    return raw_delta_scores - raw_delta_scores.mean()


def _compose_shared_plus_candidate_delta_scores(
    *,
    base_scores: torch.Tensor,
    shared_residual_scores: torch.Tensor,
    conditioned_residual_scores: torch.Tensor,
    shared_scale: float,
    delta_scale: float,
    gate_tau: float,
) -> dict[str, torch.Tensor | float]:
    raw_candidate_delta_scores = conditioned_residual_scores - shared_residual_scores
    candidate_delta_scores = _zero_mean_candidate_delta(raw_candidate_delta_scores)
    candidate_delta_gate = _compute_candidate_delta_gate(shared_residual_scores, gate_tau=gate_tau)
    shared_choice_scores = base_scores + (float(shared_scale) * shared_residual_scores)
    delta_contribution = candidate_delta_gate * float(delta_scale) * candidate_delta_scores
    final_choice_scores = shared_choice_scores + delta_contribution
    total_memory_residual_scores = final_choice_scores - base_scores
    return {
        "base_scores": base_scores,
        "shared_residual_scores": shared_residual_scores,
        "conditioned_residual_scores": conditioned_residual_scores,
        "raw_candidate_delta_scores": raw_candidate_delta_scores,
        "candidate_delta_scores": candidate_delta_scores,
        "candidate_delta_gate": float(candidate_delta_gate),
        "shared_choice_scores": shared_choice_scores,
        "memory_residual_scores": total_memory_residual_scores,
        "final_choice_scores": final_choice_scores,
    }


def _compose_shared_plus_candidate_conditioned_scores(
    *,
    base_scores: torch.Tensor,
    shared_residual_scores: torch.Tensor,
    conditioned_residual_scores: torch.Tensor,
    shared_scale: float,
    candidate_residual_scale: float,
) -> dict[str, torch.Tensor | float]:
    shared_choice_scores = base_scores + (float(shared_scale) * shared_residual_scores)
    candidate_residual_scores = float(candidate_residual_scale) * conditioned_residual_scores
    final_choice_scores = shared_choice_scores + candidate_residual_scores
    total_memory_residual_scores = final_choice_scores - base_scores
    return {
        "base_scores": base_scores,
        "shared_residual_scores": shared_residual_scores,
        "conditioned_residual_scores": conditioned_residual_scores,
        "raw_candidate_delta_scores": torch.zeros_like(base_scores),
        "candidate_delta_scores": torch.zeros_like(base_scores),
        "candidate_delta_gate": 1.0,
        "shared_choice_scores": shared_choice_scores,
        "memory_residual_scores": total_memory_residual_scores,
        "final_choice_scores": final_choice_scores,
    }


def _build_repair_anchor(
    *,
    choice_scores: torch.Tensor,
    candidate_labels: list[str],
    gold_label: str,
    margin_threshold: float,
) -> RepairAnchor:
    gold_index = candidate_labels.index(gold_label)
    predicted_index = int(torch.argmax(choice_scores).item())
    other_indices = [index for index in range(len(candidate_labels)) if index != gold_index]
    competitor_index = (
        max(other_indices, key=lambda index: float(choice_scores[index].item()))
        if other_indices
        else gold_index
    )
    margin = float(choice_scores[gold_index].item() - choice_scores[competitor_index].item())
    predicted_label = candidate_labels[predicted_index]
    competitor_label = candidate_labels[competitor_index]
    if predicted_index != gold_index:
        repair_bucket = "anchor_wrong"
        repair_weight = 1.0
    elif margin < float(margin_threshold):
        repair_bucket = "anchor_near_threshold"
        repair_weight = 1.0
    else:
        repair_bucket = "anchor_confident_correct"
        repair_weight = 0.0
    return RepairAnchor(
        choice_scores=choice_scores.detach().cpu(),
        gold_index=gold_index,
        predicted_index=predicted_index,
        competitor_index=competitor_index,
        margin=margin,
        predicted_label=predicted_label,
        competitor_label=competitor_label,
        repair_weight=float(repair_weight),
        repair_active=bool(repair_weight > 0.0),
        repair_bucket=repair_bucket,
    )


def _build_repair_anchor_lookup(
    runtime: MemoryRuntime,
    example_caches: dict[str, PilotExampleCache],
    *,
    residual_scale: float,
    margin_threshold: float,
) -> dict[str, RepairAnchor]:
    anchors: dict[str, RepairAnchor] = {}
    with torch.no_grad():
        for example_id, example_cache in example_caches.items():
            shared_residual_scores = _shared_residual_scores(
                runtime,
                example_cache,
                memory_source_cache=_resolve_memory_source_cache(
                    example_cache,
                    example_caches,
                    memory_control="real",
                ),
            )
            shared_choice_scores = example_cache.base_scores + (float(residual_scale) * shared_residual_scores)
            anchors[example_id] = _build_repair_anchor(
                choice_scores=shared_choice_scores,
                candidate_labels=example_cache.candidate_labels,
                gold_label=str(example_cache.example["label"]),
                margin_threshold=margin_threshold,
            )
    return anchors


def _choice_repair_ce_margin_from_scores(
    final_scores: torch.Tensor,
    *,
    gold_index: int,
    competitor_index: int,
    repair_active: bool,
    margin_value: float,
) -> tuple[torch.Tensor, float]:
    accuracy = float(int(torch.argmax(final_scores).item()) == gold_index)
    if not repair_active:
        return final_scores.sum() * 0.0, accuracy
    ce_loss = F.cross_entropy(
        final_scores.unsqueeze(0),
        torch.tensor([gold_index], dtype=torch.long, device=final_scores.device),
    )
    repair_pair_scores = torch.stack(
        [
            final_scores[gold_index],
            final_scores[competitor_index],
        ]
    )
    margin_loss = _pairwise_margin_loss(repair_pair_scores, margin_value=margin_value)
    return ce_loss + margin_loss, accuracy


def _score_choice_components(
    runtime: MemoryRuntime,
    example_cache: PilotExampleCache,
    *,
    decision_mode: str,
    memory_control: str,
    candidate_memory_control: str,
    cache_lookup: dict[str, PilotExampleCache],
    residual_scale: float,
    candidate_residual_scale: float,
    candidate_delta_scale: float,
    candidate_delta_gate_tau: float,
) -> dict[str, torch.Tensor | float]:
    memory_source_cache = _resolve_memory_source_cache(
        example_cache,
        cache_lookup,
        memory_control=memory_control,
    )
    zero_scores = torch.zeros_like(example_cache.base_scores)
    base_scores = example_cache.base_scores.detach().clone()
    if decision_mode == "base_only":
        return {
            "base_scores": base_scores,
            "shared_residual_scores": zero_scores.clone(),
            "conditioned_residual_scores": zero_scores.clone(),
            "raw_candidate_delta_scores": zero_scores.clone(),
            "candidate_delta_scores": zero_scores.clone(),
            "candidate_delta_gate": 0.0,
            "shared_choice_scores": base_scores.clone(),
            "memory_residual_scores": zero_scores.clone(),
            "final_choice_scores": base_scores.clone(),
        }

    if decision_mode == "shared_summary_late_fusion":
        shared_residual_scores = _shared_residual_scores(
            runtime,
            example_cache,
            memory_source_cache=memory_source_cache,
        )
        memory_residual_scores = float(residual_scale) * shared_residual_scores
        final_scores = base_scores + memory_residual_scores
        return {
            "base_scores": base_scores,
            "shared_residual_scores": shared_residual_scores,
            "conditioned_residual_scores": zero_scores.clone(),
            "raw_candidate_delta_scores": zero_scores.clone(),
            "candidate_delta_scores": zero_scores.clone(),
            "candidate_delta_gate": 0.0,
            "shared_choice_scores": final_scores.clone(),
            "memory_residual_scores": memory_residual_scores,
            "final_choice_scores": final_scores,
        }

    conditioned_residual_scores = _candidate_conditioned_residual_scores(
        runtime,
        example_cache,
        memory_source_cache=_resolve_memory_source_cache(
            example_cache,
            cache_lookup,
            memory_control=candidate_memory_control,
        ),
    )
    if decision_mode == "candidate_conditioned_late_fusion":
        memory_residual_scores = float(residual_scale) * conditioned_residual_scores
        final_scores = base_scores + memory_residual_scores
        return {
            "base_scores": base_scores,
            "shared_residual_scores": zero_scores.clone(),
            "conditioned_residual_scores": conditioned_residual_scores,
            "raw_candidate_delta_scores": conditioned_residual_scores.clone(),
            "candidate_delta_scores": conditioned_residual_scores.clone(),
            "candidate_delta_gate": 1.0,
            "shared_choice_scores": base_scores.clone(),
            "memory_residual_scores": memory_residual_scores,
            "final_choice_scores": final_scores,
        }

    if decision_mode == "shared_plus_candidate_conditioned_late_fusion":
        shared_residual_scores = _shared_residual_scores(
            runtime,
            example_cache,
            memory_source_cache=memory_source_cache,
        )
        return _compose_shared_plus_candidate_conditioned_scores(
            base_scores=base_scores,
            shared_residual_scores=shared_residual_scores,
            conditioned_residual_scores=conditioned_residual_scores,
            shared_scale=residual_scale,
            candidate_residual_scale=candidate_residual_scale,
        )

    shared_residual_scores = _shared_residual_scores(
        runtime,
        example_cache,
        memory_source_cache=memory_source_cache,
    )
    return _compose_shared_plus_candidate_delta_scores(
        base_scores=base_scores,
        shared_residual_scores=shared_residual_scores,
        conditioned_residual_scores=conditioned_residual_scores,
        shared_scale=residual_scale,
        delta_scale=candidate_delta_scale,
        gate_tau=candidate_delta_gate_tau,
    )


def _score_multiple_choice_example(
    runtime: MemoryRuntime,
    example_cache: PilotExampleCache,
    *,
    decision_mode: str,
    memory_control: str,
    candidate_memory_control: str,
    cache_lookup: dict[str, PilotExampleCache],
    residual_scale: float,
    candidate_residual_scale: float,
    candidate_delta_scale: float,
    candidate_delta_gate_tau: float,
    repair_anchor: RepairAnchor | None,
) -> dict[str, Any]:
    evaluator = _build_task_evaluator(example_cache.example)
    score_components = _score_choice_components(
        runtime,
        example_cache,
        decision_mode=decision_mode,
        memory_control=memory_control,
        candidate_memory_control=candidate_memory_control,
        cache_lookup=cache_lookup,
        residual_scale=residual_scale,
        candidate_residual_scale=candidate_residual_scale,
        candidate_delta_scale=candidate_delta_scale,
        candidate_delta_gate_tau=candidate_delta_gate_tau,
    )
    base_scores = score_components["base_scores"]
    shared_residual_scores = score_components["shared_residual_scores"]
    conditioned_residual_scores = score_components["conditioned_residual_scores"]
    raw_candidate_delta_scores = score_components["raw_candidate_delta_scores"]
    candidate_delta_scores = score_components["candidate_delta_scores"]
    shared_choice_scores = score_components["shared_choice_scores"]
    residual_scores = score_components["memory_residual_scores"]
    final_scores = score_components["final_choice_scores"]
    candidate_delta_gate = float(score_components["candidate_delta_gate"])
    base_probabilities = torch.softmax(base_scores, dim=0)
    shared_probabilities = torch.softmax(shared_choice_scores, dim=0)
    final_probabilities = torch.softmax(final_scores, dim=0)
    gold_label = str(example_cache.example["label"])
    gold_index = example_cache.candidate_labels.index(gold_label)
    base_predicted_index = int(torch.argmax(base_scores).item())
    shared_predicted_index = int(torch.argmax(shared_choice_scores).item())
    final_predicted_index = int(torch.argmax(final_scores).item())
    base_other_indices = [index for index in range(len(example_cache.candidate_labels)) if index != gold_index]
    base_competitor_index = (
        max(base_other_indices, key=lambda index: float(base_scores[index].item()))
        if base_other_indices
        else gold_index
    )
    shared_competitor_index = (
        max(base_other_indices, key=lambda index: float(shared_choice_scores[index].item()))
        if base_other_indices
        else gold_index
    )
    final_competitor_index = (
        max(base_other_indices, key=lambda index: float(final_scores[index].item()))
        if base_other_indices
        else gold_index
    )
    predicted_label = example_cache.candidate_labels[final_predicted_index]
    predicted_text = example_cache.candidate_texts[final_predicted_index]
    score_payload = evaluator.evaluate_prediction(
        {"label": predicted_label, "text": predicted_text},
        example_cache.example,
    )
    return {
        "task_score": float(score_payload["score"]),
        "task_metric_name": evaluator.metric_name,
        "task_proxy_score": float(final_probabilities[gold_index].item()),
        "task_proxy_name": "gold_choice_probability",
        "task_margin": float(final_scores[gold_index].item() - final_scores[final_competitor_index].item()),
        "objective_accuracy": float(final_predicted_index == gold_index),
        "base_choice_scores": [float(value) for value in base_scores.tolist()],
        "memory_residual_scores": [float(value) for value in residual_scores.tolist()],
        "shared_residual_scores": [float(value) for value in shared_residual_scores.tolist()],
        "conditioned_residual_scores": [float(value) for value in conditioned_residual_scores.tolist()],
        "raw_candidate_delta_scores": [float(value) for value in raw_candidate_delta_scores.tolist()],
        "candidate_delta_scores": [float(value) for value in candidate_delta_scores.tolist()],
        "candidate_delta_gate": float(candidate_delta_gate),
        "shared_choice_scores": [float(value) for value in shared_choice_scores.tolist()],
        "final_choice_scores": [float(value) for value in final_scores.tolist()],
        "base_margin": float(base_scores[gold_index].item() - base_scores[base_competitor_index].item()),
        "shared_margin": float(
            shared_choice_scores[gold_index].item() - shared_choice_scores[shared_competitor_index].item()
        ),
        "final_margin": float(final_scores[gold_index].item() - final_scores[final_competitor_index].item()),
        "base_predicted_label": example_cache.candidate_labels[base_predicted_index],
        "shared_predicted_label": example_cache.candidate_labels[shared_predicted_index],
        "final_predicted_label": predicted_label,
        "base_top_competitor_label": example_cache.candidate_labels[base_competitor_index],
        "shared_top_competitor_label": example_cache.candidate_labels[shared_competitor_index],
        "final_top_competitor_label": example_cache.candidate_labels[final_competitor_index],
        "base_top_competitor_text": example_cache.candidate_texts[base_competitor_index],
        "shared_top_competitor_text": example_cache.candidate_texts[shared_competitor_index],
        "final_top_competitor_text": example_cache.candidate_texts[final_competitor_index],
        "candidate_branch_memory_control": candidate_memory_control,
        "candidate_conditioned_residual_scores": [float(value) for value in conditioned_residual_scores.tolist()],
        "anchor_shared_choice_scores": []
        if repair_anchor is None
        else [float(value) for value in repair_anchor.choice_scores.tolist()],
        "anchor_shared_margin": float("nan") if repair_anchor is None else float(repair_anchor.margin),
        "anchor_shared_predicted_label": "" if repair_anchor is None else str(repair_anchor.predicted_label),
        "repair_competitor_label": "" if repair_anchor is None else str(repair_anchor.competitor_label),
        "repair_weight": 0.0 if repair_anchor is None else float(repair_anchor.repair_weight),
        "repair_active": False if repair_anchor is None else bool(repair_anchor.repair_active),
        "repair_bucket": "" if repair_anchor is None else str(repair_anchor.repair_bucket),
        "gold_label": gold_label,
        "gold_text": example_cache.candidate_texts[gold_index],
        "predicted_correct": bool(final_predicted_index == gold_index),
        "predicted_text": predicted_text,
        "base_probabilities": [float(value) for value in base_probabilities.tolist()],
        "shared_probabilities": [float(value) for value in shared_probabilities.tolist()],
        "final_probabilities": [float(value) for value in final_probabilities.tolist()],
        "choices": [
            {
                "label": example_cache.candidate_labels[index],
                "text": example_cache.candidate_texts[index],
                "base_score": float(base_scores[index].item()),
                "shared_residual_score": float(shared_residual_scores[index].item()),
                "conditioned_residual_score": float(conditioned_residual_scores[index].item()),
                "raw_candidate_delta_score": float(raw_candidate_delta_scores[index].item()),
                "candidate_delta_score": float(candidate_delta_scores[index].item()),
                "residual_score": float(residual_scores[index].item()),
                "shared_score": float(shared_choice_scores[index].item()),
                "final_score": float(final_scores[index].item()),
                "base_probability": float(base_probabilities[index].item()),
                "shared_probability": float(shared_probabilities[index].item()),
                "final_probability": float(final_probabilities[index].item()),
                "is_gold": index == gold_index,
                "is_base_predicted": index == base_predicted_index,
                "is_shared_predicted": index == shared_predicted_index,
                "is_final_predicted": index == final_predicted_index,
            }
            for index in range(len(example_cache.candidate_labels))
        ],
    }


def _choice_ce_plus_margin_support_loss(
    runtime: MemoryRuntime,
    example_cache: PilotExampleCache,
    *,
    cache_lookup: dict[str, PilotExampleCache],
    residual_scale: float,
    candidate_residual_scale: float,
    candidate_delta_scale: float,
    candidate_delta_gate_tau: float,
    memory_control: str,
    candidate_memory_control: str,
    margin_value: float,
    decision_mode: str,
) -> tuple[torch.Tensor, float]:
    score_components = _score_choice_components(
        runtime,
        example_cache,
        decision_mode=decision_mode,
        memory_control=memory_control,
        candidate_memory_control=candidate_memory_control,
        cache_lookup=cache_lookup,
        residual_scale=residual_scale,
        candidate_residual_scale=candidate_residual_scale,
        candidate_delta_scale=candidate_delta_scale,
        candidate_delta_gate_tau=candidate_delta_gate_tau,
    )
    final_scores = score_components["final_choice_scores"]
    gold_index = example_cache.candidate_labels.index(str(example_cache.example["label"]))
    ce_loss = F.cross_entropy(
        final_scores.unsqueeze(0),
        torch.tensor([gold_index], dtype=torch.long, device=final_scores.device),
    )
    margin_loss = _pairwise_margin_loss(
        torch.cat(
            [
                final_scores[gold_index : gold_index + 1],
                final_scores[
                    torch.tensor(
                        [index for index in range(len(example_cache.candidate_labels)) if index != gold_index],
                        device=final_scores.device,
                        dtype=torch.long,
                    )
                ]
                if len(example_cache.candidate_labels) > 1
                else final_scores.new_zeros((0,)),
            ]
        ),
        margin_value=margin_value,
    )
    accuracy = float(int(torch.argmax(final_scores).item()) == gold_index)
    return ce_loss + margin_loss, accuracy


def _choice_repair_ce_margin_support_loss(
    runtime: MemoryRuntime,
    example_cache: PilotExampleCache,
    *,
    repair_anchor: RepairAnchor,
    cache_lookup: dict[str, PilotExampleCache],
    residual_scale: float,
    candidate_residual_scale: float,
    candidate_delta_scale: float,
    candidate_delta_gate_tau: float,
    memory_control: str,
    candidate_memory_control: str,
    margin_value: float,
    decision_mode: str,
) -> tuple[torch.Tensor, float]:
    score_components = _score_choice_components(
        runtime,
        example_cache,
        decision_mode=decision_mode,
        memory_control=memory_control,
        candidate_memory_control=candidate_memory_control,
        cache_lookup=cache_lookup,
        residual_scale=residual_scale,
        candidate_residual_scale=candidate_residual_scale,
        candidate_delta_scale=candidate_delta_scale,
        candidate_delta_gate_tau=candidate_delta_gate_tau,
    )
    final_scores = score_components["final_choice_scores"]
    return _choice_repair_ce_margin_from_scores(
        final_scores,
        gold_index=repair_anchor.gold_index,
        competitor_index=repair_anchor.competitor_index,
        repair_active=repair_anchor.repair_active,
        margin_value=margin_value,
    )


def _continuation_retrieval_support_loss(
    runtime: MemoryRuntime,
    example_cache: PilotExampleCache,
    *,
    support_caches: list[PilotExampleCache],
    cache_lookup: dict[str, PilotExampleCache],
    memory_control: str,
    margin_value: float,
) -> tuple[torch.Tensor, float]:
    memory_summary = _shared_memory_summary(
        runtime,
        example_cache,
        memory_source_cache=_resolve_memory_source_cache(
            example_cache,
            cache_lookup,
            memory_control=memory_control,
        ),
    )
    candidate_caches = [example_cache] + [cache for cache in support_caches if cache.example["id"] != example_cache.example["id"]]
    candidate_states = torch.cat([cache.continuation_state for cache in candidate_caches], dim=0)
    scores = runtime.score_candidates(memory_summary, candidate_states)
    ce_loss = F.cross_entropy(
        scores.unsqueeze(0),
        torch.tensor([0], dtype=torch.long, device=scores.device),
    )
    margin_loss = _pairwise_margin_loss(scores, margin_value=margin_value)
    accuracy = float(int(torch.argmax(scores).item()) == 0)
    return ce_loss + margin_loss, accuracy


def _aggregate_evaluation(
    runtime: MemoryRuntime,
    eval_caches: list[PilotExampleCache],
    *,
    decision_mode: str,
    memory_control: str,
    candidate_memory_control: str,
    cache_lookup: dict[str, PilotExampleCache],
    residual_scale: float,
    candidate_residual_scale: float,
    candidate_delta_scale: float,
    candidate_delta_gate_tau: float,
    residual_calibration_mode: str,
    step: int,
    seed: int,
    support_ids: list[str],
    support_objective: str,
    repair_anchor_lookup: dict[str, RepairAnchor],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_scores: list[float] = []
    task_proxy_scores: list[float] = []
    task_margins: list[float] = []
    objective_accuracies: list[float] = []
    case_rows: list[dict[str, Any]] = []
    for example_cache in eval_caches:
        payload = _score_multiple_choice_example(
            runtime,
            example_cache,
            decision_mode=decision_mode,
            memory_control=memory_control,
            candidate_memory_control=candidate_memory_control,
            cache_lookup=cache_lookup,
            residual_scale=residual_scale,
            candidate_residual_scale=candidate_residual_scale,
            candidate_delta_scale=candidate_delta_scale,
            candidate_delta_gate_tau=candidate_delta_gate_tau,
            repair_anchor=repair_anchor_lookup.get(str(example_cache.example["id"])),
        )
        task_scores.append(float(payload["task_score"]))
        task_proxy_scores.append(float(payload["task_proxy_score"]))
        task_margins.append(float(payload["task_margin"]))
        objective_accuracies.append(float(payload["objective_accuracy"]))
        case_rows.append(
            {
                "step": step,
                "seed": seed,
                "backbone": runtime.backbone.name,
                "training_stage": "stage_c_real_pilot",
                "decision_mode": decision_mode,
                "memory_control_mode": memory_control,
                "candidate_branch_memory_control": candidate_memory_control,
                "choice_objective": support_objective,
                "residual_calibration_mode": residual_calibration_mode,
                "effective_residual_scale": float(residual_scale),
                "support_ids": support_ids,
                "example_id": str(example_cache.example["id"]),
                "benchmark_id": str(example_cache.example["benchmark_id"]),
                "domain": str(example_cache.example["domain"]),
                "task_name": str(example_cache.example["task_name"]),
                "segment": str(example_cache.example["segment"]),
                "story": str(example_cache.example.get("story", "")),
                "screening_bucket": str(example_cache.example.get("screening_bucket", "")),
                "screening_split": str(example_cache.example.get("screening_split", "")),
                "task_score": float(payload["task_score"]),
                "task_metric_name": str(payload["task_metric_name"]),
                "task_proxy_score": float(payload["task_proxy_score"]),
                "task_proxy_name": str(payload["task_proxy_name"]),
                "task_margin": float(payload["task_margin"]),
                **payload,
            }
        )
    metrics = {
        "objective_accuracy": sum(objective_accuracies) / len(objective_accuracies),
        "task_score": sum(task_scores) / len(task_scores),
        "task_metric_name": "accuracy",
        "task_proxy_score": sum(task_proxy_scores) / len(task_proxy_scores),
        "task_proxy_name": "gold_choice_probability",
        "task_margin": sum(task_margins) / len(task_margins),
    }
    return metrics, case_rows


def _curve_row_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["task_score"]),
        float(row["task_proxy_score"]),
        float(row["task_margin"]),
        -int(row["step"]),
    )


def _pick_residual_calibration_row(
    rows: list[dict[str, float]],
    *,
    configured_residual_scale: float,
) -> dict[str, float]:
    if not rows:
        raise ValueError("Expected at least one residual calibration row.")
    return max(
        rows,
        key=lambda row: (
            float(row["task_score"]),
            float(row["task_proxy_score"]),
            float(row["task_margin"]),
            -abs(float(row["alpha"]) - float(configured_residual_scale)),
            -float(row["alpha"]),
        ),
    )


def _resolve_effective_residual_scale(
    runtime: MemoryRuntime,
    support_caches: list[PilotExampleCache],
    *,
    decision_mode: str,
    memory_control: str,
    candidate_memory_control: str,
    cache_lookup: dict[str, PilotExampleCache],
    configured_residual_scale: float,
    calibration_mode: str,
    calibration_alpha_grid: list[float],
    candidate_residual_scale: float,
    candidate_delta_scale: float,
    candidate_delta_gate_tau: float,
    repair_anchor_lookup: dict[str, RepairAnchor],
) -> tuple[float, dict[str, float]]:
    if decision_mode == "base_only" or calibration_mode == "none" or not support_caches:
        return float(configured_residual_scale), {
            "alpha": float(configured_residual_scale),
            "task_score": 0.0,
            "task_proxy_score": 0.0,
            "task_margin": 0.0,
        }
    calibration_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for alpha in calibration_alpha_grid:
            task_scores: list[float] = []
            task_proxy_scores: list[float] = []
            task_margins: list[float] = []
            for example_cache in support_caches:
                payload = _score_multiple_choice_example(
                    runtime,
                    example_cache,
                    decision_mode=decision_mode,
                    memory_control=memory_control,
                    candidate_memory_control=candidate_memory_control,
                    cache_lookup=cache_lookup,
                    residual_scale=float(alpha),
                    candidate_residual_scale=candidate_residual_scale,
                    candidate_delta_scale=candidate_delta_scale,
                    candidate_delta_gate_tau=candidate_delta_gate_tau,
                    repair_anchor=repair_anchor_lookup.get(str(example_cache.example["id"])),
                )
                task_scores.append(float(payload["task_score"]))
                task_proxy_scores.append(float(payload["task_proxy_score"]))
                task_margins.append(float(payload["task_margin"]))
            calibration_rows.append(
                {
                    "alpha": float(alpha),
                    "task_score": sum(task_scores) / len(task_scores),
                    "task_proxy_score": sum(task_proxy_scores) / len(task_proxy_scores),
                    "task_margin": sum(task_margins) / len(task_margins),
                }
            )
    best_row = _pick_residual_calibration_row(
        calibration_rows,
        configured_residual_scale=configured_residual_scale,
    )
    return float(best_row["alpha"]), best_row


def run_stage_c_real_pilot(
    *,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    resume: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    decision_mode = _resolve_decision_mode(config)
    memory_control = _resolve_memory_control(config)
    candidate_memory_control = _resolve_candidate_memory_control(
        config,
        decision_mode=decision_mode,
        default_memory_control=memory_control,
    )
    support_objective = _resolve_choice_objective(config)
    residual_calibration_mode = _resolve_residual_calibration_mode(config)
    residual_scale = float(config["runtime"].get("stage_c_choice_residual_scale", 1.0))
    candidate_residual_scale = _resolve_candidate_residual_scale(config)
    candidate_delta_scale = _resolve_candidate_delta_scale(config)
    candidate_delta_gate_tau = _resolve_candidate_delta_gate_tau(config)
    repair_anchor_margin_threshold = _resolve_repair_anchor_margin_threshold(config)
    calibration_alpha_grid = _resolve_residual_calibration_alpha_grid(
        config,
        configured_residual_scale=residual_scale,
    )
    choice_margin = float(config["runtime"].get("stage_c_choice_margin", 0.1))
    support_learning_rate = float(config["runtime"].get("pilot_support_learning_rate", 0.2))
    adapt_steps = 0 if decision_mode == "base_only" else (1 if dry_run else int(config["runtime"].get("pilot_adapt_steps", 3)))
    support_limit = max(0, int(config["runtime"].get("pilot_support_examples", 8)))
    expected_query_learning_mode = _resolve_expected_stage_c_query_learning_mode(config)
    runtime = MemoryRuntime(config=config, seed=seed)

    support_dataset_path = str(config["task"]["support_dataset_path"])
    calibration_dataset_path = str(config["task"].get("calibration_dataset_path", "")).strip()
    eval_examples = load_task_dataset(config)
    support_examples = _load_task_dataset_from_path(config, support_dataset_path)
    calibration_examples = (
        _load_task_dataset_from_path(config, calibration_dataset_path)
        if calibration_dataset_path
        else []
    )
    if support_limit:
        support_examples = support_examples[:support_limit]
    eval_examples = eval_examples[: min(len(eval_examples), 4 if dry_run else len(eval_examples))]
    combined_examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for example in [*support_examples, *calibration_examples, *eval_examples]:
        example_id = str(example["id"])
        if example_id in seen_ids:
            continue
        seen_ids.add(example_id)
        combined_examples.append(example)

    if decision_mode != "base_only":
        writer_path = _resolve_artifact_path(resume, "writer.ckpt")
        queries_path = _resolve_artifact_path(resume, "queries_meta_init.pt")
        runtime.writer.load_from(writer_path)
        state = torch.load(queries_path, map_location="cpu")
        query_learning_mode = str(state.get("query_learning_mode", "meta_trained"))
        if expected_query_learning_mode is not None and query_learning_mode != expected_query_learning_mode:
            raise ValueError(
                f"Stage C real pilot expected query_learning_mode={expected_query_learning_mode}, "
                f"but resume artifact provides {query_learning_mode}."
            )
        runtime.reader.load_state_dict(state["reader_state"])
        if "fuser_state" in state:
            runtime.fuser.load_state_dict(state["fuser_state"])
        adaptable_parameters, trainable_module = _configure_stage_c_trainables(runtime, "q_only")
        optimizer = torch.optim.SGD(adaptable_parameters, lr=support_learning_rate)
        trainable_parameter_count = int(runtime.reader.queries.numel())
    else:
        query_learning_mode = "none"
        trainable_module = "none"
        trainable_parameter_count = 0
        optimizer = None

    cache_lookup = _build_example_caches(runtime, combined_examples)
    support_caches = [cache_lookup[str(example["id"])] for example in support_examples]
    calibration_caches = [cache_lookup[str(example["id"])] for example in calibration_examples]
    eval_caches = [cache_lookup[str(example["id"])] for example in eval_examples]
    support_ids = [str(example["id"]) for example in support_examples]
    repair_anchor_lookup = (
        _build_repair_anchor_lookup(
            runtime,
            cache_lookup,
            residual_scale=residual_scale,
            margin_threshold=repair_anchor_margin_threshold,
        )
        if support_objective == "choice_repair_ce_margin"
        else {}
    )

    profiler = ProfileTracker(
        output_dir=output_dir,
        device=str(config["runtime"].get("device", "cpu")),
        event_name="train",
    )
    curve_rows: list[dict[str, Any]] = []
    task_case_rows: list[dict[str, Any]] = []
    best_runtime = copy.deepcopy(runtime)
    best_row: dict[str, Any] | None = None
    latest_support_loss = 0.0
    latest_support_accuracy = 0.0
    latest_support_grad_norm = 0.0
    calibration_history: list[dict[str, Any]] = []

    for step in range(adapt_steps + 1):
        profiler.add_example()
        effective_residual_scale, calibration_row = _resolve_effective_residual_scale(
            runtime,
            calibration_caches or support_caches,
            decision_mode=decision_mode,
            memory_control=memory_control,
            candidate_memory_control=candidate_memory_control,
            cache_lookup=cache_lookup,
            configured_residual_scale=residual_scale,
            calibration_mode=residual_calibration_mode,
            calibration_alpha_grid=calibration_alpha_grid,
            candidate_residual_scale=candidate_residual_scale,
            candidate_delta_scale=candidate_delta_scale,
            candidate_delta_gate_tau=candidate_delta_gate_tau,
            repair_anchor_lookup=repair_anchor_lookup,
        )
        calibration_history.append(
            {
                "step": step,
                "decision_mode": decision_mode,
                "memory_control_mode": memory_control,
                "residual_calibration_mode": residual_calibration_mode,
                "configured_residual_scale": float(residual_scale),
                "effective_residual_scale": float(effective_residual_scale),
                "support_task_score": float(calibration_row["task_score"]),
                "support_task_proxy_score": float(calibration_row["task_proxy_score"]),
                "support_task_margin": float(calibration_row["task_margin"]),
            }
        )
        eval_metrics, case_rows = _aggregate_evaluation(
            runtime,
            eval_caches,
            decision_mode=decision_mode,
            memory_control=memory_control,
            candidate_memory_control=candidate_memory_control,
            cache_lookup=cache_lookup,
            residual_scale=effective_residual_scale,
            candidate_residual_scale=candidate_residual_scale,
            candidate_delta_scale=candidate_delta_scale,
            candidate_delta_gate_tau=candidate_delta_gate_tau,
            residual_calibration_mode=residual_calibration_mode,
            step=step,
            seed=seed,
            support_ids=support_ids,
            support_objective=support_objective,
            repair_anchor_lookup=repair_anchor_lookup,
        )
        for example_cache in eval_caches:
            profiler.add_tokens(runtime.backbone.count_tokens(_resolve_story_text(example_cache.example)))
            for candidate_text in example_cache.candidate_texts:
                profiler.add_tokens(runtime.backbone.count_tokens(candidate_text))
        task_case_rows.extend(case_rows)
        row = {
            "decision_mode": decision_mode,
            "memory_control_mode": memory_control,
            "candidate_branch_memory_control": candidate_memory_control,
            "choice_objective": support_objective,
            "residual_calibration_mode": residual_calibration_mode,
            "query_learning_mode": query_learning_mode,
            "adaptation_target": "q_only",
            "trainable_module": trainable_module,
            "trainable_parameter_count": trainable_parameter_count,
            "step": step,
            "effective_residual_scale": float(effective_residual_scale),
            "candidate_residual_scale": float(candidate_residual_scale),
            "candidate_delta_scale": float(candidate_delta_scale),
            "candidate_delta_gate_tau": float(candidate_delta_gate_tau),
            "support_examples": len(support_caches),
            "eval_examples": len(eval_caches),
            "support_loss": latest_support_loss,
            "support_accuracy": latest_support_accuracy,
            "support_grad_norm": latest_support_grad_norm,
            "support_calibration_task_score": float(calibration_row["task_score"]),
            "support_calibration_task_proxy_score": float(calibration_row["task_proxy_score"]),
            "support_calibration_task_margin": float(calibration_row["task_margin"]),
            "repair_active_examples": sum(int(bool(row["repair_active"])) for row in case_rows),
            "repair_active_rate": (
                sum(int(bool(row["repair_active"])) for row in case_rows) / max(1, len(case_rows))
            ),
            **eval_metrics,
        }
        curve_rows.append(row)
        if best_row is None or _curve_row_key(row) > _curve_row_key(best_row):
            best_row = dict(row)
            best_runtime = copy.deepcopy(runtime)
        if step == adapt_steps or decision_mode == "base_only":
            continue

        if not support_caches:
            raise ValueError("Stage C real pilot requires at least one support example for adaptive arms.")
        support_losses: list[torch.Tensor] = []
        support_accuracies: list[float] = []
        for example_cache in support_caches:
            profiler.add_tokens(runtime.backbone.count_tokens(_resolve_story_text(example_cache.example)))
            for candidate_text in example_cache.candidate_texts:
                profiler.add_tokens(runtime.backbone.count_tokens(candidate_text))
            if support_objective == "choice_ce_plus_margin":
                support_loss, support_accuracy = _choice_ce_plus_margin_support_loss(
                    runtime,
                    example_cache,
                    cache_lookup=cache_lookup,
                    residual_scale=effective_residual_scale,
                    candidate_residual_scale=candidate_residual_scale,
                    candidate_delta_scale=candidate_delta_scale,
                    candidate_delta_gate_tau=candidate_delta_gate_tau,
                    memory_control=memory_control,
                    candidate_memory_control=candidate_memory_control,
                    margin_value=choice_margin,
                    decision_mode=decision_mode,
                )
            elif support_objective == "choice_repair_ce_margin":
                support_loss, support_accuracy = _choice_repair_ce_margin_support_loss(
                    runtime,
                    example_cache,
                    repair_anchor=repair_anchor_lookup[str(example_cache.example["id"])],
                    cache_lookup=cache_lookup,
                    residual_scale=effective_residual_scale,
                    candidate_residual_scale=candidate_residual_scale,
                    candidate_delta_scale=candidate_delta_scale,
                    candidate_delta_gate_tau=candidate_delta_gate_tau,
                    memory_control=memory_control,
                    candidate_memory_control=candidate_memory_control,
                    margin_value=choice_margin,
                    decision_mode=decision_mode,
                )
            else:
                support_loss, support_accuracy = _continuation_retrieval_support_loss(
                    runtime,
                    example_cache,
                    support_caches=support_caches,
                    cache_lookup=cache_lookup,
                    memory_control=memory_control,
                    margin_value=choice_margin,
                )
            support_losses.append(support_loss)
            support_accuracies.append(support_accuracy)
        combined_support_loss = torch.stack(support_losses).mean()
        optimizer.zero_grad()
        combined_support_loss.backward()
        latest_support_grad_norm = float(runtime.reader.queries.grad.norm().item())
        optimizer.step()
        latest_support_loss = float(combined_support_loss.item())
        latest_support_accuracy = sum(support_accuracies) / len(support_accuracies)

    adapt_curve_path = output_dir / "adapt_curve.csv"
    with adapt_curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision_mode",
                "memory_control_mode",
                "candidate_branch_memory_control",
                "choice_objective",
                "residual_calibration_mode",
                "query_learning_mode",
                "adaptation_target",
                "trainable_module",
                "trainable_parameter_count",
                "step",
                "effective_residual_scale",
                "candidate_residual_scale",
                "candidate_delta_scale",
                "candidate_delta_gate_tau",
                "support_examples",
                "eval_examples",
                "support_loss",
                "support_accuracy",
                "support_grad_norm",
                "support_calibration_task_score",
                "support_calibration_task_proxy_score",
                "support_calibration_task_margin",
                "repair_active_examples",
                "repair_active_rate",
                "objective_accuracy",
                "task_score",
                "task_metric_name",
                "task_proxy_score",
                "task_proxy_name",
                "task_margin",
            ],
        )
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)

    task_case_dump_path = output_dir / "task_case_dump.jsonl"
    write_jsonl(task_case_dump_path, task_case_rows)
    torch.save(
        {
            "reader_state": best_runtime.reader.state_dict(),
            "fuser_state": best_runtime.fuser.state_dict(),
            "seed": seed,
            "decision_mode": decision_mode,
            "memory_control_mode": memory_control,
            "candidate_branch_memory_control": candidate_memory_control,
            "choice_objective": support_objective,
            "residual_calibration_mode": residual_calibration_mode,
            "trainable_module": trainable_module,
            "support_dataset_path": str(Path(support_dataset_path).resolve()),
            "calibration_dataset_path": str(Path(calibration_dataset_path).resolve()) if calibration_dataset_path else "",
            "eval_dataset_path": str(Path(config["task"]["dataset_path"]).resolve()),
        },
        output_dir / "queries_adapted.pt",
    )
    profile_metrics = profiler.finalize()
    zero_row = curve_rows[0]
    best_row = best_row or zero_row
    metrics = {
        "mode": "train",
        "training_stage": "stage_c_real_pilot",
        "decision_mode": decision_mode,
        "memory_control_mode": memory_control,
        "candidate_branch_memory_control": candidate_memory_control,
        "choice_objective": support_objective,
        "residual_calibration_mode": residual_calibration_mode,
        "query_learning_mode": query_learning_mode,
        "adaptation_target": "q_only",
        "trainable_module": trainable_module,
        "trainable_parameter_count": trainable_parameter_count,
        "support_dataset_path": str(Path(support_dataset_path).resolve()),
        "calibration_dataset_path": str(Path(calibration_dataset_path).resolve()) if calibration_dataset_path else "",
        "eval_dataset_path": str(Path(config["task"]["dataset_path"]).resolve()),
        "support_examples": len(support_caches),
        "calibration_examples": len(calibration_caches),
        "eval_examples": len(eval_caches),
        "pilot_split": str(config["task"].get("pilot_split", config["task"].get("smoke_subset", ""))),
        "stage_c_choice_residual_scale": residual_scale,
        "stage_c_candidate_residual_scale": candidate_residual_scale,
        "stage_c_candidate_delta_scale": candidate_delta_scale,
        "stage_c_candidate_delta_gate_tau": candidate_delta_gate_tau,
        "stage_c_repair_anchor_margin_threshold": repair_anchor_margin_threshold,
        "stage_c_residual_calibration_alpha_grid": calibration_alpha_grid,
        "stage_c_choice_margin": choice_margin,
        "pilot_support_learning_rate": support_learning_rate,
        "pilot_adapt_steps": adapt_steps,
        "zero_shot_effective_residual_scale": zero_row["effective_residual_scale"],
        "best_adapt_effective_residual_scale": best_row["effective_residual_scale"],
        "zero_shot_task_score": zero_row["task_score"],
        "zero_shot_task_proxy_score": zero_row["task_proxy_score"],
        "zero_shot_task_margin": zero_row["task_margin"],
        "best_adapt_task_score": best_row["task_score"],
        "best_adapt_task_proxy_score": best_row["task_proxy_score"],
        "best_adapt_task_margin": best_row["task_margin"],
        "best_adapt_step": best_row["step"],
        "task_metric_name": best_row["task_metric_name"],
        "task_proxy_name": best_row["task_proxy_name"],
        "adapt_curve_path": str(adapt_curve_path.resolve()),
        "task_case_dump_path": str(task_case_dump_path.resolve()),
        "task_case_dump_rows": len(task_case_rows),
        "repair_active_case_rows": sum(int(bool(row["repair_active"])) for row in task_case_rows),
        "adapted_queries_checkpoint": str((output_dir / "queries_adapted.pt").resolve()),
        "support_calibration_path": str((output_dir / "support_calibration.json").resolve()),
        **profile_metrics,
    }
    write_json(output_dir / "adapt_curve.json", {"rows": curve_rows})
    write_json(output_dir / "support_calibration.json", {"rows": calibration_history})
    write_json(output_dir / "metrics.json", metrics)
    return metrics
