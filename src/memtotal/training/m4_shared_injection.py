from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from memtotal.models import BackboneWrapper, MemoryWriter
from memtotal.tasks import load_task_dataset
from memtotal.training.m3 import _resolve_artifact_path
from memtotal.utils.io import write_json, write_jsonl
from memtotal.utils.profiling import ProfileTracker


@dataclass(frozen=True)
class SharedInjectionExampleCache:
    example: dict[str, Any]
    candidate_labels: list[str]
    candidate_texts: list[str]
    gold_index: int
    prompt_text: str


class LatentPrefixProjector(nn.Module):
    def __init__(self, hidden_size: int, prefix_tokens: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prefix_tokens = int(prefix_tokens)
        self.prefix_norm = nn.LayerNorm(hidden_size)
        self.prefix_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.prefix_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.prefix_proj.bias)

    def forward(self, memory_slots: torch.Tensor) -> torch.Tensor:
        if memory_slots.ndim != 3:
            raise ValueError("memory_slots must have shape [batch, slots, hidden_size].")
        if memory_slots.shape[1] != self.prefix_tokens:
            raise ValueError(
                f"LatentPrefixProjector expected {self.prefix_tokens} slots, got {memory_slots.shape[1]}."
            )
        return self.prefix_proj(self.prefix_norm(memory_slots))


def _load_task_dataset_with_path(config: dict[str, Any], dataset_path: str) -> list[dict[str, Any]]:
    config_copy = copy.deepcopy(config)
    config_copy["task"]["dataset_path"] = dataset_path
    return load_task_dataset(config_copy)


def _resolve_shared_injection_arm(config: dict[str, Any]) -> str:
    arm = str(config["runtime"].get("shared_injection_arm", "base_only"))
    if arm not in {"base_only", "teacher_text", "injected"}:
        raise ValueError(
            f"Unsupported runtime.shared_injection_arm={arm}. "
            "Expected one of base_only, teacher_text, injected."
        )
    return arm


def _resolve_writer_memory_control(config: dict[str, Any]) -> str:
    memory_control = str(config["runtime"].get("writer_memory_control", "real"))
    if memory_control not in {"real", "shuffled", "zero"}:
        raise ValueError(
            f"Unsupported runtime.writer_memory_control={memory_control}. "
            "Expected one of real, shuffled, zero."
        )
    return memory_control


def _resolve_pilot_arm_alias(config: dict[str, Any], *, arm: str, writer_memory_control: str) -> str:
    alias = config["runtime"].get("pilot_arm_alias")
    if alias is not None:
        return str(alias)
    if arm == "base_only":
        return "A"
    if arm == "teacher_text":
        return "T"
    return {
        "real": "I_real",
        "shuffled": "I_shuffle",
        "zero": "I_zero",
    }[writer_memory_control]


def _resolve_prompt_text(example: dict[str, Any]) -> str:
    prompt = str(example.get("segment", "")).strip()
    if prompt:
        return prompt
    claim = str(example.get("claim", "")).strip()
    evidence = str(example.get("evidence", "")).strip()
    choices = example.get("choices", [])
    labels_block = " | ".join(f"{choice['label']}: {choice['text']}" for choice in choices)
    return f"Claim: {claim} || Evidence: {evidence} || Labels: {labels_block} || Decide the correct label."


def _support_example_to_text(example: dict[str, Any]) -> str:
    claim = str(example.get("claim", "")).strip()
    evidence = str(example.get("evidence", "")).strip()
    label = str(example.get("label", "")).strip()
    return f"Claim: {claim} || Evidence: {evidence} || Label: {label}"


def _build_support_text_block(
    support_examples: list[dict[str, Any]],
    *,
    memory_control: str,
    example_lookup: dict[str, dict[str, Any]],
) -> str:
    if memory_control == "zero":
        return ""
    rows: list[dict[str, Any]] = []
    if memory_control == "real":
        rows = [dict(example) for example in support_examples]
    else:
        for example in support_examples:
            shuffled_id = str(example.get("shuffled_memory_example_id", "")).strip()
            if not shuffled_id:
                raise ValueError(
                    f"Support example {example['id']} is missing shuffled_memory_example_id for shuffled control."
                )
            shuffled_example = example_lookup[shuffled_id]
            rows.append(
                {
                    **example,
                    "evidence": shuffled_example.get("evidence", ""),
                    "label": shuffled_example.get("label", ""),
                }
            )
    return " || ".join(
        f"Support {index + 1}: {_support_example_to_text(row)}"
        for index, row in enumerate(rows)
    )


def _candidate_payload(example: dict[str, Any]) -> tuple[list[str], list[str], int]:
    choices = example.get("choices", [])
    if not choices:
        raise ValueError("Shared injection pilot currently supports only multiple-choice tasks.")
    candidate_labels = [str(choice["label"]) for choice in choices]
    candidate_texts = [str(choice["text"]) for choice in choices]
    gold_label = str(example["label"])
    try:
        gold_index = candidate_labels.index(gold_label)
    except ValueError as exc:
        raise ValueError(f"Gold label {gold_label!r} missing from candidate choices.") from exc
    return candidate_labels, candidate_texts, gold_index


def _build_example_caches(examples: list[dict[str, Any]]) -> list[SharedInjectionExampleCache]:
    caches: list[SharedInjectionExampleCache] = []
    for example in examples:
        candidate_labels, candidate_texts, gold_index = _candidate_payload(example)
        caches.append(
            SharedInjectionExampleCache(
                example=example,
                candidate_labels=candidate_labels,
                candidate_texts=candidate_texts,
                gold_index=gold_index,
                prompt_text=_resolve_prompt_text(example),
            )
        )
    return caches


def _strongest_competitor_index(scores: torch.Tensor, gold_index: int) -> int:
    competitor_indices = [index for index in range(scores.shape[0]) if index != gold_index]
    return max(competitor_indices, key=lambda index: float(scores[index].item()))


def _compute_margin(scores: torch.Tensor, gold_index: int) -> tuple[float, int]:
    competitor_index = _strongest_competitor_index(scores, gold_index)
    return float(scores[gold_index].item() - scores[competitor_index].item()), competitor_index


def _serialize_teacher_prompt(prompt_text: str, support_text_block: str) -> str:
    if not support_text_block:
        return prompt_text
    return f"Support bank: {support_text_block} || {prompt_text}"


class SharedInjectionPilotRuntime(nn.Module):
    def __init__(self, config: dict[str, Any], seed: int, *, arm: str, writer_memory_control: str) -> None:
        super().__init__()
        backbone_cfg = config["backbone"]
        runtime_device = str(config["runtime"].get("device", "cpu"))
        backbone_hidden_size = backbone_cfg.get("stub_hidden_size")
        self.backbone = BackboneWrapper(
            name=backbone_cfg["name"],
            load_mode=backbone_cfg["load_mode"],
            hidden_size=int(backbone_hidden_size) if backbone_hidden_size is not None else None,
            seed=seed,
            model_id=backbone_cfg.get("model_id"),
            device=runtime_device,
            dtype=str(backbone_cfg.get("dtype", "float32")),
            cache_dir=backbone_cfg.get("cache_dir"),
            max_new_tokens=int(backbone_cfg.get("max_new_tokens", 32)),
        )
        writer_cfg = config["method"]["writer"]
        self.writer = MemoryWriter(
            embed_dim=self.backbone.hidden_size,
            memory_slots=int(writer_cfg["memory_slots"]),
            arch=str(writer_cfg.get("arch", "mlp")),
            hidden_dim=writer_cfg.get("hidden_dim"),
            num_heads=int(writer_cfg.get("num_heads", 4)),
            transformer_layers=int(writer_cfg.get("transformer_layers", 1)),
            dropout=float(writer_cfg.get("dropout", 0.0)),
        )
        self.prefix_projector = LatentPrefixProjector(
            hidden_size=self.backbone.hidden_size,
            prefix_tokens=int(writer_cfg["memory_slots"]),
        )
        self.arm = arm
        self.writer_memory_control = writer_memory_control
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.to(self.backbone.device)

    def load_writer(self, resume: str | None) -> None:
        writer_path = _resolve_artifact_path(resume, "writer.ckpt")
        self.writer.load_from(writer_path, map_location="cpu")

    def set_writer_trainable(self, enabled: bool) -> None:
        for parameter in self.writer.parameters():
            parameter.requires_grad_(enabled)

    def build_prefix_embeddings(self, support_text_block: str) -> torch.Tensor:
        if self.writer_memory_control == "zero":
            return torch.zeros(
                1,
                self.prefix_projector.prefix_tokens,
                self.backbone.hidden_size,
                dtype=torch.float32,
                device=self.backbone.device,
            )
        support_state = self.backbone.summarize_texts([support_text_block])
        memory_slots = self.writer.write(support_state)
        return self.prefix_projector(memory_slots)

    def score_example(
        self,
        example_cache: SharedInjectionExampleCache,
        *,
        support_text_block: str,
        prefix_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.arm == "teacher_text":
            prompt = _serialize_teacher_prompt(example_cache.prompt_text, support_text_block)
            return self.backbone.score_continuations(prompt, example_cache.candidate_texts)
        return self.backbone.score_continuations(
            example_cache.prompt_text,
            example_cache.candidate_texts,
            prefix_embeddings=prefix_embeddings,
        )


def _choice_task_loss(scores: torch.Tensor, gold_index: int, *, margin_value: float) -> torch.Tensor:
    ce_loss = F.cross_entropy(
        scores.unsqueeze(0),
        torch.tensor([gold_index], dtype=torch.long, device=scores.device),
    )
    competitor_index = _strongest_competitor_index(scores, gold_index)
    gold_score = scores[gold_index].view(1)
    competitor_score = scores[competitor_index].view(1)
    target = torch.ones(1, dtype=torch.float32, device=scores.device)
    margin_loss = F.margin_ranking_loss(
        gold_score,
        competitor_score,
        target,
        margin=margin_value,
    )
    return ce_loss + margin_loss


def _prediction_row(
    *,
    arm_alias: str,
    arm: str,
    writer_memory_control: str,
    support_text_block: str,
    example_cache: SharedInjectionExampleCache,
    scores: torch.Tensor,
    prefix_embeddings: torch.Tensor | None,
) -> dict[str, Any]:
    predicted_index = int(torch.argmax(scores).item())
    margin, competitor_index = _compute_margin(scores, example_cache.gold_index)
    return {
        "example_id": str(example_cache.example["id"]),
        "arm_alias": arm_alias,
        "arm": arm,
        "writer_memory_control": writer_memory_control,
        "support_text_block_chars": len(support_text_block),
        "prefix_tokens": 0 if prefix_embeddings is None else int(prefix_embeddings.shape[1]),
        "prefix_l2": 0.0 if prefix_embeddings is None else float(prefix_embeddings.norm().item()),
        "prompt_text": example_cache.prompt_text,
        "predicted_label": example_cache.candidate_labels[predicted_index],
        "gold_label": example_cache.candidate_labels[example_cache.gold_index],
        "predicted_correct": bool(predicted_index == example_cache.gold_index),
        "task_score": float(predicted_index == example_cache.gold_index),
        "final_margin": margin,
        "competitor_label": example_cache.candidate_labels[competitor_index],
        "final_choice_scores": [float(value) for value in scores.tolist()],
        "candidate_labels": list(example_cache.candidate_labels),
        "candidate_texts": list(example_cache.candidate_texts),
    }


def run_shared_injection_pilot(
    *,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    resume: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    arm = _resolve_shared_injection_arm(config)
    writer_memory_control = _resolve_writer_memory_control(config)
    arm_alias = _resolve_pilot_arm_alias(config, arm=arm, writer_memory_control=writer_memory_control)
    support_dataset_path = str(config["task"]["support_dataset_path"])
    support_limit = max(0, int(config["runtime"].get("pilot_support_examples", 8)))
    learning_rate = float(config["runtime"].get("pilot_learning_rate", 0.05))
    train_steps = int(config["runtime"].get("pilot_train_steps", 12))
    projector_warmup_steps = int(config["runtime"].get("pilot_projector_warmup_steps", 2))
    choice_margin = float(config["runtime"].get("stage_c_choice_margin", 0.1))
    if arm in {"base_only", "teacher_text"} or writer_memory_control == "zero":
        train_steps = 0
        projector_warmup_steps = 0
    if dry_run:
        train_steps = min(train_steps, 2)

    support_examples = _build_example_caches(_load_task_dataset_with_path(config, support_dataset_path))
    eval_examples = _build_example_caches(load_task_dataset(config))
    if support_limit:
        support_examples = support_examples[: min(support_limit, len(support_examples))]
    if dry_run:
        eval_examples = eval_examples[: min(4, len(eval_examples))]

    example_lookup = {
        str(cache.example["id"]): cache.example
        for cache in [*support_examples, *eval_examples]
    }
    support_text_block = _build_support_text_block(
        [cache.example for cache in support_examples],
        memory_control=writer_memory_control if arm == "injected" else "real",
        example_lookup=example_lookup,
    )
    teacher_support_text_block = _build_support_text_block(
        [cache.example for cache in support_examples],
        memory_control="real",
        example_lookup=example_lookup,
    )

    runtime = SharedInjectionPilotRuntime(
        config=config,
        seed=seed,
        arm=arm,
        writer_memory_control=writer_memory_control,
    )
    if arm == "injected":
        runtime.load_writer(resume)

    profiler = ProfileTracker(
        output_dir=output_dir,
        device=str(config["runtime"].get("device", "cpu")),
        event_name="train",
    )
    train_events: list[dict[str, Any]] = []
    if arm == "injected" and writer_memory_control != "zero":
        runtime.writer.train()
        runtime.prefix_projector.train()
        runtime.set_writer_trainable(False)
        warmup_optimizer = torch.optim.Adam(runtime.prefix_projector.parameters(), lr=learning_rate)
        joint_optimizer = torch.optim.Adam(
            [*runtime.writer.parameters(), *runtime.prefix_projector.parameters()],
            lr=learning_rate,
        )
        for step in range(train_steps):
            projector_warmup = step < projector_warmup_steps
            if projector_warmup:
                runtime.set_writer_trainable(False)
                optimizer = warmup_optimizer
            else:
                runtime.set_writer_trainable(True)
                optimizer = joint_optimizer
            support_example = support_examples[step % len(support_examples)]
            optimizer.zero_grad(set_to_none=True)
            prefix_embeddings = runtime.build_prefix_embeddings(support_text_block)
            scores = runtime.score_example(
                support_example,
                support_text_block=teacher_support_text_block,
                prefix_embeddings=prefix_embeddings,
            )
            loss = _choice_task_loss(scores, support_example.gold_index, margin_value=choice_margin)
            loss.backward()
            optimizer.step()
            profiler.add_example()
            profiler.add_tokens(runtime.backbone.count_tokens(support_example.prompt_text))
            for candidate_text in support_example.candidate_texts:
                profiler.add_tokens(runtime.backbone.count_tokens(candidate_text))
            train_events.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "support_example_id": str(support_example.example["id"]),
                    "projector_warmup": projector_warmup,
                }
            )
        runtime.writer.eval()
        runtime.prefix_projector.eval()
        runtime.set_writer_trainable(True)

    prefix_embeddings = None
    if arm == "injected":
        prefix_embeddings = runtime.build_prefix_embeddings(support_text_block)
    prefix_embeddings_cpu = None if prefix_embeddings is None else prefix_embeddings.detach().cpu()

    case_rows: list[dict[str, Any]] = []
    active_support_text_block = teacher_support_text_block if arm == "teacher_text" else support_text_block
    for example_cache in eval_examples:
        profiler.add_example()
        profiler.add_tokens(runtime.backbone.count_tokens(example_cache.prompt_text))
        for candidate_text in example_cache.candidate_texts:
            profiler.add_tokens(runtime.backbone.count_tokens(candidate_text))
        scores = runtime.score_example(
            example_cache,
            support_text_block=teacher_support_text_block,
            prefix_embeddings=prefix_embeddings,
        ).detach().to(dtype=torch.float32).cpu()
        case_rows.append(
            _prediction_row(
                arm_alias=arm_alias,
                arm=arm,
                writer_memory_control=writer_memory_control,
                support_text_block=active_support_text_block,
                example_cache=example_cache,
                scores=scores,
                prefix_embeddings=prefix_embeddings_cpu,
            )
        )

    profile_metrics = profiler.finalize()
    task_score = sum(float(row["task_score"]) for row in case_rows) / max(1, len(case_rows))
    mean_margin = sum(float(row["final_margin"]) for row in case_rows) / max(1, len(case_rows))
    checkpoint_path = output_dir / "checkpoint.pt"
    if arm == "injected":
        torch.save(
            {
                "writer_state": runtime.writer.state_dict(),
                "prefix_projector_state": runtime.prefix_projector.state_dict(),
                "seed": seed,
                "arm_alias": arm_alias,
                "shared_injection_arm": arm,
                "writer_memory_control": writer_memory_control,
                "support_dataset_path": str(Path(support_dataset_path).resolve()),
                "support_text_block": support_text_block,
            },
            checkpoint_path,
        )

    task_case_dump_path = output_dir / "task_case_dump.jsonl"
    write_jsonl(task_case_dump_path, case_rows)
    metrics = {
        "mode": "train",
        "training_stage": "shared_injection_pilot",
        "pilot_arm_alias": arm_alias,
        "shared_injection_arm": arm,
        "writer_memory_control": writer_memory_control,
        "task_name": config["task"]["name"],
        "benchmark_id": config["task"].get("benchmark_id"),
        "pilot_split": str(config["task"].get("pilot_split", config["task"].get("smoke_subset", ""))),
        "support_dataset_path": str(Path(support_dataset_path).resolve()),
        "eval_dataset_path": str(Path(config["task"]["dataset_path"]).resolve()),
        "support_examples": len(support_examples),
        "eval_examples": len(eval_examples),
        "teacher_text_block_chars": len(teacher_support_text_block),
        "support_text_block_chars": len(support_text_block),
        "task_metric_name": str(config["task"].get("metric_name", "accuracy")),
        "best_adapt_task_score": float(task_score),
        "best_adapt_task_margin": float(mean_margin),
        "best_adapt_step": 0,
        "task_case_dump_rows": len(case_rows),
        "task_case_dump_path": str(task_case_dump_path.resolve()),
        "pilot_train_steps": train_steps,
        "pilot_learning_rate": learning_rate,
        "pilot_projector_warmup_steps": projector_warmup_steps,
        "stage_c_choice_margin": choice_margin,
        "train_final_loss": train_events[-1]["loss"] if train_events else None,
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path.exists() else "",
        **profile_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "train_events.json", {"events": train_events})
    return metrics
