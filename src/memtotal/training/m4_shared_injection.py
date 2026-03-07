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

FEVER_LABEL_ORDER = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
FEVER_PROMPT_VARIANTS = (
    "inline_short_labels",
    "answer_slot_labels",
    "verbalized_decisions",
)
FEVER_SUPPORT_SERIALIZATION_VARIANTS = (
    "flat_raw8",
    "example_blocks_raw8",
    "triad_curated6",
)
FEVER_TRIAD_NEI_EVIDENCE = "insufficient evidence available"


@dataclass(frozen=True)
class SharedInjectionExampleCache:
    example: dict[str, Any]
    candidate_labels: list[str]
    candidate_texts: list[str]
    gold_index: int
    prompt_text: str
    prompt_variant: str


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


def _resolve_prompt_variant(config: dict[str, Any]) -> str:
    prompt_variant = str(config["runtime"].get("pilot_prompt_variant", "inline_short_labels"))
    if prompt_variant not in FEVER_PROMPT_VARIANTS:
        raise ValueError(
            f"Unsupported runtime.pilot_prompt_variant={prompt_variant}. "
            f"Expected one of {', '.join(FEVER_PROMPT_VARIANTS)}."
        )
    return prompt_variant


def _resolve_support_serialization_variant(config: dict[str, Any]) -> str:
    support_variant = str(config["runtime"].get("pilot_support_serialization", "flat_raw8"))
    if support_variant not in FEVER_SUPPORT_SERIALIZATION_VARIANTS:
        raise ValueError(
            f"Unsupported runtime.pilot_support_serialization={support_variant}. "
            f"Expected one of {', '.join(FEVER_SUPPORT_SERIALIZATION_VARIANTS)}."
        )
    return support_variant


def _resolve_snapshot_steps(config: dict[str, Any], *, train_steps: int, warmup_steps: int) -> list[int]:
    explicit = config["runtime"].get("pilot_snapshot_steps")
    if explicit is not None:
        values = [int(step) for step in explicit]
    else:
        values = [0, warmup_steps, max(warmup_steps, train_steps // 2), train_steps]
    selected = sorted({step for step in values if 0 <= step <= train_steps})
    if 0 not in selected:
        selected.insert(0, 0)
    if train_steps not in selected:
        selected.append(train_steps)
    return selected


def _classification_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(row["gold_label"]) for row in rows]
    unique_labels = sorted(set(labels))
    total = max(1, len(rows))
    accuracy = sum(int(bool(row["predicted_correct"])) for row in rows) / total
    predicted_counts: dict[str, int] = {label: 0 for label in unique_labels}
    gold_counts: dict[str, int] = {label: 0 for label in unique_labels}
    tp_counts: dict[str, int] = {label: 0 for label in unique_labels}
    for row in rows:
        gold = str(row["gold_label"])
        predicted = str(row["predicted_label"])
        gold_counts[gold] = gold_counts.get(gold, 0) + 1
        predicted_counts[predicted] = predicted_counts.get(predicted, 0) + 1
        if gold == predicted:
            tp_counts[gold] = tp_counts.get(gold, 0) + 1
    f1_values: list[float] = []
    recall_by_class: dict[str, float] = {}
    for label in unique_labels:
        tp = tp_counts.get(label, 0)
        fp = predicted_counts.get(label, 0) - tp
        fn = gold_counts.get(label, 0) - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
        recall_by_class[label] = recall
    dominant_label_fraction = max(predicted_counts.values(), default=0) / total
    mean_margin = sum(float(row["final_margin"]) for row in rows) / total
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(sum(f1_values) / max(1, len(f1_values))),
        "dominant_label_fraction": float(dominant_label_fraction),
        "label_recall_by_class": recall_by_class,
        "mean_margin": float(mean_margin),
    }


def _write_snapshot_eval(
    *,
    snapshot_dir: Path,
    step: int,
    arm_alias: str,
    arm: str,
    writer_memory_control: str,
    prompt_variant: str,
    support_serialization_variant: str,
    support_text_block: str,
    case_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    task_case_dump_path = snapshot_dir / "task_case_dump.jsonl"
    write_jsonl(task_case_dump_path, case_rows)
    metrics = _classification_metrics_from_rows(case_rows)
    payload = {
        "training_stage": "shared_injection_snapshot_eval",
        "pilot_arm_alias": arm_alias,
        "shared_injection_arm": arm,
        "writer_memory_control": writer_memory_control,
        "prompt_variant": prompt_variant,
        "support_serialization_variant": support_serialization_variant,
        "support_text_block_chars": len(support_text_block),
        "step": int(step),
        "task_case_dump_rows": len(case_rows),
        "task_case_dump_path": str(task_case_dump_path.resolve()),
        "best_adapt_task_score": float(metrics["accuracy"]),
        "best_adapt_macro_f1": float(metrics["macro_f1"]),
        "best_adapt_task_margin": float(metrics["mean_margin"]),
        "dominant_label_fraction": float(metrics["dominant_label_fraction"]),
        "label_recall_by_class": metrics["label_recall_by_class"],
    }
    write_json(snapshot_dir / "metrics.json", payload)
    return payload


def _fever_candidate_texts(prompt_variant: str) -> tuple[list[str], list[str]]:
    labels = list(FEVER_LABEL_ORDER)
    if prompt_variant == "inline_short_labels":
        return labels, ["Supports", "Refutes", "Not enough info"]
    if prompt_variant == "answer_slot_labels":
        return labels, [" SUPPORTS", " REFUTES", " NOT_ENOUGH_INFO"]
    if prompt_variant == "verbalized_decisions":
        return labels, [
            " The evidence supports the claim.",
            " The evidence refutes the claim.",
            " There is not enough information in the evidence to verify the claim.",
        ]
    raise ValueError(f"Unsupported prompt_variant={prompt_variant}.")


def _resolve_prompt_text(example: dict[str, Any], *, prompt_variant: str) -> str:
    claim = str(example.get("claim", "")).strip()
    evidence = str(example.get("evidence", "")).strip()
    if prompt_variant == "inline_short_labels":
        return (
            f"Claim: {claim} || Evidence: {evidence} || "
            "Labels: SUPPORTS: Supports | REFUTES: Refutes | NOT_ENOUGH_INFO: Not enough info || "
            "Decide the correct label."
        )
    if prompt_variant == "answer_slot_labels":
        return (
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n"
            "Question: Does the evidence support, refute, or not give enough info for the claim?\n"
            "Answer:"
        )
    if prompt_variant == "verbalized_decisions":
        return f"Claim: {claim}\nEvidence: {evidence}\nDecision:"
    raise ValueError(f"Unsupported prompt_variant={prompt_variant}.")


def _candidate_payload(example: dict[str, Any], *, prompt_variant: str) -> tuple[list[str], list[str], int]:
    gold_label = str(example["label"])
    candidate_labels, candidate_texts = _fever_candidate_texts(prompt_variant)
    try:
        gold_index = candidate_labels.index(gold_label)
    except ValueError as exc:
        raise ValueError(f"Gold label {gold_label!r} missing from FEVER labels.") from exc
    return candidate_labels, candidate_texts, gold_index


def _build_example_caches(
    examples: list[dict[str, Any]],
    *,
    prompt_variant: str,
) -> list[SharedInjectionExampleCache]:
    caches: list[SharedInjectionExampleCache] = []
    for example in examples:
        candidate_labels, candidate_texts, gold_index = _candidate_payload(
            example,
            prompt_variant=prompt_variant,
        )
        caches.append(
            SharedInjectionExampleCache(
                example=example,
                candidate_labels=candidate_labels,
                candidate_texts=candidate_texts,
                gold_index=gold_index,
                prompt_text=_resolve_prompt_text(example, prompt_variant=prompt_variant),
                prompt_variant=prompt_variant,
            )
        )
    return caches


def _serialize_support_row(
    row: dict[str, Any],
    *,
    support_index: int,
    support_serialization_variant: str,
) -> str:
    claim = str(row.get("claim", "")).strip()
    evidence = str(row.get("evidence", "")).strip()
    label = str(row.get("label", "")).strip()
    if support_serialization_variant == "flat_raw8":
        return f"Support {support_index}: Claim: {claim} || Evidence: {evidence} || Label: {label}"
    if support_serialization_variant in {"example_blocks_raw8", "triad_curated6"}:
        return (
            f"Example {support_index}\n"
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n"
            f"Answer: {label}"
        )
    raise ValueError(
        f"Unsupported support_serialization_variant={support_serialization_variant}."
    )


def _select_support_rows_for_variant(
    support_examples: list[dict[str, Any]],
    *,
    support_serialization_variant: str,
) -> list[dict[str, Any]]:
    if support_serialization_variant in {"flat_raw8", "example_blocks_raw8"}:
        return [dict(example) for example in support_examples]
    if support_serialization_variant != "triad_curated6":
        raise ValueError(
            f"Unsupported support_serialization_variant={support_serialization_variant}."
        )
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in FEVER_LABEL_ORDER}
    for example in support_examples:
        label = str(example.get("label", "")).strip()
        if label in by_label:
            by_label[label].append(dict(example))
    selected: list[dict[str, Any]] = []
    for label in FEVER_LABEL_ORDER:
        rows = by_label[label][:2]
        if len(rows) < 2:
            raise ValueError(
                f"triad_curated6 requires at least 2 rows for label {label}, got {len(rows)}."
            )
        for row in rows:
            if label == "NOT_ENOUGH_INFO":
                row["evidence"] = FEVER_TRIAD_NEI_EVIDENCE
            selected.append(row)
    return selected


def _build_support_text_block(
    support_examples: list[dict[str, Any]],
    *,
    memory_control: str,
    example_lookup: dict[str, dict[str, Any]],
    support_serialization_variant: str,
) -> str:
    if memory_control == "zero":
        return ""
    selected_rows = _select_support_rows_for_variant(
        support_examples,
        support_serialization_variant=support_serialization_variant,
    )
    rows: list[dict[str, Any]] = []
    if memory_control == "real":
        rows = [dict(example) for example in selected_rows]
    else:
        for example in selected_rows:
            shuffled_id = str(example.get("shuffled_memory_example_id", "")).strip()
            if not shuffled_id:
                raise ValueError(
                    f"Support example {example['id']} is missing shuffled_memory_example_id for shuffled control."
                )
            shuffled_example = dict(example_lookup[shuffled_id])
            if support_serialization_variant == "triad_curated6" and str(shuffled_example.get("label", "")) == "NOT_ENOUGH_INFO":
                shuffled_example["evidence"] = FEVER_TRIAD_NEI_EVIDENCE
            rows.append(shuffled_example)
    return "\n\n".join(
        _serialize_support_row(
            row,
            support_index=index + 1,
            support_serialization_variant=support_serialization_variant,
        )
        for index, row in enumerate(rows)
    )


def _serialize_teacher_prompt(prompt_text: str, support_text_block: str) -> str:
    if not support_text_block:
        return prompt_text
    return f"Support bank:\n{support_text_block}\n\n{prompt_text}"


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


def _strongest_competitor_index(scores: torch.Tensor, gold_index: int) -> int:
    competitor_indices = [index for index in range(scores.shape[0]) if index != gold_index]
    return max(competitor_indices, key=lambda index: float(scores[index].item()))


def _compute_margin(scores: torch.Tensor, gold_index: int) -> tuple[float, int]:
    competitor_index = _strongest_competitor_index(scores, gold_index)
    return float(scores[gold_index].item() - scores[competitor_index].item()), competitor_index


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
    prompt_variant: str,
    support_serialization_variant: str,
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
        "prompt_variant": prompt_variant,
        "support_serialization_variant": support_serialization_variant,
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


def _evaluate_examples(
    *,
    runtime: SharedInjectionPilotRuntime,
    eval_examples: list[SharedInjectionExampleCache],
    arm_alias: str,
    arm: str,
    writer_memory_control: str,
    prompt_variant: str,
    support_serialization_variant: str,
    support_text_block: str,
    teacher_support_text_block: str,
    prefix_embeddings: torch.Tensor | None,
    profiler: ProfileTracker | None,
) -> list[dict[str, Any]]:
    prefix_embeddings_cpu = None if prefix_embeddings is None else prefix_embeddings.detach().cpu()
    active_support_text_block = teacher_support_text_block if arm == "teacher_text" else support_text_block
    case_rows: list[dict[str, Any]] = []
    for example_cache in eval_examples:
        if profiler is not None:
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
                prompt_variant=prompt_variant,
                support_serialization_variant=support_serialization_variant,
                support_text_block=active_support_text_block,
                example_cache=example_cache,
                scores=scores,
                prefix_embeddings=prefix_embeddings_cpu,
            )
        )
    return case_rows


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
    prompt_variant = _resolve_prompt_variant(config)
    support_serialization_variant = _resolve_support_serialization_variant(config)
    support_dataset_path = str(config["task"]["support_dataset_path"])
    support_limit = max(0, int(config["runtime"].get("pilot_support_examples", 8)))
    train_steps = int(config["runtime"].get("pilot_train_steps", 64))
    projector_warmup_steps = int(config["runtime"].get("pilot_projector_warmup_steps", 16))
    writer_learning_rate = float(config["runtime"].get("pilot_writer_learning_rate", 0.002))
    projector_learning_rate = float(config["runtime"].get("pilot_projector_learning_rate", 0.01))
    choice_margin = float(config["runtime"].get("stage_c_choice_margin", 0.1))
    if arm in {"base_only", "teacher_text"} or writer_memory_control == "zero":
        train_steps = 0
        projector_warmup_steps = 0
    if dry_run:
        train_steps = min(train_steps, 2)
        projector_warmup_steps = min(projector_warmup_steps, train_steps)

    support_examples = _load_task_dataset_with_path(config, support_dataset_path)
    eval_examples = load_task_dataset(config)
    if support_limit:
        support_examples = support_examples[: min(support_limit, len(support_examples))]
    if dry_run:
        eval_examples = eval_examples[: min(12, len(eval_examples))]

    eval_caches = _build_example_caches(eval_examples, prompt_variant=prompt_variant)
    support_caches = _build_example_caches(support_examples, prompt_variant=prompt_variant)
    example_lookup = {
        str(cache.example["id"]): cache.example
        for cache in [*support_caches, *eval_caches]
    }
    support_text_block = _build_support_text_block(
        [cache.example for cache in support_caches],
        memory_control=writer_memory_control if arm == "injected" else "real",
        example_lookup=example_lookup,
        support_serialization_variant=support_serialization_variant,
    )
    teacher_support_text_block = _build_support_text_block(
        [cache.example for cache in support_caches],
        memory_control="real",
        example_lookup=example_lookup,
        support_serialization_variant=support_serialization_variant,
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
    snapshot_steps = _resolve_snapshot_steps(
        config,
        train_steps=train_steps,
        warmup_steps=projector_warmup_steps,
    )
    snapshot_metrics: list[dict[str, Any]] = []

    def evaluate_snapshot(step: int) -> None:
        prefix_embeddings = None
        if arm == "injected":
            prefix_embeddings = runtime.build_prefix_embeddings(support_text_block)
        snapshot_rows = _evaluate_examples(
            runtime=runtime,
            eval_examples=eval_caches,
            arm_alias=arm_alias,
            arm=arm,
            writer_memory_control=writer_memory_control,
            prompt_variant=prompt_variant,
            support_serialization_variant=support_serialization_variant,
            support_text_block=support_text_block,
            teacher_support_text_block=teacher_support_text_block,
            prefix_embeddings=prefix_embeddings,
            profiler=None,
        )
        metrics = _classification_metrics_from_rows(snapshot_rows)
        snapshot_dir = output_dir / "snapshot_evals" / f"step_{step:04d}"
        snapshot_payload = _write_snapshot_eval(
            snapshot_dir=snapshot_dir,
            step=step,
            arm_alias=arm_alias,
            arm=arm,
            writer_memory_control=writer_memory_control,
            prompt_variant=prompt_variant,
            support_serialization_variant=support_serialization_variant,
            support_text_block=support_text_block,
            case_rows=snapshot_rows,
        )
        snapshot_metrics.append(
            {
                "step": step,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mean_margin": metrics["mean_margin"],
                "dominant_label_fraction": metrics["dominant_label_fraction"],
                "snapshot_dir": str(snapshot_dir.resolve()),
                "task_case_dump_path": snapshot_payload["task_case_dump_path"],
            }
        )

    if 0 in snapshot_steps:
        evaluate_snapshot(0)

    if arm == "injected" and writer_memory_control != "zero":
        runtime.writer.train()
        runtime.prefix_projector.train()
        runtime.set_writer_trainable(False)
        optimizer = torch.optim.AdamW(
            [
                {"params": list(runtime.prefix_projector.parameters()), "lr": projector_learning_rate},
                {"params": list(runtime.writer.parameters()), "lr": writer_learning_rate},
            ]
        )
        for step in range(train_steps):
            writer_frozen = step < projector_warmup_steps
            runtime.set_writer_trainable(not writer_frozen)
            support_example = support_caches[step % len(support_caches)]
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
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "support_example_id": str(support_example.example["id"]),
                    "writer_frozen": writer_frozen,
                }
            )
            if (step + 1) in snapshot_steps and (step + 1) != 0:
                runtime.writer.eval()
                runtime.prefix_projector.eval()
                evaluate_snapshot(step + 1)
                runtime.writer.train()
                runtime.prefix_projector.train()
        runtime.writer.eval()
        runtime.prefix_projector.eval()
        runtime.set_writer_trainable(True)

    prefix_embeddings = None
    if arm == "injected":
        prefix_embeddings = runtime.build_prefix_embeddings(support_text_block)
    case_rows = _evaluate_examples(
        runtime=runtime,
        eval_examples=eval_caches,
        arm_alias=arm_alias,
        arm=arm,
        writer_memory_control=writer_memory_control,
        prompt_variant=prompt_variant,
        support_serialization_variant=support_serialization_variant,
        support_text_block=support_text_block,
        teacher_support_text_block=teacher_support_text_block,
        prefix_embeddings=prefix_embeddings,
        profiler=profiler,
    )

    profile_metrics = profiler.finalize()
    class_metrics = _classification_metrics_from_rows(case_rows)
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
                "prompt_variant": prompt_variant,
                "support_serialization_variant": support_serialization_variant,
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
        "prompt_variant": prompt_variant,
        "support_serialization_variant": support_serialization_variant,
        "task_name": config["task"]["name"],
        "benchmark_id": config["task"].get("benchmark_id"),
        "pilot_split": str(config["task"].get("pilot_split", config["task"].get("smoke_subset", ""))),
        "support_dataset_path": str(Path(support_dataset_path).resolve()),
        "eval_dataset_path": str(Path(config["task"]["dataset_path"]).resolve()),
        "support_examples": len(support_caches),
        "eval_examples": len(eval_caches),
        "teacher_text_block_chars": len(teacher_support_text_block),
        "support_text_block_chars": len(support_text_block),
        "task_metric_name": str(config["task"].get("metric_name", "accuracy")),
        "best_adapt_task_score": float(class_metrics["accuracy"]),
        "best_adapt_macro_f1": float(class_metrics["macro_f1"]),
        "best_adapt_task_margin": float(class_metrics["mean_margin"]),
        "dominant_label_fraction": float(class_metrics["dominant_label_fraction"]),
        "label_recall_by_class": class_metrics["label_recall_by_class"],
        "best_adapt_step": 0,
        "task_case_dump_rows": len(case_rows),
        "task_case_dump_path": str(task_case_dump_path.resolve()),
        "pilot_train_steps": train_steps,
        "pilot_writer_learning_rate": writer_learning_rate,
        "pilot_projector_learning_rate": projector_learning_rate,
        "pilot_projector_warmup_steps": projector_warmup_steps,
        "pilot_snapshot_steps": snapshot_steps,
        "snapshot_metrics": snapshot_metrics,
        "stage_c_choice_margin": choice_margin,
        "train_final_loss": train_events[-1]["loss"] if train_events else None,
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path.exists() else "",
        **profile_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "train_events.json", {"events": train_events, "snapshots": snapshot_metrics})
    return metrics
