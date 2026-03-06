from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


MEMORYAGENTBENCH_DATASET = "ai-hyz/MemoryAgentBench"
MEMORYAGENTBENCH_SMOKE_CONTEXT_TOKENS = 512


@dataclass(frozen=True)
class MemoryAgentBenchCapability:
    split_name: str
    short_name: str
    display_name: str
    source: str
    primary_metric: str


CAPABILITY_SPECS = (
    MemoryAgentBenchCapability(
        split_name="Accurate_Retrieval",
        short_name="AR",
        display_name="Accurate Retrieval",
        source="ruler_qa1_197K",
        primary_metric="exact_match",
    ),
    MemoryAgentBenchCapability(
        split_name="Test_Time_Learning",
        short_name="TTL",
        display_name="Test-Time Learning",
        source="icl_trec_coarse_6600shot_balance",
        primary_metric="exact_match",
    ),
    MemoryAgentBenchCapability(
        split_name="Long_Range_Understanding",
        short_name="LRU",
        display_name="Long-Range Understanding",
        source="infbench_sum_eng_shots2",
        primary_metric="rougeLsum_f1",
    ),
    MemoryAgentBenchCapability(
        split_name="Conflict_Resolution",
        short_name="CR",
        display_name="Conflict Resolution",
        source="factconsolidation_mh_6k",
        primary_metric="exact_match",
    ),
)


def _tokenize_context(text: str) -> list[str]:
    return [token for token in text.replace("\n", " ").split(" ") if token]


def truncate_memoryagentbench_context(
    context: str,
    *,
    max_context_tokens: int = MEMORYAGENTBENCH_SMOKE_CONTEXT_TOKENS,
) -> tuple[str, int, bool]:
    tokens = _tokenize_context(context)
    if len(tokens) <= max_context_tokens:
        return context.strip(), len(tokens), False
    truncated = " ".join(tokens[:max_context_tokens]).strip()
    return truncated, len(tokens), True


def _flatten_text_answers(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_text_answers(item))
        return flattened
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _allocate_examples(total_examples: int, num_buckets: int) -> list[int]:
    if total_examples <= 0:
        return [0] * num_buckets
    base = total_examples // num_buckets
    remainder = total_examples % num_buckets
    return [base + (1 if index < remainder else 0) for index in range(num_buckets)]


def select_memoryagentbench_source_row(
    rows: list[dict[str, Any]],
    capability: MemoryAgentBenchCapability,
) -> dict[str, Any]:
    for row in rows:
        metadata = row.get("metadata") or {}
        if metadata.get("source") == capability.source:
            return row
    raise ValueError(
        f"MemoryAgentBench split {capability.split_name} is missing source '{capability.source}'."
    )


def build_memoryagentbench_smoke_examples(
    row: dict[str, Any],
    *,
    capability: MemoryAgentBenchCapability,
    take_questions: int,
    max_context_tokens: int = MEMORYAGENTBENCH_SMOKE_CONTEXT_TOKENS,
) -> list[dict[str, Any]]:
    context = str(row.get("context", "")).strip()
    truncated_context, context_tokens_total, was_truncated = truncate_memoryagentbench_context(
        context,
        max_context_tokens=max_context_tokens,
    )
    metadata = row.get("metadata") or {}
    keypoints = [text for text in _flatten_text_answers(metadata.get("keypoints")) if text]
    qa_pair_ids = [text for text in _flatten_text_answers(metadata.get("qa_pair_ids")) if text]
    question_types = [text for text in _flatten_text_answers(metadata.get("question_types")) if text]
    questions = [str(question).strip() for question in row.get("questions", [])]
    answers = row.get("answers", [])

    examples: list[dict[str, Any]] = []
    for question_index in range(min(take_questions, len(questions), len(answers))):
        aliases = _flatten_text_answers(answers[question_index])
        if not aliases:
            continue
        example = {
            "id": f"memoryagentbench-{capability.short_name.lower()}-{question_index:03d}",
            "question": questions[question_index],
            "answer": aliases[0],
            "aliases": aliases,
            "context": truncated_context,
            "capability": capability.short_name,
            "capability_name": capability.display_name,
            "memoryagent_source": capability.source,
            "question_index": question_index,
            "capability_metric_name": capability.primary_metric,
            "context_token_budget": max_context_tokens,
            "context_tokens_total": context_tokens_total,
            "context_tokens_used": min(context_tokens_total, max_context_tokens),
            "context_was_truncated": was_truncated,
            "full_context_chars": len(context),
        }
        if keypoints:
            example["keypoints"] = keypoints
        if question_index < len(qa_pair_ids):
            example["qa_pair_id"] = qa_pair_ids[question_index]
        if question_index < len(question_types):
            example["question_type"] = question_types[question_index]
        examples.append(example)
    return examples


def materialize_memoryagentbench_smoke(
    *,
    max_examples: int,
    seed: int,
    max_context_tokens: int = MEMORYAGENTBENCH_SMOKE_CONTEXT_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del seed
    allocations = _allocate_examples(max_examples, len(CAPABILITY_SPECS))
    canonical_rows: list[dict[str, Any]] = []
    capability_manifest: dict[str, dict[str, Any]] = {}

    for capability, take_questions in zip(CAPABILITY_SPECS, allocations, strict=True):
        if take_questions <= 0:
            continue
        dataset = load_dataset(MEMORYAGENTBENCH_DATASET, split=capability.split_name)
        selected_row = select_memoryagentbench_source_row([dataset[index] for index in range(len(dataset))], capability)
        examples = build_memoryagentbench_smoke_examples(
            selected_row,
            capability=capability,
            take_questions=take_questions,
            max_context_tokens=max_context_tokens,
        )
        capability_manifest[capability.short_name] = {
            "split_name": capability.split_name,
            "display_name": capability.display_name,
            "source": capability.source,
            "primary_metric": capability.primary_metric,
            "questions_materialized": len(examples),
        }
        canonical_rows.extend(examples)

    manifest = {
        "selected_capabilities": capability_manifest,
        "context_token_budget": max_context_tokens,
        "smoke_note": (
            "This MemoryAgentBench smoke materialization uses official HF rows but truncates "
            "context to a small token budget for local stub-harness validation. "
            "It is not a formal long-context benchmark run."
        ),
    }
    return canonical_rows, manifest
