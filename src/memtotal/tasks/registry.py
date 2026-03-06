from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memtotal.data import load_jsonl_dataset
from memtotal.tasks.sources import select_narrativeqa_story_segments


@dataclass(frozen=True)
class TaskSpec:
    benchmark_id: str
    display_name: str
    domain: str
    evaluator_type: str
    metric_name: str
    prompt_template: str
    answer_field: str = "answer"
    choices_field: str | None = None
    label_field: str | None = None
    choice_label_field: str = "label"
    choice_text_field: str = "text"
    normalizer: str = "text"
    supports_tools: bool = False
    passthrough_fields: tuple[str, ...] = ()


TASK_SPECS: dict[str, TaskSpec] = {
    "gsm8k": TaskSpec(
        benchmark_id="gsm8k",
        display_name="GSM8K",
        domain="math",
        evaluator_type="exact_match",
        metric_name="exact_match",
        prompt_template="Question: {question} || Solve carefully and give the final answer.",
    ),
    "math": TaskSpec(
        benchmark_id="math",
        display_name="MATH",
        domain="math",
        evaluator_type="exact_match",
        metric_name="exact_match",
        prompt_template="Problem: {question} || Show concise reasoning and give the final answer.",
    ),
    "gpqa": TaskSpec(
        benchmark_id="gpqa",
        display_name="GPQA",
        domain="qa",
        evaluator_type="multiple_choice",
        metric_name="accuracy",
        prompt_template="Question: {question} || Options: {choices_block} || Select the best option.",
        choices_field="choices",
        label_field="label",
    ),
    "triviaqa": TaskSpec(
        benchmark_id="triviaqa",
        display_name="TriviaQA",
        domain="qa",
        evaluator_type="exact_match",
        metric_name="exact_match",
        prompt_template="Question: {question} || Answer briefly.",
    ),
    "kodcode": TaskSpec(
        benchmark_id="kodcode",
        display_name="KodCode",
        domain="code",
        evaluator_type="exact_match",
        metric_name="exact_match",
        prompt_template="Coding task: {prompt} || Return the requested code snippet.",
        normalizer="code",
    ),
    "story_cloze": TaskSpec(
        benchmark_id="story_cloze",
        display_name="Story Cloze",
        domain="narrative",
        evaluator_type="multiple_choice",
        metric_name="accuracy",
        prompt_template="Story: {story} || Candidate endings: {choices_block} || Choose the most coherent ending.",
        choices_field="choices",
        label_field="label",
    ),
    "narrativeqa": TaskSpec(
        benchmark_id="narrativeqa",
        display_name="NarrativeQA",
        domain="narrative",
        evaluator_type="qa_f1",
        metric_name="f1",
        prompt_template="{story_segments_block} || Question: {question} || Answer briefly.",
        passthrough_fields=(
            "story",
            "story_segments",
            "story_chunk_pool_size",
            "summary_title",
            "document_kind",
            "story_chars",
            "story_word_count",
            "story_excerpt_chars",
            "story_segment_words",
            "story_segments_materialized",
            "story_total_segments",
            "story_selected_indexes",
            "story_start_index",
            "story_selection_strategy",
            "story_query_token_count",
            "story_runtime_segment_budget",
            "story_runtime_selector",
            "story_truncated_for_smoke",
            "narrativeqa_view",
        ),
    ),
    "rocstories": TaskSpec(
        benchmark_id="rocstories",
        display_name="ROCStories",
        domain="narrative",
        evaluator_type="multiple_choice",
        metric_name="accuracy",
        prompt_template="Story: {story} || Candidate endings: {choices_block} || Choose the best ending.",
        choices_field="choices",
        label_field="label",
    ),
    "fever": TaskSpec(
        benchmark_id="fever",
        display_name="FEVER",
        domain="qa",
        evaluator_type="multiple_choice",
        metric_name="accuracy",
        prompt_template="Claim: {claim} || Evidence: {evidence} || Labels: {choices_block} || Decide the correct label.",
        choices_field="choices",
        label_field="label",
    ),
    "alfworld": TaskSpec(
        benchmark_id="alfworld",
        display_name="ALFWorld",
        domain="agent",
        evaluator_type="exact_match",
        metric_name="exact_match",
        prompt_template="Observation: {observation} || Goal: {goal} || What is the next action?",
        normalizer="action",
        supports_tools=True,
    ),
    "memoryagentbench": TaskSpec(
        benchmark_id="memoryagentbench",
        display_name="MemoryAgentBench",
        domain="agent",
        evaluator_type="memoryagentbench",
        metric_name="memoryagent_score",
        prompt_template="Context: {context} || Instruction: {question}",
        passthrough_fields=(
            "question",
            "capability",
            "capability_name",
            "memoryagent_source",
            "question_index",
            "capability_metric_name",
            "keypoints",
            "qa_pair_id",
            "question_type",
            "context_token_budget",
            "context_tokens_total",
            "context_tokens_used",
            "context_was_truncated",
            "full_context_chars",
        ),
    ),
}


def list_task_specs() -> list[TaskSpec]:
    return [TASK_SPECS[key] for key in sorted(TASK_SPECS)]


def get_task_spec(benchmark_id: str) -> TaskSpec:
    try:
        return TASK_SPECS[benchmark_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark_id: {benchmark_id}") from exc


def _normalize_choices(raw_choices: object, spec: TaskSpec) -> list[dict[str, str]]:
    if raw_choices is None:
        return []
    if isinstance(raw_choices, dict):
        return [
            {"label": str(label), "text": str(text)}
            for label, text in raw_choices.items()
        ]
    if not isinstance(raw_choices, list):
        raise ValueError(f"{spec.benchmark_id} expects choices to be a list or dict.")
    choices: list[dict[str, str]] = []
    for item in raw_choices:
        if isinstance(item, dict):
            label = str(item.get(spec.choice_label_field, item.get("id", "")))
            text = str(item.get(spec.choice_text_field, item.get("value", "")))
        else:
            label = chr(ord("A") + len(choices))
            text = str(item)
        if not label or not text:
            raise ValueError(f"{spec.benchmark_id} contains an invalid choice item: {item!r}")
        choices.append({"label": label, "text": text})
    return choices


def _format_choices_block(choices: list[dict[str, str]]) -> str:
    return " | ".join(f"{choice['label']}: {choice['text']}" for choice in choices)


def _resolve_gold_choice_text(choices: list[dict[str, str]], gold_label: str) -> str:
    for choice in choices:
        if choice["label"] == gold_label:
            return choice["text"]
    raise ValueError(f"Gold label '{gold_label}' is missing from choices.")


def _build_canonical_benchmark_example(
    raw_row: dict[str, Any],
    spec: TaskSpec,
    *,
    task_cfg: dict[str, Any],
    task_name: str,
    smoke_subset: str | None,
) -> dict[str, Any]:
    choices = _normalize_choices(raw_row.get(spec.choices_field), spec) if spec.choices_field else []
    render_context = dict(raw_row)
    if spec.benchmark_id == "narrativeqa":
        title = str(raw_row.get("summary_title", "")).strip()
        runtime_cfg = task_cfg.get("narrativeqa_runtime", {})
        story_segments = [str(segment).strip() for segment in raw_row.get("story_segments", []) if str(segment).strip()]
        story_chunk_pool = [str(segment).strip() for segment in raw_row.get("story_chunk_pool", []) if str(segment).strip()]
        if story_chunk_pool:
            selected_segments, selected_indexes, selection_strategy = select_narrativeqa_story_segments(
                story_chunk_pool,
                max_segments=int(runtime_cfg.get("segment_budget", len(story_segments) or 6)),
                query_text=str(raw_row.get("question", "")),
            )
            story_segments = selected_segments
            render_context["story_segments"] = selected_segments
            render_context["story_selected_indexes"] = selected_indexes
            render_context["story_selection_strategy"] = selection_strategy
            render_context["story_runtime_segment_budget"] = int(runtime_cfg.get("segment_budget", len(selected_segments)))
            render_context["story_runtime_selector"] = str(runtime_cfg.get("selector", "question_aware"))
            render_context["story_segments_materialized"] = len(selected_segments)
            render_context["story_excerpt_chars"] = len(" ".join(selected_segments).strip())
            render_context["story_truncated_for_smoke"] = len(selected_segments) < len(story_chunk_pool)
        if not story_segments:
            story_text = str(raw_row.get("story", "")).strip()
            if story_text:
                story_segments = [story_text]
        story_blocks: list[str] = []
        if title:
            story_blocks.append(f"Title: {title}")
        pool_size = int(render_context.get("story_chunk_pool_size", len(story_segments)))
        selected_indexes = [int(index) for index in render_context.get("story_selected_indexes", list(range(len(story_segments))))]
        for segment_index, segment_text in enumerate(story_segments):
            pool_index = selected_indexes[segment_index] + 1 if segment_index < len(selected_indexes) else segment_index + 1
            story_blocks.append(
                f"Story segment {segment_index + 1}/{len(story_segments)} [pool {pool_index}/{pool_size}]: {segment_text}"
            )
        render_context["story_segments_block"] = " || ".join(story_blocks)
    render_context["choices_block"] = _format_choices_block(choices)
    segment = spec.prompt_template.format(**render_context)
    answer_text = str(raw_row.get(spec.answer_field, ""))
    gold_label = str(raw_row.get(spec.label_field, answer_text)) if spec.label_field else answer_text
    continuation = _resolve_gold_choice_text(choices, gold_label) if choices else answer_text
    example = {
        "id": str(raw_row.get("id", f"{spec.benchmark_id}-{task_name}-{raw_row.get('index', 0)}")),
        "benchmark_id": spec.benchmark_id,
        "task_name": task_name,
        "display_name": spec.display_name,
        "domain": str(raw_row.get("domain", spec.domain)),
        "segment": segment,
        "continuation": continuation,
        "label": gold_label,
        "gold_answer": answer_text or continuation,
        "metric_name": spec.metric_name,
        "evaluator_type": spec.evaluator_type,
        "normalizer": spec.normalizer,
        "smoke_subset": smoke_subset,
        "supports_tools": spec.supports_tools,
    }
    if choices:
        example["choices"] = choices
    aliases = raw_row.get("aliases")
    if aliases is not None:
        if not isinstance(aliases, list):
            raise ValueError(f"{spec.benchmark_id} aliases must be a list when provided.")
        example["aliases"] = [str(alias) for alias in aliases]
    for field_name in spec.passthrough_fields:
        if field_name in render_context:
            example[field_name] = render_context[field_name]
    return example


def load_task_dataset(config: dict[str, Any]) -> list[dict[str, Any]]:
    task_cfg = config["task"]
    raw_rows = load_jsonl_dataset(task_cfg["dataset_path"])
    benchmark_id = task_cfg.get("benchmark_id")
    if benchmark_id is None:
        return raw_rows
    spec = get_task_spec(str(benchmark_id))
    return [
        _build_canonical_benchmark_example(
            raw_row,
            spec,
            task_cfg=task_cfg,
            task_name=str(task_cfg["name"]),
            smoke_subset=task_cfg.get("smoke_subset"),
        )
        for raw_row in raw_rows
    ]
