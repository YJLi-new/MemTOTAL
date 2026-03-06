from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memtotal.data import load_jsonl_dataset


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
    task_name: str,
    smoke_subset: str | None,
) -> dict[str, Any]:
    choices = _normalize_choices(raw_row.get(spec.choices_field), spec) if spec.choices_field else []
    render_context = dict(raw_row)
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
            task_name=str(task_cfg["name"]),
            smoke_subset=task_cfg.get("smoke_subset"),
        )
        for raw_row in raw_rows
    ]
