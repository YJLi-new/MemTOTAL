from __future__ import annotations

from typing import Any


ZERO_SHOT_BASELINE_FAMILIES = {"prompting", "meta_prompting", "rag", "lightthinker", "memory_bank", "memgen"}


def _infer_adapter_trainable_parameter_count(config: dict[str, Any]) -> int:
    baseline_cfg = config.get("baseline", {})
    backbone_cfg = config.get("backbone", {})
    hidden_size = int(backbone_cfg.get("stub_hidden_size", 0))
    mode = str(baseline_cfg.get("mode", "prompt_tuning"))
    if mode == "prompt_tuning":
        prompt_tokens = int(baseline_cfg.get("prompt_tuning", {}).get("prompt_tokens", 0))
        return prompt_tokens * hidden_size
    if mode == "lora":
        rank = int(baseline_cfg.get("lora", {}).get("rank", 0))
        return 2 * hidden_size * rank
    if mode == "ia3":
        return hidden_size
    if mode == "prefix_tuning":
        prefix_tokens = int(baseline_cfg.get("prefix_tuning", {}).get("prefix_tokens", 0))
        return prefix_tokens * hidden_size + hidden_size * hidden_size + hidden_size
    return 0


def build_baseline_budget_fields(
    *,
    config: dict[str, Any],
    baseline_family: str,
    baseline_mode: str,
    support_examples: int | None = None,
    train_steps: int | None = None,
    trainable_parameter_count: int | None = None,
) -> dict[str, object]:
    baseline_cfg = config.get("baseline", {})
    runtime_cfg = config.get("runtime", {})
    resolved_support_examples = (
        int(support_examples)
        if support_examples is not None
        else int(baseline_cfg.get("support_examples", 0))
    )
    resolved_train_steps = (
        int(train_steps)
        if train_steps is not None
        else 0
        if baseline_family in ZERO_SHOT_BASELINE_FAMILIES
        else int(runtime_cfg.get("train_steps", 0))
    )
    resolved_trainable_parameter_count = (
        int(trainable_parameter_count)
        if trainable_parameter_count is not None
        else _infer_adapter_trainable_parameter_count(config)
        if baseline_family == "adapter"
        else 0
    )
    if baseline_family == "adapter":
        budget_scope = "few_shot_adapter"
    elif baseline_family == "lightthinker":
        budget_scope = "compressed_reasoning_prompt"
    elif baseline_family == "memory_bank":
        budget_scope = "external_memory_bank_prompt"
    elif baseline_family == "rag":
        budget_scope = "external_memory_prompt"
    elif baseline_family == "memgen":
        budget_scope = "external_eval_only"
    else:
        budget_scope = "zero_shot_prompt"
    return {
        "baseline_family": baseline_family,
        "baseline_mode": baseline_mode,
        "support_examples": resolved_support_examples,
        "train_steps": resolved_train_steps,
        "trainable_parameter_count": resolved_trainable_parameter_count,
        "budget_scope": budget_scope,
        "budget_signature": (
            f"family={baseline_family}|mode={baseline_mode}|"
            f"shots={resolved_support_examples}|steps={resolved_train_steps}|"
            f"params={resolved_trainable_parameter_count}"
        ),
    }
