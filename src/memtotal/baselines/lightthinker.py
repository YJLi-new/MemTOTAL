from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from memtotal.models import BackboneWrapper


@dataclass(frozen=True)
class LightThinkerBaselineOutput:
    prompt: str
    predicted_label: str
    predicted_text: str
    similarity: float | None
    candidate_scores: list[dict[str, float | str]] | None
    compression_prompt: str
    thought_sketch: str


class LightThinkerBaselineRuntime:
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        baseline_cfg = config.get("baseline", {})
        family = str(baseline_cfg.get("family", "lightthinker"))
        if family != "lightthinker":
            raise ValueError(f"Unsupported LightThinker baseline family: {family}")
        mode = str(baseline_cfg.get("mode", "compress_then_answer"))
        if mode not in {"compress_then_answer"}:
            raise ValueError(f"Unsupported LightThinker baseline mode: {mode}")
        backbone_cfg = config["backbone"]
        self.backbone = BackboneWrapper(
            name=backbone_cfg["name"],
            load_mode=backbone_cfg["load_mode"],
            hidden_size=int(backbone_cfg["stub_hidden_size"]),
            seed=seed,
        )
        lightthinker_cfg = baseline_cfg.get("lightthinker", {})
        self.family = family
        self.mode = mode
        self.compression_suffix = str(
            lightthinker_cfg.get(
                "compression_suffix",
                "Think through the task and compress the reasoning into a short thought sketch.",
            )
        )
        self.answer_suffix = str(
            lightthinker_cfg.get(
                "answer_suffix",
                "Use the thought sketch to answer the task succinctly.",
            )
        )
        self.max_sketch_tokens = max(1, int(lightthinker_cfg.get("max_sketch_tokens", 16)))

    def _answer_text(self, example: dict[str, Any]) -> str:
        return str(
            example.get("gold_answer")
            or example.get("continuation")
            or example.get("answer")
            or example.get("label", "")
        )

    def _format_support_examples(self, support_examples: list[dict[str, Any]] | None) -> str:
        if not support_examples:
            return ""
        blocks = []
        for index, support_example in enumerate(support_examples, start=1):
            blocks.append(
                f"Demo {index} Input: {support_example['segment']} || "
                f"Demo {index} Answer: {self._answer_text(support_example)}"
            )
        return " || ".join(blocks)

    def build_compression_prompt(
        self,
        example: dict[str, Any],
        *,
        support_examples: list[dict[str, Any]] | None = None,
    ) -> str:
        support_prefix = self._format_support_examples(support_examples)
        task_prompt = str(example["segment"])
        if support_prefix:
            task_prompt = f"{support_prefix} || Query: {task_prompt}"
        return f"{task_prompt} || {self.compression_suffix}"

    def _truncate_sketch(self, sketch: str) -> str:
        tokens = sketch.split()
        return " ".join(tokens[: self.max_sketch_tokens]).strip()

    def generate_thought_sketch(
        self,
        example: dict[str, Any],
        *,
        support_examples: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        compression_prompt = self.build_compression_prompt(example, support_examples=support_examples)
        sketch = self.backbone.generate([compression_prompt])[0]
        sketch = self._truncate_sketch(sketch)
        return compression_prompt, sketch

    def build_final_prompt(self, example: dict[str, Any], thought_sketch: str) -> str:
        return f"{example['segment']} || Thought sketch: {thought_sketch} || {self.answer_suffix}"

    def generate_text(
        self,
        example: dict[str, Any],
        *,
        support_examples: list[dict[str, Any]] | None = None,
    ) -> LightThinkerBaselineOutput:
        compression_prompt, thought_sketch = self.generate_thought_sketch(
            example,
            support_examples=support_examples,
        )
        prompt = self.build_final_prompt(example, thought_sketch)
        predicted_text = self.backbone.generate([prompt])[0]
        return LightThinkerBaselineOutput(
            prompt=prompt,
            predicted_label="",
            predicted_text=predicted_text,
            similarity=None,
            candidate_scores=None,
            compression_prompt=compression_prompt,
            thought_sketch=thought_sketch,
        )

    def predict_multiple_choice(
        self,
        example: dict[str, Any],
        *,
        candidate_labels: list[str],
        candidate_texts: list[str],
        support_examples: list[dict[str, Any]] | None = None,
    ) -> LightThinkerBaselineOutput:
        compression_prompt, thought_sketch = self.generate_thought_sketch(
            example,
            support_examples=support_examples,
        )
        prompt = self.build_final_prompt(example, thought_sketch)
        prompt_state = self.backbone.summarize_texts([prompt])
        candidate_states = self.backbone.summarize_texts(candidate_texts)
        normalized_prompt = torch.nn.functional.normalize(prompt_state, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_states, dim=-1)
        scores = torch.matmul(normalized_prompt, normalized_candidates.transpose(0, 1)).squeeze(0)
        best_index = int(torch.argmax(scores).item())
        return LightThinkerBaselineOutput(
            prompt=prompt,
            predicted_label=candidate_labels[best_index],
            predicted_text=candidate_texts[best_index],
            similarity=float(scores[best_index].item()),
            candidate_scores=[
                {
                    "label": label,
                    "text": text,
                    "score": float(scores[index].item()),
                }
                for index, (label, text) in enumerate(zip(candidate_labels, candidate_texts))
            ],
            compression_prompt=compression_prompt,
            thought_sketch=thought_sketch,
        )
