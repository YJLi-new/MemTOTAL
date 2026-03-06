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

    def build_compression_prompt(self, example: dict[str, Any]) -> str:
        return f"{example['segment']} || {self.compression_suffix}"

    def _truncate_sketch(self, sketch: str) -> str:
        tokens = sketch.split()
        return " ".join(tokens[: self.max_sketch_tokens]).strip()

    def generate_thought_sketch(self, example: dict[str, Any]) -> tuple[str, str]:
        compression_prompt = self.build_compression_prompt(example)
        sketch = self.backbone.generate([compression_prompt])[0]
        sketch = self._truncate_sketch(sketch)
        return compression_prompt, sketch

    def build_final_prompt(self, example: dict[str, Any], thought_sketch: str) -> str:
        return f"{example['segment']} || Thought sketch: {thought_sketch} || {self.answer_suffix}"

    def generate_text(self, example: dict[str, Any]) -> LightThinkerBaselineOutput:
        compression_prompt, thought_sketch = self.generate_thought_sketch(example)
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
    ) -> LightThinkerBaselineOutput:
        compression_prompt, thought_sketch = self.generate_thought_sketch(example)
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
