from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from memtotal.models import BackboneWrapper


@dataclass(frozen=True)
class PromptBaselineOutput:
    prompt: str
    predicted_label: str
    predicted_text: str
    similarity: float | None
    candidate_scores: list[dict[str, float | str]] | None


class PromptBaselineRuntime:
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        baseline_cfg = config.get("baseline", {})
        family = str(baseline_cfg.get("family", "prompting"))
        if family != "prompting":
            raise ValueError(f"Unsupported baseline family: {family}")
        mode = str(baseline_cfg.get("mode", "vanilla"))
        if mode not in {"vanilla", "cot"}:
            raise ValueError(f"Unsupported prompt baseline mode: {mode}")
        backbone_cfg = config["backbone"]
        self.backbone = BackboneWrapper(
            name=backbone_cfg["name"],
            load_mode=backbone_cfg["load_mode"],
            hidden_size=int(backbone_cfg["stub_hidden_size"]),
            seed=seed,
        )
        self.family = family
        self.mode = mode
        self.cot_suffix = str(
            baseline_cfg.get(
                "cot_suffix",
                "Think step by step, then give the final answer.",
            )
        )

    def build_prompt(self, example: dict[str, Any]) -> str:
        prompt = str(example["segment"])
        if self.mode == "cot":
            return f"{prompt} || {self.cot_suffix}"
        return prompt

    def generate_text(self, example: dict[str, Any]) -> PromptBaselineOutput:
        prompt = self.build_prompt(example)
        predicted_text = self.backbone.generate([prompt])[0]
        return PromptBaselineOutput(
            prompt=prompt,
            predicted_label="",
            predicted_text=predicted_text,
            similarity=None,
            candidate_scores=None,
        )

    def predict_multiple_choice(
        self,
        example: dict[str, Any],
        *,
        candidate_labels: list[str],
        candidate_texts: list[str],
    ) -> PromptBaselineOutput:
        prompt = self.build_prompt(example)
        prompt_state = self.backbone.summarize_texts([prompt])
        candidate_states = self.backbone.summarize_texts(candidate_texts)
        normalized_prompt = torch.nn.functional.normalize(prompt_state, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_states, dim=-1)
        scores = torch.matmul(normalized_prompt, normalized_candidates.transpose(0, 1)).squeeze(0)
        best_index = int(torch.argmax(scores).item())
        return PromptBaselineOutput(
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
        )
