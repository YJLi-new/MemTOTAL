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
        if family not in {"prompting", "meta_prompting"}:
            raise ValueError(f"Unsupported baseline family: {family}")
        default_mode = "planner_critic" if family == "meta_prompting" else "vanilla"
        mode = str(baseline_cfg.get("mode", default_mode))
        if family == "prompting" and mode not in {"vanilla", "cot"}:
            raise ValueError(f"Unsupported prompt baseline mode: {mode}")
        if family == "meta_prompting" and mode not in {"planner_critic"}:
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
        self.meta_roles = {
            "planner": str(baseline_cfg.get("planner_role", "Planner: decompose the task into concise reasoning steps.")),
            "solver": str(baseline_cfg.get("solver_role", "Solver: solve using the plan and inspect each option.")),
            "critic": str(baseline_cfg.get("critic_role", "Critic: verify the draft and correct any mistakes.")),
            "finalizer": str(
                baseline_cfg.get("finalizer_role", "Finalizer: provide only the final answer or option label.")
            ),
        }

    def _format_support_examples(self, support_examples: list[dict[str, Any]] | None) -> str:
        if not support_examples:
            return ""
        blocks = []
        for index, support_example in enumerate(support_examples, start=1):
            answer_text = str(
                support_example.get("gold_answer")
                or support_example.get("continuation")
                or support_example.get("answer")
                or support_example["label"]
            )
            blocks.append(
                f"Demo {index} Input: {support_example['segment']} || "
                f"Demo {index} Answer: {answer_text}"
            )
        return " || ".join(blocks)

    def build_prompt(
        self,
        example: dict[str, Any],
        *,
        support_examples: list[dict[str, Any]] | None = None,
    ) -> str:
        prompt = str(example["segment"])
        support_prefix = self._format_support_examples(support_examples)
        task_prompt = f"Task: {prompt}" if self.family == "meta_prompting" else prompt
        if support_prefix:
            task_prompt = f"{support_prefix} || Query: {task_prompt}"
        if self.family == "meta_prompting":
            return (
                f"{self.meta_roles['planner']} || "
                f"{self.meta_roles['solver']} || "
                f"{self.meta_roles['critic']} || "
                f"{self.meta_roles['finalizer']} || "
                f"{task_prompt}"
            )
        if self.mode == "cot":
            return f"{task_prompt} || {self.cot_suffix}"
        return task_prompt

    def generate_text(
        self,
        example: dict[str, Any],
        *,
        support_examples: list[dict[str, Any]] | None = None,
    ) -> PromptBaselineOutput:
        prompt = self.build_prompt(example, support_examples=support_examples)
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
        support_examples: list[dict[str, Any]] | None = None,
    ) -> PromptBaselineOutput:
        prompt = self.build_prompt(example, support_examples=support_examples)
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
