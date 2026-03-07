from __future__ import annotations

import re
from typing import Any

import torch

from memtotal.baselines.prompting import PromptBaselineOutput
from memtotal.models import BackboneWrapper


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


class RetrievalBaselineRuntime:
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        baseline_cfg = config.get("baseline", {})
        family = str(baseline_cfg.get("family", "rag"))
        if family != "rag":
            raise ValueError(f"Unsupported retrieval baseline family: {family}")
        mode = str(baseline_cfg.get("mode", "retrieval_augmented"))
        if mode not in {"retrieval_augmented"}:
            raise ValueError(f"Unsupported retrieval baseline mode: {mode}")
        backbone_cfg = config["backbone"]
        runtime_device = str(config.get("runtime", {}).get("device", "cpu"))
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
        rag_cfg = baseline_cfg.get("rag", {})
        self.family = family
        self.mode = mode
        self.retriever = str(rag_cfg.get("retriever", "lexical_overlap"))
        self.include_answer_in_memory = bool(rag_cfg.get("include_answer_in_memory", True))
        self.memory_prefix = str(rag_cfg.get("memory_prefix", "Retrieved memory"))

    def _answer_text(self, example: dict[str, Any]) -> str:
        return str(
            example.get("gold_answer")
            or example.get("continuation")
            or example.get("answer")
            or example.get("label", "")
        )

    def _memory_text(self, example: dict[str, Any]) -> str:
        parts = [str(example["segment"])]
        answer_text = self._answer_text(example)
        if self.include_answer_in_memory and answer_text:
            parts.append(answer_text)
        return " || ".join(part for part in parts if part)

    def _lexical_overlap_scores(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        query_tokens = _tokenize(query_text)
        scores: list[float] = []
        for candidate_text in candidate_texts:
            candidate_tokens = _tokenize(candidate_text)
            union = query_tokens | candidate_tokens
            overlap = query_tokens & candidate_tokens
            scores.append(float(len(overlap) / max(1, len(union))))
        return scores

    def _dense_stub_scores(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        if not candidate_texts:
            return []
        query_state = self.backbone.summarize_texts([query_text])
        candidate_states = self.backbone.summarize_texts(candidate_texts)
        normalized_query = torch.nn.functional.normalize(query_state, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_states, dim=-1)
        scores = torch.matmul(normalized_query, normalized_candidates.transpose(0, 1)).squeeze(0)
        return [float(value) for value in scores.tolist()]

    def select_support_examples(
        self,
        dataset: list[dict[str, Any]],
        example: dict[str, Any],
        support_examples: int,
    ) -> tuple[list[dict[str, Any]], list[float], str]:
        if support_examples <= 0:
            return [], [], self.retriever
        candidates = [row for row in dataset if row["id"] != example["id"]]
        candidate_texts = [self._memory_text(row) for row in candidates]
        query_text = str(example["segment"])
        if self.retriever == "lexical_overlap":
            scores = self._lexical_overlap_scores(query_text, candidate_texts)
        elif self.retriever == "dense_stub":
            scores = self._dense_stub_scores(query_text, candidate_texts)
        else:
            raise ValueError(f"Unsupported rag retriever: {self.retriever}")
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], str(item[0]["id"])),
        )
        selected = ranked[: min(support_examples, len(ranked))]
        return [row for row, _score in selected], [float(score) for _row, score in selected], self.retriever

    def _format_support_examples(self, support_examples: list[dict[str, Any]] | None) -> str:
        if not support_examples:
            return ""
        blocks = []
        for index, support_example in enumerate(support_examples, start=1):
            blocks.append(
                f"{self.memory_prefix} {index} Input: {support_example['segment']} || "
                f"{self.memory_prefix} {index} Answer: {self._answer_text(support_example)}"
            )
        return " || ".join(blocks)

    def build_prompt(
        self,
        example: dict[str, Any],
        *,
        support_examples: list[dict[str, Any]] | None = None,
    ) -> str:
        support_prefix = self._format_support_examples(support_examples)
        query_prompt = f"Query: {example['segment']}"
        if not support_prefix:
            return query_prompt
        return f"{support_prefix} || {query_prompt}"

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
