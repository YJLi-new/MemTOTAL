from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from memtotal.models import BackboneWrapper


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _truncate_words(text: str, budget: int) -> str:
    tokens = text.split()
    return " ".join(tokens[: max(1, budget)]).strip()


@dataclass(frozen=True)
class MemoryBankBaselineOutput:
    prompt: str
    predicted_label: str
    predicted_text: str
    similarity: float | None
    candidate_scores: list[dict[str, float | str]] | None
    memory_bank_entries: list[dict[str, float | int | str]]


class MemoryBankBaselineRuntime:
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        baseline_cfg = config.get("baseline", {})
        family = str(baseline_cfg.get("family", "memory_bank"))
        if family != "memory_bank":
            raise ValueError(f"Unsupported memory bank baseline family: {family}")
        mode = str(baseline_cfg.get("mode", "episodic_bank"))
        if mode not in {"episodic_bank"}:
            raise ValueError(f"Unsupported memory bank baseline mode: {mode}")
        backbone_cfg = config["backbone"]
        self.backbone = BackboneWrapper(
            name=backbone_cfg["name"],
            load_mode=backbone_cfg["load_mode"],
            hidden_size=int(backbone_cfg["stub_hidden_size"]),
            seed=seed,
        )
        memory_bank_cfg = baseline_cfg.get("memory_bank", {})
        self.family = family
        self.mode = mode
        self.selector = str(memory_bank_cfg.get("selector", "overlap_then_recency"))
        self.eviction_policy = str(memory_bank_cfg.get("eviction_policy", "topk"))
        self.bank_capacity = max(1, int(memory_bank_cfg.get("bank_capacity", 2)))
        self.include_answer_in_memory = bool(memory_bank_cfg.get("include_answer_in_memory", True))
        self.recency_weight = float(memory_bank_cfg.get("recency_weight", 0.05))
        self.cue_word_budget = max(1, int(memory_bank_cfg.get("cue_word_budget", 18)))
        self.outcome_word_budget = max(1, int(memory_bank_cfg.get("outcome_word_budget", 10)))
        self.answer_suffix = str(
            memory_bank_cfg.get(
                "answer_suffix",
                "Use the memory bank to answer the query.",
            )
        )

    def _answer_text(self, example: dict[str, Any]) -> str:
        return str(
            example.get("gold_answer")
            or example.get("continuation")
            or example.get("answer")
            or example.get("label", "")
        )

    def _memory_text(self, example: dict[str, Any]) -> str:
        parts = [str(example["segment"])]
        if self.include_answer_in_memory:
            answer_text = self._answer_text(example)
            if answer_text:
                parts.append(answer_text)
        return " || ".join(parts)

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
            return [], [], self.selector
        candidates = [(index, row) for index, row in enumerate(dataset) if row["id"] != example["id"]]
        candidate_texts = [self._memory_text(row) for _index, row in candidates]
        query_text = str(example["segment"])
        if self.selector == "overlap_then_recency":
            scores = self._lexical_overlap_scores(query_text, candidate_texts)
        elif self.selector == "dense_stub":
            scores = self._dense_stub_scores(query_text, candidate_texts)
        else:
            raise ValueError(f"Unsupported memory bank selector: {self.selector}")
        ranked: list[tuple[dict[str, Any], float, int]] = []
        candidate_count = max(1, len(candidates) - 1)
        for (index, row), score in zip(candidates, scores, strict=True):
            recency_bonus = self.recency_weight * (index / max(1, candidate_count))
            ranked.append((row, float(score + recency_bonus), index))
        ranked.sort(key=lambda item: (-item[1], -item[2], str(item[0]["id"])))
        selected = ranked[: min(support_examples, len(ranked))]
        return [row for row, _score, _index in selected], [score for _row, score, _index in selected], self.selector

    def _build_memory_bank_entries(
        self,
        support_examples: list[dict[str, Any]] | None,
        support_scores: list[float] | None = None,
    ) -> list[dict[str, float | int | str]]:
        if not support_examples:
            return []
        scored_entries = [
            (
                row,
                float(support_scores[index]) if support_scores and index < len(support_scores) else 0.0,
                index,
            )
            for index, row in enumerate(support_examples)
        ]
        if len(scored_entries) > self.bank_capacity:
            if self.eviction_policy == "topk":
                scored_entries = sorted(
                    scored_entries,
                    key=lambda item: (-item[1], str(item[0]["id"])),
                )[: self.bank_capacity]
            elif self.eviction_policy == "recency":
                scored_entries = scored_entries[-self.bank_capacity :]
            else:
                raise ValueError(f"Unsupported memory bank eviction policy: {self.eviction_policy}")
        entries: list[dict[str, float | int | str]] = []
        for slot, (row, score, _index) in enumerate(scored_entries, start=1):
            entries.append(
                {
                    "slot": slot,
                    "id": str(row["id"]),
                    "cue": _truncate_words(str(row["segment"]), self.cue_word_budget),
                    "outcome": _truncate_words(self._answer_text(row), self.outcome_word_budget),
                    "score": float(score),
                }
            )
        return entries

    def build_prompt(
        self,
        example: dict[str, Any],
        *,
        support_examples: list[dict[str, Any]] | None = None,
        support_scores: list[float] | None = None,
    ) -> tuple[str, list[dict[str, float | int | str]]]:
        entries = self._build_memory_bank_entries(support_examples, support_scores=support_scores)
        parts = [
            (
                f"Memory {entry['slot']} Cue: {entry['cue']} || "
                f"Memory {entry['slot']} Outcome: {entry['outcome']}"
            )
            for entry in entries
        ]
        parts.append(f"Query: {example['segment']}")
        parts.append(self.answer_suffix)
        return " || ".join(parts), entries

    def generate_text(
        self,
        example: dict[str, Any],
        *,
        support_examples: list[dict[str, Any]] | None = None,
        support_scores: list[float] | None = None,
    ) -> MemoryBankBaselineOutput:
        prompt, entries = self.build_prompt(
            example,
            support_examples=support_examples,
            support_scores=support_scores,
        )
        predicted_text = self.backbone.generate([prompt])[0]
        return MemoryBankBaselineOutput(
            prompt=prompt,
            predicted_label="",
            predicted_text=predicted_text,
            similarity=None,
            candidate_scores=None,
            memory_bank_entries=entries,
        )

    def predict_multiple_choice(
        self,
        example: dict[str, Any],
        *,
        candidate_labels: list[str],
        candidate_texts: list[str],
        support_examples: list[dict[str, Any]] | None = None,
        support_scores: list[float] | None = None,
    ) -> MemoryBankBaselineOutput:
        prompt, entries = self.build_prompt(
            example,
            support_examples=support_examples,
            support_scores=support_scores,
        )
        prompt_state = self.backbone.summarize_texts([prompt])
        candidate_states = self.backbone.summarize_texts(candidate_texts)
        normalized_prompt = torch.nn.functional.normalize(prompt_state, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_states, dim=-1)
        scores = torch.matmul(normalized_prompt, normalized_candidates.transpose(0, 1)).squeeze(0)
        best_index = int(torch.argmax(scores).item())
        return MemoryBankBaselineOutput(
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
            memory_bank_entries=entries,
        )
