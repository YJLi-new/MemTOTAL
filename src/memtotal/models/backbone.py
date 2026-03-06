from __future__ import annotations

import hashlib
from typing import Iterable

import torch
from torch import nn

from memtotal.utils.repro import validate_backbone_name


class BackboneWrapper(nn.Module):
    def __init__(self, name: str, load_mode: str, hidden_size: int, seed: int) -> None:
        super().__init__()
        validate_backbone_name(name)
        if load_mode != "stub":
            raise NotImplementedError(
                "M0 bootstrap only supports load_mode='stub'. Extend BackboneWrapper for real weights."
            )
        self.name = name
        self.load_mode = load_mode
        self.hidden_size = hidden_size
        self.seed = seed

    def _tokenize(self, text: str) -> list[str]:
        tokens = [token for token in text.replace("\n", " ").split(" ") if token]
        return tokens or ["<empty>"]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def _vector_from_token(self, token: str) -> torch.Tensor:
        values = []
        counter = 0
        while len(values) < self.hidden_size:
            digest = hashlib.sha256(
                f"{self.name}|{self.seed}|{counter}|{token}".encode("utf-8")
            ).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        return torch.tensor(values[: self.hidden_size], dtype=torch.float32)

    def encode_texts(self, texts: Iterable[str]) -> torch.Tensor:
        encoded = []
        max_tokens = 1
        tokenized = [self._tokenize(text) for text in texts]
        for tokens in tokenized:
            max_tokens = max(max_tokens, len(tokens))
        for tokens in tokenized:
            token_vectors = [self._vector_from_token(token) for token in tokens]
            while len(token_vectors) < max_tokens:
                token_vectors.append(torch.zeros(self.hidden_size))
            encoded.append(torch.stack(token_vectors))
        return torch.stack(encoded)

    def summarize_texts(self, texts: Iterable[str]) -> torch.Tensor:
        hidden_states = self.encode_texts(texts)
        return hidden_states.mean(dim=1)

    def generate(self, prompts: Iterable[str], memory_tokens: torch.Tensor | None = None) -> list[str]:
        memory_width = 0 if memory_tokens is None else memory_tokens.shape[1]
        generations = []
        for prompt in prompts:
            generations.append(
                f"{self.name} stub generation | memory_slots={memory_width} | prompt={prompt[:80]}"
            )
        return generations
