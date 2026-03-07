from __future__ import annotations

import hashlib
import os
from typing import Iterable

import torch
from torch import nn

from memtotal.utils.repro import validate_backbone_name


class BackboneWrapper(nn.Module):
    def __init__(
        self,
        name: str,
        load_mode: str,
        hidden_size: int | None,
        seed: int,
        *,
        model_id: str | None = None,
        device: str | torch.device = "cpu",
        dtype: str = "float32",
        cache_dir: str | None = None,
        max_new_tokens: int = 32,
    ) -> None:
        super().__init__()
        validate_backbone_name(name)
        self.name = name
        self.load_mode = load_mode
        self.seed = seed
        self.model_id = model_id or self._default_model_id(name)
        self.device = torch.device(device)
        self.dtype_name = dtype
        self.cache_dir = cache_dir or os.environ.get("HF_HOME") or "/root/autodl-tmp/hf-cache"
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = None
        self.model = None
        if load_mode == "stub":
            if hidden_size is None:
                raise ValueError("Stub backbone requires hidden_size.")
            self.hidden_size = int(hidden_size)
            return
        if load_mode != "hf_causal_lm":
            raise NotImplementedError(
                f"Unsupported backbone load_mode={load_mode}. Expected one of stub, hf_causal_lm."
            )
        self._init_hf_backbone()

    def _default_model_id(self, name: str) -> str:
        defaults = {
            "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen3-8B": "Qwen/Qwen3-8B",
        }
        return defaults[name]

    def _resolve_torch_dtype(self) -> torch.dtype:
        normalized = self.dtype_name.lower()
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32", "float"}:
            return torch.float32
        raise ValueError(
            f"Unsupported backbone dtype={self.dtype_name}. Expected one of float32, float16, bfloat16."
        )

    def _init_hf_backbone(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = self._resolve_torch_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                raise ValueError(f"{self.model_id} tokenizer has neither pad/eos/unk token configured.")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)

    def _tokenize(self, text: str) -> list[str]:
        if self.load_mode == "hf_causal_lm":
            if self.tokenizer is None:
                raise RuntimeError("Real backbone tokenizer is not initialized.")
            token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            return [str(token_id) for token_id in token_ids] or ["<empty>"]
        tokens = [token for token in text.replace("\n", " ").split(" ") if token]
        return tokens or ["<empty>"]

    def count_tokens(self, text: str) -> int:
        if self.load_mode == "hf_causal_lm":
            if self.tokenizer is None:
                raise RuntimeError("Real backbone tokenizer is not initialized.")
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
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
        text_list = list(texts)
        if self.load_mode == "hf_causal_lm":
            return self._encode_texts_hf(text_list)
        encoded = []
        max_tokens = 1
        tokenized = [self._tokenize(text) for text in text_list]
        for tokens in tokenized:
            max_tokens = max(max_tokens, len(tokens))
        for tokens in tokenized:
            token_vectors = [self._vector_from_token(token) for token in tokens]
            while len(token_vectors) < max_tokens:
                token_vectors.append(torch.zeros(self.hidden_size))
            encoded.append(torch.stack(token_vectors))
        return torch.stack(encoded)

    def summarize_texts(self, texts: Iterable[str]) -> torch.Tensor:
        if self.load_mode == "hf_causal_lm":
            text_list = list(texts)
            encoded = self._prepare_hf_inputs(text_list)
            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states[-1].to(dtype=torch.float32)
            attention_mask = encoded["attention_mask"].to(dtype=torch.float32)
            return self._masked_mean_pool(hidden_states, attention_mask)
        hidden_states = self.encode_texts(texts)
        return hidden_states.mean(dim=1)

    def _encode_texts_hf(self, texts: list[str]) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Real backbone is not initialized.")
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states[-1]
        return hidden_states.to(dtype=torch.float32)

    def _masked_mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype, device=hidden_states.device)
        masked_hidden = hidden_states * mask
        lengths = mask.sum(dim=1).clamp_min(1.0)
        return masked_hidden.sum(dim=1) / lengths

    def _prepare_hf_inputs(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Real backbone tokenizer is not initialized.")
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def score_continuations(self, prompt: str, candidate_texts: list[str]) -> torch.Tensor:
        if self.load_mode != "hf_causal_lm":
            raise RuntimeError("score_continuations is only available for load_mode='hf_causal_lm'.")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Real backbone is not initialized.")
        prompt_with_sep = prompt if prompt.endswith((" ", "\n", "\t")) else f"{prompt} "
        full_texts = [f"{prompt_with_sep}{candidate_text}" for candidate_text in candidate_texts]
        encoded = self._prepare_hf_inputs(full_texts)
        prompt_length = len(self.tokenizer(prompt_with_sep, add_special_tokens=True)["input_ids"])
        with torch.inference_mode():
            logits = self.model(**encoded, use_cache=False).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].to(dtype=torch.float32), dim=-1)
        labels = encoded["input_ids"][:, 1:]
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        scores = []
        sequence_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for row_index, sequence_length_value in enumerate(sequence_lengths):
            sequence_length = int(sequence_length_value)
            start_index = max(0, prompt_length - 1)
            end_index = max(start_index, sequence_length - 1)
            scores.append(token_log_probs[row_index, start_index:end_index].sum())
        return torch.stack(scores)

    def generate(self, prompts: Iterable[str], memory_tokens: torch.Tensor | None = None) -> list[str]:
        if self.load_mode == "hf_causal_lm":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Real backbone is not initialized.")
            prompt_list = list(prompts)
            encoded = self._prepare_hf_inputs(prompt_list)
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            completions = []
            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for row_index, prompt_length in enumerate(prompt_lengths):
                new_tokens = generated[row_index, int(prompt_length):]
                completions.append(
                    self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                )
            return completions
        memory_width = 0 if memory_tokens is None else memory_tokens.shape[1]
        generations = []
        for prompt in prompts:
            generations.append(
                f"{self.name} stub generation | memory_slots={memory_width} | prompt={prompt[:80]}"
            )
        return generations
