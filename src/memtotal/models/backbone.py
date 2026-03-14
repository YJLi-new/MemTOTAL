from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import math
import os
from typing import Any, Iterable

import torch
from torch import nn

from memtotal.utils.repro import validate_backbone_name


class MicroLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"Micro-LoRA rank must be positive, got {rank}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"Micro-LoRA dropout must be in [0, 1), got {dropout}.")
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("Micro-LoRA can wrap only nn.Linear modules.")
        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha / max(1, rank))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.down = nn.Linear(base_linear.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, base_linear.out_features, bias=False)
        self.down.to(device=base_linear.weight.device, dtype=base_linear.weight.dtype)
        self.up.to(device=base_linear.weight.device, dtype=base_linear.weight.dtype)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.down(self.dropout(inputs)))
        return self.base_linear(inputs) + (self.scale * delta)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base_linear.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base_linear.bias

    @property
    def in_features(self) -> int:
        return int(self.base_linear.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base_linear.out_features)

    def lora_parameters(self) -> list[nn.Parameter]:
        return [self.down.weight, self.up.weight]

    def lora_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.lora_parameters()))

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "down.weight": self.down.weight.detach().cpu(),
            "up.weight": self.up.weight.detach().cpu(),
        }

    def validate_lora_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        module_path: str,
        checkpoint_path: str,
    ) -> None:
        expected = {
            "down.weight": self.down.weight,
            "up.weight": self.up.weight,
        }
        for key, expected_tensor in expected.items():
            if key not in state_dict:
                raise ValueError(
                    f"Receiver micro-LoRA checkpoint {checkpoint_path} is missing '{module_path}.{key}'."
                )
            actual_tensor = state_dict[key]
            if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
                raise ValueError(
                    f"Receiver micro-LoRA checkpoint {checkpoint_path} has incompatible "
                    f"'{module_path}.{key}': expected shape {tuple(expected_tensor.shape)}, "
                    f"got {tuple(actual_tensor.shape)}."
                )

    def load_lora_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        module_path: str,
        checkpoint_path: str,
    ) -> None:
        self.validate_lora_state_dict(
            state_dict,
            module_path=module_path,
            checkpoint_path=checkpoint_path,
        )
        with torch.no_grad():
            self.down.weight.copy_(
                state_dict["down.weight"].to(
                    device=self.down.weight.device,
                    dtype=self.down.weight.dtype,
                )
            )
            self.up.weight.copy_(
                state_dict["up.weight"].to(
                    device=self.up.weight.device,
                    dtype=self.up.weight.dtype,
                )
            )


ReceiverMicroLoRALinear = MicroLoRALinear


class ReaderCrossAttentionAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        ff_hidden_dim: int,
        gate_init: float,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("ReaderCrossAttentionAdapter requires positive hidden_size.")
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(
                f"ReaderCrossAttentionAdapter requires hidden_size divisible by num_heads, got "
                f"hidden_size={hidden_size}, num_heads={num_heads}."
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size // num_heads)
        self.scale = float(1.0 / math.sqrt(self.head_dim))
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.memory_norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, int(ff_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(ff_hidden_dim), self.hidden_size),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self._last_diagnostics = {
            "gate_open_fraction": 0.0,
            "memory_token_attention_mass_mean": 0.0,
            "memory_token_attention_entropy_mean": 0.0,
            "memory_token_attention_top_mass_mean": 0.0,
        }

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tensor.shape
        return tensor.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, token_count, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch_size, token_count, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden_size].")
        if memory_tokens.ndim != 3:
            raise ValueError("memory_tokens must have shape [batch, memory_tokens, hidden_size].")
        if hidden_states.shape[-1] != self.hidden_size or memory_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"ReaderCrossAttentionAdapter expected hidden size {self.hidden_size}, got "
                f"{hidden_states.shape[-1]} and {memory_tokens.shape[-1]}."
            )
        batch_size = int(hidden_states.shape[0])
        if memory_tokens.shape[0] == 1 and batch_size > 1:
            memory_tokens = memory_tokens.expand(batch_size, -1, -1)
        if memory_tokens.shape[0] != batch_size:
            raise ValueError(
                "memory_tokens batch dimension must be 1 or match hidden_states batch dimension."
            )
        query_states = self._split_heads(self.q_proj(self.query_norm(hidden_states)))
        normalized_memory = self.memory_norm(memory_tokens)
        key_states = self._split_heads(self.k_proj(normalized_memory))
        value_states = self._split_heads(self.v_proj(normalized_memory))
        attention_logits = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        attention = torch.softmax(attention_logits.to(dtype=torch.float32), dim=-1).to(dtype=query_states.dtype)
        attended = torch.matmul(attention, value_states)
        delta = self.out_proj(self._merge_heads(attended))
        delta = delta + self.ffn(delta)
        gate_value = torch.sigmoid(self.gate.to(dtype=delta.dtype))
        if attention.numel() > 0:
            attention_fp32 = attention.detach().to(dtype=torch.float32)
            top_mass = attention_fp32.max(dim=-1).values
            entropy = -torch.sum(
                attention_fp32.clamp_min(1e-8) * torch.log(attention_fp32.clamp_min(1e-8)),
                dim=-1,
            )
            self._last_diagnostics = {
                "gate_open_fraction": float(gate_value.detach().to(dtype=torch.float32).item()),
                "memory_token_attention_mass_mean": float(attention_fp32.mean().item()),
                "memory_token_attention_entropy_mean": float(entropy.mean().item()),
                "memory_token_attention_top_mass_mean": float(top_mass.mean().item()),
            }
        else:
            self._last_diagnostics = {
                "gate_open_fraction": float(gate_value.detach().to(dtype=torch.float32).item()),
                "memory_token_attention_mass_mean": 0.0,
                "memory_token_attention_entropy_mean": 0.0,
                "memory_token_attention_top_mass_mean": 0.0,
            }
        return hidden_states + (gate_value * delta)

    def diagnostics(self) -> dict[str, float]:
        return dict(self._last_diagnostics)


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
        attn_implementation: str | None = None,
        max_new_tokens: int = 32,
        gradient_checkpointing: bool = False,
        use_chat_template: bool = False,
        chat_template_enable_thinking: bool | None = None,
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
        self.attn_implementation = attn_implementation
        self.max_new_tokens = int(max_new_tokens)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.use_chat_template = bool(use_chat_template)
        self.chat_template_enable_thinking = (
            None
            if chat_template_enable_thinking is None
            else bool(chat_template_enable_thinking)
        )
        self.tokenizer = None
        self.model = None
        self._receiver_lora_targets: dict[str, MicroLoRALinear] = {}
        self._receiver_lora_config: dict[str, Any] = {
            "enabled": False,
            "target_layers": [],
            "target_modules": [],
            "rank": 0,
            "alpha": 0.0,
            "dropout": 0.0,
        }
        self._reader_cross_attn_targets = nn.ModuleDict()
        self._reader_cross_attn_config: dict[str, Any] = {
            "enabled": False,
            "target_layers": [],
            "num_heads": 0,
            "ff_hidden_dim": 0,
            "gate_init": 0.0,
        }
        self._active_reader_cross_attn_memory: torch.Tensor | None = None
        self._reader_cross_attn_last_diagnostics: dict[str, dict[str, float]] = {}
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
            "Qwen3-4B": "Qwen/Qwen3-4B",
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
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)

    @contextmanager
    def _temporary_attention_implementation(self, implementation: str):
        if self.model is None:
            yield
            return
        config_targets: list[tuple[Any, Any]] = []
        seen_configs: set[int] = set()
        for module in self.model.modules():
            config = getattr(module, "config", None)
            if config is None or not hasattr(config, "_attn_implementation"):
                continue
            config_id = id(config)
            if config_id in seen_configs:
                continue
            seen_configs.add(config_id)
            config_targets.append((config, getattr(config, "_attn_implementation")))
        if not config_targets:
            yield
            return
        try:
            for config, _previous in config_targets:
                setattr(config, "_attn_implementation", implementation)
            yield
        finally:
            for config, previous in config_targets:
                setattr(config, "_attn_implementation", previous)

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

    def _tail_hidden_state_window(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden_size].")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, tokens].")
        if hidden_states.shape[:2] != attention_mask.shape:
            raise ValueError("hidden_states and attention_mask must agree on [batch, tokens].")
        window_tokens = max(1, int(max_tokens))
        batch_size, _, hidden_size = hidden_states.shape
        window = torch.zeros(
            batch_size,
            window_tokens,
            hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        window_mask = torch.zeros(
            batch_size,
            window_tokens,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        lengths = attention_mask.to(dtype=torch.long, device=hidden_states.device).sum(dim=1)
        for row_index in range(batch_size):
            valid_tokens = int(lengths[row_index].item())
            tail_length = max(1, min(window_tokens, valid_tokens))
            if valid_tokens > 0:
                tail_hidden = hidden_states[row_index, valid_tokens - tail_length : valid_tokens, :]
            else:
                tail_hidden = hidden_states[row_index, :1, :]
            window[row_index, :tail_length, :] = tail_hidden
            window_mask[row_index, :tail_length] = True
        return window, window_mask

    def extract_prompt_hidden_state_slice(
        self,
        texts: Iterable[str],
        *,
        max_tokens: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_list = list(texts)
        if self.load_mode == "hf_causal_lm":
            encoded = self._prepare_hf_inputs(text_list)
            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states[-1].to(dtype=torch.float32)
            return self._tail_hidden_state_window(
                hidden_states,
                encoded["attention_mask"],
                max_tokens=max_tokens,
            )
        encoded = self.encode_texts(text_list)
        lengths = [max(1, len(self._tokenize(text))) for text in text_list]
        max_length = encoded.shape[1]
        attention_mask = torch.zeros(
            len(text_list),
            max_length,
            dtype=torch.bool,
            device=encoded.device,
        )
        for row_index, length in enumerate(lengths):
            attention_mask[row_index, : min(length, max_length)] = True
        return self._tail_hidden_state_window(
            encoded.to(dtype=torch.float32),
            attention_mask,
            max_tokens=max_tokens,
        )

    def extract_layer_hidden_state_slices(
        self,
        texts: Iterable[str],
        *,
        layer_indices: Iterable[int],
        max_tokens: int = 8,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        normalized_layers = tuple(sorted({int(layer_index) for layer_index in layer_indices}))
        if not normalized_layers:
            raise ValueError("extract_layer_hidden_state_slices requires at least one layer index.")
        text_list = list(texts)
        if self.load_mode == "hf_causal_lm":
            encoded = self._prepare_hf_inputs(text_list)
            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_state_stack = outputs.hidden_states
            max_layer_index = len(hidden_state_stack) - 2
            window_by_layer: dict[int, torch.Tensor] = {}
            shared_mask = None
            for layer_index in normalized_layers:
                if layer_index < 0 or layer_index > max_layer_index:
                    raise ValueError(
                        f"Requested layer {layer_index}, but backbone exposes layers [0, {max_layer_index}]."
                    )
                layer_hidden = hidden_state_stack[layer_index + 1].to(dtype=torch.float32)
                window, window_mask = self._tail_hidden_state_window(
                    layer_hidden,
                    encoded["attention_mask"],
                    max_tokens=max_tokens,
                )
                window_by_layer[layer_index] = window
                if shared_mask is None:
                    shared_mask = window_mask
            if shared_mask is None:
                raise RuntimeError("extract_layer_hidden_state_slices failed to collect any layer windows.")
            return window_by_layer, shared_mask
        encoded = self.encode_texts(text_list)
        lengths = [max(1, len(self._tokenize(text))) for text in text_list]
        max_length = encoded.shape[1]
        attention_mask = torch.zeros(
            len(text_list),
            max_length,
            dtype=torch.bool,
            device=encoded.device,
        )
        for row_index, length in enumerate(lengths):
            attention_mask[row_index, : min(length, max_length)] = True
        window, window_mask = self._tail_hidden_state_window(
            encoded.to(dtype=torch.float32),
            attention_mask,
            max_tokens=max_tokens,
        )
        return {
            layer_index: window.clone()
            for layer_index in normalized_layers
        }, window_mask

    def collect_deep_prefix_calibration(
        self,
        texts: Iterable[str],
        *,
        layer_indices: Iterable[int],
        max_tokens: int = 8,
    ) -> dict[str, Any]:
        text_list = list(texts)
        if not text_list:
            raise ValueError("collect_deep_prefix_calibration requires at least one text.")
        semantic_anchor = self.summarize_texts(text_list).mean(dim=0, keepdim=True).to(dtype=torch.float32)
        hidden_state_anchor, hidden_state_mask = self.extract_prompt_hidden_state_slice(
            text_list,
            max_tokens=max_tokens,
        )
        hidden_state_anchor = hidden_state_anchor.mean(dim=0, keepdim=True).to(dtype=torch.float32)
        hidden_state_mask = hidden_state_mask.any(dim=0, keepdim=True).to(dtype=torch.bool)
        layer_hidden_windows, layer_hidden_mask = self.extract_layer_hidden_state_slices(
            text_list,
            layer_indices=layer_indices,
            max_tokens=max_tokens,
        )
        averaged_layer_hidden_by_layer = {
            int(layer_index): layer_hidden.mean(dim=0, keepdim=True).to(dtype=torch.float32)
            for layer_index, layer_hidden in layer_hidden_windows.items()
        }
        projection_summary = self.summarize_layer_prefix_projection(
            averaged_layer_hidden_by_layer,
            batch_size=1,
        )
        return {
            "semantic_anchor": semantic_anchor,
            "hidden_state_anchor": hidden_state_anchor,
            "hidden_state_mask": hidden_state_mask,
            "layer_hidden_anchor_by_layer": averaged_layer_hidden_by_layer,
            "layer_hidden_mask": layer_hidden_mask.any(dim=0, keepdim=True).to(dtype=torch.bool),
            "layer_key_l2_by_layer": dict(projection_summary.get("layer_key_l2_by_layer", {})),
            "layer_value_l2_by_layer": dict(projection_summary.get("layer_value_l2_by_layer", {})),
            "layer_hidden_l2_by_layer": dict(projection_summary.get("layer_hidden_l2_by_layer", {})),
        }

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

    def _normalize_memory_tokens(
        self,
        memory_tokens: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if memory_tokens.ndim != 3:
            raise ValueError("memory_tokens must have shape [batch, memory_tokens, hidden_size].")
        if memory_tokens.shape[0] == 1 and batch_size > 1:
            memory_tokens = memory_tokens.expand(batch_size, -1, -1)
        if memory_tokens.shape[0] != batch_size:
            raise ValueError(
                "memory_tokens batch dimension must be 1 or match the number of candidate sequences."
            )
        if memory_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"memory_tokens hidden size must match backbone hidden size {self.hidden_size}, "
                f"got {memory_tokens.shape[-1]}."
            )
        return memory_tokens.to(device=self.device, dtype=self._resolve_torch_dtype())

    def _chunk_pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        pool_window: int,
        slot_cap: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or attention_mask.ndim != 2:
            raise ValueError("hidden_states must be [batch, tokens, hidden_size] and attention_mask [batch, tokens].")
        window = max(1, int(pool_window))
        max_slots = max(1, int(slot_cap))
        batch_size, _, hidden_size = hidden_states.shape
        pooled = torch.zeros(
            batch_size,
            max_slots,
            hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        pooled_mask = torch.zeros(
            batch_size,
            max_slots,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        lengths = attention_mask.to(dtype=torch.long, device=hidden_states.device).sum(dim=1)
        for row_index in range(batch_size):
            valid_tokens = int(lengths[row_index].item())
            if valid_tokens <= 0:
                pooled[row_index, 0, :] = hidden_states[row_index, 0, :]
                pooled_mask[row_index, 0] = True
                continue
            token_states = hidden_states[row_index, :valid_tokens, :]
            chunk_count = max(1, math.ceil(valid_tokens / window))
            if chunk_count > max_slots:
                start = valid_tokens - (max_slots * window)
                start = max(0, start)
                token_states = token_states[start:, :]
                chunk_count = max_slots
            for chunk_index in range(chunk_count):
                chunk_start = chunk_index * window
                chunk_end = min(token_states.shape[0], chunk_start + window)
                if chunk_end <= chunk_start:
                    break
                pooled[row_index, chunk_index, :] = token_states[chunk_start:chunk_end, :].mean(dim=0)
                pooled_mask[row_index, chunk_index] = True
        return pooled, pooled_mask

    def extract_chunk_pooled_hidden_state_slots(
        self,
        texts: Iterable[str],
        *,
        layer_index: int | None = None,
        pool_window: int = 16,
        slot_cap: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_list = list(texts)
        if self.load_mode == "hf_causal_lm":
            encoded = self._prepare_hf_inputs(text_list)
            with torch.inference_mode():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_state_stack = outputs.hidden_states
            if layer_index is None:
                hidden_states = hidden_state_stack[-1].to(dtype=torch.float32)
            else:
                max_layer_index = len(hidden_state_stack) - 2
                if layer_index < 0 or layer_index > max_layer_index:
                    raise ValueError(
                        f"Requested layer {layer_index}, but backbone exposes layers [0, {max_layer_index}]."
                    )
                hidden_states = hidden_state_stack[layer_index + 1].to(dtype=torch.float32)
            return self._chunk_pool_hidden_states(
                hidden_states,
                encoded["attention_mask"],
                pool_window=pool_window,
                slot_cap=slot_cap,
            )
        encoded = self.encode_texts(text_list)
        lengths = [max(1, len(self._tokenize(text))) for text in text_list]
        attention_mask = torch.zeros(
            len(text_list),
            encoded.shape[1],
            dtype=torch.bool,
            device=encoded.device,
        )
        for row_index, length in enumerate(lengths):
            attention_mask[row_index, : min(length, encoded.shape[1])] = True
        return self._chunk_pool_hidden_states(
            encoded.to(dtype=torch.float32),
            attention_mask,
            pool_window=pool_window,
            slot_cap=slot_cap,
        )

    def _format_chat_prompts(self, texts: list[str]) -> list[str]:
        if not self.use_chat_template or self.tokenizer is None:
            return texts
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            return texts
        formatted: list[str] = []
        for text in texts:
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.chat_template_enable_thinking is not None:
                template_kwargs["enable_thinking"] = bool(self.chat_template_enable_thinking)
            formatted.append(
                str(
                    apply_chat_template(
                        [{"role": "user", "content": str(text)}],
                        **template_kwargs,
                    )
                )
            )
        return formatted

    def _prepare_hf_inputs(
        self,
        texts: list[str],
        *,
        apply_chat_template: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Real backbone tokenizer is not initialized.")
        active_texts = self._format_chat_prompts(texts) if apply_chat_template else texts
        encoded = self.tokenizer(
            active_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _prepare_prefixed_hf_inputs(
        self,
        texts: list[str],
        prefix_embeddings: torch.Tensor | None,
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None,
        *,
        memory_tokens: torch.Tensor | None = None,
        apply_chat_template: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if prefix_embeddings is not None and layer_prefix_hidden_by_layer is not None:
            raise ValueError(
                "prefix_embeddings and layer_prefix_hidden_by_layer are mutually exclusive."
            )
        if memory_tokens is not None and (
            prefix_embeddings is not None or layer_prefix_hidden_by_layer is not None
        ):
            raise ValueError(
                "memory_tokens is mutually exclusive with prefix_embeddings and layer_prefix_hidden_by_layer."
            )
        encoded = self._prepare_hf_inputs(texts, apply_chat_template=apply_chat_template)
        metadata = {
            "prefix_length": 0,
            "score_input_ids": encoded["input_ids"],
            "score_attention_mask": encoded["attention_mask"],
            "token_logit_slice_start": 0,
            "attention_query_offset": 0,
            "diagnostic_layers": [],
        }
        if prefix_embeddings is None and layer_prefix_hidden_by_layer is None and memory_tokens is None:
            return encoded, metadata
        if self.model is None:
            raise RuntimeError("Real backbone model is not initialized.")
        if memory_tokens is not None:
            batch_size = encoded["input_ids"].shape[0]
            normalized_memory = self._normalize_memory_tokens(memory_tokens, batch_size=batch_size)
            input_embeddings = self.model.get_input_embeddings()(encoded["input_ids"])
            model_kwargs = dict(encoded)
            model_kwargs.pop("input_ids")
            memory_mask = torch.ones(
                batch_size,
                normalized_memory.shape[1],
                dtype=encoded["attention_mask"].dtype,
                device=self.device,
            )
            model_kwargs["inputs_embeds"] = torch.cat([normalized_memory, input_embeddings], dim=1)
            model_kwargs["attention_mask"] = torch.cat([memory_mask, encoded["attention_mask"]], dim=1)
            metadata.update(
                {
                    "prefix_length": int(normalized_memory.shape[1]),
                    "token_logit_slice_start": int(normalized_memory.shape[1]),
                    "attention_query_offset": int(normalized_memory.shape[1]),
                    "diagnostic_layers": [max(0, int(getattr(self.model.config, "num_hidden_layers", 1)) - 1)],
                }
            )
            return model_kwargs, metadata
        if prefix_embeddings is not None:
            if prefix_embeddings.ndim != 3:
                raise ValueError("prefix_embeddings must have shape [batch, prefix_tokens, hidden_size].")
            batch_size = encoded["input_ids"].shape[0]
            if prefix_embeddings.shape[0] == 1 and batch_size > 1:
                prefix_embeddings = prefix_embeddings.expand(batch_size, -1, -1)
            if prefix_embeddings.shape[0] != batch_size:
                raise ValueError(
                    "prefix_embeddings batch dimension must be 1 or match the number of candidate sequences."
                )
            if prefix_embeddings.shape[2] != self.hidden_size:
                raise ValueError(
                    f"prefix_embeddings hidden size must match backbone hidden size {self.hidden_size}, "
                    f"got {prefix_embeddings.shape[2]}."
                )
            prefix_embeddings = prefix_embeddings.to(device=self.device, dtype=self._resolve_torch_dtype())
            input_embeddings = self.model.get_input_embeddings()(encoded["input_ids"])
            model_kwargs = dict(encoded)
            model_kwargs.pop("input_ids")
            prefix_mask = torch.ones(
                batch_size,
                prefix_embeddings.shape[1],
                dtype=encoded["attention_mask"].dtype,
                device=self.device,
            )
            model_kwargs["inputs_embeds"] = torch.cat([prefix_embeddings, input_embeddings], dim=1)
            model_kwargs["attention_mask"] = torch.cat([prefix_mask, encoded["attention_mask"]], dim=1)
            metadata.update(
                {
                    "prefix_length": int(prefix_embeddings.shape[1]),
                    "token_logit_slice_start": int(prefix_embeddings.shape[1]),
                    "attention_query_offset": int(prefix_embeddings.shape[1]),
                    "diagnostic_layers": [max(0, int(getattr(self.model.config, "num_hidden_layers", 1)) - 1)],
                }
            )
            return model_kwargs, metadata
        return self._prepare_deep_prefixed_hf_inputs(encoded, layer_prefix_hidden_by_layer or {})

    def _decoder_layers(self) -> list[nn.Module]:
        if self.model is None:
            raise RuntimeError("Real backbone model is not initialized.")
        decoder = getattr(self.model, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise NotImplementedError(
                "layer_prefix_hidden_by_layer requires a decoder backbone exposing model.layers."
            )
        return list(layers)

    def enable_receiver_micro_lora(
        self,
        *,
        layer_indices: Iterable[int],
        target_modules: Iterable[str],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        if self.load_mode != "hf_causal_lm":
            raise ValueError("Receiver micro-LoRA requires backbone.load_mode='hf_causal_lm'.")
        normalized_layers = tuple(sorted({int(layer_index) for layer_index in layer_indices}))
        normalized_targets = tuple(dict.fromkeys(str(name) for name in target_modules))
        if not normalized_layers:
            raise ValueError("Receiver micro-LoRA requires at least one target layer.")
        if not normalized_targets:
            raise ValueError("Receiver micro-LoRA requires at least one target projection.")
        if rank <= 0:
            raise ValueError(f"Receiver micro-LoRA rank must be positive, got {rank}.")
        if alpha <= 0.0:
            raise ValueError(f"Receiver micro-LoRA alpha must be positive, got {alpha}.")
        decoder_layers = self._decoder_layers()
        for layer_index in normalized_layers:
            if layer_index < 0 or layer_index >= len(decoder_layers):
                raise ValueError(
                    f"Receiver micro-LoRA layer index {layer_index} is out of range for "
                    f"{len(decoder_layers)} decoder layers."
                )
            self_attn = getattr(decoder_layers[layer_index], "self_attn", None)
            if self_attn is None:
                raise NotImplementedError(
                    "Receiver micro-LoRA requires decoder layers exposing self_attn."
                )
            for module_name in normalized_targets:
                target_module = getattr(self_attn, module_name, None)
                if target_module is None:
                    raise ValueError(
                        f"Receiver micro-LoRA target '{module_name}' is missing on decoder layer {layer_index}."
                    )
                target_path = f"layers.{layer_index}.self_attn.{module_name}"
                if isinstance(target_module, MicroLoRALinear):
                    adapter = target_module
                    if (
                        adapter.rank != int(rank)
                        or abs(adapter.alpha - float(alpha)) > 1e-8
                        or type(adapter.dropout) is not (nn.Dropout if dropout > 0.0 else nn.Identity)
                    ):
                        raise ValueError(
                            f"Receiver micro-LoRA target '{target_path}' is already configured "
                            "with a different adapter shape."
                        )
                else:
                    if not isinstance(target_module, nn.Linear):
                        raise ValueError(
                            f"Receiver micro-LoRA target '{target_path}' must be nn.Linear, "
                            f"got {type(target_module).__name__}."
                        )
                    adapter = MicroLoRALinear(
                        target_module,
                        rank=int(rank),
                        alpha=float(alpha),
                        dropout=float(dropout),
                    )
                    adapter.train(bool(self.model.training))
                    setattr(self_attn, module_name, adapter)
                self._receiver_lora_targets[target_path] = adapter
        self._receiver_lora_config = {
            "enabled": True,
            "target_layers": list(normalized_layers),
            "target_modules": list(normalized_targets),
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
        }

    def receiver_lora_enabled(self) -> bool:
        return bool(self._receiver_lora_targets)

    def receiver_lora_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for target_path in sorted(self._receiver_lora_targets):
            parameters.extend(self._receiver_lora_targets[target_path].lora_parameters())
        return parameters

    def set_receiver_lora_trainable(self, enabled: bool) -> None:
        for parameter in self.receiver_lora_parameters():
            parameter.requires_grad_(enabled)

    def receiver_lora_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.receiver_lora_parameters()))

    def receiver_lora_metadata(self) -> dict[str, Any]:
        metadata = dict(self._receiver_lora_config)
        metadata["target_paths"] = sorted(self._receiver_lora_targets)
        metadata["trainable_params"] = self.receiver_lora_parameter_count()
        return metadata

    def receiver_lora_state_dict(self) -> dict[str, dict[str, torch.Tensor]] | None:
        if not self.receiver_lora_enabled():
            return None
        return {
            target_path: adapter.lora_state_dict()
            for target_path, adapter in sorted(self._receiver_lora_targets.items())
        }

    def validate_receiver_lora_state_dict(
        self,
        state_dict: dict[str, dict[str, torch.Tensor]],
        *,
        checkpoint_path: str,
    ) -> None:
        if not self.receiver_lora_enabled():
            if state_dict:
                raise ValueError(
                    f"Receiver micro-LoRA checkpoint {checkpoint_path} includes adapter weights, "
                    "but the current backbone has receiver micro-LoRA disabled."
                )
            return
        expected_paths = set(self._receiver_lora_targets)
        actual_paths = set(state_dict)
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        if missing:
            raise ValueError(
                f"Receiver micro-LoRA checkpoint {checkpoint_path} is missing targets: {missing}."
            )
        if unexpected:
            raise ValueError(
                f"Receiver micro-LoRA checkpoint {checkpoint_path} includes unexpected targets: {unexpected}."
            )
        for target_path, adapter in self._receiver_lora_targets.items():
            adapter.validate_lora_state_dict(
                state_dict[target_path],
                module_path=target_path,
                checkpoint_path=checkpoint_path,
            )

    def load_receiver_lora_state_dict(
        self,
        state_dict: dict[str, dict[str, torch.Tensor]],
        *,
        checkpoint_path: str,
    ) -> None:
        self.validate_receiver_lora_state_dict(
            state_dict,
            checkpoint_path=checkpoint_path,
        )
        for target_path, adapter in self._receiver_lora_targets.items():
            adapter.load_lora_state_dict(
                state_dict[target_path],
                module_path=target_path,
                checkpoint_path=checkpoint_path,
            )

    def enable_reader_cross_attention(
        self,
        *,
        layer_indices: Iterable[int],
        num_heads: int,
        ff_hidden_dim: int,
        gate_init: float,
    ) -> None:
        if self.load_mode != "hf_causal_lm":
            raise ValueError("Reader cross-attention requires backbone.load_mode='hf_causal_lm'.")
        normalized_layers = tuple(sorted({int(layer_index) for layer_index in layer_indices}))
        if not normalized_layers:
            raise ValueError("Reader cross-attention requires at least one target layer.")
        decoder_layers = self._decoder_layers()
        for layer_index in normalized_layers:
            if layer_index < 0 or layer_index >= len(decoder_layers):
                raise ValueError(
                    f"Reader cross-attention layer index {layer_index} is out of range for "
                    f"{len(decoder_layers)} decoder layers."
                )
            adapter_key = f"layer_{layer_index}"
            if adapter_key not in self._reader_cross_attn_targets:
                adapter = ReaderCrossAttentionAdapter(
                    hidden_size=self.hidden_size,
                    num_heads=int(num_heads),
                    ff_hidden_dim=int(ff_hidden_dim),
                    gate_init=float(gate_init),
                )
                adapter.to(device=self.device, dtype=self._resolve_torch_dtype())
                adapter.train(bool(self.model.training))
                self._reader_cross_attn_targets[adapter_key] = adapter
        self._reader_cross_attn_config = {
            "enabled": True,
            "target_layers": list(normalized_layers),
            "num_heads": int(num_heads),
            "ff_hidden_dim": int(ff_hidden_dim),
            "gate_init": float(gate_init),
        }

    def reader_cross_attention_enabled(self) -> bool:
        return bool(self._reader_cross_attn_targets)

    def reader_cross_attn_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for adapter_key in sorted(self._reader_cross_attn_targets):
            parameters.extend(self._reader_cross_attn_targets[adapter_key].parameters())
        return parameters

    def set_reader_cross_attn_trainable(self, enabled: bool) -> None:
        for parameter in self.reader_cross_attn_parameters():
            parameter.requires_grad_(enabled)

    def reader_cross_attn_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.reader_cross_attn_parameters()))

    def reader_cross_attn_metadata(self) -> dict[str, Any]:
        metadata = dict(self._reader_cross_attn_config)
        metadata["target_paths"] = sorted(self._reader_cross_attn_targets)
        metadata["trainable_params"] = self.reader_cross_attn_parameter_count()
        return metadata

    def reader_cross_attn_state_dict(self) -> dict[str, dict[str, torch.Tensor]] | None:
        if not self.reader_cross_attention_enabled():
            return None
        return {
            adapter_key: {
                state_key: value.detach().cpu()
                for state_key, value in adapter.state_dict().items()
            }
            for adapter_key, adapter in sorted(self._reader_cross_attn_targets.items())
        }

    def validate_reader_cross_attn_state_dict(
        self,
        state_dict: dict[str, dict[str, torch.Tensor]],
        *,
        checkpoint_path: str,
    ) -> None:
        if not self.reader_cross_attention_enabled():
            if state_dict:
                raise ValueError(
                    f"Reader cross-attention checkpoint {checkpoint_path} includes adapter weights, "
                    "but the current backbone has reader cross-attention disabled."
                )
            return
        expected_paths = set(self._reader_cross_attn_targets)
        actual_paths = set(state_dict)
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        if missing:
            raise ValueError(
                f"Reader cross-attention checkpoint {checkpoint_path} is missing targets: {missing}."
            )
        if unexpected:
            raise ValueError(
                f"Reader cross-attention checkpoint {checkpoint_path} includes unexpected targets: {unexpected}."
            )
        for adapter_key, adapter in self._reader_cross_attn_targets.items():
            expected_state = adapter.state_dict()
            candidate_state = state_dict[adapter_key]
            for state_key, expected_value in expected_state.items():
                if state_key not in candidate_state:
                    raise ValueError(
                        f"Reader cross-attention checkpoint {checkpoint_path} is missing "
                        f"'{adapter_key}.{state_key}'."
                    )
                actual_value = candidate_state[state_key]
                if tuple(actual_value.shape) != tuple(expected_value.shape):
                    raise ValueError(
                        f"Reader cross-attention checkpoint {checkpoint_path} has incompatible "
                        f"'{adapter_key}.{state_key}': expected {tuple(expected_value.shape)}, "
                        f"got {tuple(actual_value.shape)}."
                    )

    def load_reader_cross_attn_state_dict(
        self,
        state_dict: dict[str, dict[str, torch.Tensor]],
        *,
        checkpoint_path: str,
    ) -> None:
        self.validate_reader_cross_attn_state_dict(
            state_dict,
            checkpoint_path=checkpoint_path,
        )
        for adapter_key, adapter in self._reader_cross_attn_targets.items():
            normalized_state = {
                state_key: value.to(
                    device=next(adapter.parameters()).device,
                    dtype=next(adapter.parameters()).dtype if state_key != "gate" else adapter.gate.dtype,
                )
                for state_key, value in state_dict[adapter_key].items()
            }
            adapter.load_state_dict(normalized_state)

    @contextmanager
    def _activate_reader_cross_attention(self, memory_tokens: torch.Tensor | None):
        if not self.reader_cross_attention_enabled() or memory_tokens is None:
            yield
            return
        handles: list[Any] = []
        decoder_layers = self._decoder_layers()
        self._active_reader_cross_attn_memory = memory_tokens.to(
            device=self.device,
            dtype=self._resolve_torch_dtype(),
        )
        self._reader_cross_attn_last_diagnostics = {}

        def make_hook(*, adapter_key: str, adapter: ReaderCrossAttentionAdapter):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
                hidden_states = output[0] if isinstance(output, tuple) else output
                updated_hidden = adapter(hidden_states, self._active_reader_cross_attn_memory)
                self._reader_cross_attn_last_diagnostics[adapter_key] = adapter.diagnostics()
                if isinstance(output, tuple):
                    return (updated_hidden, *output[1:])
                return updated_hidden

            return hook

        try:
            for adapter_key, adapter in sorted(self._reader_cross_attn_targets.items()):
                layer_index = int(adapter_key.split("_", 1)[1])
                handles.append(
                    decoder_layers[layer_index].register_forward_hook(
                        make_hook(adapter_key=adapter_key, adapter=adapter)
                    )
                )
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._active_reader_cross_attn_memory = None

    def reader_cross_attn_diagnostics(self) -> dict[str, dict[str, float]]:
        by_layer = {
            layer_key.removeprefix("layer_"): dict(payload)
            for layer_key, payload in self._reader_cross_attn_last_diagnostics.items()
        }
        if not by_layer:
            return {
                "by_layer": {},
                "cross_attn_gate_open_fraction": 0.0,
                "memory_token_attention_mass_mean": 0.0,
                "memory_token_attention_top_mass_mean": 0.0,
            }
        return {
            "by_layer": by_layer,
            "cross_attn_gate_open_fraction": float(
                sum(payload.get("gate_open_fraction", 0.0) for payload in by_layer.values())
                / max(1, len(by_layer))
            ),
            "memory_token_attention_mass_mean": float(
                sum(payload.get("memory_token_attention_mass_mean", 0.0) for payload in by_layer.values())
                / max(1, len(by_layer))
            ),
            "memory_token_attention_entropy_mean": float(
                sum(payload.get("memory_token_attention_entropy_mean", 0.0) for payload in by_layer.values())
                / max(1, len(by_layer))
            ),
            "memory_token_attention_top_mass_mean": float(
                sum(payload.get("memory_token_attention_top_mass_mean", 0.0) for payload in by_layer.values())
                / max(1, len(by_layer))
            ),
        }

    def _expand_layer_prefix_hidden(
        self,
        *,
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor],
        batch_size: int,
    ) -> tuple[dict[int, torch.Tensor], int]:
        normalized: dict[int, torch.Tensor] = {}
        prefix_length: int | None = None
        torch_dtype = self._resolve_torch_dtype()
        num_layers = len(self._decoder_layers())
        for raw_layer_index, hidden_states in sorted(layer_prefix_hidden_by_layer.items()):
            layer_index = int(raw_layer_index)
            if layer_index < 0 or layer_index >= num_layers:
                raise ValueError(
                    f"layer_prefix_hidden_by_layer includes layer {layer_index}, "
                    f"but backbone exposes layers [0, {num_layers - 1}]."
                )
            if hidden_states.ndim != 3:
                raise ValueError(
                    "Each layer_prefix_hidden_by_layer tensor must have shape [batch, prefix_tokens, hidden_size]."
                )
            if hidden_states.shape[0] == 1 and batch_size > 1:
                hidden_states = hidden_states.expand(batch_size, -1, -1)
            if hidden_states.shape[0] != batch_size:
                raise ValueError(
                    "layer_prefix_hidden_by_layer batch dimension must be 1 or match the number of candidate sequences."
                )
            if hidden_states.shape[2] != self.hidden_size:
                raise ValueError(
                    f"layer_prefix_hidden_by_layer hidden size must match backbone hidden size {self.hidden_size}, "
                    f"got {hidden_states.shape[2]}."
                )
            if prefix_length is None:
                prefix_length = int(hidden_states.shape[1])
            elif int(hidden_states.shape[1]) != prefix_length:
                raise ValueError("All layer_prefix_hidden_by_layer tensors must share the same prefix token count.")
            normalized[layer_index] = hidden_states.to(device=self.device, dtype=torch_dtype)
        if not normalized:
            raise ValueError("layer_prefix_hidden_by_layer must not be empty when deep prefix injection is enabled.")
        return normalized, int(prefix_length or 0)

    def _prepare_deep_prefixed_hf_inputs(
        self,
        encoded: dict[str, torch.Tensor],
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Real backbone model is not initialized.")
        batch_size, query_length = encoded["input_ids"].shape
        normalized_layers, prefix_length = self._expand_layer_prefix_hidden(
            layer_prefix_hidden_by_layer=layer_prefix_hidden_by_layer,
            batch_size=batch_size,
        )
        decoder_layers = self._decoder_layers()
        base_model = getattr(self.model, "model", None)
        rotary_emb = getattr(base_model, "rotary_emb", None)
        if rotary_emb is None:
            raise NotImplementedError(
                "layer_prefix_hidden_by_layer requires a backbone exposing model.rotary_emb."
            )
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        prefix_position_ids = torch.arange(prefix_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        prefix_cache = DynamicCache()
        zero_hidden_template = torch.zeros(
            batch_size,
            prefix_length,
            self.hidden_size,
            device=self.device,
            dtype=self._resolve_torch_dtype(),
        )
        for layer_index, decoder_layer in enumerate(decoder_layers):
            hidden_states = normalized_layers.get(layer_index, zero_hidden_template)
            attn_input = decoder_layer.input_layernorm(hidden_states)
            key_states = decoder_layer.self_attn.k_proj(attn_input)
            value_states = decoder_layer.self_attn.v_proj(attn_input)
            head_dim = int(decoder_layer.self_attn.head_dim)
            key_head_count = max(1, int(key_states.shape[-1] // head_dim))
            value_head_count = max(1, int(value_states.shape[-1] // head_dim))
            key_states = key_states.view(batch_size, prefix_length, key_head_count, head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, prefix_length, value_head_count, head_dim).transpose(1, 2)
            cos, sin = rotary_emb(hidden_states, prefix_position_ids)
            dummy_queries = torch.zeros_like(key_states)
            _, key_states = apply_rotary_pos_emb(dummy_queries, key_states, cos, sin)
            prefix_cache.update(key_states, value_states, layer_index)
        prefix_mask = torch.ones(
            batch_size,
            prefix_length,
            dtype=encoded["attention_mask"].dtype,
            device=self.device,
        )
        model_kwargs: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": torch.cat([prefix_mask, encoded["attention_mask"]], dim=1),
            "past_key_values": prefix_cache,
            "cache_position": torch.arange(
                prefix_length,
                prefix_length + query_length,
                device=self.device,
            ),
            "use_cache": True,
        }
        metadata = {
            "prefix_length": prefix_length,
            "score_input_ids": encoded["input_ids"],
            "score_attention_mask": encoded["attention_mask"],
            "token_logit_slice_start": 0,
            "attention_query_offset": 0,
            "diagnostic_layers": sorted(int(layer_index) for layer_index in normalized_layers),
        }
        return model_kwargs, metadata

    def summarize_layer_prefix_projection(
        self,
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor],
        *,
        batch_size: int = 1,
    ) -> dict[str, dict[str, float]]:
        if self.load_mode != "hf_causal_lm":
            return {
                "layer_hidden_l2_by_layer": {},
                "layer_key_l2_by_layer": {},
                "layer_value_l2_by_layer": {},
            }
        if self.model is None:
            raise RuntimeError("Real backbone model is not initialized.")
        normalized_layers, _ = self._expand_layer_prefix_hidden(
            layer_prefix_hidden_by_layer=layer_prefix_hidden_by_layer,
            batch_size=batch_size,
        )
        decoder_layers = self._decoder_layers()
        base_model = getattr(self.model, "model", None)
        rotary_emb = getattr(base_model, "rotary_emb", None)
        if rotary_emb is None:
            raise NotImplementedError(
                "layer_prefix_hidden_by_layer requires a backbone exposing model.rotary_emb."
            )
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        layer_hidden_l2_by_layer: dict[str, float] = {}
        layer_key_l2_by_layer: dict[str, float] = {}
        layer_value_l2_by_layer: dict[str, float] = {}
        for layer_index, hidden_states in sorted(normalized_layers.items()):
            decoder_layer = decoder_layers[layer_index]
            prefix_length = hidden_states.shape[1]
            prefix_position_ids = torch.arange(prefix_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
            attn_input = decoder_layer.input_layernorm(hidden_states)
            key_states = decoder_layer.self_attn.k_proj(attn_input)
            value_states = decoder_layer.self_attn.v_proj(attn_input)
            head_dim = int(decoder_layer.self_attn.head_dim)
            key_head_count = max(1, int(key_states.shape[-1] // head_dim))
            value_head_count = max(1, int(value_states.shape[-1] // head_dim))
            key_states = key_states.view(batch_size, prefix_length, key_head_count, head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, prefix_length, value_head_count, head_dim).transpose(1, 2)
            cos, sin = rotary_emb(hidden_states, prefix_position_ids)
            dummy_queries = torch.zeros_like(key_states)
            _, key_states = apply_rotary_pos_emb(dummy_queries, key_states, cos, sin)
            layer_key = str(layer_index)
            layer_hidden_l2_by_layer[layer_key] = float(
                hidden_states.detach().to(dtype=torch.float32, device="cpu").norm().item()
            )
            layer_key_l2_by_layer[layer_key] = float(
                key_states.detach().to(dtype=torch.float32, device="cpu").norm().item()
            )
            layer_value_l2_by_layer[layer_key] = float(
                value_states.detach().to(dtype=torch.float32, device="cpu").norm().item()
            )
        return {
            "layer_hidden_l2_by_layer": layer_hidden_l2_by_layer,
            "layer_key_l2_by_layer": layer_key_l2_by_layer,
            "layer_value_l2_by_layer": layer_value_l2_by_layer,
        }

    def score_continuations(
        self,
        prompt: str,
        candidate_texts: list[str],
        *,
        memory_tokens: torch.Tensor | None = None,
        memory_consumer_mode: str = "prepend_block",
        prefix_embeddings: torch.Tensor | None = None,
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if self.load_mode != "hf_causal_lm":
            raise RuntimeError("score_continuations is only available for load_mode='hf_causal_lm'.")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Real backbone is not initialized.")
        prompt_with_sep = prompt if prompt.endswith((" ", "\n", "\t")) else f"{prompt} "
        full_texts = [f"{prompt_with_sep}{candidate_text}" for candidate_text in candidate_texts]
        if memory_tokens is not None and memory_consumer_mode not in {"prepend_block", "reader_cross_attn"}:
            raise ValueError(
                f"Unsupported memory_consumer_mode={memory_consumer_mode}. "
                "Expected one of prepend_block, reader_cross_attn."
            )
        if memory_tokens is not None and memory_consumer_mode == "reader_cross_attn":
            model_kwargs, metadata = self._prepare_prefixed_hf_inputs(
                full_texts,
                None,
                None,
                apply_chat_template=self.use_chat_template,
            )
        else:
            model_kwargs, metadata = self._prepare_prefixed_hf_inputs(
                full_texts,
                prefix_embeddings,
                layer_prefix_hidden_by_layer,
                memory_tokens=memory_tokens,
                apply_chat_template=self.use_chat_template,
            )
        prefix_length = int(metadata["prefix_length"])
        prompt_length = len(self.tokenizer(prompt_with_sep, add_special_tokens=True)["input_ids"])
        model_context = (
            torch.inference_mode
            if prefix_embeddings is None and layer_prefix_hidden_by_layer is None and memory_tokens is None
            else nullcontext
        )
        model_kwargs = dict(model_kwargs)
        if "use_cache" not in model_kwargs:
            model_kwargs["use_cache"] = False
        if return_diagnostics:
            model_kwargs["output_attentions"] = True
        attention_context = nullcontext()
        if return_diagnostics:
            attention_context = self._temporary_attention_implementation("eager")
        with attention_context:
            with self._activate_reader_cross_attention(
                memory_tokens if memory_consumer_mode == "reader_cross_attn" else None
            ):
                with model_context():
                    outputs = self.model(**model_kwargs)
        logits = outputs.logits
        token_logit_slice_start = int(metadata["token_logit_slice_start"])
        log_probs = torch.log_softmax(
            logits[:, token_logit_slice_start:-1, :].to(dtype=torch.float32),
            dim=-1,
        )
        labels_source = metadata["score_input_ids"]
        labels = labels_source[:, 1:]
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        scores: list[torch.Tensor] = []
        prefix_attention_masses: list[float] = []
        attention_query_offset = int(metadata["attention_query_offset"])
        diagnostic_layers = [int(layer_index) for layer_index in metadata["diagnostic_layers"]]
        attention_by_layer: dict[int, list[float]] = {layer_index: [] for layer_index in diagnostic_layers}
        sequence_lengths = metadata["score_attention_mask"].sum(dim=1).tolist()
        for row_index, sequence_length_value in enumerate(sequence_lengths):
            sequence_length = int(sequence_length_value)
            start_index = max(0, prompt_length - 1)
            end_index = max(start_index, sequence_length - 1)
            scores.append(token_log_probs[row_index, start_index:end_index].sum())
            if return_diagnostics:
                if prefix_length <= 0:
                    prefix_attention_masses.append(0.0)
                else:
                    if outputs.attentions is None:
                        raise RuntimeError("Attention diagnostics requested but model did not return attentions.")
                    attention_start = max(0, attention_query_offset + prompt_length - 1)
                    attention_end = max(attention_start, attention_query_offset + sequence_length - 1)
                    layer_masses: list[float] = []
                    for layer_index in diagnostic_layers:
                        layer_attention = outputs.attentions[layer_index][row_index].to(dtype=torch.float32)
                        candidate_query_attention = layer_attention[
                            :,
                            attention_start:attention_end,
                            :prefix_length,
                        ]
                        if candidate_query_attention.numel() == 0:
                            layer_mass = 0.0
                        else:
                            layer_mass = float(candidate_query_attention.mean().item())
                        attention_by_layer[layer_index].append(layer_mass)
                        layer_masses.append(layer_mass)
                    prefix_attention_masses.append(
                        float(sum(layer_masses) / max(1, len(layer_masses)))
                    )
        score_tensor = torch.stack(scores)
        if not return_diagnostics:
            return score_tensor
        diagnostics: dict[str, object] = {
            "prefix_length": int(prefix_length),
            "prefix_attention_mass_by_candidate": prefix_attention_masses,
            "prompt_length": int(prompt_length),
            "prefix_attention_mass_by_candidate_by_layer": attention_by_layer,
            "diagnostic_layers": diagnostic_layers,
            "memory_consumer_mode": str(memory_consumer_mode),
        }
        if memory_tokens is not None and memory_consumer_mode == "reader_cross_attn":
            reader_cross_attn_by_layer = self.reader_cross_attn_diagnostics()
            by_layer = reader_cross_attn_by_layer.get("by_layer", {})
            top_mass_by_layer = {
                str(layer_key): float(payload.get("memory_token_attention_top_mass_mean", 0.0))
                for layer_key, payload in by_layer.items()
            }
            gate_values = [
                float(payload.get("gate_open_fraction", 0.0))
                for payload in by_layer.values()
            ]
            diagnostics["by_layer"] = by_layer
            diagnostics["cross_attn_gate_open_fraction"] = float(
                reader_cross_attn_by_layer.get("cross_attn_gate_open_fraction", 0.0)
            )
            if diagnostics["cross_attn_gate_open_fraction"] == 0.0 and gate_values:
                diagnostics["cross_attn_gate_open_fraction"] = float(sum(gate_values) / len(gate_values))
            diagnostics["memory_token_attention_mass_mean"] = float(
                reader_cross_attn_by_layer.get("memory_token_attention_top_mass_mean", 0.0)
            )
            if diagnostics["memory_token_attention_mass_mean"] == 0.0 and top_mass_by_layer:
                diagnostics["memory_token_attention_mass_mean"] = float(
                    sum(top_mass_by_layer.values()) / len(top_mass_by_layer)
                )
            diagnostics["memory_token_attention_mass_mean_by_layer"] = top_mass_by_layer
        return score_tensor, diagnostics

    def generate(
        self,
        prompts: Iterable[str],
        memory_tokens: torch.Tensor | None = None,
        *,
        memory_consumer_mode: str = "prepend_block",
        prefix_embeddings: torch.Tensor | None = None,
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None = None,
    ) -> list[str]:
        if self.load_mode == "hf_causal_lm":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Real backbone is not initialized.")
            prompt_list = list(prompts)
            if prefix_embeddings is not None and memory_tokens is not None:
                raise ValueError("memory_tokens and prefix_embeddings are mutually exclusive.")
            if layer_prefix_hidden_by_layer is not None and memory_tokens is not None:
                raise ValueError("memory_tokens and layer_prefix_hidden_by_layer are mutually exclusive.")
            if memory_tokens is not None and memory_consumer_mode not in {"prepend_block", "reader_cross_attn"}:
                raise ValueError(
                    f"Unsupported memory_consumer_mode={memory_consumer_mode}. "
                    "Expected one of prepend_block, reader_cross_attn."
                )
            model_kwargs: dict[str, Any]
            prompt_attention_mask: torch.Tensor
            if memory_tokens is not None and memory_consumer_mode == "reader_cross_attn":
                encoded = self._prepare_hf_inputs(prompt_list, apply_chat_template=self.use_chat_template)
                model_kwargs = dict(encoded)
                prompt_attention_mask = encoded["attention_mask"]
            elif prefix_embeddings is not None or layer_prefix_hidden_by_layer is not None or memory_tokens is not None:
                model_kwargs, metadata = self._prepare_prefixed_hf_inputs(
                    prompt_list,
                    prefix_embeddings,
                    layer_prefix_hidden_by_layer,
                    memory_tokens=memory_tokens,
                    apply_chat_template=self.use_chat_template,
                )
                prompt_attention_mask = metadata["score_attention_mask"]
                if "input_ids" not in model_kwargs:
                    model_kwargs["input_ids"] = metadata["score_input_ids"]
            else:
                encoded = self._prepare_hf_inputs(prompt_list, apply_chat_template=self.use_chat_template)
                model_kwargs = dict(encoded)
                prompt_attention_mask = encoded["attention_mask"]
            with self._activate_reader_cross_attention(
                memory_tokens if memory_consumer_mode == "reader_cross_attn" else None
            ):
                with torch.inference_mode():
                    generated = self.model.generate(
                        **model_kwargs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            completions = []
            prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()
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
