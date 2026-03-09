from __future__ import annotations

from contextlib import nullcontext
import hashlib
import os
from typing import Any, Iterable

import torch
from torch import nn

from memtotal.utils.repro import validate_backbone_name


class ReceiverMicroLoRALinear(nn.Module):
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
            raise ValueError(f"Receiver micro-LoRA rank must be positive, got {rank}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"Receiver micro-LoRA dropout must be in [0, 1), got {dropout}.")
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("Receiver micro-LoRA can wrap only nn.Linear modules.")
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
        self.tokenizer = None
        self.model = None
        self._receiver_lora_targets: dict[str, ReceiverMicroLoRALinear] = {}
        self._receiver_lora_config: dict[str, Any] = {
            "enabled": False,
            "target_layers": [],
            "target_modules": [],
            "rank": 0,
            "alpha": 0.0,
            "dropout": 0.0,
        }
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

    def _prepare_prefixed_hf_inputs(
        self,
        texts: list[str],
        prefix_embeddings: torch.Tensor | None,
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if prefix_embeddings is not None and layer_prefix_hidden_by_layer is not None:
            raise ValueError(
                "prefix_embeddings and layer_prefix_hidden_by_layer are mutually exclusive."
            )
        encoded = self._prepare_hf_inputs(texts)
        metadata = {
            "prefix_length": 0,
            "score_input_ids": encoded["input_ids"],
            "score_attention_mask": encoded["attention_mask"],
            "token_logit_slice_start": 0,
            "attention_query_offset": 0,
            "diagnostic_layers": [],
        }
        if prefix_embeddings is None and layer_prefix_hidden_by_layer is None:
            return encoded, metadata
        if self.model is None:
            raise RuntimeError("Real backbone model is not initialized.")
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
                if isinstance(target_module, ReceiverMicroLoRALinear):
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
                    adapter = ReceiverMicroLoRALinear(
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
        model_kwargs, metadata = self._prepare_prefixed_hf_inputs(
            full_texts,
            prefix_embeddings,
            layer_prefix_hidden_by_layer,
        )
        prefix_length = int(metadata["prefix_length"])
        prompt_length = len(self.tokenizer(prompt_with_sep, add_special_tokens=True)["input_ids"])
        model_context = (
            torch.inference_mode
            if prefix_embeddings is None and layer_prefix_hidden_by_layer is None
            else nullcontext
        )
        model_kwargs = dict(model_kwargs)
        if "use_cache" not in model_kwargs:
            model_kwargs["use_cache"] = False
        if return_diagnostics:
            model_kwargs["output_attentions"] = True
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
        }
        return score_tensor, diagnostics

    def generate(
        self,
        prompts: Iterable[str],
        memory_tokens: torch.Tensor | None = None,
        *,
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
            model_kwargs: dict[str, Any]
            prompt_attention_mask: torch.Tensor
            if prefix_embeddings is not None or layer_prefix_hidden_by_layer is not None:
                model_kwargs, metadata = self._prepare_prefixed_hf_inputs(
                    prompt_list,
                    prefix_embeddings,
                    layer_prefix_hidden_by_layer,
                )
                prompt_attention_mask = metadata["score_attention_mask"]
                if "input_ids" not in model_kwargs:
                    model_kwargs["input_ids"] = metadata["score_input_ids"]
            else:
                encoded = self._prepare_hf_inputs(prompt_list)
                model_kwargs = dict(encoded)
                prompt_attention_mask = encoded["attention_mask"]
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
