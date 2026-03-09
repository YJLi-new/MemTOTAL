from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from memtotal.models.backbone import MicroLoRALinear


def _ensure_rank(name: str, tensor: torch.Tensor, expected_rank: int) -> None:
    if tensor.ndim != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got rank {tensor.ndim}.")


class ManagedMemoryModule(nn.Module):
    def freeze(self) -> "ManagedMemoryModule":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def unfreeze(self) -> "ManagedMemoryModule":
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self

    def save_to(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination)
        return destination

    def load_from(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "ManagedMemoryModule":
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)
        return self


class _WriterWeaverConditioningBlock(nn.Module):
    def __init__(self, *, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.context_cross_attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.support_cross_attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.context_attn_norm = nn.LayerNorm(int(embed_dim))
        self.support_attn_norm = nn.LayerNorm(int(embed_dim))

    def forward(
        self,
        slots: torch.Tensor,
        *,
        context_states: torch.Tensor | None,
        support_states: torch.Tensor | None,
        stimulus_mode: str,
        context_key_padding_mask: torch.Tensor | None,
        support_key_padding_mask: torch.Tensor | None,
        context_query_residual_scale: float,
        support_query_residual_scale: float,
        collect_support_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        current_slots = slots
        if context_states is not None and stimulus_mode in {"context_only", "support_and_context"}:
            context_attended, _ = self.context_cross_attn(
                query=current_slots,
                key=context_states,
                value=context_states,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            current_slots = self.context_attn_norm(
                context_attended + (float(context_query_residual_scale) * current_slots)
            )
        support_attention_weights = None
        if support_states is not None and stimulus_mode in {"support_only", "support_and_context"}:
            support_attended, support_attention_weights = self.support_cross_attn(
                query=current_slots,
                key=support_states,
                value=support_states,
                key_padding_mask=support_key_padding_mask,
                need_weights=bool(collect_support_diagnostics),
                average_attn_weights=False,
            )
            current_slots = self.support_attn_norm(
                support_attended + (float(support_query_residual_scale) * current_slots)
            )
        return current_slots, support_attention_weights


class WriterAuxProjectionHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        resolved_hidden_dim = int(hidden_dim or max(int(projection_dim), int(input_dim)))
        self.proj = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), resolved_hidden_dim),
            nn.GELU(),
            nn.Linear(resolved_hidden_dim, int(projection_dim)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.proj(inputs)
        return F.normalize(projected, dim=-1)


class SourceStubMemory(ManagedMemoryModule):
    def __init__(
        self,
        *,
        embed_dim: int,
        memory_slots: int,
        init_scale: float = 0.02,
        slot_max_norm: float | None = None,
        total_max_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.memory_slots = int(memory_slots)
        self.init_scale = float(init_scale)
        self.slot_max_norm = None if slot_max_norm is None else float(slot_max_norm)
        self.total_max_norm = None if total_max_norm is None else float(total_max_norm)
        self.slot_embeddings = nn.Parameter(
            torch.randn(self.memory_slots, self.embed_dim) * self.init_scale
        )

    def _apply_norm_caps(self, slots: torch.Tensor) -> torch.Tensor:
        capped = slots
        if self.slot_max_norm is not None and self.slot_max_norm > 0.0:
            slot_norms = capped.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            slot_scales = torch.clamp(self.slot_max_norm / slot_norms, max=1.0)
            capped = capped * slot_scales
        if self.total_max_norm is not None and self.total_max_norm > 0.0:
            total_norms = capped.flatten(start_dim=1).norm(dim=1, keepdim=True).clamp_min(1e-6)
            total_scales = torch.clamp(self.total_max_norm / total_norms, max=1.0).view(-1, 1, 1)
            capped = capped * total_scales
        return capped

    def _fit_anchor_tokens(self, anchor_tokens: torch.Tensor) -> torch.Tensor:
        if anchor_tokens.ndim == 3:
            anchor_tokens = anchor_tokens.mean(dim=0)
        if anchor_tokens.ndim == 1:
            anchor_tokens = anchor_tokens.unsqueeze(0)
        if anchor_tokens.ndim != 2:
            raise ValueError(
                "SourceStubMemory anchor tokens must have shape [tokens, hidden_size] "
                "or [batch, tokens, hidden_size]."
            )
        if anchor_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"SourceStubMemory expected hidden size {self.embed_dim}, got {anchor_tokens.shape[-1]}."
            )
        token_count = int(anchor_tokens.shape[0])
        if token_count == self.memory_slots:
            return anchor_tokens
        if token_count > self.memory_slots:
            return anchor_tokens[: self.memory_slots, :]
        repeats = math.ceil(self.memory_slots / max(1, token_count))
        return anchor_tokens.repeat(repeats, 1)[: self.memory_slots, :]

    def initialize_from_anchor(self, anchor_tokens: torch.Tensor) -> None:
        fitted = self._fit_anchor_tokens(anchor_tokens).to(
            device=self.slot_embeddings.device,
            dtype=self.slot_embeddings.dtype,
        )
        with torch.no_grad():
            self.slot_embeddings.copy_(fitted)

    def forward(self, *, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"SourceStubMemory batch_size must be positive, got {batch_size}.")
        slots = self.slot_embeddings.unsqueeze(0).expand(int(batch_size), -1, -1)
        return self._apply_norm_caps(slots)


class WriterDeepPrefixProjector(ManagedMemoryModule):
    def __init__(
        self,
        *,
        hidden_size: int,
        prefix_tokens: int,
        layer_indices: list[int],
        bottleneck_rank: int = 32,
        slot_max_norm: float | None = None,
        total_max_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prefix_tokens = int(prefix_tokens)
        self.layer_indices = tuple(int(layer_index) for layer_index in layer_indices)
        self.bottleneck_rank = int(bottleneck_rank)
        self.slot_max_norm = None if slot_max_norm is None else float(slot_max_norm)
        self.total_max_norm = None if total_max_norm is None else float(total_max_norm)
        self.prefix_norm = nn.LayerNorm(self.hidden_size)
        self.down_proj = nn.Linear(self.hidden_size, self.bottleneck_rank, bias=False)
        self.up_proj = nn.Linear(
            self.bottleneck_rank,
            len(self.layer_indices) * self.hidden_size,
        )
        self.layer_slot_bias = nn.Parameter(
            torch.zeros(len(self.layer_indices), self.prefix_tokens, self.hidden_size)
        )
        self.layer_output_scale = nn.Parameter(torch.ones(len(self.layer_indices), 1, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up_proj.bias)
        with torch.no_grad():
            self.layer_slot_bias.zero_()
            self.layer_output_scale.fill_(1.0)

    def _fit_anchor_tokens(self, anchor_tokens: torch.Tensor) -> torch.Tensor:
        if anchor_tokens.ndim == 3:
            anchor_tokens = anchor_tokens.mean(dim=0)
        if anchor_tokens.ndim == 1:
            anchor_tokens = anchor_tokens.unsqueeze(0)
        if anchor_tokens.ndim != 2:
            raise ValueError(
                "WriterDeepPrefixProjector anchor tokens must have shape [tokens, hidden_size] "
                "or [batch, tokens, hidden_size]."
            )
        if anchor_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"WriterDeepPrefixProjector expected hidden size {self.hidden_size}, got {anchor_tokens.shape[-1]}."
            )
        token_count = int(anchor_tokens.shape[0])
        if token_count == self.prefix_tokens:
            return anchor_tokens
        if token_count > self.prefix_tokens:
            return anchor_tokens[: self.prefix_tokens, :]
        repeats = math.ceil(self.prefix_tokens / max(1, token_count))
        return anchor_tokens.repeat(repeats, 1)[: self.prefix_tokens, :]

    def _apply_anchor_stack(self, anchor_stack: torch.Tensor) -> None:
        if anchor_stack.shape != self.layer_slot_bias.shape:
            raise ValueError(
                f"WriterDeepPrefixProjector expected anchor stack shape {tuple(self.layer_slot_bias.shape)}, "
                f"got {tuple(anchor_stack.shape)}."
            )
        anchor_stack = anchor_stack.to(
            device=self.layer_slot_bias.device,
            dtype=self.layer_slot_bias.dtype,
        )
        with torch.no_grad():
            self.layer_slot_bias.copy_(anchor_stack)
            self.layer_output_scale.fill_(1.0)

    def initialize_from_semantic_anchor(self, semantic_anchor: torch.Tensor) -> None:
        fitted = self._fit_anchor_tokens(semantic_anchor)
        anchor_stack = fitted.unsqueeze(0).expand(len(self.layer_indices), -1, -1)
        self._apply_anchor_stack(anchor_stack)

    def initialize_from_hidden_state_anchor(self, hidden_state_anchor: torch.Tensor) -> None:
        fitted = self._fit_anchor_tokens(hidden_state_anchor)
        anchor_stack = fitted.unsqueeze(0).expand(len(self.layer_indices), -1, -1)
        self._apply_anchor_stack(anchor_stack)

    def initialize_from_layer_hidden_anchors(
        self,
        layer_hidden_anchor_by_layer: dict[int, torch.Tensor],
    ) -> None:
        if not layer_hidden_anchor_by_layer:
            raise ValueError("WriterDeepPrefixProjector requires non-empty layer hidden anchors.")
        fallback = None
        anchor_tensors: list[torch.Tensor] = []
        for layer_index in self.layer_indices:
            layer_anchor = layer_hidden_anchor_by_layer.get(int(layer_index))
            if layer_anchor is None:
                if fallback is None:
                    continue
                layer_anchor = fallback
            fitted = self._fit_anchor_tokens(layer_anchor)
            if fallback is None:
                fallback = fitted
            anchor_tensors.append(fitted)
        if len(anchor_tensors) != len(self.layer_indices):
            if fallback is None:
                raise ValueError("WriterDeepPrefixProjector could not resolve any usable layer anchors.")
            while len(anchor_tensors) < len(self.layer_indices):
                anchor_tensors.append(fallback.clone())
        self._apply_anchor_stack(torch.stack(anchor_tensors, dim=0))

    def initialize_from_calibration(
        self,
        *,
        mode: str,
        semantic_anchor: torch.Tensor | None = None,
        hidden_state_anchor: torch.Tensor | None = None,
        layer_hidden_anchor_by_layer: dict[int, torch.Tensor] | None = None,
    ) -> None:
        normalized_mode = str(mode)
        if normalized_mode == "random":
            self.reset_parameters()
            return
        if normalized_mode == "semantic_anchor":
            if semantic_anchor is None:
                raise ValueError("semantic_anchor init requires a semantic anchor tensor.")
            self.initialize_from_semantic_anchor(semantic_anchor)
            return
        if normalized_mode == "hidden_state_anchor":
            if hidden_state_anchor is None:
                raise ValueError("hidden_state_anchor init requires a hidden-state anchor tensor.")
            self.initialize_from_hidden_state_anchor(hidden_state_anchor)
            return
        if normalized_mode == "kv_stat_match":
            if not layer_hidden_anchor_by_layer:
                raise ValueError("kv_stat_match init requires per-layer hidden anchors.")
            self.initialize_from_layer_hidden_anchors(layer_hidden_anchor_by_layer)
            return
        raise ValueError(
            f"Unsupported WriterDeepPrefixProjector init mode={normalized_mode}. "
            "Expected one of random, kv_stat_match, semantic_anchor, hidden_state_anchor."
        )

    def _apply_norm_caps(self, stacked_hidden: torch.Tensor) -> torch.Tensor:
        clipped = stacked_hidden
        if self.slot_max_norm is not None and self.slot_max_norm > 0.0:
            slot_norms = clipped.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            slot_scales = torch.clamp(self.slot_max_norm / slot_norms, max=1.0)
            clipped = clipped * slot_scales
        if self.total_max_norm is not None and self.total_max_norm > 0.0:
            total_norms = clipped.flatten(start_dim=1).norm(dim=1, keepdim=True).clamp_min(1e-6)
            total_scales = torch.clamp(self.total_max_norm / total_norms, max=1.0).view(-1, 1, 1, 1)
            clipped = clipped * total_scales
        return clipped

    def forward(self, memory_slots: torch.Tensor) -> dict[int, torch.Tensor]:
        if memory_slots.ndim != 3:
            raise ValueError("memory_slots must have shape [batch, slots, hidden_size].")
        if memory_slots.shape[1] != self.prefix_tokens:
            raise ValueError(
                f"WriterDeepPrefixProjector expected {self.prefix_tokens} slots, got {memory_slots.shape[1]}."
            )
        batch_size, slot_count, _ = memory_slots.shape
        bottleneck = F.gelu(self.down_proj(self.prefix_norm(memory_slots)))
        expanded = self.up_proj(bottleneck)
        expanded = expanded.view(batch_size, slot_count, len(self.layer_indices), self.hidden_size)
        expanded = expanded.permute(0, 2, 1, 3).contiguous()
        expanded = expanded + self.layer_slot_bias.unsqueeze(0)
        expanded = expanded * self.layer_output_scale.unsqueeze(0)
        expanded = self._apply_norm_caps(expanded)
        return {
            int(layer_index): expanded[:, layer_offset, :, :]
            for layer_offset, layer_index in enumerate(self.layer_indices)
        }


class MemoryWriter(ManagedMemoryModule):
    def __init__(
        self,
        embed_dim: int,
        memory_slots: int,
        *,
        arch: str = "mlp",
        hidden_dim: int | None = None,
        num_heads: int = 4,
        transformer_layers: int = 1,
        dropout: float = 0.0,
        support_query_residual_scale: float = 0.0,
        output_slot_basis_scale: float = 0.0,
        slot_conditioning_mode: str = "shared_add",
        shared_state_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_slots = memory_slots
        self.arch = arch
        self.support_query_residual_scale = float(support_query_residual_scale)
        self.output_slot_basis_scale = float(output_slot_basis_scale)
        resolved_slot_conditioning_mode = str(slot_conditioning_mode)
        if resolved_slot_conditioning_mode not in {
            "shared_add",
            "shared_add_scaled",
            "slot_query_only",
            "slot_query_small_shared",
        }:
            raise ValueError(
                f"Unsupported writer slot_conditioning_mode: {resolved_slot_conditioning_mode}. "
                "Expected one of shared_add/shared_add_scaled/slot_query_only/slot_query_small_shared."
            )
        self.slot_conditioning_mode = resolved_slot_conditioning_mode
        self.shared_state_scale = float(shared_state_scale)
        hidden_dim = hidden_dim or (2 * embed_dim)
        if arch == "mlp":
            self.proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, memory_slots * embed_dim),
            )
        elif arch == "transformer":
            self.state_proj = nn.Linear(embed_dim, embed_dim)
            self.slot_embeddings = nn.Parameter(torch.randn(memory_slots, embed_dim) * 0.02)
            self.support_cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            self.output_norm = nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"Unsupported writer architecture: {arch}")

    def orthogonalize_slot_embeddings_(self) -> None:
        if self.arch != "transformer":
            return
        with torch.no_grad():
            original_mean_norm = self.slot_embeddings.norm(dim=-1).mean().clamp_min(1e-6)
            orthogonal = torch.empty_like(self.slot_embeddings)
            nn.init.orthogonal_(orthogonal)
            orthogonal = orthogonal * original_mean_norm
            self.slot_embeddings.copy_(orthogonal)

    def _pool_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            return state
        if state.ndim == 3:
            return state.mean(dim=1)
        raise ValueError(f"Unsupported state rank for writer: {state.ndim}")

    @staticmethod
    def common_mode_vector(slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim == 2:
            return slots.mean(dim=0)
        if slots.ndim == 3:
            return slots.mean(dim=1)
        raise ValueError(f"Unsupported slot rank for common_mode_vector: {slots.ndim}")

    @staticmethod
    def center_slots(slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim == 2:
            return slots - slots.mean(dim=0, keepdim=True)
        if slots.ndim == 3:
            return slots - slots.mean(dim=1, keepdim=True)
        raise ValueError(f"Unsupported slot rank for center_slots: {slots.ndim}")

    @staticmethod
    def slot_norms(slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim not in {2, 3}:
            raise ValueError(f"Unsupported slot rank for slot_norms: {slots.ndim}")
        return slots.norm(dim=-1)

    def _condition_slots(
        self,
        slots: torch.Tensor,
        shared_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.slot_conditioning_mode == "slot_query_only":
            return slots
        shared_delta = self.state_proj(shared_state).unsqueeze(1)
        # `slot_query_small_shared` is the same path as scaled shared add; the
        # distinct mode name preserves experiment attribution in configs/results.
        if self.slot_conditioning_mode in {"shared_add_scaled", "slot_query_small_shared"}:
            shared_delta = self.shared_state_scale * shared_delta
        return slots + shared_delta

    def _encode_slots(self, slot_inputs: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        encoded_slots = self.output_norm(self.encoder(slot_inputs))
        if self.output_slot_basis_scale != 0.0:
            encoded_slots = encoded_slots + (self.output_slot_basis_scale * slots)
        return encoded_slots

    def write(self, state: torch.Tensor, *, input_schema: str = "pooled_state") -> torch.Tensor:
        if input_schema not in {"pooled_state", "support_set"}:
            raise ValueError(
                f"Unsupported writer input_schema={input_schema}. "
                "Expected one of pooled_state, support_set."
            )
        if state.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Writer expected state hidden size {self.embed_dim}, got {state.shape[-1]}."
            )
        if input_schema == "pooled_state":
            pooled_state = self._pool_state(state)
            batch_size = pooled_state.shape[0]
        else:
            _ensure_rank("state", state, 3)
            if self.arch != "transformer":
                raise ValueError("MemoryWriter(input_schema='support_set') requires arch='transformer'.")
            batch_size = state.shape[0]
        if self.arch == "mlp":
            if input_schema != "pooled_state":
                raise ValueError("MLP MemoryWriter only supports input_schema='pooled_state'.")
            memory = self.proj(pooled_state)
            return memory.view(batch_size, self.memory_slots, self.embed_dim)

        slots = self.slot_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        if input_schema == "pooled_state":
            conditioned_slots = self._condition_slots(slots, pooled_state)
            return self._encode_slots(conditioned_slots, slots)

        pooled_support_state = state.mean(dim=1)
        conditioned_slots = self._condition_slots(slots, pooled_support_state)
        attended_slots, _ = self.support_cross_attn(
            query=conditioned_slots,
            key=state,
            value=state,
            need_weights=False,
        )
        support_slots = attended_slots + (self.support_query_residual_scale * conditioned_slots)
        return self._encode_slots(support_slots, slots)


class WriterWeaverHead(ManagedMemoryModule):
    def __init__(
        self,
        embed_dim: int,
        memory_slots: int,
        *,
        hidden_dim: int | None = None,
        num_heads: int = 4,
        transformer_layers: int = 1,
        conditioning_layers: int = 1,
        dropout: float = 0.0,
        context_query_residual_scale: float = 1.0,
        support_query_residual_scale: float = 1.0,
        output_slot_basis_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.memory_slots = int(memory_slots)
        self.arch = "weaver"
        self.context_query_residual_scale = float(context_query_residual_scale)
        self.support_query_residual_scale = float(support_query_residual_scale)
        self.output_slot_basis_scale = float(output_slot_basis_scale)
        self.slot_conditioning_mode = "writer_weaver"
        self.shared_state_scale = 1.0
        self.conditioning_layers = int(conditioning_layers)
        if self.conditioning_layers < 1:
            raise ValueError("WriterWeaverHead conditioning_layers must be >= 1.")
        resolved_hidden_dim = int(hidden_dim or (2 * self.embed_dim))
        self.slot_embeddings = nn.Parameter(torch.randn(self.memory_slots, self.embed_dim) * 0.02)
        self.context_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.support_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.context_attn_norm = nn.LayerNorm(self.embed_dim)
        self.support_attn_norm = nn.LayerNorm(self.embed_dim)
        self.extra_conditioning_blocks = nn.ModuleList(
            [
                _WriterWeaverConditioningBlock(
                    embed_dim=self.embed_dim,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                )
                for _ in range(self.conditioning_layers - 1)
            ]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=resolved_hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self._writer_lora_targets: dict[str, MicroLoRALinear] = {}
        self._writer_lora_config: dict[str, object] = {
            "enabled": False,
            "target_modules": [],
            "rank": 0,
            "alpha": 0.0,
            "dropout": 0.0,
        }
        self.aux_projection_head: WriterAuxProjectionHead | None = None

    @staticmethod
    def common_mode_vector(slots: torch.Tensor) -> torch.Tensor:
        return MemoryWriter.common_mode_vector(slots)

    @staticmethod
    def center_slots(slots: torch.Tensor) -> torch.Tensor:
        return MemoryWriter.center_slots(slots)

    @staticmethod
    def slot_norms(slots: torch.Tensor) -> torch.Tensor:
        return MemoryWriter.slot_norms(slots)

    def orthogonalize_slot_embeddings_(self) -> None:
        with torch.no_grad():
            original_mean_norm = self.slot_embeddings.norm(dim=-1).mean().clamp_min(1e-6)
            orthogonal = torch.empty_like(self.slot_embeddings)
            nn.init.orthogonal_(orthogonal)
            orthogonal = orthogonal * original_mean_norm
            self.slot_embeddings.copy_(orthogonal)

    def _iter_writer_lora_candidate_targets(
        self,
        target_modules: tuple[str, ...],
    ) -> list[tuple[nn.Module, str, str]]:
        targets: list[tuple[nn.Module, str, str]] = []
        requested = set(target_modules)
        if "conditioning_out_proj" in requested:
            targets.extend(
                [
                    (self.context_cross_attn, "out_proj", "context_cross_attn.out_proj"),
                    (self.support_cross_attn, "out_proj", "support_cross_attn.out_proj"),
                ]
            )
            for block_index, block in enumerate(self.extra_conditioning_blocks):
                prefix = f"extra_conditioning_blocks.{block_index}"
                targets.extend(
                    [
                        (
                            block.context_cross_attn,
                            "out_proj",
                            f"{prefix}.context_cross_attn.out_proj",
                        ),
                        (
                            block.support_cross_attn,
                            "out_proj",
                            f"{prefix}.support_cross_attn.out_proj",
                        ),
                    ]
                )
        if "encoder_self_attn_out_proj" in requested:
            for layer_index, layer in enumerate(getattr(self.encoder, "layers", [])):
                targets.append(
                    (
                        layer.self_attn,
                        "out_proj",
                        f"encoder.layers.{layer_index}.self_attn.out_proj",
                    )
                )
        if "encoder_ffn" in requested:
            for layer_index, layer in enumerate(getattr(self.encoder, "layers", [])):
                targets.extend(
                    [
                        (layer, "linear1", f"encoder.layers.{layer_index}.linear1"),
                        (layer, "linear2", f"encoder.layers.{layer_index}.linear2"),
                    ]
                )
        return targets

    def enable_writer_micro_lora(
        self,
        *,
        target_modules: tuple[str, ...],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        normalized_targets = tuple(str(module_name) for module_name in target_modules)
        unsupported = sorted(
            set(normalized_targets)
            - {"conditioning_out_proj", "encoder_self_attn_out_proj", "encoder_ffn"}
        )
        if unsupported:
            raise ValueError(
                "Unsupported WriterWeaverHead writer micro-LoRA targets: "
                f"{unsupported}. Expected only conditioning_out_proj, "
                "encoder_self_attn_out_proj, and/or encoder_ffn."
            )
        if not normalized_targets:
            raise ValueError("WriterWeaverHead writer micro-LoRA requires at least one target module.")
        if rank <= 0:
            raise ValueError(f"WriterWeaverHead writer micro-LoRA rank must be positive, got {rank}.")
        if alpha <= 0.0:
            raise ValueError(f"WriterWeaverHead writer micro-LoRA alpha must be positive, got {alpha}.")
        candidate_targets = self._iter_writer_lora_candidate_targets(normalized_targets)
        if not candidate_targets:
            raise ValueError(
                "WriterWeaverHead writer micro-LoRA found no addressable target modules for the requested targets."
            )
        for parent_module, module_name, target_path in candidate_targets:
            target_module = getattr(parent_module, module_name)
            if isinstance(target_module, MicroLoRALinear):
                adapter = target_module
                if (
                    adapter.rank != int(rank)
                    or abs(adapter.alpha - float(alpha)) > 1e-8
                    or type(adapter.dropout) is not (nn.Dropout if dropout > 0.0 else nn.Identity)
                ):
                    raise ValueError(
                        f"WriterWeaverHead writer micro-LoRA target '{target_path}' is already configured "
                        "with a different adapter shape."
                    )
            else:
                if not isinstance(target_module, nn.Linear):
                    raise TypeError(
                        f"WriterWeaverHead writer micro-LoRA target '{target_path}' must be nn.Linear, "
                        f"got {type(target_module)!r}."
                    )
                adapter = MicroLoRALinear(
                    target_module,
                    rank=int(rank),
                    alpha=float(alpha),
                    dropout=float(dropout),
                )
                adapter.train(bool(self.training))
                setattr(parent_module, module_name, adapter)
            self._writer_lora_targets[target_path] = adapter
        self._writer_lora_config = {
            "enabled": True,
            "target_modules": list(normalized_targets),
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
        }

    def writer_lora_enabled(self) -> bool:
        return bool(self._writer_lora_targets)

    def writer_lora_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for target_path in sorted(self._writer_lora_targets):
            parameters.extend(self._writer_lora_targets[target_path].lora_parameters())
        return parameters

    def writer_base_parameters(self) -> list[nn.Parameter]:
        adapter_parameter_ids = {id(parameter) for parameter in self.writer_lora_parameters()}
        for adapter in self._writer_lora_targets.values():
            adapter_parameter_ids.update(id(parameter) for parameter in adapter.base_linear.parameters())
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in adapter_parameter_ids
        ]

    def set_base_trainable(self, enabled: bool) -> None:
        for parameter in self.writer_base_parameters():
            parameter.requires_grad_(enabled)

    def set_writer_lora_trainable(self, enabled: bool) -> None:
        for parameter in self.writer_lora_parameters():
            parameter.requires_grad_(enabled)

    def writer_lora_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.writer_lora_parameters()))

    def writer_lora_metadata(self) -> dict[str, object]:
        metadata = dict(self._writer_lora_config)
        metadata["target_paths"] = sorted(self._writer_lora_targets)
        metadata["trainable_params"] = self.writer_lora_parameter_count()
        return metadata

    def _normalize_sequence(self, name: str, states: torch.Tensor | None) -> torch.Tensor | None:
        if states is None:
            return None
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if states.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, tokens, hidden_size].")
        if states.shape[-1] != self.embed_dim:
            raise ValueError(
                f"{name} expected hidden size {self.embed_dim}, got {states.shape[-1]}."
            )
        return states

    def _apply_conditioning_layer(
        self,
        slots: torch.Tensor,
        *,
        context_states: torch.Tensor | None,
        support_states: torch.Tensor | None,
        stimulus_mode: str,
        context_key_padding_mask: torch.Tensor | None,
        support_key_padding_mask: torch.Tensor | None,
        collect_support_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        current_slots = slots
        if context_states is not None and stimulus_mode in {"context_only", "support_and_context"}:
            context_attended, _ = self.context_cross_attn(
                query=current_slots,
                key=context_states,
                value=context_states,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            current_slots = self.context_attn_norm(
                context_attended + (self.context_query_residual_scale * current_slots)
            )
        support_attention_weights = None
        if support_states is not None and stimulus_mode in {"support_only", "support_and_context"}:
            support_attended, support_attention_weights = self.support_cross_attn(
                query=current_slots,
                key=support_states,
                value=support_states,
                key_padding_mask=support_key_padding_mask,
                need_weights=bool(collect_support_diagnostics),
                average_attn_weights=False,
            )
            current_slots = self.support_attn_norm(
                support_attended + (self.support_query_residual_scale * current_slots)
            )
        return current_slots, support_attention_weights

    def enable_aux_projection_head(
        self,
        *,
        projection_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        if projection_dim <= 0:
            raise ValueError(
                f"WriterWeaverHead aux projection_dim must be positive, got {projection_dim}."
            )
        self.aux_projection_head = WriterAuxProjectionHead(
            input_dim=self.embed_dim,
            projection_dim=int(projection_dim),
            hidden_dim=hidden_dim,
        )

    def project_auxiliary(
        self,
        slots: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if self.aux_projection_head is None:
            raise RuntimeError("WriterWeaverHead auxiliary projection head is not enabled.")
        if slots.ndim != 3 or slots.shape[-1] != self.embed_dim:
            raise ValueError(
                f"WriterWeaverHead.project_auxiliary expected [batch, slots, {self.embed_dim}] input, "
                f"got {tuple(slots.shape)}."
            )
        if reduction == "mean":
            inputs = slots.mean(dim=1)
        elif reduction == "slots":
            inputs = slots
        else:
            raise ValueError(
                f"Unsupported WriterWeaverHead auxiliary reduction={reduction}. "
                "Expected one of mean, slots."
            )
        return self.aux_projection_head(inputs)

    def write(
        self,
        *,
        context_states: torch.Tensor | None,
        support_states: torch.Tensor | None,
        stimulus_mode: str = "support_and_context",
        context_key_padding_mask: torch.Tensor | None = None,
        support_key_padding_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if stimulus_mode not in {"support_only", "context_only", "support_and_context"}:
            raise ValueError(
                f"Unsupported WriterWeaverHead stimulus_mode={stimulus_mode}. "
                "Expected one of support_only, context_only, support_and_context."
            )
        normalized_context = self._normalize_sequence("context_states", context_states)
        normalized_support = self._normalize_sequence("support_states", support_states)
        if stimulus_mode in {"context_only", "support_and_context"} and normalized_context is None:
            raise ValueError(
                f"WriterWeaverHead stimulus_mode={stimulus_mode} requires non-empty context_states."
            )
        if stimulus_mode in {"support_only", "support_and_context"} and normalized_support is None:
            raise ValueError(
                f"WriterWeaverHead stimulus_mode={stimulus_mode} requires non-empty support_states."
            )
        source_state = normalized_context if normalized_context is not None else normalized_support
        if source_state is None:
            raise ValueError("WriterWeaverHead requires at least one stimulus source.")
        batch_size = int(source_state.shape[0])
        slots = self.slot_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        support_attention_by_layer: dict[str, torch.Tensor] = {}
        current_slots, support_attention = self._apply_conditioning_layer(
            slots,
            context_states=normalized_context,
            support_states=normalized_support,
            stimulus_mode=stimulus_mode,
            context_key_padding_mask=context_key_padding_mask,
            support_key_padding_mask=support_key_padding_mask,
            collect_support_diagnostics=return_diagnostics,
        )
        if support_attention is not None:
            support_attention_by_layer["0"] = support_attention
        for block in self.extra_conditioning_blocks:
            current_slots, support_attention = block(
                current_slots,
                context_states=normalized_context,
                support_states=normalized_support,
                stimulus_mode=stimulus_mode,
                context_key_padding_mask=context_key_padding_mask,
                support_key_padding_mask=support_key_padding_mask,
                context_query_residual_scale=self.context_query_residual_scale,
                support_query_residual_scale=self.support_query_residual_scale,
                collect_support_diagnostics=return_diagnostics,
            )
            if support_attention is not None:
                support_attention_by_layer[str(len(support_attention_by_layer))] = support_attention
        encoded = self.output_norm(self.encoder(current_slots))
        if self.output_slot_basis_scale != 0.0:
            encoded = encoded + (self.output_slot_basis_scale * slots)
        if not return_diagnostics:
            return encoded
        return encoded, {"support_attention_weights_by_layer": support_attention_by_layer}


class MemoryReader(ManagedMemoryModule):
    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        use_query_gating: bool = False,
        *,
        gating_mode: str | None = None,
        num_heads: int = 4,
        condition_on_context: bool = True,
        conditioning_mode: str | None = None,
        gated_add_scale: float = 0.1,
        attention_mode: str = "standard",
        masked_partition: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
        dropout: float = 0.0,
        query_residual_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.query_residual_scale = float(query_residual_scale)
        resolved_conditioning_mode = conditioning_mode or ("add" if condition_on_context else "none")
        if resolved_conditioning_mode not in {"add", "gated_add", "none"}:
            raise ValueError(
                f"Unsupported reader conditioning mode: {resolved_conditioning_mode}. "
                "Expected one of add/gated_add/none."
            )
        if attention_mode not in {"standard", "competitive_slots", "masked_partition"}:
            raise ValueError(
                f"Unsupported reader attention mode: {attention_mode}. "
                "Expected one of standard/competitive_slots/masked_partition."
            )
        resolved_gating_mode = gating_mode or ("learned" if use_query_gating else "off")
        if resolved_gating_mode not in {"off", "random", "learned"}:
            raise ValueError(
                f"Unsupported reader gating mode: {resolved_gating_mode}. "
                "Expected one of off/random/learned."
            )
        self.gating_mode = resolved_gating_mode
        self.condition_on_context = bool(condition_on_context)
        self.conditioning_mode = resolved_conditioning_mode
        self.gated_add_scale = float(gated_add_scale)
        self.attention_mode = attention_mode
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.context_proj = nn.Linear(embed_dim, embed_dim) if self.condition_on_context else None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.readout_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim, num_queries) if self.gating_mode == "learned" else None
        self.masked_partition = self._normalize_masked_partition(masked_partition)

    def _pool_context(self, context: torch.Tensor | None) -> torch.Tensor | None:
        if context is None:
            return None
        if context.ndim == 2:
            return context
        if context.ndim == 3:
            return context.mean(dim=1)
        raise ValueError(f"Unsupported context rank for reader: {context.ndim}")

    def _build_key_padding_mask(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if memory_mask is None:
            return None
        _ensure_rank("memory_mask", memory_mask, 2)
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError(
                "memory_mask must have shape [batch, memory_slots] matching memory."
        )
        return ~memory_mask.to(dtype=torch.bool, device=memory.device)

    def _project_attention_logits(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        projected_queries, projected_keys, _ = self._project_attention_inputs(queries, memory)
        logits = torch.matmul(projected_queries, projected_keys.transpose(-2, -1))
        logits = logits * (1.0 / math.sqrt(projected_queries.shape[-1]))
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        return logits

    def _normalize_masked_partition(
        self,
        masked_partition: list[list[int]] | tuple[tuple[int, ...], ...] | None,
    ) -> tuple[tuple[int, ...], ...] | None:
        if masked_partition is None:
            return None
        normalized: list[tuple[int, ...]] = []
        for index, slot_group in enumerate(masked_partition):
            group = tuple(int(slot) for slot in slot_group)
            if not group:
                raise ValueError(
                    f"masked_partition query {index} must contain at least one allowed slot."
                )
            normalized.append(group)
        if len(normalized) != self.num_queries:
            raise ValueError(
                f"masked_partition must define exactly {self.num_queries} query groups, "
                f"got {len(normalized)}."
            )
        return tuple(normalized)

    def _project_attention_inputs(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_heads = int(self.cross_attn.num_heads)
        head_dim = self.embed_dim // max(1, num_heads)
        if head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"Reader embed_dim={self.embed_dim} must be divisible by num_heads={num_heads}."
            )
        if self.cross_attn._qkv_same_embed_dim:
            q_proj_weight, k_proj_weight, v_proj_weight = self.cross_attn.in_proj_weight.chunk(3, dim=0)
            if self.cross_attn.in_proj_bias is None:
                q_proj_bias = None
                k_proj_bias = None
                v_proj_bias = None
            else:
                q_proj_bias, k_proj_bias, v_proj_bias = self.cross_attn.in_proj_bias.chunk(3, dim=0)
        else:
            q_proj_weight = self.cross_attn.q_proj_weight
            k_proj_weight = self.cross_attn.k_proj_weight
            v_proj_weight = self.cross_attn.v_proj_weight
            if self.cross_attn.in_proj_bias is None:
                q_proj_bias = None
                k_proj_bias = None
                v_proj_bias = None
            else:
                q_proj_bias, k_proj_bias, v_proj_bias = self.cross_attn.in_proj_bias.chunk(3, dim=0)

        projected_queries = F.linear(queries, q_proj_weight, q_proj_bias)
        projected_keys = F.linear(memory, k_proj_weight, k_proj_bias)
        projected_values = F.linear(memory, v_proj_weight, v_proj_bias)
        projected_queries = projected_queries.view(
            queries.shape[0],
            queries.shape[1],
            num_heads,
            head_dim,
        ).transpose(1, 2)
        projected_keys = projected_keys.view(
            memory.shape[0],
            memory.shape[1],
            num_heads,
            head_dim,
        ).transpose(1, 2)
        projected_values = projected_values.view(
            memory.shape[0],
            memory.shape[1],
            num_heads,
            head_dim,
        ).transpose(1, 2)
        return projected_queries, projected_keys, projected_values

    def _query_slot_mask(
        self,
        memory_slots: int,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.attention_mode != "masked_partition":
            return None
        mask = torch.zeros(self.num_queries, memory_slots, dtype=torch.bool, device=device)
        if self.masked_partition is None:
            slot_indices = torch.arange(memory_slots, device=device)
            for query_index, chunk in enumerate(torch.tensor_split(slot_indices, self.num_queries)):
                if chunk.numel() == 0:
                    mask[query_index, min(query_index, memory_slots - 1)] = True
                else:
                    mask[query_index, chunk] = True
            return mask
        for query_index, slot_group in enumerate(self.masked_partition):
            for slot_index in slot_group:
                if slot_index < 0 or slot_index >= memory_slots:
                    raise ValueError(
                        f"masked_partition slot index {slot_index} is invalid for memory_slots={memory_slots}."
                    )
                mask[query_index, slot_index] = True
        return mask

    def _safe_softmax(
        self,
        logits: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))
        no_valid_entries = ~valid_mask.any(dim=dim, keepdim=True)
        masked_logits = masked_logits.masked_fill(no_valid_entries.expand_as(masked_logits), 0.0)
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights.masked_fill(~valid_mask, 0.0)
        normalizer = weights.sum(dim=dim, keepdim=True)
        return torch.where(
            normalizer > 0,
            weights / normalizer.clamp_min(1e-9),
            torch.zeros_like(weights),
        )

    def _manual_attention(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_queries, projected_keys, projected_values = self._project_attention_inputs(
            queries,
            memory,
        )
        value_projected_slots = projected_values.transpose(1, 2).contiguous().view(
            memory.shape[0],
            memory.shape[1],
            self.embed_dim,
        )
        logits = torch.matmul(projected_queries, projected_keys.transpose(-2, -1))
        logits = logits * (1.0 / math.sqrt(projected_queries.shape[-1]))
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
        if key_padding_mask is not None:
            valid_mask = valid_mask & (~key_padding_mask[:, None, None, :])
        query_slot_mask = self._query_slot_mask(
            memory.shape[1],
            device=memory.device,
        )
        if query_slot_mask is not None:
            valid_mask = valid_mask & query_slot_mask[None, None, :, :]
        attention_logits = logits.masked_fill(~valid_mask, float("-inf"))
        if self.attention_mode == "competitive_slots":
            competitive = self._safe_softmax(attention_logits, valid_mask=valid_mask, dim=-2)
            attention_weights = competitive.masked_fill(~valid_mask, 0.0)
            attention_normalizer = attention_weights.sum(dim=-1, keepdim=True)
            attention_weights = torch.where(
                attention_normalizer > 0,
                attention_weights / attention_normalizer.clamp_min(1e-9),
                torch.zeros_like(attention_weights),
            )
        else:
            attention_weights = self._safe_softmax(attention_logits, valid_mask=valid_mask, dim=-1)
        attention_for_output = F.dropout(
            attention_weights,
            p=float(self.cross_attn.dropout),
            training=self.training,
        )
        readouts = torch.matmul(attention_for_output, projected_values)
        readouts = readouts.transpose(1, 2).contiguous().view(
            queries.shape[0],
            queries.shape[1],
            self.embed_dim,
        )
        readouts = F.linear(
            readouts,
            self.cross_attn.out_proj.weight,
            self.cross_attn.out_proj.bias,
        )
        return readouts, attention_weights.mean(dim=1), attention_logits, value_projected_slots

    def read(
        self,
        memory: torch.Tensor,
        context: torch.Tensor | None = None,
        candidate_state: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ensure_rank("memory", memory, 3)
        if memory.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Reader expected memory hidden size {self.embed_dim}, got {memory.shape[-1]}."
            )
        batch_size = memory.shape[0]
        base_queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        conditioned_queries = base_queries
        pooled_context = self._pool_context(context)
        if pooled_context is not None and pooled_context.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Reader expected context hidden size {self.embed_dim}, got {pooled_context.shape[-1]}."
            )
        pooled_candidate_state = self._pool_context(candidate_state)
        if pooled_candidate_state is not None and pooled_candidate_state.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Reader expected candidate_state hidden size {self.embed_dim}, got {pooled_candidate_state.shape[-1]}."
            )
        if self.context_proj is not None and pooled_context is not None:
            projected_context = self.context_proj(pooled_context).unsqueeze(1)
            if self.conditioning_mode == "add":
                conditioned_queries = conditioned_queries + projected_context
            elif self.conditioning_mode == "gated_add":
                conditioned_queries = conditioned_queries + (
                    self.gated_add_scale * projected_context
                )
        if pooled_candidate_state is not None and self.conditioning_mode != "none":
            projected_candidate = pooled_candidate_state.unsqueeze(1)
            if self.conditioning_mode == "gated_add":
                conditioned_queries = conditioned_queries + (
                    self.gated_add_scale * projected_candidate
                )
            else:
                conditioned_queries = conditioned_queries + projected_candidate
        context_shift = conditioned_queries - base_queries

        key_padding_mask = self._build_key_padding_mask(memory, memory_mask)
        readouts, attention, attention_logits, value_projected_slots = self._manual_attention(
            conditioned_queries,
            memory,
            key_padding_mask=key_padding_mask,
        )
        if self.gating_mode == "learned":
            gate_source = conditioned_queries.mean(dim=1)
            if pooled_context is not None and self.conditioning_mode != "none":
                gate_source = pooled_context
            gates = torch.sigmoid(self.gate(gate_source))
            readouts = readouts * gates.unsqueeze(-1)
        elif self.gating_mode == "random":
            gates = (
                torch.rand(batch_size, self.num_queries, device=memory.device) > 0.5
            ).to(dtype=readouts.dtype)
            readouts = readouts * gates.unsqueeze(-1)
        else:
            gates = torch.ones(batch_size, self.num_queries, device=memory.device)
        normalized_readouts = self.readout_norm(readouts)
        if self.query_residual_scale != 0.0:
            normalized_readouts = normalized_readouts + (
                self.query_residual_scale * conditioned_queries * gates.unsqueeze(-1)
            )
        return {
            "readouts": normalized_readouts,
            "raw_readouts": readouts,
            "attention": attention,
            "gates": gates,
            "queries": conditioned_queries,
            "base_queries": base_queries,
            "conditioned_queries": conditioned_queries,
            "context_shift": context_shift,
            "attention_logits": attention_logits,
            "value_projected_slots": value_projected_slots,
        }


class MemoryFuser(ManagedMemoryModule):
    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        short_slots: int,
        *,
        arch: str = "linear",
        hidden_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.short_slots = short_slots
        self.arch = arch
        hidden_dim = hidden_dim or (2 * embed_dim)
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(short_slots * embed_dim),
            nn.Linear(short_slots * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        if arch == "linear":
            self.proj = nn.Sequential(
                nn.LayerNorm(num_queries * embed_dim),
                nn.Linear(num_queries * embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, short_slots * embed_dim),
            )
        elif arch == "resampler":
            self.short_queries = nn.Parameter(torch.randn(short_slots, embed_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.output_norm = nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"Unsupported fuser architecture: {arch}")

    def fuse(self, readouts: torch.Tensor) -> torch.Tensor:
        _ensure_rank("readouts", readouts, 3)
        if readouts.shape[1] != self.num_queries:
            raise ValueError(
                f"Fuser expected {self.num_queries} readouts, got {readouts.shape[1]}."
            )
        if readouts.shape[2] != self.embed_dim:
            raise ValueError(
                f"Fuser expected readout hidden size {self.embed_dim}, got {readouts.shape[2]}."
            )
        batch_size = readouts.shape[0]
        if self.arch == "linear":
            flattened = readouts.reshape(batch_size, self.num_queries * self.embed_dim)
            fused = self.proj(flattened)
            return fused.view(batch_size, self.short_slots, self.embed_dim)

        queries = self.short_queries.unsqueeze(0).expand(batch_size, -1, -1)
        fused, _ = self.cross_attn(
            query=queries,
            key=readouts,
            value=readouts,
            need_weights=False,
        )
        # Preserve slot identity from the learned short queries so resampled slots
        # do not collapse to an order-insensitive average when attention is similar.
        return self.output_norm(fused + queries)

    def summarize(self, memory_short: torch.Tensor) -> torch.Tensor:
        _ensure_rank("memory_short", memory_short, 3)
        if memory_short.shape[1] != self.short_slots:
            raise ValueError(
                f"Fuser expected {self.short_slots} short slots, got {memory_short.shape[1]}."
            )
        if memory_short.shape[2] != self.embed_dim:
            raise ValueError(
                f"Fuser expected short-slot hidden size {self.embed_dim}, got {memory_short.shape[2]}."
            )
        flattened = memory_short.reshape(memory_short.shape[0], self.short_slots * self.embed_dim)
        return self.summary_proj(flattened)


class MemoryInjector(ManagedMemoryModule):
    def __init__(
        self,
        mode: str = "prefix",
        *,
        enabled: bool = True,
        position: str = "segment",
    ) -> None:
        super().__init__()
        if mode != "prefix":
            raise NotImplementedError("M0 bootstrap currently supports only prefix injection.")
        if position not in {"segment", "delimiter", "random", "none"}:
            raise ValueError(
                f"Unsupported injection position: {position}. "
                "Expected one of segment/delimiter/random/none."
            )
        self.mode = mode
        self.enabled = enabled
        self.position = position

    def memory_for_generation(self, memory_short: torch.Tensor) -> torch.Tensor | None:
        return memory_short if self.enabled and self.position != "none" else None

    def inject(self, memory_short: torch.Tensor, next_inputs: torch.Tensor) -> torch.Tensor:
        _ensure_rank("memory_short", memory_short, 3)
        _ensure_rank("next_inputs", next_inputs, 3)
        if memory_short.shape[0] != next_inputs.shape[0]:
            raise ValueError("memory_short and next_inputs must have the same batch size.")
        if memory_short.shape[2] != next_inputs.shape[2]:
            raise ValueError("memory_short and next_inputs must share the same hidden size.")
        if not self.enabled or self.position == "none":
            return next_inputs
        return torch.cat([memory_short, next_inputs], dim=1)

    def compose(
        self,
        *,
        segment_memories: list[torch.Tensor],
        segment_inputs: list[torch.Tensor],
        delimiter_inputs: torch.Tensor | None,
        suffix_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        if len(segment_memories) != len(segment_inputs):
            raise ValueError("segment_memories and segment_inputs must have the same length.")
        if not segment_memories:
            raise ValueError("compose() requires at least one segment memory/input pair.")
        for index, (memory_short, segment_input) in enumerate(zip(segment_memories, segment_inputs)):
            _ensure_rank(f"segment_memories[{index}]", memory_short, 3)
            _ensure_rank(f"segment_inputs[{index}]", segment_input, 3)
            if memory_short.shape[0] != segment_input.shape[0]:
                raise ValueError("All segment memories and inputs must share the same batch size.")
            if memory_short.shape[2] != segment_input.shape[2]:
                raise ValueError("All segment memories and inputs must share the same hidden size.")
        _ensure_rank("suffix_inputs", suffix_inputs, 3)
        if delimiter_inputs is not None:
            _ensure_rank("delimiter_inputs", delimiter_inputs, 3)
        if not self.enabled or self.position == "none":
            return self._concat_base_sequence(segment_inputs, delimiter_inputs, suffix_inputs), None, []
        if self.position == "segment":
            return self._compose_segment(segment_memories, segment_inputs, delimiter_inputs, suffix_inputs)
        if self.position == "delimiter":
            return self._compose_delimiter(segment_memories, segment_inputs, delimiter_inputs, suffix_inputs)
        return self._compose_random(segment_memories, segment_inputs, delimiter_inputs, suffix_inputs)

    def _concat_base_sequence(
        self,
        segment_inputs: list[torch.Tensor],
        delimiter_inputs: torch.Tensor | None,
        suffix_inputs: torch.Tensor,
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for index, segment_input in enumerate(segment_inputs):
            chunks.append(segment_input)
            if delimiter_inputs is not None and index < len(segment_inputs) - 1:
                chunks.append(delimiter_inputs)
        chunks.append(suffix_inputs)
        return torch.cat(chunks, dim=1)

    def _compose_segment(
        self,
        segment_memories: list[torch.Tensor],
        segment_inputs: list[torch.Tensor],
        delimiter_inputs: torch.Tensor | None,
        suffix_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        chunks: list[torch.Tensor] = []
        generation_chunks: list[torch.Tensor] = []
        anchors: list[str] = []
        for index, (segment_input, segment_memory) in enumerate(zip(segment_inputs, segment_memories)):
            chunks.append(segment_input)
            chunks.append(segment_memory)
            generation_chunks.append(segment_memory)
            anchors.append(f"segment:{index}")
            if delimiter_inputs is not None and index < len(segment_inputs) - 1:
                chunks.append(delimiter_inputs)
        chunks.append(suffix_inputs)
        return torch.cat(chunks, dim=1), torch.cat(generation_chunks, dim=1), anchors

    def _compose_delimiter(
        self,
        segment_memories: list[torch.Tensor],
        segment_inputs: list[torch.Tensor],
        delimiter_inputs: torch.Tensor | None,
        suffix_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        chunks: list[torch.Tensor] = []
        generation_chunks: list[torch.Tensor] = []
        anchors: list[str] = []
        for index, (segment_input, segment_memory) in enumerate(zip(segment_inputs, segment_memories)):
            chunks.append(segment_input)
            has_delimiter = delimiter_inputs is not None and index < len(segment_inputs) - 1
            if has_delimiter:
                chunks.append(delimiter_inputs)
                chunks.append(segment_memory)
                generation_chunks.append(segment_memory)
                anchors.append(f"delimiter:{index}")
        if not generation_chunks:
            chunks.append(segment_memories[-1])
            generation_chunks.append(segment_memories[-1])
            anchors.append("delimiter-fallback:last")
        chunks.append(suffix_inputs)
        return torch.cat(chunks, dim=1), torch.cat(generation_chunks, dim=1), anchors

    def _compose_random(
        self,
        segment_memories: list[torch.Tensor],
        segment_inputs: list[torch.Tensor],
        delimiter_inputs: torch.Tensor | None,
        suffix_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        base_chunks: list[torch.Tensor] = []
        for index, segment_input in enumerate(segment_inputs):
            base_chunks.append(segment_input)
            if delimiter_inputs is not None and index < len(segment_inputs) - 1:
                base_chunks.append(delimiter_inputs)
        base_chunks.append(suffix_inputs)

        insertions: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(base_chunks) + 1)}
        anchors: list[str] = []
        for segment_index, segment_memory in enumerate(segment_memories):
            boundary = int(torch.randint(0, len(base_chunks) + 1, (1,)).item())
            insertions[boundary].append(segment_memory)
            anchors.append(f"random:{segment_index}@{boundary}")

        chunks: list[torch.Tensor] = []
        generation_chunks: list[torch.Tensor] = []
        for boundary in range(len(base_chunks) + 1):
            for segment_memory in insertions[boundary]:
                chunks.append(segment_memory)
                generation_chunks.append(segment_memory)
            if boundary < len(base_chunks):
                chunks.append(base_chunks[boundary])
        return torch.cat(chunks, dim=1), torch.cat(generation_chunks, dim=1), anchors
