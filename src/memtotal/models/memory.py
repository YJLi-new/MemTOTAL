from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


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
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_slots = memory_slots
        self.arch = arch
        self.support_query_residual_scale = float(support_query_residual_scale)
        self.output_slot_basis_scale = float(output_slot_basis_scale)
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
            conditioned_slots = slots + self.state_proj(pooled_state).unsqueeze(1)
            encoded_slots = self.output_norm(self.encoder(conditioned_slots))
            if self.output_slot_basis_scale != 0.0:
                encoded_slots = encoded_slots + (self.output_slot_basis_scale * slots)
            return encoded_slots

        pooled_support_state = state.mean(dim=1)
        conditioned_slots = slots + self.state_proj(pooled_support_state).unsqueeze(1)
        attended_slots, _ = self.support_cross_attn(
            query=conditioned_slots,
            key=state,
            value=state,
            need_weights=False,
        )
        support_slots = attended_slots + (self.support_query_residual_scale * conditioned_slots)
        encoded_slots = self.output_norm(self.encoder(support_slots))
        if self.output_slot_basis_scale != 0.0:
            encoded_slots = encoded_slots + (self.output_slot_basis_scale * slots)
        return encoded_slots


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
        dropout: float = 0.0,
        query_residual_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.query_residual_scale = float(query_residual_scale)
        resolved_gating_mode = gating_mode or ("learned" if use_query_gating else "off")
        if resolved_gating_mode not in {"off", "random", "learned"}:
            raise ValueError(
                f"Unsupported reader gating mode: {resolved_gating_mode}. "
                "Expected one of off/random/learned."
            )
        self.gating_mode = resolved_gating_mode
        self.condition_on_context = condition_on_context
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.context_proj = nn.Linear(embed_dim, embed_dim) if condition_on_context else None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.readout_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim, num_queries) if self.gating_mode == "learned" else None

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
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
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
            queries = queries + self.context_proj(pooled_context).unsqueeze(1)
        if pooled_candidate_state is not None:
            queries = queries + pooled_candidate_state.unsqueeze(1)

        key_padding_mask = self._build_key_padding_mask(memory, memory_mask)
        readouts, attention = self.cross_attn(
            query=queries,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attention = attention.mean(dim=1)
        if self.gating_mode == "learned":
            gate_source = pooled_context if pooled_context is not None else queries.mean(dim=1)
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
                self.query_residual_scale * queries * gates.unsqueeze(-1)
            )
        return {
            "readouts": normalized_readouts,
            "attention": attention,
            "gates": gates,
            "queries": queries,
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
