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
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_slots = memory_slots
        self.arch = arch
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

    def _pool_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            return state
        if state.ndim == 3:
            return state.mean(dim=1)
        raise ValueError(f"Unsupported state rank for writer: {state.ndim}")

    def write(self, state: torch.Tensor) -> torch.Tensor:
        pooled_state = self._pool_state(state)
        if pooled_state.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Writer expected state hidden size {self.embed_dim}, got {pooled_state.shape[-1]}."
            )
        batch_size = pooled_state.shape[0]
        if self.arch == "mlp":
            memory = self.proj(pooled_state)
            return memory.view(batch_size, self.memory_slots, self.embed_dim)

        slots = self.slot_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        conditioned_slots = slots + self.state_proj(pooled_state).unsqueeze(1)
        return self.output_norm(self.encoder(conditioned_slots))


class MemoryReader(ManagedMemoryModule):
    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        use_query_gating: bool = False,
        *,
        num_heads: int = 4,
        condition_on_context: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.use_query_gating = use_query_gating
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
        self.gate = nn.Linear(embed_dim, num_queries) if use_query_gating else None

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
        if self.context_proj is not None and pooled_context is not None:
            queries = queries + self.context_proj(pooled_context).unsqueeze(1)

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
        if self.gate is not None and pooled_context is not None:
            gates = torch.sigmoid(self.gate(pooled_context))
            readouts = readouts * gates.unsqueeze(-1)
        else:
            gates = torch.ones(batch_size, self.num_queries, device=memory.device)
        return {
            "readouts": self.readout_norm(readouts),
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
        return self.output_norm(fused)


class MemoryInjector(ManagedMemoryModule):
    def __init__(self, mode: str = "prefix", *, enabled: bool = True) -> None:
        super().__init__()
        if mode != "prefix":
            raise NotImplementedError("M0 bootstrap currently supports only prefix injection.")
        self.mode = mode
        self.enabled = enabled

    def memory_for_generation(self, memory_short: torch.Tensor) -> torch.Tensor | None:
        return memory_short if self.enabled else None

    def inject(self, memory_short: torch.Tensor, next_inputs: torch.Tensor) -> torch.Tensor:
        _ensure_rank("memory_short", memory_short, 3)
        _ensure_rank("next_inputs", next_inputs, 3)
        if memory_short.shape[0] != next_inputs.shape[0]:
            raise ValueError("memory_short and next_inputs must have the same batch size.")
        if memory_short.shape[2] != next_inputs.shape[2]:
            raise ValueError("memory_short and next_inputs must share the same hidden size.")
        if not self.enabled:
            return next_inputs
        return torch.cat([memory_short, next_inputs], dim=1)
