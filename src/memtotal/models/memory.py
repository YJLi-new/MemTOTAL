from __future__ import annotations

import math

import torch
from torch import nn


class MemoryWriter(nn.Module):
    def __init__(self, embed_dim: int, memory_slots: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_slots = memory_slots
        self.proj = nn.Linear(embed_dim, memory_slots * embed_dim)

    def write(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        memory = self.proj(state)
        return memory.view(batch_size, self.memory_slots, self.embed_dim)


class MemoryReader(nn.Module):
    def __init__(self, embed_dim: int, num_queries: int, use_query_gating: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.use_query_gating = use_query_gating
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.gate = nn.Linear(embed_dim, num_queries) if use_query_gating else None

    def read(self, memory: torch.Tensor, context: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch_size = memory.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        logits = torch.einsum("bhd,bld->bhl", queries, memory) / math.sqrt(self.embed_dim)
        attention = torch.softmax(logits, dim=-1)
        readouts = torch.einsum("bhl,bld->bhd", attention, memory)
        if self.gate is not None and context is not None:
            gates = torch.sigmoid(self.gate(context))
            readouts = readouts * gates.unsqueeze(-1)
        else:
            gates = torch.ones(batch_size, self.num_queries, device=memory.device)
        return {"readouts": readouts, "attention": attention, "gates": gates}


class MemoryFuser(nn.Module):
    def __init__(self, embed_dim: int, num_queries: int, short_slots: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.short_slots = short_slots
        self.proj = nn.Linear(num_queries * embed_dim, short_slots * embed_dim)

    def fuse(self, readouts: torch.Tensor) -> torch.Tensor:
        batch_size = readouts.shape[0]
        flattened = readouts.reshape(batch_size, self.num_queries * self.embed_dim)
        fused = self.proj(flattened)
        return fused.view(batch_size, self.short_slots, self.embed_dim)


class MemoryInjector(nn.Module):
    def __init__(self, mode: str = "prefix") -> None:
        super().__init__()
        if mode != "prefix":
            raise NotImplementedError("M0 bootstrap currently supports only prefix injection.")
        self.mode = mode

    def inject(self, memory_short: torch.Tensor, next_inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([memory_short, next_inputs], dim=1)

