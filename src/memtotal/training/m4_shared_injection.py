from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from memtotal.models import (
    BackboneWrapper,
    MemoryFuser,
    MemoryReader,
    MemoryWriter,
    SourceStubMemory,
    WriterDeepPrefixProjector,
    WriterWeaverHead,
)
from memtotal.tasks import build_task_evaluator, load_task_dataset
from memtotal.training.m3 import _resolve_artifact_path
from memtotal.utils.io import write_json, write_jsonl
from memtotal.utils.profiling import ProfileTracker

FEVER_LABEL_ORDER = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
FEVER_PROMPT_VARIANTS = (
    "inline_short_labels",
    "answer_slot_labels",
    "verbalized_decisions",
)
FEVER_SUPPORT_SERIALIZATION_VARIANTS = (
    "flat_raw8",
    "example_blocks_raw8",
    "triad_curated6",
)
GENERIC_SUPPORT_SERIALIZATION_VARIANTS = (
    "flat_raw8",
    "example_blocks_raw8",
)
FEVER_TRIAD_NEI_EVIDENCE = "insufficient evidence available"
TASK_NATIVE_PROMPT_VARIANT = "task_native"
TASK_CANDIDATE_SELECTION_EVALUATOR_TYPES = (
    "dataset_label_classification",
    "multiple_choice",
)
TASK_GENERATION_EVALUATOR_TYPES = (
    "exact_match",
    "qa_f1",
    "memoryagentbench",
)


@dataclass(frozen=True)
class SharedInjectionExampleCache:
    example: dict[str, Any]
    candidate_labels: list[str]
    candidate_texts: list[str]
    gold_index: int
    prompt_text: str
    prompt_variant: str
    evaluator_type: str
    task_mode: str


@dataclass(frozen=True)
class PrefixInjectionArtifacts:
    prefix_embeddings: torch.Tensor | None
    layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None
    prefix_stats: dict[str, Any]
    memory_slots: torch.Tensor | None = None
    support_item_states: torch.Tensor | None = None
    writer_support_states: torch.Tensor | None = None
    writer_context_states: torch.Tensor | None = None
    writer_context_mask: torch.Tensor | None = None
    memory_long: torch.Tensor | None = None
    memory_short: torch.Tensor | None = None
    reader_attention: torch.Tensor | None = None
    reader_gates: torch.Tensor | None = None
    reader_queries: torch.Tensor | None = None
    reader_context: torch.Tensor | None = None
    reader_conditioned_queries: torch.Tensor | None = None
    reader_value_projected_slots: torch.Tensor | None = None
    reader_readouts: torch.Tensor | None = None


@dataclass(frozen=True)
class SupportEpisode:
    episode_id: str
    support_rows: list[dict[str, Any]]
    source_split: str
    label_counts: dict[str, int]


class StructuredSupportSetEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        label_count: int,
        max_items: int = 8,
        num_heads: int = 4,
        transformer_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_items = int(max_items)
        self.label_embeddings = nn.Embedding(int(label_count), self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_items, self.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(num_heads),
            dim_feedforward=2 * self.hidden_size,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        self.output_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, item_states: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        if item_states.ndim != 3:
            raise ValueError("item_states must have shape [batch, item_count, hidden_size].")
        if label_ids.ndim != 2:
            raise ValueError("label_ids must have shape [batch, item_count].")
        if item_states.shape[:2] != label_ids.shape:
            raise ValueError("item_states and label_ids must agree on [batch, item_count].")
        if item_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"StructuredSupportSetEncoder expected hidden size {self.hidden_size}, got {item_states.shape[-1]}."
            )
        item_count = int(item_states.shape[1])
        if item_count > self.max_items:
            raise ValueError(
                f"StructuredSupportSetEncoder max_items={self.max_items}, got item_count={item_count}."
            )
        position_ids = torch.arange(
            item_count,
            device=item_states.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(item_states.shape[0], -1)
        enriched = (
            item_states
            + self.label_embeddings(label_ids.to(device=item_states.device, dtype=torch.long))
            + self.position_embeddings(position_ids)
        )
        return self.output_norm(self.encoder(enriched))


class LatentPrefixProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        prefix_tokens: int,
        *,
        slot_max_norm: float | None = None,
        total_max_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prefix_tokens = int(prefix_tokens)
        self.slot_max_norm = None if slot_max_norm is None else float(slot_max_norm)
        self.total_max_norm = None if total_max_norm is None else float(total_max_norm)
        self.prefix_norm = nn.LayerNorm(hidden_size)
        self.prefix_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.prefix_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.prefix_proj.bias)

    def _apply_norm_caps(self, prefix_embeddings: torch.Tensor) -> torch.Tensor:
        clipped = prefix_embeddings
        if self.slot_max_norm is not None and self.slot_max_norm > 0.0:
            slot_norms = clipped.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            slot_scales = torch.clamp(self.slot_max_norm / slot_norms, max=1.0)
            clipped = clipped * slot_scales
        if self.total_max_norm is not None and self.total_max_norm > 0.0:
            total_norms = clipped.flatten(start_dim=1).norm(dim=1, keepdim=True).clamp_min(1e-6)
            total_scales = torch.clamp(self.total_max_norm / total_norms, max=1.0).view(-1, 1, 1)
            clipped = clipped * total_scales
        return clipped

    def forward(self, memory_slots: torch.Tensor) -> torch.Tensor:
        if memory_slots.ndim != 3:
            raise ValueError("memory_slots must have shape [batch, slots, hidden_size].")
        if memory_slots.shape[1] != self.prefix_tokens:
            raise ValueError(
                f"LatentPrefixProjector expected {self.prefix_tokens} slots, got {memory_slots.shape[1]}."
            )
        projected = self.prefix_proj(self.prefix_norm(memory_slots))
        return self._apply_norm_caps(projected)


class SharedLowRankDeepPrefixProjector(WriterDeepPrefixProjector):
    pass


def _load_task_dataset_with_path(config: dict[str, Any], dataset_path: str) -> list[dict[str, Any]]:
    config_copy = copy.deepcopy(config)
    config_copy["task"]["dataset_path"] = dataset_path
    return load_task_dataset(config_copy)


def _load_support_episode_bank(bank_path: str) -> list[SupportEpisode]:
    payload = json.loads(Path(bank_path).read_text())
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Support episode bank at {bank_path} is missing a non-empty episodes list.")
    parsed: list[SupportEpisode] = []
    for raw_episode in episodes:
        support_rows = [dict(row) for row in raw_episode.get("support_rows", [])]
        if not support_rows:
            raise ValueError(f"Support episode bank at {bank_path} contains an empty episode.")
        parsed.append(
            SupportEpisode(
                episode_id=str(raw_episode.get("episode_id", "")),
                support_rows=support_rows,
                source_split=str(raw_episode.get("source_split", "")),
                label_counts={
                    str(label): int(count)
                    for label, count in dict(raw_episode.get("label_counts", {})).items()
                },
            )
        )
    return parsed


def _merge_example_lookup(*example_groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for group in example_groups:
        for example in group:
            example_id = str(example.get("id", "")).strip()
            if not example_id:
                continue
            lookup[example_id] = dict(example)
    return lookup


def _resolve_support_lookup_dataset_paths(
    config: dict[str, Any],
    *,
    default_paths: list[str],
) -> list[str]:
    paths: list[str] = []
    for raw_path in [*default_paths, *list(config["task"].get("support_lookup_dataset_paths", []))]:
        path = str(raw_path).strip()
        if not path or path in paths:
            continue
        paths.append(path)
    return paths


def _resolve_shared_injection_arm(config: dict[str, Any]) -> str:
    arm = str(config["runtime"].get("shared_injection_arm", "base_only"))
    if arm not in {"base_only", "teacher_text", "injected"}:
        raise ValueError(
            f"Unsupported runtime.shared_injection_arm={arm}. "
            "Expected one of base_only, teacher_text, injected."
        )
    return arm


def _resolve_writer_memory_control(config: dict[str, Any]) -> str:
    memory_control = str(config["runtime"].get("writer_memory_control", "real"))
    if memory_control not in {"real", "shuffled", "zero"}:
        raise ValueError(
            f"Unsupported runtime.writer_memory_control={memory_control}. "
            "Expected one of real, shuffled, zero."
        )
    return memory_control


def _resolve_pilot_arm_alias(config: dict[str, Any], *, arm: str, writer_memory_control: str) -> str:
    alias = config["runtime"].get("pilot_arm_alias")
    if alias is not None:
        return str(alias)
    if arm == "base_only":
        return "A"
    if arm == "teacher_text":
        return "T"
    return {
        "real": "I_real",
        "shuffled": "I_shuffle",
        "zero": "I_zero",
    }[writer_memory_control]


def _benchmark_id(config: dict[str, Any]) -> str:
    return str(config.get("task", {}).get("benchmark_id", "")).strip().lower()


def _is_fever_task(config: dict[str, Any]) -> bool:
    return _benchmark_id(config) == "fever"


def _resolve_prompt_variant(config: dict[str, Any]) -> str:
    if not _is_fever_task(config):
        return str(config["runtime"].get("pilot_prompt_variant", TASK_NATIVE_PROMPT_VARIANT))
    prompt_variant = str(config["runtime"].get("pilot_prompt_variant", "inline_short_labels"))
    if prompt_variant not in FEVER_PROMPT_VARIANTS:
        raise ValueError(
            f"Unsupported runtime.pilot_prompt_variant={prompt_variant}. "
            f"Expected one of {', '.join(FEVER_PROMPT_VARIANTS)}."
        )
    return prompt_variant


def _resolve_support_serialization_variant(config: dict[str, Any]) -> str:
    default_variant = "flat_raw8" if _is_fever_task(config) else "example_blocks_raw8"
    support_variant = str(config["runtime"].get("pilot_support_serialization", default_variant))
    allowed_variants = (
        FEVER_SUPPORT_SERIALIZATION_VARIANTS
        if _is_fever_task(config)
        else GENERIC_SUPPORT_SERIALIZATION_VARIANTS
    )
    if support_variant not in allowed_variants:
        raise ValueError(
            f"Unsupported runtime.pilot_support_serialization={support_variant}. "
            f"Expected one of {', '.join(allowed_variants)}."
        )
    return support_variant


def _resolve_injection_mode(config: dict[str, Any]) -> str:
    injection_mode = str(config["runtime"].get("pilot_injection_mode", "shallow_prefix"))
    if injection_mode not in {"shallow_prefix", "sparse_deep_prefix"}:
        raise ValueError(
            f"Unsupported runtime.pilot_injection_mode={injection_mode}. "
            "Expected one of shallow_prefix, sparse_deep_prefix."
        )
    return injection_mode


def _resolve_bridge_mode(config: dict[str, Any]) -> str:
    mode = str(config["runtime"].get("pilot_bridge_mode", "legacy"))
    if mode not in {"legacy", "writer_direct"}:
        raise ValueError(
            f"Unsupported runtime.pilot_bridge_mode={mode}. "
            "Expected one of legacy, writer_direct."
        )
    return mode


def _resolve_memory_path_variant(config: dict[str, Any]) -> str:
    variant = str(config["runtime"].get("pilot_memory_path_variant", "single_level"))
    if variant not in {"single_level", "two_level"}:
        raise ValueError(
            f"Unsupported runtime.pilot_memory_path_variant={variant}. "
            "Expected one of single_level, two_level."
        )
    return variant


def _resolve_deep_prefix_layers(config: dict[str, Any]) -> list[int]:
    explicit = config["runtime"].get("pilot_deep_prefix_layers")
    if explicit is None:
        return [0, 7, 14, 21, 27]
    return [int(layer_index) for layer_index in explicit]


def _resolve_deep_prefix_rank(config: dict[str, Any]) -> int:
    return int(config["runtime"].get("pilot_deep_prefix_rank", 32))


def _resolve_reader_context_mode(config: dict[str, Any]) -> str:
    mode = str(config["runtime"].get("pilot_reader_context_mode", "prompt_summary"))
    if mode not in {"prompt_summary", "none"}:
        raise ValueError(
            f"Unsupported runtime.pilot_reader_context_mode={mode}. "
            "Expected one of prompt_summary, none."
        )
    return mode


def _resolve_reader_num_queries(config: dict[str, Any]) -> int:
    if "pilot_reader_num_queries" in config["runtime"]:
        return int(config["runtime"]["pilot_reader_num_queries"])
    return int(config.get("method", {}).get("reader", {}).get("num_queries", 4))


def _resolve_fuser_short_slots(config: dict[str, Any]) -> int:
    if "pilot_fuser_short_slots" in config["runtime"]:
        return int(config["runtime"]["pilot_fuser_short_slots"])
    method_cfg = config.get("method", {})
    writer_slots = int(method_cfg.get("writer", {}).get("memory_slots", 8))
    return int(method_cfg.get("fuser", {}).get("short_slots", writer_slots))


def _resolve_projector_token_source(config: dict[str, Any]) -> str:
    variant = _resolve_memory_path_variant(config)
    source = str(
        config["runtime"].get(
            "pilot_projector_token_source",
            "short_slots" if variant == "two_level" else "writer_slots",
        )
    )
    if source not in {"writer_slots", "short_slots"}:
        raise ValueError(
            f"Unsupported runtime.pilot_projector_token_source={source}. "
            "Expected one of writer_slots, short_slots."
        )
    if variant == "single_level" and source != "writer_slots":
        raise ValueError(
            "runtime.pilot_projector_token_source=short_slots requires "
            "runtime.pilot_memory_path_variant=two_level."
        )
    return source


def _resolve_prefix_source_mode(config: dict[str, Any]) -> str:
    mode = str(config["runtime"].get("pilot_prefix_source_mode", "writer"))
    if mode not in {"writer", "source_stub"}:
        raise ValueError(
            f"Unsupported runtime.pilot_prefix_source_mode={mode}. "
            "Expected one of writer, source_stub."
        )
    return mode


def _resolve_deep_prefix_init_mode(config: dict[str, Any]) -> str:
    mode = str(config["runtime"].get("pilot_deep_prefix_init_mode", "random"))
    if mode not in {"random", "kv_stat_match", "semantic_anchor", "hidden_state_anchor"}:
        raise ValueError(
            f"Unsupported runtime.pilot_deep_prefix_init_mode={mode}. "
            "Expected one of random, kv_stat_match, semantic_anchor, hidden_state_anchor."
        )
    return mode


def _resolve_source_stub_learning_rate(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_source_stub_learning_rate", 5e-4))


def _resolve_source_stub_weight_decay(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_source_stub_weight_decay", 0.0))


def _resolve_support_encoder_mode(config: dict[str, Any]) -> str:
    mode = str(config["runtime"].get("pilot_support_encoder_mode", "pooled_block"))
    if mode not in {"pooled_block", "structured_support_set"}:
        raise ValueError(
            f"Unsupported runtime.pilot_support_encoder_mode={mode}. "
            "Expected one of pooled_block, structured_support_set."
        )
    return mode


def _resolve_trainable_variant(config: dict[str, Any]) -> str:
    variant = str(config["runtime"].get("pilot_trainable_variant", "full"))
    if variant not in {"full", "projector_only", "writer_adapter_only"}:
        raise ValueError(
            f"Unsupported runtime.pilot_trainable_variant={variant}. "
            "Expected one of full, projector_only, writer_adapter_only."
        )
    return variant


def _resolve_alignment_aux_mode(config: dict[str, Any]) -> str:
    raw_mode = config["runtime"].get("pilot_alignment_aux_mode", "off")
    if isinstance(raw_mode, bool):
        mode = "teacher_margin" if raw_mode else "off"
    elif raw_mode is None:
        mode = "off"
    else:
        mode = str(raw_mode)
    if mode not in {"off", "teacher_margin", "teacher_choice_kl", "teacher_choice_js"}:
        raise ValueError(
            f"Unsupported runtime.pilot_alignment_aux_mode={mode}. "
            "Expected one of off, teacher_margin, teacher_choice_kl, teacher_choice_js."
        )
    return mode


def _resolve_alignment_aux_weight_max(config: dict[str, Any]) -> float:
    if "pilot_alignment_aux_weight_max" in config["runtime"]:
        return float(config["runtime"].get("pilot_alignment_aux_weight_max", 0.0))
    return float(config["runtime"].get("pilot_alignment_aux_weight", 0.0))


def _resolve_alignment_aux_start_step(config: dict[str, Any]) -> int:
    return int(config["runtime"].get("pilot_alignment_aux_start_step", 0))


def _resolve_alignment_aux_ramp_steps(config: dict[str, Any]) -> int:
    return int(config["runtime"].get("pilot_alignment_aux_ramp_steps", 0))


def _resolve_alignment_aux_apply_only_to_real_memory(config: dict[str, Any]) -> bool:
    return bool(config["runtime"].get("pilot_alignment_aux_apply_only_to_real_memory", False))


def _resolve_alignment_aux_temperature(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_alignment_aux_temperature", 1.0))


def _resolve_alignment_aux_advantage_center(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_alignment_aux_advantage_center", 0.0))


def _resolve_alignment_aux_advantage_scale(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_alignment_aux_advantage_scale", 0.25))


def _resolve_init_checkpoint_path(config: dict[str, Any]) -> str:
    path = str(config["runtime"].get("pilot_init_checkpoint_path", "")).strip()
    return path


def _resolve_choice_ce_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_choice_ce_weight", 1.0))


def _resolve_competitor_hinge_weight_max(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_competitor_hinge_weight_max", 1.0))


def _resolve_competitor_hinge_start_step(config: dict[str, Any]) -> int:
    return int(config["runtime"].get("pilot_competitor_hinge_start_step", 0))


def _resolve_competitor_hinge_ramp_steps(config: dict[str, Any]) -> int:
    return int(config["runtime"].get("pilot_competitor_hinge_ramp_steps", 0))


def _resolve_latent_anchor_weight_start(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_latent_anchor_weight_start", 0.0))


def _resolve_latent_anchor_weight_end(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_latent_anchor_weight_end", 0.0))


def _resolve_latent_anchor_decay_steps(config: dict[str, Any]) -> int:
    return int(config["runtime"].get("pilot_latent_anchor_decay_steps", 0))


def _resolve_memory_long_diversity_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_memory_long_diversity_weight", 0.0))


def _resolve_memory_short_diversity_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_memory_short_diversity_weight", 0.0))


def _resolve_reader_attention_diversity_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_reader_attention_diversity_weight", 0.0))


def _resolve_reader_conditioned_query_orthogonality_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_reader_conditioned_query_orthogonality_weight", 0.0))


def _resolve_reader_short_reconstruction_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_reader_short_reconstruction_weight", 0.0))


def _resolve_reader_fuser_bootstrap_steps(config: dict[str, Any]) -> int:
    return int(config["runtime"].get("pilot_reader_fuser_bootstrap_steps", 0))


def _resolve_writer_slot_basis_orthogonality_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_writer_slot_basis_orthogonality_weight", 0.0))


def _resolve_writer_slot_energy_balance_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_writer_slot_energy_balance_weight", 0.0))


def _resolve_writer_common_mode_penalty_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_writer_common_mode_penalty_weight", 0.0))


def _resolve_writer_orthogonalize_slot_basis(config: dict[str, Any]) -> bool:
    return bool(config["runtime"].get("pilot_writer_orthogonalize_slot_basis", False))


def _resolve_writer_stimulus_mode(config: dict[str, Any]) -> str:
    mode = str(config["runtime"].get("pilot_writer_stimulus_mode", "support_and_context"))
    if mode not in {"support_only", "context_only", "support_and_context"}:
        raise ValueError(
            f"Unsupported runtime.pilot_writer_stimulus_mode={mode}. "
            "Expected one of support_only, context_only, support_and_context."
        )
    return mode


def _resolve_writer_context_tokens(config: dict[str, Any]) -> int:
    return max(1, int(config["runtime"].get("pilot_writer_context_tokens", 8)))


def _resolve_writer_gain_margin(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_writer_gain_margin", 0.0))


def _resolve_writer_gain_margin_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_writer_gain_margin_weight", 0.0))


def _resolve_writer_covariance_diversity_weight(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_writer_covariance_diversity_weight", 0.0))


def _resolve_writer_adapter_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("method", {}).get("writer_adapter", {}).get("enabled", False))


def _resolve_writer_adapter_target_modules(config: dict[str, Any]) -> tuple[str, ...]:
    target_modules = config.get("method", {}).get("writer_adapter", {}).get(
        "target_modules",
        ("conditioning_out_proj",),
    )
    normalized = tuple(str(module_name) for module_name in target_modules)
    unsupported = sorted(
        set(normalized) - {"conditioning_out_proj", "encoder_self_attn_out_proj", "encoder_ffn"}
    )
    if unsupported:
        raise ValueError(
            f"Unsupported method.writer_adapter.target_modules={unsupported}. "
            "Expected only conditioning_out_proj, encoder_self_attn_out_proj, and/or encoder_ffn."
        )
    return normalized


def _resolve_writer_adapter_rank(config: dict[str, Any]) -> int:
    return int(config.get("method", {}).get("writer_adapter", {}).get("rank", 0))


def _resolve_writer_adapter_alpha(config: dict[str, Any]) -> float:
    return float(config.get("method", {}).get("writer_adapter", {}).get("alpha", 4.0))


def _resolve_writer_adapter_dropout(config: dict[str, Any]) -> float:
    return float(config.get("method", {}).get("writer_adapter", {}).get("dropout", 0.0))


def _resolve_writer_adapter_learning_rate(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_writer_adapter_learning_rate", 5e-4))


def _resolve_writer_adapter_weight_decay(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_writer_adapter_weight_decay", 0.0))


def _resolve_receiver_lora_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("method", {}).get("receiver_lora", {}).get("enabled", False))


def _resolve_receiver_lora_target_layers(config: dict[str, Any]) -> tuple[int, ...]:
    layers = config.get("method", {}).get("receiver_lora", {}).get("target_layers", [])
    return tuple(int(layer_index) for layer_index in layers)


def _resolve_receiver_lora_target_modules(config: dict[str, Any]) -> tuple[str, ...]:
    target_modules = config.get("method", {}).get("receiver_lora", {}).get(
        "target_modules",
        ("k_proj", "v_proj"),
    )
    normalized = tuple(str(module_name) for module_name in target_modules)
    unsupported = sorted(set(normalized) - {"k_proj", "v_proj"})
    if unsupported:
        raise ValueError(
            f"Unsupported method.receiver_lora.target_modules={unsupported}. "
            "Expected only k_proj and/or v_proj."
        )
    return normalized


def _resolve_receiver_lora_rank(config: dict[str, Any]) -> int:
    return int(config.get("method", {}).get("receiver_lora", {}).get("rank", 0))


def _resolve_receiver_lora_alpha(config: dict[str, Any]) -> float:
    return float(config.get("method", {}).get("receiver_lora", {}).get("alpha", 4.0))


def _resolve_receiver_lora_dropout(config: dict[str, Any]) -> float:
    return float(config.get("method", {}).get("receiver_lora", {}).get("dropout", 0.0))


def _resolve_receiver_lora_learning_rate(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_receiver_lora_learning_rate", 5e-5))


def _resolve_receiver_lora_weight_decay(config: dict[str, Any]) -> float:
    return float(config["runtime"].get("pilot_receiver_lora_weight_decay", 0.0))


def _resolve_train_support_mode(config: dict[str, Any]) -> str:
    mode = str(config["runtime"].get("pilot_train_support_mode", "static_support_rows"))
    if mode not in {"static_support_rows", "episode_bank"}:
        raise ValueError(
            f"Unsupported runtime.pilot_train_support_mode={mode}. "
            "Expected one of static_support_rows, episode_bank."
        )
    return mode


def _resolve_snapshot_steps(config: dict[str, Any], *, train_steps: int, warmup_steps: int) -> list[int]:
    explicit = config["runtime"].get("pilot_snapshot_steps")
    if explicit is not None:
        values = [int(step) for step in explicit]
    else:
        values = [0, warmup_steps, max(warmup_steps, train_steps // 2), train_steps]
    selected = sorted({step for step in values if 0 <= step <= train_steps})
    if 0 not in selected:
        selected.insert(0, 0)
    if train_steps not in selected:
        selected.append(train_steps)
    return selected


def _classification_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluator_type = str(rows[0].get("evaluator_type", "dataset_label_classification")) if rows else "dataset_label_classification"
    total = max(1, len(rows))
    answer_logprob_with_memory = sum(float(row.get("answer_logprob_with_memory", 0.0)) for row in rows) / total
    answer_logprob_without_memory = sum(float(row.get("answer_logprob_without_memory", 0.0)) for row in rows) / total
    delta_answer_logprob = sum(float(row.get("delta_answer_logprob", 0.0)) for row in rows) / total
    prefix_attention_metrics = _aggregate_prefix_attention_metrics(rows)
    if evaluator_type in TASK_GENERATION_EVALUATOR_TYPES:
        task_score = sum(float(row.get("task_score", 0.0)) for row in rows) / total
        exact_match = sum(int(bool(row.get("predicted_correct", False))) for row in rows) / total
        prediction_keys = [
            str(row.get("normalized_prediction", row.get("predicted_text", ""))).strip()
            for row in rows
        ]
        predicted_counts: dict[str, int] = {}
        for prediction_key in prediction_keys:
            predicted_counts[prediction_key] = predicted_counts.get(prediction_key, 0) + 1
        dominant_label_fraction = max(predicted_counts.values(), default=0) / total
        mean_margin = sum(float(row.get("final_margin", 0.0)) for row in rows) / total
        return {
            "accuracy": float(task_score),
            "macro_f1": float(task_score if evaluator_type == "qa_f1" else exact_match),
            "dominant_label_fraction": float(dominant_label_fraction),
            "label_recall_by_class": {},
            "mean_margin": float(mean_margin),
            "exact_match": float(exact_match),
            "answer_logprob_with_memory": float(answer_logprob_with_memory),
            "answer_logprob_without_memory": float(answer_logprob_without_memory),
            "delta_answer_logprob": float(delta_answer_logprob),
            **prefix_attention_metrics,
        }
    labels = [str(row["gold_label"]) for row in rows]
    unique_labels = sorted(set(labels))
    accuracy = sum(int(bool(row["predicted_correct"])) for row in rows) / total
    predicted_counts: dict[str, int] = {label: 0 for label in unique_labels}
    gold_counts: dict[str, int] = {label: 0 for label in unique_labels}
    tp_counts: dict[str, int] = {label: 0 for label in unique_labels}
    for row in rows:
        gold = str(row["gold_label"])
        predicted = str(row["predicted_label"])
        gold_counts[gold] = gold_counts.get(gold, 0) + 1
        predicted_counts[predicted] = predicted_counts.get(predicted, 0) + 1
        if gold == predicted:
            tp_counts[gold] = tp_counts.get(gold, 0) + 1
    f1_values: list[float] = []
    recall_by_class: dict[str, float] = {}
    for label in unique_labels:
        tp = tp_counts.get(label, 0)
        fp = predicted_counts.get(label, 0) - tp
        fn = gold_counts.get(label, 0) - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
        recall_by_class[label] = recall
    dominant_label_fraction = max(predicted_counts.values(), default=0) / total
    mean_margin = sum(float(row["final_margin"]) for row in rows) / total
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(sum(f1_values) / max(1, len(f1_values))),
        "dominant_label_fraction": float(dominant_label_fraction),
        "label_recall_by_class": recall_by_class,
        "mean_margin": float(mean_margin),
        "exact_match": float(accuracy),
        "answer_logprob_with_memory": float(answer_logprob_with_memory),
        "answer_logprob_without_memory": float(answer_logprob_without_memory),
        "delta_answer_logprob": float(delta_answer_logprob),
        **prefix_attention_metrics,
    }


def _write_snapshot_eval(
    *,
    snapshot_dir: Path,
    step: int,
    arm_alias: str,
    arm: str,
    writer_memory_control: str,
    prompt_variant: str,
    support_serialization_variant: str,
    support_text_block: str,
    case_rows: list[dict[str, Any]],
    prefix_stats: dict[str, float] | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    task_case_dump_path = snapshot_dir / "task_case_dump.jsonl"
    write_jsonl(task_case_dump_path, case_rows)
    metrics = _classification_metrics_from_rows(case_rows)
    scalar_prefix_stats = _prefix_scalar_summary(prefix_stats or {})
    payload = {
        "training_stage": "shared_injection_snapshot_eval",
        "pilot_arm_alias": arm_alias,
        "shared_injection_arm": arm,
        "writer_memory_control": writer_memory_control,
        "pilot_memory_path_variant": str((prefix_stats or {}).get("pilot_memory_path_variant", "single_level")),
        "pilot_reader_context_mode": str((prefix_stats or {}).get("pilot_reader_context_mode", "prompt_summary")),
        "pilot_projector_token_source": str((prefix_stats or {}).get("pilot_projector_token_source", "writer_slots")),
        "pilot_reader_num_queries": float((prefix_stats or {}).get("reader_num_queries", 0.0)),
        "pilot_fuser_short_slots": float((prefix_stats or {}).get("memory_short_slots", 0.0)),
        "pilot_support_encoder_mode": str((prefix_stats or {}).get("pilot_support_encoder_mode", "pooled_block")),
        "prompt_variant": prompt_variant,
        "support_serialization_variant": support_serialization_variant,
        "support_text_block_chars": len(support_text_block),
        "step": int(step),
        "task_case_dump_rows": len(case_rows),
        "task_case_dump_path": str(task_case_dump_path.resolve()),
        "task_evaluator_type": str(case_rows[0].get("evaluator_type", "dataset_label_classification")) if case_rows else "dataset_label_classification",
        "best_adapt_task_score": float(metrics["accuracy"]),
        "best_adapt_macro_f1": float(metrics["macro_f1"]),
        "best_adapt_exact_match": float(metrics.get("exact_match", 0.0)),
        "best_adapt_task_margin": float(metrics["mean_margin"]),
        "answer_logprob_with_memory": float(metrics.get("answer_logprob_with_memory", 0.0)),
        "answer_logprob_without_memory": float(metrics.get("answer_logprob_without_memory", 0.0)),
        "delta_answer_logprob": float(metrics.get("delta_answer_logprob", 0.0)),
        "dominant_label_fraction": float(metrics["dominant_label_fraction"]),
        "label_recall_by_class": metrics["label_recall_by_class"],
        "prefix_attention_mass_mean": float(metrics.get("prefix_attention_mass_mean", 0.0)),
        "prefix_to_content_attention_ratio_mean": float(
            metrics.get("prefix_to_content_attention_ratio_mean", 0.0)
        ),
        "gold_prefix_attention_mass_mean": float(metrics.get("gold_prefix_attention_mass_mean", 0.0)),
        "competitor_prefix_attention_mass_mean": float(
            metrics.get("competitor_prefix_attention_mass_mean", 0.0)
        ),
        "prefix_attention_mass_mean_by_layer": dict(
            metrics.get("prefix_attention_mass_mean_by_layer", {})
        ),
        "prefix_to_content_attention_ratio_mean_by_layer": dict(
            metrics.get("prefix_to_content_attention_ratio_mean_by_layer", {})
        ),
        "prefix_attention_nontrivial_layer_count": int(
            metrics.get("prefix_attention_nontrivial_layer_count", 0)
        ),
        "prefix_artifact_stats": prefix_stats or {},
        "prefix_tokens": int(scalar_prefix_stats.get("prefix_tokens", 0.0)),
        "prefix_l2": float(scalar_prefix_stats.get("prefix_l2", 0.0)),
        "prefix_slot_norm_mean": float(scalar_prefix_stats.get("prefix_slot_norm_mean", 0.0)),
        "prefix_slot_norm_std": float(scalar_prefix_stats.get("prefix_slot_norm_std", 0.0)),
        "prefix_slot_norm_max": float(scalar_prefix_stats.get("prefix_slot_norm_max", 0.0)),
        "writer_memory_l2": float(scalar_prefix_stats.get("writer_memory_l2", 0.0)),
        "writer_slot_norm_mean": float(scalar_prefix_stats.get("writer_slot_norm_mean", 0.0)),
        "writer_slot_norm_std": float(scalar_prefix_stats.get("writer_slot_norm_std", 0.0)),
        "writer_slot_norm_max": float(scalar_prefix_stats.get("writer_slot_norm_max", 0.0)),
        "memory_long_l2": float(scalar_prefix_stats.get("memory_long_l2", 0.0)),
        "memory_long_slots": float(scalar_prefix_stats.get("memory_long_slots", 0.0)),
        "memory_long_slot_norm_mean": float(scalar_prefix_stats.get("memory_long_slot_norm_mean", 0.0)),
        "memory_long_slot_norm_std": float(scalar_prefix_stats.get("memory_long_slot_norm_std", 0.0)),
        "memory_long_slot_norm_max": float(scalar_prefix_stats.get("memory_long_slot_norm_max", 0.0)),
        "memory_short_l2": float(scalar_prefix_stats.get("memory_short_l2", 0.0)),
        "memory_short_slots": float(scalar_prefix_stats.get("memory_short_slots", 0.0)),
        "memory_short_slot_norm_mean": float(scalar_prefix_stats.get("memory_short_slot_norm_mean", 0.0)),
        "memory_short_slot_norm_std": float(scalar_prefix_stats.get("memory_short_slot_norm_std", 0.0)),
        "memory_short_slot_norm_max": float(scalar_prefix_stats.get("memory_short_slot_norm_max", 0.0)),
        "memory_short_pairwise_cosine_mean": float(
            scalar_prefix_stats.get("memory_short_pairwise_cosine_mean", 0.0)
        ),
        "reader_num_queries": float(scalar_prefix_stats.get("reader_num_queries", 0.0)),
        "reader_slot_count": float(scalar_prefix_stats.get("reader_slot_count", 0.0)),
        "reader_attention_entropy_mean": float(
            scalar_prefix_stats.get("reader_attention_entropy_mean", 0.0)
        ),
        "reader_attention_entropy_min": float(scalar_prefix_stats.get("reader_attention_entropy_min", 0.0)),
        "reader_attention_entropy_max": float(scalar_prefix_stats.get("reader_attention_entropy_max", 0.0)),
        "reader_attention_pairwise_cosine_mean": float(
            scalar_prefix_stats.get("reader_attention_pairwise_cosine_mean", 0.0)
        ),
        "reader_slot_coverage_fraction": float(
            scalar_prefix_stats.get("reader_slot_coverage_fraction", 0.0)
        ),
        "support_item_count": float(scalar_prefix_stats.get("support_item_count", 0.0)),
        "support_item_hidden_l2": float(scalar_prefix_stats.get("support_item_hidden_l2", 0.0)),
        "support_item_hidden_norm_mean": float(scalar_prefix_stats.get("support_item_hidden_norm_mean", 0.0)),
        "support_item_hidden_norm_std": float(scalar_prefix_stats.get("support_item_hidden_norm_std", 0.0)),
        "support_item_hidden_norm_max": float(scalar_prefix_stats.get("support_item_hidden_norm_max", 0.0)),
        "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path.resolve()),
        **scalar_prefix_stats,
    }
    write_json(snapshot_dir / "metrics.json", payload)
    return payload


def _slot_norm_summary(slots: torch.Tensor) -> dict[str, float]:
    slots_cpu = slots.detach().to(dtype=torch.float32, device="cpu")
    slot_norms = slots_cpu.norm(dim=-1)
    return {
        "l2": float(slots_cpu.norm().item()),
        "slot_norm_mean": float(slot_norms.mean().item()),
        "slot_norm_std": float(slot_norms.std(unbiased=False).item()),
        "slot_norm_max": float(slot_norms.max().item()),
    }


def _pairwise_cosine_mean(slots: torch.Tensor | None) -> float:
    if slots is None or slots.numel() == 0:
        return 0.0
    slots_fp32 = slots.detach().to(dtype=torch.float32)
    if slots_fp32.ndim != 3 or slots_fp32.shape[1] < 2:
        return 0.0
    normalized = F.normalize(slots_fp32, dim=-1, eps=1e-8)
    cosine = torch.matmul(normalized, normalized.transpose(1, 2))
    slot_count = cosine.shape[1]
    mask = ~torch.eye(slot_count, dtype=torch.bool, device=cosine.device).unsqueeze(0)
    if not bool(mask.any()):
        return 0.0
    return float(cosine.masked_select(mask).mean().item())


PREFIX_SCALAR_KEYS = (
    "prefix_tokens",
    "prefix_l2",
    "prefix_slot_norm_mean",
    "prefix_slot_norm_std",
    "prefix_slot_norm_max",
    "writer_memory_l2",
    "writer_slot_norm_mean",
    "writer_slot_norm_std",
    "writer_slot_norm_max",
    "support_item_count",
    "support_item_hidden_l2",
    "support_item_hidden_norm_mean",
    "support_item_hidden_norm_std",
    "support_item_hidden_norm_max",
    "writer_support_state_count",
    "writer_support_hidden_l2",
    "writer_support_hidden_norm_mean",
    "writer_support_hidden_norm_std",
    "writer_support_hidden_norm_max",
    "writer_context_token_count",
    "writer_context_hidden_l2",
    "writer_context_hidden_norm_mean",
    "writer_context_hidden_norm_std",
    "writer_context_hidden_norm_max",
    "memory_long_l2",
    "memory_long_slots",
    "memory_long_slot_norm_mean",
    "memory_long_slot_norm_std",
    "memory_long_slot_norm_max",
    "memory_long_slot_energy_cv",
    "memory_long_slot_variance_cv",
    "memory_long_common_mode_vector_l2",
    "memory_long_common_mode_energy_ratio",
    "memory_long_centered_effective_rank",
    "memory_long_top1_top2_ratio",
    "memory_long_centered_top1_top2_ratio",
    "memory_long_pairwise_cosine_mean",
    "memory_long_singular_value_top1",
    "memory_long_singular_value_top2",
    "memory_long_singular_value_top3",
    "memory_long_slot_norm_histogram_min",
    "memory_long_slot_norm_histogram_max",
    "memory_long_slot_energy_histogram_min",
    "memory_long_slot_energy_histogram_max",
    "memory_short_l2",
    "memory_short_slots",
    "memory_short_slot_norm_mean",
    "memory_short_slot_norm_std",
    "memory_short_slot_norm_max",
    "memory_short_pairwise_cosine_mean",
    "reader_num_queries",
    "reader_slot_count",
    "reader_base_query_pairwise_cosine_mean",
    "reader_conditioned_query_pairwise_cosine_mean",
    "reader_base_query_norm_mean",
    "reader_conditioned_query_norm_mean",
    "reader_context_shift_norm_mean",
    "reader_context_overwrite_ratio",
    "reader_qk_logit_mean",
    "reader_qk_logit_std",
    "reader_qk_logit_range",
    "reader_qk_logit_pairwise_cosine_mean",
    "reader_attention_entropy_mean",
    "reader_attention_entropy_min",
    "reader_attention_entropy_max",
    "reader_attention_pairwise_cosine_mean",
    "reader_slot_coverage_fraction",
    "reader_argmax_mass_mean",
    "reader_argmax_mass_std",
    "reader_value_projected_effective_rank",
    "reader_value_projected_pairwise_cosine_mean",
    "reader_readout_pairwise_cosine_mean",
    "reader_readout_effective_rank",
    "reader_readout_centered_effective_rank",
    "reader_query_to_slot_top1_agreement_rate",
    "reader_query_to_slot_top2_coverage_fraction",
    "fuser_input_pairwise_cosine_mean",
    "fuser_input_effective_rank",
    "fuser_output_pairwise_cosine_mean",
    "fuser_output_effective_rank",
    "fuser_rank_gain_over_readout",
    "fuser_diversity_without_semantic_gain_flag",
    "fuser_short_query_pairwise_cosine_mean",
    "fuser_linear_output_singular_value_top1",
    "fuser_linear_output_singular_value_top2",
    "fuser_linear_output_singular_value_top3",
    "projected_memory_effective_rank",
    "answer_logprob_with_memory",
    "answer_logprob_without_memory",
    "delta_answer_logprob",
)


def _finite_values(tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None or tensor.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    tensor_fp32 = tensor.detach().to(dtype=torch.float32, device="cpu")
    return tensor_fp32[torch.isfinite(tensor_fp32)]


def _coefficient_of_variation(values: torch.Tensor | None) -> float:
    finite = _finite_values(values)
    if finite.numel() == 0:
        return 0.0
    mean_abs = finite.abs().mean().item()
    if mean_abs <= 1e-8:
        return 0.0
    return float(finite.std(unbiased=False).item() / mean_abs)


def _histogram_counts(values: torch.Tensor | None, *, bins: int = 4) -> list[float]:
    finite = _finite_values(values)
    if finite.numel() == 0:
        return [0.0 for _ in range(max(1, bins))]
    min_value = float(finite.min().item())
    max_value = float(finite.max().item())
    if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-8):
        counts = [0.0 for _ in range(max(1, bins))]
        counts[0] = float(finite.numel())
        return counts
    histogram = torch.histc(finite, bins=max(1, bins), min=min_value, max=max_value)
    return [float(value) for value in histogram.tolist()]


def _top_singular_values(tensor: torch.Tensor | None, *, top_k: int = 3) -> list[float]:
    if tensor is None or tensor.numel() == 0:
        return [0.0 for _ in range(max(1, top_k))]
    matrix = tensor.detach().to(dtype=torch.float32)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    elif matrix.ndim != 3:
        raise ValueError(
            f"Expected rank-2 or rank-3 tensor for singular values, got shape={tuple(matrix.shape)}."
        )
    singular_values = torch.linalg.svdvals(matrix)
    means = singular_values.mean(dim=0)
    padded = torch.zeros(max(1, top_k), dtype=torch.float32, device=means.device)
    padded[: min(len(means), max(1, top_k))] = means[: max(1, top_k)]
    return [float(value) for value in padded.cpu().tolist()]


def _top1_top2_ratio(tensor: torch.Tensor | None) -> float:
    top_values = _top_singular_values(tensor, top_k=2)
    top1 = float(top_values[0]) if top_values else 0.0
    top2 = float(top_values[1]) if len(top_values) > 1 else 0.0
    if top1 <= 0.0:
        return 0.0
    return float(top1 / max(top2, 1e-8))


def _center_slot_matrix(slots: torch.Tensor | None) -> torch.Tensor | None:
    if slots is None or slots.numel() == 0:
        return None
    if slots.ndim == 2:
        slots = slots.unsqueeze(0)
    elif slots.ndim != 3:
        raise ValueError(f"Expected rank-2 or rank-3 slot tensor, got shape={tuple(slots.shape)}.")
    centered = slots.detach().to(dtype=torch.float32) - slots.detach().to(dtype=torch.float32).mean(
        dim=1,
        keepdim=True,
    )
    return centered


def _common_mode_vector_l2(slots: torch.Tensor | None) -> float:
    if slots is None or slots.numel() == 0:
        return 0.0
    slots_fp32 = slots.detach().to(dtype=torch.float32)
    if slots_fp32.ndim == 2:
        slots_fp32 = slots_fp32.unsqueeze(0)
    elif slots_fp32.ndim != 3:
        raise ValueError(f"Expected rank-2 or rank-3 slot tensor, got shape={tuple(slots_fp32.shape)}.")
    slot_mean = slots_fp32.mean(dim=1)
    return float(slot_mean.norm(dim=-1).mean().item())


def _common_mode_energy_ratio(slots: torch.Tensor | None) -> float:
    if slots is None or slots.numel() == 0:
        return 0.0
    slots_fp32 = slots.detach().to(dtype=torch.float32)
    if slots_fp32.ndim == 2:
        slots_fp32 = slots_fp32.unsqueeze(0)
    elif slots_fp32.ndim != 3:
        raise ValueError(f"Expected rank-2 or rank-3 slot tensor, got shape={tuple(slots_fp32.shape)}.")
    common_mode = slots_fp32.mean(dim=1, keepdim=True).expand_as(slots_fp32)
    common_energy = common_mode.square().sum(dim=(1, 2))
    total_energy = slots_fp32.square().sum(dim=(1, 2))
    ratio = common_energy / total_energy.clamp_min(1e-8)
    return float(ratio.mean().item())


def _reader_pre_attention_stats(
    *,
    base_queries: torch.Tensor | None,
    conditioned_queries: torch.Tensor | None,
    attention_logits: torch.Tensor | None,
) -> dict[str, float]:
    zero_stats = {
        "reader_base_query_pairwise_cosine_mean": 0.0,
        "reader_conditioned_query_pairwise_cosine_mean": 0.0,
        "reader_base_query_norm_mean": 0.0,
        "reader_conditioned_query_norm_mean": 0.0,
        "reader_context_shift_norm_mean": 0.0,
        "reader_context_overwrite_ratio": 0.0,
        "reader_qk_logit_mean": 0.0,
        "reader_qk_logit_std": 0.0,
        "reader_qk_logit_range": 0.0,
        "reader_qk_logit_pairwise_cosine_mean": 0.0,
    }
    if base_queries is None or conditioned_queries is None:
        return zero_stats
    shift = conditioned_queries - base_queries
    base_norm_mean = float(base_queries.detach().to(dtype=torch.float32).norm(dim=-1).mean().item())
    conditioned_norm_mean = float(
        conditioned_queries.detach().to(dtype=torch.float32).norm(dim=-1).mean().item()
    )
    shift_norm_mean = float(shift.detach().to(dtype=torch.float32).norm(dim=-1).mean().item())
    logits_mean = 0.0
    logits_std = 0.0
    logits_range = 0.0
    logits_pairwise = 0.0
    if attention_logits is not None and attention_logits.numel() > 0:
        logits_source = attention_logits.detach().to(dtype=torch.float32)
        if logits_source.ndim == 4:
            logits_source = logits_source.mean(dim=1)
        finite_logits = _finite_values(logits_source)
        if finite_logits.numel() > 0:
            logits_mean = float(finite_logits.mean().item())
            logits_std = float(finite_logits.std(unbiased=False).item())
            logits_range = float((finite_logits.max() - finite_logits.min()).item())
        logits_pairwise = _pairwise_cosine_mean(
            torch.where(torch.isfinite(logits_source), logits_source, torch.zeros_like(logits_source))
        )
    return {
        "reader_base_query_pairwise_cosine_mean": _pairwise_cosine_mean(base_queries),
        "reader_conditioned_query_pairwise_cosine_mean": _pairwise_cosine_mean(conditioned_queries),
        "reader_base_query_norm_mean": base_norm_mean,
        "reader_conditioned_query_norm_mean": conditioned_norm_mean,
        "reader_context_shift_norm_mean": shift_norm_mean,
        "reader_context_overwrite_ratio": (
            0.0 if base_norm_mean <= 1e-8 else float(shift_norm_mean / base_norm_mean)
        ),
        "reader_qk_logit_mean": logits_mean,
        "reader_qk_logit_std": logits_std,
        "reader_qk_logit_range": logits_range,
        "reader_qk_logit_pairwise_cosine_mean": logits_pairwise,
    }


def _reader_top1_agreement_rate(attention: torch.Tensor) -> float:
    top1_indices = attention.argmax(dim=-1)
    rates: list[float] = []
    query_count = max(1, int(attention.shape[1]))
    slot_count = max(1, int(attention.shape[2]))
    for batch_index in range(top1_indices.shape[0]):
        counts = torch.bincount(top1_indices[batch_index], minlength=slot_count)
        rates.append(float(counts.max().item()) / query_count)
    return float(sum(rates) / max(1, len(rates)))


def _reader_top2_coverage_fraction(attention: torch.Tensor) -> float:
    slot_count = max(1, int(attention.shape[2]))
    top_k = min(2, slot_count)
    topk_indices = torch.topk(attention, k=top_k, dim=-1).indices
    fractions: list[float] = []
    for batch_index in range(topk_indices.shape[0]):
        unique_slots = torch.unique(topk_indices[batch_index].reshape(-1))
        fractions.append(float(unique_slots.numel()) / slot_count)
    return float(sum(fractions) / max(1, len(fractions)))


def _memory_long_geometry_stats(memory_long: torch.Tensor | None) -> dict[str, Any]:
    zero_stats: dict[str, Any] = {
        "memory_long_slot_energy_cv": 0.0,
        "memory_long_slot_variance_cv": 0.0,
        "memory_long_common_mode_vector_l2": 0.0,
        "memory_long_common_mode_energy_ratio": 0.0,
        "memory_long_centered_effective_rank": 0.0,
        "memory_long_top1_top2_ratio": 0.0,
        "memory_long_centered_top1_top2_ratio": 0.0,
        "memory_long_pairwise_cosine_mean": 0.0,
        "memory_long_singular_value_top1": 0.0,
        "memory_long_singular_value_top2": 0.0,
        "memory_long_singular_value_top3": 0.0,
        "memory_long_slot_norm_histogram_counts": [0.0, 0.0, 0.0, 0.0],
        "memory_long_slot_norm_histogram_min": 0.0,
        "memory_long_slot_norm_histogram_max": 0.0,
        "memory_long_slot_energy_histogram_counts": [0.0, 0.0, 0.0, 0.0],
        "memory_long_slot_energy_histogram_min": 0.0,
        "memory_long_slot_energy_histogram_max": 0.0,
    }
    if memory_long is None or memory_long.numel() == 0:
        return zero_stats
    memory_long_fp32 = memory_long.detach().to(dtype=torch.float32, device="cpu")
    slot_norms = memory_long_fp32.norm(dim=-1)
    slot_energy = memory_long_fp32.pow(2).sum(dim=-1)
    slot_variance = memory_long_fp32.var(dim=-1, unbiased=False)
    centered_memory_long = _center_slot_matrix(memory_long_fp32)
    singular_values = _top_singular_values(memory_long_fp32, top_k=3)
    return {
        "memory_long_slot_energy_cv": _coefficient_of_variation(slot_energy),
        "memory_long_slot_variance_cv": _coefficient_of_variation(slot_variance),
        "memory_long_common_mode_vector_l2": _common_mode_vector_l2(memory_long_fp32),
        "memory_long_common_mode_energy_ratio": _common_mode_energy_ratio(memory_long_fp32),
        "memory_long_centered_effective_rank": _effective_rank(centered_memory_long),
        "memory_long_top1_top2_ratio": _top1_top2_ratio(memory_long_fp32),
        "memory_long_centered_top1_top2_ratio": _top1_top2_ratio(centered_memory_long),
        "memory_long_pairwise_cosine_mean": _pairwise_cosine_mean(memory_long_fp32),
        "memory_long_singular_value_top1": singular_values[0],
        "memory_long_singular_value_top2": singular_values[1],
        "memory_long_singular_value_top3": singular_values[2],
        "memory_long_slot_norm_histogram_counts": _histogram_counts(slot_norms, bins=4),
        "memory_long_slot_norm_histogram_min": float(slot_norms.min().item()),
        "memory_long_slot_norm_histogram_max": float(slot_norms.max().item()),
        "memory_long_slot_energy_histogram_counts": _histogram_counts(slot_energy, bins=4),
        "memory_long_slot_energy_histogram_min": float(slot_energy.min().item()),
        "memory_long_slot_energy_histogram_max": float(slot_energy.max().item()),
    }


def _fuser_stats(
    *,
    fuser: MemoryFuser | None,
    fuser_input: torch.Tensor | None,
    memory_short: torch.Tensor | None,
) -> dict[str, float]:
    arch = "" if fuser is None else str(getattr(fuser, "arch", ""))
    short_queries = None
    if arch == "resampler" and fuser is not None:
        short_queries = getattr(fuser, "short_queries", None)
    input_rank = _effective_rank(fuser_input)
    output_rank = _effective_rank(memory_short)
    input_pairwise = _pairwise_cosine_mean(fuser_input)
    output_high_rank_threshold = 0.75 * float(memory_short.shape[1]) if memory_short is not None else 0.0
    diversity_without_semantic_gain = bool(
        memory_short is not None
        and output_rank >= max(2.5, output_high_rank_threshold)
        and input_rank <= 1.5
        and input_pairwise >= 0.95
    )
    linear_singular_values = _top_singular_values(memory_short, top_k=3) if arch == "linear" else [0.0, 0.0, 0.0]
    return {
        "fuser_input_pairwise_cosine_mean": input_pairwise,
        "fuser_input_effective_rank": input_rank,
        "fuser_output_pairwise_cosine_mean": _pairwise_cosine_mean(memory_short),
        "fuser_output_effective_rank": output_rank,
        "fuser_rank_gain_over_readout": float(output_rank - input_rank),
        "fuser_diversity_without_semantic_gain_flag": float(diversity_without_semantic_gain),
        "fuser_short_query_pairwise_cosine_mean": _pairwise_cosine_mean(
            None if short_queries is None else short_queries.unsqueeze(0)
        ),
        "fuser_linear_output_singular_value_top1": linear_singular_values[0],
        "fuser_linear_output_singular_value_top2": linear_singular_values[1],
        "fuser_linear_output_singular_value_top3": linear_singular_values[2],
    }


def _reader_attention_stats(
    reader_attention: torch.Tensor | None,
    *,
    memory_long_slots: int,
    reader_value_projected_slots: torch.Tensor | None = None,
    reader_readouts: torch.Tensor | None = None,
) -> dict[str, Any]:
    if reader_attention is None or reader_attention.numel() == 0:
        return {
            "reader_num_queries": 0.0,
            "reader_slot_count": float(memory_long_slots),
            "reader_attention_entropy_mean": 0.0,
            "reader_attention_entropy_min": 0.0,
            "reader_attention_entropy_max": 0.0,
            "reader_attention_pairwise_cosine_mean": 0.0,
            "reader_slot_coverage_fraction": 0.0,
            "reader_argmax_mass_mean": 0.0,
            "reader_argmax_mass_std": 0.0,
            "reader_value_projected_effective_rank": 0.0,
            "reader_value_projected_pairwise_cosine_mean": 0.0,
            "reader_readout_pairwise_cosine_mean": 0.0,
            "reader_readout_effective_rank": 0.0,
            "reader_readout_centered_effective_rank": 0.0,
            "reader_query_to_slot_top1_agreement_rate": 0.0,
            "reader_query_to_slot_top2_coverage_fraction": 0.0,
            "reader_query_entropy_by_query": [],
        }
    attention = reader_attention.detach().to(dtype=torch.float32)
    probs = attention.clamp_min(1e-8)
    query_entropy = -(probs * probs.log()).sum(dim=-1)
    pairwise = _pairwise_cosine_mean(attention)
    slot_attention = attention.max(dim=1).values
    coverage_threshold = max(0.05, 1.0 / max(1, int(memory_long_slots)))
    coverage_fraction = float((slot_attention > coverage_threshold).to(dtype=torch.float32).mean().item())
    argmax_mass = attention.max(dim=-1).values
    return {
        "reader_num_queries": float(attention.shape[1]),
        "reader_slot_count": float(memory_long_slots),
        "reader_attention_entropy_mean": float(query_entropy.mean().item()),
        "reader_attention_entropy_min": float(query_entropy.min().item()),
        "reader_attention_entropy_max": float(query_entropy.max().item()),
        "reader_attention_pairwise_cosine_mean": pairwise,
        "reader_slot_coverage_fraction": coverage_fraction,
        "reader_argmax_mass_mean": float(argmax_mass.mean().item()),
        "reader_argmax_mass_std": float(argmax_mass.std(unbiased=False).item()),
        "reader_value_projected_effective_rank": _effective_rank(reader_value_projected_slots),
        "reader_value_projected_pairwise_cosine_mean": _pairwise_cosine_mean(
            reader_value_projected_slots
        ),
        "reader_readout_pairwise_cosine_mean": _pairwise_cosine_mean(reader_readouts),
        "reader_readout_effective_rank": _effective_rank(reader_readouts),
        "reader_readout_centered_effective_rank": _effective_rank(
            _center_slot_matrix(reader_readouts)
        ),
        "reader_query_to_slot_top1_agreement_rate": _reader_top1_agreement_rate(attention),
        "reader_query_to_slot_top2_coverage_fraction": _reader_top2_coverage_fraction(attention),
        "reader_query_entropy_by_query": [
            float(value)
            for value in query_entropy.mean(dim=0).detach().to(device="cpu").tolist()
        ],
    }


def _prefix_scalar_summary(prefix_stats: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(prefix_stats.get(key, 0.0))
        for key in PREFIX_SCALAR_KEYS
    }


def _prefix_stats(
    prefix_embeddings: torch.Tensor | None = None,
    *,
    layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None = None,
    memory_slots: torch.Tensor | None = None,
    memory_long: torch.Tensor | None = None,
    memory_short: torch.Tensor | None = None,
    support_item_states: torch.Tensor | None = None,
    writer_support_states: torch.Tensor | None = None,
    writer_context_states: torch.Tensor | None = None,
    reader_attention: torch.Tensor | None = None,
    reader_base_queries: torch.Tensor | None = None,
    reader_conditioned_queries: torch.Tensor | None = None,
    reader_attention_logits: torch.Tensor | None = None,
    reader_value_projected_slots: torch.Tensor | None = None,
    reader_readouts: torch.Tensor | None = None,
    fuser: MemoryFuser | None = None,
    memory_path_variant: str = "single_level",
    projector_token_source: str = "writer_slots",
    prefix_source_mode: str = "writer",
    deep_prefix_init_mode: str = "random",
) -> dict[str, Any]:
    memory_long = memory_slots if memory_long is None else memory_long
    memory_short = memory_long if memory_short is None else memory_short
    writer_stats = {
        "writer_memory_l2": 0.0,
        "writer_slot_norm_mean": 0.0,
        "writer_slot_norm_std": 0.0,
        "writer_slot_norm_max": 0.0,
    }
    if memory_long is not None and memory_long.numel() > 0:
        writer_summary = _slot_norm_summary(memory_long)
        writer_stats = {
            "writer_memory_l2": float(writer_summary["l2"]),
            "writer_slot_norm_mean": float(writer_summary["slot_norm_mean"]),
            "writer_slot_norm_std": float(writer_summary["slot_norm_std"]),
            "writer_slot_norm_max": float(writer_summary["slot_norm_max"]),
        }
    memory_long_stats = {
        "memory_long_l2": 0.0,
        "memory_long_slots": 0.0,
        "memory_long_slot_norm_mean": 0.0,
        "memory_long_slot_norm_std": 0.0,
        "memory_long_slot_norm_max": 0.0,
    }
    if memory_long is not None and memory_long.numel() > 0:
        memory_long_summary = _slot_norm_summary(memory_long)
        memory_long_stats = {
            "memory_long_l2": float(memory_long_summary["l2"]),
            "memory_long_slots": float(memory_long.shape[1]),
            "memory_long_slot_norm_mean": float(memory_long_summary["slot_norm_mean"]),
            "memory_long_slot_norm_std": float(memory_long_summary["slot_norm_std"]),
            "memory_long_slot_norm_max": float(memory_long_summary["slot_norm_max"]),
        }
    memory_short_stats = {
        "memory_short_l2": 0.0,
        "memory_short_slots": 0.0,
        "memory_short_slot_norm_mean": 0.0,
        "memory_short_slot_norm_std": 0.0,
        "memory_short_slot_norm_max": 0.0,
        "memory_short_pairwise_cosine_mean": 0.0,
    }
    if memory_short is not None and memory_short.numel() > 0:
        memory_short_summary = _slot_norm_summary(memory_short)
        memory_short_stats = {
            "memory_short_l2": float(memory_short_summary["l2"]),
            "memory_short_slots": float(memory_short.shape[1]),
            "memory_short_slot_norm_mean": float(memory_short_summary["slot_norm_mean"]),
            "memory_short_slot_norm_std": float(memory_short_summary["slot_norm_std"]),
            "memory_short_slot_norm_max": float(memory_short_summary["slot_norm_max"]),
            "memory_short_pairwise_cosine_mean": _pairwise_cosine_mean(memory_short),
        }
    support_stats = {
        "support_item_count": 0.0,
        "support_item_hidden_l2": 0.0,
        "support_item_hidden_norm_mean": 0.0,
        "support_item_hidden_norm_std": 0.0,
        "support_item_hidden_norm_max": 0.0,
    }
    if support_item_states is not None and support_item_states.numel() > 0:
        support_summary = _slot_norm_summary(support_item_states)
        support_stats = {
            "support_item_count": float(support_item_states.shape[1]),
            "support_item_hidden_l2": float(support_summary["l2"]),
            "support_item_hidden_norm_mean": float(support_summary["slot_norm_mean"]),
            "support_item_hidden_norm_std": float(support_summary["slot_norm_std"]),
            "support_item_hidden_norm_max": float(support_summary["slot_norm_max"]),
        }
    writer_support_stats = {
        "writer_support_state_count": 0.0,
        "writer_support_hidden_l2": 0.0,
        "writer_support_hidden_norm_mean": 0.0,
        "writer_support_hidden_norm_std": 0.0,
        "writer_support_hidden_norm_max": 0.0,
    }
    if writer_support_states is not None and writer_support_states.numel() > 0:
        writer_support_summary = _slot_norm_summary(writer_support_states)
        writer_support_stats = {
            "writer_support_state_count": float(writer_support_states.shape[1]),
            "writer_support_hidden_l2": float(writer_support_summary["l2"]),
            "writer_support_hidden_norm_mean": float(writer_support_summary["slot_norm_mean"]),
            "writer_support_hidden_norm_std": float(writer_support_summary["slot_norm_std"]),
            "writer_support_hidden_norm_max": float(writer_support_summary["slot_norm_max"]),
        }
    writer_context_stats = {
        "writer_context_token_count": 0.0,
        "writer_context_hidden_l2": 0.0,
        "writer_context_hidden_norm_mean": 0.0,
        "writer_context_hidden_norm_std": 0.0,
        "writer_context_hidden_norm_max": 0.0,
    }
    if writer_context_states is not None and writer_context_states.numel() > 0:
        writer_context_summary = _slot_norm_summary(writer_context_states)
        writer_context_stats = {
            "writer_context_token_count": float(writer_context_states.shape[1]),
            "writer_context_hidden_l2": float(writer_context_summary["l2"]),
            "writer_context_hidden_norm_mean": float(writer_context_summary["slot_norm_mean"]),
            "writer_context_hidden_norm_std": float(writer_context_summary["slot_norm_std"]),
            "writer_context_hidden_norm_max": float(writer_context_summary["slot_norm_max"]),
        }
    reader_pre_stats = _reader_pre_attention_stats(
        base_queries=reader_base_queries,
        conditioned_queries=reader_conditioned_queries,
        attention_logits=reader_attention_logits,
    )
    reader_stats = _reader_attention_stats(
        reader_attention,
        memory_long_slots=0 if memory_long is None else int(memory_long.shape[1]),
        reader_value_projected_slots=reader_value_projected_slots,
        reader_readouts=reader_readouts,
    )
    memory_long_geometry_stats = _memory_long_geometry_stats(memory_long)
    fuser_geometry_stats = _fuser_stats(
        fuser=fuser,
        fuser_input=reader_readouts,
        memory_short=memory_short,
    )
    projected_memory_effective_rank = (
        _effective_rank(prefix_embeddings)
        if layer_prefix_hidden_by_layer is None
        else _effective_rank(
            torch.cat(
                [layer_prefix_hidden_by_layer[layer_index] for layer_index in sorted(layer_prefix_hidden_by_layer)],
                dim=1,
            )
        )
    )
    if layer_prefix_hidden_by_layer is not None:
        if not layer_prefix_hidden_by_layer:
            return {
                "pilot_memory_path_variant": memory_path_variant,
                "pilot_projector_token_source": projector_token_source,
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_prefix_source_mode": prefix_source_mode,
                "pilot_deep_prefix_init_mode": deep_prefix_init_mode,
                "pilot_support_encoder_mode": "pooled_block",
                "active_prefix_layers": [],
                "layer_hidden_l2_by_layer": {},
                "layer_slot_norm_mean_by_layer": {},
                "layer_slot_norm_std_by_layer": {},
                "layer_slot_norm_max_by_layer": {},
                "layer_key_l2_by_layer": {},
                "layer_value_l2_by_layer": {},
                "prefix_tokens": 0.0,
                "prefix_l2": 0.0,
                "prefix_slot_norm_mean": 0.0,
                "prefix_slot_norm_std": 0.0,
                "prefix_slot_norm_max": 0.0,
                **writer_stats,
                **memory_long_stats,
                **memory_long_geometry_stats,
                **memory_short_stats,
                **support_stats,
                **writer_support_stats,
                **writer_context_stats,
                **reader_pre_stats,
                **reader_stats,
                **fuser_geometry_stats,
                "projected_memory_effective_rank": float(projected_memory_effective_rank),
            }
        ordered_layers = sorted(int(layer_index) for layer_index in layer_prefix_hidden_by_layer)
        stacked = torch.stack(
            [layer_prefix_hidden_by_layer[layer_index] for layer_index in ordered_layers],
            dim=1,
        )
        stacked_cpu = stacked.detach().to(dtype=torch.float32, device="cpu")
        slot_norms = stacked_cpu.norm(dim=-1)
        layer_hidden_l2_by_layer: dict[str, float] = {}
        layer_slot_norm_mean_by_layer: dict[str, float] = {}
        layer_slot_norm_std_by_layer: dict[str, float] = {}
        layer_slot_norm_max_by_layer: dict[str, float] = {}
        for layer_offset, layer_index in enumerate(ordered_layers):
            layer_tensor = stacked_cpu[:, layer_offset, :, :]
            layer_summary = _slot_norm_summary(layer_tensor)
            layer_hidden_l2_by_layer[str(layer_index)] = float(layer_summary["l2"])
            layer_slot_norm_mean_by_layer[str(layer_index)] = float(layer_summary["slot_norm_mean"])
            layer_slot_norm_std_by_layer[str(layer_index)] = float(layer_summary["slot_norm_std"])
            layer_slot_norm_max_by_layer[str(layer_index)] = float(layer_summary["slot_norm_max"])
        return {
            "pilot_memory_path_variant": memory_path_variant,
            "pilot_projector_token_source": projector_token_source,
            "pilot_injection_mode": "sparse_deep_prefix",
            "pilot_prefix_source_mode": prefix_source_mode,
            "pilot_deep_prefix_init_mode": deep_prefix_init_mode,
            "pilot_support_encoder_mode": (
                "structured_support_set" if support_item_states is not None and support_item_states.numel() > 0 else "pooled_block"
            ),
            "active_prefix_layers": ordered_layers,
            "layer_hidden_l2_by_layer": layer_hidden_l2_by_layer,
            "layer_slot_norm_mean_by_layer": layer_slot_norm_mean_by_layer,
            "layer_slot_norm_std_by_layer": layer_slot_norm_std_by_layer,
            "layer_slot_norm_max_by_layer": layer_slot_norm_max_by_layer,
            "layer_key_l2_by_layer": {},
            "layer_value_l2_by_layer": {},
            "prefix_tokens": float(stacked_cpu.shape[2]),
            "prefix_l2": float(stacked_cpu.norm().item()),
            "prefix_slot_norm_mean": float(slot_norms.mean().item()),
            "prefix_slot_norm_std": float(slot_norms.std(unbiased=False).item()),
            "prefix_slot_norm_max": float(slot_norms.max().item()),
            **writer_stats,
            **memory_long_stats,
            **memory_long_geometry_stats,
            **memory_short_stats,
            **support_stats,
            **writer_support_stats,
            **writer_context_stats,
            **reader_pre_stats,
            **reader_stats,
            **fuser_geometry_stats,
            "projected_memory_effective_rank": float(projected_memory_effective_rank),
        }
    if prefix_embeddings is None or prefix_embeddings.numel() == 0:
        return {
            "pilot_memory_path_variant": memory_path_variant,
            "pilot_projector_token_source": projector_token_source,
            "pilot_injection_mode": "shallow_prefix",
            "pilot_prefix_source_mode": prefix_source_mode,
            "pilot_deep_prefix_init_mode": deep_prefix_init_mode,
            "pilot_support_encoder_mode": (
                "structured_support_set" if support_item_states is not None and support_item_states.numel() > 0 else "pooled_block"
            ),
            "active_prefix_layers": [],
            "layer_hidden_l2_by_layer": {},
            "layer_slot_norm_mean_by_layer": {},
            "layer_slot_norm_std_by_layer": {},
            "layer_slot_norm_max_by_layer": {},
            "layer_key_l2_by_layer": {},
            "layer_value_l2_by_layer": {},
            "prefix_tokens": 0.0,
            "prefix_l2": 0.0,
            "prefix_slot_norm_mean": 0.0,
            "prefix_slot_norm_std": 0.0,
            "prefix_slot_norm_max": 0.0,
            **writer_stats,
            **memory_long_stats,
            **memory_long_geometry_stats,
            **memory_short_stats,
            **support_stats,
            **writer_support_stats,
            **writer_context_stats,
            **reader_pre_stats,
            **reader_stats,
            **fuser_geometry_stats,
            "projected_memory_effective_rank": float(projected_memory_effective_rank),
        }
    prefix_summary = _slot_norm_summary(prefix_embeddings)
    return {
        "pilot_memory_path_variant": memory_path_variant,
        "pilot_projector_token_source": projector_token_source,
        "pilot_injection_mode": "shallow_prefix",
        "pilot_prefix_source_mode": prefix_source_mode,
        "pilot_deep_prefix_init_mode": deep_prefix_init_mode,
        "pilot_support_encoder_mode": (
            "structured_support_set" if support_item_states is not None and support_item_states.numel() > 0 else "pooled_block"
        ),
        "active_prefix_layers": [],
        "layer_hidden_l2_by_layer": {},
        "layer_slot_norm_mean_by_layer": {},
        "layer_slot_norm_std_by_layer": {},
        "layer_slot_norm_max_by_layer": {},
        "layer_key_l2_by_layer": {},
        "layer_value_l2_by_layer": {},
        "prefix_tokens": float(prefix_embeddings.shape[1]),
        "prefix_l2": float(prefix_summary["l2"]),
        "prefix_slot_norm_mean": float(prefix_summary["slot_norm_mean"]),
        "prefix_slot_norm_std": float(prefix_summary["slot_norm_std"]),
        "prefix_slot_norm_max": float(prefix_summary["slot_norm_max"]),
        **writer_stats,
        **memory_long_stats,
        **memory_long_geometry_stats,
        **memory_short_stats,
        **support_stats,
        **writer_support_stats,
        **writer_context_stats,
        **reader_pre_stats,
        **reader_stats,
        **fuser_geometry_stats,
        "projected_memory_effective_rank": float(projected_memory_effective_rank),
    }


def _aggregate_prefix_stats(prefix_stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not prefix_stats_list:
        return _prefix_stats()
    aggregated = copy.deepcopy(prefix_stats_list[0])
    numeric_keys = [
        key
        for key, value in aggregated.items()
        if isinstance(value, (int, float))
    ]
    for key in numeric_keys:
        aggregated[key] = float(
            sum(float(stats.get(key, 0.0)) for stats in prefix_stats_list) / len(prefix_stats_list)
        )
    for key, value in list(aggregated.items()):
        if isinstance(value, dict):
            all_nested_keys = sorted({nested for stats in prefix_stats_list for nested in stats.get(key, {})})
            aggregated[key] = {
                nested: float(
                    sum(float(stats.get(key, {}).get(nested, 0.0)) for stats in prefix_stats_list)
                    / len(prefix_stats_list)
                )
                for nested in all_nested_keys
            }
        elif isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
            width = max(len(stats.get(key, [])) for stats in prefix_stats_list)
            aggregated[key] = [
                float(
                    sum(
                        float(stats.get(key, [0.0] * width)[index])
                        if index < len(stats.get(key, []))
                        else 0.0
                        for stats in prefix_stats_list
                    )
                    / len(prefix_stats_list)
                )
                for index in range(width)
            ]
    return aggregated


def _save_shared_injection_checkpoint(
    *,
    runtime: "SharedInjectionPilotRuntime",
    output_path: Path,
    seed: int,
    arm_alias: str,
    arm: str,
    writer_memory_control: str,
    support_dataset_path: str,
    support_text_block: str,
    prompt_variant: str,
    support_serialization_variant: str,
    step: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "writer_state": runtime.writer.state_dict(),
            "source_stub_state": (
                None if runtime.source_stub is None else runtime.source_stub.state_dict()
            ),
            "support_encoder_state": (
                None if runtime.support_encoder is None else runtime.support_encoder.state_dict()
            ),
            "reader_state": None if runtime.reader is None else runtime.reader.state_dict(),
            "fuser_state": None if runtime.fuser is None else runtime.fuser.state_dict(),
            "prefix_projector_state": runtime.prefix_projector.state_dict(),
            "seed": seed,
            "arm_alias": arm_alias,
            "shared_injection_arm": arm,
            "writer_memory_control": writer_memory_control,
            "support_dataset_path": str(Path(support_dataset_path).resolve()),
            "support_text_block": support_text_block,
            "prompt_variant": prompt_variant,
            "support_serialization_variant": support_serialization_variant,
            "pilot_support_encoder_mode": str(runtime.support_encoder_mode),
            "backbone_hidden_size": int(runtime.backbone.hidden_size),
            "writer_memory_slots": int(runtime.writer.memory_slots),
            "pilot_memory_path_variant": str(runtime.memory_path_variant),
            "pilot_bridge_mode": str(runtime.bridge_mode),
            "pilot_reader_context_mode": str(runtime.reader_context_mode),
            "pilot_reader_conditioning_mode": (
                None if runtime.reader is None else str(runtime.reader.conditioning_mode)
            ),
            "pilot_reader_gated_add_scale": (
                None if runtime.reader is None else float(runtime.reader.gated_add_scale)
            ),
            "pilot_reader_attention_mode": (
                None if runtime.reader is None else str(runtime.reader.attention_mode)
            ),
            "pilot_reader_masked_partition": (
                []
                if runtime.reader is None or runtime.reader.masked_partition is None
                else [list(group) for group in runtime.reader.masked_partition]
            ),
            "pilot_projector_token_source": str(runtime.projector_token_source),
            "pilot_prefix_source_mode": str(runtime.prefix_source_mode),
            "pilot_reader_num_queries": int(runtime.reader_num_queries),
            "pilot_fuser_short_slots": int(runtime.fuser_short_slots),
            "pilot_projector_prefix_tokens": int(runtime.prefix_projector.prefix_tokens),
            "pilot_deep_prefix_rank": int(getattr(runtime.prefix_projector, "bottleneck_rank", runtime.backbone.hidden_size)),
            "pilot_deep_prefix_init_mode": str(runtime.deep_prefix_init_mode),
            "pilot_writer_output_slot_basis_scale": float(
                getattr(runtime.writer, "output_slot_basis_scale", 0.0)
            ),
            "pilot_writer_support_query_residual_scale": float(
                getattr(runtime.writer, "support_query_residual_scale", 0.0)
            ),
            "pilot_writer_context_query_residual_scale": float(
                getattr(runtime.writer, "context_query_residual_scale", 0.0)
            ),
            "pilot_writer_conditioning_layers": int(
                getattr(runtime.writer, "conditioning_layers", 1)
            ),
            "pilot_writer_stimulus_mode": str(runtime.writer_stimulus_mode),
            "pilot_writer_context_tokens": int(runtime.writer_context_tokens),
            "pilot_writer_adapter_enabled": bool(runtime.writer_adapter_enabled),
            "pilot_writer_adapter_target_modules": list(runtime.writer_adapter_target_modules),
            "pilot_writer_adapter_rank": int(runtime.writer_adapter_rank),
            "pilot_writer_adapter_alpha": float(runtime.writer_adapter_alpha),
            "pilot_writer_adapter_dropout": float(runtime.writer_adapter_dropout),
            "pilot_writer_adapter_trainable_params": int(runtime.writer_adapter_trainable_params),
            "pilot_source_stub_trainable_params": int(runtime.source_stub_trainable_params),
            "receiver_lora_state": (
                None
                if not hasattr(runtime.backbone, "receiver_lora_state_dict")
                else runtime.backbone.receiver_lora_state_dict()
            ),
            "pilot_receiver_lora_enabled": bool(runtime.receiver_lora_enabled),
            "pilot_receiver_lora_target_layers": list(runtime.receiver_lora_target_layers),
            "pilot_receiver_lora_target_modules": list(runtime.receiver_lora_target_modules),
            "pilot_receiver_lora_rank": int(runtime.receiver_lora_rank),
            "pilot_receiver_lora_alpha": float(runtime.receiver_lora_alpha),
            "pilot_receiver_lora_dropout": float(runtime.receiver_lora_dropout),
            "pilot_receiver_lora_trainable_params": int(runtime.receiver_lora_trainable_params),
            "pilot_support_encoder_max_items": (
                None if runtime.support_encoder is None else int(runtime.support_encoder.max_items)
            ),
            "step": int(step),
            "pilot_injection_mode": str(runtime.injection_mode),
            "pilot_deep_prefix_layers": list(runtime.deep_prefix_layers),
        },
        output_path,
    )


def _validate_module_state_shapes(
    *,
    module_name: str,
    expected_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
    checkpoint_path: Path,
) -> None:
    for key, expected_tensor in expected_state.items():
        if key not in checkpoint_state:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} is missing {module_name} parameter '{key}'."
            )
        actual_tensor = checkpoint_state[key]
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} has incompatible {module_name} parameter "
                f"'{key}': expected shape {tuple(expected_tensor.shape)}, got {tuple(actual_tensor.shape)}."
            )


def _active_competitor_hinge_weight(
    *,
    current_step: int,
    max_weight: float,
    start_step: int,
    ramp_steps: int,
) -> float:
    capped_weight = max(0.0, float(max_weight))
    if capped_weight == 0.0:
        return 0.0
    if current_step <= int(start_step):
        return 0.0
    if ramp_steps <= 0:
        return capped_weight
    progress = min(1.0, float(current_step - int(start_step)) / float(ramp_steps))
    return float(capped_weight * progress)


def _scheduled_linear_decay_weight(
    *,
    current_step: int,
    start_weight: float,
    end_weight: float,
    decay_steps: int,
) -> float:
    start = float(start_weight)
    end = float(end_weight)
    if decay_steps <= 1:
        return end
    progress = min(1.0, max(0.0, float(current_step - 1)) / float(decay_steps - 1))
    return float(start + ((end - start) * progress))


def _fever_candidate_texts(prompt_variant: str) -> tuple[list[str], list[str]]:
    labels = list(FEVER_LABEL_ORDER)
    if prompt_variant == "inline_short_labels":
        return labels, ["Supports", "Refutes", "Not enough info"]
    if prompt_variant == "answer_slot_labels":
        return labels, [" SUPPORTS", " REFUTES", " NOT_ENOUGH_INFO"]
    if prompt_variant == "verbalized_decisions":
        return labels, [
            " The evidence supports the claim.",
            " The evidence refutes the claim.",
            " There is not enough information in the evidence to verify the claim.",
        ]
    raise ValueError(f"Unsupported prompt_variant={prompt_variant}.")


def _resolve_prompt_text(example: dict[str, Any], *, prompt_variant: str) -> str:
    if prompt_variant == TASK_NATIVE_PROMPT_VARIANT:
        return str(example.get("segment", "")).strip()
    claim = str(example.get("claim", "")).strip()
    evidence = str(example.get("evidence", "")).strip()
    if prompt_variant == "inline_short_labels":
        return (
            f"Claim: {claim} || Evidence: {evidence} || "
            "Labels: SUPPORTS: Supports | REFUTES: Refutes | NOT_ENOUGH_INFO: Not enough info || "
            "Decide the correct label."
        )
    if prompt_variant == "answer_slot_labels":
        return (
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n"
            "Question: Does the evidence support, refute, or not give enough info for the claim?\n"
            "Answer:"
        )
    if prompt_variant == "verbalized_decisions":
        return f"Claim: {claim}\nEvidence: {evidence}\nDecision:"
    raise ValueError(f"Unsupported prompt_variant={prompt_variant}.")


def _task_mode_for_evaluator_type(evaluator_type: str) -> str:
    if evaluator_type in TASK_CANDIDATE_SELECTION_EVALUATOR_TYPES:
        return "candidate_selection"
    if evaluator_type in TASK_GENERATION_EVALUATOR_TYPES:
        return "generation"
    raise ValueError(f"Unsupported evaluator_type={evaluator_type!r} for shared injection pilot.")


def _candidate_payload(example: dict[str, Any], *, prompt_variant: str) -> tuple[list[str], list[str], int, str]:
    evaluator_type = str(example.get("evaluator_type", "dataset_label_classification"))
    task_mode = _task_mode_for_evaluator_type(evaluator_type)
    if prompt_variant != TASK_NATIVE_PROMPT_VARIANT:
        gold_label = str(example["label"])
        candidate_labels, candidate_texts = _fever_candidate_texts(prompt_variant)
        try:
            gold_index = candidate_labels.index(gold_label)
        except ValueError as exc:
            raise ValueError(f"Gold label {gold_label!r} missing from FEVER labels.") from exc
        return candidate_labels, candidate_texts, gold_index, task_mode
    if task_mode == "candidate_selection":
        choices = list(example.get("choices", []))
        if not choices:
            raise ValueError("Candidate-selection shared-injection examples require non-empty choices.")
        candidate_labels = [str(choice["label"]) for choice in choices]
        candidate_texts = [str(choice["text"]) for choice in choices]
        gold_label = str(example["label"])
        try:
            gold_index = candidate_labels.index(gold_label)
        except ValueError as exc:
            raise ValueError(f"Gold label {gold_label!r} missing from benchmark choices.") from exc
        return candidate_labels, candidate_texts, gold_index, task_mode
    gold_text = str(example.get("continuation", example.get("gold_answer", ""))).strip()
    if not gold_text:
        raise ValueError("Generative shared-injection examples require a non-empty continuation/gold_answer.")
    return [str(example.get("label", gold_text))], [gold_text], 0, task_mode


def _build_example_caches(
    examples: list[dict[str, Any]],
    *,
    prompt_variant: str,
) -> list[SharedInjectionExampleCache]:
    caches: list[SharedInjectionExampleCache] = []
    for example in examples:
        candidate_labels, candidate_texts, gold_index, task_mode = _candidate_payload(
            example,
            prompt_variant=prompt_variant,
        )
        caches.append(
            SharedInjectionExampleCache(
                example=example,
                candidate_labels=candidate_labels,
                candidate_texts=candidate_texts,
                gold_index=gold_index,
                prompt_text=_resolve_prompt_text(example, prompt_variant=prompt_variant),
                prompt_variant=prompt_variant,
                evaluator_type=str(example.get("evaluator_type", "dataset_label_classification")),
                task_mode=task_mode,
            )
        )
    return caches


def _serialize_support_row(
    row: dict[str, Any],
    *,
    support_index: int,
    support_serialization_variant: str,
) -> str:
    benchmark_id = str(row.get("benchmark_id", "")).strip().lower()
    if benchmark_id and benchmark_id != "fever":
        prompt = str(row.get("segment", "")).strip()
        answer = str(row.get("gold_answer", row.get("continuation", ""))).strip()
        if support_serialization_variant == "flat_raw8":
            return f"Support {support_index}: Prompt: {prompt} || Answer: {answer}"
        if support_serialization_variant == "example_blocks_raw8":
            return (
                f"Example {support_index}\n"
                f"Prompt: {prompt}\n"
                f"Answer: {answer}"
            )
        raise ValueError(
            f"Unsupported support_serialization_variant={support_serialization_variant}."
        )
    claim = str(row.get("claim", "")).strip()
    evidence = str(row.get("evidence", "")).strip()
    label = str(row.get("label", "")).strip()
    if support_serialization_variant == "flat_raw8":
        return f"Support {support_index}: Claim: {claim} || Evidence: {evidence} || Label: {label}"
    if support_serialization_variant in {"example_blocks_raw8", "triad_curated6"}:
        return (
            f"Example {support_index}\n"
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n"
            f"Answer: {label}"
        )
    raise ValueError(
        f"Unsupported support_serialization_variant={support_serialization_variant}."
    )


def _serialize_support_rows(
    support_rows: list[dict[str, Any]],
    *,
    support_serialization_variant: str,
) -> list[str]:
    return [
        _serialize_support_row(
            row,
            support_index=index + 1,
            support_serialization_variant=support_serialization_variant,
        )
        for index, row in enumerate(support_rows)
    ]


def _select_support_rows_for_variant(
    support_examples: list[dict[str, Any]],
    *,
    support_serialization_variant: str,
) -> list[dict[str, Any]]:
    if support_serialization_variant in {"flat_raw8", "example_blocks_raw8"}:
        return [dict(example) for example in support_examples]
    if support_serialization_variant != "triad_curated6":
        raise ValueError(
            f"Unsupported support_serialization_variant={support_serialization_variant}."
        )
    benchmark_ids = {
        str(example.get("benchmark_id", "")).strip().lower()
        for example in support_examples
        if str(example.get("benchmark_id", "")).strip()
    }
    if benchmark_ids and benchmark_ids != {"fever"}:
        raise ValueError("triad_curated6 is only supported for FEVER support banks.")
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in FEVER_LABEL_ORDER}
    for example in support_examples:
        label = str(example.get("label", "")).strip()
        if label in by_label:
            by_label[label].append(dict(example))
    selected: list[dict[str, Any]] = []
    for label in FEVER_LABEL_ORDER:
        rows = by_label[label][:2]
        if len(rows) < 2:
            raise ValueError(
                f"triad_curated6 requires at least 2 rows for label {label}, got {len(rows)}."
            )
        for row in rows:
            if label == "NOT_ENOUGH_INFO":
                row["evidence"] = FEVER_TRIAD_NEI_EVIDENCE
            selected.append(row)
    return selected


def _resolve_support_rows_for_memory_control(
    support_examples: list[dict[str, Any]],
    *,
    memory_control: str,
    example_lookup: dict[str, dict[str, Any]],
    support_serialization_variant: str,
    preselected_rows: bool = False,
) -> list[dict[str, Any]]:
    if memory_control == "zero":
        return []
    if preselected_rows:
        selected_rows = [dict(example) for example in support_examples]
    else:
        selected_rows = _select_support_rows_for_variant(
            support_examples,
            support_serialization_variant=support_serialization_variant,
        )
    if memory_control == "real":
        return [dict(example) for example in selected_rows]
    rows: list[dict[str, Any]] = []
    for example in selected_rows:
        shuffled_id = str(example.get("shuffled_memory_example_id", "")).strip()
        if not shuffled_id:
            raise ValueError(
                f"Support example {example['id']} is missing shuffled_memory_example_id for shuffled control."
            )
        shuffled_example = dict(example_lookup[shuffled_id])
        if (
            support_serialization_variant == "triad_curated6"
            and str(shuffled_example.get("label", "")) == "NOT_ENOUGH_INFO"
        ):
            shuffled_example["evidence"] = FEVER_TRIAD_NEI_EVIDENCE
        rows.append(shuffled_example)
    return rows


def _build_support_text_block(
    support_examples: list[dict[str, Any]],
    *,
    memory_control: str,
    example_lookup: dict[str, dict[str, Any]],
    support_serialization_variant: str,
    preselected_rows: bool = False,
) -> str:
    rows = _resolve_support_rows_for_memory_control(
        support_examples,
        memory_control=memory_control,
        example_lookup=example_lookup,
        support_serialization_variant=support_serialization_variant,
        preselected_rows=preselected_rows,
    )
    return "\n\n".join(
        _serialize_support_rows(
            rows,
            support_serialization_variant=support_serialization_variant,
        )
    )


def _support_label_ids(
    support_rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    label_ids = []
    for row in support_rows:
        label = str(row.get("label", "")).strip()
        if label in FEVER_LABEL_ORDER:
            label_ids.append(FEVER_LABEL_ORDER.index(label))
        else:
            # Non-FEVER support banks do not expose a small fixed label space; keep them in a shared bucket
            # so the warm-started FEVER support encoder can still be reused.
            label_ids.append(0)
    return torch.tensor([label_ids], dtype=torch.long, device=device)


def _resolve_train_support_mask_count(
    config: dict[str, Any],
    *,
    support_serialization_variant: str,
) -> int | None:
    explicit = config["runtime"].get("pilot_train_support_mask_count")
    if explicit is not None:
        return int(explicit)
    if support_serialization_variant == "triad_curated6":
        return 4
    if support_serialization_variant in {"flat_raw8", "example_blocks_raw8"}:
        return 5
    return None


def _sample_support_examples_for_training(
    *,
    support_examples: list[dict[str, Any]],
    support_serialization_variant: str,
    generator: torch.Generator,
    target_count: int | None = None,
) -> list[dict[str, Any]]:
    selected_rows = _select_support_rows_for_variant(
        support_examples,
        support_serialization_variant=support_serialization_variant,
    )
    if target_count is None:
        if support_serialization_variant == "triad_curated6":
            target_count = 4
        elif support_serialization_variant in {"flat_raw8", "example_blocks_raw8"}:
            target_count = 5
    if target_count is None or target_count >= len(selected_rows):
        return [dict(row) for row in selected_rows]
    permutation = torch.randperm(len(selected_rows), generator=generator).tolist()
    return [dict(selected_rows[index]) for index in permutation[:target_count]]


def _serialize_teacher_prompt(prompt_text: str, support_text_block: str) -> str:
    if not support_text_block:
        return prompt_text
    return f"Support bank:\n{support_text_block}\n\n{prompt_text}"


class SharedInjectionPilotRuntime(nn.Module):
    def __init__(self, config: dict[str, Any], seed: int, *, arm: str, writer_memory_control: str) -> None:
        super().__init__()
        backbone_cfg = config["backbone"]
        runtime_device = str(config["runtime"].get("device", "cpu"))
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
            attn_implementation=backbone_cfg.get("attn_implementation"),
            max_new_tokens=int(backbone_cfg.get("max_new_tokens", 32)),
        )
        writer_cfg = config["method"]["writer"]
        self.bridge_mode = _resolve_bridge_mode(config)
        self.writer_stimulus_mode = _resolve_writer_stimulus_mode(config)
        self.writer_context_tokens = _resolve_writer_context_tokens(config)
        if self.bridge_mode == "writer_direct":
            self.writer = WriterWeaverHead(
                embed_dim=self.backbone.hidden_size,
                memory_slots=int(writer_cfg["memory_slots"]),
                hidden_dim=writer_cfg.get("hidden_dim"),
                num_heads=int(writer_cfg.get("num_heads", 4)),
                transformer_layers=int(writer_cfg.get("transformer_layers", 1)),
                conditioning_layers=int(writer_cfg.get("conditioning_layers", 1)),
                dropout=float(writer_cfg.get("dropout", 0.0)),
                context_query_residual_scale=float(
                    config["runtime"].get("pilot_writer_context_query_residual_scale", 1.0)
                ),
                support_query_residual_scale=float(writer_cfg.get("support_query_residual_scale", 1.0)),
                output_slot_basis_scale=float(writer_cfg.get("output_slot_basis_scale", 0.0)),
            )
        else:
            self.writer = MemoryWriter(
                embed_dim=self.backbone.hidden_size,
                memory_slots=int(writer_cfg["memory_slots"]),
                arch=str(writer_cfg.get("arch", "mlp")),
                hidden_dim=writer_cfg.get("hidden_dim"),
                num_heads=int(writer_cfg.get("num_heads", 4)),
                transformer_layers=int(writer_cfg.get("transformer_layers", 1)),
                dropout=float(writer_cfg.get("dropout", 0.0)),
                support_query_residual_scale=float(writer_cfg.get("support_query_residual_scale", 0.0)),
                output_slot_basis_scale=float(writer_cfg.get("output_slot_basis_scale", 0.0)),
                slot_conditioning_mode=str(writer_cfg.get("slot_conditioning_mode", "shared_add")),
                shared_state_scale=float(writer_cfg.get("shared_state_scale", 1.0)),
            )
        self.memory_path_variant = _resolve_memory_path_variant(config)
        self.injection_mode = _resolve_injection_mode(config)
        self.reader_context_mode = _resolve_reader_context_mode(config)
        self.projector_token_source = _resolve_projector_token_source(config)
        self.support_encoder_mode = _resolve_support_encoder_mode(config)
        self.trainable_variant = _resolve_trainable_variant(config)
        self.alignment_aux_mode = _resolve_alignment_aux_mode(config)
        self.alignment_aux_weight_max = _resolve_alignment_aux_weight_max(config)
        self.alignment_aux_temperature = _resolve_alignment_aux_temperature(config)
        self.alignment_aux_advantage_center = _resolve_alignment_aux_advantage_center(config)
        self.alignment_aux_advantage_scale = _resolve_alignment_aux_advantage_scale(config)
        self.deep_prefix_layers = tuple(_resolve_deep_prefix_layers(config))
        self.prefix_source_mode = _resolve_prefix_source_mode(config)
        self.deep_prefix_init_mode = _resolve_deep_prefix_init_mode(config)
        self.support_serialization_variant = _resolve_support_serialization_variant(config)
        self.writer_adapter_enabled = _resolve_writer_adapter_enabled(config)
        self.writer_adapter_target_modules = _resolve_writer_adapter_target_modules(config)
        self.writer_adapter_rank = _resolve_writer_adapter_rank(config)
        self.writer_adapter_alpha = _resolve_writer_adapter_alpha(config)
        self.writer_adapter_dropout = _resolve_writer_adapter_dropout(config)
        self.receiver_lora_enabled = _resolve_receiver_lora_enabled(config)
        self.receiver_lora_target_layers = _resolve_receiver_lora_target_layers(config)
        self.receiver_lora_target_modules = _resolve_receiver_lora_target_modules(config)
        self.receiver_lora_rank = _resolve_receiver_lora_rank(config)
        self.receiver_lora_alpha = _resolve_receiver_lora_alpha(config)
        self.receiver_lora_dropout = _resolve_receiver_lora_dropout(config)
        self._validate_planv5_runtime_contract()
        if self.writer_adapter_enabled:
            if self.bridge_mode != "writer_direct":
                raise ValueError(
                    "method.writer_adapter.enabled=true currently requires runtime.pilot_bridge_mode=writer_direct."
                )
            if self.prefix_source_mode != "writer":
                raise ValueError(
                    "method.writer_adapter.enabled=true requires runtime.pilot_prefix_source_mode=writer."
                )
            if not hasattr(self.writer, "enable_writer_micro_lora"):
                raise RuntimeError("Configured writer micro-LoRA, but the active writer lacks support.")
            self.writer.enable_writer_micro_lora(
                target_modules=self.writer_adapter_target_modules,
                rank=self.writer_adapter_rank,
                alpha=self.writer_adapter_alpha,
                dropout=self.writer_adapter_dropout,
            )
        self.writer_adapter_trainable_params = int(
            getattr(self.writer, "writer_lora_parameter_count", lambda: 0)()
        )
        self.source_stub: SourceStubMemory | None = None
        if self.prefix_source_mode == "source_stub":
            self.source_stub = SourceStubMemory(
                embed_dim=self.backbone.hidden_size,
                memory_slots=int(self.writer.memory_slots),
                slot_max_norm=config["runtime"].get("pilot_prefix_slot_max_norm"),
                total_max_norm=config["runtime"].get("pilot_prefix_total_max_norm"),
            )
        if self.receiver_lora_enabled:
            if not hasattr(self.backbone, "enable_receiver_micro_lora"):
                raise RuntimeError("Configured receiver micro-LoRA, but BackboneWrapper lacks support.")
            self.backbone.enable_receiver_micro_lora(
                layer_indices=self.receiver_lora_target_layers,
                target_modules=self.receiver_lora_target_modules,
                rank=self.receiver_lora_rank,
                alpha=self.receiver_lora_alpha,
                dropout=self.receiver_lora_dropout,
            )
        self.receiver_lora_trainable_params = int(
            getattr(self.backbone, "receiver_lora_parameter_count", lambda: 0)()
        )
        self.source_stub_trainable_params = int(
            0 if self.source_stub is None else sum(parameter.numel() for parameter in self.source_stub.parameters())
        )
        self.reader: MemoryReader | None = None
        self.fuser: MemoryFuser | None = None
        self.support_encoder: StructuredSupportSetEncoder | None = None
        if self.support_encoder_mode == "structured_support_set":
            self.support_encoder = StructuredSupportSetEncoder(
                hidden_size=self.backbone.hidden_size,
                label_count=len(FEVER_LABEL_ORDER),
                max_items=int(config["runtime"].get("pilot_support_encoder_max_items", 8)),
                num_heads=int(config["runtime"].get("pilot_support_encoder_num_heads", 4)),
                transformer_layers=int(config["runtime"].get("pilot_support_encoder_layers", 1)),
                dropout=float(config["runtime"].get("pilot_support_encoder_dropout", 0.0)),
            )
        self.reader_num_queries = 0
        self.fuser_short_slots = int(self.writer.memory_slots)
        if self.memory_path_variant == "two_level":
            reader_cfg = config["method"].get("reader", {})
            fuser_cfg = config["method"].get("fuser", {})
            self.reader_num_queries = _resolve_reader_num_queries(config)
            self.fuser_short_slots = _resolve_fuser_short_slots(config)
            self.reader = MemoryReader(
                embed_dim=self.backbone.hidden_size,
                num_queries=self.reader_num_queries,
                use_query_gating=bool(reader_cfg.get("use_query_gating", False)),
                gating_mode=reader_cfg.get("gating_mode", "off"),
                num_heads=int(reader_cfg.get("num_heads", 4)),
                condition_on_context=bool(reader_cfg.get("condition_on_context", True)),
                conditioning_mode=reader_cfg.get("conditioning_mode"),
                gated_add_scale=float(reader_cfg.get("gated_add_scale", 0.1)),
                attention_mode=str(reader_cfg.get("attention_mode", "standard")),
                masked_partition=reader_cfg.get("masked_partition"),
                dropout=float(reader_cfg.get("dropout", 0.0)),
                query_residual_scale=float(reader_cfg.get("query_residual_scale", 0.0)),
            )
            self.fuser = MemoryFuser(
                embed_dim=self.backbone.hidden_size,
                num_queries=self.reader_num_queries,
                short_slots=self.fuser_short_slots,
                arch=str(fuser_cfg.get("arch", "resampler")),
                hidden_dim=fuser_cfg.get("hidden_dim"),
                num_heads=int(fuser_cfg.get("num_heads", 4)),
                dropout=float(fuser_cfg.get("dropout", 0.0)),
            )
        projector_prefix_tokens = (
            int(self.writer.memory_slots)
            if self.projector_token_source == "writer_slots"
            else int(self.fuser_short_slots)
        )
        if self.injection_mode == "shallow_prefix":
            self.prefix_projector = LatentPrefixProjector(
                hidden_size=self.backbone.hidden_size,
                prefix_tokens=projector_prefix_tokens,
                slot_max_norm=config["runtime"].get("pilot_prefix_slot_max_norm"),
                total_max_norm=config["runtime"].get("pilot_prefix_total_max_norm"),
            )
        else:
            self.prefix_projector = SharedLowRankDeepPrefixProjector(
                hidden_size=self.backbone.hidden_size,
                prefix_tokens=projector_prefix_tokens,
                layer_indices=list(self.deep_prefix_layers),
                bottleneck_rank=_resolve_deep_prefix_rank(config),
                slot_max_norm=config["runtime"].get("pilot_prefix_slot_max_norm"),
                total_max_norm=config["runtime"].get("pilot_prefix_total_max_norm"),
            )
        self.arm = arm
        self.writer_memory_control = writer_memory_control
        self._prompt_summary_cache: dict[str, torch.Tensor] = {}
        self._prompt_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._deep_prefix_projector_initialized = bool(
            self.injection_mode == "shallow_prefix" or self.deep_prefix_init_mode == "random"
        )
        self._source_stub_initialized = bool(
            self.source_stub is None or self.deep_prefix_init_mode == "random"
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        set_receiver_lora_trainable = getattr(self.backbone, "set_receiver_lora_trainable", None)
        if callable(set_receiver_lora_trainable):
            set_receiver_lora_trainable(self.receiver_lora_enabled)
        self.to(self.backbone.device)

    def _validate_planv5_runtime_contract(self) -> None:
        if self.bridge_mode == "writer_direct" and self.memory_path_variant != "single_level":
            raise ValueError("runtime.pilot_bridge_mode=writer_direct requires pilot_memory_path_variant=single_level.")
        if self.prefix_source_mode == "source_stub":
            if self.bridge_mode != "writer_direct":
                raise ValueError(
                    "runtime.pilot_prefix_source_mode=source_stub currently requires runtime.pilot_bridge_mode=writer_direct."
                )
            if self.injection_mode != "sparse_deep_prefix":
                raise ValueError(
                    "runtime.pilot_prefix_source_mode=source_stub requires runtime.pilot_injection_mode=sparse_deep_prefix."
                )
        if self.bridge_mode != "writer_direct" and self.receiver_lora_enabled:
            return
        if self.bridge_mode == "writer_direct" and self.receiver_lora_enabled:
            if self.injection_mode != "sparse_deep_prefix":
                raise ValueError(
                    "runtime.pilot_bridge_mode=writer_direct allows receiver micro-LoRA only when "
                    "runtime.pilot_injection_mode=sparse_deep_prefix."
                )
            if not self.receiver_lora_target_layers:
                raise ValueError(
                    "runtime.pilot_bridge_mode=writer_direct with receiver micro-LoRA requires at least one target layer."
                )
            if len(self.receiver_lora_target_layers) > 5:
                raise ValueError(
                    "runtime.pilot_bridge_mode=writer_direct supports only tiny receiver micro-LoRA sets "
                    "(at most 5 target layers)."
                )
            if self.receiver_lora_rank <= 0 or self.receiver_lora_rank > 4:
                raise ValueError(
                    "runtime.pilot_bridge_mode=writer_direct supports receiver micro-LoRA rank in [1, 4]."
                )
            if self.receiver_lora_alpha <= 0.0 or self.receiver_lora_alpha > 8.0:
                raise ValueError(
                    "runtime.pilot_bridge_mode=writer_direct supports receiver micro-LoRA alpha in (0, 8]."
                )

    def orthogonalize_writer_slot_basis(self) -> None:
        self.writer.orthogonalize_slot_embeddings_()

    def source_stub_parameters(self) -> list[nn.Parameter]:
        if self.source_stub is None:
            return []
        return list(self.source_stub.parameters())

    def set_source_stub_trainable(self, enabled: bool) -> None:
        if self.source_stub is None:
            return
        for parameter in self.source_stub.parameters():
            parameter.requires_grad_(enabled)

    def load_writer(self, resume: str | None) -> None:
        writer_path = _resolve_artifact_path(resume, "writer.ckpt")
        self.writer.load_from(writer_path, map_location="cpu", strict=False)

    def _validate_injection_checkpoint_schema(
        self,
        *,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
        allow_single_level_init_for_two_level: bool = False,
        allow_missing_receiver_lora_state: bool = False,
    ) -> None:
        checkpoint_bridge_mode = str(checkpoint.get("pilot_bridge_mode", "legacy"))
        if checkpoint_bridge_mode != self.bridge_mode:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} uses pilot_bridge_mode={checkpoint_bridge_mode}, "
                f"expected {self.bridge_mode}."
            )
        checkpoint_memory_path_variant = str(
            checkpoint.get(
                "pilot_memory_path_variant",
                "two_level"
                if checkpoint.get("reader_state") is not None or checkpoint.get("fuser_state") is not None
                else "single_level",
            )
        )
        if checkpoint_memory_path_variant != self.memory_path_variant:
            if not (
                allow_single_level_init_for_two_level
                and self.memory_path_variant == "two_level"
                and checkpoint_memory_path_variant == "single_level"
            ):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_memory_path_variant="
                    f"{checkpoint_memory_path_variant}, expected {self.memory_path_variant}."
                )
        checkpoint_support_encoder_mode = str(
            checkpoint.get(
                "pilot_support_encoder_mode",
                "structured_support_set" if checkpoint.get("support_encoder_state") is not None else "pooled_block",
            )
        )
        if checkpoint_support_encoder_mode != self.support_encoder_mode:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} uses pilot_support_encoder_mode="
                f"{checkpoint_support_encoder_mode}, expected {self.support_encoder_mode}."
            )
        checkpoint_injection_mode = str(
            checkpoint.get(
                "pilot_injection_mode",
                "sparse_deep_prefix" if "down_proj.weight" in checkpoint.get("prefix_projector_state", {}) else "shallow_prefix",
            )
        )
        if checkpoint_injection_mode != self.injection_mode:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} uses pilot_injection_mode="
                f"{checkpoint_injection_mode}, expected {self.injection_mode}."
            )
        checkpoint_layers = tuple(int(layer) for layer in checkpoint.get("pilot_deep_prefix_layers", []))
        if self.injection_mode == "sparse_deep_prefix" and checkpoint_layers != self.deep_prefix_layers:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} uses pilot_deep_prefix_layers="
                f"{list(checkpoint_layers)}, expected {list(self.deep_prefix_layers)}."
            )
        checkpoint_hidden_size = int(checkpoint.get("backbone_hidden_size", self.backbone.hidden_size))
        if checkpoint_hidden_size != int(self.backbone.hidden_size):
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} uses hidden_size={checkpoint_hidden_size}, "
                f"expected {self.backbone.hidden_size}."
            )
        checkpoint_slots = int(checkpoint.get("writer_memory_slots", self.writer.memory_slots))
        if checkpoint_slots != int(self.writer.memory_slots):
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} uses writer_memory_slots={checkpoint_slots}, "
                f"expected {self.writer.memory_slots}."
            )
        if checkpoint_bridge_mode == "writer_direct":
            checkpoint_writer_stimulus_mode = str(
                checkpoint.get("pilot_writer_stimulus_mode", self.writer_stimulus_mode)
            )
            if checkpoint_writer_stimulus_mode != self.writer_stimulus_mode:
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_writer_stimulus_mode="
                    f"{checkpoint_writer_stimulus_mode}, expected {self.writer_stimulus_mode}."
                )
            checkpoint_writer_context_tokens = int(
                checkpoint.get("pilot_writer_context_tokens", self.writer_context_tokens)
            )
            if checkpoint_writer_context_tokens != int(self.writer_context_tokens):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_writer_context_tokens="
                    f"{checkpoint_writer_context_tokens}, expected {self.writer_context_tokens}."
                )
            checkpoint_writer_conditioning_layers = int(
                checkpoint.get(
                    "pilot_writer_conditioning_layers",
                    getattr(self.writer, "conditioning_layers", 1),
                )
            )
            if checkpoint_writer_conditioning_layers != int(getattr(self.writer, "conditioning_layers", 1)):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_writer_conditioning_layers="
                    f"{checkpoint_writer_conditioning_layers}, expected "
                    f"{getattr(self.writer, 'conditioning_layers', 1)}."
                )
            checkpoint_writer_adapter_enabled = bool(
                checkpoint.get("pilot_writer_adapter_enabled", False)
            )
            if checkpoint_writer_adapter_enabled != self.writer_adapter_enabled:
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_writer_adapter_enabled="
                    f"{checkpoint_writer_adapter_enabled}, expected {self.writer_adapter_enabled}."
                )
            if self.writer_adapter_enabled:
                checkpoint_writer_adapter_target_modules = tuple(
                    str(module_name)
                    for module_name in checkpoint.get("pilot_writer_adapter_target_modules", [])
                )
                if checkpoint_writer_adapter_target_modules != self.writer_adapter_target_modules:
                    raise ValueError(
                        f"Warm-start checkpoint {checkpoint_path} uses pilot_writer_adapter_target_modules="
                        f"{list(checkpoint_writer_adapter_target_modules)}, expected "
                        f"{list(self.writer_adapter_target_modules)}."
                    )
                checkpoint_writer_adapter_rank = int(
                    checkpoint.get("pilot_writer_adapter_rank", self.writer_adapter_rank)
                )
                checkpoint_writer_adapter_alpha = float(
                    checkpoint.get("pilot_writer_adapter_alpha", self.writer_adapter_alpha)
                )
                checkpoint_writer_adapter_dropout = float(
                    checkpoint.get("pilot_writer_adapter_dropout", self.writer_adapter_dropout)
                )
                if checkpoint_writer_adapter_rank != int(self.writer_adapter_rank):
                    raise ValueError(
                        f"Warm-start checkpoint {checkpoint_path} uses pilot_writer_adapter_rank="
                        f"{checkpoint_writer_adapter_rank}, expected {self.writer_adapter_rank}."
                    )
                if abs(checkpoint_writer_adapter_alpha - float(self.writer_adapter_alpha)) > 1e-8:
                    raise ValueError(
                        f"Warm-start checkpoint {checkpoint_path} uses pilot_writer_adapter_alpha="
                        f"{checkpoint_writer_adapter_alpha}, expected {self.writer_adapter_alpha}."
                    )
                if abs(checkpoint_writer_adapter_dropout - float(self.writer_adapter_dropout)) > 1e-8:
                    raise ValueError(
                        f"Warm-start checkpoint {checkpoint_path} uses pilot_writer_adapter_dropout="
                        f"{checkpoint_writer_adapter_dropout}, expected {self.writer_adapter_dropout}."
                    )
        checkpoint_projector_prefix_tokens = int(
            checkpoint.get("pilot_projector_prefix_tokens", checkpoint_slots)
        )
        if checkpoint_projector_prefix_tokens != int(self.prefix_projector.prefix_tokens):
            if not (
                allow_single_level_init_for_two_level
                and self.memory_path_variant == "two_level"
                and checkpoint_memory_path_variant == "single_level"
            ):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_projector_prefix_tokens="
                    f"{checkpoint_projector_prefix_tokens}, expected {self.prefix_projector.prefix_tokens}."
                )
        checkpoint_prefix_source_mode = str(checkpoint.get("pilot_prefix_source_mode", "writer"))
        if checkpoint_prefix_source_mode != self.prefix_source_mode:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} uses pilot_prefix_source_mode="
                f"{checkpoint_prefix_source_mode}, expected {self.prefix_source_mode}."
            )
        checkpoint_deep_prefix_init_mode = str(
            checkpoint.get("pilot_deep_prefix_init_mode", self.deep_prefix_init_mode)
        )
        if checkpoint_deep_prefix_init_mode != self.deep_prefix_init_mode:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} uses pilot_deep_prefix_init_mode="
                f"{checkpoint_deep_prefix_init_mode}, expected {self.deep_prefix_init_mode}."
            )
        writer_state = checkpoint.get("writer_state")
        if not isinstance(writer_state, dict):
            raise ValueError(f"Warm-start checkpoint {checkpoint_path} is missing writer_state.")
        _validate_module_state_shapes(
            module_name="writer",
            expected_state=self.writer.state_dict(),
            checkpoint_state=writer_state,
            checkpoint_path=checkpoint_path,
        )
        source_stub_state = checkpoint.get("source_stub_state")
        if self.source_stub is None and source_stub_state is not None:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} includes source_stub_state, but the current "
                f"runtime uses pilot_prefix_source_mode={self.prefix_source_mode}."
            )
        if self.source_stub is not None:
            if not isinstance(source_stub_state, dict):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} is missing source_stub_state for "
                    f"pilot_prefix_source_mode={self.prefix_source_mode}."
                )
            _validate_module_state_shapes(
                module_name="source_stub",
                expected_state=self.source_stub.state_dict(),
                checkpoint_state=source_stub_state,
                checkpoint_path=checkpoint_path,
            )
        support_encoder_state = checkpoint.get("support_encoder_state")
        if self.support_encoder is None and support_encoder_state is not None:
            raise ValueError(
                f"Warm-start checkpoint {checkpoint_path} includes support_encoder_state, but the current "
                f"runtime uses pilot_support_encoder_mode={self.support_encoder_mode}."
            )
        if self.support_encoder is not None:
            if not isinstance(support_encoder_state, dict):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} is missing support_encoder_state for "
                    f"pilot_support_encoder_mode={self.support_encoder_mode}."
                )
            _validate_module_state_shapes(
                module_name="support_encoder",
                expected_state=self.support_encoder.state_dict(),
                checkpoint_state=support_encoder_state,
                checkpoint_path=checkpoint_path,
            )
        prefix_projector_state = checkpoint.get("prefix_projector_state")
        if not isinstance(prefix_projector_state, dict):
            raise ValueError(f"Warm-start checkpoint {checkpoint_path} is missing prefix_projector_state.")
        if not (
            allow_single_level_init_for_two_level
            and self.memory_path_variant == "two_level"
            and checkpoint_memory_path_variant == "single_level"
        ):
            _validate_module_state_shapes(
                module_name="prefix_projector",
                expected_state=self.prefix_projector.state_dict(),
                checkpoint_state=prefix_projector_state,
                checkpoint_path=checkpoint_path,
            )
        if self.memory_path_variant == "two_level":
            checkpoint_reader_context_mode = str(
                checkpoint.get("pilot_reader_context_mode", "prompt_summary")
            )
            if (
                checkpoint_memory_path_variant == "two_level"
                and checkpoint_reader_context_mode != self.reader_context_mode
            ):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_reader_context_mode="
                    f"{checkpoint_reader_context_mode}, expected {self.reader_context_mode}."
                )
            checkpoint_projector_token_source = str(
                checkpoint.get("pilot_projector_token_source", "short_slots")
            )
            if (
                checkpoint_memory_path_variant == "two_level"
                and checkpoint_projector_token_source != self.projector_token_source
            ):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_projector_token_source="
                    f"{checkpoint_projector_token_source}, expected {self.projector_token_source}."
                )
            if checkpoint_memory_path_variant == "two_level":
                checkpoint_reader_queries = int(
                    checkpoint.get("pilot_reader_num_queries", self.reader_num_queries)
                )
                checkpoint_short_slots = int(
                    checkpoint.get("pilot_fuser_short_slots", self.fuser_short_slots)
                )
                if checkpoint_reader_queries != int(self.reader_num_queries):
                    raise ValueError(
                        f"Warm-start checkpoint {checkpoint_path} uses pilot_reader_num_queries="
                        f"{checkpoint_reader_queries}, expected {self.reader_num_queries}."
                    )
                if checkpoint_short_slots != int(self.fuser_short_slots):
                    raise ValueError(
                        f"Warm-start checkpoint {checkpoint_path} uses pilot_fuser_short_slots="
                        f"{checkpoint_short_slots}, expected {self.fuser_short_slots}."
                    )
                reader_state = checkpoint.get("reader_state")
                fuser_state = checkpoint.get("fuser_state")
                if not isinstance(reader_state, dict):
                    raise ValueError(f"Warm-start checkpoint {checkpoint_path} is missing reader_state.")
                if not isinstance(fuser_state, dict):
                    raise ValueError(f"Warm-start checkpoint {checkpoint_path} is missing fuser_state.")
                if self.reader is None or self.fuser is None:
                    raise RuntimeError("two_level runtime requires reader and fuser modules.")
                _validate_module_state_shapes(
                    module_name="reader",
                    expected_state=self.reader.state_dict(),
                    checkpoint_state=reader_state,
                    checkpoint_path=checkpoint_path,
                )
                _validate_module_state_shapes(
                    module_name="fuser",
                    expected_state=self.fuser.state_dict(),
                    checkpoint_state=fuser_state,
                    checkpoint_path=checkpoint_path,
                )
        checkpoint_receiver_lora_enabled = bool(
            checkpoint.get("pilot_receiver_lora_enabled", checkpoint.get("receiver_lora_state") is not None)
        )
        if checkpoint_receiver_lora_enabled != self.receiver_lora_enabled:
            if not (
                allow_missing_receiver_lora_state
                and self.receiver_lora_enabled
                and not checkpoint_receiver_lora_enabled
            ):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_receiver_lora_enabled="
                    f"{checkpoint_receiver_lora_enabled}, expected {self.receiver_lora_enabled}."
                )
        if self.receiver_lora_enabled and checkpoint_receiver_lora_enabled:
            checkpoint_receiver_lora_layers = tuple(
                int(layer_index) for layer_index in checkpoint.get("pilot_receiver_lora_target_layers", [])
            )
            checkpoint_receiver_lora_modules = tuple(
                str(module_name)
                for module_name in checkpoint.get("pilot_receiver_lora_target_modules", [])
            )
            if checkpoint_receiver_lora_layers != self.receiver_lora_target_layers:
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_receiver_lora_target_layers="
                    f"{list(checkpoint_receiver_lora_layers)}, expected {list(self.receiver_lora_target_layers)}."
                )
            if checkpoint_receiver_lora_modules != self.receiver_lora_target_modules:
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_receiver_lora_target_modules="
                    f"{list(checkpoint_receiver_lora_modules)}, expected {list(self.receiver_lora_target_modules)}."
                )
            checkpoint_receiver_lora_rank = int(
                checkpoint.get("pilot_receiver_lora_rank", self.receiver_lora_rank)
            )
            checkpoint_receiver_lora_alpha = float(
                checkpoint.get("pilot_receiver_lora_alpha", self.receiver_lora_alpha)
            )
            checkpoint_receiver_lora_dropout = float(
                checkpoint.get("pilot_receiver_lora_dropout", self.receiver_lora_dropout)
            )
            if checkpoint_receiver_lora_rank != int(self.receiver_lora_rank):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_receiver_lora_rank="
                    f"{checkpoint_receiver_lora_rank}, expected {self.receiver_lora_rank}."
                )
            if abs(checkpoint_receiver_lora_alpha - float(self.receiver_lora_alpha)) > 1e-8:
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_receiver_lora_alpha="
                    f"{checkpoint_receiver_lora_alpha}, expected {self.receiver_lora_alpha}."
                )
            if abs(checkpoint_receiver_lora_dropout - float(self.receiver_lora_dropout)) > 1e-8:
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} uses pilot_receiver_lora_dropout="
                    f"{checkpoint_receiver_lora_dropout}, expected {self.receiver_lora_dropout}."
                )
            receiver_lora_state = checkpoint.get("receiver_lora_state")
            if receiver_lora_state is None:
                if not allow_missing_receiver_lora_state:
                    raise ValueError(
                        f"Warm-start checkpoint {checkpoint_path} is missing receiver_lora_state."
                    )
            elif not isinstance(receiver_lora_state, dict):
                raise ValueError(
                    f"Warm-start checkpoint {checkpoint_path} has invalid receiver_lora_state."
                )
            else:
                self.backbone.validate_receiver_lora_state_dict(
                    receiver_lora_state,
                    checkpoint_path=str(checkpoint_path),
                )

    def warm_start_from_injection_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._validate_injection_checkpoint_schema(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            allow_single_level_init_for_two_level=True,
            allow_missing_receiver_lora_state=True,
        )
        self.writer.load_state_dict(checkpoint["writer_state"])
        if self.source_stub is not None and checkpoint.get("source_stub_state") is not None:
            self.source_stub.load_state_dict(checkpoint["source_stub_state"])
        if self.support_encoder is not None and checkpoint.get("support_encoder_state") is not None:
            self.support_encoder.load_state_dict(checkpoint["support_encoder_state"])
        if not (
            self.memory_path_variant == "two_level"
            and str(checkpoint.get("pilot_memory_path_variant", "single_level")) == "single_level"
        ):
            self.prefix_projector.load_state_dict(checkpoint["prefix_projector_state"])
        if self.receiver_lora_enabled and checkpoint.get("receiver_lora_state") is not None:
            self.backbone.load_receiver_lora_state_dict(
                checkpoint["receiver_lora_state"],
                checkpoint_path=str(checkpoint_path),
            )
        if (
            self.memory_path_variant == "two_level"
            and checkpoint.get("pilot_memory_path_variant") == "two_level"
        ):
            if self.reader is None or self.fuser is None:
                raise RuntimeError("two_level runtime requires reader and fuser modules.")
            self.reader.load_state_dict(checkpoint["reader_state"])
            self.fuser.load_state_dict(checkpoint["fuser_state"])
        return checkpoint

    def load_injection_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._validate_injection_checkpoint_schema(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            allow_single_level_init_for_two_level=False,
            allow_missing_receiver_lora_state=False,
        )
        self.writer.load_state_dict(checkpoint["writer_state"])
        if self.source_stub is not None and checkpoint.get("source_stub_state") is not None:
            self.source_stub.load_state_dict(checkpoint["source_stub_state"])
        if self.support_encoder is not None and checkpoint.get("support_encoder_state") is not None:
            self.support_encoder.load_state_dict(checkpoint["support_encoder_state"])
        self.prefix_projector.load_state_dict(checkpoint["prefix_projector_state"])
        if self.receiver_lora_enabled and checkpoint.get("receiver_lora_state") is not None:
            self.backbone.load_receiver_lora_state_dict(
                checkpoint["receiver_lora_state"],
                checkpoint_path=str(checkpoint_path),
            )
        if self.memory_path_variant == "two_level":
            if self.reader is None or self.fuser is None:
                raise RuntimeError("two_level runtime requires reader and fuser modules.")
            self.reader.load_state_dict(checkpoint["reader_state"])
            self.fuser.load_state_dict(checkpoint["fuser_state"])
        return checkpoint

    def set_writer_trainable(self, enabled: bool) -> None:
        self.set_writer_base_trainable(enabled)
        self.set_writer_adapter_trainable(enabled)

    def writer_base_parameters(self) -> list[nn.Parameter]:
        parameter_getter = getattr(self.writer, "writer_base_parameters", None)
        if callable(parameter_getter):
            return list(parameter_getter())
        return list(self.writer.parameters())

    def writer_adapter_parameters(self) -> list[nn.Parameter]:
        parameter_getter = getattr(self.writer, "writer_lora_parameters", None)
        if callable(parameter_getter):
            return list(parameter_getter())
        return []

    def set_writer_base_trainable(self, enabled: bool) -> None:
        setter = getattr(self.writer, "set_base_trainable", None)
        if callable(setter):
            setter(enabled)
            return
        for parameter in self.writer.parameters():
            parameter.requires_grad_(enabled)

    def set_writer_adapter_trainable(self, enabled: bool) -> None:
        setter = getattr(self.writer, "set_writer_lora_trainable", None)
        if callable(setter):
            setter(enabled)

    def set_support_encoder_trainable(self, enabled: bool) -> None:
        if self.support_encoder is None:
            return
        for parameter in self.support_encoder.parameters():
            parameter.requires_grad_(enabled)

    def set_prefix_projector_trainable(self, enabled: bool) -> None:
        for parameter in self.prefix_projector.parameters():
            parameter.requires_grad_(enabled)

    def _prompt_summary(self, prompt_text: str | None) -> torch.Tensor | None:
        if self.reader_context_mode == "none" or not prompt_text:
            return None
        cached = self._prompt_summary_cache.get(prompt_text)
        if cached is None:
            with torch.no_grad():
                cached = self.backbone.summarize_texts([prompt_text]).detach()
            self._prompt_summary_cache[prompt_text] = cached
        return cached.to(device=self.backbone.device, dtype=torch.float32)

    def _prompt_context_states(
        self,
        prompt_text: str | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            not prompt_text
            or self.bridge_mode != "writer_direct"
            or self.writer_stimulus_mode == "support_only"
        ):
            return None, None
        cached = self._prompt_context_cache.get(prompt_text)
        if cached is None:
            extract_hidden = getattr(self.backbone, "extract_prompt_hidden_state_slice", None)
            if callable(extract_hidden):
                with torch.no_grad():
                    hidden_states, hidden_mask = extract_hidden(
                        [prompt_text],
                        max_tokens=self.writer_context_tokens,
                    )
            else:
                with torch.no_grad():
                    hidden_states = self.backbone.summarize_texts([prompt_text]).unsqueeze(1).detach()
                hidden_mask = torch.ones(
                    1,
                    1,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
            cached = (
                hidden_states.detach().to(dtype=torch.float32),
                hidden_mask.detach().to(dtype=torch.bool),
            )
            self._prompt_context_cache[prompt_text] = cached
        hidden_states, hidden_mask = cached
        return (
            hidden_states.to(device=self.backbone.device, dtype=torch.float32),
            hidden_mask.to(device=self.backbone.device, dtype=torch.bool),
        )

    def _augment_prefix_stats_with_projection(
        self,
        *,
        prefix_stats: dict[str, Any],
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None,
    ) -> dict[str, Any]:
        if not layer_prefix_hidden_by_layer:
            return prefix_stats
        projection_stats = self.backbone.summarize_layer_prefix_projection(
            layer_prefix_hidden_by_layer,
            batch_size=int(next(iter(layer_prefix_hidden_by_layer.values())).shape[0]),
        )
        merged = dict(prefix_stats)
        merged.update(projection_stats)
        return merged

    def _deep_prefix_calibration_texts(
        self,
        *,
        support_text_block: str,
        prompt_text: str | None,
    ) -> list[str]:
        texts: list[str] = []
        if prompt_text is not None and str(prompt_text).strip():
            texts.append(str(prompt_text))
        if str(support_text_block).strip():
            texts.append(str(support_text_block))
        if not texts:
            texts.append("<empty>")
        return texts

    def _maybe_initialize_prefix_source(
        self,
        *,
        support_text_block: str,
        prompt_text: str | None,
    ) -> None:
        if self.injection_mode != "sparse_deep_prefix":
            return
        calibration = None
        if (
            not self._deep_prefix_projector_initialized
            or (self.source_stub is not None and not self._source_stub_initialized)
        ):
            calibration = self.backbone.collect_deep_prefix_calibration(
                self._deep_prefix_calibration_texts(
                    support_text_block=support_text_block,
                    prompt_text=prompt_text,
                ),
                layer_indices=self.deep_prefix_layers,
                max_tokens=int(self.prefix_projector.prefix_tokens),
            )
        if not self._deep_prefix_projector_initialized:
            assert calibration is not None
            if isinstance(self.prefix_projector, SharedLowRankDeepPrefixProjector):
                self.prefix_projector.initialize_from_calibration(
                    mode=self.deep_prefix_init_mode,
                    semantic_anchor=calibration.get("semantic_anchor"),
                    hidden_state_anchor=calibration.get("hidden_state_anchor"),
                    layer_hidden_anchor_by_layer=calibration.get("layer_hidden_anchor_by_layer"),
                )
            self._deep_prefix_projector_initialized = True
        if self.source_stub is not None and not self._source_stub_initialized:
            assert calibration is not None
            if self.deep_prefix_init_mode == "semantic_anchor":
                self.source_stub.initialize_from_anchor(calibration["semantic_anchor"])
            else:
                self.source_stub.initialize_from_anchor(calibration["hidden_state_anchor"])
            self._source_stub_initialized = True

    def _writer_support_states(
        self,
        support_text_block: str,
        *,
        support_rows: list[dict[str, Any]] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.support_encoder_mode == "structured_support_set":
            if not support_rows:
                raise ValueError(
                    "Structured support-set encoding requires non-empty support_rows for injected arms."
                )
            support_row_texts = _serialize_support_rows(
                support_rows,
                support_serialization_variant=self.support_serialization_variant,
            )
            support_item_states = self.backbone.summarize_texts(support_row_texts).unsqueeze(0)
            if self.support_encoder is None:
                raise RuntimeError("support_encoder_mode=structured_support_set requires support_encoder.")
            encoded_support_states = self.support_encoder(
                support_item_states,
                _support_label_ids(support_rows, device=self.backbone.device),
            )
            return support_item_states, encoded_support_states
        support_state = self.backbone.summarize_texts([support_text_block]).unsqueeze(1)
        return support_state, support_state

    def _two_level_memory_short(
        self,
        *,
        memory_long: torch.Tensor,
        prompt_text: str | None,
    ) -> dict[str, torch.Tensor | None]:
        if self.reader is None or self.fuser is None:
            raise RuntimeError("two_level runtime requires reader and fuser modules.")
        reader_context = self._prompt_summary(prompt_text)
        reader_outputs = self.reader.read(memory_long, context=reader_context)
        memory_short = self.fuser.fuse(reader_outputs["readouts"])
        return {
            "memory_short": memory_short,
            "reader_attention": reader_outputs["attention"],
            "reader_gates": reader_outputs["gates"],
            "reader_context": reader_context,
            "reader_base_queries": reader_outputs.get("base_queries"),
            "reader_conditioned_queries": reader_outputs.get("conditioned_queries"),
            "reader_attention_logits": reader_outputs.get("attention_logits"),
            "reader_value_projected_slots": reader_outputs.get("value_projected_slots"),
            "reader_readouts": reader_outputs.get("readouts"),
        }

    def build_prefix_artifacts(
        self,
        support_text_block: str,
        *,
        support_rows: list[dict[str, Any]] | None = None,
        prompt_text: str | None = None,
    ) -> PrefixInjectionArtifacts:
        support_item_states = None
        writer_support_states = None
        writer_context_states = None
        writer_context_mask = None
        if self.writer_memory_control == "zero":
            memory_long = torch.zeros(
                1,
                int(self.writer.memory_slots),
                self.backbone.hidden_size,
                dtype=torch.float32,
                device=self.backbone.device,
            )
        elif self.prefix_source_mode == "source_stub":
            if prompt_text is not None:
                writer_context_states, writer_context_mask = self._prompt_context_states(prompt_text)
            self._maybe_initialize_prefix_source(
                support_text_block=support_text_block,
                prompt_text=prompt_text,
            )
            if self.source_stub is None:
                raise RuntimeError("pilot_prefix_source_mode=source_stub requires SourceStubMemory.")
            memory_long = self.source_stub(batch_size=1)
        else:
            support_item_states, writer_support_states = self._writer_support_states(
                support_text_block,
                support_rows=support_rows,
            )
            if self.bridge_mode == "writer_direct":
                writer_context_states, writer_context_mask = self._prompt_context_states(prompt_text)
                if not isinstance(self.writer, WriterWeaverHead):
                    raise RuntimeError("writer_direct bridge mode requires WriterWeaverHead.")
                memory_long = self.writer.write(
                    context_states=writer_context_states,
                    support_states=writer_support_states,
                    stimulus_mode=self.writer_stimulus_mode,
                    context_key_padding_mask=(
                        None if writer_context_mask is None else ~writer_context_mask.to(dtype=torch.bool)
                    ),
                    support_key_padding_mask=None,
                )
            elif writer_support_states is not None and writer_support_states.shape[1] > 1:
                memory_long = self.writer.write(
                    writer_support_states,
                    input_schema="support_set",
                )
            else:
                pooled_support_state = (
                    writer_support_states[:, 0, :]
                    if writer_support_states is not None and writer_support_states.ndim == 3
                    else self.backbone.summarize_texts([support_text_block])
                )
                memory_long = self.writer.write(pooled_support_state, input_schema="pooled_state")
        memory_short = None
        reader_attention = None
        reader_gates = None
        reader_context = None
        reader_base_queries = None
        reader_conditioned_queries = None
        reader_attention_logits = None
        reader_value_projected_slots = None
        reader_readouts = None
        projector_source = memory_long
        if self.memory_path_variant == "two_level":
            if self.writer_memory_control == "zero":
                memory_short = torch.zeros(
                    1,
                    int(self.fuser_short_slots),
                    self.backbone.hidden_size,
                    dtype=torch.float32,
                    device=self.backbone.device,
                )
                reader_attention = torch.zeros(
                    1,
                    int(self.reader_num_queries),
                    int(self.writer.memory_slots),
                    dtype=torch.float32,
                    device=self.backbone.device,
                )
                reader_gates = torch.zeros(
                    1,
                    int(self.reader_num_queries),
                    dtype=torch.float32,
                    device=self.backbone.device,
                )
            else:
                two_level_outputs = self._two_level_memory_short(
                    memory_long=memory_long,
                    prompt_text=prompt_text,
                )
                memory_short = two_level_outputs["memory_short"]
                reader_attention = two_level_outputs["reader_attention"]
                reader_gates = two_level_outputs["reader_gates"]
                reader_context = two_level_outputs["reader_context"]
                reader_base_queries = two_level_outputs["reader_base_queries"]
                reader_conditioned_queries = two_level_outputs["reader_conditioned_queries"]
                reader_attention_logits = two_level_outputs["reader_attention_logits"]
                reader_value_projected_slots = two_level_outputs["reader_value_projected_slots"]
                reader_readouts = two_level_outputs["reader_readouts"]
            projector_source = memory_long if self.projector_token_source == "writer_slots" else memory_short
            if projector_source is None:
                raise RuntimeError("two_level projector source resolved to None.")
        if self.injection_mode == "shallow_prefix":
            prefix_embeddings = self.prefix_projector(projector_source)
            prefix_stats = _prefix_stats(
                prefix_embeddings,
                memory_slots=memory_long,
                memory_long=memory_long,
                memory_short=memory_short,
                support_item_states=support_item_states,
                writer_support_states=writer_support_states,
                writer_context_states=writer_context_states,
                reader_attention=reader_attention,
                reader_base_queries=reader_base_queries,
                reader_conditioned_queries=reader_conditioned_queries,
                reader_attention_logits=reader_attention_logits,
                reader_value_projected_slots=reader_value_projected_slots,
                reader_readouts=reader_readouts,
                fuser=self.fuser,
                memory_path_variant=self.memory_path_variant,
                projector_token_source=self.projector_token_source,
                prefix_source_mode=self.prefix_source_mode,
                deep_prefix_init_mode=self.deep_prefix_init_mode,
            )
            return PrefixInjectionArtifacts(
                prefix_embeddings=prefix_embeddings,
                layer_prefix_hidden_by_layer=None,
                prefix_stats=prefix_stats,
                memory_slots=memory_long,
                support_item_states=support_item_states,
                writer_support_states=writer_support_states,
                writer_context_states=writer_context_states,
                writer_context_mask=writer_context_mask,
                memory_long=memory_long,
                memory_short=memory_short,
                reader_attention=reader_attention,
                reader_gates=reader_gates,
                reader_queries=None if self.reader is None else self.reader.queries.detach(),
                reader_context=reader_context,
                reader_conditioned_queries=reader_conditioned_queries,
                reader_value_projected_slots=reader_value_projected_slots,
                reader_readouts=reader_readouts,
            )
        self._maybe_initialize_prefix_source(
            support_text_block=support_text_block,
            prompt_text=prompt_text,
        )
        layer_prefix_hidden_by_layer = self.prefix_projector(projector_source)
        prefix_stats = _prefix_stats(
            layer_prefix_hidden_by_layer=layer_prefix_hidden_by_layer,
            memory_slots=memory_long,
            memory_long=memory_long,
            memory_short=memory_short,
            support_item_states=support_item_states,
            writer_support_states=writer_support_states,
            writer_context_states=writer_context_states,
            reader_attention=reader_attention,
            reader_base_queries=reader_base_queries,
            reader_conditioned_queries=reader_conditioned_queries,
            reader_attention_logits=reader_attention_logits,
            reader_value_projected_slots=reader_value_projected_slots,
            reader_readouts=reader_readouts,
            fuser=self.fuser,
            memory_path_variant=self.memory_path_variant,
            projector_token_source=self.projector_token_source,
            prefix_source_mode=self.prefix_source_mode,
            deep_prefix_init_mode=self.deep_prefix_init_mode,
        )
        prefix_stats = self._augment_prefix_stats_with_projection(
            prefix_stats=prefix_stats,
            layer_prefix_hidden_by_layer=layer_prefix_hidden_by_layer,
        )
        return PrefixInjectionArtifacts(
            prefix_embeddings=None,
            layer_prefix_hidden_by_layer=layer_prefix_hidden_by_layer,
            prefix_stats=prefix_stats,
            memory_slots=memory_long,
            support_item_states=support_item_states,
            writer_support_states=writer_support_states,
            writer_context_states=writer_context_states,
            writer_context_mask=writer_context_mask,
            memory_long=memory_long,
            memory_short=memory_short,
            reader_attention=reader_attention,
            reader_gates=reader_gates,
            reader_queries=None if self.reader is None else self.reader.queries.detach(),
            reader_context=reader_context,
            reader_conditioned_queries=reader_conditioned_queries,
            reader_value_projected_slots=reader_value_projected_slots,
            reader_readouts=reader_readouts,
        )

    def score_example(
        self,
        example_cache: SharedInjectionExampleCache,
        *,
        support_text_block: str,
        prefix_embeddings: torch.Tensor | None,
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if self.arm == "teacher_text":
            prompt = _serialize_teacher_prompt(example_cache.prompt_text, support_text_block)
            return self.backbone.score_continuations(
                prompt,
                example_cache.candidate_texts,
                return_diagnostics=return_diagnostics,
            )
        return self.backbone.score_continuations(
            example_cache.prompt_text,
            example_cache.candidate_texts,
            prefix_embeddings=prefix_embeddings,
            layer_prefix_hidden_by_layer=layer_prefix_hidden_by_layer,
            return_diagnostics=return_diagnostics,
        )

    def generate_text(
        self,
        *,
        prompt_text: str,
        support_text_block: str,
        prefix_embeddings: torch.Tensor | None,
        layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None,
    ) -> str:
        if self.arm == "teacher_text":
            prompt = _serialize_teacher_prompt(prompt_text, support_text_block)
            return self.backbone.generate([prompt])[0]
        return self.backbone.generate(
            [prompt_text],
            prefix_embeddings=prefix_embeddings,
            layer_prefix_hidden_by_layer=layer_prefix_hidden_by_layer,
        )[0]


def _strongest_competitor_index(scores: torch.Tensor, gold_index: int) -> int:
    competitor_indices = [index for index in range(scores.shape[0]) if index != gold_index]
    if not competitor_indices:
        return gold_index
    detached_scores = scores.detach()
    return max(competitor_indices, key=lambda index: float(detached_scores[index].item()))


def _compute_margin(scores: torch.Tensor, gold_index: int) -> tuple[float, int]:
    competitor_index = _strongest_competitor_index(scores, gold_index)
    if competitor_index == gold_index:
        return float(scores[gold_index].item()), competitor_index
    return float(scores[gold_index].item() - scores[competitor_index].item()), competitor_index


def _task_training_loss(
    scores: torch.Tensor,
    example_cache: SharedInjectionExampleCache,
    *,
    margin_value: float,
    ce_weight: float = 1.0,
    competitor_hinge_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if example_cache.task_mode == "generation":
        zero = torch.zeros((), dtype=scores.dtype, device=scores.device)
        return -scores[example_cache.gold_index], zero, zero
    return _choice_task_loss(
        scores,
        example_cache.gold_index,
        margin_value=margin_value,
        ce_weight=ce_weight,
        competitor_hinge_weight=competitor_hinge_weight,
    )


def _choice_task_loss(
    scores: torch.Tensor,
    gold_index: int,
    *,
    margin_value: float,
    ce_weight: float = 1.0,
    competitor_hinge_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = F.cross_entropy(
        scores.unsqueeze(0),
        torch.tensor([gold_index], dtype=torch.long, device=scores.device),
    )
    competitor_index = _strongest_competitor_index(scores, gold_index)
    gold_score = scores[gold_index].view(1)
    competitor_score = scores[competitor_index].view(1)
    target = torch.ones(1, dtype=torch.float32, device=scores.device)
    margin_loss = F.margin_ranking_loss(
        gold_score,
        competitor_score,
        target,
        margin=margin_value,
    )
    total_loss = (float(ce_weight) * ce_loss) + (float(competitor_hinge_weight) * margin_loss)
    return total_loss, ce_loss, margin_loss


def _score_margin_tensor(scores: torch.Tensor, gold_index: int) -> torch.Tensor:
    competitor_index = _strongest_competitor_index(scores, gold_index)
    if competitor_index == gold_index:
        return scores[gold_index]
    return scores[gold_index] - scores[competitor_index]


def _answer_logprob(scores: torch.Tensor, gold_index: int) -> torch.Tensor:
    return scores[gold_index]


def _writer_gain_margin_loss(
    *,
    with_memory_scores: torch.Tensor,
    without_memory_scores: torch.Tensor,
    gold_index: int,
    margin_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = _answer_logprob(with_memory_scores, gold_index) - _answer_logprob(
        without_memory_scores,
        gold_index,
    )
    margin_loss = torch.clamp(float(margin_value) - delta, min=0.0)
    return margin_loss, delta


def _writer_covariance_diversity_loss(memory_long: torch.Tensor | None) -> torch.Tensor | None:
    return _slot_diversity_loss(memory_long)


def _teacher_advantage_weight(
    teacher_margin: torch.Tensor,
    base_margin: torch.Tensor,
    *,
    center: float,
    scale: float,
) -> torch.Tensor:
    safe_scale = max(float(scale), 1e-6)
    return torch.sigmoid(((teacher_margin - base_margin) - float(center)) / safe_scale)


def _choice_distribution(scores: torch.Tensor, *, temperature: float) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-6)
    return torch.softmax(scores.to(dtype=torch.float32) / safe_temperature, dim=-1)


def _alignment_aux_choice_loss(
    *,
    mode: str,
    active_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-6)
    active_scores_fp32 = active_scores.to(dtype=torch.float32)
    teacher_scores_fp32 = teacher_scores.to(dtype=torch.float32).detach()
    teacher_probs = _choice_distribution(teacher_scores_fp32, temperature=safe_temperature).unsqueeze(0)
    student_log_probs = F.log_softmax(active_scores_fp32 / safe_temperature, dim=-1).unsqueeze(0)
    tau_sq = safe_temperature * safe_temperature
    if mode == "teacher_choice_kl":
        return tau_sq * F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    if mode != "teacher_choice_js":
        raise ValueError(f"Unsupported choice-space alignment aux mode: {mode}.")
    student_probs = _choice_distribution(active_scores_fp32, temperature=safe_temperature)
    mixture = 0.5 * (teacher_probs.squeeze(0) + student_probs.detach())
    mixture = mixture.clamp_min(1e-8)
    student_kl = F.kl_div(student_log_probs, mixture.unsqueeze(0), reduction="batchmean")
    teacher_kl = F.kl_div(
        torch.log(mixture).unsqueeze(0),
        teacher_probs,
        reduction="batchmean",
    )
    return tau_sq * 0.5 * (student_kl + teacher_kl)


def _effective_rank(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    matrix = tensor.to(dtype=torch.float32)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    elif matrix.ndim != 3:
        raise ValueError(f"Expected rank-2 or rank-3 tensor for effective rank, got shape={tuple(matrix.shape)}.")
    singular_values = torch.linalg.svdvals(matrix)
    totals = singular_values.sum(dim=-1, keepdim=True)
    nonzero = totals.squeeze(-1) > 0.0
    probs = singular_values / totals.clamp_min(1e-8)
    probs = probs.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    effective_rank = torch.exp(entropy)
    effective_rank = torch.where(nonzero, effective_rank, torch.zeros_like(effective_rank))
    return float(effective_rank.mean().item())


def _class_entropy(scores: torch.Tensor, *, temperature: float = 1.0) -> float:
    probs = _choice_distribution(scores, temperature=temperature).clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum()
    return float(entropy.item())


def _alignment_aux_loss(
    *,
    mode: str,
    active_scores: torch.Tensor,
    base_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    gold_index: int,
    temperature: float = 1.0,
    advantage_center: float = 0.0,
    advantage_scale: float = 0.25,
) -> tuple[torch.Tensor | None, bool, dict[str, float]]:
    active_scores_fp32 = active_scores.to(dtype=torch.float32)
    base_scores_fp32 = base_scores.to(dtype=torch.float32)
    teacher_scores_fp32 = teacher_scores.to(dtype=torch.float32)
    active_margin = _score_margin_tensor(active_scores_fp32, gold_index)
    base_margin = _score_margin_tensor(base_scores_fp32, gold_index).detach()
    teacher_margin = _score_margin_tensor(teacher_scores_fp32, gold_index).detach()
    advantage_weight = _teacher_advantage_weight(
        teacher_margin,
        base_margin,
        center=advantage_center,
        scale=advantage_scale,
    ).detach()
    teacher_choice_kl = _alignment_aux_choice_loss(
        mode="teacher_choice_kl",
        active_scores=active_scores_fp32,
        teacher_scores=teacher_scores_fp32,
        temperature=temperature,
    )
    teacher_choice_js = _alignment_aux_choice_loss(
        mode="teacher_choice_js",
        active_scores=active_scores_fp32,
        teacher_scores=teacher_scores_fp32,
        temperature=temperature,
    )
    diagnostics = {
        "teacher_choice_kl": float(teacher_choice_kl.detach().item()),
        "teacher_choice_js": float(teacher_choice_js.detach().item()),
        "teacher_advantage_weight_mean": float(advantage_weight.item()),
        "teacher_advantage_weight_max": float(advantage_weight.item()),
        "teacher_margin_minus_base_margin": float((teacher_margin - base_margin).item()),
        "teacher_margin_minus_active_margin": float((teacher_margin - active_margin.detach()).item()),
        "active_class_entropy": _class_entropy(active_scores_fp32),
        "teacher_class_entropy": _class_entropy(teacher_scores_fp32),
        "base_class_entropy": _class_entropy(base_scores_fp32),
    }
    if mode == "off":
        return None, False, diagnostics
    if mode == "teacher_margin":
        if float(teacher_margin.item()) <= float(base_margin.item()):
            return None, False, diagnostics
        target = teacher_margin.to(device=active_scores_fp32.device, dtype=torch.float32)
        aux_loss = F.smooth_l1_loss(active_margin, target)
        return aux_loss, True, diagnostics
    if mode == "teacher_choice_kl":
        aux_loss = advantage_weight.to(device=active_scores_fp32.device, dtype=torch.float32) * teacher_choice_kl
        return aux_loss, bool(float(advantage_weight.item()) > 1e-6), diagnostics
    if mode == "teacher_choice_js":
        aux_loss = advantage_weight.to(device=active_scores_fp32.device, dtype=torch.float32) * teacher_choice_js
        return aux_loss, bool(float(advantage_weight.item()) > 1e-6), diagnostics
    raise ValueError(f"Unsupported alignment aux mode: {mode}.")


def _cosine_anchor_loss(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(current.shape) != tuple(reference.shape):
        raise ValueError(
            f"Anchor tensors must share the same shape, got current={tuple(current.shape)} "
            f"and reference={tuple(reference.shape)}."
        )
    current_flat = current.reshape(current.shape[0], -1)
    reference_flat = reference.reshape(reference.shape[0], -1)
    cosine = F.cosine_similarity(current_flat, reference_flat, dim=1).mean()
    return 1.0 - cosine, cosine


def _reference_latent_targets(
    *,
    runtime: "SharedInjectionPilotRuntime",
    reference_support_encoder: nn.Module | None,
    reference_writer: MemoryWriter,
    support_text_block: str,
    support_rows: list[dict[str, Any]] | None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if runtime.writer_memory_control == "zero":
        reference_memory_slots = torch.zeros(
            1,
            int(reference_writer.memory_slots),
            runtime.backbone.hidden_size,
            dtype=torch.float32,
            device=runtime.backbone.device,
        )
        return None, reference_memory_slots
    if runtime.support_encoder_mode == "structured_support_set":
        if not support_rows:
            raise ValueError("Latent anchor reference requires support_rows for structured support-set mode.")
        if reference_support_encoder is None:
            raise RuntimeError("Structured latent anchor reference requires reference_support_encoder.")
        support_row_texts = _serialize_support_rows(
            support_rows,
            support_serialization_variant=runtime.support_serialization_variant,
        )
        support_item_states = runtime.backbone.summarize_texts(support_row_texts).unsqueeze(0)
        encoded_support_states = reference_support_encoder(
            support_item_states,
            _support_label_ids(support_rows, device=runtime.backbone.device),
        )
        reference_memory_slots = reference_writer.write(
            encoded_support_states,
            input_schema="support_set",
        )
        return encoded_support_states, reference_memory_slots
    support_state = runtime.backbone.summarize_texts([support_text_block])
    reference_memory_slots = reference_writer.write(support_state, input_schema="pooled_state")
    return None, reference_memory_slots


def _latent_anchor_loss(
    *,
    current_support_states: torch.Tensor | None,
    reference_support_states: torch.Tensor | None,
    current_memory_slots: torch.Tensor | None,
    reference_memory_slots: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if current_memory_slots is None or reference_memory_slots is None:
        raise ValueError("Latent anchor loss requires current and reference memory slots.")
    writer_anchor_loss, writer_cosine = _cosine_anchor_loss(current_memory_slots, reference_memory_slots)
    if current_support_states is None or reference_support_states is None:
        zero = torch.zeros((), dtype=writer_anchor_loss.dtype, device=writer_anchor_loss.device)
        return writer_anchor_loss, zero, writer_anchor_loss, zero, writer_cosine
    support_anchor_loss, support_cosine = _cosine_anchor_loss(current_support_states, reference_support_states)
    total_anchor_loss = 0.5 * support_anchor_loss + 0.5 * writer_anchor_loss
    return total_anchor_loss, support_anchor_loss, writer_anchor_loss, support_cosine, writer_cosine


def _slot_diversity_loss(slots: torch.Tensor | None) -> torch.Tensor | None:
    if slots is None or slots.ndim != 3 or slots.shape[1] <= 1:
        return None
    normalized = F.normalize(slots.to(dtype=torch.float32), dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    slot_count = similarity.shape[-1]
    mask = ~torch.eye(slot_count, dtype=torch.bool, device=similarity.device).unsqueeze(0)
    off_diag = similarity.masked_select(mask)
    if off_diag.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=slots.device)
    return torch.mean(off_diag.square())


def _reader_attention_diversity_loss(attention: torch.Tensor | None) -> torch.Tensor | None:
    if attention is None or attention.ndim != 3 or attention.shape[1] <= 1:
        return None
    normalized = F.normalize(attention.to(dtype=torch.float32), dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    query_count = similarity.shape[-1]
    mask = ~torch.eye(query_count, dtype=torch.bool, device=similarity.device).unsqueeze(0)
    off_diag = similarity.masked_select(mask)
    if off_diag.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=attention.device)
    return torch.mean(off_diag)


def _conditioned_query_orthogonality_loss(conditioned_queries: torch.Tensor | None) -> torch.Tensor | None:
    if conditioned_queries is None or conditioned_queries.ndim != 3 or conditioned_queries.shape[1] <= 1:
        return None
    return _slot_diversity_loss(conditioned_queries)


def _reader_short_reconstruction_loss(
    memory_short: torch.Tensor | None,
    reader_readouts: torch.Tensor | None,
) -> torch.Tensor | None:
    if memory_short is None or reader_readouts is None:
        return None
    if tuple(memory_short.shape) != tuple(reader_readouts.shape):
        return None
    return F.smooth_l1_loss(
        memory_short.to(dtype=torch.float32),
        reader_readouts.detach().to(dtype=torch.float32),
    )


def _writer_slot_basis_orthogonality_loss(writer: MemoryWriter | None) -> torch.Tensor | None:
    if writer is None or not hasattr(writer, "slot_embeddings"):
        return None
    slot_embeddings = getattr(writer, "slot_embeddings")
    if slot_embeddings.ndim != 2 or slot_embeddings.shape[0] <= 1:
        return None
    return _slot_diversity_loss(slot_embeddings.unsqueeze(0))


def _writer_slot_energy_balance_loss(memory_long: torch.Tensor | None) -> torch.Tensor | None:
    if memory_long is None:
        return None
    slot_norms = MemoryWriter.slot_norms(memory_long.to(dtype=torch.float32))
    if slot_norms.ndim == 1:
        slot_norms = slot_norms.unsqueeze(0)
    if slot_norms.ndim != 2 or slot_norms.shape[1] <= 1:
        return None
    slot_norm_mean = slot_norms.mean(dim=1).clamp_min(1e-6)
    slot_norm_std = slot_norms.std(dim=1, unbiased=False)
    return torch.mean(slot_norm_std / slot_norm_mean)


def _writer_common_mode_penalty(memory_long: torch.Tensor | None) -> torch.Tensor | None:
    if memory_long is None:
        return None
    slots_fp32 = memory_long.to(dtype=torch.float32)
    if slots_fp32.ndim == 2:
        slots_fp32 = slots_fp32.unsqueeze(0)
    if slots_fp32.ndim != 3 or slots_fp32.shape[1] <= 1:
        return None
    common_mode = MemoryWriter.common_mode_vector(slots_fp32)
    common_energy = common_mode.square().sum(dim=-1) * float(slots_fp32.shape[1])
    total_energy = slots_fp32.square().sum(dim=(1, 2)).clamp_min(1e-6)
    return torch.mean(common_energy / total_energy)


def _reader_fuser_bootstrap_active(*, current_step: int, bootstrap_steps: int) -> bool:
    return int(current_step) <= max(0, int(bootstrap_steps))


def _pairwise_cosine_mean(slots: torch.Tensor | None) -> float:
    if slots is None:
        return 0.0
    if slots.ndim == 2:
        slots = slots.unsqueeze(0)
    if slots.ndim != 3 or slots.shape[1] <= 1:
        return 0.0
    normalized = F.normalize(slots.to(dtype=torch.float32), dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    slot_count = similarity.shape[-1]
    mask = ~torch.eye(slot_count, dtype=torch.bool, device=similarity.device).unsqueeze(0)
    off_diag = similarity.masked_select(mask)
    if off_diag.numel() == 0:
        return 0.0
    return float(torch.mean(off_diag).item())


def _grad_norm(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(dtype=torch.float32)
        total += float(torch.sum(grad * grad).item())
    return float(total ** 0.5)


def _parameters_grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(dtype=torch.float32)
        total += float(torch.sum(grad * grad).item())
    return float(total ** 0.5)


def _trainable_parameter_count(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def _module_trainable(module: nn.Module | None) -> bool:
    if module is None:
        return False
    return any(parameter.requires_grad for parameter in module.parameters())


def _safe_grad_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-8))


def _median_train_event_metric(
    train_events: list[dict[str, Any]],
    *,
    key: str,
    step_start: int,
    step_end: int,
) -> float:
    values = [
        float(event.get(key, 0.0))
        for event in train_events
        if step_start <= int(event.get("step", 0)) <= step_end
    ]
    if not values:
        return 0.0
    values_tensor = torch.tensor(values, dtype=torch.float32)
    return float(values_tensor.median().item())


def _prefix_attention_diagnostic_fields(
    *,
    example_cache: SharedInjectionExampleCache,
    scores: torch.Tensor,
    diagnostics: dict[str, object] | None,
) -> dict[str, Any]:
    if not diagnostics:
        return {
            "diagnostic_layers": [],
            "prefix_attention_mass_mean": 0.0,
            "prefix_to_content_attention_ratio_mean": 0.0,
            "gold_prefix_attention_mass": 0.0,
            "competitor_prefix_attention_mass": 0.0,
            "prefix_attention_mass_mean_by_layer": {},
            "prefix_to_content_attention_ratio_mean_by_layer": {},
            "gold_prefix_attention_mass_by_layer": {},
            "competitor_prefix_attention_mass_by_layer": {},
        }
    masses = [float(value) for value in diagnostics.get("prefix_attention_mass_by_candidate", [])]
    layer_masses_raw = diagnostics.get("prefix_attention_mass_by_candidate_by_layer", {})
    diagnostic_layers = [int(layer_index) for layer_index in diagnostics.get("diagnostic_layers", [])]
    predicted_index = int(torch.argmax(scores).item()) if scores.numel() else 0
    gold_index = int(example_cache.gold_index)
    competitor_candidates = [index for index in range(len(example_cache.candidate_texts)) if index != gold_index]
    competitor_index = (
        max(competitor_candidates, key=lambda index: float(scores[index].item()))
        if competitor_candidates
        else predicted_index
    )
    mean_mass = float(sum(masses) / max(1, len(masses)))
    gold_mass = float(masses[gold_index]) if gold_index < len(masses) else 0.0
    competitor_mass = float(masses[competitor_index]) if competitor_index < len(masses) else 0.0
    mean_by_layer: dict[str, float] = {}
    ratio_by_layer: dict[str, float] = {}
    gold_by_layer: dict[str, float] = {}
    competitor_by_layer: dict[str, float] = {}
    for layer_index in diagnostic_layers:
        layer_values = [float(value) for value in layer_masses_raw.get(layer_index, layer_masses_raw.get(str(layer_index), []))]
        if not layer_values:
            continue
        layer_mean = float(sum(layer_values) / len(layer_values))
        mean_by_layer[str(layer_index)] = layer_mean
        ratio_by_layer[str(layer_index)] = float(layer_mean / max(1e-8, 1.0 - layer_mean))
        if gold_index < len(layer_values):
            gold_by_layer[str(layer_index)] = float(layer_values[gold_index])
        if competitor_index < len(layer_values):
            competitor_by_layer[str(layer_index)] = float(layer_values[competitor_index])
    return {
        "diagnostic_layers": diagnostic_layers,
        "prefix_attention_mass_mean": mean_mass,
        "prefix_to_content_attention_ratio_mean": float(mean_mass / max(1e-8, 1.0 - mean_mass)),
        "gold_prefix_attention_mass": gold_mass,
        "competitor_prefix_attention_mass": competitor_mass,
        "prefix_attention_mass_mean_by_layer": mean_by_layer,
        "prefix_to_content_attention_ratio_mean_by_layer": ratio_by_layer,
        "gold_prefix_attention_mass_by_layer": gold_by_layer,
        "competitor_prefix_attention_mass_by_layer": competitor_by_layer,
    }


def _aggregate_prefix_attention_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostic_rows = [row for row in rows if "prefix_attention_mass_mean" in row]
    if not diagnostic_rows:
        return {
            "prefix_attention_mass_mean": 0.0,
            "prefix_to_content_attention_ratio_mean": 0.0,
            "gold_prefix_attention_mass_mean": 0.0,
            "competitor_prefix_attention_mass_mean": 0.0,
            "prefix_attention_mass_mean_by_layer": {},
            "prefix_to_content_attention_ratio_mean_by_layer": {},
            "gold_prefix_attention_mass_mean_by_layer": {},
            "competitor_prefix_attention_mass_mean_by_layer": {},
            "prefix_attention_nontrivial_layer_count": 0,
        }
    metric_keys = [
        "prefix_attention_mass_mean",
        "prefix_to_content_attention_ratio_mean",
        "gold_prefix_attention_mass",
        "competitor_prefix_attention_mass",
    ]
    aggregated = {
        key: float(
            sum(float(row.get(key, 0.0)) for row in diagnostic_rows) / len(diagnostic_rows)
        )
        for key in metric_keys
    }
    aggregated["gold_prefix_attention_mass_mean"] = aggregated.pop("gold_prefix_attention_mass")
    aggregated["competitor_prefix_attention_mass_mean"] = aggregated.pop("competitor_prefix_attention_mass")
    nested_mappings = {
        "prefix_attention_mass_mean_by_layer": "prefix_attention_mass_mean_by_layer",
        "prefix_to_content_attention_ratio_mean_by_layer": "prefix_to_content_attention_ratio_mean_by_layer",
        "gold_prefix_attention_mass_mean_by_layer": "gold_prefix_attention_mass_by_layer",
        "competitor_prefix_attention_mass_mean_by_layer": "competitor_prefix_attention_mass_by_layer",
    }
    for output_key, row_key in nested_mappings.items():
        layer_keys = sorted({str(layer_key) for row in diagnostic_rows for layer_key in row.get(row_key, {})})
        aggregated[output_key] = {
            layer_key: float(
                sum(float(row.get(row_key, {}).get(layer_key, 0.0)) for row in diagnostic_rows)
                / len(diagnostic_rows)
            )
            for layer_key in layer_keys
        }
    aggregated["prefix_attention_nontrivial_layer_count"] = int(
        sum(
            1
            for value in aggregated["prefix_attention_mass_mean_by_layer"].values()
            if float(value) > 1e-3
        )
    )
    return aggregated


def _prediction_row(
    *,
    arm_alias: str,
    arm: str,
    writer_memory_control: str,
    prompt_variant: str,
    support_serialization_variant: str,
    support_text_block: str,
    example_cache: SharedInjectionExampleCache,
    scores: torch.Tensor,
    prefix_stats: dict[str, Any] | None,
    task_evaluator: Any,
    generated_text: str | None = None,
    no_memory_scores: torch.Tensor | None = None,
    diagnostics: dict[str, object] | None = None,
) -> dict[str, Any]:
    margin, competitor_index = _compute_margin(scores, example_cache.gold_index)
    scalar_prefix_stats = _prefix_scalar_summary(prefix_stats or {})
    answer_logprob_with_memory = float(_answer_logprob(scores, example_cache.gold_index).item())
    answer_logprob_without_memory = (
        float(_answer_logprob(no_memory_scores, example_cache.gold_index).item())
        if no_memory_scores is not None
        else 0.0
    )
    delta_answer_logprob = answer_logprob_with_memory - answer_logprob_without_memory
    if example_cache.task_mode == "generation":
        predicted_text = str(generated_text or "")
        score_payload = task_evaluator.evaluate_prediction({"text": predicted_text}, example_cache.example)
        return {
            "example_id": str(example_cache.example["id"]),
            "arm_alias": arm_alias,
            "arm": arm,
            "writer_memory_control": writer_memory_control,
            "prompt_variant": prompt_variant,
            "support_serialization_variant": support_serialization_variant,
            "support_text_block_chars": len(support_text_block),
            "prefix_tokens": int(scalar_prefix_stats.get("prefix_tokens", 0.0)),
            "prefix_l2": float(scalar_prefix_stats.get("prefix_l2", 0.0)),
            "prompt_text": example_cache.prompt_text,
            "evaluator_type": example_cache.evaluator_type,
            "predicted_label": "",
            "gold_label": str(example_cache.example.get("label", "")),
            "predicted_text": predicted_text,
            "gold_answer": str(example_cache.example.get("gold_answer", example_cache.candidate_texts[0])),
            "predicted_correct": bool(score_payload["correct"]),
            "task_score": float(score_payload["score"]),
            "final_margin": margin,
            "final_choice_scores": [float(value) for value in scores.tolist()],
            "candidate_labels": list(example_cache.candidate_labels),
            "candidate_texts": list(example_cache.candidate_texts),
            "answer_logprob_with_memory": answer_logprob_with_memory,
            "answer_logprob_without_memory": answer_logprob_without_memory,
            "delta_answer_logprob": delta_answer_logprob,
            "normalized_prediction": str(score_payload.get("normalized_prediction", "")),
            "normalized_reference": str(score_payload.get("normalized_reference", "")),
            "extra_metrics": dict(score_payload.get("extra_metrics", {})),
            **_prefix_attention_diagnostic_fields(
                example_cache=example_cache,
                scores=scores,
                diagnostics=diagnostics,
            ),
        }
    predicted_index = int(torch.argmax(scores).item())
    return {
        "example_id": str(example_cache.example["id"]),
        "arm_alias": arm_alias,
        "arm": arm,
        "writer_memory_control": writer_memory_control,
        "prompt_variant": prompt_variant,
        "support_serialization_variant": support_serialization_variant,
        "support_text_block_chars": len(support_text_block),
        "prefix_tokens": int(scalar_prefix_stats.get("prefix_tokens", 0.0)),
        "prefix_l2": float(scalar_prefix_stats.get("prefix_l2", 0.0)),
        "prompt_text": example_cache.prompt_text,
        "evaluator_type": example_cache.evaluator_type,
        "predicted_label": example_cache.candidate_labels[predicted_index],
        "gold_label": example_cache.candidate_labels[example_cache.gold_index],
        "predicted_correct": bool(predicted_index == example_cache.gold_index),
        "task_score": float(predicted_index == example_cache.gold_index),
        "final_margin": margin,
        "competitor_label": example_cache.candidate_labels[competitor_index],
        "final_choice_scores": [float(value) for value in scores.tolist()],
        "candidate_labels": list(example_cache.candidate_labels),
        "candidate_texts": list(example_cache.candidate_texts),
        "answer_logprob_with_memory": answer_logprob_with_memory,
        "answer_logprob_without_memory": answer_logprob_without_memory,
        "delta_answer_logprob": delta_answer_logprob,
        **_prefix_attention_diagnostic_fields(
            example_cache=example_cache,
            scores=scores,
            diagnostics=diagnostics,
        ),
    }


def _evaluate_examples(
    *,
    runtime: SharedInjectionPilotRuntime,
    eval_examples: list[SharedInjectionExampleCache],
    arm_alias: str,
    arm: str,
    writer_memory_control: str,
    prompt_variant: str,
    support_serialization_variant: str,
    support_text_block: str,
    teacher_support_text_block: str,
    prefix_embeddings: torch.Tensor | None,
    layer_prefix_hidden_by_layer: dict[int, torch.Tensor] | None,
    prefix_stats: dict[str, Any] | None,
    task_evaluator: Any,
    support_rows: list[dict[str, Any]] | None = None,
    profiler: ProfileTracker | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_support_text_block = teacher_support_text_block if arm == "teacher_text" else support_text_block
    case_rows: list[dict[str, Any]] = []
    example_prefix_stats: list[dict[str, Any]] = []
    for example_cache in eval_examples:
        if profiler is not None:
            profiler.add_example()
            profiler.add_tokens(runtime.backbone.count_tokens(example_cache.prompt_text))
            for candidate_text in example_cache.candidate_texts:
                profiler.add_tokens(runtime.backbone.count_tokens(candidate_text))
        active_prefix_embeddings = prefix_embeddings
        active_layer_prefix_hidden_by_layer = layer_prefix_hidden_by_layer
        active_prefix_stats = prefix_stats
        if arm == "injected" and (
            runtime.memory_path_variant == "two_level" or runtime.bridge_mode == "writer_direct"
        ):
            prefix_artifacts = runtime.build_prefix_artifacts(
                support_text_block,
                support_rows=support_rows,
                prompt_text=example_cache.prompt_text,
            )
            active_prefix_embeddings = prefix_artifacts.prefix_embeddings
            active_layer_prefix_hidden_by_layer = prefix_artifacts.layer_prefix_hidden_by_layer
            active_prefix_stats = prefix_artifacts.prefix_stats
            example_prefix_stats.append(active_prefix_stats)
        score_output = runtime.score_example(
            example_cache,
            support_text_block=teacher_support_text_block,
            prefix_embeddings=active_prefix_embeddings,
            layer_prefix_hidden_by_layer=active_layer_prefix_hidden_by_layer,
            return_diagnostics=bool(
                arm == "injected"
                and runtime.injection_mode == "sparse_deep_prefix"
                and active_layer_prefix_hidden_by_layer is not None
            ),
        )
        diagnostics = None
        if isinstance(score_output, tuple):
            scores_tensor, diagnostics = score_output
        else:
            scores_tensor = score_output
        scores = scores_tensor.detach().to(dtype=torch.float32).cpu()
        no_memory_scores = None
        if arm == "injected":
            no_memory_scores = runtime.backbone.score_continuations(
                example_cache.prompt_text,
                example_cache.candidate_texts,
            ).detach().to(dtype=torch.float32).cpu()
        generated_text = None
        if example_cache.task_mode == "generation":
            generated_text = runtime.generate_text(
                prompt_text=example_cache.prompt_text,
                support_text_block=teacher_support_text_block,
                prefix_embeddings=active_prefix_embeddings,
                layer_prefix_hidden_by_layer=active_layer_prefix_hidden_by_layer,
            )
            if profiler is not None and generated_text:
                profiler.add_tokens(runtime.backbone.count_tokens(generated_text))
        case_rows.append(
            _prediction_row(
                arm_alias=arm_alias,
                arm=arm,
                writer_memory_control=writer_memory_control,
                prompt_variant=prompt_variant,
                support_serialization_variant=support_serialization_variant,
                support_text_block=active_support_text_block,
                example_cache=example_cache,
                scores=scores,
                prefix_stats=active_prefix_stats,
                task_evaluator=task_evaluator,
                generated_text=generated_text,
                no_memory_scores=no_memory_scores,
                diagnostics=diagnostics,
            )
        )
    return case_rows, (_aggregate_prefix_stats(example_prefix_stats) if example_prefix_stats else (prefix_stats or _prefix_stats()))


def run_shared_injection_pilot(
    *,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    resume: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    arm = _resolve_shared_injection_arm(config)
    writer_memory_control = _resolve_writer_memory_control(config)
    arm_alias = _resolve_pilot_arm_alias(config, arm=arm, writer_memory_control=writer_memory_control)
    prompt_variant = _resolve_prompt_variant(config)
    support_serialization_variant = _resolve_support_serialization_variant(config)
    bridge_mode = _resolve_bridge_mode(config)
    memory_path_variant = _resolve_memory_path_variant(config)
    injection_mode = _resolve_injection_mode(config)
    reader_context_mode = _resolve_reader_context_mode(config)
    reader_num_queries = _resolve_reader_num_queries(config)
    fuser_short_slots = _resolve_fuser_short_slots(config)
    projector_token_source = _resolve_projector_token_source(config)
    support_encoder_mode = _resolve_support_encoder_mode(config)
    trainable_variant = _resolve_trainable_variant(config)
    writer_adapter_enabled = _resolve_writer_adapter_enabled(config)
    alignment_aux_mode = _resolve_alignment_aux_mode(config)
    alignment_aux_weight_max = _resolve_alignment_aux_weight_max(config)
    alignment_aux_start_step = _resolve_alignment_aux_start_step(config)
    alignment_aux_ramp_steps = _resolve_alignment_aux_ramp_steps(config)
    alignment_aux_apply_only_to_real_memory = _resolve_alignment_aux_apply_only_to_real_memory(config)
    alignment_aux_temperature = _resolve_alignment_aux_temperature(config)
    alignment_aux_advantage_center = _resolve_alignment_aux_advantage_center(config)
    alignment_aux_advantage_scale = _resolve_alignment_aux_advantage_scale(config)
    init_checkpoint_path = _resolve_init_checkpoint_path(config)
    deep_prefix_layers = _resolve_deep_prefix_layers(config)
    deep_prefix_rank = _resolve_deep_prefix_rank(config)
    train_support_mode = _resolve_train_support_mode(config)
    choice_ce_weight = _resolve_choice_ce_weight(config)
    competitor_hinge_weight_max = _resolve_competitor_hinge_weight_max(config)
    competitor_hinge_start_step = _resolve_competitor_hinge_start_step(config)
    competitor_hinge_ramp_steps = _resolve_competitor_hinge_ramp_steps(config)
    latent_anchor_weight_start = _resolve_latent_anchor_weight_start(config)
    latent_anchor_weight_end = _resolve_latent_anchor_weight_end(config)
    latent_anchor_decay_steps = _resolve_latent_anchor_decay_steps(config)
    memory_long_diversity_weight = _resolve_memory_long_diversity_weight(config)
    memory_short_diversity_weight = _resolve_memory_short_diversity_weight(config)
    reader_attention_diversity_weight = _resolve_reader_attention_diversity_weight(config)
    reader_conditioned_query_orthogonality_weight = (
        _resolve_reader_conditioned_query_orthogonality_weight(config)
    )
    reader_short_reconstruction_weight = _resolve_reader_short_reconstruction_weight(config)
    reader_fuser_bootstrap_steps = _resolve_reader_fuser_bootstrap_steps(config)
    writer_slot_basis_orthogonality_weight = _resolve_writer_slot_basis_orthogonality_weight(config)
    writer_slot_energy_balance_weight = _resolve_writer_slot_energy_balance_weight(config)
    writer_common_mode_penalty_weight = _resolve_writer_common_mode_penalty_weight(config)
    writer_gain_margin = _resolve_writer_gain_margin(config)
    writer_gain_margin_weight = _resolve_writer_gain_margin_weight(config)
    writer_covariance_diversity_weight = _resolve_writer_covariance_diversity_weight(config)
    orthogonalize_writer_slot_basis = _resolve_writer_orthogonalize_slot_basis(config)
    support_dataset_path = str(config["task"]["support_dataset_path"])
    train_dataset_path = str(config["task"].get("train_dataset_path", support_dataset_path))
    train_support_dataset_path = str(config["task"].get("train_support_dataset_path", support_dataset_path))
    train_support_episode_bank_path = str(config["task"].get("train_support_episode_bank_path", "")).strip()
    support_limit = max(0, int(config["runtime"].get("pilot_support_examples", 8)))
    train_steps = int(config["runtime"].get("pilot_train_steps", 96))
    projector_warmup_steps = int(config["runtime"].get("pilot_projector_warmup_steps", 32))
    writer_learning_rate = float(config["runtime"].get("pilot_writer_learning_rate", 1e-4))
    writer_adapter_learning_rate = _resolve_writer_adapter_learning_rate(config)
    projector_learning_rate = float(config["runtime"].get("pilot_projector_learning_rate", 2e-3))
    source_stub_learning_rate = _resolve_source_stub_learning_rate(config)
    receiver_lora_learning_rate = _resolve_receiver_lora_learning_rate(config)
    writer_weight_decay = float(config["runtime"].get("pilot_writer_weight_decay", 0.0))
    writer_adapter_weight_decay = _resolve_writer_adapter_weight_decay(config)
    projector_weight_decay = float(config["runtime"].get("pilot_projector_weight_decay", 0.0))
    source_stub_weight_decay = _resolve_source_stub_weight_decay(config)
    receiver_lora_weight_decay = _resolve_receiver_lora_weight_decay(config)
    gradient_clip_norm = float(config["runtime"].get("pilot_gradient_clip_norm", 0.0))
    choice_margin = float(config["runtime"].get("stage_c_choice_margin", 0.1))
    injection_checkpoint_path = config["runtime"].get("pilot_checkpoint_path")
    if init_checkpoint_path and injection_checkpoint_path:
        raise ValueError(
            "runtime.pilot_init_checkpoint_path and runtime.pilot_checkpoint_path are mutually exclusive."
        )
    if bridge_mode == "writer_direct" and injection_mode != "sparse_deep_prefix" and init_checkpoint_path:
        raise ValueError("runtime.pilot_bridge_mode=writer_direct forbids pilot_init_checkpoint_path outside sparse deep-prefix runs.")
    if bridge_mode == "writer_direct" and injection_mode != "sparse_deep_prefix" and injection_checkpoint_path:
        raise ValueError("runtime.pilot_bridge_mode=writer_direct forbids pilot_checkpoint_path outside sparse deep-prefix runs.")
    if trainable_variant == "writer_adapter_only" and not writer_adapter_enabled:
        raise ValueError(
            "runtime.pilot_trainable_variant=writer_adapter_only requires method.writer_adapter.enabled=true."
        )
    if arm in {"base_only", "teacher_text"} or writer_memory_control == "zero":
        train_steps = 0
        projector_warmup_steps = 0
    if injection_checkpoint_path:
        train_steps = 0
        projector_warmup_steps = 0
    if dry_run:
        train_steps = min(train_steps, 2)
        projector_warmup_steps = min(projector_warmup_steps, train_steps)

    task_evaluator = build_task_evaluator(config)
    support_examples = _load_task_dataset_with_path(config, support_dataset_path)
    train_examples = _load_task_dataset_with_path(config, train_dataset_path)
    eval_examples = load_task_dataset(config)
    train_support_examples: list[dict[str, Any]] = []
    train_support_episodes: list[SupportEpisode] = []
    if train_support_mode == "episode_bank":
        if not train_support_episode_bank_path:
            raise ValueError(
                "runtime.pilot_train_support_mode=episode_bank requires task.train_support_episode_bank_path."
            )
        train_support_episodes = _load_support_episode_bank(train_support_episode_bank_path)
        if dry_run:
            train_support_episodes = train_support_episodes[: min(4, len(train_support_episodes))]
    else:
        train_support_examples = _load_task_dataset_with_path(config, train_support_dataset_path)
    if support_limit:
        support_examples = support_examples[: min(support_limit, len(support_examples))]
        if train_support_examples:
            train_support_examples = train_support_examples[: min(support_limit, len(train_support_examples))]
    if dry_run:
        train_examples = train_examples[: min(24, len(train_examples))]
        eval_examples = eval_examples[: min(12, len(eval_examples))]

    eval_caches = _build_example_caches(eval_examples, prompt_variant=prompt_variant)
    support_caches = _build_example_caches(support_examples, prompt_variant=prompt_variant)
    train_caches = _build_example_caches(train_examples, prompt_variant=prompt_variant)
    default_prefix_prompt_text = (
        eval_caches[0].prompt_text
        if eval_caches
        else (train_caches[0].prompt_text if train_caches else None)
    )
    lookup_paths = _resolve_support_lookup_dataset_paths(
        config,
        default_paths=[
            support_dataset_path,
            train_dataset_path,
            str(config["task"]["dataset_path"]),
            train_support_dataset_path,
        ],
    )
    lookup_examples: list[dict[str, Any]] = []
    for lookup_path in lookup_paths:
        if train_support_mode == "episode_bank" and lookup_path == train_support_episode_bank_path:
            continue
        lookup_examples.extend(_load_task_dataset_with_path(config, lookup_path))
    if train_support_episodes:
        for episode in train_support_episodes:
            lookup_examples.extend(episode.support_rows)
    example_lookup = _merge_example_lookup(
        [cache.example for cache in support_caches],
        [cache.example for cache in train_caches],
        [cache.example for cache in eval_caches],
        train_support_examples,
        lookup_examples,
    )
    support_text_block = _build_support_text_block(
        [cache.example for cache in support_caches],
        memory_control=writer_memory_control if arm == "injected" else "real",
        example_lookup=example_lookup,
        support_serialization_variant=support_serialization_variant,
    )
    support_rows_for_prefix = _resolve_support_rows_for_memory_control(
        [cache.example for cache in support_caches],
        memory_control=writer_memory_control if arm == "injected" else "real",
        example_lookup=example_lookup,
        support_serialization_variant=support_serialization_variant,
    )
    teacher_support_text_block = _build_support_text_block(
        [cache.example for cache in support_caches],
        memory_control="real",
        example_lookup=example_lookup,
        support_serialization_variant=support_serialization_variant,
    )

    runtime = SharedInjectionPilotRuntime(
        config=config,
        seed=seed,
        arm=arm,
        writer_memory_control=writer_memory_control,
    )
    if arm == "injected":
        if bridge_mode != "writer_direct":
            runtime.load_writer(resume)
        if init_checkpoint_path:
            runtime.warm_start_from_injection_checkpoint(init_checkpoint_path)
        if injection_checkpoint_path:
            runtime.load_injection_checkpoint(injection_checkpoint_path)
        if orthogonalize_writer_slot_basis:
            runtime.orthogonalize_writer_slot_basis()
    reference_support_encoder: nn.Module | None = None
    reference_writer: nn.Module | None = None
    latent_anchor_enabled = bool(
        arm == "injected"
        and writer_memory_control != "zero"
        and bridge_mode != "writer_direct"
        and (
            abs(float(latent_anchor_weight_start)) > 0.0
            or abs(float(latent_anchor_weight_end)) > 0.0
        )
    )
    if latent_anchor_enabled:
        reference_writer = copy.deepcopy(runtime.writer).eval()
        for parameter in reference_writer.parameters():
            parameter.requires_grad_(False)
        if runtime.support_encoder is not None:
            reference_support_encoder = copy.deepcopy(runtime.support_encoder).eval()
            for parameter in reference_support_encoder.parameters():
                parameter.requires_grad_(False)

    profiler = ProfileTracker(
        output_dir=output_dir,
        device=str(config["runtime"].get("device", "cpu")),
        event_name="train",
    )
    train_events: list[dict[str, Any]] = []
    snapshot_steps = _resolve_snapshot_steps(
        config,
        train_steps=train_steps,
        warmup_steps=projector_warmup_steps,
    )
    snapshot_metrics: list[dict[str, Any]] = []
    support_mask_generator = torch.Generator(device="cpu")
    support_mask_generator.manual_seed(seed + 101)
    train_support_mask_count = _resolve_train_support_mask_count(
        config,
        support_serialization_variant=support_serialization_variant,
    )

    def evaluate_snapshot(step: int) -> None:
        prefix_artifacts = PrefixInjectionArtifacts(
            prefix_embeddings=None,
            layer_prefix_hidden_by_layer=None,
            prefix_stats=_prefix_stats(),
        )
        snapshot_checkpoint_path = None
        if arm == "injected":
            prefix_artifacts = runtime.build_prefix_artifacts(
                support_text_block,
                support_rows=support_rows_for_prefix,
                prompt_text=default_prefix_prompt_text if bridge_mode == "writer_direct" else None,
            )
            if writer_memory_control != "zero":
                snapshot_checkpoint_path = (
                    output_dir / "snapshot_evals" / f"step_{step:04d}" / "checkpoint.pt"
                )
                _save_shared_injection_checkpoint(
                    runtime=runtime,
                    output_path=snapshot_checkpoint_path,
                    seed=seed,
                    arm_alias=arm_alias,
                    arm=arm,
                    writer_memory_control=writer_memory_control,
                    support_dataset_path=support_dataset_path,
                    support_text_block=support_text_block,
                    prompt_variant=prompt_variant,
                    support_serialization_variant=support_serialization_variant,
                    step=step,
                )
        snapshot_rows, snapshot_prefix_stats = _evaluate_examples(
            runtime=runtime,
            eval_examples=eval_caches,
            arm_alias=arm_alias,
            arm=arm,
            writer_memory_control=writer_memory_control,
            prompt_variant=prompt_variant,
            support_serialization_variant=support_serialization_variant,
            support_text_block=support_text_block,
            teacher_support_text_block=teacher_support_text_block,
            prefix_embeddings=prefix_artifacts.prefix_embeddings,
            layer_prefix_hidden_by_layer=prefix_artifacts.layer_prefix_hidden_by_layer,
            prefix_stats=prefix_artifacts.prefix_stats,
            task_evaluator=task_evaluator,
            support_rows=support_rows_for_prefix,
            profiler=None,
        )
        metrics = _classification_metrics_from_rows(snapshot_rows)
        snapshot_dir = output_dir / "snapshot_evals" / f"step_{step:04d}"
        snapshot_payload = _write_snapshot_eval(
            snapshot_dir=snapshot_dir,
            step=step,
            arm_alias=arm_alias,
            arm=arm,
            writer_memory_control=writer_memory_control,
            prompt_variant=prompt_variant,
            support_serialization_variant=support_serialization_variant,
            support_text_block=support_text_block,
            case_rows=snapshot_rows,
            prefix_stats=snapshot_prefix_stats,
            checkpoint_path=snapshot_checkpoint_path,
        )
        snapshot_metrics.append(
            {
                "step": step,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mean_margin": metrics["mean_margin"],
                "dominant_label_fraction": metrics["dominant_label_fraction"],
                "snapshot_dir": str(snapshot_dir.resolve()),
                "task_case_dump_path": snapshot_payload["task_case_dump_path"],
                "checkpoint_path": snapshot_payload["checkpoint_path"],
                "prefix_tokens": snapshot_payload["prefix_tokens"],
                "prefix_l2": snapshot_payload["prefix_l2"],
                "prefix_slot_norm_mean": snapshot_payload["prefix_slot_norm_mean"],
                "prefix_slot_norm_std": snapshot_payload["prefix_slot_norm_std"],
                "prefix_slot_norm_max": snapshot_payload["prefix_slot_norm_max"],
                "writer_memory_l2": snapshot_payload["writer_memory_l2"],
                "writer_slot_norm_mean": snapshot_payload["writer_slot_norm_mean"],
                "writer_slot_norm_std": snapshot_payload["writer_slot_norm_std"],
                "writer_slot_norm_max": snapshot_payload["writer_slot_norm_max"],
                "support_item_count": snapshot_payload["support_item_count"],
                "support_item_hidden_l2": snapshot_payload["support_item_hidden_l2"],
                "support_item_hidden_norm_mean": snapshot_payload["support_item_hidden_norm_mean"],
                "support_item_hidden_norm_std": snapshot_payload["support_item_hidden_norm_std"],
                "support_item_hidden_norm_max": snapshot_payload["support_item_hidden_norm_max"],
                "prefix_artifact_stats": snapshot_payload["prefix_artifact_stats"],
                **_prefix_scalar_summary(snapshot_payload),
            }
        )

    if 0 in snapshot_steps:
        evaluate_snapshot(0)

    if arm == "injected" and writer_memory_control != "zero" and train_steps > 0:
        runtime.writer.train()
        if runtime.source_stub is not None:
            runtime.source_stub.train()
        if runtime.support_encoder is not None:
            runtime.support_encoder.train()
        if runtime.reader is not None:
            runtime.reader.train()
        if runtime.fuser is not None:
            runtime.fuser.train()
        runtime.prefix_projector.train()
        runtime.set_writer_trainable(False)
        runtime.set_source_stub_trainable(False)
        runtime.set_support_encoder_trainable(False)
        writer_base_parameters = runtime.writer_base_parameters()
        writer_adapter_parameters = runtime.writer_adapter_parameters()
        source_stub_parameters = runtime.source_stub_parameters()
        optimizer_groups = [
            {
                "params": list(runtime.prefix_projector.parameters()),
                "lr": projector_learning_rate,
                "weight_decay": projector_weight_decay,
            },
        ]
        if source_stub_parameters:
            optimizer_groups.append(
                {
                    "params": source_stub_parameters,
                    "lr": source_stub_learning_rate,
                    "weight_decay": source_stub_weight_decay,
                }
            )
        if writer_base_parameters:
            optimizer_groups.append(
                {
                    "params": writer_base_parameters,
                    "lr": writer_learning_rate,
                    "weight_decay": writer_weight_decay,
                }
            )
        if writer_adapter_parameters:
            optimizer_groups.append(
                {
                    "params": writer_adapter_parameters,
                    "lr": writer_adapter_learning_rate,
                    "weight_decay": writer_adapter_weight_decay,
                }
            )
        if runtime.support_encoder is not None:
            optimizer_groups.append(
                {
                    "params": list(runtime.support_encoder.parameters()),
                    "lr": writer_learning_rate,
                    "weight_decay": writer_weight_decay,
                }
            )
        if runtime.reader is not None:
            optimizer_groups.append(
                {
                    "params": list(runtime.reader.parameters()),
                    "lr": writer_learning_rate,
                    "weight_decay": writer_weight_decay,
                }
            )
        if runtime.fuser is not None:
            optimizer_groups.append(
                {
                    "params": list(runtime.fuser.parameters()),
                    "lr": writer_learning_rate,
                    "weight_decay": writer_weight_decay,
                }
            )
        receiver_lora_parameters = list(
            getattr(runtime.backbone, "receiver_lora_parameters", lambda: [])()
        )
        if receiver_lora_parameters:
            optimizer_groups.append(
                {
                    "params": receiver_lora_parameters,
                    "lr": receiver_lora_learning_rate,
                    "weight_decay": receiver_lora_weight_decay,
                }
            )
        optimizer = torch.optim.AdamW(
            optimizer_groups
        )
        for step in range(train_steps):
            bootstrap_active = _reader_fuser_bootstrap_active(
                current_step=step + 1,
                bootstrap_steps=reader_fuser_bootstrap_steps,
            )
            writer_adapter_frozen = bool(
                trainable_variant == "projector_only"
                or step < projector_warmup_steps
                or bootstrap_active
            )
            writer_base_frozen = bool(
                writer_adapter_frozen
                or (
                    runtime.writer_adapter_enabled
                    and trainable_variant == "writer_adapter_only"
                )
            )
            if runtime.source_stub is not None:
                writer_base_frozen = True
                writer_adapter_frozen = True
            projector_frozen = bool(bootstrap_active)
            runtime.set_writer_base_trainable(not writer_base_frozen)
            runtime.set_writer_adapter_trainable(not writer_adapter_frozen)
            runtime.set_source_stub_trainable(not projector_frozen)
            runtime.set_support_encoder_trainable(not writer_base_frozen)
            runtime.set_prefix_projector_trainable(not projector_frozen)
            train_example = train_caches[step % len(train_caches)]
            selected_episode_id = ""
            selected_episode_source_split = ""
            selected_episode_label_counts: dict[str, int] = {}
            train_support_source_rows = [cache.example for cache in support_caches]
            if train_support_mode == "episode_bank":
                if not train_support_episodes:
                    raise ValueError("Episode-bank training requested but no episodes were loaded.")
                episode_index = int(
                    torch.randint(
                        len(train_support_episodes),
                        (1,),
                        generator=support_mask_generator,
                    ).item()
                )
                selected_episode = train_support_episodes[episode_index]
                selected_episode_id = selected_episode.episode_id
                selected_episode_source_split = selected_episode.source_split
                selected_episode_label_counts = dict(selected_episode.label_counts)
                train_support_source_rows = selected_episode.support_rows
            elif train_support_examples:
                train_support_source_rows = train_support_examples
            masked_support_rows = _sample_support_examples_for_training(
                support_examples=train_support_source_rows,
                support_serialization_variant=support_serialization_variant,
                generator=support_mask_generator,
                target_count=train_support_mask_count,
            )
            masked_support_text_block = _build_support_text_block(
                masked_support_rows,
                memory_control=writer_memory_control,
                example_lookup=example_lookup,
                support_serialization_variant=support_serialization_variant,
                preselected_rows=True,
            )
            masked_support_rows_for_prefix = _resolve_support_rows_for_memory_control(
                masked_support_rows,
                memory_control=writer_memory_control,
                example_lookup=example_lookup,
                support_serialization_variant=support_serialization_variant,
                preselected_rows=True,
            )
            optimizer.zero_grad(set_to_none=True)
            prefix_artifacts = runtime.build_prefix_artifacts(
                masked_support_text_block,
                support_rows=masked_support_rows_for_prefix,
                prompt_text=train_example.prompt_text,
            )
            memory_slot_effective_rank = _effective_rank(prefix_artifacts.memory_long)
            memory_short_effective_rank = _effective_rank(prefix_artifacts.memory_short)
            support_state_effective_rank = _effective_rank(prefix_artifacts.support_item_states)
            score_output = runtime.score_example(
                train_example,
                support_text_block=teacher_support_text_block,
                prefix_embeddings=prefix_artifacts.prefix_embeddings,
                layer_prefix_hidden_by_layer=prefix_artifacts.layer_prefix_hidden_by_layer,
                return_diagnostics=bool(
                    runtime.injection_mode == "sparse_deep_prefix"
                    and prefix_artifacts.layer_prefix_hidden_by_layer is not None
                ),
            )
            train_diagnostics = None
            if isinstance(score_output, tuple):
                scores, train_diagnostics = score_output
            else:
                scores = score_output
            with torch.no_grad():
                no_memory_scores = runtime.backbone.score_continuations(
                    train_example.prompt_text,
                    train_example.candidate_texts,
                )
            active_competitor_hinge_weight = _active_competitor_hinge_weight(
                current_step=step + 1,
                max_weight=competitor_hinge_weight_max,
                start_step=competitor_hinge_start_step,
                ramp_steps=competitor_hinge_ramp_steps,
            )
            loss, choice_ce_loss, competitor_hinge_loss = _task_training_loss(
                scores,
                train_example,
                margin_value=choice_margin,
                ce_weight=choice_ce_weight,
                competitor_hinge_weight=active_competitor_hinge_weight,
            )
            writer_gain_margin_loss = None
            delta_answer_logprob_tensor = _answer_logprob(scores, train_example.gold_index) - _answer_logprob(
                no_memory_scores,
                train_example.gold_index,
            )
            if writer_gain_margin_weight > 0.0:
                writer_gain_margin_loss, delta_answer_logprob_tensor = _writer_gain_margin_loss(
                    with_memory_scores=scores,
                    without_memory_scores=no_memory_scores,
                    gold_index=train_example.gold_index,
                    margin_value=writer_gain_margin,
                )
                loss = loss + (writer_gain_margin_weight * writer_gain_margin_loss)
            active_latent_anchor_weight = _scheduled_linear_decay_weight(
                current_step=step + 1,
                start_weight=latent_anchor_weight_start,
                end_weight=latent_anchor_weight_end,
                decay_steps=latent_anchor_decay_steps,
            )
            latent_anchor_loss = None
            support_anchor_loss = None
            writer_anchor_loss = None
            anchor_support_cosine = None
            anchor_writer_slot_cosine = None
            if (
                active_latent_anchor_weight > 0.0
                and reference_writer is not None
            ):
                with torch.no_grad():
                    reference_support_states, reference_memory_slots = _reference_latent_targets(
                        runtime=runtime,
                        reference_support_encoder=reference_support_encoder,
                        reference_writer=reference_writer,
                        support_text_block=masked_support_text_block,
                        support_rows=masked_support_rows_for_prefix,
                    )
                (
                    latent_anchor_loss,
                    support_anchor_loss,
                    writer_anchor_loss,
                    anchor_support_cosine,
                    anchor_writer_slot_cosine,
                ) = _latent_anchor_loss(
                    current_support_states=prefix_artifacts.support_item_states,
                    reference_support_states=reference_support_states,
                    current_memory_slots=prefix_artifacts.memory_slots,
                    reference_memory_slots=reference_memory_slots,
                )
                loss = loss + (active_latent_anchor_weight * latent_anchor_loss)
            memory_long_diversity_loss = _slot_diversity_loss(prefix_artifacts.memory_long)
            memory_short_diversity_loss = _slot_diversity_loss(prefix_artifacts.memory_short)
            reader_attention_diversity_loss = _reader_attention_diversity_loss(
                prefix_artifacts.reader_attention
            )
            reader_conditioned_query_orthogonality_loss = None
            if runtime.reader is not None and runtime.reader.conditioning_mode != "none":
                reader_conditioned_query_orthogonality_loss = _conditioned_query_orthogonality_loss(
                    prefix_artifacts.reader_conditioned_queries
                )
            reader_short_reconstruction_loss = _reader_short_reconstruction_loss(
                prefix_artifacts.memory_short,
                prefix_artifacts.reader_readouts,
            )
            writer_slot_basis_orthogonality_loss = _writer_slot_basis_orthogonality_loss(runtime.writer)
            writer_slot_energy_balance_loss = _writer_slot_energy_balance_loss(prefix_artifacts.memory_long)
            writer_common_mode_penalty_loss = _writer_common_mode_penalty(prefix_artifacts.memory_long)
            writer_covariance_diversity_loss = _writer_covariance_diversity_loss(
                prefix_artifacts.memory_long
            )
            if memory_long_diversity_weight > 0.0 and memory_long_diversity_loss is not None:
                loss = loss + (memory_long_diversity_weight * memory_long_diversity_loss)
            if memory_short_diversity_weight > 0.0 and memory_short_diversity_loss is not None:
                loss = loss + (memory_short_diversity_weight * memory_short_diversity_loss)
            if (
                reader_attention_diversity_weight > 0.0
                and reader_attention_diversity_loss is not None
            ):
                loss = loss + (reader_attention_diversity_weight * reader_attention_diversity_loss)
            if (
                reader_conditioned_query_orthogonality_weight > 0.0
                and reader_conditioned_query_orthogonality_loss is not None
            ):
                loss = loss + (
                    reader_conditioned_query_orthogonality_weight
                    * reader_conditioned_query_orthogonality_loss
                )
            if (
                reader_short_reconstruction_weight > 0.0
                and reader_short_reconstruction_loss is not None
            ):
                loss = loss + (
                    reader_short_reconstruction_weight * reader_short_reconstruction_loss
                )
            if (
                writer_slot_basis_orthogonality_weight > 0.0
                and writer_slot_basis_orthogonality_loss is not None
            ):
                loss = loss + (
                    writer_slot_basis_orthogonality_weight * writer_slot_basis_orthogonality_loss
                )
            if (
                writer_slot_energy_balance_weight > 0.0
                and writer_slot_energy_balance_loss is not None
            ):
                loss = loss + (
                    writer_slot_energy_balance_weight * writer_slot_energy_balance_loss
                )
            if (
                writer_common_mode_penalty_weight > 0.0
                and writer_common_mode_penalty_loss is not None
            ):
                loss = loss + (
                    writer_common_mode_penalty_weight * writer_common_mode_penalty_loss
                )
            if (
                writer_covariance_diversity_weight > 0.0
                and writer_covariance_diversity_loss is not None
            ):
                loss = loss + (
                    writer_covariance_diversity_weight * writer_covariance_diversity_loss
                )
            active_alignment_aux_weight = _active_competitor_hinge_weight(
                current_step=step + 1,
                max_weight=alignment_aux_weight_max,
                start_step=alignment_aux_start_step,
                ramp_steps=alignment_aux_ramp_steps,
            )
            alignment_aux_loss = None
            alignment_aux_active = False
            alignment_aux_diagnostics = {
                "teacher_choice_kl": 0.0,
                "teacher_choice_js": 0.0,
                "teacher_advantage_weight_mean": 0.0,
                "teacher_advantage_weight_max": 0.0,
                "teacher_margin_minus_base_margin": 0.0,
                "teacher_margin_minus_active_margin": 0.0,
                "active_class_entropy": 0.0,
                "teacher_class_entropy": 0.0,
                "base_class_entropy": 0.0,
            }
            if train_example.task_mode == "candidate_selection":
                with torch.no_grad():
                    base_scores = runtime.backbone.score_continuations(
                        train_example.prompt_text,
                        train_example.candidate_texts,
                    )
                    teacher_scores = runtime.backbone.score_continuations(
                        _serialize_teacher_prompt(train_example.prompt_text, teacher_support_text_block),
                        train_example.candidate_texts,
                    )
                alignment_aux_allowed = not (
                    alignment_aux_apply_only_to_real_memory and writer_memory_control != "real"
                )
                requested_alignment_mode = alignment_aux_mode if alignment_aux_allowed else "off"
                alignment_aux_loss, alignment_aux_active, alignment_aux_diagnostics = _alignment_aux_loss(
                    mode=requested_alignment_mode,
                    active_scores=scores,
                    base_scores=base_scores,
                    teacher_scores=teacher_scores,
                    gold_index=train_example.gold_index,
                    temperature=alignment_aux_temperature,
                    advantage_center=alignment_aux_advantage_center,
                    advantage_scale=alignment_aux_advantage_scale,
                )
                if (
                    alignment_aux_mode != "off"
                    and alignment_aux_allowed
                    and active_alignment_aux_weight > 0.0
                    and alignment_aux_loss is not None
                ):
                    loss = loss + (active_alignment_aux_weight * alignment_aux_loss)
            loss.backward()
            source_stub_grad_norm = _grad_norm(runtime.source_stub)
            support_encoder_grad_norm = _grad_norm(runtime.support_encoder)
            prefix_projector_grad_norm = _grad_norm(runtime.prefix_projector)
            writer_grad_norm = _grad_norm(runtime.writer)
            writer_adapter_grad_norm = _parameters_grad_norm(runtime.writer_adapter_parameters())
            reader_grad_norm = _grad_norm(runtime.reader)
            fuser_grad_norm = _grad_norm(runtime.fuser)
            receiver_lora_grad_norm = _parameters_grad_norm(
                list(getattr(runtime.backbone, "receiver_lora_parameters", lambda: [])())
            )
            source_to_projector_grad_ratio = _safe_grad_ratio(
                source_stub_grad_norm,
                prefix_projector_grad_norm,
            )
            writer_to_projector_grad_ratio = _safe_grad_ratio(
                writer_grad_norm + support_encoder_grad_norm,
                prefix_projector_grad_norm,
            )
            reader_to_support_grad_ratio = _safe_grad_ratio(
                reader_grad_norm,
                support_encoder_grad_norm,
            )
            fuser_to_support_grad_ratio = _safe_grad_ratio(
                fuser_grad_norm,
                support_encoder_grad_norm,
            )
            receiver_lora_to_reader_grad_ratio = _safe_grad_ratio(
                receiver_lora_grad_norm,
                reader_grad_norm,
            )
            total_grad_norm = 0.0
            if gradient_clip_norm > 0.0:
                parameters = [
                    parameter
                    for parameter in runtime.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if parameters:
                    total_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm).item()
                    )
            optimizer.step()
            profiler.add_example()
            profiler.add_tokens(runtime.backbone.count_tokens(train_example.prompt_text))
            profiler.add_tokens(runtime.backbone.count_tokens(masked_support_text_block))
            for candidate_text in train_example.candidate_texts:
                profiler.add_tokens(runtime.backbone.count_tokens(candidate_text))
            train_prefix_attention_metrics = _prefix_attention_diagnostic_fields(
                example_cache=train_example,
                scores=scores.detach().to(dtype=torch.float32).cpu(),
                diagnostics=train_diagnostics,
            )
            train_events.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "choice_ce_loss": float(choice_ce_loss.item()),
                    "competitor_hinge_loss": float(competitor_hinge_loss.item()),
                    "active_competitor_hinge_weight": float(active_competitor_hinge_weight),
                    "answer_logprob_with_memory": float(
                        _answer_logprob(scores, train_example.gold_index).item()
                    ),
                    "answer_logprob_without_memory": float(
                        _answer_logprob(no_memory_scores, train_example.gold_index).item()
                    ),
                    "delta_answer_logprob": float(delta_answer_logprob_tensor.item()),
                    "writer_gain_margin_loss": (
                        0.0
                        if writer_gain_margin_loss is None
                        else float(writer_gain_margin_loss.item())
                    ),
                    "writer_gain_margin": float(writer_gain_margin),
                    "writer_gain_margin_weight": float(writer_gain_margin_weight),
                    "latent_anchor_loss": (
                        0.0 if latent_anchor_loss is None else float(latent_anchor_loss.item())
                    ),
                    "latent_anchor_weight": float(active_latent_anchor_weight),
                    "support_anchor_loss": (
                        0.0 if support_anchor_loss is None else float(support_anchor_loss.item())
                    ),
                    "writer_anchor_loss": (
                        0.0 if writer_anchor_loss is None else float(writer_anchor_loss.item())
                    ),
                    "anchor_support_cosine": (
                        0.0 if anchor_support_cosine is None else float(anchor_support_cosine.item())
                    ),
                    "anchor_writer_slot_cosine": (
                        0.0 if anchor_writer_slot_cosine is None else float(anchor_writer_slot_cosine.item())
                    ),
                    "memory_long_diversity_loss": (
                        0.0
                        if memory_long_diversity_loss is None
                        else float(memory_long_diversity_loss.item())
                    ),
                    "memory_long_diversity_weight": float(memory_long_diversity_weight),
                    "memory_short_diversity_loss": (
                        0.0
                        if memory_short_diversity_loss is None
                        else float(memory_short_diversity_loss.item())
                    ),
                    "memory_short_diversity_weight": float(memory_short_diversity_weight),
                    "reader_attention_diversity_loss": (
                        0.0
                        if reader_attention_diversity_loss is None
                        else float(reader_attention_diversity_loss.item())
                    ),
                    "reader_attention_diversity_weight": float(reader_attention_diversity_weight),
                    "reader_conditioned_query_orthogonality_loss": (
                        0.0
                        if reader_conditioned_query_orthogonality_loss is None
                        else float(reader_conditioned_query_orthogonality_loss.item())
                    ),
                    "reader_conditioned_query_orthogonality_weight": float(
                        reader_conditioned_query_orthogonality_weight
                    ),
                    "reader_short_reconstruction_loss": (
                        0.0
                        if reader_short_reconstruction_loss is None
                        else float(reader_short_reconstruction_loss.item())
                    ),
                    "reader_short_reconstruction_weight": float(reader_short_reconstruction_weight),
                    "writer_slot_basis_orthogonality_loss": (
                        0.0
                        if writer_slot_basis_orthogonality_loss is None
                        else float(writer_slot_basis_orthogonality_loss.item())
                    ),
                    "writer_slot_basis_orthogonality_weight": float(
                        writer_slot_basis_orthogonality_weight
                    ),
                    "writer_slot_energy_balance_loss": (
                        0.0
                        if writer_slot_energy_balance_loss is None
                        else float(writer_slot_energy_balance_loss.item())
                    ),
                    "writer_slot_energy_balance_weight": float(
                        writer_slot_energy_balance_weight
                    ),
                    "writer_common_mode_penalty_loss": (
                        0.0
                        if writer_common_mode_penalty_loss is None
                        else float(writer_common_mode_penalty_loss.item())
                    ),
                    "writer_common_mode_penalty_weight": float(
                        writer_common_mode_penalty_weight
                    ),
                    "writer_covariance_diversity_loss": (
                        0.0
                        if writer_covariance_diversity_loss is None
                        else float(writer_covariance_diversity_loss.item())
                    ),
                    "writer_covariance_diversity_weight": float(
                        writer_covariance_diversity_weight
                    ),
                    "writer_slot_basis_pairwise_cosine_mean": _pairwise_cosine_mean(
                        runtime.writer.slot_embeddings.unsqueeze(0)
                        if hasattr(runtime.writer, "slot_embeddings")
                        else None
                    ),
                    "train_example_id": str(train_example.example["id"]),
                    "writer_frozen": writer_base_frozen and writer_adapter_frozen,
                    "writer_base_frozen": writer_base_frozen,
                    "writer_adapter_frozen": writer_adapter_frozen,
                    "projector_frozen": projector_frozen,
                    "reader_fuser_bootstrap_active": bootstrap_active,
                    "train_support_mode": train_support_mode,
                    "support_episode_id": selected_episode_id,
                    "support_episode_source_split": selected_episode_source_split,
                    "support_episode_label_counts": selected_episode_label_counts,
                    "masked_support_rows": len(masked_support_rows),
                    "masked_support_ids": [str(row["id"]) for row in masked_support_rows],
                    "pilot_bridge_mode": runtime.bridge_mode,
                    "pilot_support_encoder_mode": support_encoder_mode,
                    "pilot_trainable_variant": trainable_variant,
                    "pilot_writer_stimulus_mode": runtime.writer_stimulus_mode,
                    "pilot_writer_context_tokens": int(runtime.writer_context_tokens),
                    "pilot_writer_adapter_enabled": bool(runtime.writer_adapter_enabled),
                    "pilot_writer_adapter_target_modules": list(runtime.writer_adapter_target_modules),
                    "pilot_writer_adapter_rank": int(runtime.writer_adapter_rank),
                    "pilot_writer_adapter_alpha": float(runtime.writer_adapter_alpha),
                    "pilot_writer_adapter_dropout": float(runtime.writer_adapter_dropout),
                    "pilot_writer_adapter_trainable_params": int(runtime.writer_adapter_trainable_params),
                    "pilot_prefix_source_mode": runtime.prefix_source_mode,
                    "pilot_deep_prefix_init_mode": runtime.deep_prefix_init_mode,
                    "alignment_aux_mode": alignment_aux_mode,
                    "alignment_aux_active": bool(alignment_aux_active),
                    "alignment_aux_loss": (
                        0.0 if alignment_aux_loss is None else float(alignment_aux_loss.item())
                    ),
                    "teacher_margin_aux_loss": (
                        0.0 if alignment_aux_loss is None else float(alignment_aux_loss.item())
                    ),
                    "teacher_margin_aux_weight": float(active_alignment_aux_weight),
                    "teacher_margin_aux_active": bool(alignment_aux_active),
                    "teacher_choice_kl": float(alignment_aux_diagnostics["teacher_choice_kl"]),
                    "teacher_choice_js": float(alignment_aux_diagnostics["teacher_choice_js"]),
                    "teacher_advantage_weight_mean": float(
                        alignment_aux_diagnostics["teacher_advantage_weight_mean"]
                    ),
                    "teacher_advantage_weight_max": float(
                        alignment_aux_diagnostics["teacher_advantage_weight_max"]
                    ),
                    "teacher_margin_minus_base_margin": float(
                        alignment_aux_diagnostics["teacher_margin_minus_base_margin"]
                    ),
                    "teacher_margin_minus_active_margin": float(
                        alignment_aux_diagnostics["teacher_margin_minus_active_margin"]
                    ),
                    "active_class_entropy": float(alignment_aux_diagnostics["active_class_entropy"]),
                    "teacher_class_entropy": float(alignment_aux_diagnostics["teacher_class_entropy"]),
                    "base_class_entropy": float(alignment_aux_diagnostics["base_class_entropy"]),
                    "memory_slot_effective_rank": float(memory_slot_effective_rank),
                    "memory_long_effective_rank": float(memory_slot_effective_rank),
                    "memory_short_effective_rank": float(memory_short_effective_rank),
                    "support_state_effective_rank": float(support_state_effective_rank),
                    "pilot_memory_path_variant": runtime.memory_path_variant,
                    "pilot_writer_slot_conditioning_mode": getattr(
                        runtime.writer,
                        "slot_conditioning_mode",
                        "shared_add",
                    ),
                    "pilot_writer_shared_state_scale": float(
                        getattr(runtime.writer, "shared_state_scale", 1.0)
                    ),
                    "pilot_writer_context_query_residual_scale": float(
                        getattr(runtime.writer, "context_query_residual_scale", 0.0)
                    ),
                    "pilot_reader_context_mode": runtime.reader_context_mode,
                    "pilot_reader_conditioning_mode": (
                        None if runtime.reader is None else runtime.reader.conditioning_mode
                    ),
                    "pilot_reader_gated_add_scale": (
                        None if runtime.reader is None else float(runtime.reader.gated_add_scale)
                    ),
                    "pilot_reader_attention_mode": (
                        None if runtime.reader is None else runtime.reader.attention_mode
                    ),
                    "pilot_projector_token_source": runtime.projector_token_source,
                    "pilot_reader_num_queries": int(runtime.reader_num_queries),
                    "pilot_fuser_short_slots": int(runtime.fuser_short_slots),
                    "pilot_receiver_lora_enabled": bool(runtime.receiver_lora_enabled),
                    "pilot_receiver_lora_target_layers": list(runtime.receiver_lora_target_layers),
                    "pilot_receiver_lora_target_modules": list(runtime.receiver_lora_target_modules),
                    "pilot_receiver_lora_rank": int(runtime.receiver_lora_rank),
                    "pilot_receiver_lora_alpha": float(runtime.receiver_lora_alpha),
                    "pilot_receiver_lora_dropout": float(runtime.receiver_lora_dropout),
                    "pilot_receiver_lora_trainable_params": int(runtime.receiver_lora_trainable_params),
                    "support_encoder_grad_norm": support_encoder_grad_norm,
                    "source_stub_grad_norm": source_stub_grad_norm,
                    "prefix_projector_grad_norm": prefix_projector_grad_norm,
                    "reader_grad_norm": reader_grad_norm,
                    "fuser_grad_norm": fuser_grad_norm,
                    "writer_grad_norm": writer_grad_norm,
                    "grad_norm_support_encoder": support_encoder_grad_norm,
                    "grad_norm_source_stub": source_stub_grad_norm,
                    "grad_norm_projector": prefix_projector_grad_norm,
                    "grad_norm_prefix_projector": prefix_projector_grad_norm,
                    "grad_norm_reader": reader_grad_norm,
                    "grad_norm_fuser": fuser_grad_norm,
                    "grad_norm_writer": writer_grad_norm,
                    "grad_norm_writer_adapter": writer_adapter_grad_norm,
                    "grad_norm_receiver_lora": receiver_lora_grad_norm,
                    "reader_to_support_grad_ratio": reader_to_support_grad_ratio,
                    "fuser_to_support_grad_ratio": fuser_to_support_grad_ratio,
                    "source_to_projector_grad_ratio": source_to_projector_grad_ratio,
                    "receiver_lora_to_reader_grad_ratio": receiver_lora_to_reader_grad_ratio,
                    "writer_to_projector_grad_ratio": writer_to_projector_grad_ratio,
                    "source_stub_trainable": _module_trainable(runtime.source_stub),
                    "writer_trainable": _module_trainable(runtime.writer),
                    "writer_adapter_trainable": bool(
                        any(parameter.requires_grad for parameter in runtime.writer_adapter_parameters())
                    ),
                    "support_encoder_trainable": _module_trainable(runtime.support_encoder),
                    "reader_trainable": _module_trainable(runtime.reader),
                    "fuser_trainable": _module_trainable(runtime.fuser),
                    "prefix_projector_trainable": _module_trainable(runtime.prefix_projector),
                    "writer_trainable_params": _trainable_parameter_count(runtime.writer),
                    "writer_adapter_trainable_params": int(
                        sum(
                            parameter.numel()
                            for parameter in runtime.writer_adapter_parameters()
                            if parameter.requires_grad
                        )
                    ),
                    "source_stub_trainable_params": _trainable_parameter_count(runtime.source_stub),
                    "support_encoder_trainable_params": _trainable_parameter_count(runtime.support_encoder),
                    "reader_trainable_params": _trainable_parameter_count(runtime.reader),
                    "fuser_trainable_params": _trainable_parameter_count(runtime.fuser),
                    "prefix_projector_trainable_params": _trainable_parameter_count(runtime.prefix_projector),
                    "total_grad_norm_pre_clip": total_grad_norm,
                    "prefix_artifact_stats": prefix_artifacts.prefix_stats,
                    **train_prefix_attention_metrics,
                    **_prefix_scalar_summary(prefix_artifacts.prefix_stats),
                }
            )
            if (step + 1) in snapshot_steps and (step + 1) != 0:
                runtime.writer.eval()
                if runtime.source_stub is not None:
                    runtime.source_stub.eval()
                if runtime.support_encoder is not None:
                    runtime.support_encoder.eval()
                if runtime.reader is not None:
                    runtime.reader.eval()
                if runtime.fuser is not None:
                    runtime.fuser.eval()
                runtime.prefix_projector.eval()
                evaluate_snapshot(step + 1)
                runtime.writer.train()
                if runtime.source_stub is not None:
                    runtime.source_stub.train()
                if runtime.support_encoder is not None:
                    runtime.support_encoder.train()
                if runtime.reader is not None:
                    runtime.reader.train()
                if runtime.fuser is not None:
                    runtime.fuser.train()
                runtime.prefix_projector.train()
        runtime.writer.eval()
        if runtime.source_stub is not None:
            runtime.source_stub.eval()
        if runtime.support_encoder is not None:
            runtime.support_encoder.eval()
        if runtime.reader is not None:
            runtime.reader.eval()
        if runtime.fuser is not None:
            runtime.fuser.eval()
        runtime.prefix_projector.eval()
        runtime.set_writer_trainable(True)
        runtime.set_source_stub_trainable(True)
        runtime.set_support_encoder_trainable(True)
        runtime.set_prefix_projector_trainable(True)

    prefix_artifacts = PrefixInjectionArtifacts(
        prefix_embeddings=None,
        layer_prefix_hidden_by_layer=None,
        prefix_stats=_prefix_stats(),
    )
    if arm == "injected":
        prefix_artifacts = runtime.build_prefix_artifacts(
            support_text_block,
            support_rows=support_rows_for_prefix,
            prompt_text=default_prefix_prompt_text if bridge_mode == "writer_direct" else None,
        )
    case_rows, final_prefix_stats = _evaluate_examples(
        runtime=runtime,
        eval_examples=eval_caches,
        arm_alias=arm_alias,
        arm=arm,
        writer_memory_control=writer_memory_control,
        prompt_variant=prompt_variant,
        support_serialization_variant=support_serialization_variant,
        support_text_block=support_text_block,
        teacher_support_text_block=teacher_support_text_block,
        prefix_embeddings=prefix_artifacts.prefix_embeddings,
        layer_prefix_hidden_by_layer=prefix_artifacts.layer_prefix_hidden_by_layer,
        prefix_stats=prefix_artifacts.prefix_stats,
        task_evaluator=task_evaluator,
        support_rows=support_rows_for_prefix,
        profiler=profiler,
    )

    profile_metrics = profiler.finalize()
    class_metrics = _classification_metrics_from_rows(case_rows)
    checkpoint_path = output_dir / "checkpoint.pt"
    if arm == "injected":
        _save_shared_injection_checkpoint(
            runtime=runtime,
            output_path=checkpoint_path,
            seed=seed,
            arm_alias=arm_alias,
            arm=arm,
            writer_memory_control=writer_memory_control,
            support_dataset_path=support_dataset_path,
            support_text_block=support_text_block,
            prompt_variant=prompt_variant,
            support_serialization_variant=support_serialization_variant,
            step=train_steps,
        )

    task_case_dump_path = output_dir / "task_case_dump.jsonl"
    write_jsonl(task_case_dump_path, case_rows)
    final_prefix_scalar_stats = _prefix_scalar_summary(final_prefix_stats)
    train_reader_to_support_grad_ratio_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="reader_to_support_grad_ratio",
        step_start=1,
        step_end=4,
    )
    train_fuser_to_support_grad_ratio_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="fuser_to_support_grad_ratio",
        step_start=1,
        step_end=4,
    )
    train_source_to_projector_grad_ratio_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="source_to_projector_grad_ratio",
        step_start=1,
        step_end=4,
    )
    train_receiver_lora_to_reader_grad_ratio_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="receiver_lora_to_reader_grad_ratio",
        step_start=1,
        step_end=4,
    )
    train_grad_norm_reader_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="grad_norm_reader",
        step_start=1,
        step_end=4,
    )
    train_grad_norm_source_stub_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="grad_norm_source_stub",
        step_start=1,
        step_end=4,
    )
    train_grad_norm_fuser_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="grad_norm_fuser",
        step_start=1,
        step_end=4,
    )
    train_grad_norm_receiver_lora_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="grad_norm_receiver_lora",
        step_start=1,
        step_end=4,
    )
    train_grad_norm_writer_adapter_steps_1_4_median = _median_train_event_metric(
        train_events,
        key="grad_norm_writer_adapter",
        step_start=1,
        step_end=4,
    )
    train_reader_to_support_grad_ratio_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="reader_to_support_grad_ratio",
        step_start=5,
        step_end=8,
    )
    train_fuser_to_support_grad_ratio_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="fuser_to_support_grad_ratio",
        step_start=5,
        step_end=8,
    )
    train_source_to_projector_grad_ratio_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="source_to_projector_grad_ratio",
        step_start=5,
        step_end=8,
    )
    train_receiver_lora_to_reader_grad_ratio_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="receiver_lora_to_reader_grad_ratio",
        step_start=5,
        step_end=8,
    )
    train_grad_norm_reader_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="grad_norm_reader",
        step_start=5,
        step_end=8,
    )
    train_grad_norm_source_stub_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="grad_norm_source_stub",
        step_start=5,
        step_end=8,
    )
    train_grad_norm_fuser_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="grad_norm_fuser",
        step_start=5,
        step_end=8,
    )
    train_grad_norm_receiver_lora_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="grad_norm_receiver_lora",
        step_start=5,
        step_end=8,
    )
    train_grad_norm_writer_adapter_steps_5_8_median = _median_train_event_metric(
        train_events,
        key="grad_norm_writer_adapter",
        step_start=5,
        step_end=8,
    )
    metrics = {
        "mode": "train",
        "training_stage": "shared_injection_pilot",
        "pilot_arm_alias": arm_alias,
        "shared_injection_arm": arm,
        "writer_memory_control": writer_memory_control,
        "pilot_bridge_mode": bridge_mode,
        "pilot_injection_mode": injection_mode,
        "pilot_memory_path_variant": memory_path_variant,
        "pilot_reader_context_mode": reader_context_mode,
        "pilot_reader_conditioning_mode": (
            None if runtime.reader is None else runtime.reader.conditioning_mode
        ),
        "pilot_reader_gated_add_scale": (
            None if runtime.reader is None else float(runtime.reader.gated_add_scale)
        ),
        "pilot_reader_attention_mode": (
            None if runtime.reader is None else runtime.reader.attention_mode
        ),
        "pilot_reader_masked_partition": (
            []
            if runtime.reader is None or runtime.reader.masked_partition is None
            else [list(group) for group in runtime.reader.masked_partition]
        ),
        "pilot_reader_num_queries": reader_num_queries,
        "pilot_fuser_short_slots": fuser_short_slots,
        "pilot_projector_token_source": projector_token_source,
        "pilot_prefix_source_mode": runtime.prefix_source_mode,
        "pilot_deep_prefix_init_mode": runtime.deep_prefix_init_mode,
        "pilot_support_encoder_mode": support_encoder_mode,
        "pilot_trainable_variant": trainable_variant,
        "pilot_writer_stimulus_mode": runtime.writer_stimulus_mode,
        "pilot_writer_context_tokens": int(runtime.writer_context_tokens),
        "pilot_writer_adapter_enabled": bool(runtime.writer_adapter_enabled),
        "pilot_writer_adapter_target_modules": list(runtime.writer_adapter_target_modules),
        "pilot_writer_adapter_rank": int(runtime.writer_adapter_rank),
        "pilot_writer_adapter_alpha": float(runtime.writer_adapter_alpha),
        "pilot_writer_adapter_dropout": float(runtime.writer_adapter_dropout),
        "pilot_writer_adapter_trainable_params": int(runtime.writer_adapter_trainable_params),
        "pilot_source_stub_trainable_params": int(runtime.source_stub_trainable_params),
        "pilot_writer_support_query_residual_scale": float(
            config["method"].get("writer", {}).get("support_query_residual_scale", 0.0)
        ),
        "pilot_writer_context_query_residual_scale": float(
            config["runtime"].get("pilot_writer_context_query_residual_scale", 1.0)
        ),
        "pilot_writer_conditioning_layers": int(
            config["method"].get("writer", {}).get("conditioning_layers", 1)
        ),
        "pilot_writer_output_slot_basis_scale": float(
            getattr(runtime.writer, "output_slot_basis_scale", 0.0)
        ),
        "pilot_writer_slot_conditioning_mode": str(
            getattr(runtime.writer, "slot_conditioning_mode", "shared_add")
        ),
        "pilot_writer_shared_state_scale": float(
            getattr(runtime.writer, "shared_state_scale", 1.0)
        ),
        "pilot_writer_orthogonalize_slot_basis": orthogonalize_writer_slot_basis,
        "pilot_writer_gain_margin": writer_gain_margin,
        "pilot_writer_gain_margin_weight": writer_gain_margin_weight,
        "pilot_writer_covariance_diversity_weight": writer_covariance_diversity_weight,
        "pilot_writer_slot_energy_balance_weight": writer_slot_energy_balance_weight,
        "pilot_writer_common_mode_penalty_weight": writer_common_mode_penalty_weight,
        "pilot_receiver_lora_enabled": bool(runtime.receiver_lora_enabled),
        "pilot_receiver_lora_target_layers": list(runtime.receiver_lora_target_layers),
        "pilot_receiver_lora_target_modules": list(runtime.receiver_lora_target_modules),
        "pilot_receiver_lora_rank": int(runtime.receiver_lora_rank),
        "pilot_receiver_lora_alpha": float(runtime.receiver_lora_alpha),
        "pilot_receiver_lora_dropout": float(runtime.receiver_lora_dropout),
        "pilot_receiver_lora_trainable_params": int(runtime.receiver_lora_trainable_params),
        "pilot_alignment_aux_mode": alignment_aux_mode,
        "pilot_alignment_aux_weight": alignment_aux_weight_max,
        "pilot_alignment_aux_weight_max": alignment_aux_weight_max,
        "pilot_alignment_aux_start_step": alignment_aux_start_step,
        "pilot_alignment_aux_ramp_steps": alignment_aux_ramp_steps,
        "pilot_alignment_aux_apply_only_to_real_memory": alignment_aux_apply_only_to_real_memory,
        "pilot_alignment_aux_temperature": alignment_aux_temperature,
        "pilot_alignment_aux_advantage_center": alignment_aux_advantage_center,
        "pilot_alignment_aux_advantage_scale": alignment_aux_advantage_scale,
        "pilot_init_checkpoint_path": "" if not init_checkpoint_path else str(Path(init_checkpoint_path).resolve()),
        "pilot_deep_prefix_layers": list(deep_prefix_layers),
        "pilot_deep_prefix_rank": deep_prefix_rank,
        "pilot_train_support_mode": train_support_mode,
        "prompt_variant": prompt_variant,
        "support_serialization_variant": support_serialization_variant,
        "task_name": config["task"]["name"],
        "benchmark_id": config["task"].get("benchmark_id"),
        "pilot_split": str(config["task"].get("pilot_split", config["task"].get("smoke_subset", ""))),
        "support_dataset_path": str(Path(support_dataset_path).resolve()),
        "train_dataset_path": str(Path(train_dataset_path).resolve()),
        "train_support_dataset_path": str(Path(train_support_dataset_path).resolve()),
        "train_support_episode_bank_path": (
            "" if not train_support_episode_bank_path else str(Path(train_support_episode_bank_path).resolve())
        ),
        "eval_dataset_path": str(Path(config["task"]["dataset_path"]).resolve()),
        "support_lookup_dataset_paths": [str(Path(path).resolve()) for path in lookup_paths],
        "support_examples": len(support_caches),
        "train_support_examples": len(train_support_examples),
        "train_support_episode_count": len(train_support_episodes),
        "train_examples": len(train_caches),
        "eval_examples": len(eval_caches),
        "teacher_text_block_chars": len(teacher_support_text_block),
        "support_text_block_chars": len(support_text_block),
        "task_metric_name": str(config["task"].get("metric_name", "accuracy")),
        "task_evaluator_type": task_evaluator.evaluator_type,
        "best_adapt_task_score": float(class_metrics["accuracy"]),
        "best_adapt_macro_f1": float(class_metrics["macro_f1"]),
        "best_adapt_exact_match": float(class_metrics.get("exact_match", 0.0)),
        "best_adapt_task_margin": float(class_metrics["mean_margin"]),
        "answer_logprob_with_memory": float(class_metrics.get("answer_logprob_with_memory", 0.0)),
        "answer_logprob_without_memory": float(class_metrics.get("answer_logprob_without_memory", 0.0)),
        "delta_answer_logprob": float(class_metrics.get("delta_answer_logprob", 0.0)),
        "prefix_attention_mass_mean": float(class_metrics.get("prefix_attention_mass_mean", 0.0)),
        "prefix_to_content_attention_ratio_mean": float(
            class_metrics.get("prefix_to_content_attention_ratio_mean", 0.0)
        ),
        "gold_prefix_attention_mass_mean": float(
            class_metrics.get("gold_prefix_attention_mass_mean", 0.0)
        ),
        "competitor_prefix_attention_mass_mean": float(
            class_metrics.get("competitor_prefix_attention_mass_mean", 0.0)
        ),
        "prefix_attention_mass_mean_by_layer": dict(
            class_metrics.get("prefix_attention_mass_mean_by_layer", {})
        ),
        "prefix_to_content_attention_ratio_mean_by_layer": dict(
            class_metrics.get("prefix_to_content_attention_ratio_mean_by_layer", {})
        ),
        "gold_prefix_attention_mass_mean_by_layer": dict(
            class_metrics.get("gold_prefix_attention_mass_mean_by_layer", {})
        ),
        "competitor_prefix_attention_mass_mean_by_layer": dict(
            class_metrics.get("competitor_prefix_attention_mass_mean_by_layer", {})
        ),
        "prefix_attention_nontrivial_layer_count": int(
            class_metrics.get("prefix_attention_nontrivial_layer_count", 0)
        ),
        "dominant_label_fraction": float(class_metrics["dominant_label_fraction"]),
        "label_recall_by_class": class_metrics["label_recall_by_class"],
        "best_adapt_step": 0,
        "task_case_dump_rows": len(case_rows),
        "task_case_dump_path": str(task_case_dump_path.resolve()),
        "pilot_train_steps": train_steps,
        "pilot_writer_learning_rate": writer_learning_rate,
        "pilot_source_stub_learning_rate": source_stub_learning_rate,
        "pilot_projector_learning_rate": projector_learning_rate,
        "pilot_receiver_lora_learning_rate": receiver_lora_learning_rate,
        "pilot_writer_weight_decay": writer_weight_decay,
        "pilot_source_stub_weight_decay": source_stub_weight_decay,
        "pilot_projector_weight_decay": projector_weight_decay,
        "pilot_receiver_lora_weight_decay": receiver_lora_weight_decay,
        "pilot_projector_warmup_steps": projector_warmup_steps,
        "pilot_train_support_mask_count": train_support_mask_count,
        "pilot_choice_ce_weight": choice_ce_weight,
        "pilot_competitor_hinge_weight_max": competitor_hinge_weight_max,
        "pilot_competitor_hinge_start_step": competitor_hinge_start_step,
        "pilot_competitor_hinge_ramp_steps": competitor_hinge_ramp_steps,
        "pilot_latent_anchor_weight_start": latent_anchor_weight_start,
        "pilot_latent_anchor_weight_end": latent_anchor_weight_end,
        "pilot_latent_anchor_decay_steps": latent_anchor_decay_steps,
        "pilot_memory_long_diversity_weight": memory_long_diversity_weight,
        "pilot_memory_short_diversity_weight": memory_short_diversity_weight,
        "pilot_reader_attention_diversity_weight": reader_attention_diversity_weight,
        "pilot_reader_conditioned_query_orthogonality_weight": (
            reader_conditioned_query_orthogonality_weight
        ),
        "pilot_reader_short_reconstruction_weight": reader_short_reconstruction_weight,
        "pilot_reader_fuser_bootstrap_steps": reader_fuser_bootstrap_steps,
        "pilot_writer_slot_basis_orthogonality_weight": writer_slot_basis_orthogonality_weight,
        "pilot_gradient_clip_norm": gradient_clip_norm,
        "pilot_prefix_slot_max_norm": float(config["runtime"].get("pilot_prefix_slot_max_norm", 0.0)),
        "pilot_prefix_total_max_norm": float(config["runtime"].get("pilot_prefix_total_max_norm", 0.0)),
        "pilot_snapshot_steps": snapshot_steps,
        "snapshot_metrics": snapshot_metrics,
        "stage_c_choice_margin": choice_margin,
        "train_final_loss": train_events[-1]["loss"] if train_events else None,
        "train_final_choice_ce_loss": train_events[-1]["choice_ce_loss"] if train_events else None,
        "train_final_competitor_hinge_loss": (
            train_events[-1]["competitor_hinge_loss"] if train_events else None
        ),
        "train_final_active_competitor_hinge_weight": (
            train_events[-1]["active_competitor_hinge_weight"] if train_events else None
        ),
        "train_final_answer_logprob_with_memory": (
            train_events[-1]["answer_logprob_with_memory"] if train_events else None
        ),
        "train_final_answer_logprob_without_memory": (
            train_events[-1]["answer_logprob_without_memory"] if train_events else None
        ),
        "train_final_delta_answer_logprob": (
            train_events[-1]["delta_answer_logprob"] if train_events else None
        ),
        "train_final_writer_gain_margin_loss": (
            train_events[-1]["writer_gain_margin_loss"] if train_events else None
        ),
        "train_final_latent_anchor_loss": train_events[-1]["latent_anchor_loss"] if train_events else None,
        "train_final_latent_anchor_weight": train_events[-1]["latent_anchor_weight"] if train_events else None,
        "train_final_support_anchor_loss": train_events[-1]["support_anchor_loss"] if train_events else None,
        "train_final_writer_anchor_loss": train_events[-1]["writer_anchor_loss"] if train_events else None,
        "train_final_anchor_support_cosine": (
            train_events[-1]["anchor_support_cosine"] if train_events else None
        ),
        "train_final_anchor_writer_slot_cosine": (
            train_events[-1]["anchor_writer_slot_cosine"] if train_events else None
        ),
        "train_final_memory_long_diversity_loss": (
            train_events[-1]["memory_long_diversity_loss"] if train_events else None
        ),
        "train_final_memory_short_diversity_loss": (
            train_events[-1]["memory_short_diversity_loss"] if train_events else None
        ),
        "train_final_reader_attention_diversity_loss": (
            train_events[-1]["reader_attention_diversity_loss"] if train_events else None
        ),
        "train_final_reader_conditioned_query_orthogonality_loss": (
            train_events[-1]["reader_conditioned_query_orthogonality_loss"] if train_events else None
        ),
        "train_final_reader_short_reconstruction_loss": (
            train_events[-1]["reader_short_reconstruction_loss"] if train_events else None
        ),
        "train_final_writer_slot_basis_orthogonality_loss": (
            train_events[-1]["writer_slot_basis_orthogonality_loss"] if train_events else None
        ),
        "train_final_writer_slot_energy_balance_loss": (
            train_events[-1]["writer_slot_energy_balance_loss"] if train_events else None
        ),
        "train_final_writer_common_mode_penalty_loss": (
            train_events[-1]["writer_common_mode_penalty_loss"] if train_events else None
        ),
        "train_final_writer_covariance_diversity_loss": (
            train_events[-1]["writer_covariance_diversity_loss"] if train_events else None
        ),
        "train_final_writer_slot_basis_pairwise_cosine_mean": (
            train_events[-1]["writer_slot_basis_pairwise_cosine_mean"] if train_events else None
        ),
        "train_final_alignment_aux_loss": train_events[-1]["alignment_aux_loss"] if train_events else None,
        "train_final_teacher_margin_aux_loss": (
            train_events[-1]["teacher_margin_aux_loss"] if train_events else None
        ),
        "train_final_teacher_margin_aux_weight": (
            train_events[-1]["teacher_margin_aux_weight"] if train_events else None
        ),
        "train_final_teacher_choice_kl": train_events[-1]["teacher_choice_kl"] if train_events else None,
        "train_final_teacher_choice_js": train_events[-1]["teacher_choice_js"] if train_events else None,
        "train_final_teacher_advantage_weight_mean": (
            train_events[-1]["teacher_advantage_weight_mean"] if train_events else None
        ),
        "train_final_teacher_advantage_weight_max": (
            train_events[-1]["teacher_advantage_weight_max"] if train_events else None
        ),
        "train_final_teacher_margin_minus_base_margin": (
            train_events[-1]["teacher_margin_minus_base_margin"] if train_events else None
        ),
        "train_final_teacher_margin_minus_active_margin": (
            train_events[-1]["teacher_margin_minus_active_margin"] if train_events else None
        ),
        "train_final_active_class_entropy": train_events[-1]["active_class_entropy"] if train_events else None,
        "train_final_teacher_class_entropy": train_events[-1]["teacher_class_entropy"] if train_events else None,
        "train_final_base_class_entropy": train_events[-1]["base_class_entropy"] if train_events else None,
        "train_final_memory_slot_effective_rank": (
            train_events[-1]["memory_slot_effective_rank"] if train_events else None
        ),
        "train_final_memory_long_effective_rank": (
            train_events[-1]["memory_long_effective_rank"] if train_events else None
        ),
        "train_final_memory_short_effective_rank": (
            train_events[-1]["memory_short_effective_rank"] if train_events else None
        ),
        "train_final_support_state_effective_rank": (
            train_events[-1]["support_state_effective_rank"] if train_events else None
        ),
        "train_final_grad_norm_support_encoder": (
            train_events[-1]["grad_norm_support_encoder"] if train_events else None
        ),
        "train_final_grad_norm_source_stub": (
            train_events[-1]["grad_norm_source_stub"] if train_events else None
        ),
        "train_final_grad_norm_projector": (
            train_events[-1]["grad_norm_projector"] if train_events else None
        ),
        "train_final_grad_norm_reader": train_events[-1]["grad_norm_reader"] if train_events else None,
        "train_final_grad_norm_fuser": train_events[-1]["grad_norm_fuser"] if train_events else None,
        "train_final_grad_norm_writer": train_events[-1]["grad_norm_writer"] if train_events else None,
        "train_final_grad_norm_receiver_lora": (
            train_events[-1]["grad_norm_receiver_lora"] if train_events else None
        ),
        "train_reader_to_support_grad_ratio_steps_1_4_median": (
            train_reader_to_support_grad_ratio_steps_1_4_median
        ),
        "train_fuser_to_support_grad_ratio_steps_1_4_median": (
            train_fuser_to_support_grad_ratio_steps_1_4_median
        ),
        "train_source_to_projector_grad_ratio_steps_1_4_median": (
            train_source_to_projector_grad_ratio_steps_1_4_median
        ),
        "train_receiver_lora_to_reader_grad_ratio_steps_1_4_median": (
            train_receiver_lora_to_reader_grad_ratio_steps_1_4_median
        ),
        "train_grad_norm_reader_steps_1_4_median": train_grad_norm_reader_steps_1_4_median,
        "train_grad_norm_source_stub_steps_1_4_median": train_grad_norm_source_stub_steps_1_4_median,
        "train_grad_norm_fuser_steps_1_4_median": train_grad_norm_fuser_steps_1_4_median,
        "train_grad_norm_writer_adapter_steps_1_4_median": (
            train_grad_norm_writer_adapter_steps_1_4_median
        ),
        "train_grad_norm_receiver_lora_steps_1_4_median": (
            train_grad_norm_receiver_lora_steps_1_4_median
        ),
        "train_reader_to_support_grad_ratio_steps_5_8_median": (
            train_reader_to_support_grad_ratio_steps_5_8_median
        ),
        "train_fuser_to_support_grad_ratio_steps_5_8_median": (
            train_fuser_to_support_grad_ratio_steps_5_8_median
        ),
        "train_source_to_projector_grad_ratio_steps_5_8_median": (
            train_source_to_projector_grad_ratio_steps_5_8_median
        ),
        "train_receiver_lora_to_reader_grad_ratio_steps_5_8_median": (
            train_receiver_lora_to_reader_grad_ratio_steps_5_8_median
        ),
        "train_grad_norm_reader_steps_5_8_median": train_grad_norm_reader_steps_5_8_median,
        "train_grad_norm_source_stub_steps_5_8_median": train_grad_norm_source_stub_steps_5_8_median,
        "train_grad_norm_fuser_steps_5_8_median": train_grad_norm_fuser_steps_5_8_median,
        "train_grad_norm_writer_adapter_steps_5_8_median": (
            train_grad_norm_writer_adapter_steps_5_8_median
        ),
        "train_grad_norm_receiver_lora_steps_5_8_median": (
            train_grad_norm_receiver_lora_steps_5_8_median
        ),
        "train_final_reader_to_support_grad_ratio": (
            train_events[-1]["reader_to_support_grad_ratio"] if train_events else None
        ),
        "train_final_fuser_to_support_grad_ratio": (
            train_events[-1]["fuser_to_support_grad_ratio"] if train_events else None
        ),
        "train_final_source_to_projector_grad_ratio": (
            train_events[-1]["source_to_projector_grad_ratio"] if train_events else None
        ),
        "train_final_receiver_lora_to_reader_grad_ratio": (
            train_events[-1]["receiver_lora_to_reader_grad_ratio"] if train_events else None
        ),
        "train_final_source_stub_trainable": (
            train_events[-1]["source_stub_trainable"] if train_events else None
        ),
        "train_final_writer_trainable": train_events[-1]["writer_trainable"] if train_events else None,
        "train_final_writer_adapter_trainable": (
            train_events[-1]["writer_adapter_trainable"] if train_events else None
        ),
        "train_final_support_encoder_trainable": (
            train_events[-1]["support_encoder_trainable"] if train_events else None
        ),
        "train_final_reader_trainable": train_events[-1]["reader_trainable"] if train_events else None,
        "train_final_fuser_trainable": train_events[-1]["fuser_trainable"] if train_events else None,
        "train_final_prefix_projector_trainable": (
            train_events[-1]["prefix_projector_trainable"] if train_events else None
        ),
        "train_final_writer_trainable_params": (
            train_events[-1]["writer_trainable_params"] if train_events else None
        ),
        "train_final_writer_adapter_trainable_params": (
            train_events[-1]["writer_adapter_trainable_params"] if train_events else None
        ),
        "train_final_source_stub_trainable_params": (
            train_events[-1]["source_stub_trainable_params"] if train_events else None
        ),
        "train_final_support_encoder_trainable_params": (
            train_events[-1]["support_encoder_trainable_params"] if train_events else None
        ),
        "train_final_reader_trainable_params": (
            train_events[-1]["reader_trainable_params"] if train_events else None
        ),
        "train_final_fuser_trainable_params": (
            train_events[-1]["fuser_trainable_params"] if train_events else None
        ),
        "train_final_prefix_projector_trainable_params": (
            train_events[-1]["prefix_projector_trainable_params"] if train_events else None
        ),
        "train_final_projector_frozen": train_events[-1]["projector_frozen"] if train_events else None,
        "train_final_reader_fuser_bootstrap_active": (
            train_events[-1]["reader_fuser_bootstrap_active"] if train_events else None
        ),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path.exists() else "",
        "pilot_checkpoint_path": "" if not injection_checkpoint_path else str(Path(injection_checkpoint_path).resolve()),
        "prefix_artifact_stats": final_prefix_stats,
        **final_prefix_scalar_stats,
        **profile_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "train_events.json", {"events": train_events, "snapshots": snapshot_metrics})
    return metrics
