from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from memtotal.models import BackboneWrapper, MemoryFuser, MemoryInjector, MemoryReader, MemoryWriter, Segmenter


@dataclass
class ExampleForward:
    memory_long: torch.Tensor
    readouts: torch.Tensor
    memory_short: torch.Tensor
    generation_memory: torch.Tensor | None
    injected_inputs: torch.Tensor
    predicted_state: torch.Tensor
    target_state: torch.Tensor
    gating: torch.Tensor
    segments: list[str]
    segment_stats: list[dict[str, Any]]
    conditioning: dict[str, str]
    injection_anchors: list[str]
    next_prompt: str


class MemoryRuntime(nn.Module):
    def __init__(self, config: dict, seed: int) -> None:
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
            max_new_tokens=int(backbone_cfg.get("max_new_tokens", 32)),
        )
        embed_dim_raw = config["method"].get("embed_dim", self.backbone.hidden_size)
        embed_dim = self.backbone.hidden_size if str(embed_dim_raw) == "auto" else int(embed_dim_raw)
        self.task_name = str(config["task"]["name"])
        if self.backbone.hidden_size != embed_dim:
            raise ValueError(
                f"method.embed_dim={embed_dim} must match backbone hidden size {self.backbone.hidden_size}."
            )
        self.segmenter = Segmenter(
            mode=config["method"]["segmenter"]["mode"],
            delimiter=config["method"]["segmenter"]["delimiter"],
        )
        writer_cfg = config["method"]["writer"]
        reader_cfg = config["method"]["reader"]
        fuser_cfg = config["method"]["fuser"]
        self.writer = MemoryWriter(
            embed_dim=embed_dim,
            memory_slots=writer_cfg["memory_slots"],
            arch=writer_cfg.get("arch", "mlp"),
            hidden_dim=writer_cfg.get("hidden_dim"),
            num_heads=writer_cfg.get("num_heads", 4),
            transformer_layers=writer_cfg.get("transformer_layers", 1),
            dropout=writer_cfg.get("dropout", 0.0),
            support_query_residual_scale=float(writer_cfg.get("support_query_residual_scale", 0.0)),
            output_slot_basis_scale=float(writer_cfg.get("output_slot_basis_scale", 0.0)),
        )
        self.reader = MemoryReader(
            embed_dim=embed_dim,
            num_queries=reader_cfg["num_queries"],
            use_query_gating=bool(reader_cfg.get("use_query_gating", False)),
            gating_mode=reader_cfg.get("gating_mode"),
            num_heads=reader_cfg.get("num_heads", 4),
            condition_on_context=reader_cfg.get("condition_on_context", True),
            dropout=reader_cfg.get("dropout", 0.0),
            query_residual_scale=float(reader_cfg.get("query_residual_scale", 0.0)),
        )
        self.fuser = MemoryFuser(
            embed_dim=embed_dim,
            num_queries=reader_cfg["num_queries"],
            short_slots=fuser_cfg["short_slots"],
            arch=fuser_cfg.get("arch", "linear"),
            hidden_dim=fuser_cfg.get("hidden_dim"),
            num_heads=fuser_cfg.get("num_heads", 4),
            dropout=fuser_cfg.get("dropout", 0.0),
        )
        self.injector = MemoryInjector(
            mode=config["method"]["injector"]["mode"],
            enabled=bool(config["method"]["injector"].get("enabled", True)),
            position=str(config["method"]["injector"].get("position", "segment")),
        )
        conditioning_cfg = reader_cfg.get("conditioning", {})
        self.conditioning_cfg = {
            "domain_key": str(conditioning_cfg.get("domain_key", "domain")),
            "include_task_name": bool(conditioning_cfg.get("include_task_name", True)),
        }
        self.runtime_device = torch.device(runtime_device)
        self.to(self.runtime_device)

    def _resolve_conditioning(self, example: dict[str, str]) -> tuple[dict[str, str], torch.Tensor]:
        domain_key = self.conditioning_cfg["domain_key"]
        if domain_key not in example:
            raise ValueError(
                f"Example is missing conditioning domain key '{domain_key}'. "
                "Update method.reader.conditioning.domain_key or dataset fields."
            )
        conditioning = {
            "domain_name": str(example[domain_key]),
        }
        if self.conditioning_cfg["include_task_name"]:
            conditioning["task_name"] = self.task_name
        conditioning_text = " || ".join(f"{key}: {value}" for key, value in conditioning.items())
        conditioning_state = self.backbone.summarize_texts([conditioning_text])
        return conditioning, conditioning_state

    def _aggregate_tensors(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensors, dim=0).mean(dim=0)

    def forward_example(self, example: dict[str, str]) -> ExampleForward:
        segments = self.segmenter.split(example["segment"])
        conditioning, conditioning_state = self._resolve_conditioning(example)
        segment_memories: list[torch.Tensor] = []
        segment_inputs: list[torch.Tensor] = []
        memory_longs: list[torch.Tensor] = []
        readouts: list[torch.Tensor] = []
        gatings: list[torch.Tensor] = []
        segment_stats: list[dict[str, Any]] = []

        for segment_index, segment_text in enumerate(segments):
            segment_state = self.backbone.summarize_texts([segment_text])
            reader_context = (segment_state + conditioning_state) / 2.0
            memory_long = self.writer.write(segment_state)
            reader_output = self.reader.read(memory_long, context=reader_context)
            memory_short = self.fuser.fuse(reader_output["readouts"])
            segment_memories.append(memory_short)
            segment_inputs.append(self.backbone.encode_texts([segment_text]))
            memory_longs.append(memory_long)
            readouts.append(reader_output["readouts"])
            gatings.append(reader_output["gates"])
            segment_stats.append(
                {
                    "segment_index": segment_index,
                    "segment_text": segment_text,
                    "mean_gate": float(reader_output["gates"].mean().item()),
                    "active_queries": int((reader_output["gates"] > 0.5).sum().item()),
                    "gates": [float(value) for value in reader_output["gates"].squeeze(0).tolist()],
                    "injection_anchor": "not-injected",
                }
            )

        memory_long = self._aggregate_tensors(memory_longs)
        readout_tensor = self._aggregate_tensors(readouts)
        memory_short = self._aggregate_tensors(segment_memories)
        gating = self._aggregate_tensors(gatings)
        next_prompt = f"{example['segment']} || Continue:"
        delimiter_inputs = self.backbone.encode_texts([self.segmenter.delimiter]) if len(segments) > 1 else None
        suffix_inputs = self.backbone.encode_texts(["Continue:"])
        injected_inputs, generation_memory, injection_anchors = self.injector.compose(
            segment_memories=segment_memories,
            segment_inputs=segment_inputs,
            delimiter_inputs=delimiter_inputs,
            suffix_inputs=suffix_inputs,
        )
        predicted_state = injected_inputs.mean(dim=1)
        target_state = self.backbone.summarize_texts([example["continuation"]])
        for anchor in injection_anchors:
            segment_index = None
            if ":" in anchor:
                anchor_body = anchor.split(":", maxsplit=1)[1]
                if "@" in anchor_body:
                    maybe_index = anchor_body.split("@", maxsplit=1)[0]
                else:
                    maybe_index = anchor_body
                if maybe_index.isdigit():
                    segment_index = int(maybe_index)
            if segment_index is None and anchor.endswith("last"):
                segment_index = len(segment_stats) - 1
            if segment_index is not None and 0 <= segment_index < len(segment_stats):
                segment_stats[segment_index]["injection_anchor"] = anchor
        return ExampleForward(
            memory_long=memory_long,
            readouts=readout_tensor,
            memory_short=memory_short,
            generation_memory=generation_memory,
            injected_inputs=injected_inputs,
            predicted_state=predicted_state,
            target_state=target_state,
            gating=gating,
            segments=segments,
            segment_stats=segment_stats,
            conditioning=conditioning,
            injection_anchors=injection_anchors,
            next_prompt=next_prompt,
        )

    def predict_label(
        self,
        example: dict[str, str],
        candidate_states: torch.Tensor,
        candidate_labels: list[str],
    ) -> tuple[str, float, ExampleForward]:
        forward = self.forward_example(example)
        scores = self.score_candidates(self.summarize_memory_short(forward.memory_short), candidate_states)
        best_index = int(torch.argmax(scores).item())
        return candidate_labels[best_index], float(scores[best_index].item()), forward

    def summarize_memory_short(self, memory_short: torch.Tensor) -> torch.Tensor:
        return self.fuser.summarize(memory_short)

    def score_candidates(
        self,
        predicted_state: torch.Tensor,
        candidate_states: torch.Tensor,
    ) -> torch.Tensor:
        normalized_pred = torch.nn.functional.normalize(predicted_state, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_states, dim=-1)
        return torch.matmul(normalized_pred, normalized_candidates.transpose(0, 1)).squeeze(0)
