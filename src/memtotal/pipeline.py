from __future__ import annotations

from dataclasses import dataclass

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
    next_prompt: str


class MemoryRuntime(nn.Module):
    def __init__(self, config: dict, seed: int) -> None:
        super().__init__()
        embed_dim = int(config["method"]["embed_dim"])
        backbone_cfg = config["backbone"]
        self.backbone = BackboneWrapper(
            name=backbone_cfg["name"],
            load_mode=backbone_cfg["load_mode"],
            hidden_size=int(backbone_cfg["stub_hidden_size"]),
            seed=seed,
        )
        if self.backbone.hidden_size != embed_dim:
            raise ValueError(
                "For M0 stub mode, method.embed_dim must match backbone.stub_hidden_size."
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
        )
        self.reader = MemoryReader(
            embed_dim=embed_dim,
            num_queries=reader_cfg["num_queries"],
            use_query_gating=bool(reader_cfg.get("use_query_gating", False)),
            gating_mode=reader_cfg.get("gating_mode"),
            num_heads=reader_cfg.get("num_heads", 4),
            condition_on_context=reader_cfg.get("condition_on_context", True),
            dropout=reader_cfg.get("dropout", 0.0),
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
        )

    def forward_example(self, example: dict[str, str]) -> ExampleForward:
        segments = self.segmenter.split(example["segment"])
        segment_state = self.backbone.summarize_texts(segments).mean(dim=0, keepdim=True)
        memory_long = self.writer.write(segment_state)
        reader_output = self.reader.read(memory_long, context=segment_state)
        memory_short = self.fuser.fuse(reader_output["readouts"])
        next_prompt = f"{example['segment']} || Continue:"
        next_inputs = self.backbone.encode_texts([next_prompt])
        generation_memory = self.injector.memory_for_generation(memory_short)
        injected_inputs = self.injector.inject(memory_short, next_inputs)
        predicted_state = injected_inputs.mean(dim=1)
        target_state = self.backbone.summarize_texts([example["continuation"]])
        return ExampleForward(
            memory_long=memory_long,
            readouts=reader_output["readouts"],
            memory_short=memory_short,
            generation_memory=generation_memory,
            injected_inputs=injected_inputs,
            predicted_state=predicted_state,
            target_state=target_state,
            gating=reader_output["gates"],
            segments=segments,
            next_prompt=next_prompt,
        )

    def predict_label(
        self,
        example: dict[str, str],
        candidate_states: torch.Tensor,
        candidate_labels: list[str],
    ) -> tuple[str, float, ExampleForward]:
        forward = self.forward_example(example)
        normalized_pred = torch.nn.functional.normalize(forward.predicted_state, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_states, dim=-1)
        scores = torch.matmul(normalized_pred, normalized_candidates.transpose(0, 1)).squeeze(0)
        best_index = int(torch.argmax(scores).item())
        return candidate_labels[best_index], float(scores[best_index].item()), forward
