from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from memtotal.baselines.budgeting import build_baseline_budget_fields
from memtotal.models import BackboneWrapper
from memtotal.tasks import build_task_evaluator, load_task_dataset
from memtotal.utils.io import write_json
from memtotal.utils.profiling import ProfileTracker


@dataclass(frozen=True)
class AdapterBaselineOutput:
    prompt: str
    predicted_label: str
    predicted_text: str
    similarity: float
    candidate_scores: list[dict[str, float | str]]


class PromptTuningAdapter(nn.Module):
    def __init__(self, hidden_size: int, prompt_tokens: int) -> None:
        super().__init__()
        self.soft_prompts = nn.Parameter(torch.zeros(prompt_tokens, hidden_size))
        nn.init.normal_(self.soft_prompts, mean=0.0, std=0.02)

    def forward(self, prompt_state: torch.Tensor) -> torch.Tensor:
        return prompt_state + self.soft_prompts.mean(dim=0, keepdim=True)


class LoRAAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.scale = alpha / max(1, rank)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, prompt_state: torch.Tensor) -> torch.Tensor:
        return prompt_state + self.scale * self.up(self.down(prompt_state))


class IA3Adapter(nn.Module):
    def __init__(self, hidden_size: int, init_scale: float) -> None:
        super().__init__()
        self.gates = nn.Parameter(torch.full((hidden_size,), float(init_scale - 1.0)))

    def forward(self, prompt_state: torch.Tensor) -> torch.Tensor:
        return prompt_state * (1.0 + self.gates.unsqueeze(0))


class AdapterBaselineRuntime(nn.Module):
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__()
        baseline_cfg = config.get("baseline", {})
        family = str(baseline_cfg.get("family", "adapter"))
        if family != "adapter":
            raise ValueError(f"Unsupported baseline family: {family}")
        mode = str(baseline_cfg.get("mode", "prompt_tuning"))
        if mode not in {"prompt_tuning", "lora", "ia3"}:
            raise ValueError(f"Unsupported adapter baseline mode: {mode}")
        backbone_cfg = config["backbone"]
        self.backbone = BackboneWrapper(
            name=backbone_cfg["name"],
            load_mode=backbone_cfg["load_mode"],
            hidden_size=int(backbone_cfg["stub_hidden_size"]),
            seed=seed,
        )
        self.family = family
        self.mode = mode
        if mode == "prompt_tuning":
            prompt_tokens = int(baseline_cfg.get("prompt_tuning", {}).get("prompt_tokens", 4))
            self.adapter = PromptTuningAdapter(self.backbone.hidden_size, prompt_tokens)
        elif mode == "lora":
            lora_cfg = baseline_cfg.get("lora", {})
            self.adapter = LoRAAdapter(
                hidden_size=self.backbone.hidden_size,
                rank=int(lora_cfg.get("rank", 4)),
                alpha=float(lora_cfg.get("alpha", 8.0)),
            )
        else:
            ia3_cfg = baseline_cfg.get("ia3", {})
            self.adapter = IA3Adapter(
                hidden_size=self.backbone.hidden_size,
                init_scale=float(ia3_cfg.get("init_scale", 1.0)),
            )

    def build_prompt(self, example: dict[str, Any]) -> str:
        return str(example["segment"])

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        prompt_state = self.backbone.summarize_texts([prompt])
        return self.adapter(prompt_state)

    def score_candidates(self, prompt: str, candidate_texts: list[str]) -> torch.Tensor:
        prompt_state = self.encode_prompt(prompt)
        candidate_states = self.backbone.summarize_texts(candidate_texts)
        normalized_prompt = torch.nn.functional.normalize(prompt_state, dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_states, dim=-1)
        return torch.matmul(normalized_prompt, normalized_candidates.transpose(0, 1)).squeeze(0)

    def predict_multiple_choice(
        self,
        example: dict[str, Any],
        *,
        candidate_labels: list[str],
        candidate_texts: list[str],
        support_examples: list[dict[str, Any]] | None = None,
    ) -> AdapterBaselineOutput:
        prompt = self.build_prompt(example)
        scores = self.score_candidates(prompt, candidate_texts)
        best_index = int(torch.argmax(scores).item())
        return AdapterBaselineOutput(
            prompt=prompt,
            predicted_label=candidate_labels[best_index],
            predicted_text=candidate_texts[best_index],
            similarity=float(scores[best_index].item()),
            candidate_scores=[
                {
                    "label": label,
                    "text": text,
                    "score": float(scores[index].item()),
                }
                for index, (label, text) in enumerate(zip(candidate_labels, candidate_texts))
            ],
        )


def _candidate_payload(
    example: dict[str, Any],
    dataset: list[dict[str, Any]],
    evaluator_type: str,
) -> tuple[list[str], list[str], int]:
    if evaluator_type == "multiple_choice":
        choices = example.get("choices", [])
        if not choices:
            raise ValueError("Adapter baselines require per-example `choices` for multiple_choice tasks.")
        candidate_labels = [str(choice["label"]) for choice in choices]
        candidate_texts = [str(choice["text"]) for choice in choices]
    else:
        candidate_labels = [str(row["label"]) for row in dataset]
        candidate_texts = [str(row["continuation"]) for row in dataset]
    try:
        gold_index = candidate_labels.index(str(example["label"]))
    except ValueError as exc:
        raise ValueError("Gold label is missing from adapter baseline candidates.") from exc
    return candidate_labels, candidate_texts, gold_index


def run_adapter_baseline_train(
    *,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    dry_run: bool,
) -> None:
    dataset = load_task_dataset(config)
    evaluator = build_task_evaluator(config)
    if evaluator.evaluator_type not in {"multiple_choice", "dataset_label_classification"}:
        raise ValueError(
            "Adapter baseline smoke currently supports only multiple_choice or dataset_label_classification tasks."
        )
    runtime = AdapterBaselineRuntime(config=config, seed=seed)
    baseline_cfg = config.get("baseline", {})
    support_examples = min(len(dataset), max(0, int(baseline_cfg.get("support_examples", 1))))
    train_steps = min(
        int(config["runtime"].get("train_steps", 1)),
        2 if dry_run else int(config["runtime"].get("train_steps", 1)),
    )
    if support_examples <= 0 and train_steps > 0:
        raise ValueError("Adapter baseline training with train_steps > 0 requires baseline.support_examples >= 1.")
    optimizer = torch.optim.Adam(runtime.parameters(), lr=float(config["runtime"].get("learning_rate", 0.01)))
    events = []
    profiler = ProfileTracker(
        output_dir=output_dir,
        device=str(config["runtime"].get("device", "cpu")),
        event_name="train",
    )

    support_set = dataset[:support_examples]
    for step in range(train_steps):
        example = support_set[step % len(support_set)]
        candidate_labels, candidate_texts, gold_index = _candidate_payload(
            example,
            dataset,
            evaluator.evaluator_type,
        )
        profiler.add_example()
        prompt = runtime.build_prompt(example)
        profiler.add_tokens(runtime.backbone.count_tokens(prompt))
        for candidate_text in candidate_texts:
            profiler.add_tokens(runtime.backbone.count_tokens(candidate_text))
        optimizer.zero_grad()
        scores = runtime.score_candidates(prompt, candidate_texts)
        loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([gold_index], dtype=torch.long))
        loss.backward()
        optimizer.step()
        events.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "support_example_id": str(example["id"]),
                "prompt_tokens": runtime.backbone.count_tokens(prompt),
            }
        )

    profile_metrics = profiler.finalize()
    trainable_parameter_count = sum(parameter.numel() for parameter in runtime.parameters() if parameter.requires_grad)
    budget_fields = build_baseline_budget_fields(
        config=config,
        baseline_family=runtime.family,
        baseline_mode=runtime.mode,
        support_examples=support_examples,
        train_steps=train_steps,
        trainable_parameter_count=trainable_parameter_count,
    )
    checkpoint = {
        "model_state": runtime.state_dict(),
        "seed": seed,
        "config_path": str(config["_meta"]["config_path"]),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    write_json(
        output_dir / "metrics.json",
        {
            "mode": "train_baseline",
            "examples_seen": train_steps,
            "final_loss": events[-1]["loss"] if events else None,
            "mean_loss": (sum(item["loss"] for item in events) / len(events)) if events else None,
            "task_name": config["task"]["name"],
            "benchmark_id": config["task"].get("benchmark_id"),
            "task_domain": config["task"].get("domain"),
            "smoke_subset": config["task"].get("smoke_subset"),
            "backbone": config["backbone"]["name"],
            **budget_fields,
            **profile_metrics,
        },
    )
    write_json(output_dir / "train_events.json", {"events": events})
