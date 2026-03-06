from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from memtotal.data import load_toy_dataset
from memtotal.pipeline import MemoryRuntime
from memtotal.training.m3 import run_stage_a, run_stage_b, run_stage_c
from memtotal.utils.config import load_config
from memtotal.utils.io import initialize_run_artifacts, write_json
from memtotal.utils.profiling import ProfileTracker
from memtotal.utils.repro import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MemTOTAL bootstrap train entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_bootstrap_train(args: argparse.Namespace, config: dict) -> int:
    dataset = load_toy_dataset(config["task"]["dataset_path"])
    runtime = MemoryRuntime(config=config, seed=args.seed)
    optimizer = torch.optim.Adam(runtime.parameters(), lr=float(config["runtime"]["learning_rate"]))
    steps = min(len(dataset), 2 if args.dry_run else int(config["runtime"]["train_steps"]))
    events = []
    profiler = ProfileTracker(
        output_dir=Path(args.output_dir),
        device=str(config["runtime"].get("device", "cpu")),
        event_name="train",
    )

    for step in range(steps):
        example = dataset[step % len(dataset)]
        profiler.add_example()
        optimizer.zero_grad()
        forward = runtime.forward_example(example)
        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
        profiler.add_tokens(runtime.backbone.count_tokens(forward.next_prompt))
        loss = F.mse_loss(forward.predicted_state, forward.target_state)
        loss_has_grad = bool(loss.requires_grad)
        if loss_has_grad:
            loss.backward()
            query_grad_norm = float(runtime.reader.queries.grad.norm().item())
        else:
            query_grad_norm = 0.0
        mean_gate = float(forward.gating.mean().item())
        active_queries = int((forward.gating > 0.5).sum().item())
        segment_gate_means = [float(item["mean_gate"]) for item in forward.segment_stats]
        segment_active_queries = [int(item["active_queries"]) for item in forward.segment_stats]
        if loss_has_grad:
            optimizer.step()
        events.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "loss_has_grad": loss_has_grad,
                "query_grad_norm": query_grad_norm,
                "mean_gate": mean_gate,
                "active_queries": active_queries,
                "segment_gate_means": segment_gate_means,
                "segment_active_queries": segment_active_queries,
                "num_segments": len(forward.segment_stats),
                "conditioning": forward.conditioning,
                "injection_anchors": forward.injection_anchors,
                "segments": len(forward.segments),
            }
        )

    profile_metrics = profiler.finalize()
    metrics = {
        "mode": "train",
        "examples_seen": steps,
        "final_loss": events[-1]["loss"],
        "mean_loss": sum(item["loss"] for item in events) / len(events),
        "final_query_grad_norm": events[-1]["query_grad_norm"],
        "loss_has_grad": events[-1]["loss_has_grad"],
        "gating_mode": runtime.reader.gating_mode,
        "mean_gate": sum(item["mean_gate"] for item in events) / len(events),
        "mean_active_queries": sum(item["active_queries"] for item in events) / len(events),
        "mean_segments_per_example": sum(item["num_segments"] for item in events) / len(events),
        "mean_segment_gate": (
            sum(sum(item["segment_gate_means"]) for item in events)
            / sum(len(item["segment_gate_means"]) for item in events)
        ),
        "mean_segment_active_queries": (
            sum(sum(item["segment_active_queries"]) for item in events)
            / sum(len(item["segment_active_queries"]) for item in events)
        ),
        "segment_gate_means": events[-1]["segment_gate_means"],
        "segment_active_queries": events[-1]["segment_active_queries"],
        "injection_position": runtime.injector.position,
        "conditioning_schema": {
            "domain_name": runtime.conditioning_cfg["domain_key"],
            "task_name": "config.task.name" if runtime.conditioning_cfg["include_task_name"] else "disabled",
        },
        "memory_long_shape": list(forward.memory_long.shape),
        "memory_short_shape": list(forward.memory_short.shape),
        **profile_metrics,
    }
    checkpoint = {
        "model_state": runtime.state_dict(),
        "seed": args.seed,
        "config_path": str(Path(args.config).resolve()),
    }
    torch.save(checkpoint, Path(args.output_dir) / "checkpoint.pt")
    write_json(Path(args.output_dir) / "metrics.json", metrics)
    write_json(Path(args.output_dir) / "train_events.json", {"events": events})
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    set_seed(args.seed)
    initialize_run_artifacts(
        output_dir=args.output_dir,
        config=config,
        seed=args.seed,
        argv=sys.argv if argv is None else ["train", *argv],
    )
    training_stage = str(config["runtime"].get("training_stage", "bootstrap"))
    output_dir = Path(args.output_dir)
    if training_stage == "stage_a":
        run_stage_a(config=config, seed=args.seed, output_dir=output_dir, dry_run=args.dry_run)
        return 0
    if training_stage == "stage_b":
        run_stage_b(
            config=config,
            seed=args.seed,
            output_dir=output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        return 0
    if training_stage == "stage_c":
        run_stage_c(
            config=config,
            seed=args.seed,
            output_dir=output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        return 0
    return _run_bootstrap_train(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
