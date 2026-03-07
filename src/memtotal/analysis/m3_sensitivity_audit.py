from __future__ import annotations

import copy
import csv
from math import ceil
from pathlib import Path

import torch

from memtotal.data import split_target_domain_examples
from memtotal.pipeline import ExampleForward, MemoryRuntime
from memtotal.training.m3 import (
    _build_meta_context,
    _resolve_artifact_path,
    _resolve_expected_stage_c_query_learning_mode,
)
from memtotal.utils.io import write_json
from memtotal.utils.profiling import ProfileTracker


def _compute_candidate_states(runtime: MemoryRuntime, examples: list[dict[str, str]]) -> torch.Tensor:
    return runtime.backbone.summarize_texts([str(row["continuation"]) for row in examples])


def _forward_with_memory_shift(
    runtime: MemoryRuntime,
    example: dict[str, str],
    *,
    memory_shift: float,
) -> ExampleForward:
    segments = runtime.segmenter.split(example["segment"])
    conditioning, conditioning_state = runtime._resolve_conditioning(example)
    segment_memories: list[torch.Tensor] = []
    segment_inputs: list[torch.Tensor] = []
    memory_longs: list[torch.Tensor] = []
    readouts: list[torch.Tensor] = []
    gatings: list[torch.Tensor] = []
    segment_stats: list[dict[str, object]] = []

    for segment_index, segment_text in enumerate(segments):
        segment_state = runtime.backbone.summarize_texts([segment_text])
        reader_context = (segment_state + conditioning_state) / 2.0
        memory_long = runtime.writer.write(segment_state) + memory_shift
        reader_output = runtime.reader.read(memory_long, context=reader_context)
        memory_short = runtime.fuser.fuse(reader_output["readouts"])
        segment_memories.append(memory_short)
        segment_inputs.append(runtime.backbone.encode_texts([segment_text]))
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

    memory_long = runtime._aggregate_tensors(memory_longs)
    readout_tensor = runtime._aggregate_tensors(readouts)
    memory_short = runtime._aggregate_tensors(segment_memories)
    gating = runtime._aggregate_tensors(gatings)
    delimiter_inputs = runtime.backbone.encode_texts([runtime.segmenter.delimiter]) if len(segments) > 1 else None
    suffix_inputs = runtime.backbone.encode_texts(["Continue:"])
    injected_inputs, generation_memory, injection_anchors = runtime.injector.compose(
        segment_memories=segment_memories,
        segment_inputs=segment_inputs,
        delimiter_inputs=delimiter_inputs,
        suffix_inputs=suffix_inputs,
    )
    predicted_state = injected_inputs.mean(dim=1)
    target_state = runtime.backbone.summarize_texts([example["continuation"]])
    next_prompt = f"{example['segment']} || Continue:"
    for anchor in injection_anchors:
        segment_index = None
        if ":" in anchor:
            anchor_body = anchor.split(":", maxsplit=1)[1]
            maybe_index = anchor_body.split("@", maxsplit=1)[0]
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


def _delta_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm().item())


def _write_sensitivity_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "query_delta_readouts_norm",
                "memory_delta_readouts_norm",
                "query_delta_memory_short_norm",
                "memory_delta_memory_short_norm",
                "query_delta_summary_norm",
                "memory_delta_summary_norm",
                "query_delta_score_norm",
                "memory_delta_score_norm",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_sensitivity_svg(path: Path, query_value: float, memory_value: float) -> None:
    max_value = max(query_value, memory_value, 1e-9)
    left = 240
    width = 380
    parts = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='780' height='190'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='34' font-size='20' font-family='monospace'>M3 Stage C sensitivity audit</text>",
        "<text x='24' y='56' font-size='13' font-family='monospace'>mean candidate-score delta under +epsilon perturbation</text>",
    ]
    for index, (label, value, color) in enumerate(
        [
            ("query shift", query_value, "#2f5aa8"),
            ("memory shift", memory_value, "#b5651d"),
        ]
    ):
        top = 82 + index * 46
        scaled = ceil((value / max_value) * width) if max_value > 0 else 0
        parts.append(f"<text x='24' y='{top + 18}' font-size='13' font-family='monospace'>{label}</text>")
        parts.append(f"<rect x='{left}' y='{top}' width='{width}' height='24' fill='#efe6d0' rx='4' />")
        if scaled > 0:
            parts.append(f"<rect x='{left}' y='{top}' width='{scaled}' height='24' fill='{color}' rx='4' />")
        parts.append(
            f"<text x='{left + width + 12}' y='{top + 18}' font-size='13' font-family='monospace'>{value:.6e}</text>"
        )
    parts.append("</svg>")
    path.write_text("".join(parts))


def run_m3_stage_c_sensitivity_audit(
    *,
    config: dict,
    seed: int,
    output_dir: Path,
    resume: str | None,
    dry_run: bool,
) -> dict[str, object]:
    grouped_examples, manifest = _build_meta_context(config)
    write_json(output_dir / "meta_data_manifest.json", manifest)

    writer_path = _resolve_artifact_path(resume, "writer.ckpt")
    queries_path = _resolve_artifact_path(resume, "queries_meta_init.pt")
    runtime = MemoryRuntime(config=config, seed=seed)
    runtime.writer.load_from(writer_path)
    state = torch.load(queries_path, map_location="cpu")
    runtime.reader.load_state_dict(state["reader_state"])
    if "fuser_state" in state:
        runtime.fuser.load_state_dict(state["fuser_state"])

    expected_query_learning_mode = _resolve_expected_stage_c_query_learning_mode(config)
    actual_query_learning_mode = str(state.get("query_learning_mode", "unknown"))
    if expected_query_learning_mode is not None and actual_query_learning_mode != expected_query_learning_mode:
        raise ValueError(
            f"Sensitivity audit expected query_learning_mode={expected_query_learning_mode}, "
            f"but resume artifact provides {actual_query_learning_mode}."
        )

    epsilon = float(config["runtime"].get("sensitivity_epsilon", 0.1))
    target_episode = split_target_domain_examples(
        grouped_examples,
        target_domain=manifest["target_domain"],
        support_size=int(manifest["support_size"]),
        query_size=int(manifest["query_size"]),
        seed=seed,
        sampling_policy=str(manifest["sampling_policy"]),
    )
    support_examples = target_episode.support_examples[:1] if dry_run else target_episode.support_examples
    candidate_states = _compute_candidate_states(runtime, target_episode.query_examples)

    rows: list[dict[str, object]] = []
    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="analysis")
    for example in support_examples:
        baseline_forward = runtime.forward_example(example)
        baseline_summary = runtime.summarize_memory_short(baseline_forward.memory_short)
        baseline_scores = runtime.score_candidates(baseline_summary, candidate_states)

        query_runtime = copy.deepcopy(runtime)
        with torch.no_grad():
            query_runtime.reader.queries.add_(epsilon)
        query_forward = query_runtime.forward_example(example)
        query_summary = query_runtime.summarize_memory_short(query_forward.memory_short)
        query_scores = query_runtime.score_candidates(query_summary, candidate_states)

        memory_forward = _forward_with_memory_shift(runtime, example, memory_shift=epsilon)
        memory_summary = runtime.summarize_memory_short(memory_forward.memory_short)
        memory_scores = runtime.score_candidates(memory_summary, candidate_states)

        rows.append(
            {
                "example_id": str(example["id"]),
                "query_delta_readouts_norm": _delta_norm(query_forward.readouts, baseline_forward.readouts),
                "memory_delta_readouts_norm": _delta_norm(memory_forward.readouts, baseline_forward.readouts),
                "query_delta_memory_short_norm": _delta_norm(query_forward.memory_short, baseline_forward.memory_short),
                "memory_delta_memory_short_norm": _delta_norm(memory_forward.memory_short, baseline_forward.memory_short),
                "query_delta_summary_norm": _delta_norm(query_summary, baseline_summary),
                "memory_delta_summary_norm": _delta_norm(memory_summary, baseline_summary),
                "query_delta_score_norm": _delta_norm(query_scores, baseline_scores),
                "memory_delta_score_norm": _delta_norm(memory_scores, baseline_scores),
            }
        )
        profiler.add_example()
        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))

    csv_path = output_dir / "sensitivity_audit.csv"
    _write_sensitivity_csv(csv_path, rows)
    query_score_mean = sum(float(row["query_delta_score_norm"]) for row in rows) / len(rows)
    memory_score_mean = sum(float(row["memory_delta_score_norm"]) for row in rows) / len(rows)
    svg_path = output_dir / "sensitivity_audit.svg"
    _write_sensitivity_svg(svg_path, query_score_mean, memory_score_mean)

    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_sensitivity_audit",
        "backbone": str(config["backbone"]["name"]),
        "query_learning_mode": actual_query_learning_mode,
        "target_domain": manifest["target_domain"],
        "support_examples_evaluated": len(rows),
        "sensitivity_epsilon": epsilon,
        "mean_query_delta_readouts_norm": sum(float(row["query_delta_readouts_norm"]) for row in rows) / len(rows),
        "mean_memory_delta_readouts_norm": sum(float(row["memory_delta_readouts_norm"]) for row in rows) / len(rows),
        "mean_query_delta_memory_short_norm": sum(float(row["query_delta_memory_short_norm"]) for row in rows) / len(rows),
        "mean_memory_delta_memory_short_norm": sum(float(row["memory_delta_memory_short_norm"]) for row in rows)
        / len(rows),
        "mean_query_delta_summary_norm": sum(float(row["query_delta_summary_norm"]) for row in rows) / len(rows),
        "mean_memory_delta_summary_norm": sum(float(row["memory_delta_summary_norm"]) for row in rows) / len(rows),
        "mean_query_delta_score_norm": query_score_mean,
        "mean_memory_delta_score_norm": memory_score_mean,
        "query_to_memory_score_delta_ratio": query_score_mean / max(memory_score_mean, 1e-12),
        "sensitivity_csv": str(csv_path.resolve()),
        "sensitivity_plot": str(svg_path.resolve()),
        **profiler.finalize(),
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
