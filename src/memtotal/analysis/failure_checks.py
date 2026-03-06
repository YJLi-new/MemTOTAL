from __future__ import annotations

import csv
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F

from memtotal.data import build_meta_manifest, load_domain_dataset, validate_meta_split
from memtotal.pipeline import MemoryRuntime
from memtotal.training.m3 import (
    _load_meta_config,
    _resolve_artifact_path,
    _resolve_stage_b_query_learning_mode,
)
from memtotal.utils.io import write_json
from memtotal.utils.profiling import ProfileTracker


def _build_meta_context(config: dict) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    meta_cfg = _load_meta_config(config)
    grouped_examples = load_domain_dataset(config["task"]["dataset_path"])
    validate_meta_split(grouped_examples, **meta_cfg)
    manifest = build_meta_manifest(
        dataset_path=config["task"]["dataset_path"],
        grouped_examples=grouped_examples,
        **meta_cfg,
    )
    return grouped_examples, manifest


def _build_candidate_bank(
    runtime: MemoryRuntime,
    grouped_examples: dict[str, list[dict[str, str]]],
) -> dict[str, tuple[torch.Tensor, list[str]]]:
    bank: dict[str, tuple[torch.Tensor, list[str]]] = {}
    for domain, rows in grouped_examples.items():
        labels = sorted({str(row["label"]) for row in rows})
        states = []
        for label in labels:
            label_texts = [row["continuation"] for row in rows if str(row["label"]) == label]
            states.append(runtime.backbone.summarize_texts(label_texts).mean(dim=0))
        bank[domain] = (torch.stack(states, dim=0), labels)
    return bank


def _pairwise_slot_diversity(memory_short: torch.Tensor) -> float:
    if memory_short.shape[1] <= 1:
        return 0.0
    normalized = torch.nn.functional.normalize(memory_short.squeeze(0), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    off_diag = similarity[mask]
    return float((1.0 - off_diag.mean()).item())


def _variant_memory_short(
    runtime: MemoryRuntime,
    example: dict[str, str],
    *,
    variant: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    segments = runtime.segmenter.split(example["segment"])
    conditioning, conditioning_state = runtime._resolve_conditioning(example)
    del conditioning
    shorts: list[torch.Tensor] = []
    diversities: list[float] = []
    for segment_text in segments:
        segment_state = runtime.backbone.summarize_texts([segment_text])
        reader_context = (segment_state + conditioning_state) / 2.0
        memory_long = runtime.writer.write(segment_state)
        base_readouts = runtime.reader.read(memory_long, context=reader_context)["readouts"]
        if variant == "base":
            memory_short = runtime.fuser.fuse(base_readouts)
        elif variant == "zero_memory":
            zero_readouts = runtime.reader.read(torch.zeros_like(memory_long), context=reader_context)["readouts"]
            memory_short = runtime.fuser.fuse(zero_readouts)
        elif variant == "writer_noise":
            memory_std = float(memory_long.std().item())
            noise = torch.randn(
                memory_long.shape,
                generator=generator,
                device=memory_long.device,
                dtype=memory_long.dtype,
            ) * max(memory_std, 1e-6)
            noise = noise + memory_long.mean()
            noise_readouts = runtime.reader.read(noise, context=reader_context)["readouts"]
            memory_short = runtime.fuser.fuse(noise_readouts)
        elif variant == "collapsed_fuser":
            base_short = runtime.fuser.fuse(base_readouts)
            memory_short = base_short.mean(dim=1, keepdim=True).expand_as(base_short)
        else:
            raise ValueError(f"Unsupported failure-check variant: {variant}")
        shorts.append(memory_short)
        diversities.append(_pairwise_slot_diversity(memory_short))
    return runtime._aggregate_tensors(shorts), sum(diversities) / len(diversities)


def _evaluate_variant(
    runtime: MemoryRuntime,
    examples: list[dict[str, str]],
    candidate_bank: dict[str, tuple[torch.Tensor, list[str]]],
    *,
    variant: str,
    generator: torch.Generator,
    profiler: ProfileTracker,
    writer_noise_trials: int = 1,
) -> dict[str, float | int | str]:
    losses = []
    accuracies = []
    diversities = []
    for example in examples:
        profiler.add_example()
        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
        candidate_states, candidate_labels = candidate_bank[str(example["domain"])]
        gold_index = candidate_labels.index(str(example["label"]))
        variant_trials = writer_noise_trials if variant == "writer_noise" else 1
        trial_losses = []
        trial_accuracies = []
        trial_diversities = []
        for _ in range(variant_trials):
            memory_short, diversity = _variant_memory_short(
                runtime,
                example,
                variant=variant,
                generator=generator,
            )
            scores = runtime.score_candidates(runtime.summarize_memory_short(memory_short), candidate_states)
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([gold_index], dtype=torch.long))
            trial_losses.append(float(loss.item()))
            trial_accuracies.append(float(int(torch.argmax(scores).item() == gold_index)))
            trial_diversities.append(diversity)
        losses.append(sum(trial_losses) / len(trial_losses))
        accuracies.append(sum(trial_accuracies) / len(trial_accuracies))
        diversities.append(sum(trial_diversities) / len(trial_diversities))
    return {
        "variant": variant,
        "num_examples": len(examples),
        "query_loss": sum(losses) / len(losses),
        "query_accuracy": sum(accuracies) / len(accuracies),
        "mean_short_slot_diversity": sum(diversities) / len(diversities),
    }


def _write_variant_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "num_examples", "query_loss", "query_accuracy", "mean_short_slot_diversity"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_variant_svg(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    max_value = max(float(row["query_loss"]) for row in rows)
    min_value = min(float(row["query_loss"]) for row in rows)
    span = max(max_value - min_value, 1e-6)
    width = 720
    height = 260
    left = 120
    bar_width = 450
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='32' font-size='20' font-family='monospace'>M3 failure checks</text>",
        "<text x='24' y='54' font-size='13' font-family='monospace'>lower query_loss is better</text>",
    ]
    palette = {
        "base": "#2f5aa8",
        "zero_memory": "#b5651d",
        "writer_noise": "#8a2b2b",
        "collapsed_fuser": "#2b8a3e",
    }
    for index, row in enumerate(rows):
        top = 80 + index * 40
        value = float(row["query_loss"])
        scaled = ceil(((value - min_value) / span) * max(bar_width - 40, 1))
        width_px = max(scaled, 6)
        parts.append(
            f"<text x='24' y='{top + 18}' font-size='13' font-family='monospace'>{row['variant']}</text>"
        )
        parts.append(
            f"<rect x='{left}' y='{top}' width='{bar_width}' height='24' fill='#efe6d0' rx='4' />"
        )
        parts.append(
            f"<rect x='{left}' y='{top}' width='{width_px}' height='24' "
            f"fill='{palette.get(str(row['variant']), '#4a6fa5')}' rx='4' />"
        )
        parts.append(
            f"<text x='{left + bar_width + 12}' y='{top + 18}' font-size='13' font-family='monospace'>{value:.4f}</text>"
        )
    parts.append("</svg>")
    path.write_text("".join(parts))


def run_m3_failure_checks(
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
    query_learning_mode = str(state.get("query_learning_mode", "unknown"))
    expected_query_learning_mode = config["runtime"].get("expected_query_learning_mode")
    if expected_query_learning_mode is not None:
        normalized_expected = _resolve_stage_b_query_learning_mode(
            {"runtime": {"query_learning_mode": expected_query_learning_mode}}
        )
        if query_learning_mode != normalized_expected:
            raise ValueError(
                f"Failure checks expected query_learning_mode={normalized_expected}, "
                f"but resume artifact provides {query_learning_mode}."
            )

    eval_domains = list(config["runtime"]["failure_checks"].get("eval_domains", manifest["source_domains"]))
    if manifest["target_domain"] not in eval_domains:
        eval_domains.append(str(manifest["target_domain"]))
    per_domain_cap = int(config["runtime"]["failure_checks"].get("max_examples_per_domain", 0))
    examples = []
    for domain in eval_domains:
        domain_examples = list(grouped_examples[domain])
        if dry_run:
            domain_examples = domain_examples[:1]
        elif per_domain_cap > 0:
            domain_examples = domain_examples[:per_domain_cap]
        examples.extend(domain_examples)

    candidate_bank = _build_candidate_bank(runtime, grouped_examples)
    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="analysis")
    noise_generator = torch.Generator(device="cpu").manual_seed(seed)
    writer_noise_trials = max(1, int(config["runtime"]["failure_checks"].get("writer_noise_trials", 1)))
    variant_rows = []
    with torch.no_grad():
        for variant in ["base", "zero_memory", "writer_noise", "collapsed_fuser"]:
            variant_rows.append(
                _evaluate_variant(
                    runtime,
                    examples,
                    candidate_bank,
                    variant=variant,
                    generator=noise_generator,
                    profiler=profiler,
                    writer_noise_trials=writer_noise_trials,
                )
            )

    by_variant = {str(row["variant"]): row for row in variant_rows}
    thresholds = config["runtime"]["failure_checks"]["thresholds"]
    checks = {
        "reader_uses_memory": {
            "passed": (
                float(by_variant["zero_memory"]["query_loss"]) - float(by_variant["base"]["query_loss"])
                >= float(thresholds["reader_memory_loss_gap_min"])
            ),
            "observed_loss_gap": float(by_variant["zero_memory"]["query_loss"]) - float(by_variant["base"]["query_loss"]),
            "threshold": float(thresholds["reader_memory_loss_gap_min"]),
            "base_loss": float(by_variant["base"]["query_loss"]),
            "ablation_loss": float(by_variant["zero_memory"]["query_loss"]),
        },
        "writer_beats_noise": {
            "passed": (
                float(by_variant["writer_noise"]["query_loss"]) - float(by_variant["base"]["query_loss"])
                >= float(thresholds["writer_noise_loss_gap_min"])
            ),
            "observed_loss_gap": float(by_variant["writer_noise"]["query_loss"])
            - float(by_variant["base"]["query_loss"]),
            "threshold": float(thresholds["writer_noise_loss_gap_min"]),
            "base_loss": float(by_variant["base"]["query_loss"]),
            "ablation_loss": float(by_variant["writer_noise"]["query_loss"]),
        },
        "fuser_avoids_collapse": {
            "passed": (
                float(by_variant["base"]["mean_short_slot_diversity"])
                >= float(thresholds["fuser_slot_diversity_min"])
                and (
                    float(by_variant["collapsed_fuser"]["query_loss"]) - float(by_variant["base"]["query_loss"])
                    >= float(thresholds["fuser_collapse_loss_gap_min"])
                )
            ),
            "observed_slot_diversity": float(by_variant["base"]["mean_short_slot_diversity"]),
            "slot_diversity_threshold": float(thresholds["fuser_slot_diversity_min"]),
            "observed_loss_gap": float(by_variant["collapsed_fuser"]["query_loss"])
            - float(by_variant["base"]["query_loss"]),
            "loss_gap_threshold": float(thresholds["fuser_collapse_loss_gap_min"]),
            "base_loss": float(by_variant["base"]["query_loss"]),
            "ablation_loss": float(by_variant["collapsed_fuser"]["query_loss"]),
        },
    }

    variant_csv = output_dir / "failure_ablation_summary.csv"
    variant_svg = output_dir / "failure_ablation_summary.svg"
    _write_variant_csv(variant_csv, variant_rows)
    _write_variant_svg(variant_svg, variant_rows)
    profile_metrics = profiler.finalize()
    metrics = {
        "mode": "analysis_failure_checks",
        "analysis_mode": "m3_failure_checks",
        "query_learning_mode": query_learning_mode,
        "num_examples": len(examples),
        "checks_total": len(checks),
        "checks_passed": sum(int(item["passed"]) for item in checks.values()),
        "checks_pass_rate": sum(int(item["passed"]) for item in checks.values()) / len(checks),
        "base_query_loss": float(by_variant["base"]["query_loss"]),
        "base_query_accuracy": float(by_variant["base"]["query_accuracy"]),
        "base_short_slot_diversity": float(by_variant["base"]["mean_short_slot_diversity"]),
        "zero_memory_query_loss": float(by_variant["zero_memory"]["query_loss"]),
        "writer_noise_query_loss": float(by_variant["writer_noise"]["query_loss"]),
        "writer_noise_trials": writer_noise_trials,
        "collapsed_fuser_query_loss": float(by_variant["collapsed_fuser"]["query_loss"]),
        "failure_ablation_summary_csv": str(variant_csv.resolve()),
        "failure_ablation_summary_svg": str(variant_svg.resolve()),
        "writer_checkpoint": str(writer_path.resolve()),
        "queries_meta_init": str(queries_path.resolve()),
        "target_domain": manifest["target_domain"],
        "dataset_sha256": manifest["dataset_sha256"],
        **profile_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "failure_checks.json",
        {
            "checks": checks,
            "variants": variant_rows,
        },
    )
    return metrics
