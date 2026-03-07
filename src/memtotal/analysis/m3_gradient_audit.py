from __future__ import annotations

import csv
from math import ceil
from pathlib import Path

import torch

from memtotal.data import split_target_domain_examples
from memtotal.pipeline import MemoryRuntime
from memtotal.training.m3 import (
    _build_meta_context,
    _classification_loss,
    _configure_stage_c_trainables,
    _continuation_retrieval_loss,
    _count_unique_parameters,
    _resolve_artifact_path,
    _resolve_expected_stage_c_query_learning_mode,
    _resolve_retrieval_negative_count,
    _resolve_stage_c_adaptation_target,
)
from memtotal.utils.io import write_json
from memtotal.utils.profiling import ProfileTracker


def _parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> tuple[float, float]:
    total = 0.0
    max_abs = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(grad.pow(2).sum().item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
    return total ** 0.5, max_abs


def _write_gradient_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["module", "grad_norm", "grad_max_abs"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_gradient_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    max_value = max(float(row["grad_norm"]) for row in rows) or 1.0
    width = 760
    height = 120 + 54 * len(rows)
    left = 180
    bar_width = 420
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf7' />",
        "<text x='24' y='34' font-size='20' font-family='monospace'>M3 Stage C gradient audit</text>",
        "<text x='24' y='56' font-size='13' font-family='monospace'>support-loss grad_norm by module</text>",
    ]
    palette = {
        "queries": "#2f5aa8",
        "reader_non_query": "#7a4fa3",
        "fuser": "#2b8a3e",
        "writer": "#b5651d",
    }
    for index, row in enumerate(rows):
        top = 82 + index * 46
        value = float(row["grad_norm"])
        scaled = ceil((value / max_value) * bar_width) if max_value > 0 else 0
        parts.append(
            f"<text x='24' y='{top + 18}' font-size='13' font-family='monospace'>{row['module']}</text>"
        )
        parts.append(f"<rect x='{left}' y='{top}' width='{bar_width}' height='24' fill='#efe6d0' rx='4' />")
        if scaled > 0:
            parts.append(
                f"<rect x='{left}' y='{top}' width='{scaled}' height='24' fill='{palette.get(str(row['module']), '#4a6fa5')}' rx='4' />"
            )
        parts.append(
            f"<text x='{left + bar_width + 12}' y='{top + 18}' font-size='13' font-family='monospace'>{value:.6e}</text>"
        )
    parts.append("</svg>")
    path.write_text("".join(parts))


def run_m3_stage_c_gradient_audit(
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
            f"Gradient audit expected query_learning_mode={expected_query_learning_mode}, "
            f"but resume artifact provides {actual_query_learning_mode}."
        )

    adaptation_target = _resolve_stage_c_adaptation_target(config)
    adaptable_parameters, trainable_module = _configure_stage_c_trainables(runtime, adaptation_target)
    trainable_parameter_count = _count_unique_parameters(adaptable_parameters)
    runtime.writer.unfreeze()
    runtime.reader.unfreeze()
    runtime.fuser.unfreeze()
    runtime.zero_grad(set_to_none=True)

    target_episode = split_target_domain_examples(
        grouped_examples,
        target_domain=manifest["target_domain"],
        support_size=int(manifest["support_size"]),
        query_size=int(manifest["query_size"]),
        seed=seed,
        sampling_policy=str(manifest["sampling_policy"]),
    )
    if dry_run:
        target_episode = target_episode.__class__(
            support_examples=target_episode.support_examples[:1],
            query_examples=target_episode.query_examples[:1],
        )
    domain_examples = list(grouped_examples[manifest["target_domain"]])
    retrieval_negative_count = _resolve_retrieval_negative_count(config)
    query_objective = str(config["runtime"].get("query_objective", "label_prototype"))

    if query_objective == "label_prototype":
        candidate_states, candidate_labels = None, None
        from memtotal.training.m3 import _build_label_prototypes

        candidate_states, candidate_labels = _build_label_prototypes(runtime, domain_examples)
        support_loss = torch.stack(
            [
                _classification_loss(
                    runtime,
                    example,
                    candidate_states=candidate_states,
                    candidate_labels=candidate_labels,
                )
                for example in target_episode.support_examples
            ]
        ).mean()
    else:
        support_candidate_pool = list(target_episode.support_examples) or list(domain_examples)
        support_loss = torch.stack(
            [
                _continuation_retrieval_loss(
                    runtime,
                    example,
                    candidate_pool=support_candidate_pool,
                    negative_count=retrieval_negative_count,
                )[0]
                for example in target_episode.support_examples
            ]
        ).mean()
    support_loss.backward()

    reader_non_query_parameters = [
        parameter
        for name, parameter in runtime.reader.named_parameters()
        if name != "queries"
    ]
    rows = []
    for module_name, parameters in [
        ("queries", [runtime.reader.queries]),
        ("reader_non_query", reader_non_query_parameters),
        ("fuser", list(runtime.fuser.parameters())),
        ("writer", list(runtime.writer.parameters())),
    ]:
        grad_norm, grad_max_abs = _parameter_grad_norm(parameters)
        rows.append(
            {
                "module": module_name,
                "grad_norm": grad_norm,
                "grad_max_abs": grad_max_abs,
            }
        )

    csv_path = output_dir / "gradient_audit.csv"
    svg_path = output_dir / "gradient_audit.svg"
    _write_gradient_csv(csv_path, rows)
    _write_gradient_svg(svg_path, rows)

    profiler = ProfileTracker(output_dir=output_dir, device=str(config["runtime"]["device"]), event_name="analysis")
    for example in target_episode.support_examples:
        profiler.add_example()
        profiler.add_tokens(runtime.backbone.count_tokens(example["segment"]))
        profiler.add_tokens(runtime.backbone.count_tokens(example["continuation"]))
    profile_metrics = profiler.finalize()

    row_map = {str(row["module"]): row for row in rows}
    query_grad_norm = float(row_map["queries"]["grad_norm"])
    fuser_grad_norm = float(row_map["fuser"]["grad_norm"])
    writer_grad_norm = float(row_map["writer"]["grad_norm"])
    reader_non_query_grad_norm = float(row_map["reader_non_query"]["grad_norm"])
    metrics = {
        "mode": "analysis",
        "analysis_mode": "m3_stage_c_gradient_audit",
        "backbone": str(config["backbone"]["name"]),
        "query_learning_mode": actual_query_learning_mode,
        "query_objective": query_objective,
        "adaptation_target": adaptation_target,
        "trainable_module": trainable_module,
        "trainable_parameter_count": trainable_parameter_count,
        "target_domain": manifest["target_domain"],
        "support_loss": float(support_loss.item()),
        "queries_grad_norm": query_grad_norm,
        "queries_grad_max_abs": float(row_map["queries"]["grad_max_abs"]),
        "reader_non_query_grad_norm": reader_non_query_grad_norm,
        "reader_non_query_grad_max_abs": float(row_map["reader_non_query"]["grad_max_abs"]),
        "fuser_grad_norm": fuser_grad_norm,
        "fuser_grad_max_abs": float(row_map["fuser"]["grad_max_abs"]),
        "writer_grad_norm": writer_grad_norm,
        "writer_grad_max_abs": float(row_map["writer"]["grad_max_abs"]),
        "query_to_fuser_grad_ratio": query_grad_norm / max(fuser_grad_norm, 1e-12),
        "query_to_writer_grad_ratio": query_grad_norm / max(writer_grad_norm, 1e-12),
        "gradient_csv": str(csv_path.resolve()),
        "gradient_plot": str(svg_path.resolve()),
        **profile_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
