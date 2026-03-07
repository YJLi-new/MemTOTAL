from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from memtotal.analysis.reporting import write_sanity_plot, write_summary_csv
from memtotal.models import BackboneWrapper, MemoryWriter
from memtotal.tasks import load_task_dataset
from memtotal.training.m3 import _resolve_artifact_path
from memtotal.training.m4_shared_injection import (
    FEVER_LABEL_ORDER,
    FEVER_PROMPT_VARIANTS,
    FEVER_SUPPORT_SERIALIZATION_VARIANTS,
    SharedInjectionPilotRuntime,
    _build_example_caches,
    _build_support_text_block,
    _classification_metrics_from_rows,
    _evaluate_examples,
)
from memtotal.training.m4_shared_injection import _load_task_dataset_with_path
from memtotal.utils.io import write_json, write_jsonl


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_case_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    case_rows = [
        json.loads(line)
        for line in (run_dir / "task_case_dump.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return metrics, case_rows


def _collect_shared_injection_runs(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    runs: dict[str, tuple[dict[str, Any], Path]] = {}
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("training_stage") != "shared_injection_pilot":
            continue
        alias = str(metrics.get("pilot_arm_alias", "")).strip()
        if not alias:
            continue
        runs[alias] = (metrics, metrics_path.parent)
    return runs


def _build_writer(config: dict[str, Any], resume: str, seed: int) -> tuple[BackboneWrapper, MemoryWriter]:
    backbone_cfg = config["backbone"]
    runtime_device = str(config["runtime"].get("device", "cpu"))
    backbone_hidden_size = backbone_cfg.get("stub_hidden_size")
    backbone = BackboneWrapper(
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
    writer_cfg = config["method"]["writer"]
    writer = MemoryWriter(
        embed_dim=backbone.hidden_size,
        memory_slots=int(writer_cfg["memory_slots"]),
        arch=str(writer_cfg.get("arch", "mlp")),
        hidden_dim=writer_cfg.get("hidden_dim"),
        num_heads=int(writer_cfg.get("num_heads", 4)),
        transformer_layers=int(writer_cfg.get("transformer_layers", 1)),
        dropout=float(writer_cfg.get("dropout", 0.0)),
    )
    writer.load_from(_resolve_artifact_path(resume, "writer.ckpt"), map_location="cpu")
    writer.to(backbone.device)
    writer.eval()
    for parameter in writer.parameters():
        parameter.requires_grad_(False)
    return backbone, writer


def _example_text(example: dict[str, Any]) -> str:
    claim = str(example.get("claim", "")).strip()
    evidence = str(example.get("evidence", "")).strip()
    return f"Claim: {claim} || Evidence: {evidence}"


def _extract_writer_features(
    *,
    backbone: BackboneWrapper,
    writer: MemoryWriter,
    examples: list[dict[str, Any]],
    example_lookup: dict[str, dict[str, Any]],
    mode: str,
) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(
            len(examples),
            writer.memory_slots * writer.embed_dim,
            dtype=torch.float32,
        )
    texts: list[str] = []
    for example in examples:
        if mode == "real":
            source = example
        else:
            shuffled_id = str(example.get("shuffled_memory_example_id", "")).strip()
            if not shuffled_id:
                raise ValueError(
                    f"Example {example['id']} is missing shuffled_memory_example_id for writer information audit."
                )
            source = example_lookup[shuffled_id]
        texts.append(_example_text(source))
    with torch.inference_mode():
        states = backbone.summarize_texts(texts)
        memory = writer.write(states).detach().to(dtype=torch.float32).cpu()
    return memory.flatten(start_dim=1)


def _standardize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            rank_sum += rank
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _macro_f1(y_true: list[int], y_pred: list[int], labels: list[int]) -> float:
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
    return float(sum(f1_values) / max(1, len(f1_values)))


def _balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    positives = [index for index, label in enumerate(y_true) if label == 1]
    negatives = [index for index, label in enumerate(y_true) if label == 0]
    if not positives or not negatives:
        return float("nan")
    tpr = sum(int(y_pred[index] == 1) for index in positives) / len(positives)
    tnr = sum(int(y_pred[index] == 0) for index in negatives) / len(negatives)
    return float((tpr + tnr) / 2.0)


def _build_probe_model(
    *,
    input_dim: int,
    output_dim: int,
    model_kind: str,
) -> nn.Module:
    if model_kind == "linear":
        return nn.Linear(input_dim, output_dim)
    if model_kind == "mlp":
        hidden_dim = min(128, max(32, input_dim // 8))
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    raise ValueError(f"Unsupported probe model_kind={model_kind}")


def _fit_probe_cv(
    *,
    features: torch.Tensor,
    targets: list[int],
    task_kind: str,
    model_kind: str,
    seed: int,
    labels: list[int] | None = None,
) -> dict[str, float]:
    if len(targets) != features.shape[0]:
        raise ValueError("features and targets must have the same number of rows.")
    indices = list(range(features.shape[0]))
    num_folds = min(4, len(indices))
    fold_splits = [indices[fold::num_folds] for fold in range(num_folds)]
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    balanced_accuracies: list[float] = []
    aucs: list[float] = []
    for fold_index, eval_indices in enumerate(fold_splits):
        train_indices = [index for index in indices if index not in eval_indices]
        train_x = features[train_indices]
        eval_x = features[eval_indices]
        train_y = torch.tensor([targets[index] for index in train_indices], dtype=torch.long)
        eval_y = torch.tensor([targets[index] for index in eval_indices], dtype=torch.long)
        train_x, eval_x = _standardize(train_x, eval_x)
        output_dim = len(labels) if task_kind == "multiclass" else 1
        model = _build_probe_model(
            input_dim=int(features.shape[1]),
            output_dim=output_dim,
            model_kind=model_kind,
        )
        torch.manual_seed(seed + fold_index)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            if task_kind == "multiclass":
                loss = nn.functional.cross_entropy(logits, train_y)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),
                    train_y.to(dtype=torch.float32),
                )
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            logits = model(eval_x)
            if task_kind == "multiclass":
                predictions = torch.argmax(logits, dim=-1)
                predicted = predictions.tolist()
                gold = eval_y.tolist()
                accuracies.append(float((predictions == eval_y).to(dtype=torch.float32).mean().item()))
                macro_f1s.append(_macro_f1(gold, predicted, labels=labels or sorted(set(targets))))
            else:
                probabilities = torch.sigmoid(logits.squeeze(-1))
                predictions = (probabilities >= 0.5).to(dtype=torch.long)
                predicted = predictions.tolist()
                gold = eval_y.tolist()
                accuracies.append(float((predictions == eval_y).to(dtype=torch.float32).mean().item()))
                balanced_accuracies.append(_balanced_accuracy(gold, predicted))
                aucs.append(_binary_auc(probabilities.tolist(), gold))
    auc_values = [value for value in aucs if not math.isnan(value)]
    bal_values = [value for value in balanced_accuracies if not math.isnan(value)]
    return {
        "accuracy": sum(accuracies) / max(1, len(accuracies)),
        "macro_f1": float("nan") if not macro_f1s else sum(macro_f1s) / len(macro_f1s),
        "balanced_accuracy": float("nan") if not bal_values else sum(bal_values) / len(bal_values),
        "auroc": float("nan") if not auc_values else sum(auc_values) / len(auc_values),
    }


def _label_recall_count(metrics: dict[str, Any]) -> int:
    recalls = metrics.get("label_recall_by_class", {})
    return sum(int(float(value) >= 0.10) for value in recalls.values())


def run_m4_phase0_gate_sweep(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> None:
    support_dataset_path = str(config["task"]["support_dataset_path"])
    support_examples = _load_task_dataset_with_path(config, support_dataset_path)
    eval_examples = load_task_dataset(config)
    if dry_run:
        eval_examples = eval_examples[: min(24, len(eval_examples))]
    example_lookup = {str(example["id"]): dict(example) for example in [*support_examples, *eval_examples]}
    base_runtime = SharedInjectionPilotRuntime(
        config=config,
        seed=int(config["runtime"].get("phase0_seed", 0)),
        arm="base_only",
        writer_memory_control="real",
    )
    teacher_runtime = SharedInjectionPilotRuntime(
        config=config,
        seed=int(config["runtime"].get("phase0_seed", 0)) + 1,
        arm="teacher_text",
        writer_memory_control="real",
    )
    arm_summary_rows: list[dict[str, Any]] = []
    arm_case_dump_dir = output_dir / "arm_case_dumps"
    arm_case_dump_dir.mkdir(parents=True, exist_ok=True)
    selected_a_rows: list[dict[str, Any]] = []
    selected_t_rows: list[dict[str, Any]] = []
    for prompt_variant in FEVER_PROMPT_VARIANTS:
        eval_caches = _build_example_caches(eval_examples, prompt_variant=prompt_variant)
        a_case_rows = _evaluate_examples(
            runtime=base_runtime,
            eval_examples=eval_caches,
            arm_alias=f"A::{prompt_variant}",
            arm="base_only",
            writer_memory_control="real",
            prompt_variant=prompt_variant,
            support_serialization_variant="none",
            support_text_block="",
            teacher_support_text_block="",
            prefix_embeddings=None,
            profiler=None,
        )
        a_metrics = _classification_metrics_from_rows(a_case_rows)
        arm_summary_rows.append(
            {
                "arm_type": "A",
                "prompt_variant": prompt_variant,
                "support_serialization_variant": "none",
                **a_metrics,
            }
        )
        write_jsonl(arm_case_dump_dir / f"A__{prompt_variant}.jsonl", a_case_rows)
        for support_variant in FEVER_SUPPORT_SERIALIZATION_VARIANTS:
            support_text_block = _build_support_text_block(
                support_examples,
                memory_control="real",
                example_lookup=example_lookup,
                support_serialization_variant=support_variant,
            )
            t_case_rows = _evaluate_examples(
                runtime=teacher_runtime,
                eval_examples=eval_caches,
                arm_alias=f"T::{prompt_variant}::{support_variant}",
                arm="teacher_text",
                writer_memory_control="real",
                prompt_variant=prompt_variant,
                support_serialization_variant=support_variant,
                support_text_block=support_text_block,
                teacher_support_text_block=support_text_block,
                prefix_embeddings=None,
                profiler=None,
            )
            t_metrics = _classification_metrics_from_rows(t_case_rows)
            arm_summary_rows.append(
                {
                    "arm_type": "T",
                    "prompt_variant": prompt_variant,
                    "support_serialization_variant": support_variant,
                    **t_metrics,
                }
            )
            write_jsonl(
                arm_case_dump_dir / f"T__{prompt_variant}__{support_variant}.jsonl",
                t_case_rows,
            )

    a_rows = [row for row in arm_summary_rows if row["arm_type"] == "A"]
    a_rows_sorted = sorted(
        a_rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["accuracy"]),
            -float(row["dominant_label_fraction"]),
        ),
        reverse=True,
    )
    a_winner = dict(a_rows_sorted[0])
    a_winner["valid_prompt_surface"] = bool(
        float(a_winner["dominant_label_fraction"]) <= 0.85
        and _label_recall_count(a_winner) >= 2
    )
    t_rows = [
        row
        for row in arm_summary_rows
        if row["arm_type"] == "T" and row["prompt_variant"] == a_winner["prompt_variant"]
    ]
    t_rows_sorted = sorted(
        t_rows,
        key=lambda row: (float(row["macro_f1"]), float(row["accuracy"])),
        reverse=True,
    )
    t_winner = dict(t_rows_sorted[0])
    phase0_gate_passed = bool(
        a_winner["valid_prompt_surface"]
        and (float(t_winner["accuracy"]) - float(a_winner["accuracy"])) >= (2.0 / max(1, len(eval_examples)))
        and (float(t_winner["macro_f1"]) - float(a_winner["macro_f1"])) >= 0.05
        and float(t_winner["dominant_label_fraction"]) <= float(a_winner["dominant_label_fraction"])
    )
    summary_csv = output_dir / "phase0_summary.csv"
    summary_plot = output_dir / "phase0_summary.svg"
    write_summary_csv(
        summary_csv,
        [
            {
                "mode": f"{row['arm_type']}:{row['prompt_variant']}:{row['support_serialization_variant']}",
                "run_dir": output_dir,
                "primary_metric": "macro_f1",
                "primary_score": float(row["macro_f1"]),
                "accuracy": float(row["accuracy"]),
                "dominant_label_fraction": float(row["dominant_label_fraction"]),
            }
            for row in arm_summary_rows
        ],
    )
    write_sanity_plot(
        summary_plot,
        [
            {
                "mode": f"{row['arm_type']}:{row['prompt_variant']}:{row['support_serialization_variant']}",
                "run_dir": f"{row['arm_type']}-{row['prompt_variant']}-{row['support_serialization_variant']}",
                "primary_metric": "macro_f1",
                "primary_score": float(row["macro_f1"]),
            }
            for row in arm_summary_rows
        ],
    )
    _write_csv(output_dir / "phase0_arm_summary.csv", arm_summary_rows)
    report_lines = [
        "# M4 Phase 0 Gate Sweep",
        "",
        f"- phase0_gate_passed: {phase0_gate_passed}",
        f"- a_winner_prompt_variant: {a_winner['prompt_variant']}",
        f"- t_winner_support_serialization: {t_winner['support_serialization_variant']}",
        "",
        "## A Winner",
        (
            f"- macro_f1={float(a_winner['macro_f1']):.4f}, accuracy={float(a_winner['accuracy']):.4f}, "
            f"dominant_label_fraction={float(a_winner['dominant_label_fraction']):.4f}, "
            f"label_recall_by_class={json.dumps(a_winner['label_recall_by_class'], sort_keys=True)}"
        ),
        "",
        "## T Winner",
        (
            f"- macro_f1={float(t_winner['macro_f1']):.4f}, accuracy={float(t_winner['accuracy']):.4f}, "
            f"dominant_label_fraction={float(t_winner['dominant_label_fraction']):.4f}, "
            f"label_recall_by_class={json.dumps(t_winner['label_recall_by_class'], sort_keys=True)}"
        ),
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "m4_phase0_gate_sweep",
            "phase0_gate_passed": phase0_gate_passed,
            "a_winner": a_winner,
            "t_winner": t_winner,
            "selected_prompt_variant": str(a_winner["prompt_variant"]),
            "selected_support_serialization": str(t_winner["support_serialization_variant"]),
            "summary_csv": str(summary_csv.resolve()),
            "summary_plot": str(summary_plot.resolve()),
            "report_path": str((output_dir / "report.md").resolve()),
        },
    )


def run_m4_writer_information_audit(
    *,
    config: dict[str, Any],
    output_dir: Path,
    input_root: str,
    resume: str | None,
    dry_run: bool,
) -> None:
    if resume is None:
        raise ValueError("Writer information audit requires --resume pointing at a writer checkpoint root.")
    phase0_metrics = json.loads((Path(input_root).resolve() / "metrics.json").read_text())
    phase0_gate_passed = bool(phase0_metrics.get("phase0_gate_passed", False))
    full_examples = load_task_dataset(config)
    examples = full_examples[: min(32, len(full_examples))] if dry_run else list(full_examples)
    example_lookup = {str(example["id"]): example for example in full_examples}
    backbone, writer = _build_writer(config, resume, seed=int(config["runtime"].get("probe_seed", 0)))
    feature_modes = {
        "real": _extract_writer_features(
            backbone=backbone,
            writer=writer,
            examples=examples,
            example_lookup=example_lookup,
            mode="real",
        ),
        "shuffle": _extract_writer_features(
            backbone=backbone,
            writer=writer,
            examples=examples,
            example_lookup=example_lookup,
            mode="shuffled",
        ),
        "zero": _extract_writer_features(
            backbone=backbone,
            writer=writer,
            examples=examples,
            example_lookup=example_lookup,
            mode="zero",
        ),
    }

    label_names = list(FEVER_LABEL_ORDER)
    label_to_index = {label: index for index, label in enumerate(label_names)}
    verifiability_targets = [int(str(example["label"]) == "NOT_ENOUGH_INFO") for example in examples]
    polarity_indices = [index for index, example in enumerate(examples) if str(example["label"]) != "NOT_ENOUGH_INFO"]
    polarity_targets = [
        int(str(examples[index]["label"]) == "SUPPORTS")
        for index in polarity_indices
    ]

    probe_specs = {
        "label_probe_3way": {
            "task_kind": "multiclass",
            "targets": [label_to_index[str(example["label"])] for example in examples],
            "labels": list(range(len(label_names))),
            "primary_metric": "macro_f1",
        },
        "verifiability_probe": {
            "task_kind": "binary",
            "targets": verifiability_targets,
            "labels": None,
            "primary_metric": "auroc",
        },
        "polarity_probe": {
            "task_kind": "binary",
            "targets": polarity_targets,
            "labels": None,
            "primary_metric": "auroc",
            "subset_indices": polarity_indices,
        },
    }

    rows: list[dict[str, Any]] = []
    base_seed = int(config["runtime"].get("probe_seed", 0))
    for probe_offset, (probe_name, probe_spec) in enumerate(probe_specs.items()):
        for mode_offset, (feature_mode, features) in enumerate(feature_modes.items()):
            active_features = features
            active_targets = probe_spec["targets"]
            if "subset_indices" in probe_spec:
                subset_indices = list(probe_spec["subset_indices"])
                active_features = features[subset_indices]
                active_targets = [int(str(examples[index]["label"]) == "SUPPORTS") for index in subset_indices]
            for model_offset, model_kind in enumerate(("linear", "mlp")):
                probe_metrics = _fit_probe_cv(
                    features=active_features,
                    targets=active_targets,
                    task_kind=str(probe_spec["task_kind"]),
                    model_kind=model_kind,
                    seed=base_seed + (100 * probe_offset) + (10 * mode_offset) + model_offset,
                    labels=probe_spec.get("labels"),
                )
                rows.append(
                    {
                        "probe_name": probe_name,
                        "feature_mode": feature_mode,
                        "model_kind": model_kind,
                        "primary_metric": str(probe_spec["primary_metric"]),
                        "accuracy": probe_metrics["accuracy"],
                        "macro_f1": probe_metrics["macro_f1"],
                        "balanced_accuracy": probe_metrics["balanced_accuracy"],
                        "auroc": probe_metrics["auroc"],
                    }
                )

    def best_metric(probe_name: str, feature_mode: str, metric_name: str) -> float:
        values = [
            float(row[metric_name])
            for row in rows
            if row["probe_name"] == probe_name and row["feature_mode"] == feature_mode and not math.isnan(float(row[metric_name]))
        ]
        return max(values, default=float("-inf"))

    label_real = best_metric("label_probe_3way", "real", "macro_f1")
    label_control = max(
        best_metric("label_probe_3way", "shuffle", "macro_f1"),
        best_metric("label_probe_3way", "zero", "macro_f1"),
    )
    verif_real = best_metric("verifiability_probe", "real", "auroc")
    verif_control = max(
        best_metric("verifiability_probe", "shuffle", "auroc"),
        best_metric("verifiability_probe", "zero", "auroc"),
    )
    polarity_real = best_metric("polarity_probe", "real", "auroc")
    polarity_control = max(
        best_metric("polarity_probe", "shuffle", "auroc"),
        best_metric("polarity_probe", "zero", "auroc"),
    )

    label_probe_passed = bool(
        label_real >= float(config["runtime"].get("audit_label_macro_f1_min", 0.40))
        and (label_real - label_control) >= float(config["runtime"].get("audit_label_macro_f1_gap", 0.05))
    )
    semantic_probe_passed = bool(
        (
            verif_real >= float(config["runtime"].get("audit_binary_auroc_min", 0.60))
            and (verif_real - verif_control) >= float(config["runtime"].get("audit_binary_auroc_gap", 0.05))
        )
        or (
            polarity_real >= float(config["runtime"].get("audit_binary_auroc_min", 0.60))
            and (polarity_real - polarity_control) >= float(config["runtime"].get("audit_binary_auroc_gap", 0.05))
        )
    )
    phase1_probe_passed = bool(label_probe_passed and semantic_probe_passed)
    phase1_gate_passed = bool(phase0_gate_passed and phase1_probe_passed)

    summary_csv = output_dir / "summary.csv"
    summary_plot = output_dir / "summary.svg"
    probe_csv = output_dir / "probe_results.csv"
    write_summary_csv(
        summary_csv,
        [
            {
                "mode": f"{row['probe_name']}:{row['feature_mode']}:{row['model_kind']}",
                "run_dir": output_dir,
                "primary_metric": str(row["primary_metric"]),
                "primary_score": float(row[str(row["primary_metric"])]),
            }
            for row in rows
        ],
    )
    write_sanity_plot(
        summary_plot,
        [
            {
                "mode": f"{row['probe_name']}:{row['feature_mode']}:{row['model_kind']}",
                "run_dir": f"{row['probe_name']}-{row['feature_mode']}-{row['model_kind']}",
                "primary_metric": str(row["primary_metric"]),
                "primary_score": float(row[str(row["primary_metric"])]),
            }
            for row in rows
        ],
    )
    _write_csv(probe_csv, rows)
    report_lines = [
        "# M4 Writer Information Audit",
        "",
        f"- phase0_gate_passed: {phase0_gate_passed}",
        f"- label_probe_passed: {label_probe_passed}",
        f"- semantic_probe_passed: {semantic_probe_passed}",
        f"- phase1_probe_passed: {phase1_probe_passed}",
        f"- phase1_gate_passed: {phase1_gate_passed}",
        "",
        f"- label_real_macro_f1: {label_real:.4f}",
        f"- label_control_macro_f1: {label_control:.4f}",
        f"- verifiability_real_auroc: {verif_real:.4f}",
        f"- verifiability_control_auroc: {verif_control:.4f}",
        f"- polarity_real_auroc: {polarity_real:.4f}",
        f"- polarity_control_auroc: {polarity_control:.4f}",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "m4_writer_information_audit",
            "phase0_gate_passed": phase0_gate_passed,
            "label_probe_passed": label_probe_passed,
            "semantic_probe_passed": semantic_probe_passed,
            "phase1_probe_passed": phase1_probe_passed,
            "phase1_gate_passed": phase1_gate_passed,
            "probe_results_csv": str(probe_csv.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "summary_plot": str(summary_plot.resolve()),
            "report_path": str((output_dir / "report.md").resolve()),
        },
    )


def _pairwise_compare(left_rows: dict[str, dict[str, Any]], right_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    shared_ids = sorted(set(left_rows) & set(right_rows))
    if not shared_ids:
        raise ValueError("Pairwise compare requires overlapping example ids.")
    left_correct_to_right_wrong = 0
    left_wrong_to_right_correct = 0
    margin_gains: list[float] = []
    for example_id in shared_ids:
        left_row = left_rows[example_id]
        right_row = right_rows[example_id]
        left_correct = bool(left_row["predicted_correct"])
        right_correct = bool(right_row["predicted_correct"])
        if left_correct and not right_correct:
            left_correct_to_right_wrong += 1
        if not left_correct and right_correct:
            left_wrong_to_right_correct += 1
        margin_gains.append(float(right_row["final_margin"]) - float(left_row["final_margin"]))
    return {
        "example_count": len(shared_ids),
        "left_wrong_to_right_correct": left_wrong_to_right_correct,
        "left_correct_to_right_wrong": left_correct_to_right_wrong,
        "flip_count_delta": left_wrong_to_right_correct - left_correct_to_right_wrong,
        "mean_margin_gain": sum(margin_gains) / max(1, len(margin_gains)),
    }


def run_m4_shared_injection_compare(
    *,
    config: dict[str, Any],
    output_dir: Path,
    input_root: str,
    dry_run: bool,
) -> None:
    del dry_run
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _collect_shared_injection_runs(Path(input_root).resolve())
    required = {"A", "T", "I_real", "I_shuffle", "I_zero"}
    missing = sorted(required - set(runs))
    if missing:
        raise ValueError(f"M4 shared injection compare is missing run aliases: {', '.join(missing)}")
    arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
    arm_summaries: list[dict[str, Any]] = []
    for alias in sorted(required):
        metrics, run_dir = runs[alias]
        _, case_rows = _load_case_rows(run_dir)
        class_metrics = _classification_metrics_from_rows(case_rows)
        arm_rows[alias] = {str(row["example_id"]): row for row in case_rows}
        arm_summaries.append(
            {
                "mode": alias,
                "run_dir": str(run_dir.resolve()),
                "primary_metric": str(metrics.get("task_metric_name", "accuracy")),
                "primary_score": float(metrics.get("best_adapt_task_score", 0.0)),
                "macro_f1": float(class_metrics["macro_f1"]),
                "mean_margin": float(class_metrics["mean_margin"]),
                "dominant_label_fraction": float(class_metrics["dominant_label_fraction"]),
                "case_rows": len(case_rows),
            }
        )
    pairwise_rows: list[dict[str, Any]] = []
    for left_alias, right_alias in [
        ("A", "T"),
        ("A", "I_real"),
        ("I_shuffle", "I_real"),
        ("I_zero", "I_real"),
    ]:
        compare = _pairwise_compare(arm_rows[left_alias], arm_rows[right_alias])
        pairwise_rows.append({"left_alias": left_alias, "right_alias": right_alias, **compare})
    pairwise_lookup = {(row["left_alias"], row["right_alias"]): row for row in pairwise_rows}
    real_vs_shuffle_path = output_dir / "real_vs_shuffle_gap.csv"
    real_vs_zero_path = output_dir / "real_vs_zero_gap.csv"
    _write_csv(
        real_vs_shuffle_path,
        [
            {
                "example_id": example_id,
                "real_margin": float(arm_rows["I_real"][example_id]["final_margin"]),
                "shuffle_margin": float(arm_rows["I_shuffle"][example_id]["final_margin"]),
                "margin_gap": float(arm_rows["I_real"][example_id]["final_margin"])
                - float(arm_rows["I_shuffle"][example_id]["final_margin"]),
            }
            for example_id in sorted(set(arm_rows["I_real"]) & set(arm_rows["I_shuffle"]))
        ],
    )
    _write_csv(
        real_vs_zero_path,
        [
            {
                "example_id": example_id,
                "real_margin": float(arm_rows["I_real"][example_id]["final_margin"]),
                "zero_margin": float(arm_rows["I_zero"][example_id]["final_margin"]),
                "margin_gap": float(arm_rows["I_real"][example_id]["final_margin"])
                - float(arm_rows["I_zero"][example_id]["final_margin"]),
            }
            for example_id in sorted(set(arm_rows["I_real"]) & set(arm_rows["I_zero"]))
        ],
    )
    summary_csv = output_dir / "arm_summary.csv"
    summary_plot = output_dir / "summary.svg"
    pairwise_csv = output_dir / "arm_pairwise_compare.csv"
    write_summary_csv(summary_csv, arm_summaries)
    write_sanity_plot(summary_plot, arm_summaries)
    _write_csv(pairwise_csv, pairwise_rows)
    summary_lookup = {row["mode"]: row for row in arm_summaries}
    regressions_vs_base = int(pairwise_lookup[("A", "I_real")]["left_correct_to_right_wrong"])
    gate_passed = bool(
        int(pairwise_lookup[("I_shuffle", "I_real")]["flip_count_delta"]) >= 2
        and (float(summary_lookup["I_real"]["macro_f1"]) - float(summary_lookup["I_shuffle"]["macro_f1"])) >= 0.05
        and int(pairwise_lookup[("I_zero", "I_real")]["flip_count_delta"]) >= 2
        and (float(summary_lookup["I_real"]["macro_f1"]) - float(summary_lookup["I_zero"]["macro_f1"])) >= 0.05
        and float(summary_lookup["I_real"]["primary_score"]) >= float(summary_lookup["A"]["primary_score"])
        and regressions_vs_base <= 1
    )
    report_lines = [
        "# M4 Shared Injection Compare",
        "",
        f"- gate_passed: {gate_passed}",
        f"- regressions_vs_base: {regressions_vs_base}",
        "",
        "## Arm Summary",
    ]
    for row in arm_summaries:
        report_lines.append(
            f"- {row['mode']}: task_score={row['primary_score']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, mean_margin={row['mean_margin']:.4f}, "
            f"dominant_label_fraction={row['dominant_label_fraction']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## Pairwise Compare")
    for row in pairwise_rows:
        report_lines.append(
            f"- {row['left_alias']} -> {row['right_alias']}: "
            f"flip_delta={row['flip_count_delta']}, "
            f"left_wrong_to_right_correct={row['left_wrong_to_right_correct']}, "
            f"left_correct_to_right_wrong={row['left_correct_to_right_wrong']}, "
            f"mean_margin_gain={row['mean_margin_gain']:.4f}"
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "m4_shared_injection_compare",
            "gate_passed": gate_passed,
            "regressions_vs_base": regressions_vs_base,
            "summary_csv": str(summary_csv.resolve()),
            "summary_plot": str(summary_plot.resolve()),
            "pairwise_compare_csv": str(pairwise_csv.resolve()),
            "real_vs_shuffle_gap_csv": str(real_vs_shuffle_path.resolve()),
            "real_vs_zero_gap_csv": str(real_vs_zero_path.resolve()),
            "report_path": str((output_dir / "report.md").resolve()),
            "task_score_gap_to_T": float(summary_lookup["I_real"]["primary_score"]) - float(summary_lookup["T"]["primary_score"]),
            "macro_f1_gap_to_T": float(summary_lookup["I_real"]["macro_f1"]) - float(summary_lookup["T"]["macro_f1"]),
            "mean_margin_gap_to_T": float(summary_lookup["I_real"]["mean_margin"]) - float(summary_lookup["T"]["mean_margin"]),
        },
    )
