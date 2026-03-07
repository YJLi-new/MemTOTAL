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
from memtotal.utils.io import write_json


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


def _row_task_score(row: dict[str, Any]) -> float:
    if "task_score" in row:
        return float(row["task_score"])
    return float(bool(row.get("predicted_correct", False)))


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
) -> dict[str, float]:
    if len(targets) != features.shape[0]:
        raise ValueError("features and targets must have the same number of rows.")
    indices = list(range(features.shape[0]))
    num_folds = min(4, len(indices))
    fold_splits = [indices[fold::num_folds] for fold in range(num_folds)]
    accuracies: list[float] = []
    aucs: list[float] = []
    for fold_index, eval_indices in enumerate(fold_splits):
        train_indices = [index for index in indices if index not in eval_indices]
        train_x = features[train_indices]
        eval_x = features[eval_indices]
        train_y = torch.tensor([targets[index] for index in train_indices], dtype=torch.long)
        eval_y = torch.tensor([targets[index] for index in eval_indices], dtype=torch.long)
        train_x, eval_x = _standardize(train_x, eval_x)
        output_dim = 3 if task_kind == "multiclass" else 1
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
                accuracies.append(float((predictions == eval_y).to(dtype=torch.float32).mean().item()))
            else:
                probabilities = torch.sigmoid(logits.squeeze(-1))
                predictions = (probabilities >= 0.5).to(dtype=torch.long)
                accuracies.append(float((predictions == eval_y).to(dtype=torch.float32).mean().item()))
                aucs.append(_binary_auc(probabilities.tolist(), eval_y.tolist()))
    auc_values = [value for value in aucs if not math.isnan(value)]
    return {
        "accuracy": sum(accuracies) / max(1, len(accuracies)),
        "auroc": float("nan") if not auc_values else sum(auc_values) / len(auc_values),
    }


def _metric_value(row: dict[str, Any], *, target_name: str) -> float:
    if target_name == "teacher_gain_probe" and not math.isnan(float(row.get("auroc", float("nan")))):
        return float(row["auroc"])
    return float(row["accuracy"])


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
    runs = _collect_shared_injection_runs(Path(input_root).resolve())
    if "A" not in runs or "T" not in runs:
        raise ValueError("Writer information audit requires both A and T runs under --input_root.")
    a_metrics, a_run_dir = runs["A"]
    t_metrics, t_run_dir = runs["T"]
    _, a_rows = _load_case_rows(a_run_dir)
    _, t_rows = _load_case_rows(t_run_dir)
    a_lookup = {str(row["example_id"]): row for row in a_rows}
    t_lookup = {str(row["example_id"]): row for row in t_rows}

    full_examples = load_task_dataset(config)
    example_lookup = {str(example["id"]): example for example in full_examples}
    examples = full_examples[: min(16, len(full_examples))] if dry_run else list(full_examples)
    examples = [example for example in examples if str(example["id"]) in a_lookup and str(example["id"]) in t_lookup]
    backbone, writer = _build_writer(config, resume, seed=int(config["runtime"].get("probe_seed", 0)))
    feature_modes = {
        "real": _extract_writer_features(
            backbone=backbone,
            writer=writer,
            examples=examples,
            example_lookup=example_lookup,
            mode="real",
        ),
        "shuffled": _extract_writer_features(
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

    label_names = sorted({str(example["label"]) for example in examples})
    label_to_index = {label: index for index, label in enumerate(label_names)}
    targets_by_name: dict[str, tuple[list[int], str]] = {
        "label_probe": ([label_to_index[str(example["label"])] for example in examples], "multiclass"),
        "base_margin_sign_probe": (
            [int(float(a_lookup[str(example["id"])]["final_margin"]) < 0.0) for example in examples],
            "binary",
        ),
        "teacher_gain_probe": (
            [
                int(
                    (
                        _row_task_score(t_lookup[str(example["id"])])
                        > _row_task_score(a_lookup[str(example["id"])])
                    )
                    or (
                        float(t_lookup[str(example["id"])]["final_margin"])
                        > float(a_lookup[str(example["id"])]["final_margin"])
                    )
                )
                for example in examples
            ],
            "binary",
        ),
    }

    rows: list[dict[str, Any]] = []
    base_seed = int(config["runtime"].get("probe_seed", 0))
    for target_offset, (target_name, (targets, task_kind)) in enumerate(targets_by_name.items()):
        for mode_offset, (feature_mode, features) in enumerate(feature_modes.items()):
            for model_offset, model_kind in enumerate(("linear", "mlp")):
                probe_metrics = _fit_probe_cv(
                    features=features,
                    targets=targets,
                    task_kind=task_kind,
                    model_kind=model_kind,
                    seed=base_seed + (100 * target_offset) + (10 * mode_offset) + model_offset,
                )
                rows.append(
                    {
                        "target_name": target_name,
                        "task_kind": task_kind,
                        "feature_mode": feature_mode,
                        "model_kind": model_kind,
                        "accuracy": probe_metrics["accuracy"],
                        "auroc": probe_metrics["auroc"],
                    }
                )

    phase0_support_has_value = bool(
        float(t_metrics.get("best_adapt_task_score", 0.0)) > float(a_metrics.get("best_adapt_task_score", 0.0))
        or float(t_metrics.get("best_adapt_task_margin", 0.0))
        > float(a_metrics.get("best_adapt_task_margin", 0.0))
    )
    classification_gap = float(config["runtime"].get("audit_classification_min_gap", 0.05))
    teacher_auc_floor = float(config["runtime"].get("audit_teacher_gain_min_auc", 0.60))
    teacher_auc_gap = float(config["runtime"].get("audit_teacher_gain_auc_gap", 0.05))
    probe_gate_passed = False
    target_summaries: list[dict[str, Any]] = []
    for target_name in targets_by_name:
        target_rows = [row for row in rows if row["target_name"] == target_name]
        real_best = max((_metric_value(row, target_name=target_name) for row in target_rows if row["feature_mode"] == "real"), default=float("-inf"))
        control_best = max(
            (_metric_value(row, target_name=target_name) for row in target_rows if row["feature_mode"] in {"shuffled", "zero"}),
            default=float("-inf"),
        )
        target_summary = {
            "target_name": target_name,
            "best_real_metric": real_best,
            "best_control_metric": control_best,
            "metric_gap": real_best - control_best,
        }
        target_summaries.append(target_summary)
        if target_name == "teacher_gain_probe":
            if real_best >= teacher_auc_floor and (real_best - control_best) >= teacher_auc_gap:
                probe_gate_passed = True
        elif (real_best - control_best) >= classification_gap:
            probe_gate_passed = True

    phase1_gate_passed = bool(phase0_support_has_value and probe_gate_passed)
    probe_results_path = output_dir / "probe_results.csv"
    summary_path = output_dir / "summary.csv"
    plot_path = output_dir / "summary.svg"
    report_path = output_dir / "report.md"
    write_summary_csv(summary_path, rows)
    plot_rows = [
        {
            "mode": row["feature_mode"],
            "run_dir": f"{row['target_name']}-{row['model_kind']}",
            "primary_score": _metric_value(row, target_name=str(row["target_name"])),
            "primary_metric": "auroc" if str(row["target_name"]) == "teacher_gain_probe" else "accuracy",
        }
        for row in rows
    ]
    write_sanity_plot(plot_path, plot_rows)
    _write_csv(probe_results_path, rows)
    _write_csv(output_dir / "target_summaries.csv", target_summaries)
    report_lines = [
        "# M4 Writer Information Audit",
        "",
        f"- phase0_support_has_value: {phase0_support_has_value}",
        f"- probe_gate_passed: {probe_gate_passed}",
        f"- phase1_gate_passed: {phase1_gate_passed}",
        "",
        "## Target Summaries",
    ]
    for row in target_summaries:
        report_lines.append(
            f"- {row['target_name']}: best_real={row['best_real_metric']:.4f}, "
            f"best_control={row['best_control_metric']:.4f}, gap={row['metric_gap']:.4f}"
        )
    report_path.write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "m4_writer_information_audit",
            "base_run_dir": str(a_run_dir.resolve()),
            "teacher_run_dir": str(t_run_dir.resolve()),
            "base_task_score": float(a_metrics.get("best_adapt_task_score", 0.0)),
            "teacher_task_score": float(t_metrics.get("best_adapt_task_score", 0.0)),
            "base_margin": float(a_metrics.get("best_adapt_task_margin", 0.0)),
            "teacher_margin": float(t_metrics.get("best_adapt_task_margin", 0.0)),
            "phase0_support_has_value": phase0_support_has_value,
            "probe_gate_passed": probe_gate_passed,
            "phase1_gate_passed": phase1_gate_passed,
            "probe_results_csv": str(probe_results_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "summary_plot": str(plot_path.resolve()),
            "report_path": str(report_path.resolve()),
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
        arm_rows[alias] = {str(row["example_id"]): row for row in case_rows}
        arm_summaries.append(
            {
                "mode": alias,
                "run_dir": str(run_dir.resolve()),
                "primary_metric": str(metrics.get("task_metric_name", "accuracy")),
                "primary_score": float(metrics.get("best_adapt_task_score", 0.0)),
                "mean_margin": float(metrics.get("best_adapt_task_margin", 0.0)),
                "case_rows": len(case_rows),
            }
        )
    pairwise_rows: list[dict[str, Any]] = []
    for left_alias, right_alias in [("A", "T"), ("A", "I_real"), ("I_shuffle", "I_real"), ("I_zero", "I_real")]:
        compare = _pairwise_compare(arm_rows[left_alias], arm_rows[right_alias])
        pairwise_rows.append(
            {
                "left_alias": left_alias,
                "right_alias": right_alias,
                **compare,
            }
        )
    real_vs_shuffle_gap_rows: list[dict[str, Any]] = []
    real_vs_zero_gap_rows: list[dict[str, Any]] = []
    for example_id in sorted(set(arm_rows["I_real"]) & set(arm_rows["I_shuffle"])):
        real_row = arm_rows["I_real"][example_id]
        shuffle_row = arm_rows["I_shuffle"][example_id]
        real_vs_shuffle_gap_rows.append(
            {
                "example_id": example_id,
                "real_correct": bool(real_row["predicted_correct"]),
                "shuffle_correct": bool(shuffle_row["predicted_correct"]),
                "real_margin": float(real_row["final_margin"]),
                "shuffle_margin": float(shuffle_row["final_margin"]),
                "margin_gap": float(real_row["final_margin"]) - float(shuffle_row["final_margin"]),
            }
        )
    for example_id in sorted(set(arm_rows["I_real"]) & set(arm_rows["I_zero"])):
        real_row = arm_rows["I_real"][example_id]
        zero_row = arm_rows["I_zero"][example_id]
        real_vs_zero_gap_rows.append(
            {
                "example_id": example_id,
                "real_correct": bool(real_row["predicted_correct"]),
                "zero_correct": bool(zero_row["predicted_correct"]),
                "real_margin": float(real_row["final_margin"]),
                "zero_margin": float(zero_row["final_margin"]),
                "margin_gap": float(real_row["final_margin"]) - float(zero_row["final_margin"]),
            }
        )
    summary_path = output_dir / "arm_summary.csv"
    plot_path = output_dir / "summary.svg"
    pairwise_path = output_dir / "arm_pairwise_compare.csv"
    real_vs_shuffle_path = output_dir / "real_vs_shuffle_gap.csv"
    real_vs_zero_path = output_dir / "real_vs_zero_gap.csv"
    report_path = output_dir / "report.md"
    write_summary_csv(summary_path, arm_summaries)
    write_sanity_plot(plot_path, arm_summaries)
    _write_csv(pairwise_path, pairwise_rows)
    _write_csv(real_vs_shuffle_path, real_vs_shuffle_gap_rows)
    _write_csv(real_vs_zero_path, real_vs_zero_gap_rows)
    pairwise_lookup = {(row["left_alias"], row["right_alias"]): row for row in pairwise_rows}
    teacher_helpful = bool(
        float(runs["T"][0].get("best_adapt_task_score", 0.0)) > float(runs["A"][0].get("best_adapt_task_score", 0.0))
        or float(runs["T"][0].get("best_adapt_task_margin", 0.0)) > float(runs["A"][0].get("best_adapt_task_margin", 0.0))
    )
    regressions_vs_base = int(pairwise_lookup[("A", "I_real")]["left_correct_to_right_wrong"])
    gate_passed = bool(
        teacher_helpful
        and int(pairwise_lookup[("I_shuffle", "I_real")]["flip_count_delta"]) >= 2
        and int(pairwise_lookup[("I_zero", "I_real")]["flip_count_delta"]) >= 2
        and float(runs["I_real"][0].get("best_adapt_task_score", 0.0))
        >= float(runs["A"][0].get("best_adapt_task_score", 0.0))
        and regressions_vs_base <= 1
    )
    report_lines = [
        "# M4 Shared Injection Compare",
        "",
        f"- teacher_helpful: {teacher_helpful}",
        f"- gate_passed: {gate_passed}",
        "",
        "## Arm Summary",
    ]
    for row in arm_summaries:
        report_lines.append(
            f"- {row['mode']}: task_score={row['primary_score']:.4f}, mean_margin={row['mean_margin']:.4f}"
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
    report_path.write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "m4_shared_injection_compare",
            "teacher_helpful": teacher_helpful,
            "gate_passed": gate_passed,
            "regressions_vs_base": regressions_vs_base,
            "summary_csv": str(summary_path.resolve()),
            "summary_plot": str(plot_path.resolve()),
            "pairwise_compare_csv": str(pairwise_path.resolve()),
            "real_vs_shuffle_gap_csv": str(real_vs_shuffle_path.resolve()),
            "real_vs_zero_gap_csv": str(real_vs_zero_path.resolve()),
            "report_path": str(report_path.resolve()),
        },
    )
