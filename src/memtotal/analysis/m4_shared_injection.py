from __future__ import annotations

import hashlib
import csv
import itertools
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
    _merge_example_lookup,
    _prefix_stats,
    _resolve_support_rows_for_memory_control,
    _resolve_support_lookup_dataset_paths,
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


def _load_snapshot_eval_rows(
    run_dir: Path,
) -> dict[int, tuple[dict[str, Any], list[dict[str, Any]]]]:
    snapshot_root = run_dir / "snapshot_evals"
    snapshots: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    if not snapshot_root.exists():
        return snapshots
    for metrics_path in sorted(snapshot_root.glob("step_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        case_path = metrics_path.parent / "task_case_dump.jsonl"
        case_rows = [
            json.loads(line)
            for line in case_path.read_text().splitlines()
            if line.strip()
        ]
        snapshots[int(metrics["step"])] = (metrics, case_rows)
    return snapshots


def _load_train_events(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "train_events.json"
    if not events_path.exists():
        return []
    payload = json.loads(events_path.read_text())
    return list(payload.get("events", []))


def _resolve_phase2_suite_roots(root: Path) -> dict[str, Path]:
    suites: dict[str, Path] = {}

    def register(path: Path, suite_name: str) -> None:
        if not path.exists():
            return
        direct_runs = []
        for child in path.iterdir():
            if not child.is_dir():
                continue
            metrics_path = child / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("training_stage") == "shared_injection_pilot":
                direct_runs.append(child.name)
        if direct_runs:
            suites[suite_name] = path

    if root.exists():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            register(child, child.name)
            register(child / "phase2-selected", child.name)
    return suites


def _stable_hash_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in FEVER_LABEL_ORDER}
    for row in rows:
        label = str(row.get("label", "")).strip()
        if label in counts:
            counts[label] += 1
    return counts


def _selection_bucket(case_row: dict[str, Any]) -> str:
    if bool(case_row["predicted_correct"]):
        return "A_correct"
    if abs(float(case_row["final_margin"])) < 0.25:
        return "A_near_threshold"
    return "A_wrong"


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
        attn_implementation=backbone_cfg.get("attn_implementation"),
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


def run_m4_prepare_fever_validation_splits(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> None:
    del dry_run
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _load_task_dataset_with_path(config, str(config["task"]["dataset_path"]))
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in FEVER_LABEL_ORDER}
    for row in source_rows:
        label = str(row.get("label", "")).strip()
        if label in by_label:
            by_label[label].append(dict(row))
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for label in FEVER_LABEL_ORDER:
        ordered = sorted(by_label[label], key=lambda row: _stable_hash_rank(str(row["id"])))
        val_count = max(1, round(len(ordered) * 0.25))
        val_ids = {str(row["id"]) for row in ordered[:val_count]}
        for row in ordered:
            if str(row["id"]) in val_ids:
                row["m4_split"] = "screen_val"
                val_rows.append(row)
            else:
                row["m4_split"] = "screen_train"
                train_rows.append(row)
    val_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in FEVER_LABEL_ORDER}
    for row in val_rows:
        val_by_label[str(row["label"])].append(row)
    audit_rows: list[dict[str, Any]] = []
    for label in FEVER_LABEL_ORDER:
        ordered = sorted(val_by_label[label], key=lambda row: _stable_hash_rank(f"audit::{row['id']}"))
        audit_rows.extend(ordered[:4])

    data_root = Path(config["runtime"].get("split_output_root", "data/benchmarks/pilots/fever"))
    data_root.mkdir(parents=True, exist_ok=True)
    train_path = data_root / "screen-train.jsonl"
    val_path = data_root / "screen-val.jsonl"
    audit_path = data_root / "screen-val-audit12.jsonl"
    manifest_path = data_root / "screen-train-val-manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(audit_path, audit_rows)
    metrics = {
        "analysis_mode": "m4_prepare_fever_validation_splits",
        "source_dataset_path": str(Path(config["task"]["dataset_path"]).resolve()),
        "train_dataset_path": str(train_path.resolve()),
        "val_dataset_path": str(val_path.resolve()),
        "audit_dataset_path": str(audit_path.resolve()),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "audit_examples": len(audit_rows),
        "train_label_distribution": _label_distribution(train_rows),
        "val_label_distribution": _label_distribution(val_rows),
        "audit_label_distribution": _label_distribution(audit_rows),
        "manifest_path": str(manifest_path.resolve()),
    }
    write_json(manifest_path, metrics)
    write_json(output_dir / "metrics.json", metrics)


def _ordered_label_pairs(
    rows: list[dict[str, Any]],
    *,
    namespace: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(rows) < 2:
        raise ValueError(f"Need at least two rows to build support pairs for namespace={namespace}.")
    ranked_rows = sorted(rows, key=lambda row: _stable_hash_rank(f"{namespace}::row::{row['id']}"))
    pairs = list(itertools.combinations(ranked_rows, 2))
    pairs.sort(
        key=lambda pair: _stable_hash_rank(
            f"{namespace}::pair::{pair[0]['id']}::{pair[1]['id']}"
        )
    )
    return pairs


def _episode_from_pair_lookup(
    pair_lookup: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    *,
    pair_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for label in FEVER_LABEL_ORDER:
        pairs = pair_lookup[label]
        if pair_index >= len(pairs):
            raise ValueError(
                f"Pair index {pair_index} is out of range for label {label}; only {len(pairs)} pairs available."
            )
        for row in pairs[pair_index]:
            row_copy = dict(row)
            row_id = str(row_copy["id"])
            if row_id in seen_ids:
                raise ValueError(f"Duplicate support id {row_id} encountered in pair_index={pair_index}.")
            seen_ids.add(row_id)
            rows.append(row_copy)
    return rows


def _support_bank_signature(rows: list[dict[str, Any]]) -> str:
    return "|".join(sorted(str(row["id"]) for row in rows))


def run_m4_prepare_fever_support_banks(
    *,
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> None:
    del dry_run
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset_path = str(config["task"].get("train_dataset_path", config["task"]["dataset_path"]))
    train_rows = _load_task_dataset_with_path(config, train_dataset_path)
    train_rows = [dict(row) for row in train_rows]
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in FEVER_LABEL_ORDER}
    for row in train_rows:
        label = str(row.get("label", "")).strip()
        if label in by_label:
            by_label[label].append(row)
    pair_lookup = {
        label: _ordered_label_pairs(rows, namespace=f"m46::{label}")
        for label, rows in by_label.items()
    }
    episode_count = int(config["runtime"].get("support_bank_episode_count", 32))
    reserved_banks = (
        ("screen_val_canonical", 0),
        ("screen248_test_canonical", 1),
        ("screen248_test_heldout_a", 2),
        ("screen248_test_heldout_b", 3),
    )
    required_pair_count = len(reserved_banks) + episode_count
    for label, pairs in pair_lookup.items():
        if len(pairs) < required_pair_count:
            raise ValueError(
                f"Not enough unique pairs for label {label}; need {required_pair_count}, got {len(pairs)}."
            )
    data_root = Path(config["runtime"].get("support_bank_output_root", "data/benchmarks/pilots/fever"))
    data_root.mkdir(parents=True, exist_ok=True)
    train_episode_bank_path = data_root / "m46-train-triad-episode-bank32.json"
    val_canonical_path = data_root / "m46-screen-val-canonical-support6.jsonl"
    test_canonical_path = data_root / "m46-screen248-test-canonical-support6.jsonl"
    heldout_a_path = data_root / "m46-screen248-test-heldout-a-support6.jsonl"
    heldout_b_path = data_root / "m46-screen248-test-heldout-b-support6.jsonl"
    manifest_path = data_root / "m46-support-bank-manifest.json"
    path_by_bank = {
        "screen_val_canonical": val_canonical_path,
        "screen248_test_canonical": test_canonical_path,
        "screen248_test_heldout_a": heldout_a_path,
        "screen248_test_heldout_b": heldout_b_path,
    }
    seen_signatures: set[str] = set()
    bank_rows: dict[str, list[dict[str, Any]]] = {}
    for bank_name, pair_index in reserved_banks:
        rows = _episode_from_pair_lookup(pair_lookup, pair_index=pair_index)
        signature = _support_bank_signature(rows)
        if signature in seen_signatures:
            raise ValueError(f"Duplicate support bank signature detected for {bank_name}.")
        seen_signatures.add(signature)
        bank_rows[bank_name] = rows
        write_jsonl(path_by_bank[bank_name], rows)
    train_episodes: list[dict[str, Any]] = []
    for episode_offset in range(episode_count):
        pair_index = len(reserved_banks) + episode_offset
        rows = _episode_from_pair_lookup(pair_lookup, pair_index=pair_index)
        signature = _support_bank_signature(rows)
        if signature in seen_signatures:
            raise ValueError(f"Duplicate train support episode signature detected at pair_index={pair_index}.")
        seen_signatures.add(signature)
        train_episodes.append(
            {
                "episode_id": f"train_ep_{episode_offset:03d}",
                "source_split": "screen_train",
                "label_counts": {label: 2 for label in FEVER_LABEL_ORDER},
                "support_rows": rows,
            }
        )
    write_json(
        train_episode_bank_path,
        {
            "analysis_mode": "m4_prepare_fever_support_banks",
            "source_dataset_path": str(Path(train_dataset_path).resolve()),
            "episode_count": episode_count,
            "episodes": train_episodes,
        },
    )
    manifest = {
        "analysis_mode": "m4_prepare_fever_support_banks",
        "source_dataset_path": str(Path(train_dataset_path).resolve()),
        "train_label_distribution": _label_distribution(train_rows),
        "train_episode_bank_path": str(train_episode_bank_path.resolve()),
        "train_episode_count": len(train_episodes),
        "screen_val_canonical_bank_path": str(val_canonical_path.resolve()),
        "screen248_test_canonical_bank_path": str(test_canonical_path.resolve()),
        "screen248_test_heldout_a_bank_path": str(heldout_a_path.resolve()),
        "screen248_test_heldout_b_bank_path": str(heldout_b_path.resolve()),
        "screen_val_canonical_ids": [str(row["id"]) for row in bank_rows["screen_val_canonical"]],
        "screen248_test_canonical_ids": [str(row["id"]) for row in bank_rows["screen248_test_canonical"]],
        "screen248_test_heldout_a_ids": [str(row["id"]) for row in bank_rows["screen248_test_heldout_a"]],
        "screen248_test_heldout_b_ids": [str(row["id"]) for row in bank_rows["screen248_test_heldout_b"]],
        "train_episode_ids": [str(episode["episode_id"]) for episode in train_episodes],
        "train_episode_label_counts": {label: 2 for label in FEVER_LABEL_ORDER},
    }
    write_json(manifest_path, manifest)
    write_json(
        output_dir / "metrics.json",
        {
            **manifest,
            "manifest_path": str(manifest_path.resolve()),
        },
    )


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


def _valid_prompt_surface(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics["dominant_label_fraction"]) <= 0.85
        and _label_recall_count(metrics) >= 2
    )


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
            layer_prefix_hidden_by_layer=None,
            prefix_stats=_prefix_stats(),
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
                layer_prefix_hidden_by_layer=None,
                prefix_stats=_prefix_stats(),
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

    a_rows_by_prompt = {
        str(row["prompt_variant"]): dict(row)
        for row in arm_summary_rows
        if row["arm_type"] == "A"
    }
    prompt_pair_rows: list[dict[str, Any]] = []
    for prompt_variant in FEVER_PROMPT_VARIANTS:
        a_row = dict(a_rows_by_prompt[prompt_variant])
        a_row["valid_prompt_surface"] = _valid_prompt_surface(a_row)
        t_rows = [
            dict(row)
            for row in arm_summary_rows
            if row["arm_type"] == "T" and row["prompt_variant"] == prompt_variant
        ]
        t_rows_sorted = sorted(
            t_rows,
            key=lambda row: (
                float(row["macro_f1"]),
                float(row["accuracy"]),
                -float(row["dominant_label_fraction"]),
            ),
            reverse=True,
        )
        t_row = dict(t_rows_sorted[0])
        t_row["valid_support_surface"] = _valid_prompt_surface(t_row)
        accuracy_gain = float(t_row["accuracy"]) - float(a_row["accuracy"])
        macro_f1_gain = float(t_row["macro_f1"]) - float(a_row["macro_f1"])
        prompt_pair_rows.append(
            {
                "prompt_variant": prompt_variant,
                "a_row": a_row,
                "t_row": t_row,
                "a_valid_prompt_surface": bool(a_row["valid_prompt_surface"]),
                "t_valid_support_surface": bool(t_row["valid_support_surface"]),
                "accuracy_gain": accuracy_gain,
                "macro_f1_gain": macro_f1_gain,
                "pair_gate_passed": bool(
                    t_row["valid_support_surface"]
                    and accuracy_gain >= (2.0 / max(1, len(eval_examples)))
                    and macro_f1_gain >= 0.05
                    and float(t_row["dominant_label_fraction"]) <= float(a_row["dominant_label_fraction"])
                ),
            }
        )

    prompt_pair_rows_sorted = sorted(
        prompt_pair_rows,
        key=lambda row: (
            int(bool(row["pair_gate_passed"])),
            int(bool(row["t_valid_support_surface"])),
            float(row["macro_f1_gain"]),
            float(row["accuracy_gain"]),
            float(row["t_row"]["macro_f1"]),
            float(row["t_row"]["accuracy"]),
            -float(row["t_row"]["dominant_label_fraction"]),
        ),
        reverse=True,
    )
    prompt_pair_winner = prompt_pair_rows_sorted[0]
    a_winner = dict(prompt_pair_winner["a_row"])
    t_winner = dict(prompt_pair_winner["t_row"])
    phase0_gate_passed = bool(prompt_pair_winner["pair_gate_passed"])
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
    _write_csv(
        output_dir / "phase0_prompt_pairs.csv",
        [
            {
                "prompt_variant": row["prompt_variant"],
                "selected_support_serialization_variant": row["t_row"]["support_serialization_variant"],
                "a_accuracy": float(row["a_row"]["accuracy"]),
                "a_macro_f1": float(row["a_row"]["macro_f1"]),
                "a_dominant_label_fraction": float(row["a_row"]["dominant_label_fraction"]),
                "a_valid_prompt_surface": bool(row["a_valid_prompt_surface"]),
                "t_accuracy": float(row["t_row"]["accuracy"]),
                "t_macro_f1": float(row["t_row"]["macro_f1"]),
                "t_dominant_label_fraction": float(row["t_row"]["dominant_label_fraction"]),
                "t_valid_support_surface": bool(row["t_valid_support_surface"]),
                "accuracy_gain": float(row["accuracy_gain"]),
                "macro_f1_gain": float(row["macro_f1_gain"]),
                "pair_gate_passed": bool(row["pair_gate_passed"]),
            }
            for row in prompt_pair_rows
        ],
    )
    report_lines = [
        "# M4 Phase 0 Gate Sweep",
        "",
        f"- phase0_gate_passed: {phase0_gate_passed}",
        f"- a_winner_prompt_variant: {a_winner['prompt_variant']}",
        f"- t_winner_support_serialization: {t_winner['support_serialization_variant']}",
        (
            f"- selected_pair_accuracy_gain={float(prompt_pair_winner['accuracy_gain']):.4f}, "
            f"selected_pair_macro_f1_gain={float(prompt_pair_winner['macro_f1_gain']):.4f}"
        ),
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
            "selected_pair": {
                "prompt_variant": str(prompt_pair_winner["prompt_variant"]),
                "support_serialization_variant": str(t_winner["support_serialization_variant"]),
                "accuracy_gain": float(prompt_pair_winner["accuracy_gain"]),
                "macro_f1_gain": float(prompt_pair_winner["macro_f1_gain"]),
                "pair_gate_passed": bool(prompt_pair_winner["pair_gate_passed"]),
            },
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
    runtime_cfg = dict(config.get("runtime", {}))
    flip_gain_min = int(runtime_cfg.get("pilot_compare_flip_gain_min", 2))
    macro_f1_gap_min = float(runtime_cfg.get("pilot_compare_macro_f1_gap_min", 0.05))
    regressions_max = int(runtime_cfg.get("pilot_compare_regressions_max", 1))
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
    flip_gain_vs_shuffle = int(pairwise_lookup[("I_shuffle", "I_real")]["flip_count_delta"])
    flip_gain_vs_zero = int(pairwise_lookup[("I_zero", "I_real")]["flip_count_delta"])
    macro_f1_gap_vs_shuffle = float(summary_lookup["I_real"]["macro_f1"]) - float(summary_lookup["I_shuffle"]["macro_f1"])
    macro_f1_gap_vs_zero = float(summary_lookup["I_real"]["macro_f1"]) - float(summary_lookup["I_zero"]["macro_f1"])
    gate_passed = bool(
        flip_gain_vs_shuffle >= flip_gain_min
        and macro_f1_gap_vs_shuffle >= macro_f1_gap_min
        and flip_gain_vs_zero >= flip_gain_min
        and macro_f1_gap_vs_zero >= macro_f1_gap_min
        and float(summary_lookup["I_real"]["primary_score"]) >= float(summary_lookup["A"]["primary_score"])
        and regressions_vs_base <= regressions_max
    )
    report_lines = [
        "# M4 Shared Injection Compare",
        "",
        f"- gate_passed: {gate_passed}",
        f"- regressions_vs_base: {regressions_vs_base}",
        f"- flip_gain_vs_shuffle: {flip_gain_vs_shuffle}",
        f"- flip_gain_vs_zero: {flip_gain_vs_zero}",
        f"- macro_f1_gap_vs_shuffle: {macro_f1_gap_vs_shuffle:.4f}",
        f"- macro_f1_gap_vs_zero: {macro_f1_gap_vs_zero:.4f}",
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
            "flip_gain_vs_shuffle": flip_gain_vs_shuffle,
            "flip_gain_vs_zero": flip_gain_vs_zero,
            "macro_f1_gap_vs_shuffle": macro_f1_gap_vs_shuffle,
            "macro_f1_gap_vs_zero": macro_f1_gap_vs_zero,
            "i_real_task_score": float(summary_lookup["I_real"]["primary_score"]),
            "a_task_score": float(summary_lookup["A"]["primary_score"]),
            "i_shuffle_task_score": float(summary_lookup["I_shuffle"]["primary_score"]),
            "i_zero_task_score": float(summary_lookup["I_zero"]["primary_score"]),
            "i_real_macro_f1": float(summary_lookup["I_real"]["macro_f1"]),
            "a_macro_f1": float(summary_lookup["A"]["macro_f1"]),
            "i_shuffle_macro_f1": float(summary_lookup["I_shuffle"]["macro_f1"]),
            "i_zero_macro_f1": float(summary_lookup["I_zero"]["macro_f1"]),
            "compare_flip_gain_min": flip_gain_min,
            "compare_macro_f1_gap_min": macro_f1_gap_min,
            "compare_regressions_max": regressions_max,
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


def run_m4_shared_injection_dynamics_audit(
    *,
    config: dict[str, Any],
    output_dir: Path,
    input_root: str,
    dry_run: bool,
) -> None:
    del config, dry_run
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_roots = _resolve_phase2_suite_roots(Path(input_root).resolve())
    if not suite_roots:
        raise ValueError(f"No phase2-selected suite roots found under {input_root}.")

    summary_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    best_suite_metrics: dict[str, Any] = {}
    report_lines = ["# M4 Shared Injection Dynamics Audit", ""]

    for suite_name, suite_root in sorted(suite_roots.items()):
        runs = _collect_shared_injection_runs(suite_root)
        required = {"A", "T", "I_real", "I_shuffle", "I_zero"}
        missing = sorted(required - set(runs))
        if missing:
            raise ValueError(
                f"Suite {suite_name} is missing required shared injection runs: {', '.join(missing)}"
            )
        snapshots = {
            alias: _load_snapshot_eval_rows(run_dir)
            for alias, (_, run_dir) in runs.items()
        }
        static_rows: dict[str, list[dict[str, Any]]] = {}
        for alias in ("A", "T", "I_zero"):
            if 0 in snapshots[alias]:
                static_rows[alias] = snapshots[alias][0][1]
            else:
                _, static_rows[alias] = _load_case_rows(runs[alias][1])
        common_steps = sorted(set(snapshots["I_real"]) & set(snapshots["I_shuffle"]))
        if not common_steps:
            raise ValueError(f"Suite {suite_name} has no overlapping I_real/I_shuffle snapshot steps.")

        suite_i_real_rows: list[dict[str, Any]] = []
        for step in common_steps:
            step_case_rows = {
                "A": static_rows["A"],
                "T": static_rows["T"],
                "I_zero": static_rows["I_zero"],
                "I_real": snapshots["I_real"][step][1],
                "I_shuffle": snapshots["I_shuffle"][step][1],
            }
            step_metrics = {
                alias: _classification_metrics_from_rows(rows)
                for alias, rows in step_case_rows.items()
            }
            for alias, metrics in step_metrics.items():
                summary_rows.append(
                    {
                        "suite": suite_name,
                        "step": int(step),
                        "alias": alias,
                        "task_score": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "mean_margin": float(metrics["mean_margin"]),
                        "dominant_label_fraction": float(metrics["dominant_label_fraction"]),
                    }
                )
            for left_alias, right_alias in [("A", "I_real"), ("I_shuffle", "I_real"), ("I_zero", "I_real")]:
                compare = _pairwise_compare(
                    {str(row["example_id"]): row for row in step_case_rows[left_alias]},
                    {str(row["example_id"]): row for row in step_case_rows[right_alias]},
                )
                pairwise_rows.append(
                    {
                        "suite": suite_name,
                        "step": int(step),
                        "left_alias": left_alias,
                        "right_alias": right_alias,
                        **compare,
                    }
                )
            pairwise_lookup = {
                (row["left_alias"], row["right_alias"]): row
                for row in pairwise_rows
                if row["suite"] == suite_name and int(row["step"]) == int(step)
            }
            suite_i_real_rows.append(
                {
                    "step": int(step),
                    "task_score": float(step_metrics["I_real"]["accuracy"]),
                    "macro_f1": float(step_metrics["I_real"]["macro_f1"]),
                    "mean_margin": float(step_metrics["I_real"]["mean_margin"]),
                    "flip_gain_vs_shuffle": int(pairwise_lookup[("I_shuffle", "I_real")]["flip_count_delta"]),
                    "flip_gain_vs_zero": int(pairwise_lookup[("I_zero", "I_real")]["flip_count_delta"]),
                }
            )

        best_row = max(
            suite_i_real_rows,
            key=lambda row: (
                float(row["macro_f1"]),
                float(row["task_score"]),
                float(row["mean_margin"]),
            ),
        )
        final_step = max(common_steps)
        final_row = next(row for row in suite_i_real_rows if int(row["step"]) == int(final_step))
        best_suite_metrics[suite_name] = {
            "best_step": int(best_row["step"]),
            "best_macro_f1": float(best_row["macro_f1"]),
            "best_task_score": float(best_row["task_score"]),
            "best_mean_margin": float(best_row["mean_margin"]),
            "final_step": int(final_row["step"]),
            "final_macro_f1": float(final_row["macro_f1"]),
            "final_task_score": float(final_row["task_score"]),
            "final_mean_margin": float(final_row["mean_margin"]),
            "overshoot_detected": bool(int(best_row["step"]) != int(final_row["step"])),
        }
        report_lines.extend(
            [
                f"## {suite_name}",
                f"- best_step_by_macro_f1: {best_row['step']}",
                (
                    f"- best_real: task_score={best_row['task_score']:.4f}, "
                    f"macro_f1={best_row['macro_f1']:.4f}, mean_margin={best_row['mean_margin']:.4f}, "
                    f"flip_gain_vs_shuffle={best_row['flip_gain_vs_shuffle']}, "
                    f"flip_gain_vs_zero={best_row['flip_gain_vs_zero']}"
                ),
                (
                    f"- final_real: task_score={final_row['task_score']:.4f}, "
                    f"macro_f1={final_row['macro_f1']:.4f}, mean_margin={final_row['mean_margin']:.4f}"
                ),
                f"- overshoot_detected: {best_suite_metrics[suite_name]['overshoot_detected']}",
                "",
            ]
        )

    summary_csv = output_dir / "dynamics_summary.csv"
    pairwise_csv = output_dir / "dynamics_pairwise.csv"
    summary_plot = output_dir / "dynamics_summary.svg"
    _write_csv(summary_csv, summary_rows)
    _write_csv(pairwise_csv, pairwise_rows)
    write_sanity_plot(
        summary_plot,
        [
            {
                "mode": f"{row['suite']}:{row['alias']}:step{int(row['step']):04d}",
                "run_dir": f"{row['suite']}-{row['alias']}-step{int(row['step']):04d}",
                "primary_metric": "macro_f1",
                "primary_score": float(row["macro_f1"]),
            }
            for row in summary_rows
            if row["alias"] == "I_real"
        ],
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "m4_shared_injection_dynamics_audit",
            "suite_count": len(suite_roots),
            "summary_csv": str(summary_csv.resolve()),
            "pairwise_csv": str(pairwise_csv.resolve()),
            "summary_plot": str(summary_plot.resolve()),
            "report_path": str(report_path.resolve()),
            "suite_best_metrics": best_suite_metrics,
        },
    )


def _build_case_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["example_id"]): row for row in rows}


def _stage_case_bucket(a_rows: dict[str, dict[str, Any]], example_id: str) -> str:
    row = a_rows[example_id]
    if bool(row["predicted_correct"]):
        return "A_correct"
    if abs(float(row["final_margin"])) < 0.25:
        return "A_near_threshold"
    return "A_wrong"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _collect_content_gap_rows(
    *,
    suite_name: str,
    step: int,
    a_rows: dict[str, dict[str, Any]],
    real_rows: dict[str, dict[str, Any]],
    shuffle_rows: dict[str, dict[str, Any]],
    zero_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared_ids = sorted(set(a_rows) & set(real_rows) & set(shuffle_rows) & set(zero_rows))
    for bucket_name in ("overall", "A_wrong", "A_near_threshold", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
        if bucket_name == "overall":
            bucket_ids = shared_ids
        elif bucket_name in {"A_wrong", "A_near_threshold"}:
            bucket_ids = [example_id for example_id in shared_ids if _stage_case_bucket(a_rows, example_id) == bucket_name]
        else:
            bucket_ids = [
                example_id
                for example_id in shared_ids
                if str(real_rows[example_id]["gold_label"]) == bucket_name
            ]
        if not bucket_ids:
            continue
        real_minus_shuffle = [
            float(real_rows[example_id]["final_margin"]) - float(shuffle_rows[example_id]["final_margin"])
            for example_id in bucket_ids
        ]
        real_minus_zero = [
            float(real_rows[example_id]["final_margin"]) - float(zero_rows[example_id]["final_margin"])
            for example_id in bucket_ids
        ]
        rows.append(
            {
                "suite": suite_name,
                "step": int(step),
                "bucket": bucket_name,
                "example_count": len(bucket_ids),
                "mean_margin_gap_real_vs_shuffle": _mean(real_minus_shuffle),
                "mean_margin_gap_real_vs_zero": _mean(real_minus_zero),
            }
        )
    return rows


def _load_audit_examples(config: dict[str, Any], dataset_path: str, prompt_variant: str) -> list[SharedInjectionExampleCache]:
    rows = _load_task_dataset_with_path(config, dataset_path)
    return _build_example_caches(rows, prompt_variant=prompt_variant)


def _attention_rows_for_snapshot(
    *,
    config: dict[str, Any],
    support_examples: list[dict[str, Any]],
    audit_examples: list[SharedInjectionExampleCache],
    example_lookup: dict[str, dict[str, Any]],
    support_serialization_variant: str,
    writer_memory_control: str,
    checkpoint_path: str | None,
    step: int,
    suite_name: str,
    arm_alias: str,
) -> list[dict[str, Any]]:
    runtime = SharedInjectionPilotRuntime(
        config=config,
        seed=0,
        arm="injected",
        writer_memory_control=writer_memory_control,
    )
    if checkpoint_path:
        runtime.load_injection_checkpoint(checkpoint_path)
    support_text_block = _build_support_text_block(
        support_examples,
        memory_control=writer_memory_control,
        example_lookup=example_lookup,
        support_serialization_variant=support_serialization_variant,
    )
    support_rows = _resolve_support_rows_for_memory_control(
        support_examples,
        memory_control=writer_memory_control,
        example_lookup=example_lookup,
        support_serialization_variant=support_serialization_variant,
    )
    prefix_artifacts = runtime.build_prefix_artifacts(
        support_text_block,
        support_rows=support_rows,
    )
    prefix_stats = prefix_artifacts.prefix_stats
    rows: list[dict[str, Any]] = []
    for cache in audit_examples:
        scores, diagnostics = runtime.score_example(
            cache,
            support_text_block=support_text_block,
            prefix_embeddings=prefix_artifacts.prefix_embeddings,
            layer_prefix_hidden_by_layer=prefix_artifacts.layer_prefix_hidden_by_layer,
            return_diagnostics=True,
        )
        scores_cpu = scores.detach().to(dtype=torch.float32).cpu()
        competitor_index = max(
            [index for index in range(len(cache.candidate_labels)) if index != cache.gold_index],
            key=lambda index: float(scores_cpu[index].item()),
        )
        masses = [float(value) for value in diagnostics["prefix_attention_mass_by_candidate"]]
        masses_by_layer = {
            int(layer_index): [float(value) for value in values]
            for layer_index, values in diagnostics.get("prefix_attention_mass_by_candidate_by_layer", {}).items()
        }
        for layer_index in diagnostics.get("diagnostic_layers", []):
            layer_masses = masses_by_layer.get(int(layer_index), masses)
            rows.append(
                {
                    "suite": suite_name,
                    "step": int(step),
                    "arm_alias": arm_alias,
                    "layer_index": int(layer_index),
                    "example_id": str(cache.example["id"]),
                    "gold_label": cache.candidate_labels[cache.gold_index],
                    "predicted_label": cache.candidate_labels[int(torch.argmax(scores_cpu).item())],
                    "mean_prefix_attention_mass": _mean(layer_masses),
                    "gold_prefix_attention_mass": layer_masses[cache.gold_index],
                    "competitor_prefix_attention_mass": layer_masses[competitor_index],
                    "prefix_tokens": int(prefix_stats["prefix_tokens"]),
                    "prefix_l2": float(prefix_stats["prefix_l2"]),
                }
            )
    return rows


def _collect_prefix_norm_rows_for_metrics(
    *,
    suite_name: str,
    step: int,
    arm_alias: str,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    prefix_stats = dict(metrics.get("prefix_artifact_stats", {}))
    rows: list[dict[str, Any]] = [
        {
            "suite": suite_name,
            "step": int(step),
            "arm_alias": arm_alias,
            "row_type": "snapshot_aggregate",
            "layer_index": "",
            "pilot_support_encoder_mode": str(metrics.get("pilot_support_encoder_mode", "pooled_block")),
            "prefix_tokens": float(metrics.get("prefix_tokens", 0.0)),
            "prefix_l2": float(metrics.get("prefix_l2", 0.0)),
            "prefix_slot_norm_mean": float(metrics.get("prefix_slot_norm_mean", 0.0)),
            "prefix_slot_norm_std": float(metrics.get("prefix_slot_norm_std", 0.0)),
            "prefix_slot_norm_max": float(metrics.get("prefix_slot_norm_max", 0.0)),
            "writer_memory_l2": float(metrics.get("writer_memory_l2", 0.0)),
            "writer_slot_norm_mean": float(metrics.get("writer_slot_norm_mean", 0.0)),
            "writer_slot_norm_std": float(metrics.get("writer_slot_norm_std", 0.0)),
            "writer_slot_norm_max": float(metrics.get("writer_slot_norm_max", 0.0)),
            "support_item_count": float(metrics.get("support_item_count", 0.0)),
            "support_item_hidden_l2": float(metrics.get("support_item_hidden_l2", 0.0)),
            "support_item_hidden_norm_mean": float(metrics.get("support_item_hidden_norm_mean", 0.0)),
            "support_item_hidden_norm_std": float(metrics.get("support_item_hidden_norm_std", 0.0)),
            "support_item_hidden_norm_max": float(metrics.get("support_item_hidden_norm_max", 0.0)),
        }
    ]
    layer_indices = sorted(
        {
            *prefix_stats.get("layer_hidden_l2_by_layer", {}).keys(),
            *prefix_stats.get("layer_key_l2_by_layer", {}).keys(),
            *prefix_stats.get("layer_value_l2_by_layer", {}).keys(),
        },
        key=int,
    )
    for layer_index in layer_indices:
        rows.append(
            {
                "suite": suite_name,
                "step": int(step),
                "arm_alias": arm_alias,
                "row_type": "snapshot_layer",
                "layer_index": int(layer_index),
                "prefix_tokens": float(metrics.get("prefix_tokens", 0.0)),
                "prefix_hidden_l2": float(prefix_stats.get("layer_hidden_l2_by_layer", {}).get(layer_index, 0.0)),
                "prefix_hidden_slot_norm_mean": float(
                    prefix_stats.get("layer_slot_norm_mean_by_layer", {}).get(layer_index, 0.0)
                ),
                "prefix_hidden_slot_norm_std": float(
                    prefix_stats.get("layer_slot_norm_std_by_layer", {}).get(layer_index, 0.0)
                ),
                "prefix_hidden_slot_norm_max": float(
                    prefix_stats.get("layer_slot_norm_max_by_layer", {}).get(layer_index, 0.0)
                ),
                "projected_k_l2": float(prefix_stats.get("layer_key_l2_by_layer", {}).get(layer_index, 0.0)),
                "projected_v_l2": float(prefix_stats.get("layer_value_l2_by_layer", {}).get(layer_index, 0.0)),
            }
        )
    return rows


def _collect_train_event_norm_rows(
    *,
    suite_name: str,
    arm_alias: str,
    train_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in train_events:
        rows.append(
            {
                "suite": suite_name,
                "step": int(event.get("step", 0)),
                "arm_alias": arm_alias,
                "row_type": "train_event",
                "layer_index": "",
                "prefix_tokens": float(event.get("prefix_tokens", 0.0)),
                "prefix_l2": float(event.get("prefix_l2", 0.0)),
                "writer_memory_l2": float(event.get("writer_memory_l2", 0.0)),
                "writer_slot_norm_mean": float(event.get("writer_slot_norm_mean", 0.0)),
                "writer_slot_norm_std": float(event.get("writer_slot_norm_std", 0.0)),
                "writer_slot_norm_max": float(event.get("writer_slot_norm_max", 0.0)),
                "support_item_count": float(event.get("support_item_count", 0.0)),
                "support_item_hidden_l2": float(event.get("support_item_hidden_l2", 0.0)),
                "support_item_hidden_norm_mean": float(event.get("support_item_hidden_norm_mean", 0.0)),
                "support_item_hidden_norm_std": float(event.get("support_item_hidden_norm_std", 0.0)),
                "support_item_hidden_norm_max": float(event.get("support_item_hidden_norm_max", 0.0)),
                "pilot_support_encoder_mode": str(event.get("pilot_support_encoder_mode", "pooled_block")),
                "pilot_trainable_variant": str(event.get("pilot_trainable_variant", "")),
                "alignment_aux_mode": str(event.get("alignment_aux_mode", "off")),
                "alignment_aux_active": bool(event.get("alignment_aux_active", False)),
                "alignment_aux_loss": float(event.get("alignment_aux_loss", 0.0)),
                "support_encoder_grad_norm": float(event.get("support_encoder_grad_norm", 0.0)),
                "prefix_projector_grad_norm": float(event.get("prefix_projector_grad_norm", 0.0)),
                "writer_grad_norm": float(event.get("writer_grad_norm", 0.0)),
                "writer_to_projector_grad_ratio": float(event.get("writer_to_projector_grad_ratio", 0.0)),
                "total_grad_norm_pre_clip": float(event.get("total_grad_norm_pre_clip", 0.0)),
                "loss": float(event.get("loss", 0.0)),
            }
        )
    return rows


def run_m4_shared_injection_dynamics_recovery(
    *,
    config: dict[str, Any],
    output_dir: Path,
    input_root: str,
    dry_run: bool,
) -> None:
    del dry_run
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_roots = _resolve_phase2_suite_roots(Path(input_root).resolve())
    if not suite_roots:
        raise ValueError(f"No phase2-selected suite roots found under {input_root}.")

    summary_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    content_gap_rows: list[dict[str, Any]] = []
    prefix_norm_rows: list[dict[str, Any]] = []
    candidate_selections: list[dict[str, Any]] = []
    suite_support_variant: dict[str, str] = {}
    report_lines = ["# M4 Shared Injection Dynamics Recovery", ""]

    for suite_name, suite_root in sorted(suite_roots.items()):
        runs = _collect_shared_injection_runs(suite_root)
        required = {"A", "T", "I_real", "I_shuffle", "I_zero"}
        missing = sorted(required - set(runs))
        if missing:
            raise ValueError(f"Suite {suite_name} is missing required runs: {', '.join(missing)}")
        snapshots = {alias: _load_snapshot_eval_rows(run_dir) for alias, (_, run_dir) in runs.items()}
        train_events_by_alias = {
            alias: _load_train_events(run_dir)
            for alias, (_, run_dir) in runs.items()
            if alias in {"I_real", "I_shuffle"}
        }
        static_rows: dict[str, list[dict[str, Any]]] = {}
        for alias in ("A", "T", "I_zero"):
            if 0 in snapshots[alias]:
                static_rows[alias] = snapshots[alias][0][1]
            else:
                _, static_rows[alias] = _load_case_rows(runs[alias][1])
        common_steps = sorted(set(snapshots["I_real"]) & set(snapshots["I_shuffle"]))
        if not common_steps:
            raise ValueError(f"Suite {suite_name} has no overlapping snapshot steps.")
        suite_support_variant[suite_name] = str(runs["I_real"][0].get("support_serialization_variant", suite_name))
        for step in common_steps:
            step_case_rows = {
                "A": static_rows["A"],
                "T": static_rows["T"],
                "I_zero": static_rows["I_zero"],
                "I_real": snapshots["I_real"][step][1],
                "I_shuffle": snapshots["I_shuffle"][step][1],
            }
            step_metrics = {
                alias: _classification_metrics_from_rows(rows)
                for alias, rows in step_case_rows.items()
            }
            for alias, metrics in step_metrics.items():
                summary_rows.append(
                    {
                        "suite": suite_name,
                        "step": int(step),
                        "alias": alias,
                        "task_score": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "mean_margin": float(metrics["mean_margin"]),
                        "dominant_label_fraction": float(metrics["dominant_label_fraction"]),
                    }
                )
            step_pairwise_lookup: dict[tuple[str, str], dict[str, Any]] = {}
            for left_alias, right_alias in [("A", "I_real"), ("I_shuffle", "I_real"), ("I_zero", "I_real")]:
                compare = _pairwise_compare(
                    _build_case_lookup(step_case_rows[left_alias]),
                    _build_case_lookup(step_case_rows[right_alias]),
                )
                row = {
                    "suite": suite_name,
                    "step": int(step),
                    "left_alias": left_alias,
                    "right_alias": right_alias,
                    **compare,
                }
                pairwise_rows.append(row)
                step_pairwise_lookup[(left_alias, right_alias)] = row
            content_gap_rows.extend(
                _collect_content_gap_rows(
                    suite_name=suite_name,
                    step=int(step),
                    a_rows=_build_case_lookup(step_case_rows["A"]),
                    real_rows=_build_case_lookup(step_case_rows["I_real"]),
                    shuffle_rows=_build_case_lookup(step_case_rows["I_shuffle"]),
                    zero_rows=_build_case_lookup(step_case_rows["I_zero"]),
                )
            )
            snapshot_metrics = snapshots["I_real"][step][0]
            prefix_norm_rows.extend(
                _collect_prefix_norm_rows_for_metrics(
                    suite_name=suite_name,
                    step=int(step),
                    arm_alias="I_real",
                    metrics=snapshot_metrics,
                )
            )
            shuffle_snapshot_metrics = snapshots["I_shuffle"][step][0]
            prefix_norm_rows.extend(
                _collect_prefix_norm_rows_for_metrics(
                    suite_name=suite_name,
                    step=int(step),
                    arm_alias="I_shuffle",
                    metrics=shuffle_snapshot_metrics,
                )
            )
            zero_metrics = snapshots["I_zero"].get(0, (runs["I_zero"][0], static_rows["I_zero"]))[0]
            prefix_norm_rows.extend(
                _collect_prefix_norm_rows_for_metrics(
                    suite_name=suite_name,
                    step=int(step),
                    arm_alias="I_zero",
                    metrics=zero_metrics,
                )
            )
            real_summary = next(row for row in summary_rows if row["suite"] == suite_name and row["step"] == step and row["alias"] == "I_real")
            shuffle_summary = next(row for row in summary_rows if row["suite"] == suite_name and row["step"] == step and row["alias"] == "I_shuffle")
            zero_summary = next(row for row in summary_rows if row["suite"] == suite_name and row["step"] == step and row["alias"] == "I_zero")
            a_summary = next(row for row in summary_rows if row["suite"] == suite_name and row["step"] == step and row["alias"] == "A")
            regressions_vs_base = int(step_pairwise_lookup[("A", "I_real")]["left_correct_to_right_wrong"])
            passes = bool(
                int(step_pairwise_lookup[("I_shuffle", "I_real")]["flip_count_delta"]) >= 2
                and (float(real_summary["macro_f1"]) - float(shuffle_summary["macro_f1"])) >= 0.03
                and int(step_pairwise_lookup[("I_zero", "I_real")]["flip_count_delta"]) >= 2
                and (float(real_summary["macro_f1"]) - float(zero_summary["macro_f1"])) >= 0.03
                and float(real_summary["task_score"]) >= float(a_summary["task_score"])
                and regressions_vs_base <= 1
            )
            candidate_selections.append(
                {
                    "suite": suite_name,
                    "step": int(step),
                    "support_serialization_variant": suite_support_variant[suite_name],
                    "passes_selection": passes,
                    "task_score": float(real_summary["task_score"]),
                    "macro_f1": float(real_summary["macro_f1"]),
                    "regressions_vs_base": regressions_vs_base,
                    "flip_gain_vs_shuffle": int(step_pairwise_lookup[("I_shuffle", "I_real")]["flip_count_delta"]),
                    "flip_gain_vs_zero": int(step_pairwise_lookup[("I_zero", "I_real")]["flip_count_delta"]),
                    "i_real_checkpoint_path": str(snapshots["I_real"][step][0].get("checkpoint_path", "")),
                    "i_shuffle_checkpoint_path": str(snapshots["I_shuffle"][step][0].get("checkpoint_path", "")),
                }
            )
        for arm_alias, train_events in sorted(train_events_by_alias.items()):
            prefix_norm_rows.extend(
                _collect_train_event_norm_rows(
                    suite_name=suite_name,
                    arm_alias=arm_alias,
                    train_events=train_events,
                )
            )

    passed_candidates = [row for row in candidate_selections if bool(row["passes_selection"])]
    selected = None
    if passed_candidates:
        selected = sorted(
            passed_candidates,
            key=lambda row: (
                int(row["step"]),
                -float(row["macro_f1"]),
                int(row["regressions_vs_base"]),
            ),
        )[0]
    selection_payload = {
        "selection_passed": selected is not None,
        "selected_suite": None if selected is None else str(selected["suite"]),
        "selected_step": None if selected is None else int(selected["step"]),
        "selected_support_serialization": None if selected is None else str(selected["support_serialization_variant"]),
        "selected_prompt_variant": None,
        "screen248_test_gate_passed": False,
        "support_bank_brittle": False,
        "heldout_sane_bank_count": 0,
        "fixed64_report_generated": False,
        "fixed64_gate_passed": False,
        "milestone_gate_passed": False,
        "i_real_checkpoint_path": "" if selected is None else str(selected["i_real_checkpoint_path"]),
        "i_shuffle_checkpoint_path": "" if selected is None else str(selected["i_shuffle_checkpoint_path"]),
        "candidate_rows": candidate_selections,
    }
    if selected is not None:
        selected_suite_root = suite_roots[str(selected["suite"])]
        selected_runs = _collect_shared_injection_runs(selected_suite_root)
        selection_payload["selected_prompt_variant"] = str(selected_runs["I_real"][0].get("prompt_variant", ""))

    attention_rows: list[dict[str, Any]] = []
    audit_dataset_path = str(config["runtime"].get("pilot_val_audit_dataset_path", "")).strip()
    if audit_dataset_path:
        support_examples = _load_task_dataset_with_path(config, str(config["task"]["support_dataset_path"]))
        lookup_paths = _resolve_support_lookup_dataset_paths(
            config,
            default_paths=[
                str(config["task"]["support_dataset_path"]),
                str(config["task"].get("train_dataset_path", config["task"]["support_dataset_path"])),
                str(config["task"]["dataset_path"]),
            ],
        )
        lookup_examples: list[dict[str, Any]] = []
        for lookup_path in lookup_paths:
            lookup_examples.extend(_load_task_dataset_with_path(config, lookup_path))
        for suite_name, suite_root in sorted(suite_roots.items()):
            runs = _collect_shared_injection_runs(suite_root)
            prompt_variant = str(runs["I_real"][0].get("prompt_variant", ""))
            audit_examples = _load_audit_examples(
                config,
                audit_dataset_path,
                prompt_variant=prompt_variant,
            )
            attention_lookup = _merge_example_lookup(
                support_examples,
                [cache.example for cache in audit_examples],
                lookup_examples,
            )
            snapshots = {alias: _load_snapshot_eval_rows(run_dir) for alias, (_, run_dir) in runs.items()}
            for step in sorted(set(snapshots["I_real"]) & set(snapshots["I_shuffle"])):
                for arm_alias, memory_control in (("I_real", "real"), ("I_shuffle", "shuffled"), ("I_zero", "zero")):
                    checkpoint_path = ""
                    if arm_alias in {"I_real", "I_shuffle"}:
                        checkpoint_path = str(snapshots[arm_alias][step][0].get("checkpoint_path", ""))
                    attention_rows.extend(
                        _attention_rows_for_snapshot(
                            config=config,
                            support_examples=support_examples,
                            audit_examples=audit_examples,
                            example_lookup=attention_lookup,
                            support_serialization_variant=suite_support_variant[suite_name],
                            writer_memory_control=memory_control,
                            checkpoint_path=checkpoint_path,
                            step=int(step),
                            suite_name=suite_name,
                            arm_alias=arm_alias,
                        )
                    )

    summary_csv = output_dir / "dynamics_recovery_summary.csv"
    pairwise_csv = output_dir / "dynamics_recovery_pairwise.csv"
    content_gap_csv = output_dir / "content_gap_curve.csv"
    prefix_norm_csv = output_dir / "prefix_norm_drift.csv"
    attention_csv = output_dir / "prefix_attention_consumption.csv"
    summary_plot = output_dir / "summary.svg"
    selection_json = output_dir / "selection.json"
    dual_gate_summary_json = output_dir / "dual_gate_summary.json"
    _write_csv(summary_csv, summary_rows)
    _write_csv(pairwise_csv, pairwise_rows)
    _write_csv(content_gap_csv, content_gap_rows)
    _write_csv(prefix_norm_csv, prefix_norm_rows)
    _write_csv(attention_csv, attention_rows)
    write_sanity_plot(
        summary_plot,
        [
            {
                "mode": f"{row['suite']}:I_real:step{int(row['step']):04d}",
                "run_dir": f"{row['suite']}-I_real-step{int(row['step']):04d}",
                "primary_metric": "macro_f1",
                "primary_score": float(row["macro_f1"]),
            }
            for row in summary_rows
            if row["alias"] == "I_real"
        ],
    )
    write_json(selection_json, selection_payload)
    write_json(
        dual_gate_summary_json,
        {
            "selection_passed": bool(selection_payload["selection_passed"]),
            "screen248_test_gate_passed": bool(selection_payload["screen248_test_gate_passed"]),
            "support_bank_brittle": bool(selection_payload["support_bank_brittle"]),
            "heldout_sane_bank_count": int(selection_payload["heldout_sane_bank_count"]),
            "fixed64_report_generated": bool(selection_payload["fixed64_report_generated"]),
            "fixed64_gate_passed": bool(selection_payload["fixed64_gate_passed"]),
            "milestone_gate_passed": bool(selection_payload["milestone_gate_passed"]),
        },
    )
    report_lines.extend(
        [
            f"- selection_passed: {selection_payload['selection_passed']}",
            f"- selected_suite: {selection_payload['selected_suite']}",
            f"- selected_step: {selection_payload['selected_step']}",
            f"- selected_support_serialization: {selection_payload['selected_support_serialization']}",
            f"- screen248_test_gate_passed: {selection_payload['screen248_test_gate_passed']}",
            f"- support_bank_brittle: {selection_payload['support_bank_brittle']}",
            f"- heldout_sane_bank_count: {selection_payload['heldout_sane_bank_count']}",
            f"- fixed64_report_generated: {selection_payload['fixed64_report_generated']}",
            f"- fixed64_gate_passed: {selection_payload['fixed64_gate_passed']}",
            f"- milestone_gate_passed: {selection_payload['milestone_gate_passed']}",
            "",
            "## Candidate Checkpoints",
        ]
    )
    for row in candidate_selections:
        report_lines.append(
            f"- {row['suite']} step{int(row['step']):04d}: passes={row['passes_selection']}, "
            f"macro_f1={row['macro_f1']:.4f}, flip_gain_vs_shuffle={row['flip_gain_vs_shuffle']}, "
            f"flip_gain_vs_zero={row['flip_gain_vs_zero']}, regressions_vs_base={row['regressions_vs_base']}"
        )
    report_path = output_dir / "val_selection_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    write_json(
        output_dir / "metrics.json",
        {
            "analysis_mode": "m4_shared_injection_dynamics_recovery",
            "selection_passed": bool(selection_payload["selection_passed"]),
            "selected_suite": selection_payload["selected_suite"],
            "selected_step": selection_payload["selected_step"],
            "selected_support_serialization": selection_payload["selected_support_serialization"],
            "support_bank_brittle": bool(selection_payload["support_bank_brittle"]),
            "heldout_sane_bank_count": int(selection_payload["heldout_sane_bank_count"]),
            "fixed64_report_generated": bool(selection_payload["fixed64_report_generated"]),
            "summary_csv": str(summary_csv.resolve()),
            "pairwise_csv": str(pairwise_csv.resolve()),
            "content_gap_csv": str(content_gap_csv.resolve()),
            "prefix_norm_csv": str(prefix_norm_csv.resolve()),
            "attention_csv": str(attention_csv.resolve()),
            "summary_plot": str(summary_plot.resolve()),
            "selection_json": str(selection_json.resolve()),
            "dual_gate_summary_json": str(dual_gate_summary_json.resolve()),
            "report_path": str(report_path.resolve()),
        },
    )


def _load_optional_metrics(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def _load_csv_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _onset_step_from_prefix_norm_csv(prefix_norm_csv: str, *, total_cap: float) -> int | None:
    if total_cap <= 0.0:
        return None
    threshold = total_cap * 0.95
    rows = _load_csv_rows(prefix_norm_csv)
    matching_steps = sorted(
        {
            int(row["step"])
            for row in rows
            if row.get("row_type") == "snapshot_aggregate"
            and row.get("arm_alias") == "I_real"
            and float(row.get("prefix_l2", 0.0) or 0.0) >= threshold
        }
    )
    return None if not matching_steps else matching_steps[0]


def _onset_step_from_dominant_label_csv(summary_csv: str) -> int | None:
    rows = _load_csv_rows(summary_csv)
    matching_steps = sorted(
        {
            int(row["step"])
            for row in rows
            if row.get("alias") == "I_real"
            and float(row.get("dominant_label_fraction", 0.0) or 0.0) >= 0.99
        }
    )
    return None if not matching_steps else matching_steps[0]


def _heldout_bank_is_sane(compare_metrics: dict[str, Any]) -> bool:
    if not compare_metrics:
        return False
    return bool(
        float(compare_metrics.get("i_real_task_score", 0.0)) >= float(compare_metrics.get("a_task_score", 0.0))
        and int(compare_metrics.get("flip_gain_vs_shuffle", -10**9)) >= 0
        and int(compare_metrics.get("flip_gain_vs_zero", -10**9)) >= 0
        and int(compare_metrics.get("regressions_vs_base", 10**9)) <= 4
    )


def summarize_m4_support_bank_run(
    *,
    selection_json: str,
    run_metrics_json: str,
    dynamics_summary_csv: str,
    prefix_norm_csv: str,
    screen248_test_metrics_json: str | None = None,
    heldout_metrics_by_name: dict[str, str] | None = None,
    fixed64_metrics_json: str | None = None,
) -> dict[str, Any]:
    selection = json.loads(Path(selection_json).read_text())
    run_metrics = json.loads(Path(run_metrics_json).read_text())
    screen248_test_metrics = _load_optional_metrics(screen248_test_metrics_json)
    heldout_results: dict[str, Any] = {}
    sane_count = 0
    for name, path in sorted((heldout_metrics_by_name or {}).items()):
        compare_metrics = _load_optional_metrics(path)
        sane = _heldout_bank_is_sane(compare_metrics)
        sane_count += int(sane)
        heldout_results[name] = {
            "metrics_path": str(Path(path).resolve()),
            "gate_passed": bool(compare_metrics.get("gate_passed", False)),
            "sane": sane,
            "regressions_vs_base": int(compare_metrics.get("regressions_vs_base", 0)),
            "flip_gain_vs_shuffle": int(compare_metrics.get("flip_gain_vs_shuffle", 0)),
            "flip_gain_vs_zero": int(compare_metrics.get("flip_gain_vs_zero", 0)),
            "i_real_task_score": float(compare_metrics.get("i_real_task_score", 0.0)),
            "a_task_score": float(compare_metrics.get("a_task_score", 0.0)),
        }
    screen248_test_gate_passed = bool(screen248_test_metrics.get("gate_passed", False))
    support_bank_brittle = bool(
        screen248_test_gate_passed and heldout_results and sane_count == 0
    )
    fixed64_metrics = _load_optional_metrics(fixed64_metrics_json)
    fixed64_report_generated = bool(fixed64_metrics_json)
    total_cap = float(run_metrics.get("pilot_prefix_total_max_norm", 0.0))
    summary = {
        "selection_passed": bool(selection.get("selection_passed", False)),
        "selected_suite": selection.get("selected_suite"),
        "selected_step": selection.get("selected_step"),
        "selected_support_serialization": selection.get("selected_support_serialization"),
        "selected_prompt_variant": selection.get("selected_prompt_variant"),
        "screen248_test_gate_passed": screen248_test_gate_passed,
        "screen248_test_metrics_path": (
            "" if not screen248_test_metrics_json else str(Path(screen248_test_metrics_json).resolve())
        ),
        "heldout_results": heldout_results,
        "heldout_sane_bank_count": sane_count,
        "support_bank_brittle": support_bank_brittle,
        "fixed64_report_generated": fixed64_report_generated,
        "fixed64_gate_passed": bool(fixed64_metrics.get("gate_passed", False)),
        "fixed64_metrics_path": "" if not fixed64_metrics_json else str(Path(fixed64_metrics_json).resolve()),
        "milestone_gate_passed": bool(
            selection.get("selection_passed", False)
            and screen248_test_gate_passed
            and not support_bank_brittle
        ),
        "cap_saturation_onset_step": _onset_step_from_prefix_norm_csv(prefix_norm_csv, total_cap=total_cap),
        "dominant_label_collapse_onset_step": _onset_step_from_dominant_label_csv(dynamics_summary_csv),
        "pilot_prefix_total_max_norm": total_cap,
    }
    return summary


def compare_m4_anti_shortcut_runs(
    *,
    run_a_summary_json: str,
    run_b_summary_json: str,
) -> dict[str, Any]:
    run_a = json.loads(Path(run_a_summary_json).read_text())
    run_b = json.loads(Path(run_b_summary_json).read_text())
    conclusion = "run_a_equals_run_b"
    if bool(run_a.get("milestone_gate_passed")) and not bool(run_b.get("milestone_gate_passed")):
        conclusion = "run_a_passes_run_b_fails"
    elif not bool(run_a.get("milestone_gate_passed")) and not bool(run_b.get("milestone_gate_passed")):
        a_cap = run_a.get("cap_saturation_onset_step")
        b_cap = run_b.get("cap_saturation_onset_step")
        a_collapse = run_a.get("dominant_label_collapse_onset_step")
        b_collapse = run_b.get("dominant_label_collapse_onset_step")
        a_less_saturated = a_cap is None or (b_cap is not None and int(a_cap) > int(b_cap))
        a_less_collapsed = a_collapse is None or (b_collapse is not None and int(a_collapse) > int(b_collapse))
        if a_less_saturated or a_less_collapsed:
            conclusion = "both_fail_run_a_less_collapsed"
    return {
        "run_a_selection_passed": bool(run_a.get("selection_passed", False)),
        "run_b_selection_passed": bool(run_b.get("selection_passed", False)),
        "run_a_primary_gate_passed": bool(run_a.get("screen248_test_gate_passed", False)),
        "run_b_primary_gate_passed": bool(run_b.get("screen248_test_gate_passed", False)),
        "run_a_support_bank_brittle": bool(run_a.get("support_bank_brittle", False)),
        "run_b_support_bank_brittle": bool(run_b.get("support_bank_brittle", False)),
        "run_a_fixed64_gate_passed": bool(run_a.get("fixed64_gate_passed", False)),
        "run_b_fixed64_gate_passed": bool(run_b.get("fixed64_gate_passed", False)),
        "run_a_selected_step": run_a.get("selected_step"),
        "run_b_selected_step": run_b.get("selected_step"),
        "run_a_cap_saturation_onset_step": run_a.get("cap_saturation_onset_step"),
        "run_b_cap_saturation_onset_step": run_b.get("cap_saturation_onset_step"),
        "run_a_dominant_label_collapse_onset_step": run_a.get("dominant_label_collapse_onset_step"),
        "run_b_dominant_label_collapse_onset_step": run_b.get("dominant_label_collapse_onset_step"),
        "comparison_conclusion": conclusion,
    }


def compare_m4_alignment_runs(
    *,
    canonical_summary_json: str,
    freeze_writer_summary_json: str,
    pooled_block_summary_json: str,
) -> dict[str, Any]:
    canonical = json.loads(Path(canonical_summary_json).read_text())
    freeze_writer = json.loads(Path(freeze_writer_summary_json).read_text())
    pooled_block = json.loads(Path(pooled_block_summary_json).read_text())

    canonical_primary = bool(canonical.get("screen248_test_gate_passed", False))
    canonical_not_brittle = not bool(canonical.get("support_bank_brittle", False))
    freeze_primary = bool(freeze_writer.get("screen248_test_gate_passed", False))
    pooled_primary = bool(pooled_block.get("screen248_test_gate_passed", False))
    alignment_claim_supported = bool(
        canonical_primary
        and canonical_not_brittle
        and not freeze_primary
        and not pooled_primary
    )
    if alignment_claim_supported:
        conclusion = "canonical_passes_both_ablations_fail"
    elif canonical_primary and (freeze_primary or pooled_primary):
        conclusion = "canonical_pass_ambiguous"
    elif bool(canonical.get("selection_passed", False)):
        conclusion = "canonical_selected_but_primary_gate_failed"
    else:
        conclusion = "canonical_failed_selection"

    return {
        "canonical_selection_passed": bool(canonical.get("selection_passed", False)),
        "canonical_selected_step": canonical.get("selected_step"),
        "canonical_primary_gate_passed": canonical_primary,
        "canonical_support_bank_brittle": bool(canonical.get("support_bank_brittle", False)),
        "canonical_fixed64_report_generated": bool(canonical.get("fixed64_report_generated", False)),
        "canonical_fixed64_gate_passed": bool(canonical.get("fixed64_gate_passed", False)),
        "freeze_writer_selection_passed": bool(freeze_writer.get("selection_passed", False)),
        "freeze_writer_selected_step": freeze_writer.get("selected_step"),
        "freeze_writer_primary_gate_passed": freeze_primary,
        "pooled_block_selection_passed": bool(pooled_block.get("selection_passed", False)),
        "pooled_block_selected_step": pooled_block.get("selected_step"),
        "pooled_block_primary_gate_passed": pooled_primary,
        "alignment_claim_supported": alignment_claim_supported,
        "comparison_conclusion": conclusion,
    }


def compare_m5_alignment_runs(
    *,
    canonical_summary_json: str,
    freeze_writer_summary_json: str,
    pooled_block_summary_json: str,
) -> dict[str, Any]:
    canonical = json.loads(Path(canonical_summary_json).read_text())
    freeze_writer = json.loads(Path(freeze_writer_summary_json).read_text())
    pooled_block = json.loads(Path(pooled_block_summary_json).read_text())

    canonical_selection = bool(canonical.get("selection_passed", False))
    canonical_primary = bool(canonical.get("screen248_test_gate_passed", False))
    canonical_brittle = bool(canonical.get("support_bank_brittle", False))
    freeze_primary = bool(freeze_writer.get("screen248_test_gate_passed", False))
    pooled_primary = bool(pooled_block.get("screen248_test_gate_passed", False))
    success = bool(
        canonical_primary
        and not canonical_brittle
        and not freeze_primary
        and not pooled_primary
    )
    ambiguous = bool(
        canonical_primary
        and not canonical_brittle
        and (freeze_primary or pooled_primary)
    )
    if success:
        conclusion = "success"
        failure_reason = ""
    elif ambiguous:
        conclusion = "ambiguous_pass"
        failure_reason = ""
    else:
        conclusion = "failure"
        if not canonical_selection:
            failure_reason = "canonical_failed_selection"
        elif not canonical_primary:
            failure_reason = "canonical_selected_but_primary_gate_failed"
        elif canonical_brittle:
            failure_reason = "canonical_support_bank_brittle"
        else:
            failure_reason = "canonical_did_not_beat_ablations"

    return {
        "canonical_selection_passed": canonical_selection,
        "canonical_selected_step": canonical.get("selected_step"),
        "canonical_primary_gate_passed": canonical_primary,
        "canonical_support_bank_brittle": canonical_brittle,
        "canonical_fixed64_report_generated": bool(canonical.get("fixed64_report_generated", False)),
        "canonical_fixed64_gate_passed": bool(canonical.get("fixed64_gate_passed", False)),
        "freeze_writer_selection_passed": bool(freeze_writer.get("selection_passed", False)),
        "freeze_writer_selected_step": freeze_writer.get("selected_step"),
        "freeze_writer_primary_gate_passed": freeze_primary,
        "pooled_block_selection_passed": bool(pooled_block.get("selection_passed", False)),
        "pooled_block_selected_step": pooled_block.get("selected_step"),
        "pooled_block_primary_gate_passed": pooled_primary,
        "alignment_claim_supported": success,
        "comparison_conclusion": conclusion,
        "failure_reason": failure_reason,
    }


def compare_m5_objective_runs(
    *,
    canonical_summary_json: str,
    anchor_only_summary_json: str,
    task_only_control_summary_json: str,
) -> dict[str, Any]:
    canonical = json.loads(Path(canonical_summary_json).read_text())
    anchor_only = json.loads(Path(anchor_only_summary_json).read_text())
    task_only_control = json.loads(Path(task_only_control_summary_json).read_text())

    canonical_selection = bool(canonical.get("selection_passed", False))
    canonical_primary = bool(canonical.get("screen248_test_gate_passed", False))
    canonical_brittle = bool(canonical.get("support_bank_brittle", False))
    anchor_primary = bool(anchor_only.get("screen248_test_gate_passed", False))
    task_only_primary = bool(task_only_control.get("screen248_test_gate_passed", False))
    objective_rewrite_supported = bool(canonical_primary and not task_only_primary)
    teacher_margin_increment_supported = bool(canonical_primary and not anchor_primary)
    if objective_rewrite_supported and teacher_margin_increment_supported:
        conclusion = "success"
        failure_reason = ""
    elif objective_rewrite_supported and anchor_primary:
        conclusion = "anchor_supported_teacher_optional"
        failure_reason = ""
    else:
        conclusion = "failure"
        if not canonical_selection:
            failure_reason = "canonical_failed_selection"
        elif not canonical_primary:
            failure_reason = "canonical_selected_but_primary_gate_failed"
        elif canonical_brittle:
            failure_reason = "canonical_support_bank_brittle"
        elif task_only_primary:
            failure_reason = "task_only_control_also_passed"
        else:
            failure_reason = "anchor_only_or_teacher_margin_did_not_improve"

    return {
        "canonical_selection_passed": canonical_selection,
        "canonical_selected_step": canonical.get("selected_step"),
        "canonical_primary_gate_passed": canonical_primary,
        "canonical_support_bank_brittle": canonical_brittle,
        "canonical_fixed64_report_generated": bool(canonical.get("fixed64_report_generated", False)),
        "canonical_fixed64_gate_passed": bool(canonical.get("fixed64_gate_passed", False)),
        "anchor_only_selection_passed": bool(anchor_only.get("selection_passed", False)),
        "anchor_only_selected_step": anchor_only.get("selected_step"),
        "anchor_only_primary_gate_passed": anchor_primary,
        "task_only_control_selection_passed": bool(task_only_control.get("selection_passed", False)),
        "task_only_control_selected_step": task_only_control.get("selected_step"),
        "task_only_control_primary_gate_passed": task_only_primary,
        "objective_rewrite_supported": objective_rewrite_supported,
        "teacher_margin_increment_supported": teacher_margin_increment_supported,
        "comparison_conclusion": conclusion,
        "failure_reason": failure_reason,
    }
