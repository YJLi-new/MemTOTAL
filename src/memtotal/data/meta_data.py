from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memtotal.data.toy_data import load_toy_dataset


@dataclass
class Episode:
    domain: str
    support_examples: list[dict[str, str]]
    query_examples: list[dict[str, str]]


def compute_dataset_sha256(dataset_path: str | Path) -> str:
    path = Path(dataset_path).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_domain_dataset(dataset_path: str | Path) -> dict[str, list[dict[str, str]]]:
    dataset = load_toy_dataset(dataset_path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in dataset:
        domain = str(row["domain"])
        grouped.setdefault(domain, []).append(row)
    return grouped


def load_meta_grouped_examples(task_cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    meta_cfg = task_cfg.get("meta", {})
    dataset_sources = meta_cfg.get("dataset_sources")
    if not dataset_sources:
        return load_domain_dataset(task_cfg["dataset_path"])

    # Delayed import avoids a package cycle with memtotal.tasks.registry.
    from memtotal.tasks import load_task_dataset

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_source in dataset_sources:
        source = dict(raw_source)
        benchmark_id = str(source["benchmark_id"])
        domain = str(source["domain"])
        source_task_cfg: dict[str, Any] = {
            "name": str(source.get("task_name", f"{benchmark_id}_{domain}_meta")),
            "benchmark_id": benchmark_id,
            "dataset_path": source["dataset_path"],
        }
        if "smoke_subset" in source:
            source_task_cfg["smoke_subset"] = source["smoke_subset"]
        if "metric_name" in source:
            source_task_cfg["metric_name"] = source["metric_name"]
        if "evaluator" in source:
            source_task_cfg["evaluator"] = source["evaluator"]
        if "narrativeqa_runtime" in source:
            source_task_cfg["narrativeqa_runtime"] = source["narrativeqa_runtime"]
        rows = load_task_dataset({"task": source_task_cfg})
        for row in rows:
            canonical_row = dict(row)
            canonical_row["domain"] = domain
            canonical_row["benchmark_id"] = benchmark_id
            grouped.setdefault(domain, []).append(canonical_row)
    return grouped


def build_meta_manifest(
    *,
    dataset_path: str | Path | None = None,
    dataset_sources: list[dict[str, Any]] | None = None,
    grouped_examples: dict[str, list[dict[str, str]]],
    general_domains: list[str],
    source_domains: list[str],
    target_domain: str,
    support_size: int,
    query_size: int,
    sampling_policy: str = "stratified_labels",
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "general_domains": general_domains,
        "source_domains": source_domains,
        "target_domain": target_domain,
        "support_size": support_size,
        "query_size": query_size,
        "examples_per_domain": {domain: len(rows) for domain, rows in grouped_examples.items()},
        "sampling_policy": sampling_policy,
    }
    if dataset_sources:
        normalized_sources: list[dict[str, object]] = []
        dataset_sha256s: dict[str, str] = {}
        benchmarks_by_domain: dict[str, str] = {}
        combined = hashlib.sha256()
        for source in dataset_sources:
            resolved_path = str(Path(source["dataset_path"]).resolve())
            benchmark_id = str(source["benchmark_id"])
            domain = str(source["domain"])
            normalized_source = {
                "benchmark_id": benchmark_id,
                "dataset_path": resolved_path,
                "domain": domain,
            }
            if "smoke_subset" in source:
                normalized_source["smoke_subset"] = source["smoke_subset"]
            normalized_sources.append(normalized_source)
            source_key = f"{domain}:{benchmark_id}"
            source_sha256 = compute_dataset_sha256(resolved_path)
            dataset_sha256s[source_key] = source_sha256
            benchmarks_by_domain[domain] = benchmark_id
            combined.update(source_key.encode("utf-8"))
            combined.update(source_sha256.encode("utf-8"))
        manifest["dataset_sources"] = normalized_sources
        manifest["dataset_sha256s"] = dataset_sha256s
        manifest["benchmarks_by_domain"] = benchmarks_by_domain
        manifest["dataset_sha256"] = combined.hexdigest()
        return manifest

    if dataset_path is None:
        raise ValueError("build_meta_manifest requires dataset_path when dataset_sources are not provided.")
    manifest["dataset_path"] = str(Path(dataset_path).resolve())
    manifest["dataset_sha256"] = compute_dataset_sha256(dataset_path)
    return manifest


def validate_meta_split(
    grouped_examples: dict[str, list[dict[str, str]]],
    *,
    general_domains: list[str],
    source_domains: list[str],
    target_domain: str,
    support_size: int,
    query_size: int,
    sampling_policy: str = "stratified_labels",
) -> None:
    required_domains = set(general_domains) | set(source_domains) | {target_domain}
    missing = sorted(domain for domain in required_domains if domain not in grouped_examples)
    if missing:
        raise ValueError(f"Meta split refers to missing domains: {missing}")
    for domain in set(source_domains) | {target_domain}:
        available = len(grouped_examples[domain])
        if available < support_size + query_size:
            raise ValueError(
                f"Domain '{domain}' has only {available} examples, "
                f"but support_size + query_size = {support_size + query_size}."
            )
        if sampling_policy == "uniform_examples":
            continue
        label_groups: dict[str, list[dict[str, str]]] = {}
        for row in grouped_examples[domain]:
            label_groups.setdefault(str(row["label"]), []).append(row)
        num_labels = len(label_groups)
        if num_labels == 0:
            raise ValueError(f"Domain '{domain}' has no labels for stratified sampling.")
        if support_size % num_labels != 0 or query_size % num_labels != 0:
            raise ValueError(
                f"support_size={support_size} and query_size={query_size} must be divisible by "
                f"the number of labels ({num_labels}) in domain '{domain}'."
            )
        per_label = (support_size // num_labels) + (query_size // num_labels)
        for label, rows in label_groups.items():
            if len(rows) < per_label:
                raise ValueError(
                    f"Label '{label}' in domain '{domain}' has only {len(rows)} examples, "
                    f"but needs at least {per_label} for stratified support/query sampling."
                )
    if sampling_policy not in {"stratified_labels", "uniform_examples"}:
        raise ValueError(
            f"Unsupported sampling_policy={sampling_policy}. "
            "Expected one of stratified_labels, uniform_examples."
        )


class EpisodeSampler:
    def __init__(
        self,
        grouped_examples: dict[str, list[dict[str, str]]],
        *,
        source_domains: list[str],
        support_size: int,
        query_size: int,
        seed: int,
        sampling_policy: str = "stratified_labels",
    ) -> None:
        self.grouped_examples = grouped_examples
        self.source_domains = list(source_domains)
        self.support_size = support_size
        self.query_size = query_size
        self.rng = random.Random(seed)
        self.sampling_policy = sampling_policy

    def sample_episode(self) -> Episode:
        domain = self.rng.choice(self.source_domains)
        support_examples, query_examples = _split_examples(
            self.grouped_examples[domain],
            support_size=self.support_size,
            query_size=self.query_size,
            rng=self.rng,
            sampling_policy=self.sampling_policy,
        )
        return Episode(
            domain=domain,
            support_examples=support_examples,
            query_examples=query_examples,
        )


def split_target_domain_examples(
    grouped_examples: dict[str, list[dict[str, str]]],
    *,
    target_domain: str,
    support_size: int,
    query_size: int,
    seed: int,
    sampling_policy: str = "stratified_labels",
) -> Episode:
    rng = random.Random(seed)
    support_examples, query_examples = _split_examples(
        grouped_examples[target_domain],
        support_size=support_size,
        query_size=query_size,
        rng=rng,
        sampling_policy=sampling_policy,
    )
    return Episode(
        domain=target_domain,
        support_examples=support_examples,
        query_examples=query_examples,
    )


def _split_examples(
    examples: list[dict[str, str]],
    *,
    support_size: int,
    query_size: int,
    rng: random.Random,
    sampling_policy: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if sampling_policy == "uniform_examples":
        shuffled = list(examples)
        rng.shuffle(shuffled)
        support_examples = shuffled[:support_size]
        query_examples = shuffled[support_size : support_size + query_size]
        return support_examples, query_examples
    if sampling_policy != "stratified_labels":
        raise ValueError(
            f"Unsupported sampling_policy={sampling_policy}. "
            "Expected one of stratified_labels, uniform_examples."
        )
    grouped_by_label: dict[str, list[dict[str, str]]] = {}
    for row in examples:
        grouped_by_label.setdefault(str(row["label"]), []).append(row)
    labels = sorted(grouped_by_label)
    support_per_label = support_size // len(labels)
    query_per_label = query_size // len(labels)
    support_examples: list[dict[str, str]] = []
    query_examples: list[dict[str, str]] = []
    for label in labels:
        shuffled = list(grouped_by_label[label])
        rng.shuffle(shuffled)
        support_examples.extend(shuffled[:support_per_label])
        query_examples.extend(shuffled[support_per_label : support_per_label + query_per_label])
    rng.shuffle(support_examples)
    rng.shuffle(query_examples)
    return support_examples, query_examples
