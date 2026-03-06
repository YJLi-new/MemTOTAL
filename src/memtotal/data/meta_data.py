from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

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


def build_meta_manifest(
    *,
    dataset_path: str | Path,
    grouped_examples: dict[str, list[dict[str, str]]],
    general_domains: list[str],
    source_domains: list[str],
    target_domain: str,
    support_size: int,
    query_size: int,
) -> dict[str, object]:
    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": compute_dataset_sha256(dataset_path),
        "general_domains": general_domains,
        "source_domains": source_domains,
        "target_domain": target_domain,
        "support_size": support_size,
        "query_size": query_size,
        "examples_per_domain": {domain: len(rows) for domain, rows in grouped_examples.items()},
    }


def validate_meta_split(
    grouped_examples: dict[str, list[dict[str, str]]],
    *,
    general_domains: list[str],
    source_domains: list[str],
    target_domain: str,
    support_size: int,
    query_size: int,
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


class EpisodeSampler:
    def __init__(
        self,
        grouped_examples: dict[str, list[dict[str, str]]],
        *,
        source_domains: list[str],
        support_size: int,
        query_size: int,
        seed: int,
    ) -> None:
        self.grouped_examples = grouped_examples
        self.source_domains = list(source_domains)
        self.support_size = support_size
        self.query_size = query_size
        self.rng = random.Random(seed)

    def sample_episode(self) -> Episode:
        domain = self.rng.choice(self.source_domains)
        support_examples, query_examples = _stratified_split(
            self.grouped_examples[domain],
            support_size=self.support_size,
            query_size=self.query_size,
            rng=self.rng,
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
) -> Episode:
    rng = random.Random(seed)
    support_examples, query_examples = _stratified_split(
        grouped_examples[target_domain],
        support_size=support_size,
        query_size=query_size,
        rng=rng,
    )
    return Episode(
        domain=target_domain,
        support_examples=support_examples,
        query_examples=query_examples,
    )


def _stratified_split(
    examples: list[dict[str, str]],
    *,
    support_size: int,
    query_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
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
