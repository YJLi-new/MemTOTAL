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
        shuffled = list(self.grouped_examples[domain])
        self.rng.shuffle(shuffled)
        support_examples = shuffled[: self.support_size]
        query_examples = shuffled[self.support_size : self.support_size + self.query_size]
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
    shuffled = list(grouped_examples[target_domain])
    rng.shuffle(shuffled)
    support_examples = shuffled[:support_size]
    query_examples = shuffled[support_size : support_size + query_size]
    return Episode(
        domain=target_domain,
        support_examples=support_examples,
        query_examples=query_examples,
    )
