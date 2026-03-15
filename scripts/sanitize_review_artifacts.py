#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
REPLACEMENT = b"[REDACTED_GITHUB_TOKEN]"
TOKEN_PATTERNS = (
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
)


def _should_scan(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS


def _sanitize_file(path: Path) -> int:
    payload = path.read_bytes()
    total_replacements = 0
    sanitized = payload
    for pattern in TOKEN_PATTERNS:
        sanitized, replacements = pattern.subn(REPLACEMENT, sanitized)
        total_replacements += int(replacements)
    if total_replacements > 0 and sanitized != payload:
        path.write_bytes(sanitized)
    return total_replacements


def sanitize_paths(paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not _should_scan(path):
                continue
            replacements = _sanitize_file(path)
            if replacements > 0:
                counts[str(path)] = replacements
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Redact token-like secrets from review artifacts.")
    parser.add_argument("paths", nargs="+")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    counts = sanitize_paths([Path(raw).resolve() for raw in args.paths])
    for path, replacements in counts.items():
        print(f"sanitized {path} replacements={replacements}")
    print(f"sanitized_files={len(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
