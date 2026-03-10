#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".tmp" / "github-review-snapshot"
RESULTS_ROOT = ROOT / "results" / "generated" / "review"
MAX_SNAPSHOT_BYTES = 31 * 1024 * 1024
MAX_RESULT_FILE_BYTES = 1_000_000
ALLOWED_RESULT_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".png", ".svg", ".tex"}
EXCLUDED_RESULT_FILENAMES = {
    "task_case_dump.jsonl",
    "train_events.json",
}
ROOT_FILES = [
    "README.md",
    "AGENTS.md",
    "PLANv6.md",
]
STATIC_REVIEW_DIRS = [
    "writer-circuit-opening-qwen25",
    "writer-deep-prefix-jointpeft-qwen25",
]


def _git_commit() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dest_root: Path, *, manifest: list[dict[str, object]]) -> None:
    rel = src.relative_to(ROOT)
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    manifest.append(
        {
            "path": rel.as_posix(),
            "size_bytes": src.stat().st_size,
        }
    )


def _copy_doc_tree(dest_root: Path, *, manifest: list[dict[str, object]]) -> None:
    docs_root = ROOT / "docs"
    for src in sorted(path for path in docs_root.rglob("*") if path.is_file()):
        _copy_file(src, dest_root, manifest=manifest)


def _result_file_allowed(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in EXCLUDED_RESULT_FILENAMES:
        return False
    if path.suffix not in ALLOWED_RESULT_SUFFIXES:
        return False
    return path.stat().st_size <= MAX_RESULT_FILE_BYTES


def _completed_planv6_review_dirs() -> list[str]:
    completed: list[str] = []
    for path in sorted(RESULTS_ROOT.glob("planv6-v6-*-qwen25")):
        if not path.is_dir():
            continue
        if any(path.glob("*summary.json")) or any(path.glob("*summary.md")):
            completed.append(path.name)
    return completed


def _review_dirs_to_copy() -> list[str]:
    selected = list(STATIC_REVIEW_DIRS)
    selected.extend(_completed_planv6_review_dirs())
    deduped: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        if (RESULTS_ROOT / name).is_dir():
            deduped.append(name)
    return deduped


def _copy_review_results(dest_root: Path, *, manifest: list[dict[str, object]]) -> list[str]:
    copied_dirs: list[str] = []
    for dirname in _review_dirs_to_copy():
        src_root = RESULTS_ROOT / dirname
        copied_any = False
        for src in sorted(path for path in src_root.rglob("*") if _result_file_allowed(path)):
            _copy_file(src, dest_root, manifest=manifest)
            copied_any = True
        if copied_any:
            copied_dirs.append(dirname)
    return copied_dirs


def _total_snapshot_bytes(dest_root: Path) -> int:
    return sum(path.stat().st_size for path in dest_root.rglob("*") if path.is_file())


def _zip_snapshot_bytes(dest_root: Path) -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "review-snapshot.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for src in sorted(path for path in dest_root.rglob("*") if path.is_file()):
                archive.write(src, arcname=src.relative_to(dest_root))
        return archive_path.stat().st_size


def build_snapshot(output_root: Path, *, source_commit: str | None = None) -> dict[str, object]:
    source_commit = source_commit or _git_commit()
    _reset_output_dir(output_root)

    manifest_entries: list[dict[str, object]] = []
    for relative_path in ROOT_FILES:
        src = ROOT / relative_path
        if src.is_file():
            _copy_file(src, output_root, manifest=manifest_entries)
    _copy_doc_tree(output_root, manifest=manifest_entries)
    copied_review_dirs = _copy_review_results(output_root, manifest=manifest_entries)

    total_bytes = _total_snapshot_bytes(output_root)
    zip_size_bytes = _zip_snapshot_bytes(output_root)

    manifest = {
        "source_commit": source_commit,
        "max_snapshot_bytes": MAX_SNAPSHOT_BYTES,
        "max_result_file_bytes": MAX_RESULT_FILE_BYTES,
        "total_size_bytes": total_bytes,
        "zip_size_bytes": zip_size_bytes,
        "review_dirs": copied_review_dirs,
        "included_files": manifest_entries,
        "notes": [
            "This snapshot is the lightweight GitHub review export.",
            "It intentionally excludes raw traces such as train_events.json and task_case_dump.jsonl.",
            "It does not rewrite or shrink the full local working repository history.",
        ],
    }
    manifest_path = output_root / "REVIEW_SNAPSHOT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total_bytes = _total_snapshot_bytes(output_root)
    zip_size_bytes = _zip_snapshot_bytes(output_root)
    if total_bytes > MAX_SNAPSHOT_BYTES:
        raise SystemExit(
            f"Snapshot exceeds budget: {total_bytes} bytes > {MAX_SNAPSHOT_BYTES} bytes"
        )
    if zip_size_bytes > MAX_SNAPSHOT_BYTES:
        raise SystemExit(
            f"Snapshot zip exceeds budget: {zip_size_bytes} bytes > {MAX_SNAPSHOT_BYTES} bytes"
        )
    return {
        **manifest,
        "total_size_bytes": total_bytes,
        "zip_size_bytes": zip_size_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    source_commit = args.source_commit.strip() or None
    manifest = build_snapshot(output_root, source_commit=source_commit)
    if args.print_summary:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "source_commit": manifest["source_commit"],
                    "total_size_bytes": manifest["total_size_bytes"],
                    "review_dirs": manifest["review_dirs"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
