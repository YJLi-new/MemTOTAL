#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".tmp" / "github-review-snapshot"
RESULTS_ROOT = ROOT / "results" / "generated" / "review"
RUNS_ROOT = ROOT / "runs" / "review"
MAX_SNAPSHOT_ZIP_BYTES = 31 * 1024 * 1024
MAX_RESULT_FILE_BYTES = 1_000_000
ALLOWED_REVIEW_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_REVIEW_FILENAMES = {
    "task_case_dump.jsonl",
    "train_events.json",
    "tmux-session.log",
}
ROOT_TEXT_SUFFIXES = {".md", ".toml", ".py", ".txt", ".yaml", ".yml", ".sh"}
ROOT_TEXT_FILENAMES = {".gitignore"}
SOURCE_DIRS = [
    "src",
    "scripts",
    "configs",
    "tests",
]


def _review_tmpdir() -> Path | None:
    candidates: list[Path] = []
    env_value = os.environ.get("MEMTOTAL_REVIEW_TMPDIR", "").strip()
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend([Path("/root/autodl-tmp"), Path("/tmp")])
    for candidate in candidates:
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    return None


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


def _copy_tree(
    src_root: Path,
    dest_root: Path,
    *,
    manifest: list[dict[str, object]],
    include_predicate,
) -> None:
    for src in sorted(path for path in src_root.rglob("*") if include_predicate(path)):
        _copy_file(src, dest_root, manifest=manifest)


def _root_file_allowed(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name == "docs_review_bundle.zip":
        return False
    return path.name in ROOT_TEXT_FILENAMES or path.suffix in ROOT_TEXT_SUFFIXES


def _source_file_allowed(path: Path) -> bool:
    if not path.is_file():
        return False
    if "__pycache__" in path.parts:
        return False
    if path.suffix == ".pyc":
        return False
    return True


def _review_artifact_allowed(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in EXCLUDED_REVIEW_FILENAMES:
        return False
    if "__pycache__" in path.parts:
        return False
    if path.suffix not in ALLOWED_REVIEW_ARTIFACT_SUFFIXES:
        return False
    return path.stat().st_size <= MAX_RESULT_FILE_BYTES


def _copy_root_files(dest_root: Path, *, manifest: list[dict[str, object]]) -> None:
    for src in sorted(path for path in ROOT.iterdir() if _root_file_allowed(path)):
        _copy_file(src, dest_root, manifest=manifest)


def _copy_source_trees(dest_root: Path, *, manifest: list[dict[str, object]]) -> list[str]:
    copied_dirs: list[str] = []
    for dirname in SOURCE_DIRS:
        src_root = ROOT / dirname
        if not src_root.is_dir():
            continue
        _copy_tree(
            src_root,
            dest_root,
            manifest=manifest,
            include_predicate=_source_file_allowed,
        )
        copied_dirs.append(dirname)
    return copied_dirs


def _copy_review_tree(
    review_root: Path,
    dest_root: Path,
    *,
    manifest: list[dict[str, object]],
) -> list[str]:
    copied_dirs: list[str] = []
    if not review_root.is_dir():
        return copied_dirs
    for src_root in sorted(path for path in review_root.iterdir() if path.is_dir()):
        copied_any = False
        for src in sorted(path for path in src_root.rglob("*") if _review_artifact_allowed(path)):
            _copy_file(src, dest_root, manifest=manifest)
            copied_any = True
        if copied_any:
            copied_dirs.append(src_root.name)
    return copied_dirs


def _total_snapshot_bytes(dest_root: Path) -> int:
    return sum(path.stat().st_size for path in dest_root.rglob("*") if path.is_file())


def _zip_snapshot_bytes(dest_root: Path) -> int:
    temp_root = _review_tmpdir()
    with tempfile.TemporaryDirectory(dir=str(temp_root) if temp_root else None) as temp_dir:
        archive_path = Path(temp_dir) / "review-snapshot.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for src in sorted(path for path in dest_root.rglob("*") if path.is_file()):
                archive.write(src, arcname=src.relative_to(dest_root))
        return archive_path.stat().st_size


def _write_manifest(manifest_path: Path, manifest: dict[str, object]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build_snapshot(output_root: Path, *, source_commit: str | None = None) -> dict[str, object]:
    source_commit = source_commit or _git_commit()
    _reset_output_dir(output_root)

    manifest_entries: list[dict[str, object]] = []
    _copy_root_files(output_root, manifest=manifest_entries)
    _copy_tree(
        ROOT / "docs",
        output_root,
        manifest=manifest_entries,
        include_predicate=lambda path: path.is_file(),
    )
    copied_source_dirs = _copy_source_trees(output_root, manifest=manifest_entries)
    copied_review_result_dirs = _copy_review_tree(RESULTS_ROOT, output_root, manifest=manifest_entries)
    copied_review_run_dirs = _copy_review_tree(RUNS_ROOT, output_root, manifest=manifest_entries)

    manifest = {
        "source_commit": source_commit,
        "max_snapshot_zip_bytes": MAX_SNAPSHOT_ZIP_BYTES,
        "max_result_file_bytes": MAX_RESULT_FILE_BYTES,
        "total_size_bytes": 0,
        "zip_size_bytes": 0,
        "source_dirs": copied_source_dirs,
        "review_result_dirs": copied_review_result_dirs,
        "review_run_dirs": copied_review_run_dirs,
        "included_files": manifest_entries,
        "notes": [
            "This snapshot is the lightweight GitHub review export.",
            "It intentionally excludes bulky raw traces such as train_events.json and task_case_dump.jsonl.",
            "It includes the code, configs, scripts, tests, and governed review artifacts needed for external review and lightweight reproduction.",
            "It does not rewrite or shrink the full local working repository history.",
        ],
    }
    manifest_path = output_root / "REVIEW_SNAPSHOT_MANIFEST.json"
    for _ in range(3):
        _write_manifest(manifest_path, manifest)
        total_bytes = _total_snapshot_bytes(output_root)
        zip_size_bytes = _zip_snapshot_bytes(output_root)
        if (
            manifest["total_size_bytes"] == total_bytes
            and manifest["zip_size_bytes"] == zip_size_bytes
        ):
            break
        manifest["total_size_bytes"] = total_bytes
        manifest["zip_size_bytes"] = zip_size_bytes
    else:
        _write_manifest(manifest_path, manifest)
        total_bytes = _total_snapshot_bytes(output_root)
        zip_size_bytes = _zip_snapshot_bytes(output_root)

    if zip_size_bytes > MAX_SNAPSHOT_ZIP_BYTES:
        raise SystemExit(
            f"Snapshot zip exceeds budget: {zip_size_bytes} bytes > {MAX_SNAPSHOT_ZIP_BYTES} bytes"
        )
    return {**manifest, "total_size_bytes": total_bytes, "zip_size_bytes": zip_size_bytes}


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
                    "zip_size_bytes": manifest["zip_size_bytes"],
                    "source_dirs": manifest["source_dirs"],
                    "review_result_dirs": manifest["review_result_dirs"],
                    "review_run_dirs": manifest["review_run_dirs"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
