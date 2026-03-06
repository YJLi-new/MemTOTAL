from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class BenchmarkSourceSpec:
    benchmark_id: str
    display_name: str
    access: str
    source_kind: str
    dataset_name: str | None
    dataset_config: str | None
    split: str | None
    data_files: dict[str, str] | None
    output_filename: str
    source_url: str | None
    homepage: str | None
    license_note: str
    notes: str = ""


SOURCE_SPECS: dict[str, BenchmarkSourceSpec] = {
    "gsm8k": BenchmarkSourceSpec(
        benchmark_id="gsm8k",
        display_name="GSM8K",
        access="public",
        source_kind="huggingface",
        dataset_name="gsm8k",
        dataset_config="main",
        split="test",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/gsm8k",
        homepage=None,
        license_note="Hugging Face metadata currently does not expose a structured license field; verify upstream dataset card before redistribution.",
    ),
    "gpqa": BenchmarkSourceSpec(
        benchmark_id="gpqa",
        display_name="GPQA",
        access="gated",
        source_kind="huggingface",
        dataset_name="Idavidrein/gpqa",
        dataset_config="gpqa_diamond",
        split="train",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/Idavidrein/gpqa",
        homepage=None,
        license_note="Gated Hugging Face dataset; current metadata does not expose a structured license field. Access approval is required before use.",
        notes="MemGen uses gpqa_main for train and gpqa_diamond for eval; this scaffold materializes gpqa_diamond for eval-side smoke only.",
    ),
    "triviaqa": BenchmarkSourceSpec(
        benchmark_id="triviaqa",
        display_name="TriviaQA",
        access="public",
        source_kind="huggingface",
        dataset_name="mandarjoshi/trivia_qa",
        dataset_config="rc.wikipedia.nocontext",
        split="validation",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/mandarjoshi/trivia_qa",
        homepage=None,
        license_note="Hugging Face metadata currently does not expose a structured license field; verify upstream dataset card before redistribution.",
        notes="Validation split is used for local smoke because it is labeled and lightweight; official test split remains available for later protocol alignment.",
    ),
    "story_cloze": BenchmarkSourceSpec(
        benchmark_id="story_cloze",
        display_name="Story Cloze",
        access="public",
        source_kind="huggingface",
        dataset_name="gimmaru/story_cloze-2016",
        dataset_config=None,
        split="test",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/gimmaru/story_cloze-2016",
        homepage=None,
        license_note="Hugging Face metadata currently does not expose a structured license field; verify upstream dataset card before redistribution.",
    ),
    "kodcode": BenchmarkSourceSpec(
        benchmark_id="kodcode",
        display_name="KodCode",
        access="public",
        source_kind="huggingface",
        dataset_name="KodCode/KodCode-Light-RL-10K",
        dataset_config=None,
        split="train",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K",
        homepage=None,
        license_note="Hugging Face metadata currently does not expose a structured license field; verify upstream dataset card before redistribution.",
        notes="Current scaffold materializes eval-side smoke from the public train split because the upstream light set is single-split.",
    ),
    "rocstories": BenchmarkSourceSpec(
        benchmark_id="rocstories",
        display_name="ROCStories",
        access="public",
        source_kind="hf_csv",
        dataset_name="csv",
        dataset_config=None,
        split="test",
        data_files={"test": "hf://datasets/wza/roc_stories/ROCStories__spring2016.csv"},
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/wza/roc_stories",
        homepage=None,
        license_note="Hugging Face metadata for this CSV-backed dataset is not exposed through `load_dataset_builder`; verify the upstream dataset card before redistribution.",
        notes="The current `datasets` version no longer supports the legacy `roc_stories.py` script, so this scaffold follows MemGen and reads the CSV via `hf://` instead.",
    ),
    "fever": BenchmarkSourceSpec(
        benchmark_id="fever",
        display_name="FEVER",
        access="manual",
        source_kind="manual",
        dataset_name=None,
        dataset_config=None,
        split=None,
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url=None,
        homepage=None,
        license_note="Pending manual source registration.",
        notes="Current repo only has a local contract smoke subset. Real FEVER download/cache path is still pending.",
    ),
    "alfworld": BenchmarkSourceSpec(
        benchmark_id="alfworld",
        display_name="ALFWorld",
        access="manual",
        source_kind="manual",
        dataset_name=None,
        dataset_config=None,
        split=None,
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url=None,
        homepage="https://alfworld.github.io/",
        license_note="Pending environment/game-files registration.",
        notes="ALFWorld requires environment assets and execution harness, not just a flat JSONL download.",
    ),
}


def list_benchmark_sources() -> list[BenchmarkSourceSpec]:
    return [SOURCE_SPECS[key] for key in sorted(SOURCE_SPECS)]


def get_benchmark_source(benchmark_id: str) -> BenchmarkSourceSpec:
    try:
        return SOURCE_SPECS[benchmark_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark source: {benchmark_id}") from exc


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_serializable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _load_rows(spec: BenchmarkSourceSpec) -> list[dict[str, Any]]:
    if spec.source_kind == "manual":
        raise RuntimeError(
            f"{spec.benchmark_id} is marked as manual and cannot be materialized automatically yet."
        )
    if spec.source_kind == "huggingface":
        dataset = load_dataset(spec.dataset_name, spec.dataset_config, split=spec.split)
        return [_to_serializable(dataset[index]) for index in range(len(dataset))]
    if spec.source_kind == "hf_csv":
        dataset = load_dataset("csv", data_files=spec.data_files, split=spec.split)
        return [_to_serializable(dataset[index]) for index in range(len(dataset))]
    raise ValueError(f"Unsupported source kind: {spec.source_kind}")


def _canonicalize_gsm8k(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    answer = str(row["answer"]).split("\n####")[-1].strip()
    return {
        "id": str(row.get("id", f"gsm8k-{index}")),
        "question": str(row["question"]).strip(),
        "answer": answer,
    }


def _canonicalize_gpqa(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    candidates = [
        str(row["Correct Answer"]).strip(),
        str(row["Incorrect Answer 1"]).strip(),
        str(row["Incorrect Answer 2"]).strip(),
        str(row["Incorrect Answer 3"]).strip(),
    ]
    permutation = list(range(len(candidates)))
    random.Random(f"{seed}:{index}:{row['Question']}").shuffle(permutation)
    choices = []
    gold_label = None
    for output_index, candidate_index in enumerate(permutation):
        label = chr(ord("A") + output_index)
        text = candidates[candidate_index]
        choices.append({"label": label, "text": text})
        if candidate_index == 0:
            gold_label = label
    if gold_label is None:
        raise ValueError("GPQA canonicalization failed to identify the gold option.")
    return {
        "id": str(row.get("Record ID", f"gpqa-{index}")),
        "question": str(row["Question"]).strip(),
        "choices": choices,
        "label": gold_label,
        "answer": candidates[0],
    }


def _canonicalize_triviaqa(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    answer_block = row.get("answer", {})
    aliases = [str(alias).strip() for alias in answer_block.get("normalized_aliases", []) if str(alias).strip()]
    if not aliases:
        raise ValueError("TriviaQA row is missing normalized aliases.")
    return {
        "id": str(row.get("question_id", f"triviaqa-{index}")),
        "question": str(row["question"]).strip(),
        "answer": aliases[0],
        "aliases": aliases,
    }


def _canonicalize_story_cloze(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    story = " ".join(str(row[f"input_sentence_{offset}"]).strip() for offset in range(1, 5))
    choices = [
        {"label": "A", "text": str(row["sentence_quiz1"]).strip()},
        {"label": "B", "text": str(row["sentence_quiz2"]).strip()},
    ]
    label = "A" if int(row["answer_right_ending"]) == 1 else "B"
    answer = choices[0]["text"] if label == "A" else choices[1]["text"]
    return {
        "id": str(row.get("story_id", f"story-cloze-{index}")),
        "story": story,
        "choices": choices,
        "label": label,
        "answer": answer,
    }


def _canonicalize_kodcode(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    return {
        "id": str(row.get("id", f"kodcode-{index}")),
        "prompt": str(row["question"]).strip(),
        "answer": str(row["solution"]).rstrip(),
    }


def _canonicalize_rocstories(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    story = " ".join(str(row[f"sentence{i}"]).strip() for i in range(1, 5))
    return {
        "id": str(row.get("storyid", f"rocstories-{index}")),
        "story": story,
        "answer": str(row["sentence5"]).strip(),
    }


CANONICALIZERS = {
    "gsm8k": _canonicalize_gsm8k,
    "gpqa": _canonicalize_gpqa,
    "triviaqa": _canonicalize_triviaqa,
    "story_cloze": _canonicalize_story_cloze,
    "kodcode": _canonicalize_kodcode,
    "rocstories": _canonicalize_rocstories,
}


def materialize_benchmark_source(
    *,
    benchmark_id: str,
    output_root: str | Path,
    manifest_root: str | Path,
    max_examples: int,
    seed: int,
) -> dict[str, Any]:
    spec = get_benchmark_source(benchmark_id)
    output_dir = Path(output_root).resolve() / benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(manifest_root).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{benchmark_id}.json"

    if spec.source_kind == "manual":
        manifest = {
            **asdict(spec),
            "status": "manual_pending",
            "materialized_path": None,
            "num_rows": 0,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest

    rows = _load_rows(spec)
    if max_examples > 0:
        rows = rows[:max_examples]
    canonicalizer = CANONICALIZERS.get(benchmark_id)
    if canonicalizer is None:
        raise ValueError(f"No canonicalizer registered for benchmark_id={benchmark_id}.")
    canonical_rows = [canonicalizer(row, index, seed) for index, row in enumerate(rows)]

    output_path = output_dir / spec.output_filename
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in canonical_rows) + ("\n" if canonical_rows else "")
    )
    manifest = {
        **asdict(spec),
        "status": "materialized",
        "materialized_path": str(output_path),
        "num_rows": len(canonical_rows),
        "max_examples": max_examples,
        "seed": seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
