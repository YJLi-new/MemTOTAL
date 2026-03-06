from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from datasets import load_dataset

from memtotal.tasks.alfworld_env import materialize_alfworld_textworld_smoke
from memtotal.tasks.memoryagentbench import materialize_memoryagentbench_smoke


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
    dataset_configs: list[str] | None = None


SOURCE_SPECS: dict[str, BenchmarkSourceSpec] = {
    "gsm8k": BenchmarkSourceSpec(
        benchmark_id="gsm8k",
        display_name="GSM8K",
        access="public",
        source_kind="huggingface",
        dataset_name="gsm8k",
        dataset_config="main",
        dataset_configs=None,
        split="test",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/gsm8k",
        homepage=None,
        license_note="Hugging Face metadata currently does not expose a structured license field; verify upstream dataset card before redistribution.",
    ),
    "math": BenchmarkSourceSpec(
        benchmark_id="math",
        display_name="MATH",
        access="public",
        source_kind="multi_huggingface_configs",
        dataset_name="EleutherAI/hendrycks_math",
        dataset_config=None,
        dataset_configs=["algebra", "geometry", "number_theory", "precalculus"],
        split="test",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/EleutherAI/hendrycks_math",
        homepage="https://github.com/hendrycks/math",
        license_note="MIT (from dataset card README metadata).",
        notes="Current real-smoke aggregate samples one example from each of four configs: algebra, geometry, number_theory, precalculus.",
    ),
    "gpqa": BenchmarkSourceSpec(
        benchmark_id="gpqa",
        display_name="GPQA",
        access="gated",
        source_kind="huggingface",
        dataset_name="Idavidrein/gpqa",
        dataset_config="gpqa_diamond",
        dataset_configs=None,
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
        dataset_configs=None,
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
        dataset_configs=None,
        split="test",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/gimmaru/story_cloze-2016",
        homepage=None,
        license_note="Hugging Face metadata currently does not expose a structured license field; verify upstream dataset card before redistribution.",
    ),
    "narrativeqa": BenchmarkSourceSpec(
        benchmark_id="narrativeqa",
        display_name="NarrativeQA",
        access="public",
        source_kind="huggingface_streaming",
        dataset_name="deepmind/narrativeqa",
        dataset_config=None,
        dataset_configs=None,
        split="validation",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/deepmind/narrativeqa",
        homepage="https://github.com/deepmind/narrativeqa",
        license_note="Apache-2.0 (from the official Hugging Face dataset card metadata).",
        notes=(
            "Current smoke materializes the official summary-only view from the validation split. "
            "This keeps the real-source narrative path lightweight while staying aligned with the "
            "dataset's documented summary-vs-story task variants."
        ),
    ),
    "kodcode": BenchmarkSourceSpec(
        benchmark_id="kodcode",
        display_name="KodCode",
        access="public",
        source_kind="huggingface",
        dataset_name="KodCode/KodCode-Light-RL-10K",
        dataset_config=None,
        dataset_configs=None,
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
        dataset_configs=None,
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
        access="public",
        source_kind="huggingface",
        dataset_name="Dzeniks/fever_3way",
        dataset_config=None,
        dataset_configs=None,
        split="validation",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/Dzeniks/fever_3way",
        homepage=None,
        license_note="MIT (from dataset card README metadata).",
        notes="Uses the public 3-way FEVER variant with labels mapped to SUPPORTS / REFUTES / NOT_ENOUGH_INFO.",
    ),
    "alfworld": BenchmarkSourceSpec(
        benchmark_id="alfworld",
        display_name="ALFWorld",
        access="public",
        source_kind="alfworld_textworld",
        dataset_name="alfworld-textworld-release",
        dataset_config=None,
        dataset_configs=None,
        split="valid_seen",
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://github.com/alfworld/alfworld/releases",
        homepage="https://alfworld.github.io/",
        license_note="MIT (from the official ALFWorld GitHub repository).",
        notes="Uses the official TextWorld release assets. The smoke subset executes the first hand-coded expert action, then predicts the next expert action.",
    ),
    "memoryagentbench": BenchmarkSourceSpec(
        benchmark_id="memoryagentbench",
        display_name="MemoryAgentBench",
        access="public",
        source_kind="memoryagentbench_huggingface",
        dataset_name="ai-hyz/MemoryAgentBench",
        dataset_config=None,
        dataset_configs=None,
        split=None,
        data_files=None,
        output_filename="eval-real-smoke4.jsonl",
        source_url="https://huggingface.co/datasets/ai-hyz/MemoryAgentBench",
        homepage="https://github.com/HUST-AI-HYZ/MemoryAgentBench",
        license_note="MIT (from the official Hugging Face dataset card metadata).",
        notes=(
            "The smoke scaffold materializes one official representative row for each of the "
            "four MemoryAgentBench capabilities (AR/TTL/LRU/CR) and expands them into a small "
            "query subset. For local stub execution, context is truncated to a fixed token budget."
        ),
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


def _load_rows_streaming(spec: BenchmarkSourceSpec, max_examples: int) -> list[dict[str, Any]]:
    dataset = load_dataset(spec.dataset_name, spec.dataset_config, split=spec.split, streaming=True)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if max_examples > 0 and index >= max_examples:
            break
        rows.append(_to_serializable(row))
    return rows


def _load_multi_config_rows(spec: BenchmarkSourceSpec, max_examples: int) -> list[dict[str, Any]]:
    if not spec.dataset_configs:
        raise ValueError(f"{spec.benchmark_id} requires dataset_configs for multi-config loading.")
    per_config = max(1, max_examples // len(spec.dataset_configs))
    remainder = max_examples % len(spec.dataset_configs)
    rows: list[dict[str, Any]] = []
    for config_index, config_name in enumerate(spec.dataset_configs):
        take = per_config + (1 if config_index < remainder else 0)
        split_expr = spec.split if take <= 0 else f"{spec.split}[:{take}]"
        dataset = load_dataset(spec.dataset_name, config_name, split=split_expr)
        for index in range(len(dataset)):
            row = _to_serializable(dataset[index])
            row["_source_config"] = config_name
            rows.append(row)
    return rows


def _canonicalize_gsm8k(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    answer = str(row["answer"]).split("\n####")[-1].strip()
    return {
        "id": str(row.get("id", f"gsm8k-{index}")),
        "question": str(row["question"]).strip(),
        "answer": answer,
    }


def _extract_math_final_answer(solution: str) -> str:
    matches = list(re.finditer(r"\\boxed\{", solution))
    if matches:
        start = matches[-1].end()
        depth = 1
        collected: list[str] = []
        for char in solution[start:]:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            collected.append(char)
        boxed = "".join(collected).strip()
        if boxed:
            return boxed
    tail_match = re.search(r"(?:answer is|therefore[, ]+the answer is)\s*([^\n.]+)", solution, flags=re.IGNORECASE)
    if tail_match:
        return tail_match.group(1).strip().rstrip(".")
    return solution.strip().splitlines()[-1].strip()


def _canonicalize_math(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    solution = str(row["solution"]).strip()
    return {
        "id": f"math-{row.get('_source_config', 'unknown')}-{index}",
        "question": str(row["problem"]).strip(),
        "answer": _extract_math_final_answer(solution),
        "solution": solution,
        "math_subject": str(row.get("_source_config", row.get("type", "unknown"))),
        "math_level": str(row.get("level", "")),
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


def _canonicalize_narrativeqa(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    document = row.get("document", {})
    summary = document.get("summary", {})
    answers = [
        str(answer.get("text", "")).strip()
        for answer in row.get("answers", [])
        if str(answer.get("text", "")).strip()
    ]
    if not answers:
        raise ValueError("NarrativeQA row is missing answer texts.")
    story = str(summary.get("text", "")).strip()
    if not story:
        raise ValueError("NarrativeQA row is missing summary text for summary-only smoke.")
    return {
        "id": f"{document.get('id', 'narrativeqa')}-q{index}",
        "story": story,
        "question": str(row.get("question", {}).get("text", "")).strip(),
        "answer": answers[0],
        "aliases": answers,
        "document_kind": str(document.get("kind", "")),
        "summary_title": str(summary.get("title", "")).strip(),
        "story_chars": len(story),
        "story_word_count": int(document.get("word_count", 0) or 0),
        "narrativeqa_view": "summary_only",
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


def _canonicalize_fever(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    del seed
    label_map = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT_ENOUGH_INFO",
    }
    label = label_map[int(row["label"])]
    choices = [
        {"label": "SUPPORTS", "text": "Supports"},
        {"label": "REFUTES", "text": "Refutes"},
        {"label": "NOT_ENOUGH_INFO", "text": "Not enough info"},
    ]
    evidence = str(row.get("evidence", "")).strip() or "No gold evidence provided in this split."
    return {
        "id": str(row.get("id", f"fever-{index}")),
        "claim": str(row["claim"]).strip(),
        "evidence": evidence,
        "choices": choices,
        "label": label,
        "answer": label.replace("_", " ").title(),
    }


CANONICALIZERS = {
    "gsm8k": _canonicalize_gsm8k,
    "math": _canonicalize_math,
    "gpqa": _canonicalize_gpqa,
    "triviaqa": _canonicalize_triviaqa,
    "story_cloze": _canonicalize_story_cloze,
    "narrativeqa": _canonicalize_narrativeqa,
    "kodcode": _canonicalize_kodcode,
    "rocstories": _canonicalize_rocstories,
    "fever": _canonicalize_fever,
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

    if spec.source_kind == "multi_huggingface_configs":
        rows = _load_multi_config_rows(spec, max_examples)
    elif spec.source_kind == "huggingface_streaming":
        rows = _load_rows_streaming(spec, max_examples)
    elif spec.source_kind == "alfworld_textworld":
        asset_root = Path(output_root).resolve().parent / "external" / "alfworld"
        canonical_rows, extra_manifest = materialize_alfworld_textworld_smoke(
            asset_root=asset_root,
            max_examples=max_examples,
            split=str(spec.split),
        )
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
            **extra_manifest,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    elif spec.source_kind == "memoryagentbench_huggingface":
        canonical_rows, extra_manifest = materialize_memoryagentbench_smoke(
            max_examples=max_examples,
            seed=seed,
        )
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
            **extra_manifest,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    else:
        rows = _load_rows(spec)
    if max_examples > 0 and spec.source_kind != "multi_huggingface_configs":
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
