#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from memtotal.tasks.alfworld_env import ALFWORLD_TEXTWORLD_SPLIT_ROOT, ensure_alfworld_textworld_assets, list_alfworld_game_files
from memtotal.tasks.memoryagentbench import CAPABILITY_SPECS


ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = "/root/autodl-tmp/hf-cache"
LONGMEMEVAL_SOURCE_PATH = (
    "/root/autodl-tmp/.cache/huggingface/hub/"
    "datasets--xiaowu0162--longmemeval-cleaned/"
    "snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json"
)
ALFWORLD_SPLIT_PRIORITY = ("valid_seen", "valid_unseen", "valid_train")
ALFWORLD_FAMILIES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)
LONGMEMEVAL_PRIMARY_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
)
LONGMEMEVAL_AUX_TYPE = "knowledge-update"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"


def _chunk_words(text: str, *, chunk_words: int, stride_words: int) -> list[str]:
    words = [word for word in str(text).replace("\n", " ").split() if word]
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
        start += max(1, stride_words)
    return chunks


def _truncate_words(text: str, *, max_words: int | None) -> tuple[str, int, bool]:
    words = [word for word in str(text).replace("\n", " ").split() if word]
    if max_words is None or max_words <= 0 or len(words) <= max_words:
        return " ".join(words).strip(), len(words), False
    return " ".join(words[:max_words]).strip(), len(words), True


def _flatten_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_texts(item))
        return flattened
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _qa_aliases(answer_value: Any) -> list[str]:
    aliases = _flatten_texts(answer_value)
    return aliases or [""]


def _stable_sample_indexes(total: int, take: int, *, seed: int, tag: str) -> list[int]:
    if take <= 0:
        return []
    if take >= total:
        return list(range(total))
    indexes = list(range(total))
    random.Random(f"{seed}:{tag}").shuffle(indexes)
    return sorted(indexes[:take])


def _spread_sample_indexes(total: int, take: int) -> list[int]:
    if take <= 0:
        return []
    if take >= total:
        return list(range(total))
    if take == 1:
        return [0]
    indexes: list[int] = []
    for position in range(take):
        index = round(position * (total - 1) / (take - 1))
        if indexes and index <= indexes[-1]:
            index = min(total - 1, indexes[-1] + 1)
        indexes.append(index)
    return indexes


def _canonical_eval_example(
    *,
    example_id: str,
    task_name: str,
    display_name: str,
    domain: str,
    segment: str,
    answer: str,
    metric_name: str,
    evaluator_type: str,
    aliases: list[str] | None = None,
    normalizer: str = "text",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": example_id,
        "benchmark_id": _slugify(display_name),
        "task_name": task_name,
        "display_name": display_name,
        "domain": domain,
        "segment": segment,
        "continuation": answer,
        "label": answer,
        "gold_answer": answer,
        "metric_name": metric_name,
        "evaluator_type": evaluator_type,
        "normalizer": normalizer,
    }
    if aliases:
        row["aliases"] = aliases
    if extras:
        row.update(extras)
    return row


def _memoryagentbench_summary_text(
    *,
    context: str,
    keypoints: list[str],
    previous_events: list[str],
    max_words: int = 160,
) -> str:
    summary_parts: list[str] = []
    for event in previous_events[:6]:
        event_text = str(event).strip()
        if event_text:
            summary_parts.append(f"Event: {event_text}")
    for keypoint in keypoints[:8]:
        keypoint_text = str(keypoint).strip()
        if keypoint_text:
            summary_parts.append(f"Keypoint: {keypoint_text}")
    if not summary_parts:
        truncated, _total, _was_truncated = _truncate_words(context, max_words=max_words)
        return truncated
    return _truncate_words(" || ".join(summary_parts), max_words=max_words)[0]


def build_memoryagentbench_pilot_assets(
    *,
    capability_rows: dict[str, list[dict[str, Any]]],
    seed: int,
    examples_per_capability: int = 25,
    short_context_tokens: int = 512,
    rag_chunk_tokens: int = 384,
    rag_chunk_stride: int = 320,
    rag_support_chunks_cap: int = 48,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    datasets = {
        "b0_short_window_eval": [],
        "b1_full_history_eval": [],
        "b2_text_summary_eval": [],
        "b3_rag_eval": [],
        "b3_rag_support": [],
    }
    manifest: dict[str, Any] = {
        "benchmark_id": "memoryagentbench",
        "pilot_examples_total": 0,
        "examples_per_capability": int(examples_per_capability),
        "selected_capabilities": {},
        "context_token_budget_short_window": int(short_context_tokens),
        "rag_support_chunk_words": int(rag_chunk_tokens),
        "rag_support_chunk_stride_words": int(rag_chunk_stride),
        "rag_support_chunks_per_group_cap": int(rag_support_chunks_cap),
        "notes": (
            "V9-1 promotes MemoryAgentBench to a governed pilot by preserving full context for B1, "
            "retaining a short-window control for B0, generating deterministic text summaries for B2, "
            "and chunking per-episode support notes for B3 retrieval with a bounded, evenly spread bank."
        ),
    }
    support_written: set[str] = set()

    for capability in CAPABILITY_SPECS:
        rows = list(capability_rows.get(capability.short_name, []))
        if not rows:
            raise ValueError(f"Missing capability rows for MemoryAgentBench {capability.short_name}.")
        ranked_rows: list[tuple[int, int, str, dict[str, Any]]] = []
        for row_index, row in enumerate(rows):
            metadata = row.get("metadata") or {}
            question_count = min(len(row.get("questions", [])), len(row.get("answers", [])))
            source_name = str(metadata.get("source", f"row-{row_index}"))
            preference = 0 if source_name == capability.source else 1
            ranked_rows.append((preference, -question_count, source_name, row))
        ranked_rows.sort(key=lambda item: (item[0], item[1], item[2]))

        remaining = int(examples_per_capability)
        chosen_rows: list[dict[str, Any]] = []
        examples_for_capability = 0
        for ranked_index, (_preference, _neg_count, source_name, row) in enumerate(ranked_rows):
            if remaining <= 0:
                break
            metadata = row.get("metadata") or {}
            questions = [str(text).strip() for text in row.get("questions", [])]
            answers = row.get("answers", [])
            question_count = min(len(questions), len(answers))
            if question_count <= 0:
                continue
            take = min(remaining, question_count)
            selected_indexes = _stable_sample_indexes(
                question_count,
                take,
                seed=seed,
                tag=f"mab:{capability.short_name}:{source_name}:{ranked_index}",
            )
            context_text = str(row.get("context", "")).strip()
            full_context, full_tokens, _ = _truncate_words(context_text, max_words=None)
            short_context, _short_total, short_truncated = _truncate_words(
                context_text,
                max_words=short_context_tokens,
            )
            summary_text = _memoryagentbench_summary_text(
                context=context_text,
                keypoints=_flatten_texts(metadata.get("keypoints")),
                previous_events=_flatten_texts(metadata.get("previous_events")),
            )
            group_id = f"mab-{capability.short_name.lower()}-{_slugify(source_name)}-{ranked_index:02d}"

            if group_id not in support_written:
                support_written.add(group_id)
                chunks = _chunk_words(full_context, chunk_words=rag_chunk_tokens, stride_words=rag_chunk_stride)
                chunk_indexes = _spread_sample_indexes(len(chunks), min(len(chunks), int(rag_support_chunks_cap)))
                for chunk_index, selected_chunk_index in enumerate(chunk_indexes):
                    chunk = chunks[selected_chunk_index]
                    datasets["b3_rag_support"].append(
                        _canonical_eval_example(
                            example_id=f"{group_id}-support-{chunk_index:03d}",
                            task_name="planv9_v9_1_memoryagentbench_support",
                            display_name="MemoryAgentBench",
                            domain="agent",
                            segment=f"Episode note {chunk_index + 1}: {chunk}",
                            answer="",
                            metric_name="memoryagent_score",
                            evaluator_type="memoryagentbench",
                            extras={
                                "capability": capability.short_name,
                                "capability_name": capability.display_name,
                                "capability_metric_name": capability.primary_metric,
                                "retrieval_group": group_id,
                                "memoryagent_source": source_name,
                                "support_chunk_index": chunk_index,
                                "support_chunk_source_index": selected_chunk_index,
                            },
                        )
                    )

            for question_index in selected_indexes:
                aliases = _qa_aliases(answers[question_index])
                answer_text = aliases[0]
                example_core = {
                    "capability": capability.short_name,
                    "capability_name": capability.display_name,
                    "capability_metric_name": capability.primary_metric,
                    "memoryagent_source": source_name,
                    "question_index": int(question_index),
                    "question_type": _flatten_texts(metadata.get("question_types"))[question_index]
                    if question_index < len(_flatten_texts(metadata.get("question_types")))
                    else "",
                    "qa_pair_id": _flatten_texts(metadata.get("qa_pair_ids"))[question_index]
                    if question_index < len(_flatten_texts(metadata.get("qa_pair_ids")))
                    else "",
                    "keypoints": _flatten_texts(metadata.get("keypoints")),
                    "retrieval_group": group_id,
                    "context_tokens_total": full_tokens,
                }
                question_text = questions[question_index]
                example_slug = f"{group_id}-q{question_index:03d}"
                datasets["b0_short_window_eval"].append(
                    _canonical_eval_example(
                        example_id=f"{example_slug}-b0",
                        task_name="planv9_v9_1_memoryagentbench_b0",
                        display_name="MemoryAgentBench",
                        domain="agent",
                        segment=f"Recent context: {short_context} || Instruction: {question_text}",
                        answer=answer_text,
                        aliases=aliases,
                        metric_name="memoryagent_score",
                        evaluator_type="memoryagentbench",
                        extras={
                            **example_core,
                            "context_variant": "short_window",
                            "context_was_truncated": bool(short_truncated),
                            "context_token_budget": int(short_context_tokens),
                        },
                    )
                )
                datasets["b1_full_history_eval"].append(
                    _canonical_eval_example(
                        example_id=f"{example_slug}-b1",
                        task_name="planv9_v9_1_memoryagentbench_b1",
                        display_name="MemoryAgentBench",
                        domain="agent",
                        segment=f"Full context: {full_context} || Instruction: {question_text}",
                        answer=answer_text,
                        aliases=aliases,
                        metric_name="memoryagent_score",
                        evaluator_type="memoryagentbench",
                        extras={
                            **example_core,
                            "context_variant": "full_history",
                            "context_was_truncated": False,
                            "context_token_budget": None,
                        },
                    )
                )
                datasets["b2_text_summary_eval"].append(
                    _canonical_eval_example(
                        example_id=f"{example_slug}-b2",
                        task_name="planv9_v9_1_memoryagentbench_b2",
                        display_name="MemoryAgentBench",
                        domain="agent",
                        segment=f"Memory summary: {summary_text} || Instruction: {question_text}",
                        answer=answer_text,
                        aliases=aliases,
                        metric_name="memoryagent_score",
                        evaluator_type="memoryagentbench",
                        extras={
                            **example_core,
                            "context_variant": "text_summary",
                            "context_was_truncated": False,
                            "context_token_budget": None,
                        },
                    )
                )
                datasets["b3_rag_eval"].append(
                    _canonical_eval_example(
                        example_id=f"{example_slug}-b3",
                        task_name="planv9_v9_1_memoryagentbench_b3",
                        display_name="MemoryAgentBench",
                        domain="agent",
                        segment=f"Instruction: {question_text}",
                        answer=answer_text,
                        aliases=aliases,
                        metric_name="memoryagent_score",
                        evaluator_type="memoryagentbench",
                        extras={
                            **example_core,
                            "context_variant": "text_rag",
                            "context_was_truncated": False,
                            "context_token_budget": None,
                        },
                    )
                )
                examples_for_capability += 1
            chosen_rows.append(
                {
                    "source": source_name,
                    "questions_available": question_count,
                    "questions_selected": len(selected_indexes),
                    "preferred_source_match": bool(source_name == capability.source),
                }
            )
            remaining -= len(selected_indexes)
        if remaining > 0:
            raise ValueError(
                f"MemoryAgentBench capability {capability.short_name} could not materialize "
                f"{examples_per_capability} questions; missing {remaining}."
            )
        manifest["selected_capabilities"][capability.short_name] = {
            "display_name": capability.display_name,
            "primary_metric": capability.primary_metric,
            "preferred_source": capability.source,
            "rows_used": chosen_rows,
            "examples_materialized": examples_for_capability,
        }
        manifest["pilot_examples_total"] += examples_for_capability
    return datasets, manifest


def _render_longmemeval_turn(turn: dict[str, Any]) -> str:
    role = str(turn.get("role", "unknown")).strip() or "unknown"
    content = str(turn.get("content", "")).strip()
    return f"{role}: {content}"


def _render_longmemeval_session(session_id: str, session_date: str, session_turns: list[dict[str, Any]]) -> str:
    rendered_turns = " || ".join(_render_longmemeval_turn(turn) for turn in session_turns if str(turn.get("content", "")).strip())
    return f"Session {session_id} @ {session_date}: {rendered_turns}".strip()


def _summarize_longmemeval_session(session_id: str, session_date: str, session_turns: list[dict[str, Any]]) -> str:
    flattened = " ".join(
        str(turn.get("content", "")).strip()
        for turn in session_turns[:4]
        if str(turn.get("content", "")).strip()
    )
    compact, _tokens, _truncated = _truncate_words(flattened, max_words=48)
    return f"Session {session_id} @ {session_date}: {compact}"


def build_longmemeval_pilot_assets(
    *,
    rows: list[dict[str, Any]],
    seed: int,
    examples_per_type: int = 20,
    recent_session_budget: int = 3,
    holdout_type: str = LONGMEMEVAL_AUX_TYPE,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    datasets = {
        "b0_short_window_eval": [],
        "b1_full_history_eval": [],
        "b2_text_summary_eval": [],
        "b3_rag_eval": [],
        "b3_rag_support": [],
        "aux_holdout_eval": [],
    }
    manifest: dict[str, Any] = {
        "benchmark_id": "longmemeval",
        "primary_question_types": list(LONGMEMEVAL_PRIMARY_TYPES),
        "aux_holdout_type": holdout_type,
        "examples_per_primary_type": int(examples_per_type),
        "recent_session_budget": int(recent_session_budget),
        "notes": (
            "V9-1 uses the official cleaned LongMemEval source. The primary pilot follows PLANv9's "
            "100-item budget across five governed question types, while `knowledge-update` is preserved "
            "as an auxiliary holdout slice for later regressions."
        ),
    }
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row.get("question_type", "")), []).append(row)

    for question_type in LONGMEMEVAL_PRIMARY_TYPES:
        candidates = sorted(grouped_rows.get(question_type, []), key=lambda row: str(row.get("question_id", "")))
        if len(candidates) < examples_per_type:
            raise ValueError(
                f"LongMemEval question_type={question_type!r} has only {len(candidates)} rows; "
                f"need {examples_per_type}."
            )
        selected_indexes = _stable_sample_indexes(
            len(candidates),
            examples_per_type,
            seed=seed,
            tag=f"longmemeval:{question_type}",
        )
        selected_rows = [candidates[index] for index in selected_indexes]
        manifest[question_type] = {
            "examples_materialized": len(selected_rows),
            "question_ids": [str(row.get("question_id", "")) for row in selected_rows],
        }
        for row in selected_rows:
            question_id = str(row["question_id"])
            answer_text = str(row.get("answer", "")).strip()
            question_text = str(row.get("question", "")).strip()
            question_date = str(row.get("question_date", "")).strip()
            session_ids = [str(value).strip() for value in row.get("haystack_session_ids", [])]
            session_dates = [str(value).strip() for value in row.get("haystack_dates", [])]
            session_payloads = list(row.get("haystack_sessions", []))
            rendered_sessions = [
                _render_longmemeval_session(
                    session_ids[index] if index < len(session_ids) else f"session-{index}",
                    session_dates[index] if index < len(session_dates) else f"date-{index}",
                    session_turns,
                )
                for index, session_turns in enumerate(session_payloads)
            ]
            recent_sessions = rendered_sessions[-recent_session_budget:] if rendered_sessions else []
            summary_lines = [
                _summarize_longmemeval_session(
                    session_ids[index] if index < len(session_ids) else f"session-{index}",
                    session_dates[index] if index < len(session_dates) else f"date-{index}",
                    session_turns,
                )
                for index, session_turns in enumerate(session_payloads)
            ]
            shared_extras = {
                "question_type": question_type,
                "question_date": question_date,
                "answer_session_ids": [str(value) for value in row.get("answer_session_ids", [])],
                "haystack_session_ids": session_ids,
                "retrieval_group": question_id,
            }
            datasets["b0_short_window_eval"].append(
                _canonical_eval_example(
                    example_id=f"{question_id}-b0",
                    task_name="planv9_v9_1_longmemeval_b0",
                    display_name="LongMemEval",
                    domain="agent",
                    segment=(
                        f"Recent sessions: {' || '.join(recent_sessions)} || "
                        f"Question date: {question_date} || Question: {question_text} || Answer briefly."
                    ),
                    answer=answer_text,
                    aliases=[answer_text],
                    metric_name="f1",
                    evaluator_type="qa_f1",
                    extras={**shared_extras, "context_variant": "short_window"},
                )
            )
            datasets["b1_full_history_eval"].append(
                _canonical_eval_example(
                    example_id=f"{question_id}-b1",
                    task_name="planv9_v9_1_longmemeval_b1",
                    display_name="LongMemEval",
                    domain="agent",
                    segment=(
                        f"Conversation history: {' || '.join(rendered_sessions)} || "
                        f"Question date: {question_date} || Question: {question_text} || Answer briefly."
                    ),
                    answer=answer_text,
                    aliases=[answer_text],
                    metric_name="f1",
                    evaluator_type="qa_f1",
                    extras={**shared_extras, "context_variant": "full_history"},
                )
            )
            datasets["b2_text_summary_eval"].append(
                _canonical_eval_example(
                    example_id=f"{question_id}-b2",
                    task_name="planv9_v9_1_longmemeval_b2",
                    display_name="LongMemEval",
                    domain="agent",
                    segment=(
                        f"Session summary: {' || '.join(summary_lines)} || "
                        f"Question date: {question_date} || Question: {question_text} || Answer briefly."
                    ),
                    answer=answer_text,
                    aliases=[answer_text],
                    metric_name="f1",
                    evaluator_type="qa_f1",
                    extras={**shared_extras, "context_variant": "text_summary"},
                )
            )
            datasets["b3_rag_eval"].append(
                _canonical_eval_example(
                    example_id=f"{question_id}-b3",
                    task_name="planv9_v9_1_longmemeval_b3",
                    display_name="LongMemEval",
                    domain="agent",
                    segment=f"Question date: {question_date} || Question: {question_text} || Answer briefly.",
                    answer=answer_text,
                    aliases=[answer_text],
                    metric_name="f1",
                    evaluator_type="qa_f1",
                    extras={**shared_extras, "context_variant": "text_rag"},
                )
            )
            for session_index, rendered_session in enumerate(rendered_sessions):
                datasets["b3_rag_support"].append(
                    _canonical_eval_example(
                        example_id=f"{question_id}-support-{session_index:03d}",
                        task_name="planv9_v9_1_longmemeval_support",
                        display_name="LongMemEval",
                        domain="agent",
                        segment=rendered_session,
                        answer="",
                        metric_name="f1",
                        evaluator_type="qa_f1",
                        extras={
                            "question_type": question_type,
                            "retrieval_group": question_id,
                            "session_index": session_index,
                        },
                    )
                )

    aux_rows = sorted(grouped_rows.get(holdout_type, []), key=lambda row: str(row.get("question_id", "")))
    if len(aux_rows) >= examples_per_type:
        aux_indexes = _stable_sample_indexes(
            len(aux_rows),
            examples_per_type,
            seed=seed,
            tag=f"longmemeval:{holdout_type}",
        )
        for row in [aux_rows[index] for index in aux_indexes]:
            question_id = str(row["question_id"])
            datasets["aux_holdout_eval"].append(
                _canonical_eval_example(
                    example_id=f"{question_id}-aux",
                    task_name="planv9_v9_1_longmemeval_aux_holdout",
                    display_name="LongMemEval",
                    domain="agent",
                    segment=(
                        f"Question date: {str(row.get('question_date', '')).strip()} || "
                        f"Question: {str(row.get('question', '')).strip()} || Answer briefly."
                    ),
                    answer=str(row.get("answer", "")).strip(),
                    aliases=[str(row.get("answer", "")).strip()],
                    metric_name="f1",
                    evaluator_type="qa_f1",
                    extras={
                        "question_type": holdout_type,
                        "context_variant": "aux_holdout",
                    },
                )
            )
        manifest[holdout_type] = {
            "examples_materialized": len(datasets["aux_holdout_eval"]),
            "question_ids": [row["id"] for row in datasets["aux_holdout_eval"]],
        }
    return datasets, manifest


def select_alfworld_pilot_games(
    *,
    games_by_split: dict[str, list[str]],
    episodes_per_family: int,
    split_priority: tuple[str, ...] = ALFWORLD_SPLIT_PRIORITY,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family in ALFWORLD_FAMILIES:
        family_games: list[dict[str, Any]] = []
        for split in split_priority:
            candidates = [
                game_file
                for game_file in sorted(games_by_split.get(split, []))
                if Path(game_file).parent.parent.name.split("-", 1)[0] == family
            ]
            for game_file in candidates:
                family_games.append(
                    {
                        "task_family": family,
                        "split": split,
                        "game_file": str(Path(game_file).resolve()),
                    }
                )
                if len(family_games) >= episodes_per_family:
                    break
            if len(family_games) >= episodes_per_family:
                break
        if len(family_games) < episodes_per_family:
            raise ValueError(
                f"ALFWorld family {family!r} has only {len(family_games)} candidate games across "
                f"{split_priority}; need {episodes_per_family}."
            )
        selected.extend(family_games[:episodes_per_family])
    return selected


def materialize_planv9_v9_1_assets(
    *,
    output_root: Path,
    seed: int,
    memoryagentbench_examples_per_capability: int = 25,
    longmemeval_examples_per_type: int = 20,
    alfworld_episodes_per_family: int = 20,
    alfworld_asset_root: str | Path = ROOT / "data" / "benchmarks" / "external" / "alfworld",
    longmemeval_source_path: str | Path = LONGMEMEVAL_SOURCE_PATH,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    dataset_root = output_root / "materialized-datasets"
    manifest_root = output_root / "materialized-manifests"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    capability_rows: dict[str, list[dict[str, Any]]] = {}
    for capability in CAPABILITY_SPECS:
        dataset = load_dataset("ai-hyz/MemoryAgentBench", split=capability.split_name)
        capability_rows[capability.short_name] = [dataset[index] for index in range(len(dataset))]
    memoryagentbench_datasets, memoryagentbench_manifest = build_memoryagentbench_pilot_assets(
        capability_rows=capability_rows,
        seed=seed,
        examples_per_capability=memoryagentbench_examples_per_capability,
    )
    mab_root = dataset_root / "memoryagentbench"
    for dataset_name, rows in memoryagentbench_datasets.items():
        _write_jsonl(mab_root / f"{dataset_name}.jsonl", rows)
    _write_json(
        manifest_root / "memoryagentbench-pilot.json",
        {
            **memoryagentbench_manifest,
            "dataset_files": {
                name: str((mab_root / f"{name}.jsonl").resolve())
                for name in sorted(memoryagentbench_datasets)
            },
        },
    )

    longmemeval_rows = json.loads(Path(longmemeval_source_path).read_text())
    longmemeval_datasets, longmemeval_manifest = build_longmemeval_pilot_assets(
        rows=longmemeval_rows,
        seed=seed,
        examples_per_type=longmemeval_examples_per_type,
    )
    longmemeval_root = dataset_root / "longmemeval"
    for dataset_name, rows in longmemeval_datasets.items():
        _write_jsonl(longmemeval_root / f"{dataset_name}.jsonl", rows)
    _write_json(
        manifest_root / "longmemeval-pilot.json",
        {
            **longmemeval_manifest,
            "source_path": str(Path(longmemeval_source_path).resolve()),
            "dataset_files": {
                name: str((longmemeval_root / f"{name}.jsonl").resolve())
                for name in sorted(longmemeval_datasets)
            },
        },
    )

    asset_info = ensure_alfworld_textworld_assets(alfworld_asset_root)
    games_by_split = {
        split: [str(path) for path in list_alfworld_game_files(alfworld_asset_root, split=split)]
        for split in ALFWORLD_SPLIT_PRIORITY
    }
    alfworld_games = select_alfworld_pilot_games(
        games_by_split=games_by_split,
        episodes_per_family=alfworld_episodes_per_family,
    )
    alfworld_manifest = {
        "benchmark_id": "alfworld",
        "episodes_per_family": int(alfworld_episodes_per_family),
        "selected_episodes_total": len(alfworld_games),
        "split_priority": list(ALFWORLD_SPLIT_PRIORITY),
        "task_families": list(ALFWORLD_FAMILIES),
        "selected_episodes": alfworld_games,
        "asset_info": asset_info,
        "notes": (
            "V9-1 promotes ALFWorld to a governed 120-episode TextWorld pilot. "
            "The pilot combines valid splits to satisfy the 20x6 family quota, "
            "then evaluates success rate and step budget after one expert bootstrap action."
        ),
    }
    _write_json(manifest_root / "alfworld-pilot.json", alfworld_manifest)

    summary = {
        "phase": "V9-1",
        "seed": int(seed),
        "materialized_root": str(output_root),
        "manifests": {
            "memoryagentbench": str((manifest_root / "memoryagentbench-pilot.json").resolve()),
            "longmemeval": str((manifest_root / "longmemeval-pilot.json").resolve()),
            "alfworld": str((manifest_root / "alfworld-pilot.json").resolve()),
        },
    }
    _write_json(manifest_root / "v9-1-materialization-summary.json", summary)
    return summary


def build_planv9_v9_1_static_baseline_config(
    *,
    benchmark_name: str,
    baseline_id: str,
    dataset_path: str,
    output_config: Path,
    primary_model_dir: str,
    eval_examples: int,
    evaluator_type: str,
    metric_name: str,
    support_dataset_path: str | None = None,
    primary_backbone_name: str = "Qwen3-4B",
    hf_cache_dir: str = HF_CACHE_DIR,
) -> dict[str, Any]:
    if baseline_id not in {"b0_short_window", "b1_full_history", "b2_text_summary", "b3_text_rag"}:
        raise ValueError(f"Unsupported V9-1 baseline_id: {baseline_id}")
    baseline_family = "rag" if baseline_id == "b3_text_rag" else "prompting"
    config = {
        "experiment": {
            "name": f"planv9_v9_1_{benchmark_name}_{baseline_id}",
            "stage": "V9-1",
            "method_variant": baseline_id,
        },
        "backbone": {
            "name": str(primary_backbone_name),
            "load_mode": "hf_causal_lm",
            "model_id": str(primary_model_dir),
            "dtype": "bfloat16",
            "cache_dir": str(hf_cache_dir),
            "attn_implementation": "sdpa",
            "gradient_checkpointing": False,
            "use_chat_template": True,
            "chat_template_enable_thinking": False,
            "max_new_tokens": 96 if benchmark_name == "alfworld" else 160,
        },
        "task": {
            "name": f"planv9_v9_1_{benchmark_name}",
            "dataset_path": str(Path(dataset_path).resolve()),
            "support_dataset_path": str(Path(support_dataset_path).resolve()) if support_dataset_path else "",
            "metric_name": str(metric_name),
            "evaluator": {
                "type": str(evaluator_type),
                "normalizer": "action" if benchmark_name == "alfworld" else "text",
            },
        },
        "runtime": {
            "device": "cuda",
            "eval_examples": int(eval_examples),
        },
        "baseline": {
            "family": baseline_family,
            "mode": "retrieval_augmented" if baseline_family == "rag" else "vanilla",
            "support_examples": 4 if baseline_family == "rag" else 0,
            "rag": {
                "retriever": "lexical_overlap",
                "include_answer_in_memory": False,
                "memory_prefix": "Retrieved note",
            },
        },
    }
    _write_json(output_config, config)
    return config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PLANv9 V9-1 pilot materialization and config builder.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--output_root", required=True)
    materialize.add_argument("--seed", type=int, required=True)
    materialize.add_argument("--memoryagentbench_examples_per_capability", type=int, default=25)
    materialize.add_argument("--longmemeval_examples_per_type", type=int, default=20)
    materialize.add_argument("--alfworld_episodes_per_family", type=int, default=20)
    materialize.add_argument("--alfworld_asset_root", default=str(ROOT / "data" / "benchmarks" / "external" / "alfworld"))
    materialize.add_argument("--longmemeval_source_path", default=LONGMEMEVAL_SOURCE_PATH)

    static_config = subparsers.add_parser("static-config")
    static_config.add_argument("--benchmark_name", required=True)
    static_config.add_argument("--baseline_id", required=True)
    static_config.add_argument("--dataset_path", required=True)
    static_config.add_argument("--output_config", required=True)
    static_config.add_argument("--primary_model_dir", required=True)
    static_config.add_argument("--eval_examples", type=int, required=True)
    static_config.add_argument("--evaluator_type", required=True)
    static_config.add_argument("--metric_name", required=True)
    static_config.add_argument("--support_dataset_path", default="")
    static_config.add_argument("--primary_backbone_name", default="Qwen3-4B")
    static_config.add_argument("--hf_cache_dir", default=HF_CACHE_DIR)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.command == "materialize":
        materialize_planv9_v9_1_assets(
            output_root=Path(args.output_root),
            seed=args.seed,
            memoryagentbench_examples_per_capability=args.memoryagentbench_examples_per_capability,
            longmemeval_examples_per_type=args.longmemeval_examples_per_type,
            alfworld_episodes_per_family=args.alfworld_episodes_per_family,
            alfworld_asset_root=args.alfworld_asset_root,
            longmemeval_source_path=args.longmemeval_source_path,
        )
        return 0
    build_planv9_v9_1_static_baseline_config(
        benchmark_name=args.benchmark_name,
        baseline_id=args.baseline_id,
        dataset_path=args.dataset_path,
        output_config=Path(args.output_config),
        primary_model_dir=args.primary_model_dir,
        eval_examples=args.eval_examples,
        evaluator_type=args.evaluator_type,
        metric_name=args.metric_name,
        support_dataset_path=args.support_dataset_path or None,
        primary_backbone_name=args.primary_backbone_name,
        hf_cache_dir=args.hf_cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
