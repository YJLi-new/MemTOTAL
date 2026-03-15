from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.planv9_v9_1_config import (
    CAPABILITY_SPECS,
    LONGMEMEVAL_AUX_TYPE,
    LONGMEMEVAL_PRIMARY_TYPES,
    build_longmemeval_pilot_assets,
    build_memoryagentbench_pilot_assets,
    build_planv9_v9_1_static_baseline_config,
    select_alfworld_pilot_games,
)


def _make_memoryagentbench_row(*, source: str, questions: int, keypoint: str) -> dict:
    return {
        "context": " ".join(f"{source}-token-{index}" for index in range(800)),
        "questions": [f"{source} question {index}?" for index in range(questions)],
        "answers": [[f"{source} answer {index}"] for index in range(questions)],
        "metadata": {
            "source": source,
            "keypoints": [f"{keypoint} {index}" for index in range(4)],
            "previous_events": [f"event {index}" for index in range(4)],
            "qa_pair_ids": [f"{source}-pair-{index}" for index in range(questions)],
            "question_types": [f"type-{index % 3}" for index in range(questions)],
        },
    }


def _make_longmemeval_row(*, question_type: str, index: int) -> dict:
    return {
        "question_id": f"{question_type}-{index:03d}",
        "question_type": question_type,
        "question": f"Question for {question_type} #{index}?",
        "question_date": "2026/03/15 (Sun) 12:00",
        "answer": f"Answer {index}",
        "answer_session_ids": [f"answer-{index:03d}"],
        "haystack_session_ids": [f"session-{index:03d}-0", f"session-{index:03d}-1"],
        "haystack_dates": ["2026/03/14 (Sat) 10:00", "2026/03/15 (Sun) 09:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": f"user memory {index} 0"},
                {"role": "assistant", "content": f"assistant memory {index} 0"},
            ],
            [
                {"role": "user", "content": f"user memory {index} 1"},
                {"role": "assistant", "content": f"assistant memory {index} 1"},
            ],
        ],
    }


class PlanV9V91ConfigTest(unittest.TestCase):
    def test_memoryagentbench_builder_aggregates_rows_when_capability_row_is_too_small(self) -> None:
        capability_rows = {
            capability.short_name: [_make_memoryagentbench_row(source=capability.source, questions=30, keypoint=capability.short_name)]
            for capability in CAPABILITY_SPECS
        }
        capability_rows["LRU"] = [
            _make_memoryagentbench_row(source="small-lru-a", questions=15, keypoint="LRU"),
            _make_memoryagentbench_row(source="small-lru-b", questions=15, keypoint="LRU"),
        ]
        datasets, manifest = build_memoryagentbench_pilot_assets(
            capability_rows=capability_rows,
            seed=7,
            examples_per_capability=25,
        )

        self.assertEqual(len(datasets["b0_short_window_eval"]), 100)
        self.assertEqual(len(datasets["b1_full_history_eval"]), 100)
        self.assertEqual(len(datasets["b2_text_summary_eval"]), 100)
        self.assertEqual(len(datasets["b3_rag_eval"]), 100)
        self.assertGreater(len(datasets["b3_rag_support"]), 0)
        self.assertEqual(manifest["selected_capabilities"]["LRU"]["examples_materialized"], 25)
        self.assertEqual(len(manifest["selected_capabilities"]["LRU"]["rows_used"]), 2)
        self.assertEqual(manifest["rag_support_chunks_per_group_cap"], 48)

    def test_memoryagentbench_builder_caps_support_rows_per_retrieval_group(self) -> None:
        capability_rows = {
            capability.short_name: [_make_memoryagentbench_row(source=capability.source, questions=30, keypoint=capability.short_name)]
            for capability in CAPABILITY_SPECS
        }
        datasets, manifest = build_memoryagentbench_pilot_assets(
            capability_rows=capability_rows,
            seed=11,
            examples_per_capability=25,
            rag_chunk_tokens=32,
            rag_chunk_stride=16,
            rag_support_chunks_cap=7,
        )

        self.assertEqual(manifest["rag_support_chunks_per_group_cap"], 7)
        support_counts: dict[str, int] = {}
        for row in datasets["b3_rag_support"]:
            support_counts[row["retrieval_group"]] = support_counts.get(row["retrieval_group"], 0) + 1
        self.assertTrue(support_counts)
        self.assertTrue(all(count <= 7 for count in support_counts.values()))

    def test_longmemeval_builder_emits_primary_and_auxiliary_slices(self) -> None:
        rows = []
        for question_type in LONGMEMEVAL_PRIMARY_TYPES:
            rows.extend(_make_longmemeval_row(question_type=question_type, index=index) for index in range(20))
        rows.extend(_make_longmemeval_row(question_type=LONGMEMEVAL_AUX_TYPE, index=index) for index in range(20))

        datasets, manifest = build_longmemeval_pilot_assets(
            rows=rows,
            seed=11,
            examples_per_type=20,
        )

        self.assertEqual(len(datasets["b0_short_window_eval"]), 100)
        self.assertEqual(len(datasets["b1_full_history_eval"]), 100)
        self.assertEqual(len(datasets["b2_text_summary_eval"]), 100)
        self.assertEqual(len(datasets["b3_rag_eval"]), 100)
        self.assertEqual(len(datasets["aux_holdout_eval"]), 20)
        self.assertEqual(manifest[LONGMEMEVAL_AUX_TYPE]["examples_materialized"], 20)

    def test_alfworld_selection_uses_split_priority_to_fill_family_quota(self) -> None:
        games_by_split = {
            "valid_seen": [
                f"/tmp/valid_seen/look_at_obj_in_light-A-{index}/trial_{index}/game.tw-pddl"
                for index in range(1)
            ]
            + [
                f"/tmp/valid_seen/{family}-A-{index}/trial_{index}/game.tw-pddl"
                for family in (
                    "pick_and_place_simple",
                    "pick_clean_then_place_in_recep",
                    "pick_cool_then_place_in_recep",
                    "pick_heat_then_place_in_recep",
                    "pick_two_obj_and_place",
                )
                for index in range(2)
            ],
            "valid_unseen": [
                f"/tmp/valid_unseen/look_at_obj_in_light-B-{index}/trial_{index}/game.tw-pddl"
                for index in range(3)
            ],
        }
        selected = select_alfworld_pilot_games(
            games_by_split=games_by_split,
            episodes_per_family=2,
        )

        self.assertEqual(len(selected), 12)
        self.assertEqual(
            len([row for row in selected if row["task_family"] == "look_at_obj_in_light"]),
            2,
        )
        self.assertIn("valid_unseen", {row["split"] for row in selected if row["task_family"] == "look_at_obj_in_light"})

    def test_static_config_builder_sets_rag_support_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_config = Path(tmpdir) / "config.json"
            config = build_planv9_v9_1_static_baseline_config(
                benchmark_name="longmemeval",
                baseline_id="b3_text_rag",
                dataset_path="/tmp/eval.jsonl",
                output_config=output_config,
                primary_model_dir="/tmp/qwen34",
                eval_examples=100,
                evaluator_type="qa_f1",
                metric_name="f1",
                support_dataset_path="/tmp/support.jsonl",
            )

            self.assertEqual(config["baseline"]["family"], "rag")
            self.assertEqual(config["baseline"]["support_examples"], 4)
            self.assertEqual(config["task"]["support_dataset_path"], str(Path("/tmp/support.jsonl").resolve()))
            self.assertTrue(output_config.exists())


if __name__ == "__main__":
    unittest.main()
