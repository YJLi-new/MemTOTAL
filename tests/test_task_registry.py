from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.tasks import build_task_evaluator, list_task_specs, load_task_dataset
from memtotal.utils.config import load_config


class TaskRegistryTest(unittest.TestCase):
    def test_registry_covers_m4_smoke_entry_tasks(self) -> None:
        benchmark_ids = {spec.benchmark_id for spec in list_task_specs()}
        self.assertTrue(
            {
                "gsm8k",
                "math",
                "gpqa",
                "triviaqa",
                "kodcode",
                "story_cloze",
                "narrativeqa",
                "rocstories",
                "fever",
                "alfworld",
                "memoryagentbench",
            }.issubset(benchmark_ids)
        )

    def test_load_task_dataset_canonicalizes_exact_match_rows(self) -> None:
        config = load_config(ROOT / "configs/exp/benchmark_gsm8k_qwen25_smoke.yaml")
        dataset = load_task_dataset(config)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["benchmark_id"], "gsm8k")
        self.assertEqual(dataset[0]["evaluator_type"], "exact_match")
        self.assertIn("Question:", dataset[0]["segment"])
        self.assertEqual(dataset[0]["gold_answer"], "12")

    def test_load_task_dataset_canonicalizes_multiple_choice_rows(self) -> None:
        config = load_config(ROOT / "configs/exp/benchmark_story_cloze_qwen25_smoke.yaml")
        dataset = load_task_dataset(config)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["benchmark_id"], "story_cloze")
        self.assertEqual(dataset[0]["evaluator_type"], "multiple_choice")
        self.assertEqual(dataset[0]["label"], "A")
        self.assertEqual(len(dataset[0]["choices"]), 2)

    def test_task_evaluator_handles_exact_match_and_multiple_choice(self) -> None:
        exact_match_config = load_config(ROOT / "configs/exp/benchmark_kodcode_qwen25_smoke.yaml")
        exact_match_dataset = load_task_dataset(exact_match_config)
        exact_match_evaluator = build_task_evaluator(exact_match_config)
        exact_match_score = exact_match_evaluator.evaluate_prediction(
            {"text": exact_match_dataset[0]["gold_answer"]},
            exact_match_dataset[0],
        )
        self.assertTrue(exact_match_score["correct"])

        multiple_choice_config = load_config(ROOT / "configs/exp/benchmark_gpqa_qwen25_smoke.yaml")
        multiple_choice_dataset = load_task_dataset(multiple_choice_config)
        multiple_choice_evaluator = build_task_evaluator(multiple_choice_config)
        multiple_choice_score = multiple_choice_evaluator.evaluate_prediction(
            {"label": multiple_choice_dataset[0]["label"]},
            multiple_choice_dataset[0],
        )
        self.assertTrue(multiple_choice_score["correct"])

    def test_load_task_dataset_builds_segmented_narrativeqa_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "narrativeqa.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "narrativeqa-001",
                        "story": "Alice travels to Paris.",
                        "story_segments": [
                            "Alice leaves London.",
                            "She arrives in Paris and investigates a mystery.",
                        ],
                        "story_chunk_pool": [
                            "Alice leaves London.",
                            "She waits at the station in silence.",
                            "She arrives in Paris and investigates a mystery.",
                            "A friend sends a telegram from Rome.",
                            "The detective recovers the stolen map in Paris.",
                            "The story closes with a report from London.",
                        ],
                        "question": "Where does Alice investigate the mystery?",
                        "answer": "Paris",
                        "aliases": ["Paris"],
                        "summary_title": "Alice Story",
                        "document_kind": "movie",
                        "story_chars": 1200,
                        "story_word_count": 250,
                        "story_excerpt_chars": 78,
                        "story_segment_words": 160,
                        "story_segments_materialized": 2,
                        "story_total_segments": 6,
                        "story_chunk_pool_size": 6,
                        "story_selected_indexes": [0, 2],
                        "story_start_index": 24,
                        "story_selection_strategy": "anchors_plus_question_overlap",
                        "story_query_token_count": 3,
                        "story_truncated_for_smoke": True,
                        "narrativeqa_view": "full_text_segmented",
                    }
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "benchmark_narrativeqa_test",
                    "stage": "M4",
                    "method_variant": "ours-benchmark-real-smoke",
                },
                "task": {
                    "name": "narrativeqa_real_smoke",
                    "benchmark_id": "narrativeqa",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "hf_real_smoke4_full_text_segmented",
                    "dataset_path": str(dataset_path),
                    "metric_name": "f1",
                    "narrativeqa_runtime": {"selector": "question_aware", "segment_budget": 3},
                    "evaluator": {"type": "qa_f1"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "method": {
                    "embed_dim": 64,
                    "segmenter": {"mode": "delimiter", "delimiter": "||"},
                    "writer": {"memory_slots": 4, "arch": "mlp"},
                    "reader": {
                        "num_queries": 4,
                        "use_query_gating": False,
                        "gating_mode": "off",
                        "num_heads": 4,
                        "condition_on_context": True,
                        "conditioning": {"domain_key": "domain", "include_task_name": True},
                    },
                    "fuser": {"short_slots": 2, "arch": "linear"},
                    "injector": {"mode": "prefix", "enabled": True, "position": "segment"},
                },
                "runtime": {
                    "train_steps": 2,
                    "eval_examples": 1,
                    "learning_rate": 0.01,
                    "device": "cpu",
                },
            }
            config_path = tmp / "narrativeqa.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            dataset = load_task_dataset(load_config(config_path))
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["benchmark_id"], "narrativeqa")
            self.assertIn("Story segment 1/3 [pool 1/6]", dataset[0]["segment"])
            self.assertIn("Story segment 2/3 [pool 3/6]", dataset[0]["segment"])
            self.assertIn("Question: Where does Alice investigate the mystery?", dataset[0]["segment"])
            self.assertEqual(dataset[0]["narrativeqa_view"], "full_text_segmented")
            self.assertEqual(len(dataset[0]["story_segments"]), 3)
            self.assertIn(2, dataset[0]["story_selected_indexes"])
            self.assertEqual(dataset[0]["story_start_index"], 24)
            self.assertEqual(dataset[0]["story_query_token_count"], 3)
            self.assertEqual(dataset[0]["story_runtime_segment_budget"], 3)
            self.assertEqual(dataset[0]["story_runtime_selector"], "question_aware")

    def test_load_task_dataset_supports_narrativeqa_anchor_only_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "narrativeqa_anchor.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "narrativeqa-anchor-001",
                        "story": "Alice follows a long trail.",
                        "story_segments": [
                            "Intro paragraph.",
                            "After the gala, Alice feared betrayal but had no proof.",
                            "Victor later confessed that he betrayed the team.",
                        ],
                        "story_chunk_pool": [
                            "Intro paragraph.",
                            "Travel notes.",
                            "After the gala, Alice feared betrayal but had no proof.",
                            "A quiet marketplace scene unfolds.",
                            "Victor later confessed that he betrayed the team.",
                            "The detective revisits old notes.",
                            "An epilogue closes the book.",
                            "Credits roll.",
                        ],
                        "question": "Who betrayed Alice after the gala?",
                        "answer": "Victor",
                        "aliases": ["Victor"],
                        "summary_title": "Anchor Test",
                        "document_kind": "book",
                        "story_chars": 100,
                        "story_word_count": 20,
                        "story_excerpt_chars": 60,
                        "story_segment_words": 128,
                        "story_segments_materialized": 3,
                        "story_total_segments": 8,
                        "story_chunk_pool_size": 8,
                        "story_selected_indexes": [0, 2, 4],
                        "story_start_index": 7,
                        "story_selection_strategy": "anchors_plus_question_overlap",
                        "story_query_token_count": 4,
                        "story_truncated_for_smoke": True,
                        "narrativeqa_view": "full_text_segmented",
                    }
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "benchmark_narrativeqa_anchor_test",
                    "stage": "M4",
                    "method_variant": "ours-benchmark-real-smoke",
                },
                "task": {
                    "name": "narrativeqa_real_smoke",
                    "benchmark_id": "narrativeqa",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "hf_real_smoke4_runtime_pool_anchor6x128",
                    "dataset_path": str(dataset_path),
                    "metric_name": "f1",
                    "narrativeqa_runtime": {"selector": "anchor_only", "segment_budget": 3},
                    "evaluator": {"type": "qa_f1"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "method": {
                    "embed_dim": 64,
                    "segmenter": {"mode": "delimiter", "delimiter": "||"},
                    "writer": {"memory_slots": 4, "arch": "mlp"},
                    "reader": {
                        "num_queries": 4,
                        "use_query_gating": False,
                        "gating_mode": "off",
                        "num_heads": 4,
                        "condition_on_context": True,
                        "conditioning": {"domain_key": "domain", "include_task_name": True},
                    },
                    "fuser": {"short_slots": 2, "arch": "linear"},
                    "injector": {"mode": "prefix", "enabled": True, "position": "segment"},
                },
                "runtime": {
                    "train_steps": 2,
                    "eval_examples": 1,
                    "learning_rate": 0.01,
                    "device": "cpu",
                },
            }
            config_path = tmp / "narrativeqa_anchor.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            dataset = load_task_dataset(load_config(config_path))
            self.assertEqual(dataset[0]["story_selected_indexes"], [0, 3, 7])
            self.assertEqual(dataset[0]["story_runtime_selector"], "anchor_only")
            self.assertEqual(dataset[0]["story_selection_strategy"], "anchor_only_even_spacing")


if __name__ == "__main__":
    unittest.main()
