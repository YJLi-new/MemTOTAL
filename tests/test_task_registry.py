from __future__ import annotations

import sys
import unittest
from pathlib import Path


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
                "rocstories",
                "fever",
                "alfworld",
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


if __name__ == "__main__":
    unittest.main()
