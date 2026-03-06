from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.tasks.evaluator import TaskEvaluator
from memtotal.tasks.memoryagentbench import (
    CAPABILITY_SPECS,
    build_memoryagentbench_smoke_examples,
    truncate_memoryagentbench_context,
)


class MemoryAgentBenchTest(unittest.TestCase):
    def test_context_truncation_limits_token_budget(self) -> None:
        context = " ".join(f"token-{index}" for index in range(20))
        truncated, total_tokens, was_truncated = truncate_memoryagentbench_context(context, max_context_tokens=7)
        self.assertEqual(total_tokens, 20)
        self.assertTrue(was_truncated)
        self.assertEqual(len(truncated.split()), 7)

    def test_build_smoke_examples_preserves_capability_metadata(self) -> None:
        row = {
            "context": "Alpha beta gamma delta epsilon zeta eta theta",
            "questions": ["Question one?", "Question two?"],
            "answers": [["Answer One", "answer one"], ["Answer Two"]],
            "metadata": {
                "source": CAPABILITY_SPECS[0].source,
                "qa_pair_ids": ["pair-1", "pair-2"],
                "question_types": ["factoid", "factoid"],
                "keypoints": ["Keypoint A", "Keypoint B"],
            },
        }
        examples = build_memoryagentbench_smoke_examples(
            row,
            capability=CAPABILITY_SPECS[0],
            take_questions=1,
            max_context_tokens=4,
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["capability"], "AR")
        self.assertEqual(examples[0]["memoryagent_source"], CAPABILITY_SPECS[0].source)
        self.assertEqual(examples[0]["aliases"], ["Answer One", "answer one"])
        self.assertTrue(examples[0]["context_was_truncated"])
        self.assertEqual(examples[0]["qa_pair_id"], "pair-1")

    def test_memoryagentbench_evaluator_reports_rouge_style_metrics(self) -> None:
        evaluator = TaskEvaluator(
            evaluator_type="memoryagentbench",
            metric_name="memoryagent_score",
            benchmark_id="memoryagentbench",
        )
        example = {
            "continuation": "Alice visits Paris and writes a summary.",
            "gold_answer": "Alice visits Paris and writes a summary.",
            "aliases": ["Alice visits Paris and writes a summary."],
            "capability": "LRU",
            "capability_metric_name": "rougeLsum_f1",
            "keypoints": ["Alice visits Paris", "writes a summary"],
        }
        score = evaluator.evaluate_prediction({"text": "Alice visits Paris and writes a summary."}, example)
        self.assertEqual(score["capability"], "LRU")
        self.assertIn("rougeLsum_f1", score["extra_metrics"])
        self.assertAlmostEqual(score["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
