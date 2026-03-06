from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.eval.run_eval import main as eval_main


class BenchmarkEvalTest(unittest.TestCase):
    def test_exact_match_benchmark_eval_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "gsm8k_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/benchmark_gsm8k_qwen25_smoke.yaml"),
                        "--seed",
                        "501",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            run_info = json.loads((output_dir / "run_info.json").read_text())
            self.assertEqual(metrics["benchmark_id"], "gsm8k")
            self.assertEqual(metrics["task_domain"], "math")
            self.assertEqual(metrics["evaluator_type"], "exact_match")
            self.assertEqual(run_info["smoke_subset"], "local_contract_v1")
            self.assertEqual(len(predictions), 2)
            self.assertEqual(predictions[0]["benchmark_id"], "gsm8k")
            self.assertIn("predicted_text", predictions[0])

    def test_multiple_choice_benchmark_eval_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "gpqa_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/benchmark_gpqa_qwen25_smoke.yaml"),
                        "--seed",
                        "503",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["benchmark_id"], "gpqa")
            self.assertEqual(metrics["evaluator_type"], "multiple_choice")
            self.assertEqual(metrics["smoke_subset"], "local_contract_v1")
            self.assertEqual(len(predictions), 2)
            self.assertIn("predicted_label", predictions[0])
            self.assertIn("score", predictions[0])


if __name__ == "__main__":
    unittest.main()
