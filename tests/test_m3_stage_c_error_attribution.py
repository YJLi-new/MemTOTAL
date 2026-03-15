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

from memtotal.analysis.run_analysis import main as analysis_main


class M3StageCErrorAttributionTest(unittest.TestCase):
    def test_analysis_mode_writes_case_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "qwen25-seed-1"
            run_dir.mkdir(parents=True)
            (run_dir / "run_info.json").write_text(
                json.dumps({"seed": 1, "backbone": "Qwen2.5-1.5B-Instruct"})
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "train",
                        "training_stage": "stage_c",
                        "adaptation_target": "q_only",
                        "best_adapt_shot": 3,
                        "best_adapt_step": 3,
                    }
                )
            )
            case_rows = [
                {
                    "shot": 0,
                    "step": 0,
                    "example_id": "near-flip",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "task_name": "story_cloze_narrative_meta",
                    "segment": "Story: Anna left early. Candidate endings: A: Anna came back later. | B: Anna never returned.",
                    "gold_label": "A",
                    "gold_text": "Anna came back later.",
                    "predicted_label": "B",
                    "predicted_text": "Anna never returned.",
                    "predicted_correct": False,
                    "top_competitor_label": "B",
                    "top_competitor_text": "Anna never returned.",
                    "task_score": 0.0,
                    "task_proxy_score": 0.495,
                    "task_margin": -0.015,
                    "gold_probability": 0.495,
                    "top_competitor_probability": 0.505,
                    "support_ids": ["s1", "s2", "s3"],
                    "eval_query_set_ids": ["near-flip"],
                    "choices": [],
                },
                {
                    "shot": 3,
                    "step": 3,
                    "example_id": "near-flip",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "task_name": "story_cloze_narrative_meta",
                    "segment": "Story: Anna left early. Candidate endings: A: Anna came back later. | B: Anna never returned.",
                    "gold_label": "A",
                    "gold_text": "Anna came back later.",
                    "predicted_label": "B",
                    "predicted_text": "Anna never returned.",
                    "predicted_correct": False,
                    "top_competitor_label": "B",
                    "top_competitor_text": "Anna never returned.",
                    "task_score": 0.0,
                    "task_proxy_score": 0.499,
                    "task_margin": -0.004,
                    "gold_probability": 0.499,
                    "top_competitor_probability": 0.501,
                    "support_ids": ["s4", "s5", "s6"],
                    "eval_query_set_ids": ["near-flip"],
                    "choices": [],
                },
                {
                    "shot": 0,
                    "step": 0,
                    "example_id": "flipped",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "task_name": "story_cloze_narrative_meta",
                    "segment": "Story: Bob trained. Candidate endings: A: Bob won. | B: Bob quit.",
                    "gold_label": "A",
                    "gold_text": "Bob won.",
                    "predicted_label": "B",
                    "predicted_text": "Bob quit.",
                    "predicted_correct": False,
                    "top_competitor_label": "B",
                    "top_competitor_text": "Bob quit.",
                    "task_score": 0.0,
                    "task_proxy_score": 0.46,
                    "task_margin": -0.08,
                    "gold_probability": 0.46,
                    "top_competitor_probability": 0.54,
                    "support_ids": ["s1", "s2", "s3"],
                    "eval_query_set_ids": ["flipped"],
                    "choices": [],
                },
                {
                    "shot": 3,
                    "step": 3,
                    "example_id": "flipped",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "task_name": "story_cloze_narrative_meta",
                    "segment": "Story: Bob trained. Candidate endings: A: Bob won. | B: Bob quit.",
                    "gold_label": "A",
                    "gold_text": "Bob won.",
                    "predicted_label": "A",
                    "predicted_text": "Bob won.",
                    "predicted_correct": True,
                    "top_competitor_label": "B",
                    "top_competitor_text": "Bob quit.",
                    "task_score": 1.0,
                    "task_proxy_score": 0.54,
                    "task_margin": 0.03,
                    "gold_probability": 0.54,
                    "top_competitor_probability": 0.46,
                    "support_ids": ["s4", "s5", "s6"],
                    "eval_query_set_ids": ["flipped"],
                    "choices": [],
                },
            ]
            (run_dir / "task_case_dump.jsonl").write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in case_rows) + "\n"
            )

            output_dir = root / "report"
            result = analysis_main(
                [
                    "--config",
                    str(ROOT / "configs/exp/m3_stage_c_error_attribution.yaml"),
                    "--seed",
                    "11",
                    "--output_dir",
                    str(output_dir),
                    "--input_root",
                    str(root / "runs"),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(output_dir.joinpath("case_pairs.csv").exists())
            self.assertTrue(output_dir.joinpath("near_threshold_bad_cases.csv").exists())
            self.assertTrue(output_dir.joinpath("stubborn_wrong_cases.csv").exists())
            self.assertTrue(output_dir.joinpath("report.md").exists())
            metrics = json.loads(output_dir.joinpath("metrics.json").read_text())
            self.assertEqual(metrics["analysis_mode"], "m3_stage_c_error_attribution")
            self.assertEqual(metrics["rows_collected"], 2)
            self.assertEqual(metrics["near_threshold_bad_case_count"], 1)
            self.assertEqual(
                metrics["by_backbone"]["Qwen2.5-1.5B-Instruct"]["cross_zero_margin_count"],
                1,
            )
            report_text = output_dir.joinpath("report.md").read_text()
            self.assertIn("near-flip", report_text)
            near_threshold_csv = output_dir.joinpath("near_threshold_bad_cases.csv").read_text()
            self.assertIn("near_rank_flip", near_threshold_csv)


if __name__ == "__main__":
    unittest.main()
