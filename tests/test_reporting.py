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

from memtotal.analysis.reporting import collect_metrics, write_sanity_plot


class ReportingTest(unittest.TestCase):
    def test_collect_metrics_resolves_primary_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            memgen_dir = temp_root / "memgen"
            train_dir = temp_root / "train"
            memgen_dir.mkdir()
            train_dir.mkdir()

            (memgen_dir / "run_info.json").write_text(
                json.dumps({"backbone": "Qwen2.5-1.5B-Instruct", "task_name": "gsm8k"})
            )
            (memgen_dir / "metrics.json").write_text(
                json.dumps({"mode": "memgen_adapter", "compute_reward": 0.25})
            )

            (train_dir / "run_info.json").write_text(
                json.dumps({"backbone": "Qwen3-8B", "task_name": "toy_reasoning_smoke"})
            )
            (train_dir / "metrics.json").write_text(json.dumps({"mode": "train", "mean_loss": 0.5}))

            stage_c_dir = temp_root / "stage_c"
            stage_c_dir.mkdir()
            (stage_c_dir / "run_info.json").write_text(
                json.dumps({"backbone": "Qwen2.5-1.5B-Instruct", "task_name": "toy_meta_smoke"})
            )
            (stage_c_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "train",
                        "training_stage": "stage_c",
                        "best_adapt_query_accuracy": 0.5,
                        "best_adapt_query_loss": 1.2,
                    }
                )
            )

            stage_b_non_meta_dir = temp_root / "stage_b_non_meta"
            stage_b_non_meta_dir.mkdir()
            (stage_b_non_meta_dir / "run_info.json").write_text(
                json.dumps({"backbone": "Qwen2.5-1.5B-Instruct", "task_name": "toy_meta_smoke"})
            )
            (stage_b_non_meta_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "train",
                        "training_stage": "stage_b",
                        "query_learning_mode": "non_meta_multitask",
                        "source_eval_query_accuracy": 0.75,
                        "source_eval_query_loss": 0.67,
                    }
                )
            )

            failure_checks_dir = temp_root / "failure_checks"
            failure_checks_dir.mkdir()
            (failure_checks_dir / "run_info.json").write_text(
                json.dumps({"backbone": "Qwen2.5-1.5B-Instruct", "task_name": "toy_meta_smoke"})
            )
            (failure_checks_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "analysis_failure_checks",
                        "checks_pass_rate": 2.0 / 3.0,
                    }
                )
            )

            memoryagentbench_dir = temp_root / "memoryagentbench"
            memoryagentbench_dir.mkdir()
            (memoryagentbench_dir / "run_info.json").write_text(
                json.dumps({"backbone": "Qwen2.5-1.5B-Instruct", "task_name": "memoryagentbench_real_smoke"})
            )
            (memoryagentbench_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "eval",
                        "benchmark_id": "memoryagentbench",
                        "memoryagent_score": 0.125,
                        "accuracy": 0.0,
                        "capability_scores": {"AR": 0.0, "LRU": 0.5},
                        "capability_metric_names": {"AR": "exact_match", "LRU": "rougeLsum_f1"},
                    }
                )
            )

            rows = collect_metrics(temp_root)
            by_mode = {str(row["mode"]): row for row in rows}
            by_stage = {str(row.get("training_stage", "")): row for row in rows}
            by_mode_and_query = {
                (str(row["mode"]), str(row.get("query_learning_mode", ""))): row for row in rows
            }
            memoryagentbench_row = next(row for row in rows if row.get("benchmark_id") == "memoryagentbench")

            self.assertEqual(by_mode["memgen_adapter"]["primary_metric"], "compute_reward")
            self.assertEqual(by_mode["memgen_adapter"]["primary_score"], 0.25)
            self.assertEqual(by_mode["train"]["primary_metric"], "inv_mean_loss")
            self.assertAlmostEqual(by_mode["train"]["primary_score"], 2.0 / 3.0)
            self.assertEqual(by_stage["stage_c"]["primary_metric"], "best_adapt_query_accuracy")
            self.assertEqual(by_stage["stage_c"]["primary_score"], 0.5)
            self.assertEqual(
                by_mode_and_query[("train", "non_meta_multitask")]["primary_metric"],
                "source_eval_query_accuracy",
            )
            self.assertEqual(
                by_mode_and_query[("train", "non_meta_multitask")]["primary_score"],
                0.75,
            )
            self.assertEqual(by_mode["analysis_failure_checks"]["primary_metric"], "checks_pass_rate")
            self.assertAlmostEqual(by_mode["analysis_failure_checks"]["primary_score"], 2.0 / 3.0)
            self.assertEqual(memoryagentbench_row["primary_metric"], "memoryagent_score")
            self.assertEqual(memoryagentbench_row["primary_score"], 0.125)
            self.assertEqual(memoryagentbench_row["capability_AR_metric"], "exact_match")
            self.assertEqual(memoryagentbench_row["capability_LRU_score"], 0.5)

    def test_write_sanity_plot_uses_primary_metric_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.svg"
            rows = [
                {
                    "run_dir": str(Path(temp_dir) / "memgen-smoke"),
                    "mode": "memgen_adapter",
                    "primary_metric": "compute_reward",
                    "primary_score": 0.25,
                }
            ]
            write_sanity_plot(output_path, rows)
            text = output_path.read_text()
            self.assertIn("compute_reward", text)
            self.assertIn("0.2500", text)


if __name__ == "__main__":
    unittest.main()
