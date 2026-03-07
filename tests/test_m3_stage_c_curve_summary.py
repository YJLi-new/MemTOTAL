from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.analysis.m3_stage_c_curve_summary import collect_stage_c_curve_rows
from memtotal.analysis.run_analysis import main as analysis_main


class M3StageCCurveSummaryTest(unittest.TestCase):
    def _write_stage_c_run(
        self,
        root: Path,
        run_name: str,
        *,
        backbone: str,
        seed: int,
        zero_shot_task_score: float,
        shot3_step5_score: float,
    ) -> None:
        run_dir = root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "run_info.json").write_text(
            json.dumps({"seed": seed, "backbone": backbone, "task_name": "core4_transfer_smoke"})
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "mode": "train",
                    "training_stage": "stage_c",
                    "query_learning_mode": "meta_trained",
                    "query_objective": "continuation_retrieval",
                    "adaptation_target": "q_only",
                    "target_episode_policy": "aggregate_support",
                    "target_support_weighting": "uniform",
                    "target_split_policy": "proxy_bottomk_support",
                    "zero_shot_task_score": zero_shot_task_score,
                    "zero_shot_task_proxy_score": 0.49,
                }
            )
        )
        with (run_dir / "adapt_curve.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_learning_mode",
                    "query_objective",
                    "adaptation_target",
                    "trainable_module",
                    "trainable_parameter_count",
                    "shot",
                    "step",
                    "target_eval_repeats",
                    "target_episode_repeats",
                    "target_episode_policy",
                    "target_support_weighting",
                    "target_split_policy",
                    "evaluated_target_episodes",
                    "evaluated_query_examples",
                    "query_candidate_pool_size",
                    "support_candidate_pool_size",
                    "objective_loss",
                    "objective_accuracy",
                    "task_score",
                    "task_metric_name",
                    "task_proxy_score",
                    "task_proxy_name",
                    "task_margin",
                    "preceding_support_grad_norm",
                    "preceding_support_update_max_abs",
                    "preceding_support_update_l2",
                    "query_loss",
                    "query_accuracy",
                ],
            )
            writer.writeheader()
            for shot, step, task_score in [
                (0, 0, zero_shot_task_score),
                (1, 0, zero_shot_task_score + 0.1),
                (1, 5, zero_shot_task_score + 0.15),
                (3, 0, zero_shot_task_score + 0.2),
                (3, 5, shot3_step5_score),
            ]:
                writer.writerow(
                    {
                        "query_learning_mode": "meta_trained",
                        "query_objective": "continuation_retrieval",
                        "adaptation_target": "q_only",
                        "trainable_module": "reader.queries",
                        "trainable_parameter_count": 256,
                        "shot": shot,
                        "step": step,
                        "target_eval_repeats": 3,
                        "target_episode_repeats": 3,
                        "target_episode_policy": "aggregate_support",
                        "target_support_weighting": "uniform",
                        "target_split_policy": "proxy_bottomk_support",
                        "evaluated_target_episodes": 3,
                        "evaluated_query_examples": 9,
                        "query_candidate_pool_size": 6,
                        "support_candidate_pool_size": 3,
                        "objective_loss": 2.0 - (0.1 * shot) - (0.01 * step),
                        "objective_accuracy": 0.0,
                        "task_score": task_score,
                        "task_metric_name": "accuracy",
                        "task_proxy_score": 0.49 + (0.01 * shot) + (0.001 * step),
                        "task_proxy_name": "gold_choice_probability",
                        "task_margin": 0.02 + (0.01 * shot),
                        "preceding_support_grad_norm": 0.01,
                        "preceding_support_update_max_abs": 1.0e-3,
                        "preceding_support_update_l2": 1.0e-2,
                        "query_loss": 1.8 - (0.1 * shot) - (0.01 * step),
                        "query_accuracy": 0.0,
                    }
                )

    def test_collect_stage_c_curve_rows_reads_curve_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_stage_c_run(
                root,
                "qwen25-seed-1",
                backbone="Qwen2.5-1.5B-Instruct",
                seed=1,
                zero_shot_task_score=0.3,
                shot3_step5_score=0.8,
            )
            rows = collect_stage_c_curve_rows(root)
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["backbone"], "Qwen2.5-1.5B-Instruct")
            self.assertEqual(rows[0]["target_split_policy"], "proxy_bottomk_support")
            self.assertEqual(rows[-1]["shot"], 3)
            self.assertEqual(rows[-1]["step"], 5)

    def test_analysis_mode_writes_curve_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "runs"
            output_root = root / "summary"
            self._write_stage_c_run(
                input_root,
                "qwen25-seed-11",
                backbone="Qwen2.5-1.5B-Instruct",
                seed=11,
                zero_shot_task_score=0.3,
                shot3_step5_score=0.8,
            )
            self._write_stage_c_run(
                input_root,
                "qwen3-seed-21",
                backbone="Qwen3-8B",
                seed=21,
                zero_shot_task_score=0.4,
                shot3_step5_score=0.7,
            )
            result = analysis_main(
                [
                    "--config",
                    str(ROOT / "configs/exp/m3_stage_c_curve_summary.yaml"),
                    "--seed",
                    "21",
                    "--output_dir",
                    str(output_root),
                    "--input_root",
                    str(input_root),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(output_root.joinpath("curve_rows.csv").exists())
            self.assertTrue(output_root.joinpath("shot_curve.csv").exists())
            self.assertTrue(output_root.joinpath("step_curve.csv").exists())
            self.assertTrue(output_root.joinpath("shot_curve.svg").exists())
            self.assertTrue(output_root.joinpath("step_curve.svg").exists())
            metrics = json.loads(output_root.joinpath("metrics.json").read_text())
            self.assertEqual(metrics["analysis_mode"], "m3_stage_c_curve_summary")
            self.assertEqual(metrics["rows_collected"], 10)
            self.assertEqual(metrics["shot_curve_rows"], 6)
            self.assertEqual(metrics["step_curve_rows"], 4)
            self.assertEqual(
                metrics["best_final_by_backbone"]["Qwen2.5-1.5B-Instruct"]["target_split_policy"],
                "proxy_bottomk_support",
            )
            self.assertEqual(metrics["best_final_by_backbone"]["Qwen2.5-1.5B-Instruct"]["shot"], 3)


if __name__ == "__main__":
    unittest.main()
