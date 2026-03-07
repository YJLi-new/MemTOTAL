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

from memtotal.analysis.m3_stage_c_seed_sweep import collect_stage_c_seed_sweep_rows
from memtotal.analysis.run_analysis import main as analysis_main


class M3StageCSeedSweepTest(unittest.TestCase):
    def _write_stage_c_run(
        self,
        root: Path,
        run_name: str,
        *,
        backbone: str,
        seed: int,
        zero_shot_task_score: float,
        best_adapt_task_score: float,
        zero_shot_task_proxy_score: float,
        best_adapt_task_proxy_score: float,
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
                    "trainable_module": "reader.queries",
                    "trainable_parameter_count": 256,
                    "adapt_learning_rate": 0.2,
                    "adapt_steps": 3,
                    "target_eval_repeats": 3,
                    "zero_shot_task_score": zero_shot_task_score,
                    "best_adapt_task_score": best_adapt_task_score,
                    "task_metric_name": "accuracy",
                    "zero_shot_task_proxy_score": zero_shot_task_proxy_score,
                    "best_adapt_task_proxy_score": best_adapt_task_proxy_score,
                    "task_proxy_name": "gold_choice_probability",
                    "best_adapt_task_margin": 0.1,
                    "zero_shot_query_loss": 1.7,
                    "best_adapt_query_loss": 1.5,
                    "best_adapt_shot": 3,
                    "best_adapt_step": 3,
                    "adaptation_effective": True,
                }
            )
        )

    def test_collect_stage_c_seed_sweep_rows_keeps_seed_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_stage_c_run(
                root,
                "qwen25-seed-1",
                backbone="Qwen2.5-1.5B-Instruct",
                seed=1,
                zero_shot_task_score=0.5,
                best_adapt_task_score=0.75,
                zero_shot_task_proxy_score=0.45,
                best_adapt_task_proxy_score=0.55,
            )
            self._write_stage_c_run(
                root,
                "qwen3-seed-2",
                backbone="Qwen3-8B",
                seed=2,
                zero_shot_task_score=0.5,
                best_adapt_task_score=0.25,
                zero_shot_task_proxy_score=0.50,
                best_adapt_task_proxy_score=0.45,
            )
            rows = collect_stage_c_seed_sweep_rows(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["backbone"], "Qwen2.5-1.5B-Instruct")
            self.assertEqual(rows[0]["seed"], 1)
            self.assertEqual(rows[0]["target_eval_repeats"], 3)
            self.assertAlmostEqual(float(rows[0]["task_gain"]), 0.25)
            self.assertAlmostEqual(float(rows[1]["task_gain"]), -0.25)

    def test_analysis_mode_writes_seed_sweep_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "runs"
            output_root = root / "summary"
            self._write_stage_c_run(
                input_root,
                "qwen25-seed-11",
                backbone="Qwen2.5-1.5B-Instruct",
                seed=11,
                zero_shot_task_score=0.5,
                best_adapt_task_score=0.75,
                zero_shot_task_proxy_score=0.45,
                best_adapt_task_proxy_score=0.60,
            )
            self._write_stage_c_run(
                input_root,
                "qwen25-seed-12",
                backbone="Qwen2.5-1.5B-Instruct",
                seed=12,
                zero_shot_task_score=0.5,
                best_adapt_task_score=0.5,
                zero_shot_task_proxy_score=0.45,
                best_adapt_task_proxy_score=0.48,
            )
            self._write_stage_c_run(
                input_root,
                "qwen3-seed-21",
                backbone="Qwen3-8B",
                seed=21,
                zero_shot_task_score=0.5,
                best_adapt_task_score=0.25,
                zero_shot_task_proxy_score=0.50,
                best_adapt_task_proxy_score=0.44,
            )

            result = analysis_main(
                [
                    "--config",
                    str(ROOT / "configs/exp/m3_stage_c_seed_sweep_summary.yaml"),
                    "--seed",
                    "11",
                    "--output_dir",
                    str(output_root),
                    "--input_root",
                    str(input_root),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(output_root.joinpath("seed_sweep.csv").exists())
            self.assertTrue(output_root.joinpath("seed_sweep.svg").exists())
            metrics = json.loads(output_root.joinpath("metrics.json").read_text())
            self.assertEqual(metrics["analysis_mode"], "m3_stage_c_seed_sweep_summary")
            self.assertEqual(metrics["rows_collected"], 3)
            self.assertEqual(metrics["by_backbone"]["Qwen2.5-1.5B-Instruct"]["seed_count"], 2)
            self.assertEqual(metrics["by_backbone"]["Qwen2.5-1.5B-Instruct"]["positive_gain_count"], 1)
            self.assertAlmostEqual(
                metrics["by_backbone"]["Qwen2.5-1.5B-Instruct"]["positive_gain_rate"],
                0.5,
            )
            self.assertEqual(metrics["by_backbone"]["Qwen2.5-1.5B-Instruct"]["target_eval_repeats"], [3])
            self.assertEqual(metrics["by_backbone"]["Qwen3-8B"]["worst_seed"], 21)
            self.assertAlmostEqual(metrics["by_backbone"]["Qwen3-8B"]["mean_task_gain"], -0.25)


if __name__ == "__main__":
    unittest.main()
