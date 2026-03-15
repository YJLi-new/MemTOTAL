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

from memtotal.analysis.m3_stage_c_margin_audit import collect_stage_c_margin_rows
from memtotal.analysis.run_analysis import main as analysis_main


class M3StageCMarginAuditTest(unittest.TestCase):
    def _write_curve_run(
        self,
        root: Path,
        run_name: str,
        *,
        backbone: str,
        seed: int,
        zero_margin: float,
        step0_margin: float,
        final_margin: float,
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
                    "zero_shot_task_score": 0.4,
                    "zero_shot_task_proxy_score": 0.49,
                }
            )
        )
        (run_dir / "adapt_curve.csv").write_text(
            "\n".join(
                [
                    "query_learning_mode,query_objective,adaptation_target,trainable_module,trainable_parameter_count,shot,step,target_eval_repeats,target_episode_repeats,target_episode_policy,target_support_weighting,target_split_policy,evaluated_target_episodes,evaluated_query_examples,query_candidate_pool_size,support_candidate_pool_size,objective_loss,objective_accuracy,task_score,task_metric_name,task_proxy_score,task_proxy_name,task_margin,preceding_support_grad_norm,preceding_support_update_max_abs,preceding_support_update_l2,query_loss,query_accuracy",
                    f"meta_trained,continuation_retrieval,q_only,reader.queries,256,0,0,3,3,aggregate_support,uniform,random,3,9,6,3,2.0,0.0,0.4,accuracy,0.49,gold_choice_probability,{zero_margin},0.0,0.0,0.0,2.0,0.0",
                    f"meta_trained,continuation_retrieval,q_only,reader.queries,256,3,0,3,3,aggregate_support,uniform,random,3,9,6,3,1.8,0.0,0.4,accuracy,0.50,gold_choice_probability,{step0_margin},0.0,0.0,0.0,1.8,0.0",
                    f"meta_trained,continuation_retrieval,q_only,reader.queries,256,3,5,3,3,aggregate_support,uniform,random,3,9,6,3,1.7,0.0,0.4,accuracy,0.51,gold_choice_probability,{final_margin},0.0,0.0,0.0,1.7,0.0",
                ]
            )
        )

    def test_collect_stage_c_margin_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_curve_run(
                root,
                "qwen25-seed-1",
                backbone="Qwen2.5-1.5B-Instruct",
                seed=1,
                zero_margin=-0.30,
                step0_margin=-0.12,
                final_margin=-0.05,
            )
            rows = collect_stage_c_margin_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["zero_to_step0_margin_gain"], 0.18)
            self.assertAlmostEqual(rows[0]["step0_to_final_margin_gain"], 0.07)
            self.assertAlmostEqual(rows[0]["margin_gap_closed"], 0.25)
            self.assertTrue(rows[0]["margin_improves"])
            self.assertFalse(rows[0]["crosses_zero_margin"])

    def test_analysis_mode_writes_margin_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "runs"
            output_root = root / "summary"
            self._write_curve_run(
                input_root,
                "qwen25-seed-11",
                backbone="Qwen2.5-1.5B-Instruct",
                seed=11,
                zero_margin=-0.30,
                step0_margin=-0.12,
                final_margin=-0.05,
            )
            self._write_curve_run(
                input_root,
                "qwen3-seed-21",
                backbone="Qwen3-8B",
                seed=21,
                zero_margin=-0.20,
                step0_margin=-0.10,
                final_margin=0.02,
            )
            result = analysis_main(
                [
                    "--config",
                    str(ROOT / "configs/exp/m3_stage_c_margin_audit.yaml"),
                    "--seed",
                    "21",
                    "--output_dir",
                    str(output_root),
                    "--input_root",
                    str(input_root),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(output_root.joinpath("audit_rows.csv").exists())
            self.assertTrue(output_root.joinpath("audit_summary.svg").exists())
            metrics = json.loads(output_root.joinpath("metrics.json").read_text())
            self.assertEqual(metrics["analysis_mode"], "m3_stage_c_margin_audit")
            self.assertEqual(metrics["rows_collected"], 2)
            self.assertGreater(
                metrics["by_backbone"]["Qwen2.5-1.5B-Instruct"]["mean_margin_gap_closed"],
                0.0,
            )
            self.assertEqual(
                metrics["by_backbone_negative_only"]["Qwen2.5-1.5B-Instruct"]["seed_count"],
                1,
            )
            self.assertEqual(
                metrics["by_backbone"]["Qwen3-8B"]["cross_zero_margin_rate"],
                1.0,
            )
            self.assertEqual(
                metrics["by_backbone_negative_only"]["Qwen3-8B"]["cross_zero_margin_rate"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
