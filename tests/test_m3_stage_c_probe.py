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

from memtotal.analysis.m3_stage_c_probe import collect_stage_c_probe_rows
from memtotal.analysis.run_analysis import main as analysis_main


class M3StageCProbeTest(unittest.TestCase):
    def _write_stage_c_run(
        self,
        root: Path,
        run_name: str,
        *,
        backbone: str,
        adaptation_target: str,
        zero_shot_task_score: float,
        best_adapt_task_score: float,
        adaptation_effective: bool,
        adapt_learning_rate: float = 0.2,
        adapt_steps: int = 3,
        target_eval_repeats: int = 1,
        target_episode_repeats: int = 1,
        target_episode_policy: str = "independent",
        best_adapt_query_loss: float = 1.6,
        zero_shot_task_proxy_score: float = 0.4,
        best_adapt_task_proxy_score: float = 0.4,
    ) -> None:
        run_dir = root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "run_info.json").write_text(
            json.dumps({"seed": 3201, "backbone": backbone, "task_name": "core4_transfer_smoke"})
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "mode": "train",
                    "training_stage": "stage_c",
                    "query_learning_mode": "meta_trained",
                    "query_objective": "continuation_retrieval",
                    "adaptation_target": adaptation_target,
                    "trainable_module": "reader.queries" if adaptation_target == "q_only" else "writer",
                    "trainable_parameter_count": 256 if adaptation_target == "q_only" else 71744,
                    "adapt_learning_rate": adapt_learning_rate,
                    "adapt_steps": adapt_steps,
                    "target_eval_repeats": target_eval_repeats,
                    "target_episode_repeats": target_episode_repeats,
                    "target_episode_policy": target_episode_policy,
                    "zero_shot_task_score": zero_shot_task_score,
                    "best_adapt_task_score": best_adapt_task_score,
                    "task_metric_name": "accuracy",
                    "zero_shot_task_proxy_score": zero_shot_task_proxy_score,
                    "best_adapt_task_proxy_score": best_adapt_task_proxy_score,
                    "task_proxy_name": "gold_choice_probability",
                    "best_adapt_task_margin": 0.1,
                    "zero_shot_query_loss": 1.7,
                    "best_adapt_query_loss": best_adapt_query_loss,
                    "best_adapt_shot": 3,
                    "best_adapt_step": 2,
                    "mean_support_grad_norm": 0.01,
                    "max_support_update_max_abs": 1.0e-3 if adaptation_effective else 1.0e-9,
                    "adaptation_effective": adaptation_effective,
                }
            )
        )

    def _write_gradient_audit_run(
        self,
        root: Path,
        run_name: str,
        *,
        backbone: str,
        adaptation_target: str,
        query_to_writer_grad_ratio: float,
    ) -> None:
        run_dir = root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "run_info.json").write_text(
            json.dumps({"seed": 3201, "backbone": backbone, "task_name": "core4_transfer_smoke"})
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "mode": "analysis",
                    "analysis_mode": "m3_stage_c_gradient_audit",
                    "backbone": backbone,
                    "adaptation_target": adaptation_target,
                    "queries_grad_norm": 1.0e-8,
                    "reader_non_query_grad_norm": 1.0e-2,
                    "fuser_grad_norm": 1.0e-1,
                    "writer_grad_norm": 2.0e-2,
                    "query_to_fuser_grad_ratio": 1.0e-7,
                    "query_to_writer_grad_ratio": query_to_writer_grad_ratio,
                }
            )
        )

    def test_collect_stage_c_probe_rows_joins_gradient_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_stage_c_run(
                root,
                "qwen25-q-only",
                backbone="Qwen2.5-1.5B-Instruct",
                adaptation_target="q_only",
                zero_shot_task_score=0.25,
                best_adapt_task_score=0.25,
                adaptation_effective=False,
            )
            self._write_gradient_audit_run(
                root,
                "qwen25-q-only-gradient-audit",
                backbone="Qwen2.5-1.5B-Instruct",
                adaptation_target="q_only",
                query_to_writer_grad_ratio=1.6e-6,
            )
            rows, gradient_audits = collect_stage_c_probe_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(gradient_audits), 1)
            self.assertEqual(rows[0]["adaptation_target"], "q_only")
            self.assertEqual(rows[0]["seed"], 3201)
            self.assertEqual(rows[0]["adapt_learning_rate"], 0.2)
            self.assertEqual(rows[0]["target_eval_repeats"], 1)
            self.assertEqual(rows[0]["target_episode_repeats"], 1)
            self.assertEqual(rows[0]["target_episode_policy"], "independent")
            self.assertAlmostEqual(float(rows[0]["task_gain"]), 0.0)
            self.assertAlmostEqual(float(rows[0]["query_to_writer_grad_ratio"]), 1.6e-6)
            self.assertEqual(rows[0]["task_proxy_name"], "gold_choice_probability")

    def test_analysis_mode_writes_stage_c_probe_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "runs"
            output_root = root / "summary"
            self._write_stage_c_run(
                input_root,
                "qwen25-q-only",
                backbone="Qwen2.5-1.5B-Instruct",
                adaptation_target="q_only",
                zero_shot_task_score=0.25,
                best_adapt_task_score=0.25,
                adaptation_effective=False,
                zero_shot_task_proxy_score=0.40,
                best_adapt_task_proxy_score=0.55,
                target_eval_repeats=3,
                target_episode_repeats=3,
                target_episode_policy="aggregate_support",
            )
            self._write_stage_c_run(
                input_root,
                "qwen25-w-only",
                backbone="Qwen2.5-1.5B-Instruct",
                adaptation_target="w_only",
                zero_shot_task_score=0.25,
                best_adapt_task_score=0.5,
                adaptation_effective=True,
                zero_shot_task_proxy_score=0.42,
                best_adapt_task_proxy_score=0.75,
                target_eval_repeats=3,
                target_episode_repeats=3,
                target_episode_policy="aggregate_support",
            )
            self._write_stage_c_run(
                input_root,
                "qwen25-q-only-lr1",
                backbone="Qwen2.5-1.5B-Instruct",
                adaptation_target="q_only",
                zero_shot_task_score=0.25,
                best_adapt_task_score=0.25,
                adaptation_effective=False,
                adapt_learning_rate=1.0,
                best_adapt_query_loss=1.55,
                zero_shot_task_proxy_score=0.40,
                best_adapt_task_proxy_score=0.65,
                target_eval_repeats=3,
                target_episode_repeats=3,
                target_episode_policy="aggregate_support",
            )
            self._write_stage_c_run(
                input_root,
                "qwen3-w-plus-q",
                backbone="Qwen3-8B",
                adaptation_target="w_plus_q",
                zero_shot_task_score=0.25,
                best_adapt_task_score=0.5,
                adaptation_effective=True,
                zero_shot_task_proxy_score=0.35,
                best_adapt_task_proxy_score=0.70,
                target_eval_repeats=3,
                target_episode_repeats=3,
                target_episode_policy="aggregate_support",
            )
            self._write_gradient_audit_run(
                input_root,
                "qwen25-q-only-gradient-audit",
                backbone="Qwen2.5-1.5B-Instruct",
                adaptation_target="q_only",
                query_to_writer_grad_ratio=1.6e-6,
            )

            result = analysis_main(
                [
                    "--config",
                    str(ROOT / "configs/exp/m3_stage_c_probe_summary.yaml"),
                    "--seed",
                    "3201",
                    "--output_dir",
                    str(output_root),
                    "--input_root",
                    str(input_root),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(output_root.joinpath("probe_summary.csv").exists())
            self.assertTrue(output_root.joinpath("probe_summary.svg").exists())
            metrics = json.loads(output_root.joinpath("metrics.json").read_text())
            self.assertEqual(metrics["analysis_mode"], "m3_stage_c_probe_summary")
            self.assertEqual(metrics["rows_collected"], 4)
            self.assertEqual(metrics["gradient_audits_collected"], 1)
            self.assertEqual(
                metrics["best_by_backbone"]["Qwen2.5-1.5B-Instruct"]["adaptation_target"],
                "w_only",
            )
            self.assertEqual(metrics["best_by_backbone"]["Qwen2.5-1.5B-Instruct"]["adapt_steps"], 3)
            self.assertEqual(
                metrics["best_by_backbone"]["Qwen2.5-1.5B-Instruct"]["target_eval_repeats"],
                3,
            )
            self.assertEqual(
                metrics["best_by_backbone"]["Qwen2.5-1.5B-Instruct"]["target_episode_repeats"],
                3,
            )
            self.assertEqual(
                metrics["best_by_backbone"]["Qwen2.5-1.5B-Instruct"]["target_episode_policy"],
                "aggregate_support",
            )
            self.assertTrue(metrics["seed_consistent_by_backbone"]["Qwen2.5-1.5B-Instruct"])
            self.assertEqual(metrics["q_only_by_backbone"]["Qwen2.5-1.5B-Instruct"]["run_name"], "qwen25-q-only-lr1")
            self.assertEqual(
                metrics["q_only_by_backbone"]["Qwen2.5-1.5B-Instruct"]["task_proxy_name"],
                "gold_choice_probability",
            )
            self.assertEqual(metrics["q_only_by_backbone"]["Qwen2.5-1.5B-Instruct"]["target_eval_repeats"], 3)
            self.assertEqual(metrics["q_only_by_backbone"]["Qwen2.5-1.5B-Instruct"]["target_episode_repeats"], 3)
            self.assertEqual(
                metrics["q_only_by_backbone"]["Qwen2.5-1.5B-Instruct"]["target_episode_policy"],
                "aggregate_support",
            )
            self.assertFalse(metrics["q_only_by_backbone"]["Qwen2.5-1.5B-Instruct"]["adaptation_effective"])


if __name__ == "__main__":
    unittest.main()
