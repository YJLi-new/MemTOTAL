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

from memtotal.analysis.m3_probe import collect_stage_b_probe_rows, probe_run_matches_config
from memtotal.analysis.run_analysis import main as analysis_main
from memtotal.utils.config import load_config


class M3StageBProbeTest(unittest.TestCase):
    def _write_stage_b_run(
        self,
        root: Path,
        run_name: str,
        *,
        backbone: str,
        mean_adaptation_gain: float,
        meta_episodes: int,
        meta_learning_rate: float,
    ) -> None:
        run_dir = root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "run_info.json").write_text(json.dumps({"backbone": backbone, "task_name": "core4_transfer_smoke"}))
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "mode": "train",
                    "training_stage": "stage_b",
                    "query_learning_mode": "meta_trained",
                    "query_objective": "continuation_retrieval",
                    "stage_b_trainable_target": "queries_plus_fuser",
                    "trainable_module": "reader.queries+fuser",
                    "retrieval_negative_count": 7,
                    "meta_episodes": meta_episodes,
                    "inner_steps": 1,
                    "inner_learning_rate": 0.01,
                    "meta_learning_rate": meta_learning_rate,
                    "mean_adaptation_gain": mean_adaptation_gain,
                    "mean_zero_shot_query_loss": 1.6,
                    "mean_adapted_query_loss": 1.6 - mean_adaptation_gain,
                    "source_eval_task_score": 0.125,
                    "source_eval_metric_name": "mean_score",
                    "source_eval_query_loss": 2.08,
                    "source_eval_query_accuracy": 0.125,
                }
            )
        )

    def test_collect_stage_b_probe_rows_keeps_variant_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_stage_b_run(
                root,
                "qwen25-meta-episodes-12",
                backbone="Qwen2.5-1.5B-Instruct",
                mean_adaptation_gain=8.0e-05,
                meta_episodes=12,
                meta_learning_rate=0.05,
            )
            rows = collect_stage_b_probe_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_name"], "qwen25-meta-episodes-12")
            self.assertEqual(rows[0]["backbone"], "Qwen2.5-1.5B-Instruct")
            self.assertEqual(rows[0]["meta_episodes"], 12)
            self.assertEqual(rows[0]["meta_learning_rate"], 0.05)
            self.assertEqual(rows[0]["query_objective"], "continuation_retrieval")

    def test_analysis_mode_writes_stage_b_probe_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "runs"
            output_root = root / "summary"
            self._write_stage_b_run(
                input_root,
                "qwen25-canonical",
                backbone="Qwen2.5-1.5B-Instruct",
                mean_adaptation_gain=1.9e-05,
                meta_episodes=6,
                meta_learning_rate=0.05,
            )
            self._write_stage_b_run(
                input_root,
                "qwen25-meta-episodes-12",
                backbone="Qwen2.5-1.5B-Instruct",
                mean_adaptation_gain=8.0e-05,
                meta_episodes=12,
                meta_learning_rate=0.05,
            )
            self._write_stage_b_run(
                input_root,
                "qwen3-canonical",
                backbone="Qwen3-8B",
                mean_adaptation_gain=7.9e-04,
                meta_episodes=6,
                meta_learning_rate=0.05,
            )

            result = analysis_main(
                [
                    "--config",
                    str(ROOT / "configs/exp/m3_stage_b_probe_summary.yaml"),
                    "--seed",
                    "2701",
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
            self.assertEqual(metrics["analysis_mode"], "m3_stage_b_probe_summary")
            self.assertEqual(metrics["rows_collected"], 3)
            self.assertEqual(
                metrics["best_by_backbone"]["Qwen2.5-1.5B-Instruct"]["run_name"],
                "qwen25-meta-episodes-12",
            )
            self.assertEqual(metrics["best_by_backbone"]["Qwen3-8B"]["run_name"], "qwen3-canonical")

    def test_probe_run_matches_config_checks_seed_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "runs"
            output_root = root / "summary"
            self._write_stage_b_run(
                input_root,
                "qwen25-canonical",
                backbone="Qwen2.5-1.5B-Instruct",
                mean_adaptation_gain=1.9e-05,
                meta_episodes=16,
                meta_learning_rate=0.05,
            )
            result = analysis_main(
                [
                    "--config",
                    str(ROOT / "configs/exp/m3_stage_b_probe_summary.yaml"),
                    "--seed",
                    "2701",
                    "--output_dir",
                    str(output_root),
                    "--input_root",
                    str(input_root),
                ]
            )
            self.assertEqual(result, 0)

            run_dir = input_root / "qwen25-canonical"
            raw_config = {
                "includes": [str(ROOT / "configs/exp/m3_stage_b_core4_qwen25_smoke.yaml")],
                "experiment": {
                    "name": "m3_stage_b_core4_qwen25_probe_canonical",
                    "method_variant": "m3-core4-stage-b-probe-qwen25-canonical",
                },
            }
            config_path = root / "probe.yaml"
            config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))
            snapshot = load_config(config_path)
            (run_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "seed": 2801,
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "core4_transfer_smoke",
                    }
                )
            )
            self.assertTrue(probe_run_matches_config(run_dir, config_path, 2801))
            self.assertFalse(probe_run_matches_config(run_dir, config_path, 2803))


if __name__ == "__main__":
    unittest.main()
