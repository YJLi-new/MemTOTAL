from __future__ import annotations

import csv
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

from memtotal.baselines.grid_runner import build_grid_plan, main as grid_main
from memtotal.training.run_train import main as train_main


class BaselineGridTest(unittest.TestCase):
    def test_build_grid_plan_prunes_adapter_positive_steps_at_zero_shot(self) -> None:
        config = {
            "grid": {
                "shots": [0, 2],
                "steps": [0, 4],
                "variants": [
                    {
                        "family": "prompting",
                        "mode": "vanilla",
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "template_config": "baseline_vanilla_story_cloze_qwen25_real_smoke.yaml",
                    },
                    {
                        "family": "adapter",
                        "mode": "prompt_tuning",
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "template_config": "baseline_prompt_tuning_story_cloze_qwen25_real_smoke.yaml",
                    },
                ],
            }
        }
        cells = build_grid_plan(config)
        self.assertEqual(len([cell for cell in cells if cell.family == "prompting"]), 2)
        self.assertEqual(len([cell for cell in cells if cell.family == "adapter"]), 3)
        self.assertFalse(any(cell.family == "adapter" and cell.shot == 0 and cell.step > 0 for cell in cells))

    def test_adapter_zero_step_zero_shot_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "story_cloze.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "story-cloze-000",
                                "story": "Nina studied all week for her exam.",
                                "choices": [
                                    {"label": "A", "text": "She passed the exam with confidence."},
                                    {"label": "B", "text": "She forgot to attend the exam."},
                                ],
                                "label": "A",
                                "answer": "She passed the exam with confidence.",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "story-cloze-001",
                                "story": "Liam watered the garden every morning.",
                                "choices": [
                                    {"label": "A", "text": "The flowers stayed healthy and bright."},
                                    {"label": "B", "text": "The plants vanished overnight."},
                                ],
                                "label": "A",
                                "answer": "The flowers stayed healthy and bright.",
                            }
                        ),
                    ]
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "baseline_prompt_tuning_zero_step_test",
                    "stage": "M5",
                    "method_variant": "baseline-prompt-tuning-zero-step",
                },
                "task": {
                    "name": "story_cloze_smoke",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "local_contract_v1",
                    "dataset_path": str(dataset_path),
                    "metric_name": "accuracy",
                    "evaluator": {"type": "multiple_choice"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "baseline": {
                    "family": "adapter",
                    "mode": "prompt_tuning",
                    "support_examples": 0,
                    "prompt_tuning": {"prompt_tokens": 4},
                },
                "runtime": {
                    "train_steps": 0,
                    "eval_examples": 2,
                    "learning_rate": 0.05,
                    "device": "cpu",
                },
            }
            config_path = tmp / "zero_step.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "train"
            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "991",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(metrics["support_examples"], 0)
            self.assertEqual(metrics["train_steps"], 0)
            self.assertIsNone(metrics["final_loss"])
            self.assertTrue((output_dir / "checkpoint.pt").exists())

    def test_grid_runner_writes_adapt_curve_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "grid"
            self.assertEqual(
                grid_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m5_story_cloze_baseline_grid_smoke.yaml"),
                        "--seed",
                        "991",
                        "--output_dir",
                        str(output_dir),
                        "--dry-run",
                    ]
                ),
                0,
            )
            self.assertTrue((output_dir / "adapt_curve.csv").is_file())
            self.assertTrue((output_dir / "adapt_cost.json").is_file())
            self.assertTrue((output_dir / "summary.csv").is_file())
            with (output_dir / "adapt_curve.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            cost = json.loads((output_dir / "adapt_cost.json").read_text())
            self.assertGreaterEqual(cost["cell_count"], 1)
            self.assertGreaterEqual(cost["eval_run_count"], 1)

    def test_grid_runner_imports_external_baseline_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imported_run = tmp / "memgen-run"
            imported_run.mkdir()
            (imported_run / "run_info.json").write_text(
                json.dumps(
                    {
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "story_cloze",
                        "smoke_subset": "hf_real_smoke4",
                    }
                )
            )
            (imported_run / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "memgen_adapter",
                        "compute_reward": 0.75,
                    }
                )
            )
            config = {
                "grid": {
                    "shots": [0],
                    "steps": [0],
                    "variants": [],
                    "imports": [
                        {
                            "family": "memgen",
                            "mode": "external_eval",
                            "backbone": "Qwen2.5-1.5B-Instruct",
                            "shot": 0,
                            "step": 0,
                            "run_dir": str(imported_run),
                        }
                    ],
                }
            }
            config_path = tmp / "grid.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "grid-output"
            self.assertEqual(
                grid_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "993",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            with (output_dir / "adapt_curve.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["baseline_family"], "memgen")
            self.assertEqual(rows[0]["primary_metric"], "compute_reward")
            cost = json.loads((output_dir / "adapt_cost.json").read_text())
            self.assertEqual(cost["imported_eval_count"], 1)


if __name__ == "__main__":
    unittest.main()
