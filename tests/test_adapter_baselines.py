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

from memtotal.eval.run_eval import main as eval_main
from memtotal.training.run_train import main as train_main


class AdapterBaselineTest(unittest.TestCase):
    def _write_story_cloze_config(self, tmp: Path, mode: str) -> Path:
        dataset_path = tmp / f"{mode}_story_cloze.jsonl"
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
        baseline = {
            "family": "adapter",
            "mode": mode,
            "support_examples": 1,
        }
        if mode == "prompt_tuning":
            baseline["prompt_tuning"] = {"prompt_tokens": 4}
        elif mode == "lora":
            baseline["lora"] = {"rank": 4, "alpha": 8.0}
        elif mode == "prefix_tuning":
            baseline["prefix_tuning"] = {"prefix_tokens": 4}
        else:
            baseline["ia3"] = {"init_scale": 1.0}
        config = {
            "experiment": {
                "name": f"baseline_{mode}_story_cloze_test",
                "stage": "M5",
                "method_variant": f"baseline-{mode}",
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
            "baseline": baseline,
            "runtime": {
                "train_steps": 3,
                "eval_examples": 2,
                "learning_rate": 0.05,
                "device": "cpu",
            },
        }
        config_path = tmp / f"{mode}_story_cloze.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        return config_path

    def test_prompt_tuning_adapter_train_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = self._write_story_cloze_config(tmp, "prompt_tuning")
            output_dir = tmp / "prompt_tuning_train"
            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "901",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(metrics["mode"], "train_baseline")
            self.assertEqual(metrics["baseline_family"], "adapter")
            self.assertEqual(metrics["baseline_mode"], "prompt_tuning")
            self.assertEqual(metrics["support_examples"], 1)
            self.assertEqual(metrics["train_steps"], 3)
            self.assertEqual(metrics["budget_scope"], "few_shot_adapter")
            self.assertIn("shots=1|steps=3", metrics["budget_signature"])
            self.assertGreater(metrics["trainable_parameter_count"], 0)
            self.assertTrue((output_dir / "checkpoint.pt").exists())

    def test_lora_adapter_eval_loads_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = self._write_story_cloze_config(tmp, "lora")
            train_output_dir = tmp / "lora_train"
            eval_output_dir = tmp / "lora_eval"
            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "903",
                        "--output_dir",
                        str(train_output_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "903",
                        "--output_dir",
                        str(eval_output_dir),
                        "--checkpoint",
                        str(train_output_dir / "checkpoint.pt"),
                    ]
                ),
                0,
            )
            metrics = json.loads((eval_output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (eval_output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["mode"], "eval_baseline")
            self.assertEqual(metrics["baseline_family"], "adapter")
            self.assertEqual(metrics["baseline_mode"], "lora")
            self.assertEqual(metrics["support_examples"], 1)
            self.assertEqual(metrics["train_steps"], 3)
            self.assertEqual(metrics["budget_scope"], "few_shot_adapter")
            self.assertIn("candidate_scores", predictions[0])
            self.assertEqual(predictions[0]["baseline_mode"], "lora")
            self.assertIn("Story:", predictions[0]["baseline_prompt"])

    def test_ia3_adapter_eval_loads_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = self._write_story_cloze_config(tmp, "ia3")
            train_output_dir = tmp / "ia3_train"
            eval_output_dir = tmp / "ia3_eval"
            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "905",
                        "--output_dir",
                        str(train_output_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "905",
                        "--output_dir",
                        str(eval_output_dir),
                        "--checkpoint",
                        str(train_output_dir / "checkpoint.pt"),
                    ]
                ),
                0,
            )
            metrics = json.loads((eval_output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (eval_output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["baseline_mode"], "ia3")
            self.assertEqual(metrics["support_examples"], 1)
            self.assertEqual(metrics["train_steps"], 3)
            self.assertEqual(metrics["budget_scope"], "few_shot_adapter")
            self.assertEqual(metrics["trainable_parameter_count"], 64)
            self.assertEqual(predictions[0]["baseline_mode"], "ia3")

    def test_prefix_tuning_adapter_eval_loads_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = self._write_story_cloze_config(tmp, "prefix_tuning")
            train_output_dir = tmp / "prefix_tuning_train"
            eval_output_dir = tmp / "prefix_tuning_eval"
            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "907",
                        "--output_dir",
                        str(train_output_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "907",
                        "--output_dir",
                        str(eval_output_dir),
                        "--checkpoint",
                        str(train_output_dir / "checkpoint.pt"),
                    ]
                ),
                0,
            )
            metrics = json.loads((eval_output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (eval_output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["baseline_mode"], "prefix_tuning")
            self.assertEqual(metrics["support_examples"], 1)
            self.assertEqual(metrics["train_steps"], 3)
            self.assertEqual(metrics["budget_scope"], "few_shot_adapter")
            self.assertEqual(metrics["trainable_parameter_count"], 4416)
            self.assertEqual(predictions[0]["baseline_mode"], "prefix_tuning")


if __name__ == "__main__":
    unittest.main()
