from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from memtotal.analysis.m4_shared_injection import (
    run_m4_shared_injection_compare,
    run_m4_writer_information_audit,
)
from memtotal.training.m4_shared_injection import (
    LatentPrefixProjector,
    _build_support_text_block,
    _choice_task_loss,
)


class SharedInjectionHelpersTest(unittest.TestCase):
    def test_support_text_block_respects_real_shuffled_zero_modes(self):
        support_rows = [
            {
                "id": "a",
                "claim": "claim-a",
                "evidence": "evidence-a",
                "label": "SUPPORTS",
                "shuffled_memory_example_id": "b",
            },
            {
                "id": "b",
                "claim": "claim-b",
                "evidence": "evidence-b",
                "label": "REFUTES",
                "shuffled_memory_example_id": "a",
            },
        ]
        lookup = {str(row["id"]): row for row in support_rows}
        real_block = _build_support_text_block(support_rows, memory_control="real", example_lookup=lookup)
        shuffled_block = _build_support_text_block(support_rows, memory_control="shuffled", example_lookup=lookup)
        zero_block = _build_support_text_block(support_rows, memory_control="zero", example_lookup=lookup)
        self.assertIn("evidence-a", real_block)
        self.assertIn("evidence-b", real_block)
        self.assertIn("claim-a", shuffled_block)
        self.assertIn("evidence-b", shuffled_block)
        self.assertEqual(zero_block, "")

    def test_choice_task_loss_backpropagates(self):
        scores = torch.tensor([0.2, 0.1, -0.3], dtype=torch.float32, requires_grad=True)
        loss = _choice_task_loss(scores, 0, margin_value=0.1)
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(scores.grad)

    def test_latent_prefix_projector_preserves_slot_count(self):
        projector = LatentPrefixProjector(hidden_size=6, prefix_tokens=3)
        memory_slots = torch.randn(2, 3, 6)
        projected = projector(memory_slots)
        self.assertEqual(list(projected.shape), [2, 3, 6])


class SharedInjectionAnalysisTest(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        *,
        alias: str,
        task_score: float,
        mean_margin: float,
        case_rows: list[dict[str, object]],
    ) -> Path:
        run_dir = root / alias
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "training_stage": "shared_injection_pilot",
                    "pilot_arm_alias": alias,
                    "best_adapt_task_score": task_score,
                    "best_adapt_task_margin": mean_margin,
                    "task_metric_name": "accuracy",
                }
            )
        )
        (run_dir / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in case_rows) + "\n"
        )
        return run_dir

    def test_compare_gate_uses_real_vs_shuffle_and_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "runs"
            rows = {
                "A": [
                    {"example_id": "1", "predicted_correct": False, "final_margin": -0.2},
                    {"example_id": "2", "predicted_correct": True, "final_margin": 0.3},
                    {"example_id": "3", "predicted_correct": False, "final_margin": -0.1},
                    {"example_id": "4", "predicted_correct": True, "final_margin": 0.2},
                ],
                "T": [
                    {"example_id": "1", "predicted_correct": True, "final_margin": 0.2},
                    {"example_id": "2", "predicted_correct": True, "final_margin": 0.4},
                    {"example_id": "3", "predicted_correct": False, "final_margin": -0.05},
                    {"example_id": "4", "predicted_correct": True, "final_margin": 0.25},
                ],
                "I_real": [
                    {"example_id": "1", "predicted_correct": True, "final_margin": 0.1},
                    {"example_id": "2", "predicted_correct": True, "final_margin": 0.35},
                    {"example_id": "3", "predicted_correct": True, "final_margin": 0.05},
                    {"example_id": "4", "predicted_correct": True, "final_margin": 0.21},
                ],
                "I_shuffle": [
                    {"example_id": "1", "predicted_correct": False, "final_margin": -0.1},
                    {"example_id": "2", "predicted_correct": True, "final_margin": 0.31},
                    {"example_id": "3", "predicted_correct": False, "final_margin": -0.08},
                    {"example_id": "4", "predicted_correct": True, "final_margin": 0.2},
                ],
                "I_zero": [
                    {"example_id": "1", "predicted_correct": False, "final_margin": -0.15},
                    {"example_id": "2", "predicted_correct": True, "final_margin": 0.3},
                    {"example_id": "3", "predicted_correct": False, "final_margin": -0.06},
                    {"example_id": "4", "predicted_correct": True, "final_margin": 0.19},
                ],
            }
            for alias, case_rows in rows.items():
                task_score = sum(int(row["predicted_correct"]) for row in case_rows) / len(case_rows)
                mean_margin = sum(float(row["final_margin"]) for row in case_rows) / len(case_rows)
                self._write_run(root, alias=alias, task_score=task_score, mean_margin=mean_margin, case_rows=case_rows)
            output_dir = Path(tmpdir) / "compare"
            run_m4_shared_injection_compare(
                config={"task": {"name": "fever_real_fixed64"}},
                output_dir=output_dir,
                input_root=str(root),
                dry_run=False,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertTrue(metrics["gate_passed"])

    @mock.patch("memtotal.analysis.m4_shared_injection._build_writer", return_value=(object(), object()))
    @mock.patch("memtotal.analysis.m4_shared_injection.load_task_dataset")
    @mock.patch("memtotal.analysis.m4_shared_injection._extract_writer_features")
    def test_writer_information_audit_uses_probe_gate(self, mock_extract, mock_load_dataset, _mock_build_writer):
        with tempfile.TemporaryDirectory() as tmpdir:
            examples = []
            for index in range(8):
                label = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"][index % 3]
                examples.append(
                    {
                        "id": str(index),
                        "label": label,
                        "claim": f"claim-{index}",
                        "evidence": f"evidence-{index}",
                        "shuffled_memory_example_id": str((index + 1) % 8),
                    }
                )
            mock_load_dataset.return_value = examples
            real = torch.tensor(
                [[1.0, 0.0], [1.0, 0.1], [1.0, 0.2], [0.0, 1.0], [0.0, 1.1], [0.0, 1.2], [0.5, 0.5], [0.5, 0.6]],
                dtype=torch.float32,
            )
            shuffled = torch.zeros_like(real)
            zero = torch.zeros_like(real)
            mock_extract.side_effect = [real, shuffled, zero]

            run_root = Path(tmpdir) / "runs"
            case_rows_a = []
            case_rows_t = []
            for index, example in enumerate(examples):
                base_correct = index % 2 == 0
                teacher_correct = True
                case_rows_a.append(
                    {
                        "example_id": example["id"],
                        "predicted_correct": base_correct,
                        "final_margin": -0.1 if not base_correct else 0.1,
                    }
                )
                case_rows_t.append(
                    {
                        "example_id": example["id"],
                        "predicted_correct": teacher_correct,
                        "final_margin": 0.2,
                    }
                )
            self._write_run(run_root, alias="A", task_score=0.5, mean_margin=0.0, case_rows=case_rows_a)
            self._write_run(run_root, alias="T", task_score=1.0, mean_margin=0.2, case_rows=case_rows_t)
            output_dir = Path(tmpdir) / "audit"
            config = {
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "load_mode": "hf_causal_lm",
                    "dtype": "float32",
                    "model_id": "fake/model",
                },
                "method": {"writer": {"memory_slots": 2, "arch": "mlp"}},
                "runtime": {
                    "probe_seed": 11,
                    "audit_classification_min_gap": 0.05,
                    "audit_teacher_gain_min_auc": 0.60,
                    "audit_teacher_gain_auc_gap": 0.05,
                },
                "task": {"dataset_path": "unused", "benchmark_id": "fever", "name": "fever_real_fixed64"},
            }
            run_m4_writer_information_audit(
                config=config,
                output_dir=output_dir,
                input_root=str(run_root),
                resume="runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b",
                dry_run=False,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertTrue(metrics["phase0_support_has_value"])
            self.assertTrue(metrics["phase1_gate_passed"])


if __name__ == "__main__":
    unittest.main()
