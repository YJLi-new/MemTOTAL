from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from memtotal.analysis.m4_shared_injection import (
    run_m4_phase0_gate_sweep,
    run_m4_shared_injection_compare,
    run_m4_shared_injection_dynamics_audit,
    run_m4_writer_information_audit,
)
from memtotal.training.m4_shared_injection import (
    LatentPrefixProjector,
    _build_support_text_block,
    _choice_task_loss,
)


def _classification_rows(
    gold_labels: list[str],
    predicted_labels: list[str],
    *,
    margin: float = 0.2,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    for index, (gold_label, predicted_label) in enumerate(zip(gold_labels, predicted_labels), start=1):
        rows.append(
            {
                "example_id": str(index),
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "predicted_correct": gold_label == predicted_label,
                "task_score": float(gold_label == predicted_label),
                "final_margin": margin if gold_label == predicted_label else -margin,
                "final_choice_scores": [0.2, 0.1, -0.1],
                "candidate_labels": list(all_labels),
                "candidate_texts": ["Supports", "Refutes", "Not enough info"],
            }
        )
    return rows


class SharedInjectionHelpersTest(unittest.TestCase):
    def test_support_text_block_respects_modes_and_triad_variant(self) -> None:
        support_rows = [
            {
                "id": "s1",
                "claim": "claim-s1",
                "evidence": "evidence-s1",
                "label": "SUPPORTS",
                "shuffled_memory_example_id": "s4",
            },
            {
                "id": "s2",
                "claim": "claim-s2",
                "evidence": "evidence-s2",
                "label": "SUPPORTS",
                "shuffled_memory_example_id": "s5",
            },
            {
                "id": "s3",
                "claim": "claim-s3",
                "evidence": "evidence-s3",
                "label": "REFUTES",
                "shuffled_memory_example_id": "s6",
            },
            {
                "id": "s4",
                "claim": "claim-s4",
                "evidence": "evidence-s4",
                "label": "REFUTES",
                "shuffled_memory_example_id": "s1",
            },
            {
                "id": "s5",
                "claim": "claim-s5",
                "evidence": "No gold evidence provided in this split.",
                "label": "NOT_ENOUGH_INFO",
                "shuffled_memory_example_id": "s2",
            },
            {
                "id": "s6",
                "claim": "claim-s6",
                "evidence": "No gold evidence provided in this split.",
                "label": "NOT_ENOUGH_INFO",
                "shuffled_memory_example_id": "s3",
            },
        ]
        lookup = {row["id"]: row for row in support_rows}
        real_block = _build_support_text_block(
            support_rows,
            memory_control="real",
            example_lookup=lookup,
            support_serialization_variant="triad_curated6",
        )
        shuffled_block = _build_support_text_block(
            support_rows,
            memory_control="shuffled",
            example_lookup=lookup,
            support_serialization_variant="triad_curated6",
        )
        zero_block = _build_support_text_block(
            support_rows,
            memory_control="zero",
            example_lookup=lookup,
            support_serialization_variant="triad_curated6",
        )
        self.assertIn("Evidence: insufficient evidence available", real_block)
        self.assertIn("Answer: NOT_ENOUGH_INFO", real_block)
        self.assertIn("claim-s4", shuffled_block)
        self.assertEqual(zero_block, "")

    def test_choice_task_loss_backpropagates(self) -> None:
        scores = torch.tensor([0.2, 0.1, -0.3], dtype=torch.float32, requires_grad=True)
        loss = _choice_task_loss(scores, 0, margin_value=0.1)
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(scores.grad)

    def test_latent_prefix_projector_preserves_slot_count(self) -> None:
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
        case_rows: list[dict[str, object]],
    ) -> None:
        run_dir = root / alias
        run_dir.mkdir(parents=True)
        accuracy = sum(int(row["predicted_correct"]) for row in case_rows) / len(case_rows)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "training_stage": "shared_injection_pilot",
                    "pilot_arm_alias": alias,
                    "best_adapt_task_score": accuracy,
                    "best_adapt_task_margin": sum(float(row["final_margin"]) for row in case_rows) / len(case_rows),
                    "task_metric_name": "accuracy",
                }
            )
        )
        (run_dir / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in case_rows) + "\n"
        )

    def _write_snapshot(
        self,
        run_dir: Path,
        *,
        step: int,
        case_rows: list[dict[str, object]],
    ) -> None:
        snapshot_dir = run_dir / "snapshot_evals" / f"step_{step:04d}"
        snapshot_dir.mkdir(parents=True)
        accuracy = sum(int(row["predicted_correct"]) for row in case_rows) / len(case_rows)
        (snapshot_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "training_stage": "shared_injection_snapshot_eval",
                    "step": step,
                    "best_adapt_task_score": accuracy,
                    "best_adapt_macro_f1": accuracy,
                    "best_adapt_task_margin": sum(float(row["final_margin"]) for row in case_rows) / len(case_rows),
                }
            )
        )
        (snapshot_dir / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in case_rows) + "\n"
        )

    @mock.patch("memtotal.analysis.m4_shared_injection._build_support_text_block")
    @mock.patch("memtotal.analysis.m4_shared_injection._evaluate_examples")
    @mock.patch("memtotal.analysis.m4_shared_injection.SharedInjectionPilotRuntime")
    @mock.patch("memtotal.analysis.m4_shared_injection.load_task_dataset")
    @mock.patch("memtotal.analysis.m4_shared_injection._load_task_dataset_with_path")
    def test_phase0_gate_sweep_selects_prompt_and_support_variant(
        self,
        mock_load_support,
        mock_load_eval,
        _mock_runtime,
        mock_evaluate,
        mock_support_block,
    ) -> None:
        support_rows = [
            {"id": "s1", "label": "SUPPORTS", "shuffled_memory_example_id": "s2"},
            {"id": "s2", "label": "SUPPORTS", "shuffled_memory_example_id": "s3"},
            {"id": "s3", "label": "REFUTES", "shuffled_memory_example_id": "s4"},
            {"id": "s4", "label": "REFUTES", "shuffled_memory_example_id": "s5"},
            {"id": "s5", "label": "NOT_ENOUGH_INFO", "shuffled_memory_example_id": "s6"},
            {"id": "s6", "label": "NOT_ENOUGH_INFO", "shuffled_memory_example_id": "s1"},
        ]
        eval_rows = [{"id": str(index), "label": label} for index, label in enumerate(
            ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
            start=1,
        )]
        mock_load_support.return_value = support_rows
        mock_load_eval.return_value = eval_rows
        mock_support_block.side_effect = lambda rows, **kwargs: f"support::{kwargs['support_serialization_variant']}"

        def fake_evaluate(*, arm_alias: str, prompt_variant: str, support_serialization_variant: str, **_: object) -> list[dict[str, object]]:
            gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
            if arm_alias.startswith("A::inline_short_labels"):
                predicted = ["SUPPORTS"] * 6
            elif arm_alias.startswith("A::answer_slot_labels"):
                predicted = ["SUPPORTS"] * 6
            elif arm_alias.startswith("A::verbalized_decisions"):
                predicted = ["SUPPORTS"] * 6
            elif prompt_variant == "answer_slot_labels" and support_serialization_variant == "example_blocks_raw8":
                predicted = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
            elif prompt_variant == "answer_slot_labels" and support_serialization_variant == "triad_curated6":
                predicted = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "SUPPORTS", "SUPPORTS"]
            else:
                predicted = ["SUPPORTS", "REFUTES", "REFUTES", "SUPPORTS", "SUPPORTS", "SUPPORTS"]
            return _classification_rows(gold, predicted)

        mock_evaluate.side_effect = fake_evaluate
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "load_mode": "hf_causal_lm",
                    "dtype": "float32",
                    "model_id": "fake/model",
                },
                "method": {"writer": {"memory_slots": 2, "arch": "mlp"}},
                "runtime": {"phase0_seed": 11, "device": "cpu"},
                "task": {
                    "dataset_path": "unused",
                    "support_dataset_path": "unused-support",
                    "benchmark_id": "fever",
                    "name": "fever_real_screen248",
                },
            }
            output_dir = Path(tmpdir) / "phase0"
            run_m4_phase0_gate_sweep(config=config, output_dir=output_dir, dry_run=False)
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertTrue(metrics["phase0_gate_passed"])
            self.assertEqual(metrics["selected_prompt_variant"], "answer_slot_labels")
            self.assertEqual(metrics["selected_support_serialization"], "example_blocks_raw8")
            self.assertGreater(metrics["selected_pair"]["macro_f1_gain"], 0.05)

    @mock.patch("memtotal.analysis.m4_shared_injection._build_writer", return_value=(object(), object()))
    @mock.patch("memtotal.analysis.m4_shared_injection.load_task_dataset")
    @mock.patch("memtotal.analysis.m4_shared_injection._extract_writer_features")
    def test_writer_information_audit_uses_semantic_probe_gate(
        self,
        mock_extract,
        mock_load_dataset,
        _mock_build_writer,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            examples = []
            labels = [
                "SUPPORTS", "SUPPORTS", "REFUTES", "REFUTES",
                "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES",
                "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO",
            ]
            for index, label in enumerate(labels):
                examples.append(
                    {
                        "id": str(index),
                        "label": label,
                        "claim": f"claim-{index}",
                        "evidence": f"evidence-{index}",
                        "shuffled_memory_example_id": str((index + 1) % len(labels)),
                    }
                )
            mock_load_dataset.return_value = examples
            real = torch.tensor(
                [
                    [4.0, 0.0, 0.0], [4.1, 0.0, 0.0],
                    [0.0, 4.0, 0.0], [0.0, 4.1, 0.0],
                    [0.0, 0.0, 4.0], [0.0, 0.0, 4.1],
                    [4.2, 0.1, 0.0], [0.1, 4.2, 0.0],
                    [0.0, 0.1, 4.2], [4.3, 0.0, 0.1],
                    [0.1, 4.3, 0.0], [0.0, 0.1, 4.3],
                ],
                dtype=torch.float32,
            )
            shuffled = torch.tensor(
                [
                    [0.01, 0.00, 0.00], [0.00, 0.01, 0.00],
                    [0.00, 0.00, 0.01], [0.01, 0.01, 0.00],
                    [0.00, 0.01, 0.01], [0.01, 0.00, 0.01],
                    [0.01, 0.00, 0.00], [0.00, 0.01, 0.00],
                    [0.00, 0.00, 0.01], [0.01, 0.01, 0.00],
                    [0.00, 0.01, 0.01], [0.01, 0.00, 0.01],
                ],
                dtype=torch.float32,
            )
            zero = torch.zeros_like(real)
            mock_extract.side_effect = [real, shuffled, zero]

            phase0_dir = Path(tmpdir) / "phase0"
            phase0_dir.mkdir()
            (phase0_dir / "metrics.json").write_text(json.dumps({"phase0_gate_passed": True}))
            output_dir = Path(tmpdir) / "phase1"
            config = {
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "load_mode": "hf_causal_lm",
                    "dtype": "float32",
                    "model_id": "fake/model",
                },
                "method": {"writer": {"memory_slots": 1, "arch": "mlp"}},
                "runtime": {
                    "probe_seed": 7,
                    "audit_label_macro_f1_min": 0.40,
                    "audit_label_macro_f1_gap": 0.05,
                    "audit_binary_auroc_min": 0.60,
                    "audit_binary_auroc_gap": 0.05,
                },
                "task": {"dataset_path": "unused", "benchmark_id": "fever", "name": "fever_real_screen248"},
            }
            run_m4_writer_information_audit(
                config=config,
                output_dir=output_dir,
                input_root=str(phase0_dir),
                resume="runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b",
                dry_run=False,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertTrue(metrics["label_probe_passed"])
            self.assertTrue(metrics["semantic_probe_passed"])
            self.assertTrue(metrics["phase1_gate_passed"])

    def test_compare_gate_uses_real_vs_shuffle_zero_and_macro_f1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "runs"
            gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
            rows = {
                "A": _classification_rows(gold, ["SUPPORTS", "REFUTES", "REFUTES", "SUPPORTS", "REFUTES", "SUPPORTS"]),
                "T": _classification_rows(gold, ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "SUPPORTS"]),
                "I_real": _classification_rows(gold, ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]),
                "I_shuffle": _classification_rows(gold, ["SUPPORTS", "REFUTES", "REFUTES", "SUPPORTS", "REFUTES", "SUPPORTS"]),
                "I_zero": _classification_rows(gold, ["SUPPORTS", "REFUTES", "REFUTES", "SUPPORTS", "SUPPORTS", "SUPPORTS"]),
            }
            for alias, case_rows in rows.items():
                self._write_run(root, alias=alias, case_rows=case_rows)
            output_dir = Path(tmpdir) / "compare"
            run_m4_shared_injection_compare(
                config={"task": {"name": "fever_real_fixed64"}},
                output_dir=output_dir,
                input_root=str(root),
                dry_run=False,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertTrue(metrics["gate_passed"])

    def test_dynamics_audit_detects_overshoot_and_best_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_root = Path(tmpdir) / "raw8" / "phase2-selected"
            gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"]
            a_rows = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            t_rows = _classification_rows(gold, ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"])
            zero_rows = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            real_step0 = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            real_step16 = _classification_rows(gold, ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"])
            real_step64 = _classification_rows(gold, ["SUPPORTS", "REFUTES", "SUPPORTS", "SUPPORTS"])
            shuffle_step0 = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            shuffle_step16 = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            shuffle_step64 = _classification_rows(gold, ["SUPPORTS", "REFUTES", "SUPPORTS", "SUPPORTS"])
            rows = {
                "A": a_rows,
                "T": t_rows,
                "I_zero": zero_rows,
                "I_real": real_step64,
                "I_shuffle": shuffle_step64,
            }
            for alias, case_rows in rows.items():
                self._write_run(suite_root, alias=alias, case_rows=case_rows)
            self._write_snapshot(suite_root / "I_real", step=0, case_rows=real_step0)
            self._write_snapshot(suite_root / "I_real", step=16, case_rows=real_step16)
            self._write_snapshot(suite_root / "I_real", step=64, case_rows=real_step64)
            self._write_snapshot(suite_root / "I_shuffle", step=0, case_rows=shuffle_step0)
            self._write_snapshot(suite_root / "I_shuffle", step=16, case_rows=shuffle_step16)
            self._write_snapshot(suite_root / "I_shuffle", step=64, case_rows=shuffle_step64)
            self._write_snapshot(suite_root / "A", step=0, case_rows=a_rows)
            self._write_snapshot(suite_root / "T", step=0, case_rows=t_rows)
            self._write_snapshot(suite_root / "I_zero", step=0, case_rows=zero_rows)

            output_dir = Path(tmpdir) / "audit"
            run_m4_shared_injection_dynamics_audit(
                config={"task": {"name": "fever_real_fixed64"}},
                output_dir=output_dir,
                input_root=str(Path(tmpdir)),
                dry_run=False,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            suite_metrics = metrics["suite_best_metrics"]["raw8"]
            self.assertTrue(suite_metrics["overshoot_detected"])
            self.assertEqual(suite_metrics["best_step"], 16)


if __name__ == "__main__":
    unittest.main()
