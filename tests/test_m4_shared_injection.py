from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from memtotal.analysis.m4_shared_injection import (
    compare_m4_alignment_runs,
    run_m4_phase0_gate_sweep,
    run_m4_prepare_fever_support_banks,
    run_m4_prepare_fever_validation_splits,
    summarize_m4_support_bank_run,
    compare_m4_anti_shortcut_runs,
    run_m4_shared_injection_compare,
    run_m4_shared_injection_dynamics_recovery,
    run_m4_shared_injection_dynamics_audit,
    run_m4_writer_information_audit,
)
from memtotal.models.memory import MemoryWriter
from memtotal.training.m4_shared_injection import (
    LatentPrefixProjector,
    SharedLowRankDeepPrefixProjector,
    StructuredSupportSetEncoder,
    _alignment_aux_loss,
    _build_support_text_block,
    _choice_task_loss,
    _sample_support_examples_for_training,
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

    def test_latent_prefix_projector_applies_norm_caps(self) -> None:
        projector = LatentPrefixProjector(
            hidden_size=6,
            prefix_tokens=3,
            slot_max_norm=2.0,
            total_max_norm=4.0,
        )
        memory_slots = torch.full((1, 3, 6), 50.0)
        projected = projector(memory_slots)
        slot_norms = projected.norm(dim=-1)
        self.assertTrue(bool(torch.all(slot_norms <= 2.0001)))
        self.assertLessEqual(float(projected.flatten(start_dim=1).norm(dim=1).max().item()), 4.0001)

    def test_shared_low_rank_deep_prefix_projector_outputs_selected_layers(self) -> None:
        projector = SharedLowRankDeepPrefixProjector(
            hidden_size=6,
            prefix_tokens=3,
            layer_indices=[0, 7, 14],
            bottleneck_rank=4,
            slot_max_norm=2.0,
            total_max_norm=6.0,
        )
        memory_slots = torch.full((1, 3, 6), 50.0)
        projected = projector(memory_slots)
        self.assertEqual(sorted(projected), [0, 7, 14])
        self.assertEqual(list(projected[0].shape), [1, 3, 6])
        for tensor in projected.values():
            slot_norms = tensor.norm(dim=-1)
            self.assertTrue(bool(torch.all(slot_norms <= 2.0001)))

    def test_structured_support_set_encoder_preserves_shape(self) -> None:
        encoder = StructuredSupportSetEncoder(
            hidden_size=6,
            label_count=3,
            max_items=8,
            num_heads=2,
            transformer_layers=1,
            dropout=0.0,
        )
        item_states = torch.randn(2, 6, 6)
        label_ids = torch.tensor(
            [
                [0, 0, 1, 1, 2, 2],
                [2, 1, 0, 2, 1, 0],
            ],
            dtype=torch.long,
        )
        encoded = encoder(item_states, label_ids)
        self.assertEqual(list(encoded.shape), [2, 6, 6])

    def test_transformer_writer_support_set_mode_outputs_slots(self) -> None:
        writer = MemoryWriter(
            embed_dim=6,
            memory_slots=4,
            arch="transformer",
            num_heads=2,
            transformer_layers=1,
            dropout=0.0,
        )
        support_states = torch.randn(2, 6, 6)
        memory = writer.write(support_states, input_schema="support_set")
        self.assertEqual(list(memory.shape), [2, 4, 6])

    def test_mlp_writer_rejects_support_set_mode(self) -> None:
        writer = MemoryWriter(embed_dim=6, memory_slots=4, arch="mlp")
        support_states = torch.randn(2, 6, 6)
        with self.assertRaisesRegex(ValueError, "requires arch='transformer'"):
            writer.write(support_states, input_schema="support_set")

    def test_transformer_writer_load_from_legacy_checkpoint_without_support_cross_attn(self) -> None:
        writer = MemoryWriter(
            embed_dim=6,
            memory_slots=4,
            arch="transformer",
            num_heads=2,
            transformer_layers=1,
            dropout=0.0,
        )
        legacy_state = {
            key: value
            for key, value in writer.state_dict().items()
            if not key.startswith("support_cross_attn.")
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy_writer.pt"
            torch.save(legacy_state, checkpoint_path)
            writer.load_from(checkpoint_path, strict=False)

    def test_alignment_aux_loss_teacher_margin_is_dormant_when_teacher_not_better(self) -> None:
        active = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32)
        base = torch.tensor([0.4, 0.2, 0.1], dtype=torch.float32)
        teacher = torch.tensor([0.35, 0.2, 0.1], dtype=torch.float32)
        aux_loss, active_flag = _alignment_aux_loss(
            mode="teacher_margin",
            active_scores=active,
            base_scores=base,
            teacher_scores=teacher,
            gold_index=0,
        )
        self.assertIsNone(aux_loss)
        self.assertFalse(active_flag)

    def test_alignment_aux_loss_teacher_margin_activates_when_teacher_improves_margin(self) -> None:
        active = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32)
        base = torch.tensor([0.2, 0.25, 0.1], dtype=torch.float32)
        teacher = torch.tensor([0.6, 0.1, 0.0], dtype=torch.float32)
        aux_loss, active_flag = _alignment_aux_loss(
            mode="teacher_margin",
            active_scores=active,
            base_scores=base,
            teacher_scores=teacher,
            gold_index=0,
        )
        self.assertIsNotNone(aux_loss)
        self.assertTrue(active_flag)
        self.assertGreater(float(aux_loss.item()), 0.0)


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
        checkpoint_path: str = "",
        prefix_l2: float = 1.0,
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
                    "checkpoint_path": checkpoint_path,
                    "prefix_tokens": 2,
                    "prefix_l2": prefix_l2,
                    "prefix_slot_norm_mean": prefix_l2 / 2.0,
                    "prefix_slot_norm_std": 0.1,
                    "prefix_slot_norm_max": prefix_l2 / 1.5,
                }
            )
        )
        (snapshot_dir / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in case_rows) + "\n"
        )

    @mock.patch("memtotal.analysis.m4_shared_injection._load_task_dataset_with_path")
    def test_prepare_fever_validation_splits_is_stable_and_balanced(self, mock_load_dataset) -> None:
        rows = []
        for label in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
            for index in range(16):
                rows.append({"id": f"{label}-{index:02d}", "label": label})
        mock_load_dataset.return_value = rows
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "runtime": {"split_output_root": f"{tmpdir}/splits"},
                "task": {"dataset_path": "unused"},
            }
            output_dir = Path(tmpdir) / "analysis"
            run_m4_prepare_fever_validation_splits(config=config, output_dir=output_dir, dry_run=False)
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(metrics["train_label_distribution"], {"SUPPORTS": 12, "REFUTES": 12, "NOT_ENOUGH_INFO": 12})
            self.assertEqual(metrics["val_label_distribution"], {"SUPPORTS": 4, "REFUTES": 4, "NOT_ENOUGH_INFO": 4})
            self.assertEqual(metrics["audit_label_distribution"], {"SUPPORTS": 4, "REFUTES": 4, "NOT_ENOUGH_INFO": 4})
            audit_rows = [
                json.loads(line)
                for line in Path(metrics["audit_dataset_path"]).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(audit_rows), 12)

    @mock.patch("memtotal.analysis.m4_shared_injection._load_task_dataset_with_path")
    def test_prepare_fever_support_banks_builds_unique_triads(self, mock_load_dataset) -> None:
        rows = []
        for label in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
            for index in range(12):
                rows.append(
                    {
                        "id": f"{label}-{index:02d}",
                        "label": label,
                        "claim": f"claim-{label}-{index}",
                        "evidence": f"evidence-{label}-{index}",
                        "shuffled_memory_example_id": f"{label}-{(index + 1) % 12:02d}",
                    }
                )
        mock_load_dataset.return_value = rows
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "runtime": {
                    "support_bank_output_root": f"{tmpdir}/fever",
                    "support_bank_episode_count": 4,
                },
                "task": {
                    "dataset_path": "unused-dataset",
                    "train_dataset_path": "unused-train",
                },
            }
            output_dir = Path(tmpdir) / "analysis"
            run_m4_prepare_fever_support_banks(config=config, output_dir=output_dir, dry_run=False)
            metrics = json.loads((output_dir / "metrics.json").read_text())
            manifest = json.loads(Path(metrics["manifest_path"]).read_text())
            self.assertEqual(manifest["train_episode_count"], 4)
            bank_paths = [
                manifest["screen_val_canonical_bank_path"],
                manifest["screen248_test_canonical_bank_path"],
                manifest["screen248_test_heldout_a_bank_path"],
                manifest["screen248_test_heldout_b_bank_path"],
            ]
            signatures = []
            for path in bank_paths:
                rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
                self.assertEqual(len(rows), 6)
                signatures.append(tuple(sorted(str(row["id"]) for row in rows)))
            self.assertEqual(len(signatures), len(set(signatures)))
            episode_bank = json.loads(Path(manifest["train_episode_bank_path"]).read_text())
            self.assertEqual(len(episode_bank["episodes"]), 4)
            for episode in episode_bank["episodes"]:
                self.assertEqual(len(episode["support_rows"]), 6)
                label_counts = {}
                for row in episode["support_rows"]:
                    label_counts[str(row["label"])] = label_counts.get(str(row["label"]), 0) + 1
                self.assertEqual(label_counts, {"SUPPORTS": 2, "REFUTES": 2, "NOT_ENOUGH_INFO": 2})

    def test_support_masking_uniformly_samples_target_count(self) -> None:
        support_rows = [
            {"id": "s1", "label": "SUPPORTS", "shuffled_memory_example_id": "s2"},
            {"id": "s2", "label": "SUPPORTS", "shuffled_memory_example_id": "s3"},
            {"id": "r1", "label": "REFUTES", "shuffled_memory_example_id": "r2"},
            {"id": "r2", "label": "REFUTES", "shuffled_memory_example_id": "n1"},
            {"id": "n1", "label": "NOT_ENOUGH_INFO", "shuffled_memory_example_id": "n2"},
            {"id": "n2", "label": "NOT_ENOUGH_INFO", "shuffled_memory_example_id": "s1"},
        ]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(11)
        sampled = _sample_support_examples_for_training(
            support_examples=support_rows,
            support_serialization_variant="triad_curated6",
            generator=generator,
        )
        self.assertEqual(len(sampled), 4)
        self.assertEqual(len({row["id"] for row in sampled}), 4)
        self.assertTrue(all(row["id"] in {"s1", "s2", "r1", "r2", "n1", "n2"} for row in sampled))

    def test_support_masking_honors_explicit_target_count(self) -> None:
        support_rows = [
            {"id": "s1", "label": "SUPPORTS", "shuffled_memory_example_id": "s2"},
            {"id": "s2", "label": "SUPPORTS", "shuffled_memory_example_id": "s3"},
            {"id": "r1", "label": "REFUTES", "shuffled_memory_example_id": "r2"},
            {"id": "r2", "label": "REFUTES", "shuffled_memory_example_id": "n1"},
            {"id": "n1", "label": "NOT_ENOUGH_INFO", "shuffled_memory_example_id": "n2"},
            {"id": "n2", "label": "NOT_ENOUGH_INFO", "shuffled_memory_example_id": "s1"},
        ]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        sampled = _sample_support_examples_for_training(
            support_examples=support_rows,
            support_serialization_variant="triad_curated6",
            generator=generator,
            target_count=5,
        )
        self.assertEqual(len(sampled), 5)
        self.assertEqual(len({row["id"] for row in sampled}), 5)

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

    def test_dynamics_recovery_selects_earliest_passing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "runs"
            suite_root = root / "triad6" / "phase2-selected"
            gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
            a_rows = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            t_rows = _classification_rows(gold, ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
            zero_rows = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            real_step0 = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            real_step16 = _classification_rows(gold, ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "SUPPORTS"])
            real_step32 = _classification_rows(gold, ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
            shuffle_step0 = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            shuffle_step16 = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "REFUTES", "SUPPORTS"])
            shuffle_step32 = _classification_rows(gold, ["SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS", "SUPPORTS"])
            rows = {
                "A": a_rows,
                "T": t_rows,
                "I_zero": zero_rows,
                "I_real": real_step32,
                "I_shuffle": shuffle_step32,
            }
            for alias, case_rows in rows.items():
                self._write_run(suite_root, alias=alias, case_rows=case_rows)
            self._write_snapshot(suite_root / "A", step=0, case_rows=a_rows)
            self._write_snapshot(suite_root / "T", step=0, case_rows=t_rows)
            self._write_snapshot(suite_root / "I_zero", step=0, case_rows=zero_rows)
            self._write_snapshot(suite_root / "I_real", step=0, case_rows=real_step0, checkpoint_path="/tmp/i_real_0.pt", prefix_l2=1.0)
            self._write_snapshot(suite_root / "I_real", step=16, case_rows=real_step16, checkpoint_path="/tmp/i_real_16.pt", prefix_l2=1.5)
            self._write_snapshot(suite_root / "I_real", step=32, case_rows=real_step32, checkpoint_path="/tmp/i_real_32.pt", prefix_l2=2.0)
            self._write_snapshot(suite_root / "I_shuffle", step=0, case_rows=shuffle_step0, checkpoint_path="/tmp/i_shuffle_0.pt", prefix_l2=1.0)
            self._write_snapshot(suite_root / "I_shuffle", step=16, case_rows=shuffle_step16, checkpoint_path="/tmp/i_shuffle_16.pt", prefix_l2=1.2)
            self._write_snapshot(suite_root / "I_shuffle", step=32, case_rows=shuffle_step32, checkpoint_path="/tmp/i_shuffle_32.pt", prefix_l2=1.8)

            audit_rows = []
            for index in range(4):
                audit_rows.append({"id": f"s-{index}", "label": "SUPPORTS", "claim": "c", "evidence": "e", "shuffled_memory_example_id": f"s-{index}"})
                audit_rows.append({"id": f"r-{index}", "label": "REFUTES", "claim": "c", "evidence": "e", "shuffled_memory_example_id": f"r-{index}"})
                audit_rows.append({"id": f"n-{index}", "label": "NOT_ENOUGH_INFO", "claim": "c", "evidence": "e", "shuffled_memory_example_id": f"n-{index}"})
            audit_path = Path(tmpdir) / "audit12.jsonl"
            audit_path.write_text("\n".join(json.dumps(row) for row in audit_rows) + "\n")

            output_dir = Path(tmpdir) / "recovery"
            run_m4_shared_injection_dynamics_recovery(
                config={
                    "task": {"support_dataset_path": "unused"},
                    "runtime": {"pilot_val_audit_dataset_path": ""},
                    "backbone": {
                        "name": "Qwen2.5-1.5B-Instruct",
                        "load_mode": "stub",
                        "stub_hidden_size": 8,
                    },
                    "method": {"writer": {"memory_slots": 2, "arch": "mlp"}},
                },
                output_dir=output_dir,
                input_root=str(root),
                dry_run=False,
            )
            selection = json.loads((output_dir / "selection.json").read_text())
            self.assertTrue(selection["selection_passed"])
            self.assertEqual(selection["selected_suite"], "triad6")
            self.assertEqual(selection["selected_step"], 16)
            self.assertEqual(selection["i_real_checkpoint_path"], "/tmp/i_real_16.pt")

    def test_summarize_m4_support_bank_run_marks_brittle_when_both_heldouts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "selection_passed": True,
                        "selected_suite": "run-a",
                        "selected_step": 16,
                        "selected_support_serialization": "triad_curated6",
                        "selected_prompt_variant": "answer_slot_labels",
                    }
                )
            )
            run_metrics_path = root / "run_metrics.json"
            run_metrics_path.write_text(json.dumps({"pilot_prefix_total_max_norm": 192.0}))
            summary_csv = root / "summary.csv"
            summary_csv.write_text(
                "alias,dominant_label_fraction,step\n"
                "I_real,0.50,0\n"
                "I_real,1.00,16\n"
            )
            prefix_csv = root / "prefix.csv"
            prefix_csv.write_text(
                "arm_alias,prefix_l2,row_type,step\n"
                "I_real,10.0,snapshot_aggregate,0\n"
                "I_real,190.0,snapshot_aggregate,16\n"
            )
            canonical_metrics = root / "canonical.json"
            canonical_metrics.write_text(json.dumps({"gate_passed": True}))
            heldout_a = root / "heldout_a.json"
            heldout_a.write_text(
                json.dumps(
                    {
                        "gate_passed": False,
                        "regressions_vs_base": 8,
                        "flip_gain_vs_shuffle": -1,
                        "flip_gain_vs_zero": -1,
                        "i_real_task_score": 0.2,
                        "a_task_score": 0.4,
                    }
                )
            )
            heldout_b = root / "heldout_b.json"
            heldout_b.write_text(
                json.dumps(
                    {
                        "gate_passed": False,
                        "regressions_vs_base": 7,
                        "flip_gain_vs_shuffle": -2,
                        "flip_gain_vs_zero": 0,
                        "i_real_task_score": 0.3,
                        "a_task_score": 0.5,
                    }
                )
            )
            summary = summarize_m4_support_bank_run(
                selection_json=str(selection_path),
                run_metrics_json=str(run_metrics_path),
                dynamics_summary_csv=str(summary_csv),
                prefix_norm_csv=str(prefix_csv),
                screen248_test_metrics_json=str(canonical_metrics),
                heldout_metrics_by_name={
                    "heldout_a": str(heldout_a),
                    "heldout_b": str(heldout_b),
                },
            )
            self.assertTrue(summary["screen248_test_gate_passed"])
            self.assertTrue(summary["support_bank_brittle"])
            self.assertEqual(summary["heldout_sane_bank_count"], 0)
            self.assertEqual(summary["cap_saturation_onset_step"], 16)
            self.assertEqual(summary["dominant_label_collapse_onset_step"], 16)
            self.assertFalse(summary["milestone_gate_passed"])

    def test_compare_m4_anti_shortcut_runs_prefers_run_a_when_only_run_a_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = root / "run_a.json"
            run_b = root / "run_b.json"
            run_a.write_text(
                json.dumps(
                    {
                        "selection_passed": True,
                        "screen248_test_gate_passed": True,
                        "support_bank_brittle": False,
                        "fixed64_gate_passed": False,
                        "selected_step": 16,
                        "cap_saturation_onset_step": None,
                        "dominant_label_collapse_onset_step": None,
                        "milestone_gate_passed": True,
                    }
                )
            )
            run_b.write_text(
                json.dumps(
                    {
                        "selection_passed": True,
                        "screen248_test_gate_passed": False,
                        "support_bank_brittle": False,
                        "fixed64_gate_passed": False,
                        "selected_step": 32,
                        "cap_saturation_onset_step": 16,
                        "dominant_label_collapse_onset_step": 16,
                        "milestone_gate_passed": False,
                    }
                )
            )
            summary = compare_m4_anti_shortcut_runs(
                run_a_summary_json=str(run_a),
                run_b_summary_json=str(run_b),
            )
            self.assertEqual(summary["comparison_conclusion"], "run_a_passes_run_b_fails")
            self.assertTrue(summary["run_a_primary_gate_passed"])
            self.assertFalse(summary["run_b_primary_gate_passed"])

    def test_compare_m4_alignment_runs_requires_canonical_to_beat_both_ablations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = root / "canonical.json"
            freeze_writer = root / "freeze_writer.json"
            pooled_block = root / "pooled_block.json"
            canonical.write_text(
                json.dumps(
                    {
                        "selection_passed": True,
                        "selected_step": 16,
                        "screen248_test_gate_passed": True,
                        "support_bank_brittle": False,
                        "fixed64_report_generated": True,
                        "fixed64_gate_passed": False,
                    }
                )
            )
            freeze_writer.write_text(
                json.dumps(
                    {
                        "selection_passed": True,
                        "selected_step": 16,
                        "screen248_test_gate_passed": False,
                    }
                )
            )
            pooled_block.write_text(
                json.dumps(
                    {
                        "selection_passed": False,
                        "selected_step": None,
                        "screen248_test_gate_passed": False,
                    }
                )
            )
            summary = compare_m4_alignment_runs(
                canonical_summary_json=str(canonical),
                freeze_writer_summary_json=str(freeze_writer),
                pooled_block_summary_json=str(pooled_block),
            )
            self.assertTrue(summary["alignment_claim_supported"])
            self.assertEqual(summary["comparison_conclusion"], "canonical_passes_both_ablations_fail")


if __name__ == "__main__":
    unittest.main()
