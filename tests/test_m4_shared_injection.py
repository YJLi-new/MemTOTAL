from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from memtotal.analysis.m4_shared_injection import (
    compare_m4_alignment_runs,
    compare_m5_alignment_runs,
    compare_m5_dense_teacher_runs,
    compare_m5_objective_runs,
    compare_tl_micro_lora_runs,
    compare_tl_writer_value_runs,
    compare_tl_reader_geometry_runs,
    compare_tl_reader_rg2_runs,
    compare_tl_reader_rg3_runs,
    compare_tl_bridge_rescue_runs,
    compare_tl_slot_basis_runs,
    compare_tl_poc_runs,
    run_m4_phase0_gate_sweep,
    run_m4_prepare_fever_support_banks,
    run_m4_prepare_fever_validation_splits,
    summarize_m4_support_bank_run,
    compare_m4_anti_shortcut_runs,
    run_m4_shared_injection_compare,
    run_m4_shared_injection_dynamics_recovery,
    run_m4_shared_injection_dynamics_audit,
    run_m4_writer_information_audit,
    _v0_forensics_summary,
)
from memtotal.models.backbone import MicroLoRALinear
from memtotal.models.memory import (
    MemoryFuser,
    MemoryWriter,
    WriterAuxProjectionHead,
    WriterWeaverHead,
)
from memtotal.training.m4_shared_injection import (
    LatentPrefixProjector,
    PrefixInjectionArtifacts,
    SharedLowRankDeepPrefixProjector,
    SharedInjectionPilotRuntime,
    StructuredSupportSetEncoder,
    _active_competitor_hinge_weight,
    _alignment_aux_loss,
    _alignment_aux_choice_loss,
    _build_support_text_block,
    _candidate_payload,
    _class_entropy,
    _clip_parameter_group,
    _choice_task_loss,
    _classification_metrics_from_rows,
    _collect_gradient_probe,
    _contrastive_aux_loss,
    _conditioned_query_orthogonality_loss,
    _drop_prompt_tokens_for_aux_view,
    _drop_support_rows_for_aux_view,
    _effective_rank,
    _latent_anchor_loss,
    _median_train_event_metric,
    _prefix_stats,
    _reader_attention_diversity_loss,
    _reader_fuser_bootstrap_active,
    _reader_short_reconstruction_loss,
    _scheduled_linear_decay_weight,
    _sample_support_examples_for_training,
    _slot_diversity_loss,
    _task_training_loss,
    _teacher_advantage_weight,
    _vicreg_aux_loss,
    _writer_common_mode_penalty,
    _writer_support_coverage_loss,
    _writer_slot_energy_balance_loss,
    _writer_slot_basis_orthogonality_loss,
    run_shared_injection_pilot,
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
    def test_drop_support_rows_for_aux_view_keeps_at_least_one_row(self) -> None:
        generator = torch.Generator().manual_seed(7)
        rows = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        dropped = _drop_support_rows_for_aux_view(
            rows,
            dropout_probability=1.0,
            generator=generator,
        )
        self.assertEqual(len(dropped), 1)
        self.assertIn(dropped[0]["id"], {"1", "2", "3"})

    def test_drop_prompt_tokens_for_aux_view_keeps_at_least_one_token(self) -> None:
        generator = torch.Generator().manual_seed(11)
        dropped = _drop_prompt_tokens_for_aux_view(
            "alpha beta gamma",
            dropout_probability=1.0,
            generator=generator,
        )
        self.assertGreaterEqual(len(dropped.split()), 1)

    def test_contrastive_aux_loss_uses_positive_and_negative_pairs(self) -> None:
        anchor = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        positive = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
        negatives = [torch.tensor([0.0, 1.0], dtype=torch.float32)]
        loss, diagnostics = _contrastive_aux_loss(
            anchor=anchor,
            positive=positive,
            negative_queue=negatives,
            temperature=0.1,
        )
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(diagnostics["contrastive_positive_cosine"], 0.9)
        self.assertLess(diagnostics["contrastive_negative_cosine"], 0.5)

    def test_vicreg_aux_loss_returns_finite_components(self) -> None:
        view_a = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        view_b = torch.tensor(
            [[[0.9, 0.1], [0.1, 0.9]]],
            dtype=torch.float32,
        )
        loss, diagnostics = _vicreg_aux_loss(
            view_a=view_a,
            view_b=view_b,
            invariance_weight=1.0,
            variance_weight=1.0,
            covariance_weight=1.0,
            variance_target=1.0,
        )
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(diagnostics["vicreg_invariance_loss"], 0.0)
        self.assertGreaterEqual(diagnostics["vicreg_variance_loss"], 0.0)
        self.assertGreaterEqual(diagnostics["vicreg_covariance_loss"], 0.0)

    def test_writer_support_coverage_loss_uses_entropy_stat(self) -> None:
        support_attention = torch.tensor(
            [[[[0.97, 0.01, 0.01, 0.01], [0.97, 0.01, 0.01, 0.01]]]],
            dtype=torch.float32,
        )
        prefix_artifacts = PrefixInjectionArtifacts(
            prefix_embeddings=None,
            layer_prefix_hidden_by_layer=None,
            prefix_stats={},
            writer_diagnostics={"support_attention_weights_by_layer": {"0": support_attention}},
            memory_slots=torch.ones(1, 2, 3),
        )
        loss, diagnostics = _writer_support_coverage_loss(prefix_artifacts)
        self.assertIsNotNone(loss)
        expected_entropy = float(
            (
                -(
                    torch.tensor([0.97, 0.01, 0.01, 0.01], dtype=torch.float32)
                    * torch.tensor([0.97, 0.01, 0.01, 0.01], dtype=torch.float32).log()
                ).sum()
                / math.log(4.0)
            ).item()
        )
        self.assertAlmostEqual(
            diagnostics["writer_support_coverage_entropy_mean"],
            expected_entropy,
            places=6,
        )
        self.assertAlmostEqual(float(loss.item()), 1.0 - expected_entropy, places=6)

    def test_writer_weaver_head_supports_stimulus_modes(self) -> None:
        writer = WriterWeaverHead(
            embed_dim=6,
            memory_slots=4,
            hidden_dim=12,
            num_heads=2,
            transformer_layers=1,
        )
        context_states = torch.randn(1, 3, 6)
        support_states = torch.randn(1, 2, 6)
        support_only = writer.write(
            context_states=None,
            support_states=support_states,
            stimulus_mode="support_only",
        )
        context_only = writer.write(
            context_states=context_states,
            support_states=None,
            stimulus_mode="context_only",
        )
        support_and_context = writer.write(
            context_states=context_states,
            support_states=support_states,
            stimulus_mode="support_and_context",
        )
        self.assertEqual(list(support_only.shape), [1, 4, 6])
        self.assertEqual(list(context_only.shape), [1, 4, 6])
        self.assertEqual(list(support_and_context.shape), [1, 4, 6])
        with self.assertRaisesRegex(ValueError, "requires non-empty context_states"):
            writer.write(
                context_states=None,
                support_states=support_states,
                stimulus_mode="support_and_context",
            )

    def test_writer_weaver_head_supports_stacked_conditioning_layers(self) -> None:
        writer = WriterWeaverHead(
            embed_dim=6,
            memory_slots=4,
            hidden_dim=12,
            num_heads=2,
            transformer_layers=2,
            conditioning_layers=3,
        )
        context_states = torch.randn(1, 3, 6)
        support_states = torch.randn(1, 2, 6)
        outputs = writer.write(
            context_states=context_states,
            support_states=support_states,
            stimulus_mode="support_and_context",
        )
        self.assertEqual(writer.conditioning_layers, 3)
        self.assertEqual(len(writer.extra_conditioning_blocks), 2)
        self.assertEqual(list(outputs.shape), [1, 4, 6])

    def test_writer_weaver_head_can_return_support_attention_diagnostics(self) -> None:
        writer = WriterWeaverHead(
            embed_dim=6,
            memory_slots=4,
            hidden_dim=12,
            num_heads=2,
            transformer_layers=1,
            conditioning_layers=2,
        )
        encoded, diagnostics = writer.write(
            context_states=torch.randn(1, 3, 6),
            support_states=torch.randn(1, 5, 6),
            stimulus_mode="support_and_context",
            return_diagnostics=True,
        )
        self.assertEqual(list(encoded.shape), [1, 4, 6])
        self.assertIn("support_attention_weights_by_layer", diagnostics)
        support_attention = diagnostics["support_attention_weights_by_layer"]
        self.assertIn("0", support_attention)
        self.assertEqual(list(support_attention["0"].shape), [1, 2, 4, 5])

    def test_writer_aux_projection_head_normalizes_outputs(self) -> None:
        head = WriterAuxProjectionHead(input_dim=6, projection_dim=4, hidden_dim=8)
        outputs = head(torch.randn(3, 6))
        norms = outputs.norm(dim=-1)
        self.assertEqual(list(outputs.shape), [3, 4])
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_writer_weaver_head_supports_context_support_balance_gate(self) -> None:
        writer = WriterWeaverHead(
            embed_dim=6,
            memory_slots=4,
            hidden_dim=12,
            num_heads=2,
            transformer_layers=1,
            balance_mode="layernorm_learned_scalar",
            context_balance_scale_init=0.75,
            support_balance_scale_init=1.25,
        )
        encoded, diagnostics = writer.write(
            context_states=torch.randn(1, 3, 6),
            support_states=torch.randn(1, 2, 6),
            stimulus_mode="support_and_context",
            return_diagnostics=True,
        )
        self.assertEqual(list(encoded.shape), [1, 4, 6])
        self.assertAlmostEqual(diagnostics["context_balance_scale"], 0.75, places=5)
        self.assertAlmostEqual(diagnostics["support_balance_scale"], 1.25, places=5)
        self.assertEqual(diagnostics["balance_mode"], "layernorm_learned_scalar")

    def test_writer_weaver_head_supports_writer_micro_lora(self) -> None:
        writer = WriterWeaverHead(
            embed_dim=6,
            memory_slots=4,
            hidden_dim=12,
            num_heads=2,
            transformer_layers=2,
            conditioning_layers=3,
        )
        writer.enable_writer_micro_lora(
            target_modules=("conditioning_out_proj", "encoder_self_attn_out_proj"),
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        outputs = writer.write(
            context_states=torch.randn(1, 3, 6),
            support_states=torch.randn(1, 2, 6),
            stimulus_mode="support_and_context",
        )
        metadata = writer.writer_lora_metadata()
        self.assertEqual(list(outputs.shape), [1, 4, 6])
        self.assertTrue(writer.writer_lora_enabled())
        self.assertEqual(
            metadata["target_modules"],
            ["conditioning_out_proj", "encoder_self_attn_out_proj"],
        )
        self.assertGreater(writer.writer_lora_parameter_count(), 0)
        self.assertIsInstance(writer.context_cross_attn.out_proj, MicroLoRALinear)
        self.assertIsInstance(writer.support_cross_attn.out_proj, MicroLoRALinear)
        self.assertIsInstance(writer.extra_conditioning_blocks[0].context_cross_attn.out_proj, MicroLoRALinear)
        self.assertIsInstance(writer.encoder.layers[0].self_attn.out_proj, MicroLoRALinear)
        writer.set_base_trainable(False)
        writer.set_writer_lora_trainable(True)
        self.assertFalse(writer.slot_embeddings.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in writer.writer_lora_parameters()))

    def test_collect_gradient_probe_separates_task_and_aux_terms(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
        task_loss = (parameter ** 2).sum()
        aux_loss = (-parameter).sum()
        total_loss = task_loss + aux_loss
        probe = _collect_gradient_probe(
            enabled=True,
            module_parameters={"writer": [parameter]},
            task_loss=task_loss,
            aux_loss=aux_loss,
            total_loss=total_loss,
        )
        self.assertGreater(probe["grad_probe_writer_task_only_norm"], 0.0)
        self.assertGreater(probe["grad_probe_writer_aux_only_norm"], 0.0)
        self.assertGreater(probe["grad_probe_writer_total_norm"], 0.0)
        self.assertLessEqual(abs(probe["grad_probe_writer_task_aux_cosine"]), 1.0)
        self.assertLessEqual(abs(probe["grad_probe_writer_task_total_cosine"]), 1.0)

    def test_clip_parameter_group_reports_when_threshold_is_exceeded(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float32))
        parameter.grad = torch.tensor([3.0, 4.0], dtype=torch.float32)
        total_norm, was_clipped = _clip_parameter_group([parameter], max_norm=1.0)
        self.assertGreater(total_norm, 1.0)
        self.assertTrue(was_clipped)

    def test_median_train_event_metric_can_filter_to_probe_steps(self) -> None:
        events = [
            {"step": 11, "gradient_probe_step_active": False, "grad_probe_writer_task_only_norm": 0.0},
            {"step": 15, "gradient_probe_step_active": True, "grad_probe_writer_task_only_norm": 3.0},
            {"step": 20, "gradient_probe_step_active": False, "grad_probe_writer_task_only_norm": 0.0},
            {"step": 25, "gradient_probe_step_active": True, "grad_probe_writer_task_only_norm": 5.0},
        ]
        self.assertEqual(
            _median_train_event_metric(
                events,
                key="grad_probe_writer_task_only_norm",
                step_start=11,
                step_end=25,
            ),
            0.0,
        )
        self.assertEqual(
            _median_train_event_metric(
                events,
                key="grad_probe_writer_task_only_norm",
                step_start=11,
                step_end=25,
                active_only_key="gradient_probe_step_active",
            ),
            3.0,
        )

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

    def test_support_text_block_serializes_non_fever_examples(self) -> None:
        support_rows = [
            {
                "id": "gsm8k-1",
                "benchmark_id": "gsm8k",
                "segment": "Question: 1+1?",
                "continuation": "2",
                "gold_answer": "2",
                "label": "2",
            },
            {
                "id": "gsm8k-2",
                "benchmark_id": "gsm8k",
                "segment": "Question: 2+2?",
                "continuation": "4",
                "gold_answer": "4",
                "label": "4",
            },
        ]
        lookup = {row["id"]: row for row in support_rows}
        block = _build_support_text_block(
            support_rows,
            memory_control="real",
            example_lookup=lookup,
            support_serialization_variant="example_blocks_raw8",
        )
        self.assertIn("Prompt: Question: 1+1?", block)
        self.assertIn("Answer: 4", block)

    def test_candidate_payload_supports_task_native_generation_examples(self) -> None:
        example = {
            "id": "gsm8k-1",
            "evaluator_type": "exact_match",
            "segment": "Question: 1+1?",
            "continuation": "2",
            "gold_answer": "2",
            "label": "2",
        }
        labels, texts, gold_index, task_mode = _candidate_payload(
            example,
            prompt_variant="task_native",
        )
        self.assertEqual(labels, ["2"])
        self.assertEqual(texts, ["2"])
        self.assertEqual(gold_index, 0)
        self.assertEqual(task_mode, "generation")

    def test_choice_task_loss_backpropagates(self) -> None:
        scores = torch.tensor([0.2, 0.1, -0.3], dtype=torch.float32, requires_grad=True)
        loss, ce_loss, hinge_loss = _choice_task_loss(
            scores,
            0,
            margin_value=0.1,
            ce_weight=1.0,
            competitor_hinge_weight=0.2,
        )
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(ce_loss.item()), 0.0)
        self.assertGreaterEqual(float(hinge_loss.item()), 0.0)
        self.assertIsNotNone(scores.grad)

    def test_task_training_loss_uses_gold_score_for_generation(self) -> None:
        scores = torch.tensor([1.25], dtype=torch.float32, requires_grad=True)
        example_cache = mock.Mock(task_mode="generation", gold_index=0)
        loss, ce_loss, hinge_loss = _task_training_loss(
            scores,
            example_cache,
            margin_value=0.1,
            ce_weight=1.0,
            competitor_hinge_weight=0.2,
        )
        loss.backward()
        self.assertAlmostEqual(float(loss.item()), -1.25, places=6)
        self.assertAlmostEqual(float(ce_loss.item()), 0.0, places=6)
        self.assertAlmostEqual(float(hinge_loss.item()), 0.0, places=6)
        self.assertIsNotNone(scores.grad)

    def test_classification_metrics_handle_generation_rows(self) -> None:
        metrics = _classification_metrics_from_rows(
            [
                {
                    "evaluator_type": "qa_f1",
                    "predicted_text": "red apple",
                    "normalized_prediction": "red apple",
                    "predicted_correct": False,
                    "task_score": 0.5,
                    "final_margin": 1.0,
                },
                {
                    "evaluator_type": "qa_f1",
                    "predicted_text": "red apple",
                    "normalized_prediction": "red apple",
                    "predicted_correct": True,
                    "task_score": 1.0,
                    "final_margin": 2.0,
                },
            ]
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=6)
        self.assertAlmostEqual(metrics["macro_f1"], 0.75, places=6)
        self.assertAlmostEqual(metrics["exact_match"], 0.5, places=6)
        self.assertAlmostEqual(metrics["dominant_label_fraction"], 1.0, places=6)

    def test_active_competitor_hinge_weight_delays_and_ramps(self) -> None:
        self.assertEqual(
            _active_competitor_hinge_weight(
                current_step=8,
                max_weight=0.2,
                start_step=8,
                ramp_steps=24,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            _active_competitor_hinge_weight(
                current_step=20,
                max_weight=0.2,
                start_step=8,
                ramp_steps=24,
            ),
            0.1,
            places=6,
        )
        self.assertAlmostEqual(
            _active_competitor_hinge_weight(
                current_step=40,
                max_weight=0.2,
                start_step=8,
                ramp_steps=24,
            ),
            0.2,
            places=6,
        )

    def test_scheduled_linear_decay_weight_decays_from_start_to_end(self) -> None:
        self.assertAlmostEqual(
            _scheduled_linear_decay_weight(
                current_step=1,
                start_weight=0.1,
                end_weight=0.02,
                decay_steps=16,
            ),
            0.1,
            places=6,
        )
        self.assertAlmostEqual(
            _scheduled_linear_decay_weight(
                current_step=16,
                start_weight=0.1,
                end_weight=0.02,
                decay_steps=16,
            ),
            0.02,
            places=6,
        )
        self.assertAlmostEqual(
            _scheduled_linear_decay_weight(
                current_step=32,
                start_weight=0.1,
                end_weight=0.02,
                decay_steps=16,
            ),
            0.02,
            places=6,
        )

    def test_teacher_advantage_weight_is_monotonic(self) -> None:
        low = _teacher_advantage_weight(
            torch.tensor(0.1),
            torch.tensor(0.2),
            center=0.0,
            scale=0.25,
        )
        high = _teacher_advantage_weight(
            torch.tensor(0.8),
            torch.tensor(0.2),
            center=0.0,
            scale=0.25,
        )
        self.assertLess(float(low.item()), float(high.item()))

    def test_alignment_aux_choice_loss_is_finite_for_kl_and_js(self) -> None:
        active = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32, requires_grad=True)
        teacher = torch.tensor([0.8, -0.3, 0.1], dtype=torch.float32)
        kl = _alignment_aux_choice_loss(
            mode="teacher_choice_kl",
            active_scores=active,
            teacher_scores=teacher,
            temperature=2.0,
        )
        js = _alignment_aux_choice_loss(
            mode="teacher_choice_js",
            active_scores=active,
            teacher_scores=teacher,
            temperature=2.0,
        )
        total = kl + js
        total.backward()
        self.assertTrue(torch.isfinite(kl).item())
        self.assertTrue(torch.isfinite(js).item())
        self.assertIsNotNone(active.grad)

    def test_effective_rank_and_class_entropy_are_finite(self) -> None:
        memory = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        self.assertGreater(_effective_rank(memory), 1.0)
        entropy = _class_entropy(torch.tensor([1.0, 0.5, -0.2], dtype=torch.float32))
        self.assertGreaterEqual(entropy, 0.0)

    def test_slot_diversity_loss_prefers_distinct_slots(self) -> None:
        collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        diverse = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        self.assertGreater(float(_slot_diversity_loss(collapsed).item()), 0.5)
        self.assertLess(float(_slot_diversity_loss(diverse).item()), 0.1)

    def test_reader_attention_diversity_loss_prefers_specialized_queries(self) -> None:
        identical = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float32)
        specialized = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        self.assertGreater(float(_reader_attention_diversity_loss(identical).item()), 0.9)
        self.assertLess(float(_reader_attention_diversity_loss(specialized).item()), 0.1)

    def test_conditioned_query_orthogonality_loss_prefers_separated_queries(self) -> None:
        collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        separated = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        self.assertGreater(float(_conditioned_query_orthogonality_loss(collapsed).item()), 0.5)
        self.assertLess(float(_conditioned_query_orthogonality_loss(separated).item()), 0.1)

    def test_reader_short_reconstruction_loss_prefers_matching_readouts(self) -> None:
        memory_short = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        matching = memory_short.clone()
        mismatched = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
        matching_loss = _reader_short_reconstruction_loss(memory_short, matching)
        mismatched_loss = _reader_short_reconstruction_loss(memory_short, mismatched)
        self.assertIsNotNone(matching_loss)
        self.assertIsNotNone(mismatched_loss)
        self.assertLess(float(matching_loss.item()), 1e-6)
        self.assertGreater(float(mismatched_loss.item()), float(matching_loss.item()))

    def test_reader_fuser_bootstrap_active_respects_step_window(self) -> None:
        self.assertTrue(_reader_fuser_bootstrap_active(current_step=1, bootstrap_steps=8))
        self.assertTrue(_reader_fuser_bootstrap_active(current_step=8, bootstrap_steps=8))
        self.assertFalse(_reader_fuser_bootstrap_active(current_step=9, bootstrap_steps=8))
        self.assertFalse(_reader_fuser_bootstrap_active(current_step=1, bootstrap_steps=0))

    def test_writer_slot_basis_orthogonality_loss_prefers_orthogonal_slots(self) -> None:
        writer = MemoryWriter(embed_dim=4, memory_slots=2, arch="transformer", num_heads=2)
        with torch.no_grad():
            writer.slot_embeddings.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
        collapsed_loss = _writer_slot_basis_orthogonality_loss(writer)
        with torch.no_grad():
            writer.slot_embeddings.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                )
            )
        orthogonal_loss = _writer_slot_basis_orthogonality_loss(writer)
        self.assertIsNotNone(collapsed_loss)
        self.assertIsNotNone(orthogonal_loss)
        self.assertGreater(float(collapsed_loss.item()), 0.5)
        self.assertLess(float(orthogonal_loss.item()), 0.1)

    def test_writer_slot_energy_balance_loss_prefers_balanced_slot_norms(self) -> None:
        balanced = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        imbalanced = torch.tensor(
            [[[4.0, 0.0], [0.1, 0.0]]],
            dtype=torch.float32,
        )
        balanced_loss = _writer_slot_energy_balance_loss(balanced)
        imbalanced_loss = _writer_slot_energy_balance_loss(imbalanced)
        self.assertIsNotNone(balanced_loss)
        self.assertIsNotNone(imbalanced_loss)
        assert balanced_loss is not None
        assert imbalanced_loss is not None
        self.assertLess(float(balanced_loss.item()), float(imbalanced_loss.item()))

    def test_writer_common_mode_penalty_prefers_centered_slots(self) -> None:
        collapsed = torch.ones(1, 3, 4, dtype=torch.float32)
        centered = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        collapsed_penalty = _writer_common_mode_penalty(collapsed)
        centered_penalty = _writer_common_mode_penalty(centered)
        self.assertIsNotNone(collapsed_penalty)
        self.assertIsNotNone(centered_penalty)
        assert collapsed_penalty is not None
        assert centered_penalty is not None
        self.assertGreater(float(collapsed_penalty.item()), float(centered_penalty.item()))

    def test_latent_anchor_loss_combines_support_and_writer_cosines(self) -> None:
        current_support = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        current_writer = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        reference_support = current_support.clone()
        reference_writer = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
        total_loss, support_loss, writer_loss, support_cosine, writer_cosine = _latent_anchor_loss(
            current_support_states=current_support,
            reference_support_states=reference_support,
            current_memory_slots=current_writer,
            reference_memory_slots=reference_writer,
        )
        self.assertAlmostEqual(float(support_loss.item()), 0.0, places=6)
        self.assertAlmostEqual(float(support_cosine.item()), 1.0, places=6)
        self.assertGreater(float(writer_loss.item()), 0.0)
        self.assertLess(float(writer_cosine.item()), 1.0)
        self.assertGreater(float(total_loss.item()), 0.0)

    def test_init_and_eval_checkpoint_paths_are_mutually_exclusive(self) -> None:
        config = {
            "runtime": {
                "shared_injection_arm": "injected",
                "writer_memory_control": "real",
                "pilot_init_checkpoint_path": "/tmp/init.pt",
                "pilot_checkpoint_path": "/tmp/eval.pt",
            },
            "task": {
                "support_dataset_path": "unused-support",
                "dataset_path": "unused-eval",
            },
        }
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run_shared_injection_pilot(
                config=config,
                seed=7,
                output_dir=Path(tempfile.mkdtemp()),
                resume=None,
                dry_run=True,
            )

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

    def test_transformer_writer_support_set_residual_preserves_slot_identity(self) -> None:
        class ZeroCrossAttention(torch.nn.Module):
            def forward(self, query, key, value, need_weights=False):
                return torch.zeros_like(query), None

        writer = MemoryWriter(
            embed_dim=6,
            memory_slots=4,
            arch="transformer",
            num_heads=2,
            transformer_layers=1,
            dropout=0.0,
            support_query_residual_scale=1.0,
        )
        writer.support_cross_attn = ZeroCrossAttention()
        writer.encoder = torch.nn.Identity()
        writer.output_norm = torch.nn.Identity()
        support_states = torch.zeros(2, 6, 6)
        memory = writer.write(support_states, input_schema="support_set")
        self.assertEqual(list(memory.shape), [2, 4, 6])
        self.assertGreater(float(memory.var(dim=1).sum().item()), 0.0)

    def test_transformer_writer_output_slot_basis_preserves_basis_signal(self) -> None:
        class ZeroCrossAttention(torch.nn.Module):
            def forward(self, query, key, value, need_weights=False):
                return torch.zeros_like(query), None

        writer = MemoryWriter(
            embed_dim=6,
            memory_slots=4,
            arch="transformer",
            num_heads=2,
            transformer_layers=1,
            dropout=0.0,
            output_slot_basis_scale=1.0,
        )
        writer.support_cross_attn = ZeroCrossAttention()
        writer.encoder = torch.nn.Identity()
        writer.output_norm = torch.nn.Identity()
        with torch.no_grad():
            writer.state_proj.weight.zero_()
            writer.state_proj.bias.zero_()
        support_states = torch.zeros(2, 6, 6)
        memory = writer.write(support_states, input_schema="support_set")
        self.assertEqual(list(memory.shape), [2, 4, 6])
        self.assertGreater(float(memory.var(dim=1).sum().item()), 0.0)

    def test_transformer_writer_shared_add_scaled_respects_shared_state_scale(self) -> None:
        class ZeroCrossAttention(torch.nn.Module):
            def forward(self, query, key, value, need_weights=False):
                return torch.zeros_like(query), None

        writer = MemoryWriter(
            embed_dim=6,
            memory_slots=4,
            arch="transformer",
            num_heads=2,
            transformer_layers=1,
            dropout=0.0,
            support_query_residual_scale=1.0,
            slot_conditioning_mode="shared_add_scaled",
            shared_state_scale=0.25,
        )
        writer.support_cross_attn = ZeroCrossAttention()
        writer.encoder = torch.nn.Identity()
        writer.output_norm = torch.nn.Identity()
        with torch.no_grad():
            writer.state_proj.weight.copy_(torch.eye(6))
            writer.state_proj.bias.zero_()
        support_states = torch.ones(2, 6, 6)
        expected = writer.slot_embeddings.detach().unsqueeze(0).expand(2, -1, -1) + 0.25
        memory = writer.write(support_states, input_schema="support_set")
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6))

    def test_transformer_writer_slot_query_only_ignores_shared_state(self) -> None:
        class ZeroCrossAttention(torch.nn.Module):
            def forward(self, query, key, value, need_weights=False):
                return torch.zeros_like(query), None

        writer = MemoryWriter(
            embed_dim=6,
            memory_slots=4,
            arch="transformer",
            num_heads=2,
            transformer_layers=1,
            dropout=0.0,
            support_query_residual_scale=1.0,
            slot_conditioning_mode="slot_query_only",
        )
        writer.support_cross_attn = ZeroCrossAttention()
        writer.encoder = torch.nn.Identity()
        writer.output_norm = torch.nn.Identity()
        with torch.no_grad():
            writer.state_proj.weight.fill_(2.0)
            writer.state_proj.bias.fill_(1.0)
        support_states = torch.randn(2, 6, 6)
        expected = writer.slot_embeddings.detach().unsqueeze(0).expand(2, -1, -1)
        memory = writer.write(support_states, input_schema="support_set")
        self.assertTrue(torch.allclose(memory, expected, atol=1e-6))

    def test_writer_orthogonalize_slot_embeddings_preserves_mean_norm(self) -> None:
        writer = MemoryWriter(embed_dim=8, memory_slots=4, arch="transformer", num_heads=2)
        before_norm = float(writer.slot_embeddings.norm(dim=-1).mean().item())
        writer.orthogonalize_slot_embeddings_()
        after_norm = float(writer.slot_embeddings.norm(dim=-1).mean().item())
        cosine = torch.nn.functional.normalize(writer.slot_embeddings, dim=-1) @ torch.nn.functional.normalize(
            writer.slot_embeddings, dim=-1
        ).T
        mask = ~torch.eye(cosine.shape[0], dtype=torch.bool)
        self.assertAlmostEqual(before_norm, after_norm, delta=0.1)
        self.assertLess(float(cosine[mask].abs().mean().item()), 0.2)

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
        aux_loss, active_flag, diagnostics = _alignment_aux_loss(
            mode="teacher_margin",
            active_scores=active,
            base_scores=base,
            teacher_scores=teacher,
            gold_index=0,
        )
        self.assertIsNone(aux_loss)
        self.assertFalse(active_flag)
        self.assertIn("teacher_choice_kl", diagnostics)

    def test_alignment_aux_loss_teacher_margin_activates_when_teacher_improves_margin(self) -> None:
        active = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32)
        base = torch.tensor([0.2, 0.25, 0.1], dtype=torch.float32)
        teacher = torch.tensor([0.6, 0.1, 0.0], dtype=torch.float32)
        aux_loss, active_flag, diagnostics = _alignment_aux_loss(
            mode="teacher_margin",
            active_scores=active,
            base_scores=base,
            teacher_scores=teacher,
            gold_index=0,
        )
        self.assertIsNotNone(aux_loss)
        self.assertTrue(active_flag)
        self.assertGreater(float(aux_loss.item()), 0.0)
        self.assertGreater(diagnostics["teacher_margin_minus_base_margin"], 0.0)

    def test_alignment_aux_loss_teacher_choice_kl_uses_dense_weight(self) -> None:
        active = torch.tensor([0.25, 0.2, 0.1], dtype=torch.float32)
        base = torch.tensor([0.2, 0.25, 0.1], dtype=torch.float32)
        teacher = torch.tensor([0.8, 0.0, -0.2], dtype=torch.float32)
        aux_loss, active_flag, diagnostics = _alignment_aux_loss(
            mode="teacher_choice_kl",
            active_scores=active,
            base_scores=base,
            teacher_scores=teacher,
            gold_index=0,
            temperature=2.0,
            advantage_center=0.0,
            advantage_scale=0.25,
        )
        self.assertIsNotNone(aux_loss)
        self.assertTrue(active_flag)
        self.assertGreater(float(aux_loss.item()), 0.0)
        self.assertGreater(diagnostics["teacher_advantage_weight_mean"], 0.0)

    def test_prefix_stats_capture_two_level_memory_and_reader_metrics(self) -> None:
        memory_long = torch.randn(1, 8, 6)
        memory_short = torch.randn(1, 4, 6)
        reader_attention = torch.softmax(torch.randn(1, 4, 8), dim=-1)
        base_queries = torch.randn(1, 4, 6)
        conditioned_queries = base_queries + 0.5
        attention_logits = torch.randn(1, 2, 4, 8)
        reader_value_projected_slots = torch.randn(1, 8, 6)
        reader_readouts = torch.randn(1, 4, 6)
        fuser = MemoryFuser(embed_dim=6, num_queries=4, short_slots=4, arch="linear", hidden_dim=12)
        stats = _prefix_stats(
            layer_prefix_hidden_by_layer={0: torch.randn(1, 4, 6)},
            memory_long=memory_long,
            memory_short=memory_short,
            reader_attention=reader_attention,
            reader_base_queries=base_queries,
            reader_conditioned_queries=conditioned_queries,
            reader_attention_logits=attention_logits,
            reader_value_projected_slots=reader_value_projected_slots,
            reader_readouts=reader_readouts,
            fuser=fuser,
            memory_path_variant="two_level",
            projector_token_source="short_slots",
        )
        self.assertEqual(stats["pilot_memory_path_variant"], "two_level")
        self.assertEqual(stats["pilot_projector_token_source"], "short_slots")
        self.assertEqual(stats["memory_long_slots"], 8.0)
        self.assertEqual(stats["memory_short_slots"], 4.0)
        self.assertEqual(stats["reader_num_queries"], 4.0)
        self.assertGreaterEqual(stats["reader_attention_entropy_mean"], 0.0)
        self.assertGreaterEqual(stats["reader_context_overwrite_ratio"], 0.0)
        self.assertGreaterEqual(stats["memory_long_common_mode_energy_ratio"], 0.0)
        self.assertGreaterEqual(stats["memory_long_top1_top2_ratio"], 0.0)
        self.assertGreaterEqual(stats["memory_long_centered_effective_rank"], 0.0)
        self.assertGreaterEqual(stats["reader_value_projected_effective_rank"], 0.0)
        self.assertGreaterEqual(stats["reader_value_projected_pairwise_cosine_mean"], -1.0)
        self.assertGreater(stats["reader_readout_effective_rank"], 0.0)
        self.assertGreaterEqual(stats["reader_readout_centered_effective_rank"], 0.0)
        self.assertGreaterEqual(stats["fuser_output_effective_rank"], 0.0)
        self.assertGreaterEqual(stats["fuser_rank_gain_over_readout"], -4.0)
        self.assertEqual(len(stats["memory_long_slot_norm_histogram_counts"]), 4)
        self.assertGreaterEqual(stats["memory_long_singular_value_top1"], 0.0)

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_two_level_runtime_uses_prompt_summary_cache_and_short_slots(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"
                self._calls: list[tuple[str, ...]] = []

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                self._calls.append(tuple(texts))
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0, "1": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0, "1": 1.0},
                }

            def to(self, *_args, **_kwargs):
                return self

        fake_backbone = FakeBackbone()
        mock_backbone_cls.return_value = fake_backbone
        config = {
            "backbone": {
                "name": "fake",
                "load_mode": "stub",
                "dtype": "float32",
                "model_id": "fake/model",
            },
            "method": {
                "writer": {"memory_slots": 8, "arch": "mlp", "hidden_dim": 12},
                "reader": {"num_queries": 4, "num_heads": 2, "query_residual_scale": 1.0},
                "fuser": {"short_slots": 4, "arch": "resampler", "num_heads": 2},
            },
            "runtime": {
                "device": "cpu",
                "pilot_memory_path_variant": "two_level",
                "pilot_reader_context_mode": "prompt_summary",
                "pilot_projector_token_source": "short_slots",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }
        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=3,
            arm="injected",
            writer_memory_control="real",
        )
        prefix_artifacts = runtime.build_prefix_artifacts(
            "Support bank",
            prompt_text="Claim: cached prompt",
        )
        _ = runtime.build_prefix_artifacts(
            "Support bank",
            prompt_text="Claim: cached prompt",
        )
        self.assertEqual(list(prefix_artifacts.memory_long.shape), [1, 8, 6])
        self.assertEqual(list(prefix_artifacts.memory_short.shape), [1, 4, 6])
        self.assertIn("reader_context_overwrite_ratio", prefix_artifacts.prefix_stats)
        self.assertIn("fuser_output_effective_rank", prefix_artifacts.prefix_stats)
        prompt_calls = [call for call in fake_backbone._calls if call == ("Claim: cached prompt",)]
        self.assertEqual(len(prompt_calls), 1)

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_writer_direct_runtime_builds_context_conditioned_prefix(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"
                self.prompt_slice_calls: list[tuple[str, int]] = []

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0},
                }

            def extract_prompt_hidden_state_slice(self, texts, *, max_tokens=8):
                prompt_list = list(texts)
                self.prompt_slice_calls.append((prompt_list[0], int(max_tokens)))
                hidden = torch.full((len(prompt_list), max_tokens, self.hidden_size), 0.5, dtype=torch.float32)
                mask = torch.ones(len(prompt_list), max_tokens, dtype=torch.bool)
                return hidden, mask

            def to(self, *_args, **_kwargs):
                return self

        fake_backbone = FakeBackbone()
        mock_backbone_cls.return_value = fake_backbone
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {
                    "memory_slots": 4,
                    "hidden_dim": 12,
                    "num_heads": 2,
                    "transformer_layers": 1,
                    "conditioning_layers": 2,
                    "dropout": 0.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "pilot_bridge_mode": "writer_direct",
                "pilot_writer_stimulus_mode": "support_and_context",
                "pilot_writer_context_tokens": 3,
                "pilot_memory_path_variant": "single_level",
                "pilot_injection_mode": "shallow_prefix",
                "pilot_support_encoder_num_heads": 2,
            },
        }
        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=19,
            arm="injected",
            writer_memory_control="real",
        )
        prefix_artifacts = runtime.build_prefix_artifacts(
            "Support bank",
            prompt_text="Need context now",
        )
        self.assertEqual(runtime.bridge_mode, "writer_direct")
        self.assertIsInstance(runtime.writer, WriterWeaverHead)
        self.assertEqual(runtime.writer.conditioning_layers, 2)
        self.assertEqual(list(prefix_artifacts.memory_long.shape), [1, 4, 6])
        self.assertEqual(list(prefix_artifacts.writer_context_states.shape), [1, 3, 6])
        self.assertEqual(list(prefix_artifacts.writer_context_mask.shape), [1, 3])
        self.assertGreater(prefix_artifacts.prefix_stats["writer_context_token_count"], 0.0)
        self.assertGreater(prefix_artifacts.prefix_stats["writer_support_state_count"], 0.0)
        self.assertEqual(fake_backbone.prompt_slice_calls, [("Need context now", 3)])

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_writer_direct_runtime_enables_writer_micro_lora(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {}

            def extract_prompt_hidden_state_slice(self, texts, *, max_tokens=8):
                hidden = torch.ones(len(texts), max_tokens, self.hidden_size, dtype=torch.float32)
                mask = torch.ones(len(texts), max_tokens, dtype=torch.bool)
                return hidden, mask

            def to(self, *_args, **_kwargs):
                return self

        mock_backbone_cls.return_value = FakeBackbone()
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {
                    "memory_slots": 4,
                    "hidden_dim": 12,
                    "num_heads": 2,
                    "transformer_layers": 2,
                    "conditioning_layers": 2,
                    "dropout": 0.0,
                },
                "writer_adapter": {
                    "enabled": True,
                    "target_modules": ["conditioning_out_proj", "encoder_self_attn_out_proj"],
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "pilot_bridge_mode": "writer_direct",
                "pilot_writer_stimulus_mode": "support_and_context",
                "pilot_writer_context_tokens": 3,
                "pilot_memory_path_variant": "single_level",
                "pilot_injection_mode": "shallow_prefix",
                "pilot_trainable_variant": "writer_adapter_only",
            },
        }
        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=19,
            arm="injected",
            writer_memory_control="real",
        )
        self.assertTrue(runtime.writer_adapter_enabled)
        self.assertEqual(
            runtime.writer_adapter_target_modules,
            ("conditioning_out_proj", "encoder_self_attn_out_proj"),
        )
        self.assertGreater(runtime.writer_adapter_trainable_params, 0)
        runtime.set_writer_base_trainable(False)
        runtime.set_writer_adapter_trainable(True)
        self.assertFalse(runtime.writer.slot_embeddings.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in runtime.writer_adapter_parameters()))

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_writer_direct_runtime_threads_v6_support_modes(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                rows = []
                for index, _text in enumerate(texts):
                    rows.append(torch.full((self.hidden_size,), float(index + 1), dtype=torch.float32))
                return torch.stack(rows, dim=0)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {}

            def extract_prompt_hidden_state_slice(self, texts, *, max_tokens=8):
                hidden = torch.ones(len(texts), max_tokens, self.hidden_size, dtype=torch.float32)
                mask = torch.ones(len(texts), max_tokens, dtype=torch.bool)
                return hidden, mask

            def to(self, *_args, **_kwargs):
                return self

        mock_backbone_cls.return_value = FakeBackbone()
        support_rows = [
            {"id": "1", "label": "SUPPORTS", "claim": "c1", "evidence": "e1"},
            {"id": "2", "label": "REFUTES", "claim": "c2", "evidence": "e2"},
            {"id": "3", "label": "NOT_ENOUGH_INFO", "claim": "c3", "evidence": "e3"},
        ]
        base_config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {
                    "memory_slots": 4,
                    "hidden_dim": 12,
                    "num_heads": 2,
                    "transformer_layers": 1,
                    "conditioning_layers": 2,
                    "dropout": 0.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "pilot_bridge_mode": "writer_direct",
                "pilot_writer_stimulus_mode": "support_and_context",
                "pilot_writer_context_tokens": 3,
                "pilot_memory_path_variant": "single_level",
                "pilot_injection_mode": "shallow_prefix",
                "pilot_support_encoder_num_heads": 2,
            },
        }
        mode_expectations = {
            "pooled_block": (1, 1),
            "structured_support_set": (3, 3),
            "multi_item_cross_attn_raw": (3, 3),
            "multi_item_cross_attn_encoded": (3, 3),
            "hybrid_pooled_plus_items": (3, 4),
        }
        for support_mode, (item_count, writer_state_count) in mode_expectations.items():
            with self.subTest(support_mode=support_mode):
                config = json.loads(json.dumps(base_config))
                config["runtime"]["pilot_support_encoder_mode"] = support_mode
                if support_mode == "pooled_block":
                    config["runtime"]["pilot_context_support_balance_mode"] = "layernorm_learned_scalar"
                    config["runtime"]["pilot_context_balance_scale_init"] = 0.8
                    config["runtime"]["pilot_support_balance_scale_init"] = 1.2
                runtime = SharedInjectionPilotRuntime(
                    config=config,
                    seed=23,
                    arm="injected",
                    writer_memory_control="real",
                )
                prefix_artifacts = runtime.build_prefix_artifacts(
                    "Support bank",
                    support_rows=support_rows,
                    prompt_text="Need context now",
                )
                self.assertEqual(
                    int(prefix_artifacts.prefix_stats["support_item_count"]),
                    item_count,
                )
                self.assertEqual(
                    int(prefix_artifacts.prefix_stats["writer_support_state_count"]),
                    writer_state_count,
                )
                self.assertEqual(
                    prefix_artifacts.prefix_stats["pilot_support_encoder_mode"],
                    support_mode,
                )
                if support_mode == "pooled_block":
                    self.assertEqual(
                        prefix_artifacts.prefix_stats["pilot_context_support_balance_mode"],
                        "layernorm_learned_scalar",
                    )
                    self.assertAlmostEqual(
                        prefix_artifacts.prefix_stats["writer_context_balance_scale"],
                        0.8,
                        places=5,
                    )
                    self.assertAlmostEqual(
                        prefix_artifacts.prefix_stats["writer_support_balance_scale"],
                        1.2,
                        places=5,
                    )

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_writer_direct_runtime_allows_receiver_lora_under_sparse_deep_prefix(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"
                self.enable_calls: list[dict[str, object]] = []
                self.receiver_lora_trainable_enabled = False

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0, "1": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0, "1": 1.0},
                }

            def enable_receiver_micro_lora(self, **kwargs):
                self.enable_calls.append(dict(kwargs))

            def receiver_lora_parameter_count(self) -> int:
                return 24

            def set_receiver_lora_trainable(self, enabled: bool) -> None:
                self.receiver_lora_trainable_enabled = bool(enabled)

            def receiver_lora_state_dict(self):
                return None

            def load_receiver_lora_state_dict(self, *_args, **_kwargs):
                return None

            def validate_receiver_lora_state_dict(self, *_args, **_kwargs):
                return None

            def extract_prompt_hidden_state_slice(self, texts, *, max_tokens=8):
                prompt_list = list(texts)
                hidden = torch.ones(len(prompt_list), max_tokens, self.hidden_size, dtype=torch.float32)
                mask = torch.ones(len(prompt_list), max_tokens, dtype=torch.bool)
                return hidden, mask

            def to(self, *_args, **_kwargs):
                return self

        fake_backbone = FakeBackbone()
        mock_backbone_cls.return_value = fake_backbone
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {
                    "memory_slots": 4,
                    "hidden_dim": 12,
                    "num_heads": 2,
                    "transformer_layers": 1,
                    "conditioning_layers": 2,
                    "dropout": 0.0,
                },
                "receiver_lora": {
                    "enabled": True,
                    "target_layers": [0, 1],
                    "target_modules": ["k_proj", "v_proj"],
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "pilot_bridge_mode": "writer_direct",
                "pilot_writer_stimulus_mode": "support_and_context",
                "pilot_writer_context_tokens": 3,
                "pilot_memory_path_variant": "single_level",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }

        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=23,
            arm="injected",
            writer_memory_control="real",
        )

        self.assertTrue(runtime.receiver_lora_enabled)
        self.assertEqual(runtime.receiver_lora_target_layers, (0, 1))
        self.assertEqual(runtime.receiver_lora_target_modules, ("k_proj", "v_proj"))
        self.assertEqual(runtime.receiver_lora_trainable_params, 24)
        self.assertTrue(fake_backbone.receiver_lora_trainable_enabled)
        self.assertEqual(
            fake_backbone.enable_calls,
            [
                {
                    "layer_indices": (0, 1),
                    "target_modules": ("k_proj", "v_proj"),
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                }
            ],
        )

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_writer_direct_runtime_rejects_receiver_lora_under_shallow_prefix(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {}

            def to(self, *_args, **_kwargs):
                return self

        mock_backbone_cls.return_value = FakeBackbone()
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {
                    "memory_slots": 4,
                    "hidden_dim": 12,
                    "num_heads": 2,
                    "transformer_layers": 1,
                    "conditioning_layers": 2,
                    "dropout": 0.0,
                },
                "receiver_lora": {
                    "enabled": True,
                    "target_layers": [0, 1],
                    "target_modules": ["k_proj", "v_proj"],
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "pilot_bridge_mode": "writer_direct",
                "pilot_writer_stimulus_mode": "support_and_context",
                "pilot_writer_context_tokens": 3,
                "pilot_memory_path_variant": "single_level",
                "pilot_injection_mode": "shallow_prefix",
            },
        }

        with self.assertRaisesRegex(ValueError, "receiver micro-LoRA only when .*sparse_deep_prefix"):
            SharedInjectionPilotRuntime(
                config=config,
                seed=29,
                arm="injected",
                writer_memory_control="real",
            )

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_writer_direct_source_stub_builds_deep_prefix_and_records_init_mode(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"
                self.calibration_calls: list[tuple[tuple[str, ...], tuple[int, ...], int]] = []

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.full((len(texts), self.hidden_size), 0.25, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, layer_prefix_hidden_by_layer, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {
                        str(layer_index): float(tensor.norm().item())
                        for layer_index, tensor in layer_prefix_hidden_by_layer.items()
                    },
                    "layer_value_l2_by_layer": {
                        str(layer_index): float(tensor.norm().item())
                        for layer_index, tensor in layer_prefix_hidden_by_layer.items()
                    },
                    "layer_hidden_l2_by_layer": {
                        str(layer_index): float(tensor.norm().item())
                        for layer_index, tensor in layer_prefix_hidden_by_layer.items()
                    },
                }

            def extract_prompt_hidden_state_slice(self, texts, *, max_tokens=8):
                prompt_list = list(texts)
                hidden = torch.full((len(prompt_list), max_tokens, self.hidden_size), 0.5, dtype=torch.float32)
                mask = torch.ones(len(prompt_list), max_tokens, dtype=torch.bool)
                return hidden, mask

            def collect_deep_prefix_calibration(self, texts, *, layer_indices, max_tokens=8):
                text_list = tuple(str(text) for text in texts)
                self.calibration_calls.append((text_list, tuple(int(layer) for layer in layer_indices), int(max_tokens)))
                semantic_anchor = torch.full((1, self.hidden_size), 0.25, dtype=torch.float32)
                hidden_state_anchor = torch.full((1, max_tokens, self.hidden_size), 0.5, dtype=torch.float32)
                layer_hidden_anchor_by_layer = {
                    int(layer_index): torch.full(
                        (1, max_tokens, self.hidden_size),
                        0.75 + float(layer_index),
                        dtype=torch.float32,
                    )
                    for layer_index in layer_indices
                }
                return {
                    "semantic_anchor": semantic_anchor,
                    "hidden_state_anchor": hidden_state_anchor,
                    "hidden_state_mask": torch.ones(1, max_tokens, dtype=torch.bool),
                    "layer_hidden_anchor_by_layer": layer_hidden_anchor_by_layer,
                    "layer_hidden_mask": torch.ones(1, max_tokens, dtype=torch.bool),
                    "layer_key_l2_by_layer": {str(layer): 1.0 for layer in layer_indices},
                    "layer_value_l2_by_layer": {str(layer): 1.0 for layer in layer_indices},
                    "layer_hidden_l2_by_layer": {str(layer): 1.0 for layer in layer_indices},
                }

            def to(self, *_args, **_kwargs):
                return self

        fake_backbone = FakeBackbone()
        mock_backbone_cls.return_value = fake_backbone
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {
                    "memory_slots": 4,
                    "hidden_dim": 12,
                    "num_heads": 2,
                    "transformer_layers": 1,
                    "conditioning_layers": 2,
                    "dropout": 0.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "pilot_bridge_mode": "writer_direct",
                "pilot_memory_path_variant": "single_level",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
                "pilot_prefix_source_mode": "source_stub",
                "pilot_deep_prefix_init_mode": "kv_stat_match",
                "pilot_writer_context_tokens": 4,
            },
        }

        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=31,
            arm="injected",
            writer_memory_control="real",
        )
        prefix_artifacts = runtime.build_prefix_artifacts(
            "Support bank",
            prompt_text="Need context now",
        )

        self.assertIsNotNone(runtime.source_stub)
        self.assertEqual(list(prefix_artifacts.memory_long.shape), [1, 4, 6])
        assert prefix_artifacts.layer_prefix_hidden_by_layer is not None
        self.assertEqual(sorted(prefix_artifacts.layer_prefix_hidden_by_layer), [0, 1])
        self.assertEqual(prefix_artifacts.prefix_stats["pilot_prefix_source_mode"], "source_stub")
        self.assertEqual(prefix_artifacts.prefix_stats["pilot_deep_prefix_init_mode"], "kv_stat_match")
        self.assertEqual(len(fake_backbone.calibration_calls), 1)
        calibration_texts, calibration_layers, calibration_tokens = fake_backbone.calibration_calls[0]
        self.assertEqual(calibration_texts, ("Need context now", "Support bank"))
        self.assertEqual(calibration_layers, (0, 1))
        self.assertEqual(calibration_tokens, 4)
        checkpoint = {
            "writer_state": runtime.writer.state_dict(),
            "source_stub_state": runtime.source_stub.state_dict(),
            "support_encoder_state": None,
            "reader_state": None,
            "fuser_state": None,
            "prefix_projector_state": runtime.prefix_projector.state_dict(),
            "pilot_memory_path_variant": "single_level",
            "pilot_support_encoder_mode": "pooled_block",
            "pilot_injection_mode": "sparse_deep_prefix",
            "pilot_bridge_mode": "writer_direct",
            "pilot_prefix_source_mode": "source_stub",
            "pilot_deep_prefix_init_mode": "kv_stat_match",
            "pilot_deep_prefix_layers": [0, 1],
            "backbone_hidden_size": 6,
            "writer_memory_slots": 4,
            "pilot_projector_prefix_tokens": 4,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "source_stub_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            loaded = runtime.load_injection_checkpoint(checkpoint_path)
            self.assertIn("source_stub_state", loaded)

    def test_update_writer_weaver_summary_reports_move_to_w1(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_writer_weaver_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def write_metrics(name: str, payload: dict[str, object]) -> Path:
                path = tmp_path / name
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return path

            def task_metrics(
                *,
                benchmark_id: str,
                task_name: str,
                metric_name: str,
                task_score: float,
                exact_match: float,
                delta_answer_logprob: float,
                writer_context_token_count: float,
                common_mode_ratio: float,
                centered_rank: float,
                top1_top2_ratio: float,
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": task_name,
                    "task_metric_name": metric_name,
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": exact_match,
                    "delta_answer_logprob": delta_answer_logprob,
                    "writer_context_token_count": writer_context_token_count,
                    "memory_long_common_mode_energy_ratio": common_mode_ratio,
                    "memory_long_centered_effective_rank": centered_rank,
                    "memory_long_top1_top2_ratio": top1_top2_ratio,
                    "projected_memory_effective_rank": 2.5,
                    "memory_long_pairwise_cosine_mean": 0.15,
                }

            gsm8k_control = write_metrics(
                "gsm8k-control.json",
                task_metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    metric_name="exact_match",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    writer_context_token_count=0.0,
                    common_mode_ratio=0.98,
                    centered_rank=1.1,
                    top1_top2_ratio=21.0,
                ),
            )
            gsm8k_support_only = write_metrics(
                "gsm8k-support-only.json",
                task_metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    metric_name="exact_match",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=-0.3,
                    writer_context_token_count=0.0,
                    common_mode_ratio=0.99,
                    centered_rank=1.0,
                    top1_top2_ratio=22.0,
                ),
            )
            gsm8k_support_context = write_metrics(
                "gsm8k-support-context.json",
                task_metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    metric_name="exact_match",
                    task_score=0.5,
                    exact_match=0.5,
                    delta_answer_logprob=0.2,
                    writer_context_token_count=8.0,
                    common_mode_ratio=0.94,
                    centered_rank=1.3,
                    top1_top2_ratio=18.0,
                ),
            )
            narrativeqa_control = write_metrics(
                "narrativeqa-control.json",
                task_metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    metric_name="qa_f1",
                    task_score=0.2,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    writer_context_token_count=0.0,
                    common_mode_ratio=0.96,
                    centered_rank=1.2,
                    top1_top2_ratio=16.0,
                ),
            )
            narrativeqa_support_only = write_metrics(
                "narrativeqa-support-only.json",
                task_metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    metric_name="qa_f1",
                    task_score=0.2,
                    exact_match=0.0,
                    delta_answer_logprob=-0.1,
                    writer_context_token_count=0.0,
                    common_mode_ratio=0.95,
                    centered_rank=1.2,
                    top1_top2_ratio=16.0,
                ),
            )
            narrativeqa_support_context = write_metrics(
                "narrativeqa-support-context.json",
                task_metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    metric_name="qa_f1",
                    task_score=0.25,
                    exact_match=0.0,
                    delta_answer_logprob=0.05,
                    writer_context_token_count=8.0,
                    common_mode_ratio=0.95,
                    centered_rank=1.2,
                    top1_top2_ratio=16.0,
                ),
            )
            fever_control = write_metrics(
                "fever-control.json",
                task_metrics(
                    benchmark_id="fever",
                    task_name="fever",
                    metric_name="accuracy",
                    task_score=0.3,
                    exact_match=0.3,
                    delta_answer_logprob=0.0,
                    writer_context_token_count=0.0,
                    common_mode_ratio=0.98,
                    centered_rank=1.1,
                    top1_top2_ratio=19.0,
                ),
            )
            fever_support_only = write_metrics(
                "fever-support-only.json",
                task_metrics(
                    benchmark_id="fever",
                    task_name="fever",
                    metric_name="accuracy",
                    task_score=0.3,
                    exact_match=0.3,
                    delta_answer_logprob=-0.02,
                    writer_context_token_count=0.0,
                    common_mode_ratio=0.98,
                    centered_rank=1.1,
                    top1_top2_ratio=19.0,
                ),
            )
            fever_support_context = write_metrics(
                "fever-support-context.json",
                task_metrics(
                    benchmark_id="fever",
                    task_name="fever",
                    metric_name="accuracy",
                    task_score=0.31,
                    exact_match=0.31,
                    delta_answer_logprob=-0.01,
                    writer_context_token_count=8.0,
                    common_mode_ratio=0.98,
                    centered_rank=1.1,
                    top1_top2_ratio=19.0,
                ),
            )

            output_json = tmp_path / "w0-summary.json"
            output_report = tmp_path / "w0-summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--gsm8k_control_metrics_json",
                    str(gsm8k_control),
                    "--gsm8k_support_only_metrics_json",
                    str(gsm8k_support_only),
                    "--gsm8k_support_context_metrics_json",
                    str(gsm8k_support_context),
                    "--narrativeqa_control_metrics_json",
                    str(narrativeqa_control),
                    "--narrativeqa_support_only_metrics_json",
                    str(narrativeqa_support_only),
                    "--narrativeqa_support_context_metrics_json",
                    str(narrativeqa_support_context),
                    "--fever_control_metrics_json",
                    str(fever_control),
                    "--fever_support_only_metrics_json",
                    str(fever_support_only),
                    "--fever_support_context_metrics_json",
                    str(fever_support_context),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads(output_json.read_text())
            report = output_report.read_text()
            self.assertEqual(summary["comparison_conclusion"], "move_to_w1")
            self.assertTrue(summary["move_to_w1"])
            self.assertFalse(summary["stop_after_w0"])
            self.assertTrue(summary["all_tasks_completed"])
            self.assertTrue(summary["all_support_context_wired"])
            self.assertTrue(summary["all_support_only_context_free"])
            self.assertTrue(summary["geometry_move_any"])
            self.assertTrue(summary["nonfever_positive_delta_any"])
            self.assertIn("comparison_conclusion: move_to_w1", report)

    def test_update_writer_circuit_opening_summary_prefers_p1a(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_writer_circuit_opening_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def write_metrics(name: str, payload: dict[str, object]) -> Path:
                path = tmp_path / name
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return path

            def task_metrics(
                *,
                benchmark_id: str,
                task_name: str,
                task_score: float,
                exact_match: float,
                delta_answer_logprob: float,
                source_grad: float,
                receiver_grad: float,
                receiver_enabled: bool,
                prefix_attention_by_layer: dict[str, float],
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": task_name,
                    "task_metric_name": "accuracy",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": exact_match,
                    "delta_answer_logprob": delta_answer_logprob,
                    "train_grad_norm_source_stub_steps_1_4_median": source_grad,
                    "train_grad_norm_receiver_lora_steps_1_4_median": receiver_grad,
                    "pilot_receiver_lora_enabled": receiver_enabled,
                    "prefix_attention_mass_mean": (
                        sum(prefix_attention_by_layer.values()) / max(1, len(prefix_attention_by_layer))
                    ),
                    "prefix_to_content_attention_ratio_mean": 0.05,
                    "prefix_attention_mass_mean_by_layer": prefix_attention_by_layer,
                    "prefix_attention_nontrivial_layer_count": sum(
                        1 for value in prefix_attention_by_layer.values() if value > 1e-3
                    ),
                    "projected_memory_effective_rank": 2.0,
                    "memory_long_common_mode_energy_ratio": 0.8,
                }

            control: dict[str, Path] = {}
            p1a: dict[str, Path] = {}
            p2a: dict[str, Path] = {}
            for task_name, benchmark_id in (
                ("gsm8k", "gsm8k"),
                ("narrativeqa", "narrativeqa"),
                ("fever", "fever"),
            ):
                control[task_name] = write_metrics(
                    f"{task_name}-control.json",
                    task_metrics(
                        benchmark_id=benchmark_id,
                        task_name=task_name,
                        task_score=0.2,
                        exact_match=0.2,
                        delta_answer_logprob=0.0,
                        source_grad=0.0,
                        receiver_grad=0.0,
                        receiver_enabled=False,
                        prefix_attention_by_layer={},
                    ),
                )
                p1a[task_name] = write_metrics(
                    f"{task_name}-p1a.json",
                    task_metrics(
                        benchmark_id=benchmark_id,
                        task_name=task_name,
                        task_score=0.25 if benchmark_id != "fever" else 0.2,
                        exact_match=0.25 if benchmark_id != "fever" else 0.2,
                        delta_answer_logprob=0.08 if benchmark_id != "fever" else 0.0,
                        source_grad=1e-4 if benchmark_id != "fever" else 1e-5,
                        receiver_grad=0.0,
                        receiver_enabled=False,
                        prefix_attention_by_layer={"0": 0.02, "1": 0.01} if benchmark_id != "fever" else {"0": 0.0},
                    ),
                )
                p2a[task_name] = write_metrics(
                    f"{task_name}-p2a.json",
                    task_metrics(
                        benchmark_id=benchmark_id,
                        task_name=task_name,
                        task_score=0.22,
                        exact_match=0.22,
                        delta_answer_logprob=0.01,
                        source_grad=1e-4,
                        receiver_grad=1e-4,
                        receiver_enabled=True,
                        prefix_attention_by_layer={"0": 0.01},
                    ),
                )

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--gsm8k_control_metrics_json",
                    str(control["gsm8k"]),
                    "--gsm8k_p1a_metrics_json",
                    str(p1a["gsm8k"]),
                    "--gsm8k_p2a_metrics_json",
                    str(p2a["gsm8k"]),
                    "--narrativeqa_control_metrics_json",
                    str(control["narrativeqa"]),
                    "--narrativeqa_p1a_metrics_json",
                    str(p1a["narrativeqa"]),
                    "--narrativeqa_p2a_metrics_json",
                    str(p2a["narrativeqa"]),
                    "--fever_control_metrics_json",
                    str(control["fever"]),
                    "--fever_p1a_metrics_json",
                    str(p1a["fever"]),
                    "--fever_p2a_metrics_json",
                    str(p2a["fever"]),
                    "--output_json",
                    str(tmp_path / "summary.json"),
                    "--output_report",
                    str(tmp_path / "summary.md"),
                ],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads((tmp_path / "summary.json").read_text())
            self.assertEqual(summary["comparison_conclusion"], "move_to_p1b")
            self.assertEqual(summary["recommended_substrate"], "p1a_source_stub_no_lora")
            self.assertTrue(summary["move_to_p1b"])
            self.assertFalse(summary["move_to_p2b"])

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_two_level_runtime_accepts_single_level_warm_start_but_rejects_strict_load(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0, "1": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0, "1": 1.0},
                }

            def to(self, *_args, **_kwargs):
                return self

        mock_backbone_cls.return_value = FakeBackbone()
        single_level_config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {"writer": {"memory_slots": 8, "arch": "mlp", "hidden_dim": 12}},
            "runtime": {
                "device": "cpu",
                "pilot_memory_path_variant": "single_level",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }
        two_level_config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {"memory_slots": 8, "arch": "mlp", "hidden_dim": 12},
                "reader": {"num_queries": 4, "num_heads": 2, "query_residual_scale": 1.0},
                "fuser": {"short_slots": 4, "arch": "resampler", "num_heads": 2},
            },
            "runtime": {
                "device": "cpu",
                "pilot_memory_path_variant": "two_level",
                "pilot_reader_context_mode": "prompt_summary",
                "pilot_projector_token_source": "short_slots",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }
        single_level_runtime = SharedInjectionPilotRuntime(
            config=single_level_config,
            seed=5,
            arm="injected",
            writer_memory_control="real",
        )
        two_level_runtime = SharedInjectionPilotRuntime(
            config=two_level_config,
            seed=7,
            arm="injected",
            writer_memory_control="real",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "single_level_checkpoint.pt"
            torch.save(
                {
                    "writer_state": single_level_runtime.writer.state_dict(),
                    "support_encoder_state": None,
                    "reader_state": None,
                    "fuser_state": None,
                    "prefix_projector_state": single_level_runtime.prefix_projector.state_dict(),
                    "pilot_memory_path_variant": "single_level",
                    "pilot_support_encoder_mode": "pooled_block",
                    "pilot_injection_mode": "sparse_deep_prefix",
                    "pilot_deep_prefix_layers": [0, 1],
                    "backbone_hidden_size": 6,
                    "writer_memory_slots": 8,
                    "pilot_projector_prefix_tokens": 8,
                },
                checkpoint_path,
            )
            two_level_runtime.warm_start_from_injection_checkpoint(checkpoint_path)
            with self.assertRaisesRegex(ValueError, "pilot_memory_path_variant"):
                two_level_runtime.load_injection_checkpoint(checkpoint_path)

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_two_level_runtime_threads_reader_modes(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0, "1": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0, "1": 1.0},
                }

            def to(self, *_args, **_kwargs):
                return self

        mock_backbone_cls.return_value = FakeBackbone()
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {"memory_slots": 8, "arch": "mlp", "hidden_dim": 12},
                "reader": {
                    "num_queries": 4,
                    "num_heads": 2,
                    "conditioning_mode": "none",
                    "attention_mode": "masked_partition",
                    "masked_partition": [[0, 1], [2, 3], [4, 5], [6, 7]],
                    "query_residual_scale": 1.0,
                },
                "fuser": {"short_slots": 4, "arch": "linear", "num_heads": 2},
            },
            "runtime": {
                "device": "cpu",
                "pilot_memory_path_variant": "two_level",
                "pilot_reader_context_mode": "prompt_summary",
                "pilot_projector_token_source": "short_slots",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }

        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=11,
            arm="injected",
            writer_memory_control="real",
        )

        self.assertIsNotNone(runtime.reader)
        assert runtime.reader is not None
        self.assertEqual(runtime.reader.conditioning_mode, "none")
        self.assertEqual(runtime.reader.attention_mode, "masked_partition")
        self.assertEqual(runtime.reader.masked_partition, ((0, 1), (2, 3), (4, 5), (6, 7)))

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_two_level_runtime_threads_writer_modes(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0, "1": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0, "1": 1.0},
                }

            def to(self, *_args, **_kwargs):
                return self

        mock_backbone_cls.return_value = FakeBackbone()
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {
                    "memory_slots": 8,
                    "arch": "transformer",
                    "num_heads": 2,
                    "transformer_layers": 1,
                    "slot_conditioning_mode": "slot_query_only",
                    "shared_state_scale": 0.02,
                },
                "reader": {"num_queries": 4, "num_heads": 2},
                "fuser": {"short_slots": 4, "arch": "linear", "num_heads": 2},
            },
            "runtime": {
                "device": "cpu",
                "pilot_memory_path_variant": "two_level",
                "pilot_reader_context_mode": "prompt_summary",
                "pilot_projector_token_source": "short_slots",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }

        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=13,
            arm="injected",
            writer_memory_control="real",
        )

        self.assertEqual(runtime.writer.slot_conditioning_mode, "slot_query_only")
        self.assertAlmostEqual(runtime.writer.shared_state_scale, 0.02)

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_two_level_runtime_threads_receiver_lora_modes(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"
                self.enable_calls: list[dict[str, object]] = []
                self.receiver_lora_trainable_enabled = False

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0, "1": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0, "1": 1.0},
                }

            def enable_receiver_micro_lora(self, **kwargs):
                self.enable_calls.append(dict(kwargs))

            def receiver_lora_parameter_count(self) -> int:
                return 24

            def set_receiver_lora_trainable(self, enabled: bool) -> None:
                self.receiver_lora_trainable_enabled = bool(enabled)

            def receiver_lora_state_dict(self):
                return None

            def load_receiver_lora_state_dict(self, *_args, **_kwargs):
                return None

            def validate_receiver_lora_state_dict(self, *_args, **_kwargs):
                return None

            def to(self, *_args, **_kwargs):
                return self

        fake_backbone = FakeBackbone()
        mock_backbone_cls.return_value = fake_backbone
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {"memory_slots": 8, "arch": "transformer", "num_heads": 2, "transformer_layers": 1},
                "reader": {"num_queries": 4, "num_heads": 2},
                "fuser": {"short_slots": 4, "arch": "linear", "num_heads": 2},
                "receiver_lora": {
                    "enabled": True,
                    "target_layers": [0, 1],
                    "target_modules": ["k_proj", "v_proj"],
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "pilot_memory_path_variant": "two_level",
                "pilot_reader_context_mode": "prompt_summary",
                "pilot_projector_token_source": "short_slots",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }

        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=13,
            arm="injected",
            writer_memory_control="real",
        )

        self.assertTrue(runtime.receiver_lora_enabled)
        self.assertEqual(runtime.receiver_lora_target_layers, (0, 1))
        self.assertEqual(runtime.receiver_lora_target_modules, ("k_proj", "v_proj"))
        self.assertEqual(runtime.receiver_lora_rank, 2)
        self.assertEqual(runtime.receiver_lora_trainable_params, 24)
        self.assertEqual(
            fake_backbone.enable_calls,
            [
                {
                    "layer_indices": (0, 1),
                    "target_modules": ("k_proj", "v_proj"),
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                }
            ],
        )
        self.assertTrue(fake_backbone.receiver_lora_trainable_enabled)

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_two_level_warm_start_allows_missing_receiver_lora_state(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0, "1": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0, "1": 1.0},
                }

            def enable_receiver_micro_lora(self, **_kwargs):
                return None

            def receiver_lora_parameter_count(self) -> int:
                return 24

            def set_receiver_lora_trainable(self, *_args, **_kwargs):
                return None

            def receiver_lora_state_dict(self):
                return {
                    "layers.0.self_attn.k_proj": {
                        "down.weight": torch.zeros(2, 6),
                        "up.weight": torch.zeros(6, 2),
                    }
                }

            def validate_receiver_lora_state_dict(self, *_args, **_kwargs):
                return None

            def load_receiver_lora_state_dict(self, *_args, **_kwargs):
                return None

            def to(self, *_args, **_kwargs):
                return self

        mock_backbone_cls.return_value = FakeBackbone()
        config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {"memory_slots": 8, "arch": "mlp", "hidden_dim": 12},
                "reader": {"num_queries": 4, "num_heads": 2, "query_residual_scale": 1.0},
                "fuser": {"short_slots": 4, "arch": "linear", "num_heads": 2},
                "receiver_lora": {
                    "enabled": True,
                    "target_layers": [0],
                    "target_modules": ["k_proj"],
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "pilot_memory_path_variant": "two_level",
                "pilot_reader_context_mode": "prompt_summary",
                "pilot_projector_token_source": "short_slots",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }
        runtime = SharedInjectionPilotRuntime(
            config=config,
            seed=17,
            arm="injected",
            writer_memory_control="real",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy_checkpoint.pt"
            torch.save(
                {
                    "writer_state": runtime.writer.state_dict(),
                    "support_encoder_state": None,
                    "reader_state": runtime.reader.state_dict(),
                    "fuser_state": runtime.fuser.state_dict(),
                    "prefix_projector_state": runtime.prefix_projector.state_dict(),
                    "pilot_memory_path_variant": "two_level",
                    "pilot_support_encoder_mode": "pooled_block",
                    "pilot_injection_mode": "sparse_deep_prefix",
                    "pilot_deep_prefix_layers": [0, 1],
                    "backbone_hidden_size": 6,
                    "writer_memory_slots": 8,
                    "pilot_projector_prefix_tokens": 4,
                },
                checkpoint_path,
            )

            runtime.warm_start_from_injection_checkpoint(checkpoint_path)
            with self.assertRaisesRegex(ValueError, "pilot_receiver_lora_enabled"):
                runtime.load_injection_checkpoint(checkpoint_path)

    @mock.patch("memtotal.training.m4_shared_injection.BackboneWrapper")
    def test_two_level_warm_start_supports_conditioning_mode_none(self, mock_backbone_cls) -> None:
        class FakeBackbone:
            def __init__(self) -> None:
                self.hidden_size = 6
                self.device = "cpu"

            def parameters(self):
                return []

            def summarize_texts(self, texts):
                return torch.ones(len(texts), self.hidden_size, dtype=torch.float32)

            def summarize_layer_prefix_projection(self, *_args, **_kwargs):
                return {
                    "layer_key_l2_by_layer": {"0": 1.0, "1": 1.0},
                    "layer_value_l2_by_layer": {"0": 1.0, "1": 1.0},
                }

            def to(self, *_args, **_kwargs):
                return self

        mock_backbone_cls.return_value = FakeBackbone()
        base_config = {
            "backbone": {"name": "fake", "load_mode": "stub", "dtype": "float32", "model_id": "fake/model"},
            "method": {
                "writer": {"memory_slots": 8, "arch": "mlp", "hidden_dim": 12},
                "reader": {"num_queries": 4, "num_heads": 2, "query_residual_scale": 1.0},
                "fuser": {"short_slots": 4, "arch": "linear", "num_heads": 2},
            },
            "runtime": {
                "device": "cpu",
                "pilot_memory_path_variant": "two_level",
                "pilot_reader_context_mode": "prompt_summary",
                "pilot_projector_token_source": "short_slots",
                "pilot_injection_mode": "sparse_deep_prefix",
                "pilot_deep_prefix_layers": [0, 1],
            },
        }
        target_config = json.loads(json.dumps(base_config))
        target_config["method"]["reader"]["conditioning_mode"] = "none"

        source_runtime = SharedInjectionPilotRuntime(
            config=base_config,
            seed=13,
            arm="injected",
            writer_memory_control="real",
        )
        target_runtime = SharedInjectionPilotRuntime(
            config=target_config,
            seed=17,
            arm="injected",
            writer_memory_control="real",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "two_level_checkpoint.pt"
            torch.save(
                {
                    "writer_state": source_runtime.writer.state_dict(),
                    "support_encoder_state": None,
                    "reader_state": source_runtime.reader.state_dict(),
                    "fuser_state": source_runtime.fuser.state_dict(),
                    "prefix_projector_state": source_runtime.prefix_projector.state_dict(),
                    "pilot_memory_path_variant": "two_level",
                    "pilot_support_encoder_mode": "pooled_block",
                    "pilot_injection_mode": "sparse_deep_prefix",
                    "pilot_deep_prefix_layers": [0, 1],
                    "pilot_reader_context_mode": "prompt_summary",
                    "pilot_projector_token_source": "short_slots",
                    "pilot_reader_num_queries": 4,
                    "pilot_fuser_short_slots": 4,
                    "backbone_hidden_size": 6,
                    "writer_memory_slots": 8,
                    "pilot_projector_prefix_tokens": 4,
                },
                checkpoint_path,
            )

            target_runtime.warm_start_from_injection_checkpoint(checkpoint_path)

        assert target_runtime.reader is not None
        assert source_runtime.reader is not None
        self.assertTrue(
            torch.allclose(
                target_runtime.reader.context_proj.weight,
                source_runtime.reader.context_proj.weight,
            )
        )


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

    def test_v0_forensics_summary_prefers_common_mode_domination(self) -> None:
        summary = _v0_forensics_summary(
            {
                "memory_long_top1_top2_ratio": 72.0,
                "memory_long_common_mode_energy_ratio": 0.82,
                "memory_long_centered_effective_rank": 3.4,
                "reader_value_projected_effective_rank": 1.3,
                "reader_value_projected_pairwise_cosine_mean": 0.98,
                "reader_readout_pairwise_cosine_mean": 0.99,
                "reader_readout_effective_rank": 1.2,
                "reader_readout_centered_effective_rank": 1.3,
                "train_reader_to_support_grad_ratio_steps_1_4_median": 0.05,
                "train_fuser_to_support_grad_ratio_steps_1_4_median": 0.05,
            }
        )
        self.assertFalse(summary["v0_value_diversity_gate_passed"])
        self.assertTrue(summary["v0_common_mode_domination_flag"])
        self.assertEqual(summary["v0_primary_bottleneck"], "common_mode_domination")

    def test_v0_forensics_summary_prefers_value_projected_homogenization(self) -> None:
        summary = _v0_forensics_summary(
            {
                "memory_long_top1_top2_ratio": 9.0,
                "memory_long_common_mode_energy_ratio": 0.25,
                "memory_long_centered_effective_rank": 3.2,
                "reader_value_projected_effective_rank": 1.4,
                "reader_value_projected_pairwise_cosine_mean": 0.97,
                "reader_readout_pairwise_cosine_mean": 0.98,
                "reader_readout_effective_rank": 1.5,
                "reader_readout_centered_effective_rank": 1.4,
                "train_reader_to_support_grad_ratio_steps_1_4_median": 0.7,
                "train_fuser_to_support_grad_ratio_steps_1_4_median": 0.6,
            }
        )
        self.assertFalse(summary["v0_common_mode_domination_flag"])
        self.assertTrue(summary["v0_value_projected_homogenization_flag"])
        self.assertEqual(summary["v0_primary_bottleneck"], "value_projected_homogenization")

    def test_v0_forensics_summary_prefers_receiver_starvation(self) -> None:
        summary = _v0_forensics_summary(
            {
                "memory_long_top1_top2_ratio": 8.0,
                "memory_long_common_mode_energy_ratio": 0.20,
                "memory_long_centered_effective_rank": 3.5,
                "reader_value_projected_effective_rank": 2.5,
                "reader_value_projected_pairwise_cosine_mean": 0.80,
                "reader_readout_pairwise_cosine_mean": 0.93,
                "reader_readout_effective_rank": 2.2,
                "reader_readout_centered_effective_rank": 2.1,
                "train_reader_to_support_grad_ratio_steps_1_4_median": 0.10,
                "train_fuser_to_support_grad_ratio_steps_1_4_median": 0.12,
            }
        )
        self.assertTrue(summary["v0_value_diversity_gate_passed"])
        self.assertTrue(summary["v0_receiver_starvation_flag"])
        self.assertEqual(summary["v0_primary_bottleneck"], "receiver_starvation")

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
            with (output_dir / "prefix_norm_drift.csv").open() as handle:
                fieldnames = list(next(iter(csv.DictReader(handle))).keys())
            self.assertIn("reader_context_overwrite_ratio", fieldnames)
            self.assertIn("fuser_output_effective_rank", fieldnames)
            self.assertIn("memory_long_slot_energy_cv", fieldnames)

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
            run_metrics_path.write_text(
                json.dumps(
                    {
                        "pilot_prefix_total_max_norm": 192.0,
                        "memory_long_top1_top2_ratio": 42.0,
                        "memory_long_common_mode_energy_ratio": 0.71,
                        "memory_long_centered_effective_rank": 3.1,
                        "reader_value_projected_effective_rank": 1.4,
                        "reader_value_projected_pairwise_cosine_mean": 0.98,
                        "reader_readout_pairwise_cosine_mean": 0.99,
                        "reader_readout_effective_rank": 1.2,
                        "reader_readout_centered_effective_rank": 1.3,
                        "train_reader_to_support_grad_ratio_steps_1_4_median": 0.2,
                        "train_fuser_to_support_grad_ratio_steps_1_4_median": 0.2,
                    }
                )
            )
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
            self.assertEqual(summary["v0_primary_bottleneck"], "common_mode_domination")

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

    def test_compare_m5_alignment_runs_reports_success_ambiguous_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            success = compare_m5_alignment_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_success.json",
                        {
                            "selection_passed": True,
                            "selected_step": 16,
                            "screen248_test_gate_passed": True,
                            "support_bank_brittle": False,
                            "fixed64_report_generated": True,
                            "fixed64_gate_passed": False,
                        },
                    )
                ),
                freeze_writer_summary_json=str(
                    write_payload(
                        "freeze_success.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
                pooled_block_summary_json=str(
                    write_payload(
                        "pooled_success.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
            )
            self.assertEqual(success["comparison_conclusion"], "success")
            self.assertTrue(success["alignment_claim_supported"])

            ambiguous = compare_m5_alignment_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_ambiguous.json",
                        {
                            "selection_passed": True,
                            "selected_step": 16,
                            "screen248_test_gate_passed": True,
                            "support_bank_brittle": False,
                        },
                    )
                ),
                freeze_writer_summary_json=str(
                    write_payload(
                        "freeze_ambiguous.json",
                        {"selection_passed": True, "selected_step": 8, "screen248_test_gate_passed": True},
                    )
                ),
                pooled_block_summary_json=str(
                    write_payload(
                        "pooled_ambiguous.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
            )
            self.assertEqual(ambiguous["comparison_conclusion"], "ambiguous_pass")
            self.assertFalse(ambiguous["alignment_claim_supported"])

            failure = compare_m5_alignment_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_failure.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "support_bank_brittle": False,
                        },
                    )
                ),
                freeze_writer_summary_json=str(
                    write_payload(
                        "freeze_failure.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
                pooled_block_summary_json=str(
                    write_payload(
                        "pooled_failure.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
            )
            self.assertEqual(failure["comparison_conclusion"], "failure")
            self.assertEqual(failure["failure_reason"], "canonical_failed_selection")

    def test_compare_m5_objective_runs_reports_success_optional_teacher_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            success = compare_m5_objective_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_success.json",
                        {
                            "selection_passed": True,
                            "selected_step": 8,
                            "screen248_test_gate_passed": True,
                            "support_bank_brittle": False,
                            "fixed64_report_generated": True,
                            "fixed64_gate_passed": False,
                        },
                    )
                ),
                anchor_only_summary_json=str(
                    write_payload(
                        "anchor_success.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
                task_only_control_summary_json=str(
                    write_payload(
                        "task_success.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
            )
            self.assertEqual(success["comparison_conclusion"], "success")
            self.assertTrue(success["objective_rewrite_supported"])
            self.assertTrue(success["teacher_margin_increment_supported"])

            optional_teacher = compare_m5_objective_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_optional.json",
                        {
                            "selection_passed": True,
                            "selected_step": 8,
                            "screen248_test_gate_passed": True,
                            "support_bank_brittle": False,
                        },
                    )
                ),
                anchor_only_summary_json=str(
                    write_payload(
                        "anchor_optional.json",
                        {"selection_passed": True, "selected_step": 8, "screen248_test_gate_passed": True},
                    )
                ),
                task_only_control_summary_json=str(
                    write_payload(
                        "task_optional.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
            )
            self.assertEqual(optional_teacher["comparison_conclusion"], "anchor_supported_teacher_optional")
            self.assertTrue(optional_teacher["objective_rewrite_supported"])
            self.assertFalse(optional_teacher["teacher_margin_increment_supported"])

            failure = compare_m5_objective_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_failure.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "support_bank_brittle": False,
                        },
                    )
                ),
                anchor_only_summary_json=str(
                    write_payload(
                        "anchor_failure.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
                task_only_control_summary_json=str(
                    write_payload(
                        "task_failure.json",
                        {"selection_passed": False, "selected_step": None, "screen248_test_gate_passed": False},
                    )
                ),
            )
            self.assertEqual(failure["comparison_conclusion"], "failure")
            self.assertEqual(failure["failure_reason"], "canonical_failed_selection")

    def test_compare_m5_dense_teacher_runs_reports_success_informative_and_hinge_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            success = compare_m5_dense_teacher_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_success.json",
                        {
                            "selection_passed": True,
                            "selected_step": 12,
                            "screen248_test_gate_passed": True,
                            "support_bank_brittle": False,
                            "fixed64_report_generated": True,
                            "fixed64_gate_passed": False,
                        },
                    )
                ),
                control_summary_json=str(
                    write_payload(
                        "control_failure.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 8,
                        },
                    )
                ),
            )
            self.assertEqual(success["comparison_conclusion"], "success")

            informative = compare_m5_dense_teacher_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_informative.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 16,
                            "cap_saturation_onset_step": None,
                        },
                    )
                ),
                control_summary_json=str(
                    write_payload(
                        "control_informative.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 8,
                            "cap_saturation_onset_step": 12,
                        },
                    )
                ),
            )
            self.assertEqual(informative["comparison_conclusion"], "informative_success")
            self.assertTrue(informative["informative_success"])

            hinge_off = compare_m5_dense_teacher_runs(
                canonical_summary_json=str(
                    write_payload(
                        "canonical_hinge_off.json",
                        {
                            "selection_passed": True,
                            "selected_step": 16,
                            "screen248_test_gate_passed": False,
                            "support_bank_brittle": False,
                        },
                    )
                ),
                control_summary_json=str(
                    write_payload(
                        "control_hinge_off.json",
                        {
                            "selection_passed": True,
                            "selected_step": 16,
                            "screen248_test_gate_passed": False,
                        },
                    )
                ),
                hinge_off_audit_summary_json=str(
                    write_payload(
                        "hinge_off.json",
                        {
                            "selection_passed": True,
                            "selected_step": 16,
                            "screen248_test_gate_passed": True,
                        },
                    )
                ),
            )
            self.assertEqual(hinge_off["comparison_conclusion"], "hinge_off_better")

    def test_compare_tl_poc_runs_reports_strong_success_when_bridge_bottleneck_and_specialization_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_reader_csv(name: str, rows: list[dict[str, object]]) -> Path:
                path = root / name
                fieldnames = sorted({key for row in rows for key in row})
                with path.open("w") as handle:
                    handle.write(",".join(fieldnames) + "\n")
                    for row in rows:
                        handle.write(",".join(str(row.get(field, "")) for field in fieldnames) + "\n")
                return path

            summary = compare_tl_poc_runs(
                sl8_summary_json=str(
                    write_payload(
                        "sl8.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 8,
                            "cap_saturation_onset_step": 8,
                        },
                    )
                ),
                tl_h4_k8_summary_json=str(
                    write_payload(
                        "h4k8.json",
                        {
                            "selection_passed": True,
                            "selected_step": 8,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 16,
                            "cap_saturation_onset_step": 16,
                        },
                    )
                ),
                tl_h4_k4_summary_json=str(
                    write_payload(
                        "h4k4.json",
                        {
                            "selection_passed": True,
                            "selected_step": 8,
                            "screen248_test_gate_passed": True,
                            "dominant_label_collapse_onset_step": 24,
                            "cap_saturation_onset_step": 24,
                        },
                    )
                ),
                tl_h1_k4_summary_json=str(
                    write_payload(
                        "h1k4.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 12,
                            "cap_saturation_onset_step": 12,
                        },
                    )
                ),
                tl_h4_k8_reader_query_csv=str(
                    write_reader_csv(
                        "h4k8_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 1.0, "reader_slot_coverage_fraction": 0.5},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 1, "reader_attention_entropy": 1.1, "reader_slot_coverage_fraction": 0.5},
                        ],
                    )
                ),
                tl_h4_k4_reader_query_csv=str(
                    write_reader_csv(
                        "h4k4_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 0.9, "reader_slot_coverage_fraction": 0.5},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 1, "reader_attention_entropy": 0.8, "reader_slot_coverage_fraction": 0.5},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 2, "reader_attention_entropy": 0.7, "reader_slot_coverage_fraction": 0.5},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 3, "reader_attention_entropy": 0.6, "reader_slot_coverage_fraction": 0.5},
                        ],
                    )
                ),
                tl_h1_k4_reader_query_csv=str(
                    write_reader_csv(
                        "h1k4_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 1.2, "reader_slot_coverage_fraction": 0.5},
                        ],
                    )
                ),
            )
            self.assertEqual(summary["comparison_conclusion"], "strong_success")
            self.assertTrue(summary["bridge_supported"])
            self.assertTrue(summary["bottleneck_supported"])
            self.assertTrue(summary["specialization_supported"])

    def test_compare_tl_bridge_rescue_runs_reports_informative_when_rescue_improves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_reader_csv(name: str, rows: list[dict[str, object]]) -> Path:
                path = root / name
                fieldnames = sorted({key for row in rows for key in row})
                with path.open("w") as handle:
                    handle.write(",".join(fieldnames) + "\n")
                    for row in rows:
                        handle.write(",".join(str(row.get(field, "")) for field in fieldnames) + "\n")
                return path

            summary = compare_tl_bridge_rescue_runs(
                sl8_summary_json=str(
                    write_payload(
                        "sl8.json",
                        {
                            "selection_passed": True,
                            "selected_step": 2,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 8,
                            "cap_saturation_onset_step": None,
                        },
                    )
                ),
                tl_h4_k8_summary_json=str(
                    write_payload(
                        "h4k8.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 2,
                            "cap_saturation_onset_step": None,
                        },
                    )
                ),
                tl_h4_k8_rescue_summary_json=str(
                    write_payload(
                        "h4k8_rescue.json",
                        {
                            "selection_passed": True,
                            "selected_step": 4,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 8,
                            "cap_saturation_onset_step": None,
                        },
                    )
                ),
                tl_h4_k8_reader_query_csv=str(
                    write_reader_csv(
                        "h4k8_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.0},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.0},
                        ],
                    )
                ),
                tl_h4_k8_rescue_reader_query_csv=str(
                    write_reader_csv(
                        "h4k8_rescue_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 1.5},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 1, "reader_attention_entropy": 1.4},
                        ],
                    )
                ),
            )
            self.assertEqual(summary["comparison_conclusion"], "informative")
            self.assertTrue(summary["rescue_selection_improved"])
            self.assertTrue(summary["rescue_reader_specialization_improved"])

    def test_compare_tl_slot_basis_runs_reports_informative_when_geometry_improves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_reader_csv(name: str, rows: list[dict[str, object]]) -> Path:
                path = root / name
                fieldnames = sorted({key for row in rows for key in row})
                with path.open("w") as handle:
                    handle.write(",".join(fieldnames) + "\n")
                    for row in rows:
                        handle.write(",".join(str(row.get(field, "")) for field in fieldnames) + "\n")
                return path

            def write_train_events(name: str, event: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps({"events": [event]}))
                return path

            summary = compare_tl_slot_basis_runs(
                sl8_summary_json=str(
                    write_payload(
                        "sl8.json",
                        {
                            "selection_passed": True,
                            "selected_step": 2,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 8,
                            "cap_saturation_onset_step": None,
                        },
                    )
                ),
                tl_h4_k8_summary_json=str(
                    write_payload(
                        "h4k8.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 2,
                            "cap_saturation_onset_step": None,
                        },
                    )
                ),
                tl_bridge_rescue_summary_json=str(
                    write_payload(
                        "bridge.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 2,
                            "cap_saturation_onset_step": None,
                        },
                    )
                ),
                tl_slot_basis_summary_json=str(
                    write_payload(
                        "basis.json",
                        {
                            "selection_passed": False,
                            "selected_step": None,
                            "screen248_test_gate_passed": False,
                            "dominant_label_collapse_onset_step": 4,
                            "cap_saturation_onset_step": None,
                        },
                    )
                ),
                tl_h4_k8_reader_query_csv=str(
                    write_reader_csv(
                        "h4k8_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.1},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.1},
                        ],
                    )
                ),
                tl_bridge_rescue_reader_query_csv=str(
                    write_reader_csv(
                        "bridge_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.0},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 1, "reader_attention_entropy": 2.0},
                        ],
                    )
                ),
                tl_slot_basis_reader_query_csv=str(
                    write_reader_csv(
                        "basis_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 1.6},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 1, "reader_attention_entropy": 1.5},
                        ],
                    )
                ),
                tl_h4_k8_train_events_json=str(
                    write_train_events(
                        "h4k8_events.json",
                        {
                            "memory_long_effective_rank": 1.0,
                            "memory_short_effective_rank": 1.1,
                            "reader_attention_pairwise_cosine_mean": 1.0,
                            "writer_slot_basis_pairwise_cosine_mean": 0.8,
                        },
                    )
                ),
                tl_bridge_rescue_train_events_json=str(
                    write_train_events(
                        "bridge_events.json",
                        {
                            "memory_long_effective_rank": 1.0,
                            "memory_short_effective_rank": 1.2,
                            "reader_attention_pairwise_cosine_mean": 1.0,
                            "writer_slot_basis_pairwise_cosine_mean": 0.7,
                        },
                    )
                ),
                tl_slot_basis_train_events_json=str(
                    write_train_events(
                        "basis_events.json",
                        {
                            "memory_long_effective_rank": 1.3,
                            "memory_short_effective_rank": 1.4,
                            "reader_attention_pairwise_cosine_mean": 0.6,
                            "writer_slot_basis_pairwise_cosine_mean": 0.1,
                        },
                    )
                ),
            )
            self.assertEqual(summary["comparison_conclusion"], "success")
            self.assertTrue(summary["basis_geometry_improved"])
            self.assertTrue(summary["basis_reader_specialization_improved"])

    def test_compare_tl_reader_geometry_runs_prefers_rg1a_when_ctx_off_moves_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_reader_csv(name: str, rows: list[dict[str, object]]) -> Path:
                path = root / name
                fieldnames = sorted({key for row in rows for key in row})
                with path.open("w") as handle:
                    handle.write(",".join(fieldnames) + "\n")
                    for row in rows:
                        handle.write(",".join(str(row.get(field, "")) for field in fieldnames) + "\n")
                return path

            def write_train_events(name: str, event: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps({"events": [event]}))
                return path

            baseline_summary = write_payload("baseline.json", {"selection_passed": False, "selected_step": None})
            baseline_reader = write_reader_csv(
                "baseline_reader.csv",
                [
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.08, "reader_slot_coverage_fraction": 0.2},
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.08, "reader_slot_coverage_fraction": 0.2},
                ],
            )
            baseline_events = write_train_events(
                "baseline_events.json",
                {
                    "memory_short_effective_rank": 1.2,
                    "reader_attention_pairwise_cosine_mean": 1.0,
                    "reader_context_overwrite_ratio": 12.0,
                    "reader_readout_effective_rank": 1.05,
                    "fuser_output_effective_rank": 1.2,
                },
            )
            rg1a_summary = write_payload("rg1a.json", {"selection_passed": True, "selected_step": 8})
            rg1a_reader = write_reader_csv(
                "rg1a_reader.csv",
                [
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 1.95, "reader_slot_coverage_fraction": 0.3},
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 1, "reader_attention_entropy": 1.92, "reader_slot_coverage_fraction": 0.3},
                ],
            )
            rg1a_events = write_train_events(
                "rg1a_events.json",
                {
                    "memory_short_effective_rank": 1.6,
                    "reader_attention_pairwise_cosine_mean": 0.9,
                    "reader_context_overwrite_ratio": 0.0,
                    "reader_readout_effective_rank": 1.4,
                    "fuser_output_effective_rank": 1.5,
                },
            )
            rg1b_summary = write_payload("rg1b.json", {"selection_passed": False, "selected_step": None})
            rg1b_reader = write_reader_csv(
                "rg1b_reader.csv",
                [
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.06, "reader_slot_coverage_fraction": 0.2},
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.05, "reader_slot_coverage_fraction": 0.2},
                ],
            )
            rg1b_events = write_train_events(
                "rg1b_events.json",
                {
                    "memory_short_effective_rank": 1.25,
                    "reader_attention_pairwise_cosine_mean": 0.98,
                    "reader_context_overwrite_ratio": 0.0,
                    "reader_readout_effective_rank": 1.08,
                    "fuser_output_effective_rank": 1.22,
                },
            )
            rg1c_summary = write_payload("rg1c.json", {"selection_passed": False, "selected_step": None})
            rg1c_reader = write_reader_csv(
                "rg1c_reader.csv",
                [
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.04, "reader_slot_coverage_fraction": 0.2},
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.03, "reader_slot_coverage_fraction": 0.2},
                ],
            )
            rg1c_events = write_train_events(
                "rg1c_events.json",
                {
                    "memory_short_effective_rank": 1.22,
                    "reader_attention_pairwise_cosine_mean": 0.97,
                    "reader_context_overwrite_ratio": 0.0,
                    "reader_readout_effective_rank": 1.10,
                    "fuser_output_effective_rank": 1.25,
                },
            )

            summary = compare_tl_reader_geometry_runs(
                baseline_summary_json=str(baseline_summary),
                rg1a_summary_json=str(rg1a_summary),
                rg1b_summary_json=str(rg1b_summary),
                rg1c_summary_json=str(rg1c_summary),
                baseline_reader_query_csv=str(baseline_reader),
                rg1a_reader_query_csv=str(rg1a_reader),
                rg1b_reader_query_csv=str(rg1b_reader),
                rg1c_reader_query_csv=str(rg1c_reader),
                baseline_train_events_json=str(baseline_events),
                rg1a_train_events_json=str(rg1a_events),
                rg1b_train_events_json=str(rg1b_events),
                rg1c_train_events_json=str(rg1c_events),
            )
            self.assertEqual(summary["comparison_conclusion"], "informative")
            self.assertEqual(summary["primary_interpretation"], "B-1a_context_overwrite")
            self.assertTrue(summary["context_overwrite_supported"])
            self.assertTrue(summary["rg1a_meaningful_movement"])
            self.assertEqual(summary["recommended_control_arm"], "rg1a_ctxoff_h4_k8")

    def test_compare_tl_reader_geometry_runs_moves_to_rg2_when_no_probe_moves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_reader_csv(name: str, rows: list[dict[str, object]]) -> Path:
                path = root / name
                fieldnames = sorted({key for row in rows for key in row})
                with path.open("w") as handle:
                    handle.write(",".join(fieldnames) + "\n")
                    for row in rows:
                        handle.write(",".join(str(row.get(field, "")) for field in fieldnames) + "\n")
                return path

            def write_train_events(name: str, event: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps({"events": [event]}))
                return path

            baseline_summary = write_payload("baseline.json", {"selection_passed": False, "selected_step": None})
            baseline_reader = write_reader_csv(
                "baseline_reader.csv",
                [
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.08},
                    {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.08},
                ],
            )
            baseline_events = write_train_events(
                "baseline_events.json",
                {
                    "memory_short_effective_rank": 1.2,
                    "reader_attention_pairwise_cosine_mean": 1.0,
                    "reader_context_overwrite_ratio": 12.0,
                    "reader_readout_effective_rank": 1.05,
                    "fuser_output_effective_rank": 1.2,
                },
            )

            def same_probe(name: str) -> tuple[Path, Path, Path]:
                return (
                    write_payload(f"{name}.json", {"selection_passed": False, "selected_step": None}),
                    write_reader_csv(
                        f"{name}_reader.csv",
                        [
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.07},
                            {"arm_alias": "I_real", "step": 8, "example_id": "1", "reader_argmax_slot": 0, "reader_attention_entropy": 2.07},
                        ],
                    ),
                    write_train_events(
                        f"{name}_events.json",
                        {
                            "memory_short_effective_rank": 1.22,
                            "reader_attention_pairwise_cosine_mean": 0.98,
                            "reader_context_overwrite_ratio": 0.0,
                            "reader_readout_effective_rank": 1.06,
                            "fuser_output_effective_rank": 1.21,
                        },
                    ),
                )

            rg1a_summary, rg1a_reader, rg1a_events = same_probe("rg1a")
            rg1b_summary, rg1b_reader, rg1b_events = same_probe("rg1b")
            rg1c_summary, rg1c_reader, rg1c_events = same_probe("rg1c")

            summary = compare_tl_reader_geometry_runs(
                baseline_summary_json=str(baseline_summary),
                rg1a_summary_json=str(rg1a_summary),
                rg1b_summary_json=str(rg1b_summary),
                rg1c_summary_json=str(rg1c_summary),
                baseline_reader_query_csv=str(baseline_reader),
                rg1a_reader_query_csv=str(rg1a_reader),
                rg1b_reader_query_csv=str(rg1b_reader),
                rg1c_reader_query_csv=str(rg1c_reader),
                baseline_train_events_json=str(baseline_events),
                rg1a_train_events_json=str(rg1a_events),
                rg1b_train_events_json=str(rg1b_events),
                rg1c_train_events_json=str(rg1c_events),
            )
            self.assertEqual(summary["comparison_conclusion"], "failure")
            self.assertTrue(summary["move_to_rg2"])
            self.assertEqual(summary["primary_interpretation"], "B-1_undetermined_move_to_rg2")

    def test_compare_tl_reader_rg2_runs_marks_competitive_geometry_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_train_events(name: str, events: list[dict[str, object]]) -> Path:
                path = root / name
                path.write_text(json.dumps({"events": events}))
                return path

            control_summary = write_payload(
                "control.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 0},
            )
            control_events = write_train_events(
                "control_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 1.0,
                        "reader_attention_entropy_mean": 2.08,
                        "memory_short_effective_rank": 1.3,
                        "memory_short_pairwise_cosine_mean": 0.995,
                        "reader_readout_effective_rank": 1.1,
                        "fuser_output_effective_rank": 1.3,
                    }
                ],
            )
            competitive_summary = write_payload(
                "competitive.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 12},
            )
            competitive_events = write_train_events(
                "competitive_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.82,
                        "reader_attention_entropy_mean": 1.82,
                        "memory_short_effective_rank": 1.95,
                        "memory_short_pairwise_cosine_mean": 0.92,
                        "reader_readout_effective_rank": 1.5,
                        "fuser_output_effective_rank": 1.9,
                    }
                ],
            )
            partition_summary = write_payload(
                "partition.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 6},
            )
            partition_events = write_train_events(
                "partition_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.91,
                        "reader_attention_entropy_mean": 1.98,
                        "memory_short_effective_rank": 1.75,
                        "memory_short_pairwise_cosine_mean": 0.97,
                        "reader_readout_effective_rank": 1.4,
                        "fuser_output_effective_rank": 1.7,
                    }
                ],
            )

            summary = compare_tl_reader_rg2_runs(
                control_summary_json=str(control_summary),
                competitive_summary_json=str(competitive_summary),
                partition_summary_json=str(partition_summary),
                control_train_events_json=str(control_events),
                competitive_train_events_json=str(competitive_events),
                partition_train_events_json=str(partition_events),
            )

            self.assertEqual(summary["comparison_conclusion"], "success")
            self.assertEqual(summary["primary_interpretation"], "rg2_competitive_geometry_alive")
            self.assertTrue(summary["competitive_geometry_alive"])
            self.assertTrue(summary["competitive_reader_supported"])
            self.assertTrue(summary["geometry_alive"])
            self.assertEqual(summary["bridge_failure_submode"], "B-1b_attention_symmetry_collapse")
            self.assertFalse(summary["move_to_rg3"])

    def test_compare_tl_reader_rg2_runs_moves_to_rg3_on_partition_only_partial_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_train_events(name: str, events: list[dict[str, object]]) -> Path:
                path = root / name
                path.write_text(json.dumps({"events": events}))
                return path

            control_summary = write_payload(
                "control.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 0},
            )
            control_events = write_train_events(
                "control_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 1.0,
                        "reader_attention_entropy_mean": 2.08,
                        "memory_short_effective_rank": 1.3,
                        "memory_short_pairwise_cosine_mean": 0.995,
                        "reader_readout_effective_rank": 1.1,
                        "fuser_output_effective_rank": 1.3,
                    }
                ],
            )
            competitive_summary = write_payload(
                "competitive.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 0},
            )
            competitive_events = write_train_events(
                "competitive_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.99,
                        "reader_attention_entropy_mean": 2.07,
                        "memory_short_effective_rank": 1.35,
                        "memory_short_pairwise_cosine_mean": 0.994,
                        "reader_readout_effective_rank": 1.12,
                        "fuser_output_effective_rank": 1.33,
                    }
                ],
            )
            partition_summary = write_payload(
                "partition.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 10},
            )
            partition_events = write_train_events(
                "partition_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.93,
                        "reader_attention_entropy_mean": 2.01,
                        "memory_short_effective_rank": 1.75,
                        "memory_short_pairwise_cosine_mean": 0.97,
                        "reader_readout_effective_rank": 1.35,
                        "fuser_output_effective_rank": 1.7,
                    }
                ],
            )

            summary = compare_tl_reader_rg2_runs(
                control_summary_json=str(control_summary),
                competitive_summary_json=str(competitive_summary),
                partition_summary_json=str(partition_summary),
                control_train_events_json=str(control_events),
                competitive_train_events_json=str(competitive_events),
                partition_train_events_json=str(partition_events),
            )

            self.assertEqual(summary["comparison_conclusion"], "informative")
            self.assertEqual(summary["primary_interpretation"], "rg2_partition_only_partial_gain")
            self.assertTrue(summary["partition_partial_gain"])
            self.assertTrue(summary["partition_reader_supported"])
            self.assertFalse(summary["geometry_alive"])
            self.assertEqual(summary["bridge_failure_submode"], "B-1b_attention_symmetry_collapse")
            self.assertTrue(summary["move_to_rg3"])

    def test_compare_tl_reader_rg3_runs_marks_bootstrap_geometry_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_train_events(name: str, events: list[dict[str, object]]) -> Path:
                path = root / name
                path.write_text(json.dumps({"events": events}))
                return path

            control_summary = write_payload(
                "control.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 0},
            )
            control_events = write_train_events(
                "control_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.0,
                        "reader_attention_entropy_mean": 0.69,
                        "memory_short_effective_rank": 3.98,
                        "memory_short_pairwise_cosine_mean": 0.01,
                        "reader_readout_effective_rank": 1.23,
                        "fuser_output_effective_rank": 3.98,
                    }
                ],
            )
            bootstrap_summary = write_payload(
                "bootstrap.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 12},
            )
            bootstrap_events = write_train_events(
                "bootstrap_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.0,
                        "reader_attention_entropy_mean": 0.69,
                        "memory_short_effective_rank": 3.99,
                        "memory_short_pairwise_cosine_mean": 0.01,
                        "reader_readout_effective_rank": 1.30,
                        "fuser_output_effective_rank": 3.99,
                    }
                ],
            )
            bootstrap_reconstruction_summary = write_payload(
                "bootstrap_reconstruction.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 6},
            )
            bootstrap_reconstruction_events = write_train_events(
                "bootstrap_reconstruction_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.0,
                        "reader_attention_entropy_mean": 0.69,
                        "memory_short_effective_rank": 3.99,
                        "memory_short_pairwise_cosine_mean": 0.01,
                        "reader_readout_effective_rank": 1.28,
                        "fuser_output_effective_rank": 3.99,
                    }
                ],
            )

            summary = compare_tl_reader_rg3_runs(
                control_summary_json=str(control_summary),
                bootstrap_summary_json=str(bootstrap_summary),
                bootstrap_reconstruction_summary_json=str(bootstrap_reconstruction_summary),
                control_train_events_json=str(control_events),
                bootstrap_train_events_json=str(bootstrap_events),
                bootstrap_reconstruction_train_events_json=str(bootstrap_reconstruction_events),
            )

            self.assertEqual(summary["comparison_conclusion"], "success")
            self.assertEqual(summary["primary_interpretation"], "rg3_bootstrap_geometry_alive")
            self.assertTrue(summary["bootstrap_geometry_alive"])
            self.assertTrue(summary["geometry_alive"])
            self.assertEqual(summary["recommended_arm"], "bootstrap_only")
            self.assertTrue(summary["move_to_rg4"])
            self.assertEqual(summary["final_classification"], "B-1_resolved_geometry_alive")
            self.assertFalse(summary["stop_after_rg3"])

    def test_compare_tl_reader_rg3_runs_marks_bootstrap_reconstruction_partial_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            def write_train_events(name: str, events: list[dict[str, object]]) -> Path:
                path = root / name
                path.write_text(json.dumps({"events": events}))
                return path

            control_summary = write_payload(
                "control.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 0},
            )
            control_events = write_train_events(
                "control_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.0,
                        "reader_attention_entropy_mean": 0.69,
                        "memory_short_effective_rank": 3.98,
                        "memory_short_pairwise_cosine_mean": 0.01,
                        "reader_readout_effective_rank": 1.23,
                        "fuser_output_effective_rank": 3.98,
                    }
                ],
            )
            bootstrap_summary = write_payload(
                "bootstrap.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 2},
            )
            bootstrap_events = write_train_events(
                "bootstrap_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.0,
                        "reader_attention_entropy_mean": 0.69,
                        "memory_short_effective_rank": 3.98,
                        "memory_short_pairwise_cosine_mean": 0.01,
                        "reader_readout_effective_rank": 1.23,
                        "fuser_output_effective_rank": 3.98,
                    }
                ],
            )
            bootstrap_reconstruction_summary = write_payload(
                "bootstrap_reconstruction.json",
                {"selection_passed": False, "selected_step": None, "dominant_label_collapse_onset_step": 6},
            )
            bootstrap_reconstruction_events = write_train_events(
                "bootstrap_reconstruction_events.json",
                [
                    {
                        "step": 8,
                        "memory_short_slots": 4,
                        "reader_attention_pairwise_cosine_mean": 0.0,
                        "reader_attention_entropy_mean": 0.69,
                        "memory_short_effective_rank": 3.98,
                        "memory_short_pairwise_cosine_mean": 0.01,
                        "reader_readout_effective_rank": 1.31,
                        "fuser_output_effective_rank": 3.98,
                    }
                ],
            )

            summary = compare_tl_reader_rg3_runs(
                control_summary_json=str(control_summary),
                bootstrap_summary_json=str(bootstrap_summary),
                bootstrap_reconstruction_summary_json=str(bootstrap_reconstruction_summary),
                control_train_events_json=str(control_events),
                bootstrap_train_events_json=str(bootstrap_events),
                bootstrap_reconstruction_train_events_json=str(bootstrap_reconstruction_events),
            )

            self.assertEqual(summary["comparison_conclusion"], "informative")
            self.assertEqual(
                summary["primary_interpretation"],
                "rg3_bootstrap_reconstruction_only_partial_gain",
            )
            self.assertFalse(summary["geometry_alive"])
            self.assertTrue(summary["bootstrap_reconstruction_partial_gain"])
            self.assertEqual(summary["recommended_arm"], "bootstrap_reconstruction")
            self.assertFalse(summary["move_to_rg4"])
            self.assertEqual(summary["final_classification"], "B-2_candidate_downstream_flattening")
            self.assertTrue(summary["stop_after_rg3"])

    def test_compare_tl_writer_value_runs_prefers_slot_query_only_on_strong_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_metrics(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            control_metrics = write_metrics(
                "control_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "shared_add",
                    "pilot_writer_shared_state_scale": 1.0,
                    "best_adapt_task_score": 0.32,
                    "best_adapt_macro_f1": 0.29,
                    "dominant_label_fraction": 0.98,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.40},
                        {"step": 2, "dominant_label_fraction": 0.91},
                    ],
                    "memory_long_top1_top2_ratio": 60.0,
                    "memory_long_common_mode_energy_ratio": 0.82,
                    "memory_long_centered_effective_rank": 4.0,
                    "reader_value_projected_effective_rank": 1.5,
                    "reader_value_projected_pairwise_cosine_mean": 0.97,
                    "reader_readout_effective_rank": 1.1,
                    "reader_readout_centered_effective_rank": 1.0,
                    "reader_readout_pairwise_cosine_mean": 0.99,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )
            shared_scaled_metrics = write_metrics(
                "shared_scaled_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "shared_add_scaled",
                    "pilot_writer_shared_state_scale": 0.02,
                    "best_adapt_task_score": 0.38,
                    "best_adapt_macro_f1": 0.35,
                    "dominant_label_fraction": 0.88,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.35},
                        {"step": 2, "dominant_label_fraction": 0.60},
                        {"step": 4, "dominant_label_fraction": 0.88},
                    ],
                    "memory_long_top1_top2_ratio": 18.0,
                    "memory_long_common_mode_energy_ratio": 0.45,
                    "memory_long_centered_effective_rank": 3.2,
                    "reader_value_projected_effective_rank": 2.0,
                    "reader_value_projected_pairwise_cosine_mean": 0.92,
                    "reader_readout_effective_rank": 2.1,
                    "reader_readout_centered_effective_rank": 1.9,
                    "reader_readout_pairwise_cosine_mean": 0.94,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )
            slot_query_only_metrics = write_metrics(
                "slot_query_only_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "slot_query_only",
                    "pilot_writer_shared_state_scale": 1.0,
                    "best_adapt_task_score": 0.50,
                    "best_adapt_macro_f1": 0.47,
                    "dominant_label_fraction": 0.55,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.33},
                        {"step": 2, "dominant_label_fraction": 0.42},
                        {"step": 4, "dominant_label_fraction": 0.48},
                    ],
                    "memory_long_top1_top2_ratio": 8.0,
                    "memory_long_common_mode_energy_ratio": 0.24,
                    "memory_long_centered_effective_rank": 3.5,
                    "reader_value_projected_effective_rank": 2.6,
                    "reader_value_projected_pairwise_cosine_mean": 0.81,
                    "reader_readout_effective_rank": 2.8,
                    "reader_readout_centered_effective_rank": 2.3,
                    "reader_readout_pairwise_cosine_mean": 0.87,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )

            summary = compare_tl_writer_value_runs(
                control_metrics_json=str(control_metrics),
                shared_scaled_metrics_json=str(shared_scaled_metrics),
                slot_query_only_metrics_json=str(slot_query_only_metrics),
            )

            self.assertEqual(summary["recommended_arm"], "slot_query_only")
            self.assertEqual(summary["comparison_conclusion"], "strong_success")
            self.assertTrue(summary["move_to_v2"])
            self.assertFalse(summary["move_to_v1_penalties"])

    def test_compare_tl_writer_value_runs_moves_to_penalties_on_partial_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_metrics(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            control_metrics = write_metrics(
                "control_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "shared_add",
                    "pilot_writer_shared_state_scale": 1.0,
                    "best_adapt_task_score": 0.30,
                    "best_adapt_macro_f1": 0.27,
                    "dominant_label_fraction": 0.97,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.44},
                        {"step": 2, "dominant_label_fraction": 0.92},
                    ],
                    "memory_long_top1_top2_ratio": 60.0,
                    "memory_long_common_mode_energy_ratio": 0.81,
                    "memory_long_centered_effective_rank": 4.0,
                    "reader_value_projected_effective_rank": 1.6,
                    "reader_value_projected_pairwise_cosine_mean": 0.96,
                    "reader_readout_effective_rank": 1.2,
                    "reader_readout_centered_effective_rank": 1.0,
                    "reader_readout_pairwise_cosine_mean": 0.99,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )
            shared_scaled_metrics = write_metrics(
                "shared_scaled_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "shared_add_scaled",
                    "pilot_writer_shared_state_scale": 0.02,
                    "best_adapt_task_score": 0.34,
                    "best_adapt_macro_f1": 0.30,
                    "dominant_label_fraction": 0.91,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.40},
                        {"step": 2, "dominant_label_fraction": 0.70},
                        {"step": 4, "dominant_label_fraction": 0.91},
                    ],
                    "memory_long_top1_top2_ratio": 20.0,
                    "memory_long_common_mode_energy_ratio": 0.50,
                    "memory_long_centered_effective_rank": 3.0,
                    "reader_value_projected_effective_rank": 1.9,
                    "reader_value_projected_pairwise_cosine_mean": 0.93,
                    "reader_readout_effective_rank": 1.6,
                    "reader_readout_centered_effective_rank": 1.5,
                    "reader_readout_pairwise_cosine_mean": 0.965,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )
            slot_query_only_metrics = write_metrics(
                "slot_query_only_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "slot_query_only",
                    "pilot_writer_shared_state_scale": 1.0,
                    "best_adapt_task_score": 0.28,
                    "best_adapt_macro_f1": 0.24,
                    "dominant_label_fraction": 0.95,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.45},
                        {"step": 2, "dominant_label_fraction": 0.93},
                    ],
                    "memory_long_top1_top2_ratio": 32.0,
                    "memory_long_common_mode_energy_ratio": 0.66,
                    "memory_long_centered_effective_rank": 2.8,
                    "reader_value_projected_effective_rank": 1.7,
                    "reader_value_projected_pairwise_cosine_mean": 0.95,
                    "reader_readout_effective_rank": 1.3,
                    "reader_readout_centered_effective_rank": 1.2,
                    "reader_readout_pairwise_cosine_mean": 0.985,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )

            summary = compare_tl_writer_value_runs(
                control_metrics_json=str(control_metrics),
                shared_scaled_metrics_json=str(shared_scaled_metrics),
                slot_query_only_metrics_json=str(slot_query_only_metrics),
            )

            self.assertEqual(summary["recommended_arm"], "shared_add_scaled")
            self.assertEqual(summary["comparison_conclusion"], "needs_penalty_refinement")
            self.assertFalse(summary["move_to_v2"])
            self.assertTrue(summary["move_to_v1_penalties"])

    def test_compare_tl_writer_value_runs_hard_stop_still_moves_to_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_metrics(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            control_metrics = write_metrics(
                "control_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "shared_add",
                    "pilot_writer_shared_state_scale": 1.0,
                    "best_adapt_task_score": 0.30,
                    "best_adapt_macro_f1": 0.25,
                    "dominant_label_fraction": 0.97,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.44},
                        {"step": 2, "dominant_label_fraction": 0.92},
                    ],
                    "memory_long_top1_top2_ratio": 70.0,
                    "memory_long_common_mode_energy_ratio": 0.99,
                    "memory_long_centered_effective_rank": 6.8,
                    "reader_value_projected_effective_rank": 1.5,
                    "reader_value_projected_pairwise_cosine_mean": 0.97,
                    "reader_readout_effective_rank": 1.2,
                    "reader_readout_centered_effective_rank": 1.1,
                    "reader_readout_pairwise_cosine_mean": 0.999,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )
            shared_scaled_metrics = write_metrics(
                "shared_scaled_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "shared_add_scaled",
                    "pilot_writer_shared_state_scale": 0.02,
                    "best_adapt_task_score": 0.29,
                    "best_adapt_macro_f1": 0.24,
                    "dominant_label_fraction": 0.96,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.44},
                        {"step": 2, "dominant_label_fraction": 0.91},
                    ],
                    "memory_long_top1_top2_ratio": 71.0,
                    "memory_long_common_mode_energy_ratio": 0.99,
                    "memory_long_centered_effective_rank": 6.7,
                    "reader_value_projected_effective_rank": 1.5,
                    "reader_value_projected_pairwise_cosine_mean": 0.97,
                    "reader_readout_effective_rank": 1.19,
                    "reader_readout_centered_effective_rank": 1.1,
                    "reader_readout_pairwise_cosine_mean": 0.9991,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )
            slot_query_only_metrics = write_metrics(
                "slot_query_only_metrics.json",
                {
                    "pilot_writer_slot_conditioning_mode": "slot_query_only",
                    "pilot_writer_shared_state_scale": 1.0,
                    "best_adapt_task_score": 0.28,
                    "best_adapt_macro_f1": 0.23,
                    "dominant_label_fraction": 0.96,
                    "snapshot_metrics": [
                        {"step": 0, "dominant_label_fraction": 0.45},
                        {"step": 2, "dominant_label_fraction": 0.92},
                    ],
                    "memory_long_top1_top2_ratio": 69.5,
                    "memory_long_common_mode_energy_ratio": 0.99,
                    "memory_long_centered_effective_rank": 6.8,
                    "reader_value_projected_effective_rank": 1.5,
                    "reader_value_projected_pairwise_cosine_mean": 0.97,
                    "reader_readout_effective_rank": 1.21,
                    "reader_readout_centered_effective_rank": 1.1,
                    "reader_readout_pairwise_cosine_mean": 0.9989,
                    "train_reader_to_support_grad_ratio_steps_1_4_median": 1.0,
                    "train_fuser_to_support_grad_ratio_steps_1_4_median": 1.0,
                },
            )

            summary = compare_tl_writer_value_runs(
                control_metrics_json=str(control_metrics),
                shared_scaled_metrics_json=str(shared_scaled_metrics),
                slot_query_only_metrics_json=str(slot_query_only_metrics),
            )

            self.assertEqual(summary["comparison_conclusion"], "failure")
            self.assertTrue(summary["move_to_v2"])
            self.assertFalse(summary["move_to_v1_penalties"])
            self.assertTrue(summary["stop_after_v1_architecture"])
            self.assertEqual(summary["recommended_arm"], "control")

    def test_compare_tl_micro_lora_runs_prefers_late3_on_strong_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            control_metrics = write_payload(
                "control_metrics.json",
                {
                    "pilot_receiver_lora_enabled": False,
                    "best_adapt_task_score": 0.30,
                    "best_adapt_macro_f1": 0.25,
                    "dominant_label_fraction": 0.97,
                    "memory_long_top1_top2_ratio": 70.0,
                    "reader_readout_effective_rank": 1.2,
                    "reader_readout_pairwise_cosine_mean": 0.999,
                    "train_grad_norm_reader_steps_1_4_median": 0.01,
                    "train_grad_norm_fuser_steps_1_4_median": 0.02,
                },
            )
            control_run_summary = write_payload(
                "control_run_summary.json",
                {
                    "dominant_label_collapse_onset_step": 0,
                    "selection_passed": False,
                    "screen248_test_gate_passed": False,
                    "fixed64_gate_passed": False,
                },
            )
            late3_metrics = write_payload(
                "late3_metrics.json",
                {
                    "pilot_receiver_lora_enabled": True,
                    "pilot_receiver_lora_rank": 2,
                    "pilot_receiver_lora_alpha": 4.0,
                    "pilot_receiver_lora_dropout": 0.0,
                    "pilot_receiver_lora_target_layers": [14, 21, 27],
                    "pilot_receiver_lora_target_modules": ["k_proj", "v_proj"],
                    "pilot_receiver_lora_trainable_params": 18432,
                    "best_adapt_task_score": 0.42,
                    "best_adapt_macro_f1": 0.36,
                    "dominant_label_fraction": 0.80,
                    "memory_long_top1_top2_ratio": 70.0,
                    "reader_readout_effective_rank": 1.2,
                    "reader_readout_pairwise_cosine_mean": 0.999,
                    "train_grad_norm_reader_steps_1_4_median": 0.20,
                    "train_grad_norm_fuser_steps_1_4_median": 0.30,
                    "train_grad_norm_receiver_lora_steps_1_4_median": 0.05,
                },
            )
            late3_run_summary = write_payload(
                "late3_run_summary.json",
                {
                    "dominant_label_collapse_onset_step": 4,
                    "selection_passed": True,
                    "screen248_test_gate_passed": True,
                    "fixed64_gate_passed": False,
                },
            )

            summary = compare_tl_micro_lora_runs(
                control_metrics_json=str(control_metrics),
                control_run_summary_json=str(control_run_summary),
                late3_metrics_json=str(late3_metrics),
                late3_run_summary_json=str(late3_run_summary),
            )

            self.assertEqual(summary["comparison_conclusion"], "strong_success")
            self.assertEqual(summary["recommended_arm"], "micro_lora_r2_late3")
            self.assertTrue(summary["move_to_v3"])
            self.assertFalse(summary["continue_to_l2"])

    def test_compare_tl_micro_lora_runs_requests_l2_on_partial_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def write_payload(name: str, payload: dict[str, object]) -> Path:
                path = root / name
                path.write_text(json.dumps(payload))
                return path

            control_metrics = write_payload(
                "control_metrics.json",
                {
                    "pilot_receiver_lora_enabled": False,
                    "best_adapt_task_score": 0.30,
                    "best_adapt_macro_f1": 0.25,
                    "dominant_label_fraction": 0.97,
                    "memory_long_top1_top2_ratio": 70.0,
                    "reader_readout_effective_rank": 1.2,
                    "reader_readout_pairwise_cosine_mean": 0.999,
                    "train_grad_norm_reader_steps_1_4_median": 0.01,
                    "train_grad_norm_fuser_steps_1_4_median": 0.02,
                },
            )
            control_run_summary = write_payload(
                "control_run_summary.json",
                {
                    "dominant_label_collapse_onset_step": 0,
                    "selection_passed": False,
                    "screen248_test_gate_passed": False,
                    "fixed64_gate_passed": False,
                },
            )
            late3_metrics = write_payload(
                "late3_metrics.json",
                {
                    "pilot_receiver_lora_enabled": True,
                    "pilot_receiver_lora_rank": 2,
                    "pilot_receiver_lora_alpha": 4.0,
                    "pilot_receiver_lora_dropout": 0.0,
                    "pilot_receiver_lora_target_layers": [14, 21, 27],
                    "pilot_receiver_lora_target_modules": ["k_proj", "v_proj"],
                    "pilot_receiver_lora_trainable_params": 18432,
                    "best_adapt_task_score": 0.33,
                    "best_adapt_macro_f1": 0.28,
                    "dominant_label_fraction": 0.95,
                    "memory_long_top1_top2_ratio": 70.0,
                    "reader_readout_effective_rank": 1.2,
                    "reader_readout_pairwise_cosine_mean": 0.999,
                    "train_grad_norm_reader_steps_1_4_median": 0.04,
                    "train_grad_norm_fuser_steps_1_4_median": 0.09,
                    "train_grad_norm_receiver_lora_steps_1_4_median": 0.02,
                },
            )
            late3_run_summary = write_payload(
                "late3_run_summary.json",
                {
                    "dominant_label_collapse_onset_step": 2,
                    "selection_passed": False,
                    "screen248_test_gate_passed": False,
                    "fixed64_gate_passed": False,
                },
            )

            summary = compare_tl_micro_lora_runs(
                control_metrics_json=str(control_metrics),
                control_run_summary_json=str(control_run_summary),
                late3_metrics_json=str(late3_metrics),
                late3_run_summary_json=str(late3_run_summary),
            )

            self.assertEqual(summary["comparison_conclusion"], "expand_to_l2")
            self.assertTrue(summary["continue_to_l2"])
            self.assertFalse(summary["move_to_v4"])


if __name__ == "__main__":
    unittest.main()
