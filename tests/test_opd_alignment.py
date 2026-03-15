from __future__ import annotations

import unittest

import torch

from memtotal.training.m4_shared_injection import _build_opd_target_text, _opd_aux_loss


class OpdHintBuilderTest(unittest.TestCase):
    def test_gsm8k_answer_plus_rationale_uses_solution_prefix(self) -> None:
        target_text, diagnostics = _build_opd_target_text(
            example={
                "benchmark_id": "gsm8k",
                "gold_answer": "72",
                "solution": (
                    "Natalia sold 48/2 = <<48/2=24>>24 clips in May. "
                    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
                ),
            },
            hint_mode="answer_plus_rationale",
        )

        self.assertIsNotNone(target_text)
        self.assertIn("Answer: 72", target_text)
        self.assertIn("Rationale:", target_text)
        self.assertTrue(diagnostics["opd_target_context_available"])
        self.assertEqual(diagnostics["opd_hint_mode_effective"], "answer_plus_rationale")

    def test_triviaqa_answer_plus_evidence_falls_back_without_context(self) -> None:
        target_text, diagnostics = _build_opd_target_text(
            example={
                "benchmark_id": "triviaqa",
                "gold_answer": "York",
                "aliases": ["York", "York, England"],
                "evidence_sentences": [],
            },
            hint_mode="answer_plus_evidence",
        )

        self.assertEqual(target_text, "Answer: York")
        self.assertFalse(diagnostics["opd_target_context_available"])
        self.assertEqual(diagnostics["opd_hint_mode_effective"], "answer_only")

    def test_triviaqa_answer_plus_two_evidence_prefers_answer_matching_sentences(self) -> None:
        target_text, diagnostics = _build_opd_target_text(
            example={
                "benchmark_id": "triviaqa",
                "gold_answer": "Judi Dench",
                "aliases": ["Judi Dench", "Dame Judi Dench"],
                "evidence_sentences": [
                    "She trained at the Central School of Speech and Drama.",
                    "Dame Judi Dench was born in Heworth, York, England.",
                    "Judi Dench later appeared in many films.",
                ],
            },
            hint_mode="answer_plus_two_evidence",
        )

        self.assertIsNotNone(target_text)
        self.assertIn("Answer: Judi Dench", target_text)
        self.assertIn("Evidence:", target_text)
        self.assertIn("Dame Judi Dench was born in Heworth, York, England.", target_text)
        self.assertIn("Judi Dench later appeared in many films.", target_text)
        self.assertTrue(diagnostics["opd_target_context_available"])

    def test_fever_label_plus_evidence_ignores_placeholder(self) -> None:
        target_text, diagnostics = _build_opd_target_text(
            example={
                "benchmark_id": "fever",
                "gold_answer": "Supports",
                "evidence": "No gold evidence provided in this split.",
            },
            hint_mode="label_plus_evidence",
        )

        self.assertEqual(target_text, "Label: Supports")
        self.assertFalse(diagnostics["opd_target_context_available"])
        self.assertEqual(diagnostics["opd_hint_mode_effective"], "label_only")


class OpdAuxLossTest(unittest.TestCase):
    def test_opd_token_ce_uses_positive_teacher_advantage(self) -> None:
        loss, active, diagnostics = _opd_aux_loss(
            mode="opd_token_ce",
            active_target_score=torch.tensor(-5.0),
            base_target_score=torch.tensor(-8.0),
            teacher_target_score=torch.tensor(-6.0),
            target_token_count=5,
            center=0.0,
            scale=1.0,
            advantage_clip=5.0,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(active)
        self.assertAlmostEqual(float(loss.item()), 0.4, places=5)
        self.assertAlmostEqual(diagnostics["opd_mean_advantage"], 0.4, places=5)
        self.assertEqual(diagnostics["opd_positive_token_fraction"], 1.0)

    def test_opd_token_ce_zeroes_negative_teacher_advantage(self) -> None:
        loss, active, diagnostics = _opd_aux_loss(
            mode="opd_token_ce",
            active_target_score=torch.tensor(-5.0),
            base_target_score=torch.tensor(-6.0),
            teacher_target_score=torch.tensor(-8.0),
            target_token_count=5,
            center=0.0,
            scale=1.0,
            advantage_clip=5.0,
        )

        self.assertIsNotNone(loss)
        self.assertFalse(active)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertAlmostEqual(diagnostics["opd_mean_advantage"], 0.0, places=6)
        self.assertEqual(diagnostics["opd_positive_token_fraction"], 0.0)

    def test_opd_token_ce_centered_allows_negative_advantage(self) -> None:
        loss, active, diagnostics = _opd_aux_loss(
            mode="opd_token_ce_centered",
            active_target_score=torch.tensor(-5.0),
            base_target_score=torch.tensor(-6.0),
            teacher_target_score=torch.tensor(-8.0),
            target_token_count=5,
            center=0.0,
            scale=1.0,
            advantage_clip=5.0,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(active)
        self.assertAlmostEqual(float(loss.item()), -0.4, places=5)
        self.assertAlmostEqual(diagnostics["opd_mean_advantage"], -0.4, places=5)
        self.assertEqual(diagnostics["opd_positive_token_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
