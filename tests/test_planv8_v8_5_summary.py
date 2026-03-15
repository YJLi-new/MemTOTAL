from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_5_summary import ALL_TASKS, ARM_ORDER, build_summary


class PlanV8V85SummaryTest(unittest.TestCase):
    def _write_v84_reference(
        self,
        path: Path,
        *,
        best_arm_id: str = "w2_ext3layer64_lr2e5",
        interface_family: str = "ri0_legacy_prefix",
        next_step: str = "open_v8_5_bridge",
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": best_arm_id,
                    "base_for_v8_5_arm_id": best_arm_id,
                    "selected_interface_family_for_v8_5": interface_family,
                    "recommended_next_step": next_step,
                }
            )
            + "\n"
        )

    def _write_task(
        self,
        result_root: Path,
        arm_id: str,
        task_name: str,
        *,
        task_score: float,
        control_logprob: float,
        branch_logprob: float,
        writer_slots: int = 64,
        reader_queries: int = 0,
        short_slots: int = 0,
        memory_path_variant: str = "single_level",
        bridge_family: str = "BR0",
        reader_rank: float = 0.0,
        reader_grad: float = 0.0,
        fuser_grad: float = 0.0,
    ) -> None:
        task_root = result_root / arm_id / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        metrics = {
            "benchmark_id": task_name,
            "pilot_prompt_variant": {
                "gsm8k": "q3_gsm8k_nonthink",
                "triviaqa": "q3_trivia_nonthink",
                "fever": "answer_slot_labels",
            }[task_name],
            "best_adapt_task_score": float(task_score),
            "best_adapt_macro_f1": float(task_score),
            "pilot_memory_path_variant": memory_path_variant,
            "pilot_active_bridge_family": bridge_family,
            "writer_memory_slots": int(writer_slots),
            "pilot_reader_num_queries": int(reader_queries),
            "pilot_fuser_short_slots": int(short_slots),
            "reader_readout_effective_rank": float(reader_rank),
        }
        (task_root / "metrics.json").write_text(json.dumps(metrics) + "\n")
        (task_root / "train_events.json").write_text(
            json.dumps({"events": [{"loss": 1.0, "grad_norm_reader": reader_grad, "grad_norm_fuser": fuser_grad}]})
            + "\n"
        )
        case_rows = []
        for index in range(2):
            case_rows.append(
                {
                    "example_id": f"{task_name}-{index}",
                    "answer_logprob_with_memory": float(
                        branch_logprob if arm_id != "b0_no_bridge" else control_logprob
                    ),
                }
            )
        (task_root / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in case_rows) + "\n"
        )

    def _write_flat_phase(self, result_root: Path) -> None:
        control_scores = {"gsm8k": 0.62, "triviaqa": 0.20, "fever": 0.58}
        control_logprobs = {"gsm8k": -12.0, "triviaqa": -5.0, "fever": -1.5}
        for task_name in ALL_TASKS:
            self._write_task(
                result_root,
                "b0_no_bridge",
                task_name,
                task_score=control_scores[task_name],
                control_logprob=control_logprobs[task_name],
                branch_logprob=control_logprobs[task_name],
            )
        for arm_id in ARM_ORDER[1:]:
            for task_name in ALL_TASKS:
                self._write_task(
                    result_root,
                    arm_id,
                    task_name,
                    task_score=control_scores[task_name],
                    control_logprob=control_logprobs[task_name],
                    branch_logprob=control_logprobs[task_name],
                    reader_queries=16,
                    short_slots=16,
                    memory_path_variant="two_level",
                    bridge_family="BR1",
                    reader_rank=1.5,
                    reader_grad=0.2,
                    fuser_grad=0.2,
                )

    def test_build_summary_promotes_bridge_winner_into_v86(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v84_summary_path = Path(temp_dir) / "v8-4-summary.json"
            self._write_v84_reference(v84_summary_path)

            self._write_task(
                result_root,
                "b2_q32_s16",
                "gsm8k",
                task_score=0.70,
                control_logprob=-12.0,
                branch_logprob=-8.0,
                reader_queries=32,
                short_slots=16,
                memory_path_variant="two_level",
                bridge_family="BR2",
                reader_rank=2.0,
                reader_grad=0.3,
                fuser_grad=0.3,
            )
            self._write_task(
                result_root,
                "b2_q32_s16",
                "triviaqa",
                task_score=0.20,
                control_logprob=-5.0,
                branch_logprob=-4.0,
                reader_queries=32,
                short_slots=16,
                memory_path_variant="two_level",
                bridge_family="BR2",
                reader_rank=2.0,
                reader_grad=0.3,
                fuser_grad=0.3,
            )

            summary = build_summary(
                result_root=result_root,
                v84_summary_path=v84_summary_path,
            )

            self.assertEqual(summary["best_bridge_arm_id"], "b2_q32_s16")
            self.assertEqual(summary["best_arm_id"], "b2_q32_s16")
            self.assertTrue(summary["best_arm_acceptance_qualified"])
            self.assertEqual(summary["recommended_next_step"], "open_v8_6_writer_aux")
            self.assertEqual(
                summary["comparison_conclusion"],
                "bridge_compression_preserves_gains_open_v8_6_writer_aux",
            )
            self.assertEqual(summary["selected_bridge_family_for_v8_6"], "BR2")

    def test_build_summary_keeps_full_route_when_bridge_hurts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v84_summary_path = Path(temp_dir) / "v8-4-summary.json"
            self._write_v84_reference(v84_summary_path)

            summary = build_summary(
                result_root=result_root,
                v84_summary_path=v84_summary_path,
            )

            self.assertEqual(summary["best_arm_id"], "b0_no_bridge")
            self.assertFalse(summary["best_arm_acceptance_qualified"])
            self.assertEqual(summary["recommended_next_step"], "open_v8_6_writer_aux_full_route")
            self.assertEqual(
                summary["comparison_conclusion"],
                "bridge_compression_hurts_keep_full_route_open_v8_6_writer_aux_full_route",
            )
            self.assertEqual(summary["selected_bridge_family_for_v8_6"], "BR0")
            self.assertEqual(summary["base_for_v8_6_arm_id"], "b0_no_bridge")

    def test_build_summary_carries_v84_reference_into_v86_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v84_summary_path = Path(temp_dir) / "v8-4-summary.json"
            self._write_v84_reference(
                v84_summary_path,
                best_arm_id="w3_ext3layer96_lr1e5",
                interface_family="ri2_cross_attn",
            )

            summary = build_summary(
                result_root=result_root,
                v84_summary_path=v84_summary_path,
            )

            self.assertEqual(summary["v84_best_arm_id"], "w3_ext3layer96_lr1e5")
            self.assertEqual(summary["v84_base_for_v8_5_arm_id"], "w3_ext3layer96_lr1e5")
            self.assertEqual(summary["v84_selected_interface_family_for_v8_5"], "ri2_cross_attn")
            self.assertEqual(summary["selected_interface_family_for_v8_6"], "ri2_cross_attn")


if __name__ == "__main__":
    unittest.main()
