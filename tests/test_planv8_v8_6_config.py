from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.planv8_v8_6_config import materialize_planv8_v8_6_config


class PlanV8V86ConfigTest(unittest.TestCase):
    def _write_base_config(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "experiment": {"name": "base", "stage": "V8-5", "method_variant": "b2_q32_s16"},
                    "backbone": {"name": "Qwen3-4B", "model_id": "/tmp/model"},
                    "method": {
                        "writer": {
                            "memory_slots": 64,
                            "hidden_dim": 128,
                            "num_heads": 4,
                            "transformer_layers": 3,
                            "conditioning_layers": 2,
                            "dropout": 0.0,
                        },
                        "reader": {
                            "num_queries": 32,
                            "condition_on_context": True,
                            "conditioning_mode": "add",
                            "attention_mode": "standard",
                            "dropout": 0.05,
                            "query_residual_scale": 0.0,
                            "num_heads": 8,
                        },
                        "fuser": {
                            "arch": "resampler",
                            "hidden_dim": 1536,
                            "num_heads": 8,
                            "dropout": 0.05,
                            "short_slots": 16,
                        },
                    },
                    "runtime": {
                        "pilot_memory_path_variant": "two_level",
                        "pilot_projector_token_source": "short_slots",
                        "pilot_trainable_variant": "reader_only",
                        "pilot_reader_num_queries": 32,
                        "pilot_fuser_short_slots": 16,
                        "pilot_writer_learning_rate": 2.0e-5,
                    },
                    "task": {"benchmark_id": "gsm8k"},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _write_v85_summary(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": "b2_q32_s16",
                    "base_for_v8_6_arm_id": "b2_q32_s16",
                    "selected_interface_family_for_v8_6": "ri2_cross_attn",
                    "selected_bridge_family_for_v8_6": "BR2",
                }
            )
            + "\n"
        )

    def test_materialize_barlow_arm_restores_writer_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            base_checkpoint = tmp / "checkpoint.pt"
            v85_summary = tmp / "v8-5-summary.json"
            output_config = tmp / "barlow.json"
            self._write_base_config(base_config)
            base_checkpoint.write_bytes(b"checkpoint")
            self._write_v85_summary(v85_summary)

            config = materialize_planv8_v8_6_config(
                base_config_path=base_config,
                base_checkpoint_path=base_checkpoint,
                arm_id="a1_barlow",
                output_config=output_config,
                v85_summary_path=v85_summary,
                primary_model_dir="/tmp/qwen34",
                primary_backbone_name="Qwen3-4B",
            )

            self.assertEqual(config["experiment"]["stage"], "V8-6")
            self.assertEqual(config["runtime"]["pilot_trainable_variant"], "writer_then_joint")
            self.assertEqual(config["runtime"]["stage_a_steps"], 80)
            self.assertEqual(config["runtime"]["stage_b_steps"], 220)
            self.assertEqual(config["runtime"]["pilot_active_aux_family"], "barlow_lite")
            self.assertEqual(config["runtime"]["pilot_aux_loss_mode"], "barlow")
            self.assertEqual(config["runtime"]["pilot_aux_projection_dim"], 128)
            self.assertEqual(config["runtime"]["pilot_barlow_loss_weight"], 0.02)
            self.assertEqual(config["runtime"]["pilot_v85_selected_bridge_family"], "BR2")

    def test_materialize_control_turns_auxiliaries_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            base_checkpoint = tmp / "checkpoint.pt"
            v85_summary = tmp / "v8-5-summary.json"
            output_config = tmp / "control.json"
            self._write_base_config(base_config)
            base_checkpoint.write_bytes(b"checkpoint")
            self._write_v85_summary(v85_summary)

            config = materialize_planv8_v8_6_config(
                base_config_path=base_config,
                base_checkpoint_path=base_checkpoint,
                arm_id="a0_none",
                output_config=output_config,
                v85_summary_path=v85_summary,
            )

            self.assertEqual(config["runtime"]["pilot_aux_loss_mode"], "task_only")
            self.assertEqual(config["runtime"]["pilot_barlow_loss_weight"], 0.0)
            self.assertEqual(config["runtime"]["pilot_reconstruction_aux_mode"], "off")
            self.assertEqual(config["runtime"]["pilot_alignment_aux_mode"], "off")
            self.assertEqual(config["runtime"]["pilot_opd_weight_max"], 0.0)

    def test_materialize_opd_plus_recon_sets_contextual_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            base_checkpoint = tmp / "checkpoint.pt"
            v85_summary = tmp / "v8-5-summary.json"
            output_config = tmp / "hybrid.json"
            self._write_base_config(base_config)
            base_checkpoint.write_bytes(b"checkpoint")
            self._write_v85_summary(v85_summary)

            config = materialize_planv8_v8_6_config(
                base_config_path=base_config,
                base_checkpoint_path=base_checkpoint,
                arm_id="a5_writer_opd_plus_recon",
                output_config=output_config,
                v85_summary_path=v85_summary,
            )

            self.assertEqual(config["runtime"]["pilot_alignment_aux_mode"], "opd_token_ce")
            self.assertEqual(config["runtime"]["pilot_opd_weight_max"], 0.1)
            self.assertEqual(config["runtime"]["pilot_reconstruction_aux_mode"], "hashed_bow")
            self.assertEqual(config["runtime"]["pilot_reconstruction_aux_weight"], 0.02)
            self.assertEqual(config["runtime"]["pilot_opd_hint_mode_gsm8k"], "answer_plus_rationale")
            self.assertEqual(config["runtime"]["pilot_opd_hint_mode_triviaqa"], "answer_plus_evidence")
            self.assertEqual(config["runtime"]["pilot_opd_hint_mode_fever"], "label_plus_evidence")


if __name__ == "__main__":
    unittest.main()
