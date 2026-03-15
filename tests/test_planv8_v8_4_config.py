from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.planv8_v8_4_config import materialize_planv8_v8_4_config


class PlanV8V84ConfigTest(unittest.TestCase):
    def _write_base_config(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "experiment": {"name": "base", "stage": "V8-3", "method_variant": "p4"},
                    "backbone": {"name": "Qwen3-4B", "model_id": "/tmp/model"},
                    "method": {
                        "writer": {
                            "memory_slots": 16,
                            "transformer_layers": 2,
                            "conditioning_layers": 1,
                            "hidden_dim": 128,
                            "num_heads": 4,
                            "dropout": 0.0,
                        },
                        "receiver_lora": {
                            "enabled": True,
                            "target_layers": [16, 17, 18, 19],
                            "target_modules": ["k_proj", "v_proj"],
                            "rank": 2,
                            "alpha": 4.0,
                            "dropout": 0.0,
                        },
                    },
                    "runtime": {
                        "pilot_trainable_variant": "reader_only",
                        "pilot_alignment_aux_mode": "opd_token_ce",
                        "pilot_opd_weight_max": 0.3,
                        "pilot_prefix_source_mode": "oracle_hidden_state_slots",
                        "pilot_memory_consumer_mode": "legacy_prefix",
                        "pilot_deep_prefix_layers": [14, 15, 16, 17, 18, 19, 20, 21],
                        "pilot_deep_prefix_rank": 32,
                    },
                    "task": {"benchmark_id": "gsm8k"},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _write_v83_summary(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": "p4_opd_ansplusctx_w03",
                    "base_for_v8_4_arm_id": "p4_opd_ansplusctx_w03",
                    "selected_interface_family_for_v8_4": "ri0_legacy_prefix",
                }
            )
            + "\n"
        )

    def test_materialize_writer_arm_uses_writer_then_joint_and_consumer_only_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            base_checkpoint = tmp / "checkpoint.pt"
            v83_summary = tmp / "v8-3-summary.json"
            output_config = tmp / "writer.json"
            self._write_base_config(base_config)
            base_checkpoint.write_bytes(b"checkpoint")
            self._write_v83_summary(v83_summary)

            config = materialize_planv8_v8_4_config(
                base_config_path=base_config,
                base_checkpoint_path=base_checkpoint,
                arm_id="w2_ext3layer64_lr2e5",
                output_config=output_config,
                v83_summary_path=v83_summary,
                primary_model_dir="/tmp/qwen34",
                primary_backbone_name="Qwen3-4B",
            )

            self.assertEqual(config["experiment"]["stage"], "V8-4")
            self.assertEqual(config["runtime"]["pilot_trainable_variant"], "writer_then_joint")
            self.assertEqual(config["runtime"]["pilot_init_checkpoint_mode"], "consumer_only")
            self.assertEqual(config["runtime"]["pilot_prefix_source_mode"], "writer")
            self.assertEqual(config["runtime"]["stage_a_steps"], 80)
            self.assertEqual(config["runtime"]["stage_b_steps"], 220)
            self.assertEqual(config["runtime"]["pilot_writer_learning_rate"], 2.0e-5)
            self.assertEqual(config["method"]["writer"]["memory_slots"], 64)
            self.assertEqual(config["method"]["writer"]["transformer_layers"], 3)

    def test_materialize_oracle_control_keeps_reader_only_and_oracle_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            base_checkpoint = tmp / "checkpoint.pt"
            v83_summary = tmp / "v8-3-summary.json"
            output_config = tmp / "control.json"
            self._write_base_config(base_config)
            base_checkpoint.write_bytes(b"checkpoint")
            self._write_v83_summary(v83_summary)

            config = materialize_planv8_v8_4_config(
                base_config_path=base_config,
                base_checkpoint_path=base_checkpoint,
                arm_id="w0_oracle64",
                output_config=output_config,
                v83_summary_path=v83_summary,
            )

            self.assertEqual(config["runtime"]["pilot_trainable_variant"], "reader_only")
            self.assertEqual(config["runtime"]["pilot_prefix_source_mode"], "oracle_hidden_state_slots")
            self.assertEqual(config["runtime"]["pilot_oracle_slot_cap"], 64)
            self.assertEqual(config["runtime"]["pilot_writer_learning_rate"], 0.0)
            self.assertEqual(config["method"]["writer"]["memory_slots"], 64)


if __name__ == "__main__":
    unittest.main()
