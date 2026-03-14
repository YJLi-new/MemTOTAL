from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.planv8_v8_5_config import materialize_planv8_v8_5_config


class PlanV8V85ConfigTest(unittest.TestCase):
    def _write_base_config(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "experiment": {"name": "base", "stage": "V8-4", "method_variant": "w2_ext3layer64_lr2e5"},
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
                        "pilot_memory_path_variant": "single_level",
                        "pilot_projector_token_source": "writer_slots",
                        "pilot_memory_consumer_mode": "legacy_prefix",
                        "pilot_prefix_source_mode": "writer",
                        "pilot_trainable_variant": "writer_then_joint",
                        "pilot_writer_learning_rate": 2.0e-5,
                        "pilot_projector_learning_rate": 1.0e-4,
                    },
                    "task": {"benchmark_id": "gsm8k"},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _write_v84_summary(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": "w2_ext3layer64_lr2e5",
                    "base_for_v8_5_arm_id": "w2_ext3layer64_lr2e5",
                    "selected_interface_family_for_v8_5": "ri0_legacy_prefix",
                }
            )
            + "\n"
        )

    def test_materialize_bridge_arm_uses_two_level_reader_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            base_checkpoint = tmp / "checkpoint.pt"
            v84_summary = tmp / "v8-4-summary.json"
            output_config = tmp / "bridge.json"
            self._write_base_config(base_config)
            base_checkpoint.write_bytes(b"checkpoint")
            self._write_v84_summary(v84_summary)

            config = materialize_planv8_v8_5_config(
                base_config_path=base_config,
                base_checkpoint_path=base_checkpoint,
                arm_id="b2_q32_s16",
                output_config=output_config,
                v84_summary_path=v84_summary,
                primary_model_dir="/tmp/qwen34",
                primary_backbone_name="Qwen3-4B",
            )

            self.assertEqual(config["experiment"]["stage"], "V8-5")
            self.assertEqual(config["runtime"]["pilot_memory_path_variant"], "two_level")
            self.assertEqual(config["runtime"]["pilot_projector_token_source"], "short_slots")
            self.assertEqual(config["runtime"]["pilot_reader_num_queries"], 32)
            self.assertEqual(config["runtime"]["pilot_fuser_short_slots"], 16)
            self.assertEqual(config["runtime"]["pilot_trainable_variant"], "reader_only")
            self.assertEqual(config["runtime"]["pilot_active_bridge_family"], "BR2")
            self.assertEqual(config["runtime"]["pilot_v84_warm_start_status"], "full_from_v84")
            self.assertEqual(config["method"]["reader"]["num_queries"], 32)
            self.assertEqual(config["method"]["fuser"]["short_slots"], 16)

    def test_materialize_no_bridge_control_removes_reader_and_fuser(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            base_checkpoint = tmp / "checkpoint.pt"
            v84_summary = tmp / "v8-4-summary.json"
            output_config = tmp / "control.json"
            self._write_base_config(base_config)
            base_checkpoint.write_bytes(b"checkpoint")
            self._write_v84_summary(v84_summary)

            config = materialize_planv8_v8_5_config(
                base_config_path=base_config,
                base_checkpoint_path=base_checkpoint,
                arm_id="b0_no_bridge",
                output_config=output_config,
                v84_summary_path=v84_summary,
            )

            self.assertEqual(config["runtime"]["pilot_memory_path_variant"], "single_level")
            self.assertEqual(config["runtime"]["pilot_projector_token_source"], "writer_slots")
            self.assertNotIn("reader", config["method"])
            self.assertNotIn("fuser", config["method"])

    def test_materialize_x96_arm_cold_starts_when_base_slots_are_64(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            base_checkpoint = tmp / "checkpoint.pt"
            v84_summary = tmp / "v8-4-summary.json"
            output_config = tmp / "bridge96.json"
            self._write_base_config(base_config)
            base_checkpoint.write_bytes(b"checkpoint")
            self._write_v84_summary(v84_summary)

            config = materialize_planv8_v8_5_config(
                base_config_path=base_config,
                base_checkpoint_path=base_checkpoint,
                arm_id="b4_q48_s16_x96",
                output_config=output_config,
                v84_summary_path=v84_summary,
            )

            self.assertEqual(config["method"]["writer"]["memory_slots"], 96)
            self.assertEqual(config["runtime"]["pilot_init_checkpoint_path"], "")
            self.assertEqual(
                config["runtime"]["pilot_v84_warm_start_status"],
                "cold_start_writer_slot_mismatch",
            )


if __name__ == "__main__":
    unittest.main()
