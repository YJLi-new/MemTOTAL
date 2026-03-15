from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.planv9_v9_0_config import materialize_planv9_v9_0_config


class PlanV9V90ConfigTest(unittest.TestCase):
    def test_materialize_precache_latent_arm_uses_layer16_chunked_cache_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_config = Path(tmpdir) / "a2.json"
            config = materialize_planv9_v9_0_config(
                arm_id="a2_precache_latent_oracle",
                output_config=output_config,
                support_path="/tmp/support.jsonl",
                train_path="/tmp/train.jsonl",
                eval_path="/tmp/eval.jsonl",
                selected_prompt_variant="q3_gsm8k_nonthink",
                primary_model_dir="/tmp/qwen34",
                primary_backbone_name="Qwen3-4B",
            )
            self.assertEqual(config["experiment"]["stage"], "V9-0")
            self.assertEqual(config["runtime"]["pilot_memory_consumer_mode"], "precache_latent")
            self.assertEqual(config["runtime"]["pilot_oracle_extract_layer"], 16)
            self.assertEqual(config["runtime"]["pilot_oracle_slot_cap"], 8)
            self.assertEqual(config["runtime"]["pilot_oracle_slot_pool_window"], 16)
            self.assertEqual(config["runtime"]["shared_injection_arm"], "injected")
            self.assertEqual(config["runtime"]["pilot_prompt_variant"], "q3_gsm8k_nonthink")
            self.assertEqual(config["backbone"]["model_id"], "/tmp/qwen34")
            self.assertTrue(output_config.exists())

    def test_materialize_legacy_prefix_arm_matches_v8_prefix_oracle_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_config = Path(tmpdir) / "a1.json"
            config = materialize_planv9_v9_0_config(
                arm_id="a1_legacy_prefix_oracle",
                output_config=output_config,
                support_path="/tmp/support.jsonl",
                train_path="/tmp/train.jsonl",
                eval_path="/tmp/eval.jsonl",
                selected_prompt_variant="q3_gsm8k_nonthink",
                primary_model_dir="/tmp/qwen34",
            )
            self.assertEqual(config["runtime"]["pilot_memory_consumer_mode"], "legacy_prefix")
            self.assertEqual(config["runtime"]["pilot_deep_prefix_layers"], [16, 17, 18, 19])
            self.assertEqual(config["runtime"]["pilot_oracle_extract_layer"], 18)
            self.assertEqual(config["runtime"]["pilot_oracle_slot_cap"], 16)

    def test_materialize_unknown_arm_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                materialize_planv9_v9_0_config(
                    arm_id="unknown",
                    output_config=Path(tmpdir) / "bad.json",
                    support_path="/tmp/support.jsonl",
                    train_path="/tmp/train.jsonl",
                    eval_path="/tmp/eval.jsonl",
                    selected_prompt_variant="q3_gsm8k_nonthink",
                    primary_model_dir="/tmp/qwen34",
                )


if __name__ == "__main__":
    unittest.main()
