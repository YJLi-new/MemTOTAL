from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.planv8_v8_2_config import materialize_planv8_v8_2_config


class PlanV8V82ConfigTest(unittest.TestCase):
    def _write_v81_reference(
        self,
        path: Path,
        *,
        interface_family: str,
        best_arm_id: str = "i0_prefix_legacy_r2",
        memory_slots: int = 16,
        reader_layers: list[int] | None = None,
        rank_label: str = "r2",
    ) -> None:
        if reader_layers is None:
            reader_layers = [16, 17, 18, 19]
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": best_arm_id,
                    "base_for_v8_2_arm_id": best_arm_id,
                    "selected_interface_family_for_v8_2": interface_family,
                    "arm_summaries": {
                        best_arm_id: {
                            "memory_slots": memory_slots,
                            "reader_layers": reader_layers,
                            "rank_label": rank_label,
                        }
                    },
                }
            )
            + "\n"
        )

    def _materialize(self, *, interface_family: str, arm_id: str = "r2_mid12_r64_lr1e4") -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v81_summary_path = root / "v8-1-summary.json"
            self._write_v81_reference(v81_summary_path, interface_family=interface_family)
            output_config = root / "config.json"
            return materialize_planv8_v8_2_config(
                task_name="gsm8k",
                arm_id=arm_id,
                prompt_variant="q3_gsm8k_nonthink",
                output_config=output_config,
                support_path=str(root / "support.jsonl"),
                train_path=str(root / "train.jsonl"),
                eval_path=str(root / "eval.jsonl"),
                primary_model_dir=str(root / "Qwen3-4B"),
                primary_backbone_name="Qwen3-4B",
                v81_summary_path=v81_summary_path,
            )

    def test_legacy_prefix_sweep_keeps_base_receiver_micro_lora_tiny(self) -> None:
        config = self._materialize(interface_family="ri0_legacy_prefix")

        self.assertEqual(config["runtime"]["pilot_memory_consumer_mode"], "legacy_prefix")
        self.assertEqual(config["runtime"]["pilot_deep_prefix_layers"], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        self.assertEqual(config["runtime"]["pilot_deep_prefix_rank"], 64)
        self.assertEqual(config["runtime"]["pilot_gradient_probe_modules"], ["projector", "receiver_lora"])
        self.assertEqual(config["method"]["receiver_lora"]["target_layers"], [16, 17, 18, 19])
        self.assertEqual(config["method"]["receiver_lora"]["rank"], 2)
        self.assertEqual(config["method"]["receiver_lora"]["alpha"], 4.0)

    def test_prepend_block_sweep_tracks_selected_layer_band_and_rank(self) -> None:
        config = self._materialize(interface_family="ri1_prepend_block", arm_id="r4_late8_r32_lr1e4")

        self.assertEqual(config["runtime"]["pilot_memory_consumer_mode"], "reader_lora_sequence")
        self.assertEqual(config["runtime"]["pilot_gradient_probe_modules"], ["receiver_lora"])
        self.assertEqual(config["method"]["receiver_lora"]["target_layers"], [20, 21, 22, 23, 24, 25, 26, 27])
        self.assertEqual(config["method"]["receiver_lora"]["rank"], 32)
        self.assertEqual(config["runtime"]["pilot_reader_cross_attn_layers"], [])

    def test_cross_attention_sweep_uses_ff_hidden_proxy_rank(self) -> None:
        config = self._materialize(interface_family="ri2_cross_attn", arm_id="r5_mid8_r16_lr5e5")

        self.assertEqual(config["runtime"]["pilot_memory_consumer_mode"], "reader_cross_attn")
        self.assertEqual(config["runtime"]["pilot_gradient_probe_modules"], ["reader_cross_attn"])
        self.assertEqual(config["runtime"]["pilot_reader_cross_attn_layers"], [14, 15, 16, 17, 18, 19, 20, 21])
        self.assertEqual(config["runtime"]["pilot_reader_cross_attn_ff_hidden_dim"], 1024)
        self.assertEqual(config["method"]["receiver_lora"]["enabled"], False)


if __name__ == "__main__":
    unittest.main()
