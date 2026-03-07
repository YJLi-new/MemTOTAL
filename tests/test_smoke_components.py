from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.data import load_toy_dataset
from memtotal.models import MemoryReader, MemoryWriter
from memtotal.pipeline import MemoryRuntime
from memtotal.training.run_train import main as train_main
from memtotal.utils.config import load_config
from memtotal.utils.repro import set_seed


class SmokeComponentTest(unittest.TestCase):
    def test_memory_pipeline_backward_pass(self) -> None:
        config = load_config(ROOT / "configs/exp/smoke_qwen25.yaml")
        dataset = load_toy_dataset(ROOT / config["task"]["dataset_path"])
        set_seed(123)
        runtime = MemoryRuntime(config=config, seed=123)

        forward = runtime.forward_example(dataset[0])
        loss = F.mse_loss(forward.predicted_state, forward.target_state)
        loss.backward()

        self.assertEqual(list(forward.memory_long.shape), [1, 8, 64])
        self.assertEqual(list(forward.memory_short.shape), [1, 2, 64])
        self.assertIsNotNone(runtime.reader.queries.grad)
        self.assertGreater(float(runtime.reader.queries.grad.norm().item()), 0.0)
        self.assertEqual(forward.segments, ["Question: 2 plus 3?", "Plan: add the two integers"])
        self.assertEqual(len(forward.segment_stats), 2)
        self.assertEqual(forward.conditioning, {"domain_name": "math", "task_name": "toy_reasoning_smoke"})
        self.assertEqual(forward.injection_anchors, ["segment:0", "segment:1"])

    def test_segmenter_is_deterministic(self) -> None:
        config = load_config(ROOT / "configs/exp/smoke_qwen25.yaml")
        runtime = MemoryRuntime(config=config, seed=7)
        first = runtime.segmenter.split("A || B || C")
        second = runtime.segmenter.split("A || B || C")
        self.assertEqual(first, second)

    def test_transformer_writer_and_resampler_fuser_shapes(self) -> None:
        config = load_config(ROOT / "configs/exp/smoke_qwen25_transformer_writer.yaml")
        dataset = load_toy_dataset(ROOT / config["task"]["dataset_path"])
        set_seed(321)
        runtime = MemoryRuntime(config=config, seed=321)

        forward = runtime.forward_example(dataset[1])
        loss = F.mse_loss(forward.predicted_state, forward.target_state)
        loss.backward()

        self.assertEqual(runtime.writer.arch, "transformer")
        self.assertEqual(runtime.fuser.arch, "resampler")
        self.assertEqual(list(forward.memory_long.shape), [1, 8, 64])
        self.assertEqual(list(forward.readouts.shape), [1, 4, 64])
        self.assertEqual(list(forward.memory_short.shape), [1, 2, 64])
        self.assertEqual(list(forward.gating.shape), [1, 4])
        self.assertIsNotNone(runtime.writer.slot_embeddings.grad)
        self.assertEqual(runtime.injector.position, "segment")

    def test_query_gating_changes_readouts(self) -> None:
        config = load_config(ROOT / "configs/exp/smoke_qwen25.yaml")
        gated_config = copy.deepcopy(config)
        gated_config["method"]["reader"]["gating_mode"] = "learned"

        dataset = load_toy_dataset(ROOT / gated_config["task"]["dataset_path"])
        set_seed(11)
        runtime = MemoryRuntime(config=gated_config, seed=11)
        forward = runtime.forward_example(dataset[0])

        self.assertFalse(torch.allclose(forward.gating, torch.ones_like(forward.gating)))

    def test_random_query_gating_masks_some_queries(self) -> None:
        config = load_config(ROOT / "configs/exp/smoke_qwen25.yaml")
        gated_config = copy.deepcopy(config)
        gated_config["method"]["reader"]["gating_mode"] = "random"

        dataset = load_toy_dataset(ROOT / gated_config["task"]["dataset_path"])
        set_seed(17)
        runtime = MemoryRuntime(config=gated_config, seed=17)
        forward = runtime.forward_example(dataset[0])

        for segment_stat in forward.segment_stats:
            gates = torch.tensor(segment_stat["gates"])
            self.assertTrue(torch.all((gates == 0) | (gates == 1)))
        self.assertFalse(torch.allclose(forward.gating, torch.ones_like(forward.gating)))

    def test_reader_supports_memory_mask(self) -> None:
        reader = MemoryReader(
            embed_dim=8,
            num_queries=2,
            use_query_gating=False,
            num_heads=2,
            condition_on_context=True,
        )
        memory = torch.arange(32, dtype=torch.float32).view(1, 4, 8)
        context = torch.zeros(1, 8)
        memory_mask = torch.tensor([[True, True, False, False]])

        outputs = reader.read(memory, context=context, memory_mask=memory_mask)

        self.assertEqual(list(outputs["readouts"].shape), [1, 2, 8])
        self.assertTrue(torch.allclose(outputs["attention"][:, :, 2:], torch.zeros(1, 2, 2), atol=1e-6))

    def test_reader_query_residual_preserves_query_signal(self) -> None:
        reader = MemoryReader(
            embed_dim=4,
            num_queries=2,
            use_query_gating=False,
            num_heads=2,
            condition_on_context=False,
            query_residual_scale=1.0,
        )
        with torch.no_grad():
            reader.cross_attn.in_proj_weight.zero_()
            reader.cross_attn.in_proj_bias.zero_()
            reader.cross_attn.out_proj.weight.zero_()
            reader.cross_attn.out_proj.bias.zero_()
            reader.readout_norm.weight.fill_(1.0)
            reader.readout_norm.bias.zero_()
            reader.queries.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                )
            )
        memory = torch.zeros(1, 3, 4)
        outputs = reader.read(memory)
        self.assertGreater(float(outputs["readouts"].abs().sum().item()), 0.0)

    def test_writer_freeze_save_and_load(self) -> None:
        writer = MemoryWriter(
            embed_dim=8,
            memory_slots=3,
            arch="transformer",
            hidden_dim=16,
            num_heads=2,
            transformer_layers=1,
        )
        writer.freeze()
        self.assertTrue(all(not parameter.requires_grad for parameter in writer.parameters()))
        writer.unfreeze()
        self.assertTrue(all(parameter.requires_grad for parameter in writer.parameters()))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "writer.pt"
            original_state = {key: value.detach().clone() for key, value in writer.state_dict().items()}
            writer.save_to(checkpoint_path)
            with torch.no_grad():
                writer.slot_embeddings.add_(1.0)
            writer.load_from(checkpoint_path)
            for key, value in writer.state_dict().items():
                self.assertTrue(torch.allclose(value, original_state[key]))

    def test_injector_toggle_changes_generation(self) -> None:
        config = load_config(ROOT / "configs/exp/smoke_qwen25_transformer_writer.yaml")
        disabled_config = copy.deepcopy(config)
        disabled_config["method"]["injector"]["enabled"] = False

        dataset = load_toy_dataset(ROOT / config["task"]["dataset_path"])
        example = dataset[0]

        set_seed(19)
        runtime_on = MemoryRuntime(config=config, seed=19)
        forward_on = runtime_on.forward_example(example)
        text_on = runtime_on.backbone.generate(
            [forward_on.next_prompt],
            memory_tokens=forward_on.generation_memory,
        )[0]

        set_seed(19)
        runtime_off = MemoryRuntime(config=disabled_config, seed=19)
        forward_off = runtime_off.forward_example(example)
        text_off = runtime_off.backbone.generate(
            [forward_off.next_prompt],
            memory_tokens=forward_off.generation_memory,
        )[0]

        self.assertNotEqual(text_on, text_off)
        self.assertIsNone(forward_off.generation_memory)
        self.assertGreater(forward_on.injected_inputs.shape[1], forward_off.injected_inputs.shape[1])

    def test_injection_positions_switch_via_config(self) -> None:
        dataset = load_toy_dataset(ROOT / "data/toy/smoke_samples.jsonl")
        example = dataset[0]

        segment_config = load_config(ROOT / "configs/exp/smoke_qwen25_transformer_writer.yaml")
        delimiter_config = load_config(ROOT / "configs/exp/smoke_qwen25_transformer_writer_delimiter_injection.yaml")
        random_config = load_config(ROOT / "configs/exp/smoke_qwen25_transformer_writer_random_injection.yaml")
        none_config = load_config(ROOT / "configs/exp/smoke_qwen25_transformer_writer_no_injection.yaml")

        set_seed(23)
        segment_forward = MemoryRuntime(config=segment_config, seed=23).forward_example(example)
        set_seed(23)
        delimiter_forward = MemoryRuntime(config=delimiter_config, seed=23).forward_example(example)
        set_seed(23)
        random_forward_first = MemoryRuntime(config=random_config, seed=23).forward_example(example)
        set_seed(23)
        random_forward_second = MemoryRuntime(config=random_config, seed=23).forward_example(example)
        set_seed(23)
        none_forward = MemoryRuntime(config=none_config, seed=23).forward_example(example)

        self.assertEqual(segment_forward.injection_anchors, ["segment:0", "segment:1"])
        self.assertEqual(delimiter_forward.injection_anchors, ["delimiter:0"])
        self.assertEqual(random_forward_first.injection_anchors, random_forward_second.injection_anchors)
        self.assertEqual(none_forward.injection_anchors, [])
        self.assertIsNone(none_forward.generation_memory)
        self.assertGreater(segment_forward.injected_inputs.shape[1], delimiter_forward.injected_inputs.shape[1])
        self.assertGreater(delimiter_forward.injected_inputs.shape[1], none_forward.injected_inputs.shape[1])

    def test_missing_conditioning_key_fails_early(self) -> None:
        config = load_config(ROOT / "configs/exp/smoke_qwen25.yaml")
        broken_config = copy.deepcopy(config)
        broken_config["method"]["reader"]["conditioning"]["domain_key"] = "missing_domain"
        dataset = load_toy_dataset(ROOT / broken_config["task"]["dataset_path"])

        runtime = MemoryRuntime(config=broken_config, seed=29)
        with self.assertRaisesRegex(ValueError, "missing conditioning domain key"):
            runtime.forward_example(dataset[0])

    def test_train_entrypoint_handles_no_injection_without_grad(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "train"
            exit_code = train_main(
                [
                    "--config",
                    str(ROOT / "configs/exp/smoke_qwen25_transformer_writer_no_injection.yaml"),
                    "--seed",
                    "37",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            metrics = output_dir.joinpath("metrics.json").read_text()
            self.assertIn('"loss_has_grad": false', metrics)


if __name__ == "__main__":
    unittest.main()
