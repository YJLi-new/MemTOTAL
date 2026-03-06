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

    def test_query_gating_changes_readouts(self) -> None:
        config = load_config(ROOT / "configs/exp/smoke_qwen25.yaml")
        gated_config = copy.deepcopy(config)
        gated_config["method"]["reader"]["use_query_gating"] = True

        dataset = load_toy_dataset(ROOT / gated_config["task"]["dataset_path"])
        set_seed(11)
        runtime = MemoryRuntime(config=gated_config, seed=11)
        forward = runtime.forward_example(dataset[0])

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


if __name__ == "__main__":
    unittest.main()
