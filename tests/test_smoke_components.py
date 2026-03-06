from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.data import load_toy_dataset
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


if __name__ == "__main__":
    unittest.main()

