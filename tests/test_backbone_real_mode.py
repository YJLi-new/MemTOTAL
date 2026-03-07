from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

from memtotal.models.backbone import BackboneWrapper


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"

    def __call__(self, texts, *, padding=False, truncation=False, return_tensors=None, add_special_tokens=True):
        if isinstance(texts, str):
            input_ids = [self._encode(texts, add_special_tokens=add_special_tokens)]
        else:
            input_ids = [self._encode(text, add_special_tokens=add_special_tokens) for text in texts]
        max_length = max(len(row) for row in input_ids)
        padded = [row + [self.pad_token_id] * (max_length - len(row)) for row in input_ids]
        attention = [[1] * len(row) + [0] * (max_length - len(row)) for row in input_ids]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        if isinstance(texts, str):
            return {"input_ids": input_ids[0]}
        return {"input_ids": input_ids}

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(token_id) for token_id in token_ids if token_id != self.pad_token_id)

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        base = [max(3, (ord(char) % 17) + 3) for char in text][:12]
        if not base:
            base = [3]
        if add_special_tokens:
            return [self.eos_token_id, *base]
        return base


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(64, 8)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(64 * 8, dtype=torch.float32).view(64, 8) / 100.0)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, output_hidden_states=False, use_cache=False):
        if inputs_embeds is not None:
            hidden = inputs_embeds.to(dtype=torch.float32)
            batch, seq, _ = hidden.shape
            token_basis = hidden[..., 0].round().to(dtype=torch.long).clamp(min=0)
        else:
            assert input_ids is not None
            batch, seq = input_ids.shape
            token_basis = input_ids.to(dtype=torch.long)
            hidden = torch.stack(
                [
                    input_ids.to(dtype=torch.float32),
                    input_ids.to(dtype=torch.float32) * 0.5,
                    input_ids.to(dtype=torch.float32) * 0.25,
                    input_ids.to(dtype=torch.float32) * 0.125,
                    input_ids.to(dtype=torch.float32) * 0.0625,
                    input_ids.to(dtype=torch.float32) * 0.03125,
                    input_ids.to(dtype=torch.float32) * 0.015625,
                    input_ids.to(dtype=torch.float32) * 0.0078125,
                ],
                dim=-1,
            )
        logits = torch.zeros(batch, seq, 64, dtype=torch.float32)
        logits.scatter_(
            -1,
            (token_basis % 64).unsqueeze(-1),
            ((token_basis % 64).to(dtype=torch.float32) / 10.0).unsqueeze(-1),
        )
        return types.SimpleNamespace(logits=logits, hidden_states=[hidden, hidden])

    def generate(self, input_ids, attention_mask=None, max_new_tokens=32, do_sample=False, pad_token_id=0, eos_token_id=1):
        append = torch.full((input_ids.shape[0], 2), 7, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, append], dim=1)


class BackboneRealModeTest(unittest.TestCase):
    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_real_mode_supports_summarize_score_and_generate(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen2.5-1.5B-Instruct",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )
        summaries = backbone.summarize_texts(["alpha", "beta"])
        self.assertEqual(list(summaries.shape), [2, 8])
        encoded = backbone.encode_texts(["alpha"])
        self.assertEqual(list(encoded.shape[:2]), [1, encoded.shape[1]])
        scores = backbone.score_continuations("Prompt", ["good ending", "bad"])
        self.assertEqual(list(scores.shape), [2])
        self.assertNotEqual(float(scores[0].item()), float(scores[1].item()))
        prefix = torch.ones(1, 3, 8, dtype=torch.float32)
        prefixed_scores = backbone.score_continuations("Prompt", ["good ending", "bad"], prefix_embeddings=prefix)
        self.assertEqual(list(prefixed_scores.shape), [2])
        generations = backbone.generate(["Prompt"])
        self.assertEqual(len(generations), 1)
        self.assertTrue(generations[0])


if __name__ == "__main__":
    unittest.main()
