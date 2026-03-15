from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

from memtotal.models.backbone import BackboneWrapper, ReceiverMicroLoRALinear


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"

    def __init__(self):
        self.last_chat_kwargs = None
        self.last_chat_messages = None

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

    def apply_chat_template(self, conversation, **kwargs):
        self.last_chat_messages = conversation
        self.last_chat_kwargs = dict(kwargs)
        content = str(conversation[0]["content"])
        if kwargs.get("enable_thinking") is False:
            return f"<chat-no-think>{content}</chat-no-think>"
        return f"<chat>{content}</chat>"

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        base = [max(3, (ord(char) % 17) + 3) for char in text][:12]
        if not base:
            base = [3]
        if add_special_tokens:
            return [self.eos_token_id, *base]
        return base


class _FakeRotaryEmbedding:
    def __call__(self, x, position_ids):
        batch, seq = position_ids.shape
        head_dim = 4
        cos = torch.ones(batch, seq, head_dim, dtype=x.dtype, device=x.device)
        sin = torch.zeros(batch, seq, head_dim, dtype=x.dtype, device=x.device)
        return cos, sin


class _FakeSelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 4
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            self.k_proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
            self.v_proj.weight.copy_(self.k_proj.weight)


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(8)
        self.self_attn = _FakeSelfAttention()


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8, num_hidden_layers=2, _attn_implementation="sdpa")
        self.embedding = torch.nn.Embedding(64, 8)
        self.model = types.SimpleNamespace(
            layers=torch.nn.ModuleList([_FakeDecoderLayer(), _FakeDecoderLayer()]),
            rotary_emb=_FakeRotaryEmbedding(),
        )
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(64 * 8, dtype=torch.float32).view(64, 8) / 100.0)

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_hidden_states=False,
        use_cache=False,
        output_attentions=False,
        past_key_values=None,
        cache_position=None,
    ):
        if inputs_embeds is not None:
            hidden = inputs_embeds.to(dtype=torch.float32)
            batch, seq, _ = hidden.shape
        else:
            assert input_ids is not None
            hidden = self.embedding(input_ids).to(dtype=torch.float32)
            batch, seq = input_ids.shape
        past_length = 0
        if past_key_values is not None:
            past_length = int(past_key_values.get_seq_length())
            if past_length > 0:
                prefix_signal = past_key_values.layers[0].values.mean(dim=(1, 2, 3), keepdim=False).view(batch, 1, 1)
                hidden = hidden + prefix_signal
        token_basis = (hidden[..., 0] * 10.0).round().to(dtype=torch.long).clamp(min=0)
        logits = torch.zeros(batch, seq, 64, dtype=torch.float32)
        logits.scatter_(
            -1,
            (token_basis % 64).unsqueeze(-1),
            ((token_basis % 64).to(dtype=torch.float32) / 10.0).unsqueeze(-1),
        )
        attentions = None
        if output_attentions:
            if self.config._attn_implementation == "eager":
                attentions = []
                for layer_index in range(self.config.num_hidden_layers):
                    layer_past = 0
                    if past_key_values is not None and layer_index < len(past_key_values.layers):
                        layer = past_key_values.layers[layer_index]
                        if layer.keys is not None:
                            layer_past = int(layer.keys.shape[-2])
                    total_kv = layer_past + seq
                    layer_attention = torch.full((batch, 1, seq, total_kv), 1.0 / max(1, total_kv), dtype=torch.float32)
                    if layer_past > 0:
                        layer_attention[:, :, :, :layer_past] += 0.05
                        layer_attention = layer_attention / layer_attention.sum(dim=-1, keepdim=True)
                    attentions.append(layer_attention)
        hidden_states = [
            hidden,
            hidden + 0.1,
            hidden + 0.2,
        ]
        returned_past = past_key_values
        if use_cache:
            from transformers.cache_utils import DynamicCache

            returned_past = past_key_values if past_key_values is not None else DynamicCache()
            key_states = hidden[..., :4].view(batch, seq, 1, 4).transpose(1, 2).contiguous()
            value_states = hidden[..., 4:8].view(batch, seq, 1, 4).transpose(1, 2).contiguous()
            for layer_index in range(self.config.num_hidden_layers):
                returned_past.update(key_states, value_states, layer_index)
        return types.SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            attentions=tuple(attentions) if attentions is not None else None,
            past_key_values=returned_past,
        )

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=1,
        inputs_embeds=None,
        past_key_values=None,
        cache_position=None,
        use_cache=True,
    ):
        if input_ids is None:
            assert inputs_embeds is not None
            input_ids = torch.full(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                5,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        append = torch.full((input_ids.shape[0], 2), 7, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, append], dim=1)


class BackboneRealModeTest(unittest.TestCase):
    def test_stub_mode_default_model_id_supports_qwen34(self) -> None:
        backbone = BackboneWrapper(
            name="Qwen3-4B",
            load_mode="stub",
            hidden_size=8,
            seed=123,
        )
        self.assertEqual(backbone.model_id, "Qwen/Qwen3-4B")

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
        self.assertNotEqual(float(prefixed_scores.abs().sum().item()), 0.0)
        self.assertNotEqual(float(prefixed_scores[0].item()), float(prefixed_scores[1].item()))
        deep_prefix = {
            0: torch.ones(1, 3, 8, dtype=torch.float32),
            1: torch.full((1, 3, 8), 0.5, dtype=torch.float32),
        }
        deep_scores, diagnostics = backbone.score_continuations(
            "Prompt",
            ["good ending", "bad"],
            layer_prefix_hidden_by_layer=deep_prefix,
            return_diagnostics=True,
        )
        self.assertEqual(list(deep_scores.shape), [2])
        self.assertEqual(diagnostics["diagnostic_layers"], [0, 1])
        self.assertEqual(sorted(diagnostics["prefix_attention_mass_by_candidate_by_layer"]), [0, 1])
        self.assertEqual(backbone.model.config._attn_implementation, "sdpa")
        projection = backbone.summarize_layer_prefix_projection(deep_prefix)
        self.assertEqual(sorted(projection["layer_key_l2_by_layer"]), ["0", "1"])
        generations = backbone.generate(["Prompt"])
        self.assertEqual(len(generations), 1)
        self.assertTrue(generations[0])
        prefixed_generations = backbone.generate(["Prompt"], prefix_embeddings=prefix)
        self.assertEqual(len(prefixed_generations), 1)
        self.assertTrue(prefixed_generations[0])
        deep_prefixed_generations = backbone.generate(["Prompt"], layer_prefix_hidden_by_layer=deep_prefix)
        self.assertEqual(len(deep_prefixed_generations), 1)
        self.assertTrue(deep_prefixed_generations[0])
        prompt_hidden, prompt_mask = backbone.extract_prompt_hidden_state_slice(
            ["Prompt alpha", "B"],
            max_tokens=3,
        )
        self.assertEqual(list(prompt_hidden.shape), [2, 3, 8])
        self.assertEqual(list(prompt_mask.shape), [2, 3])
        self.assertEqual(prompt_mask.dtype, torch.bool)
        self.assertTrue(bool(prompt_mask[0].any().item()))
        self.assertTrue(bool(prompt_mask[1].any().item()))
        layer_hidden, layer_mask = backbone.extract_layer_hidden_state_slices(
            ["Prompt alpha", "B"],
            layer_indices=[0, 1],
            max_tokens=3,
        )
        self.assertEqual(sorted(layer_hidden), [0, 1])
        self.assertEqual(list(layer_hidden[0].shape), [2, 3, 8])
        self.assertEqual(list(layer_mask.shape), [2, 3])
        calibration = backbone.collect_deep_prefix_calibration(
            ["Prompt alpha", "B"],
            layer_indices=[0, 1],
            max_tokens=3,
        )
        self.assertEqual(list(calibration["semantic_anchor"].shape), [1, 8])
        self.assertEqual(list(calibration["hidden_state_anchor"].shape), [1, 3, 8])
        self.assertEqual(sorted(calibration["layer_hidden_anchor_by_layer"]), [0, 1])
        self.assertEqual(sorted(calibration["layer_key_l2_by_layer"]), ["0", "1"])
        self.assertEqual(sorted(calibration["layer_value_l2_by_layer"]), ["0", "1"])

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_real_mode_can_apply_chat_template_for_prompt_scoring_and_generation(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen3-8B",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
            use_chat_template=True,
            chat_template_enable_thinking=False,
        )
        scores = backbone.score_continuations("Prompt", ["good ending"])
        self.assertEqual(list(scores.shape), [1])
        tokenizer = backbone.tokenizer
        assert tokenizer is not None
        self.assertEqual(tokenizer.last_chat_messages, [{"role": "user", "content": "Prompt good ending"}])
        self.assertEqual(tokenizer.last_chat_kwargs["enable_thinking"], False)
        generations = backbone.generate(["Prompt"])
        self.assertEqual(len(generations), 1)
        self.assertEqual(tokenizer.last_chat_messages, [{"role": "user", "content": "Prompt"}])

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_enable_receiver_micro_lora_wraps_only_requested_targets(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen2.5-1.5B-Instruct",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )

        backbone.enable_receiver_micro_lora(
            layer_indices=[1],
            target_modules=["k_proj"],
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )

        layer0 = backbone.model.model.layers[0]
        layer1 = backbone.model.model.layers[1]
        self.assertFalse(isinstance(layer0.self_attn.k_proj, ReceiverMicroLoRALinear))
        self.assertFalse(isinstance(layer0.self_attn.v_proj, ReceiverMicroLoRALinear))
        self.assertTrue(isinstance(layer1.self_attn.k_proj, ReceiverMicroLoRALinear))
        self.assertFalse(isinstance(layer1.self_attn.v_proj, ReceiverMicroLoRALinear))
        metadata = backbone.receiver_lora_metadata()
        self.assertEqual(metadata["target_layers"], [1])
        self.assertEqual(metadata["target_modules"], ["k_proj"])
        self.assertEqual(metadata["trainable_params"], 24)

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_receiver_micro_lora_zero_init_preserves_deep_prefix_scores(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen2.5-1.5B-Instruct",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )
        deep_prefix = {
            0: torch.ones(1, 3, 8, dtype=torch.float32),
            1: torch.full((1, 3, 8), 0.5, dtype=torch.float32),
        }
        baseline_scores = backbone.score_continuations(
            "Prompt",
            ["good ending", "bad"],
            layer_prefix_hidden_by_layer=deep_prefix,
        )

        backbone.enable_receiver_micro_lora(
            layer_indices=[0, 1],
            target_modules=["k_proj", "v_proj"],
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        lora_scores = backbone.score_continuations(
            "Prompt",
            ["good ending", "bad"],
            layer_prefix_hidden_by_layer=deep_prefix,
        )

        self.assertTrue(torch.allclose(baseline_scores, lora_scores))

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_receiver_micro_lora_freezes_backbone_outside_adapter(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen2.5-1.5B-Instruct",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )
        backbone.enable_receiver_micro_lora(
            layer_indices=[1],
            target_modules=["k_proj", "v_proj"],
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.set_receiver_lora_trainable(True)

        adapter_params = backbone.receiver_lora_parameters()
        self.assertEqual(sum(parameter.numel() for parameter in adapter_params), 48)
        self.assertTrue(all(parameter.requires_grad for parameter in adapter_params))
        layer1 = backbone.model.model.layers[1]
        self.assertFalse(layer1.self_attn.k_proj.base_linear.weight.requires_grad)
        self.assertFalse(layer1.self_attn.v_proj.base_linear.weight.requires_grad)
        self.assertFalse(backbone.model.embedding.weight.requires_grad)

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_receiver_micro_lora_state_dict_round_trip(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen2.5-1.5B-Instruct",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )
        backbone.enable_receiver_micro_lora(
            layer_indices=[1],
            target_modules=["k_proj", "v_proj"],
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        state = backbone.receiver_lora_state_dict()
        assert state is not None
        with torch.no_grad():
            for target_state in state.values():
                target_state["down.weight"].fill_(0.25)
                target_state["up.weight"].fill_(0.5)
        backbone.load_receiver_lora_state_dict(state, checkpoint_path="fake-checkpoint.pt")
        reloaded_state = backbone.receiver_lora_state_dict()
        assert reloaded_state is not None
        for target_path in state:
            self.assertTrue(torch.allclose(state[target_path]["down.weight"], reloaded_state[target_path]["down.weight"]))
            self.assertTrue(torch.allclose(state[target_path]["up.weight"], reloaded_state[target_path]["up.weight"]))

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_real_mode_supports_chunk_pooled_slots_and_sequence_memory(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen3-8B",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )
        memory_tokens, memory_mask = backbone.extract_chunk_pooled_hidden_state_slots(
            ["Prompt alpha", "Prompt beta"],
            layer_index=1,
            pool_window=2,
            slot_cap=3,
        )
        self.assertEqual(list(memory_tokens.shape), [2, 3, 8])
        self.assertEqual(list(memory_mask.shape), [2, 3])
        self.assertTrue(bool(memory_mask[0].any().item()))

        baseline_scores = backbone.score_continuations("Prompt", ["good ending", "bad"])
        memory_scores, diagnostics = backbone.score_continuations(
            "Prompt",
            ["good ending", "bad"],
            memory_tokens=memory_tokens[:1],
            memory_consumer_mode="prepend_block",
            return_diagnostics=True,
        )
        self.assertEqual(list(memory_scores.shape), [2])
        self.assertEqual(diagnostics["memory_consumer_mode"], "prepend_block")
        self.assertEqual(int(diagnostics["prefix_length"]), int(memory_tokens[:1].shape[1]))
        self.assertEqual(list(baseline_scores.shape), list(memory_scores.shape))

        generations = backbone.generate(
            ["Prompt"],
            memory_tokens=memory_tokens[:1],
            memory_consumer_mode="prepend_block",
        )
        self.assertEqual(len(generations), 1)
        self.assertTrue(generations[0])

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_prepend_memory_tokens_are_rescaled_to_content_embedding_norm(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen3-8B",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )
        memory_tokens = torch.full((1, 2, 8), 100.0, dtype=torch.float32)
        model_kwargs, metadata = backbone._prepare_prefixed_hf_inputs(
            ["Prompt"],
            None,
            None,
            memory_tokens=memory_tokens,
        )
        self.assertEqual(int(metadata["prefix_length"]), 2)
        inputs_embeds = model_kwargs["inputs_embeds"].to(dtype=torch.float32)
        attention_mask = model_kwargs["attention_mask"].to(dtype=torch.bool)
        prefix_norms = inputs_embeds[:, :2, :].norm(dim=-1)
        content_norms = inputs_embeds[:, 2:, :].norm(dim=-1)
        content_mean = content_norms[attention_mask[:, 2:]].mean()
        for norm_value in prefix_norms.flatten():
            self.assertAlmostEqual(float(norm_value.item()), float(content_mean.item()), places=5)

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_real_mode_supports_precached_latent_memory(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen3-8B",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )
        memory_tokens = torch.full((1, 2, 8), 25.0, dtype=torch.float32)
        baseline_scores = backbone.score_continuations("Prompt", ["good ending", "bad"])
        precached_scores, diagnostics = backbone.score_continuations(
            "Prompt",
            ["good ending", "bad"],
            memory_tokens=memory_tokens,
            memory_consumer_mode="precache_latent",
            return_diagnostics=True,
        )
        self.assertEqual(list(precached_scores.shape), [2])
        self.assertEqual(diagnostics["memory_consumer_mode"], "precache_latent")
        self.assertEqual(int(diagnostics["prefix_length"]), 2)
        self.assertEqual(int(diagnostics["cache_growth_tokens"]), 2)
        self.assertEqual(diagnostics["diagnostic_layers"], [0, 1])
        self.assertFalse(torch.equal(baseline_scores, precached_scores))

        encoded = backbone._prepare_hf_inputs(["Prompt"])
        model_kwargs, metadata = backbone._prepare_precached_latent_hf_inputs(encoded, memory_tokens)
        self.assertEqual(int(metadata["prefix_length"]), 2)
        self.assertEqual(int(metadata["cache_growth_tokens"]), 2)
        self.assertEqual(int(model_kwargs["past_key_values"].get_seq_length()), 2)

        generations = backbone.generate(
            ["Prompt"],
            memory_tokens=memory_tokens,
            memory_consumer_mode="precache_latent",
        )
        self.assertEqual(len(generations), 1)
        self.assertTrue(generations[0])

    @mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer())
    @mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel())
    def test_reader_cross_attn_state_dict_round_trip(self, _mock_model, _mock_tokenizer):
        backbone = BackboneWrapper(
            name="Qwen3-8B",
            load_mode="hf_causal_lm",
            hidden_size=None,
            seed=123,
            model_id="fake/model",
            device="cpu",
            dtype="float32",
        )
        backbone.enable_reader_cross_attention(
            layer_indices=[0, 1],
            num_heads=2,
            ff_hidden_dim=16,
            gate_init=0.0,
        )
        metadata = backbone.reader_cross_attn_metadata()
        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["target_layers"], [0, 1])
        self.assertGreater(metadata["trainable_params"], 0)

        state = backbone.reader_cross_attn_state_dict()
        assert state is not None
        with torch.no_grad():
            for layer_state in state.values():
                for state_key, tensor in layer_state.items():
                    layer_state[state_key] = tensor + 0.25
        backbone.load_reader_cross_attn_state_dict(state, checkpoint_path="fake-reader-cross-attn.pt")
        reloaded_state = backbone.reader_cross_attn_state_dict()
        assert reloaded_state is not None
        for adapter_key in state:
            for state_key in state[adapter_key]:
                self.assertTrue(
                    torch.allclose(
                        state[adapter_key][state_key],
                        reloaded_state[adapter_key][state_key],
                    )
                )


if __name__ == "__main__":
    unittest.main()
