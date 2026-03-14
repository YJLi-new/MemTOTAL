# PLANv8 V8-0 Repair: Qwen3-4B Sequence Interface Normalization

## Purpose

Repair the qwen34 `PLANv8` `V8-0` line after the first governed pass ended at:

- `comparison_conclusion = repair_qwen34_interface_before_v8_1`
- `recommended_next_step = repair_v8_0_interface_path`

This remains a `V8-0` repair. It does not open `V8-1` unless the repaired `V8-0` summary explicitly authorizes it.

## Failure Surface

The first qwen34 `V8-0` pass completed cleanly but failed the `RI1` smoke gate:

- `o2_q34_seq_oracle16_*` and `o3_q34_seq_oracle32_*` materialized `memory_tokens_count = 8`
- the same arms stayed at `prefix_attention_nontrivial_layer_count = 0`
- `prefix_attention_mass_mean` collapsed to about `8.56e-05` on GSM8K and `5.01e-04` on TriviaQA
- the matching repaired qwen3 `V8-0` line stayed around `1.24e-02` and `3.35e-02` on the same `RI1` route

The routing path is alive, but the prepended hidden-state slots are too poorly calibrated for the qwen34 backbone to treat them as meaningful content tokens.

## Diagnosis

`BackboneWrapper._prepare_prefixed_hf_inputs()` was prepending raw oracle hidden-state slots directly as `inputs_embeds`.

On the real qwen34/qwen3 probes:

- ordinary token embedding norms were small (`~1.01` on qwen34, `~1.60` on qwen3)
- oracle hidden-state slot norms were much larger (`~93` to `115`)

That means the sequence-memory route was feeding strongly out-of-scale hidden states into the token stream.

Targeted local probes on the real governed `o2` arm confirmed:

- baseline qwen34 `RI1`: `prefix_attention_mass_mean ≈ 7.7e-05`, nontrivial layers `0`
- content-norm matched memory tokens: `prefix_attention_mass_mean ≈ 8.65e-02`, nontrivial layers `1`
- content-position preservation alone: still `≈ 8.5e-05`, nontrivial layers `0`

The repair target is therefore memory-token normalization, not position remapping.

## Repair Work

1. Rescale prepended `memory_tokens` to the mean norm of the active content-token embeddings before concatenation into `inputs_embeds`.
2. Add a backbone unit test that locks this normalization behavior in real-HF mode.
3. Extend the review publisher so a repaired qwen34 run root:
   - `/root/autodl-tmp/runs/verify/planv8-v8-0-qwen34-baselines-oracles-r1`
   - `/root/autodl-tmp/results/generated/planv8-v8-0-qwen34-baselines-oracles-r1`
   refreshes the canonical qwen34 `V8-0` review surface.
4. Re-arm the qwen34 chain against the repaired `-r1` namespace and keep the `V8-1` queue gated on the repaired summary.

## Validation

```bash
bash -n \
  scripts/arm_planv8_qwen34_chain.sh \
  scripts/publish_review_artifacts.sh
python -m py_compile src/memtotal/models/backbone.py
python -m unittest \
  tests.test_backbone_real_mode \
  tests.test_planv8_v8_0_summary \
  tests.test_repo_lints \
  tests.test_repo_contract -v
```

## Progress

- 2026-03-14 UTC: Audited the finished qwen34 `V8-0` summary and confirmed the governed stop reason is specific to `RI1`, not baseline calibration or `RI2`.
- 2026-03-14 UTC: Reproduced the qwen34 `o2` sequence-memory weakness on a 4-example governed subset from the real materialized config.
- 2026-03-14 UTC: Local probe results:
  - baseline qwen34 `o2`: `prefix_attention_mass_mean ≈ 7.7e-05`, nontrivial layers `0`
  - content-norm rescaling: `≈ 8.65e-02`, nontrivial layers `1`
  - content-position preservation only: no meaningful change
- 2026-03-14 UTC: Confirmed the same normalization rule also keeps the repaired qwen3 `RI1` route alive on its matching 4-example subset (`~1.12e-02 -> ~7.89e-02`, still `1` nontrivial layer).
