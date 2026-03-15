# PLANv9 V9-0 FlashMem Discrimination Summary

- `outcome_id = O2`
- `comparison_conclusion = flashmem_precache_collapse_matches_sequence_replay`
- `recommended_next_step = hard_fail_a2_shift_mainline_consumer_to_c0_or_c2`
- `mainline_consumer_candidate = C0_or_C2_only`
- `selected_prompt_variant = q3_gsm8k_nonthink`
- `v8_0_reference_gsm8k_score = 0.8125`

| Arm | Score | Delta vs A0 | Delta answer logprob | Gen len words | Malformed | Attention mass | Nontrivial layers | Cache growth | Peak VRAM MiB | Wall time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 no-memory control | 0.8750 | +0.0000 | +0.0000 | 70.61 | 0.0000 | 0.0000 | 0 | 0 | 8303.91 | 251.91 |
| A1 legacy-prefix oracle | 0.8750 | +0.0000 | +0.0000 | 71.27 | 0.0000 | 0.0002 | 0 | 0 | 8800.91 | 260.21 |
| A2 precache-latent oracle | 0.1250 | -0.7500 | +0.0000 | 98.48 | 0.0000 | 0.0000 | 0 | 8 | 8444.31 | 347.00 |
| A3 sequence replay oracle | 0.1406 | -0.7344 | +0.0000 | 96.50 | 0.0000 | 0.0643 | 36 | 8 | 8444.31 | 339.44 |

Interpretation:
- `A2 vs A0 = -0.7500` (-48 / 64 examples)
- `A2 vs A1 = -0.7500` (-48 / 64 examples)
- `A2 vs A3 = -0.0156` (-1 / 64 examples)

This phase resolves whether a FlashMem-style cache-prefill route is non-destructive on the qwen34 GSM8K continuity split before opening broader long-horizon engineering.
