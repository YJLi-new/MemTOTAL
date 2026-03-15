# PLANv8 V8-1 Reader Interface Scout

## Decision
- `comparison_conclusion = reader_interface_family_preference_open_v8_2`
- `recommended_next_step = open_v8_2_reader_sweep`
- `best_arm_id = i1_seq16_r32_mid8`
- `best_interface_family = ri1_prepend_block`
- `base_for_v8_2_arm_id = i1_seq16_r32_mid8`

## Ranking
- `i1_seq16_r32_mid8` score_sum=-0.375000 logprob_sum=171.603863 activation=0.058479 acceptance=False
- `i3_seq32_r64_mid8` score_sum=-0.390625 logprob_sum=164.772279 activation=0.025283 acceptance=False
- `i2_seq16_r64_mid8` score_sum=-0.437500 logprob_sum=164.961174 activation=0.034429 acceptance=False
- `i0_prefix_legacy_r2` score_sum=-0.406250 logprob_sum=164.662733 activation=0.000066 acceptance=False
- `i4_xattn16_mid4_r32` score_sum=-0.703125 logprob_sum=159.229156 activation=0.778224 acceptance=False
- `i5_xattn16_mid4_r64` score_sum=-0.781250 logprob_sum=159.518836 activation=1.249293 acceptance=False

## Control
- `gsm8k` control score = 0.671875
- `triviaqa` control score = 0.187500
- `fever` control score = 0.583333
