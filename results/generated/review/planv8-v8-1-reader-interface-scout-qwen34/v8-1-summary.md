# PLANv8 V8-1 Reader Interface Scout

## Decision
- `comparison_conclusion = reader_interface_flat_open_v8_2_last_chance`
- `recommended_next_step = open_v8_2_reader_sweep_last_chance`
- `best_arm_id = i0_prefix_legacy_r2`
- `best_interface_family = ri0_legacy_prefix`
- `base_for_v8_2_arm_id = i0_prefix_legacy_r2`

## Ranking
- `i0_prefix_legacy_r2` score_sum=0.000000 logprob_sum=202.007629 activation=0.000123 acceptance=False
- `i1_seq16_r32_mid8` score_sum=-0.515625 logprob_sum=199.441752 activation=0.099490 acceptance=False
- `i3_seq32_r64_mid8` score_sum=-0.531250 logprob_sum=202.137800 activation=0.099973 acceptance=False
- `i2_seq16_r64_mid8` score_sum=-0.593750 logprob_sum=208.740545 activation=0.097032 acceptance=False
- `i4_xattn16_mid4_r32` score_sum=-0.734375 logprob_sum=198.033161 activation=0.754893 acceptance=False
- `i5_xattn16_mid4_r64` score_sum=-0.812500 logprob_sum=194.696777 activation=1.010141 acceptance=False

## Control
- `gsm8k` control score = 0.812500
- `triviaqa` control score = 0.031250
- `fever` control score = 0.791667
