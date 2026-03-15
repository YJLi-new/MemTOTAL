# PLANv8 V8-0 Summary (Qwen3-4B)

- `comparison_conclusion = qwen34_calibrated_interfaces_alive_open_v8_1`
- `recommended_next_step = open_v8_1_reader_interface_scout`
- `ri1_passed_basic_smoke = True`
- `ri2_passed_basic_smoke = True`
- `legacy_prefix_oracle_reproduced_or_bounded = True`

## Selected Prompt Modes

- `gsm8k`: `q3_gsm8k_nonthink` via `b0_q34_gsm8k_nonthink`
- `triviaqa`: `q3_trivia_nonthink` via `b2_q34_trivia_nonthink`
- `fever`: `answer_slot_labels` via `b4_q34_fever_nonthink`

## Primary Baselines

- `gsm8k`: Qwen3-4B=`0.812500`, qwen2.5 replay=`0.000000`
- `triviaqa`: Qwen3-4B=`0.031250`, qwen2.5 replay=`0.000000`

## Reader Activation

- `o4_q34_xattn_oracle_smoke_gsm8k`: gate=`0.500000`, attention_mass=`0.000000`, grad_median=`77.220471`
- `o4_q34_xattn_oracle_smoke_triviaqa`: gate=`0.500000`, attention_mass=`0.000000`, grad_median=`156.864682`

