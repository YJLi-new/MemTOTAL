# PLANv8 V8-0 Summary

- `comparison_conclusion = qwen3_calibrated_interfaces_alive_open_v8_1`
- `recommended_next_step = open_v8_1_reader_interface_scout`
- `ri1_passed_basic_smoke = True`
- `ri2_passed_basic_smoke = True`
- `legacy_prefix_oracle_reproduced_or_bounded = True`

## Selected Prompt Modes

- `gsm8k`: `q3_gsm8k_nonthink` via `b0_q3_gsm8k_nonthink`
- `triviaqa`: `q3_trivia_think` via `b3_q3_trivia_think`
- `fever`: `answer_slot_labels` via `b4_q3_fever_nonthink`

## Primary Baselines

- `gsm8k`: qwen3=`0.671875`, qwen2.5 replay=`0.000000`
- `triviaqa`: qwen3=`0.187500`, qwen2.5 replay=`0.000000`

## Reader Activation

- `o4_q3_xattn_oracle_smoke_gsm8k`: gate=`0.500000`, attention_mass=`0.000000`, grad_median=`73.348282`
- `o4_q3_xattn_oracle_smoke_triviaqa`: gate=`0.500000`, attention_mass=`0.000000`, grad_median=`131.817748`

