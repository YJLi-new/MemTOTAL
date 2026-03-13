# PLANv8 V8-0 Summary

- `comparison_conclusion = repair_qwen3_interface_before_v8_1`
- `recommended_next_step = repair_v8_0_interface_path`
- `ri1_passed_basic_smoke = False`
- `ri2_passed_basic_smoke = True`
- `legacy_prefix_oracle_reproduced_or_bounded = True`

## Selected Prompt Modes

- `gsm8k`: `q3_gsm8k_think_boxed` via `b1_q3_gsm8k_think_boxed`
- `triviaqa`: `q3_trivia_think` via `b3_q3_trivia_think`
- `fever`: `answer_slot_labels` via `b4_q3_fever_nonthink`

## Primary Baselines

- `gsm8k`: qwen3=`0.000000`, qwen2.5 replay=`0.000000`
- `triviaqa`: qwen3=`0.000000`, qwen2.5 replay=`0.000000`

## Reader Activation

- `o4_q3_xattn_oracle_smoke_gsm8k`: gate=`0.500000`, attention_mass=`0.000000`, grad_median=`6.054741`
- `o4_q3_xattn_oracle_smoke_triviaqa`: gate=`0.500000`, attention_mass=`0.000000`, grad_median=`80.306012`

