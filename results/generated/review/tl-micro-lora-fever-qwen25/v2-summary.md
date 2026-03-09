# TL Micro-LoRA V2 Summary

- comparison_conclusion: move_to_v4
- primary_interpretation: micro_lora_no_actionable_signal
- recommended_arm: micro_lora_r2_late3
- continue_to_l2: False
- continue_to_l3: False
- move_to_v3: False
- move_to_v4: True
- failure_reason: no_gradient_or_capability_gain

## Control
- receiver_lora_enabled: False
- receiver_lora_rank: 0
- receiver_lora_target_layers: []
- receiver_lora_target_modules: []
- receiver_lora_trainable_params: 0
- best_adapt_task_score: 0.2951
- best_adapt_macro_f1: 0.1519
- dominant_label_collapse_onset_step: 0
- selection_passed: False
- screen248_test_gate_passed: False
- train_grad_norm_reader_steps_1_4_median: 0.000000
- train_grad_norm_fuser_steps_1_4_median: 0.000000

## Late3 R2
- receiver_lora_enabled: True
- receiver_lora_rank: 2
- receiver_lora_target_layers: [14, 21, 27]
- receiver_lora_target_modules: ['k_proj', 'v_proj']
- receiver_lora_trainable_params: 21504
- best_adapt_task_score: 0.2951
- best_adapt_macro_f1: 0.1519
- dominant_label_collapse_onset_step: 0
- selection_passed: False
- screen248_test_gate_passed: False
- train_grad_norm_reader_steps_1_4_median: 0.000000
- train_grad_norm_fuser_steps_1_4_median: 0.000000
- train_grad_norm_receiver_lora_steps_1_4_median: 0.000000
- reader_grad_boost: 0.0047
- fuser_grad_boost: 0.0216
- task_score_delta: 0.0000
- collapse_delayed: False
- diagnostic_success: False
- partial_evidence: False
- medium_success: False
- strong_success: False
