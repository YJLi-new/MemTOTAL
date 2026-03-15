# Writer Deep-Prefix Joint-PEFT Summary

- comparison_conclusion: fix_route_before_next_run
- recommended_next_step: debug_route_or_recipe
- any_nonfever_route_live: False
- any_nonfever_stable_training: False
- any_nonfever_usefulness_positive: False

## source_stub_health
- route_live: True
- stable_recipe: True
- loss_steps_1_50_median: 9.608093
- loss_tail_50_steps_median: 7.327192

## gsm8k
- route_live: False
- stable_training: False
- usefulness_positive: False
- loss_steps_1_50_median: 8.565849
- loss_steps_451_500_median: 3.864791
- writer_grad_norm_steps_1_50_median: 0.000000
- projector_grad_norm_steps_1_50_median: 315.032501
- receiver_lora_grad_norm_steps_1_50_median: 70.272491
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.012875
- prefix_attention_nontrivial_layer_count: 3

## narrativeqa
- route_live: False
- stable_training: False
- usefulness_positive: False
- loss_steps_1_50_median: 15.945962
- loss_steps_451_500_median: 3.641736
- writer_grad_norm_steps_1_50_median: 0.000000
- projector_grad_norm_steps_1_50_median: 188.518967
- receiver_lora_grad_norm_steps_1_50_median: 130.058655
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.002544
- prefix_attention_nontrivial_layer_count: 3

## fever
- route_live: False
- stable_training: False
- usefulness_positive: False
- loss_steps_1_50_median: 3.649772
- loss_steps_451_500_median: 0.153543
- writer_grad_norm_steps_1_50_median: 0.000000
- projector_grad_norm_steps_1_50_median: 169.193817
- receiver_lora_grad_norm_steps_1_50_median: 90.820595
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.008444
- prefix_attention_nontrivial_layer_count: 3
