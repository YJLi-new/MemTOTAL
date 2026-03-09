# Writer Deep-Prefix Joint-PEFT Summary

- comparison_conclusion: move_to_writer_usefulness_branch
- recommended_next_step: open_writer_usefulness_branch
- any_nonfever_route_live: True
- any_nonfever_task_supervision_live: True
- any_nonfever_source_not_collapsed: True
- any_nonfever_stable_training: True
- any_nonfever_usefulness_positive: True

## source_stub_health
- route_live: True
- stable_recipe: True
- loss_steps_1_50_median: 8.675567
- loss_tail_50_steps_median: 8.675567

## gsm8k
- route_live_post_unfreeze: True
- writer_task_supervision_live: True
- source_not_collapsed: False
- stable_training_v6: True
- usefulness_positive_v6: True
- loss_steps_1_50_median: 6.975586
- loss_steps_451_500_median: 4.787467
- writer_grad_norm_post_unfreeze_median: 31.127171
- writer_task_to_total_grad_ratio_post_unfreeze: 1.000000
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.004162
- prefix_attention_nontrivial_layer_count: 4

## narrativeqa
- route_live_post_unfreeze: False
- writer_task_supervision_live: True
- source_not_collapsed: True
- stable_training_v6: False
- usefulness_positive_v6: False
- loss_steps_1_50_median: 16.642006
- loss_steps_451_500_median: 6.225616
- writer_grad_norm_post_unfreeze_median: 7.233650
- writer_task_to_total_grad_ratio_post_unfreeze: 1.000000
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.000731
- prefix_attention_nontrivial_layer_count: 1

## fever
- route_live_post_unfreeze: True
- writer_task_supervision_live: True
- source_not_collapsed: False
- stable_training_v6: True
- usefulness_positive_v6: True
- loss_steps_1_50_median: 3.862473
- loss_steps_451_500_median: 0.073576
- writer_grad_norm_post_unfreeze_median: 15.879994
- writer_task_to_total_grad_ratio_post_unfreeze: 1.000000
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.006317
- prefix_attention_nontrivial_layer_count: 4
