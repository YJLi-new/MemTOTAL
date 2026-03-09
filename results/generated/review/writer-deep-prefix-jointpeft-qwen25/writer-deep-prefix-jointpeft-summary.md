# Writer Deep-Prefix Joint-PEFT Summary

- comparison_conclusion: move_to_v6_baseline_rerun
- recommended_next_step: run_clean_v6_baseline
- any_nonfever_route_live: True
- any_nonfever_task_supervision_live: False
- any_nonfever_source_not_collapsed: False
- any_nonfever_stable_training: True
- any_nonfever_usefulness_positive: True

## source_stub_health
- route_live: True
- stable_recipe: True
- loss_steps_1_50_median: 6.977210
- loss_tail_50_steps_median: 6.977210

## gsm8k
- route_live_post_unfreeze: True
- writer_task_supervision_live: False
- source_not_collapsed: False
- stable_training_v6: True
- usefulness_positive_v6: True
- loss_steps_1_50_median: 8.574252
- loss_steps_451_500_median: 4.126132
- writer_grad_norm_post_unfreeze_median: 10.807674
- writer_task_to_total_grad_ratio_post_unfreeze: 0.000000
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.012875
- prefix_attention_nontrivial_layer_count: 3

## narrativeqa
- route_live_post_unfreeze: True
- writer_task_supervision_live: False
- source_not_collapsed: False
- stable_training_v6: True
- usefulness_positive_v6: False
- loss_steps_1_50_median: 16.289518
- loss_steps_451_500_median: 3.729689
- writer_grad_norm_post_unfreeze_median: 10.616507
- writer_task_to_total_grad_ratio_post_unfreeze: 0.000000
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.002544
- prefix_attention_nontrivial_layer_count: 3

## fever
- route_live_post_unfreeze: True
- writer_task_supervision_live: False
- source_not_collapsed: True
- stable_training_v6: True
- usefulness_positive_v6: True
- loss_steps_1_50_median: 3.853199
- loss_steps_451_500_median: 0.154474
- writer_grad_norm_post_unfreeze_median: 19.449187
- writer_task_to_total_grad_ratio_post_unfreeze: 0.000000
- delta_answer_logprob: 0.000000
- prefix_attention_mass_mean: 0.008444
- prefix_attention_nontrivial_layer_count: 3
