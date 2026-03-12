# PLANv7 V7-4 Forced Consumption Summary

- comparison_conclusion: forced_consumption_diagnostic_only_move_to_v7_5
- recommended_next_step: open_v7_5_targeted_aux_revisit
- control_source_arm_id: b_w3_q16_s8
- direct_control_arm_id: d_w1_shared
- winning_depth: D1
- winning_depth_label: mid4
- base_for_v7_5_arm_id: b_w3_q16_s8
- base_for_v7_5_source_phase: v7_3

## Evidence
- any_forced_consumption_primary_score_improvement: False
- any_forced_consumption_diagnostic_only: True
- any_forced_consumption_route_live: True
- any_forced_consumption_stable_training: True

## Control
- writer_family: W3
- bridge_family: B3
- projector_family: P2
- memory_path_variant: two_level
- projector_token_source: short_slots
- active_depth_layers: [12, 13, 14, 15]
- writer_memory_slots: 64
- control.gsm8k.task_score: 0.000000
- control.triviaqa.task_score: 0.000000

## f1_num_mask
- forced_consumption_family: F1
- variant_label: num_mask
- covered_tasks: ['gsm8k']
- task_score_delta_sum: 0.000000
- actual_primary_improvement_task_count: 0
- primary_branch_success_task_count: 0
- usefulness_positive_task_count: 0
- route_live_task_count: 1
- stable_training_task_count: 1
- answer_switch_helpfulness_score: -0.633702
- acceptance_qualified: False
- diagnostic_only: False
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.primary_branch_success: False
- gsm8k.primary_usefulness_positive: False
- gsm8k.delta_answer_logprob_mean: -1.033702
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: True
- gsm8k.tail_window_source: train_loss_tail_50_steps_median

## f2_rx_only
- forced_consumption_family: F2
- variant_label: receiver_then_joint
- covered_tasks: ['gsm8k', 'triviaqa']
- task_score_delta_sum: 0.000000
- actual_primary_improvement_task_count: 0
- primary_branch_success_task_count: 0
- usefulness_positive_task_count: 0
- route_live_task_count: 2
- stable_training_task_count: 2
- answer_switch_helpfulness_score: 2.020884
- acceptance_qualified: False
- diagnostic_only: True
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.primary_branch_success: False
- gsm8k.primary_usefulness_positive: False
- gsm8k.delta_answer_logprob_mean: -0.009211
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: True
- gsm8k.tail_window_source: train_loss_tail_50_steps_median
- triviaqa.task_score_delta_vs_control: 0.000000
- triviaqa.primary_branch_success: False
- triviaqa.primary_usefulness_positive: False
- triviaqa.delta_answer_logprob_mean: 0.955096
- triviaqa.route_live_post_unfreeze: True
- triviaqa.stable_training_v6: True
- triviaqa.tail_window_source: train_loss_tail_50_steps_median

## f3_anneal
- forced_consumption_family: F3
- variant_label: starvation_anneal
- covered_tasks: ['gsm8k']
- task_score_delta_sum: 0.000000
- actual_primary_improvement_task_count: 0
- primary_branch_success_task_count: 0
- usefulness_positive_task_count: 0
- route_live_task_count: 1
- stable_training_task_count: 1
- answer_switch_helpfulness_score: -0.122094
- acceptance_qualified: False
- diagnostic_only: False
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.primary_branch_success: False
- gsm8k.primary_usefulness_positive: False
- gsm8k.delta_answer_logprob_mean: -0.597094
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: True
- gsm8k.tail_window_source: train_loss_tail_50_steps_median

## f4_dyn_budget
- forced_consumption_family: F4
- variant_label: dynamic_budget
- covered_tasks: ['gsm8k', 'triviaqa']
- task_score_delta_sum: 0.000000
- actual_primary_improvement_task_count: 0
- primary_branch_success_task_count: 0
- usefulness_positive_task_count: 0
- route_live_task_count: 2
- stable_training_task_count: 2
- answer_switch_helpfulness_score: -2.178945
- acceptance_qualified: False
- diagnostic_only: True
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.primary_branch_success: False
- gsm8k.primary_usefulness_positive: False
- gsm8k.delta_answer_logprob_mean: 0.173768
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: True
- gsm8k.tail_window_source: train_loss_tail_50_steps_median
- triviaqa.task_score_delta_vs_control: 0.000000
- triviaqa.primary_branch_success: False
- triviaqa.primary_usefulness_positive: False
- triviaqa.delta_answer_logprob_mean: -3.052713
- triviaqa.route_live_post_unfreeze: True
- triviaqa.stable_training_v6: True
- triviaqa.tail_window_source: train_loss_tail_50_steps_median
