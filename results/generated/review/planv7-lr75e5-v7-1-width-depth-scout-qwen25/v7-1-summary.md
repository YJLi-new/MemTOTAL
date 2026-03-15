# PLANv7 V7-1 Width Depth Scout Summary

- comparison_conclusion: select_mid4_for_v7_2
- recommended_next_step: open_v7_2_direct_bandwidth_mid4
- winning_depth: D1
- winning_depth_label: mid4
- fever_not_used_to_override_primary: True

## Owner LR Metadata
- owner_locked_projector_lr: 7.5e-05
- repo_confirmed_v65_projector_lr_reference: 7.5e-05
- owner_override_note: False

## Depth Comparison
- split_by_task: False
- d0_ranking_key: [0.0, 0.0, 0.0, 0.09577597677707672, 1.8517270669341086, 0.0, -8.0]
- d1_ranking_key: [0.0, 0.0, 0.0, 0.10320675000548363, -0.6001753762364387, 0.0, -8.0]
- d1_tied_or_better_on_primary: True
- d1_clearly_better_writer_metrics: False

## D0
- depth_label: early4
- arm_ids: s00, s10
- gsm8k_task_score_delta_mean: 0.000000
- triviaqa_task_score_delta_mean: 0.000000
- primary_task_score_delta_sum: 0.000000
- strict_writer_memory_task_count_total: 0
- strict_rank_fraction_mean: 0.095776
- answer_switch_helpfulness_score_mean: 1.851727
- stable_training_task_count_total: 0
- ranking_penalty_count_total: 8

## D1
- depth_label: mid4
- arm_ids: s01, s11
- gsm8k_task_score_delta_mean: 0.000000
- triviaqa_task_score_delta_mean: 0.000000
- primary_task_score_delta_sum: 0.000000
- strict_writer_memory_task_count_total: 0
- strict_rank_fraction_mean: 0.103207
- answer_switch_helpfulness_score_mean: -0.600175
- stable_training_task_count_total: 0
- ranking_penalty_count_total: 8

## s00
- writer_family: W0
- depth_family: D0
- projector_family: P0
- projector_mode: shared_low_rank
- bridge_mode: writer_direct
- memory_path_variant: single_level
- active_depth_layers: [0, 1, 2, 3]
- projector_rank: 32
- writer_memory_slots: 8
- writer_conditioning_layers: 1
- gsm8k_task_score_delta_vs_control: 0.000000
- triviaqa_task_score_delta_vs_control: 0.000000
- primary_task_score_delta_sum: 0.000000
- strict_writer_memory_task_count: 0
- strict_rank_fraction_mean: 0.126989
- answer_switch_helpfulness_score: 2.186351
- stable_training_task_count: 0
- ranking_penalty_count: 4
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.writer_memory_not_collapsed_strict: False
- gsm8k.primary_usefulness_positive: False
- gsm8k.primary_branch_success: False
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: False
- gsm8k.projector_manufactured_diversity: True
- triviaqa.task_score_delta_vs_control: 0.000000
- triviaqa.writer_memory_not_collapsed_strict: False
- triviaqa.primary_usefulness_positive: False
- triviaqa.primary_branch_success: False
- triviaqa.route_live_post_unfreeze: True
- triviaqa.stable_training_v6: False
- triviaqa.projector_manufactured_diversity: True

## s01
- writer_family: W0
- depth_family: D1
- projector_family: P0
- projector_mode: shared_low_rank
- bridge_mode: writer_direct
- memory_path_variant: single_level
- active_depth_layers: [12, 13, 14, 15]
- projector_rank: 32
- writer_memory_slots: 8
- writer_conditioning_layers: 1
- gsm8k_task_score_delta_vs_control: 0.000000
- triviaqa_task_score_delta_vs_control: 0.000000
- primary_task_score_delta_sum: 0.000000
- strict_writer_memory_task_count: 0
- strict_rank_fraction_mean: 0.139014
- answer_switch_helpfulness_score: -0.555017
- stable_training_task_count: 0
- ranking_penalty_count: 4
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.writer_memory_not_collapsed_strict: False
- gsm8k.primary_usefulness_positive: False
- gsm8k.primary_branch_success: False
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: False
- gsm8k.projector_manufactured_diversity: True
- triviaqa.task_score_delta_vs_control: 0.000000
- triviaqa.writer_memory_not_collapsed_strict: False
- triviaqa.primary_usefulness_positive: False
- triviaqa.primary_branch_success: False
- triviaqa.route_live_post_unfreeze: True
- triviaqa.stable_training_v6: False
- triviaqa.projector_manufactured_diversity: True

## s10
- writer_family: W1
- depth_family: D0
- projector_family: P1
- projector_mode: shared_low_rank
- bridge_mode: writer_direct
- memory_path_variant: single_level
- active_depth_layers: [0, 1, 2, 3]
- projector_rank: 64
- writer_memory_slots: 16
- writer_conditioning_layers: 2
- gsm8k_task_score_delta_vs_control: 0.000000
- triviaqa_task_score_delta_vs_control: 0.000000
- primary_task_score_delta_sum: 0.000000
- strict_writer_memory_task_count: 0
- strict_rank_fraction_mean: 0.064563
- answer_switch_helpfulness_score: 1.517103
- stable_training_task_count: 0
- ranking_penalty_count: 4
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.writer_memory_not_collapsed_strict: False
- gsm8k.primary_usefulness_positive: False
- gsm8k.primary_branch_success: False
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: False
- gsm8k.projector_manufactured_diversity: True
- triviaqa.task_score_delta_vs_control: 0.000000
- triviaqa.writer_memory_not_collapsed_strict: False
- triviaqa.primary_usefulness_positive: False
- triviaqa.primary_branch_success: False
- triviaqa.route_live_post_unfreeze: True
- triviaqa.stable_training_v6: False
- triviaqa.projector_manufactured_diversity: False

## s11
- writer_family: W1
- depth_family: D1
- projector_family: P1
- projector_mode: shared_low_rank
- bridge_mode: writer_direct
- memory_path_variant: single_level
- active_depth_layers: [12, 13, 14, 15]
- projector_rank: 64
- writer_memory_slots: 16
- writer_conditioning_layers: 2
- gsm8k_task_score_delta_vs_control: 0.000000
- triviaqa_task_score_delta_vs_control: 0.000000
- primary_task_score_delta_sum: 0.000000
- strict_writer_memory_task_count: 0
- strict_rank_fraction_mean: 0.067400
- answer_switch_helpfulness_score: -0.645334
- stable_training_task_count: 0
- ranking_penalty_count: 4
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.writer_memory_not_collapsed_strict: False
- gsm8k.primary_usefulness_positive: False
- gsm8k.primary_branch_success: False
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: False
- gsm8k.projector_manufactured_diversity: True
- triviaqa.task_score_delta_vs_control: 0.000000
- triviaqa.writer_memory_not_collapsed_strict: False
- triviaqa.primary_usefulness_positive: False
- triviaqa.primary_branch_success: False
- triviaqa.route_live_post_unfreeze: True
- triviaqa.stable_training_v6: False
- triviaqa.projector_manufactured_diversity: False
