# PLANv7 V7-1 Width Depth Scout Summary

- comparison_conclusion: select_mid4_for_v7_2
- recommended_next_step: open_v7_2_direct_bandwidth_mid4
- winning_depth: D1
- winning_depth_label: mid4
- fever_not_used_to_override_primary: True

## Owner LR Metadata
- owner_locked_projector_lr: 7.5e-06
- repo_confirmed_v65_projector_lr_reference: 7.5e-05
- owner_override_note: True

## Depth Comparison
- split_by_task: False
- d0_ranking_key: [0.0, 0.0, 0.0, 0.09632364846765995, 1.1712032943964006, 0.0, -8.0]
- d1_ranking_key: [0.0, 0.0, 1.0, 0.15876440703868866, -0.20643591731786737, 0.0, -6.0]
- d1_tied_or_better_on_primary: True
- d1_clearly_better_writer_metrics: True

## D0
- depth_label: early4
- arm_ids: s00, s10
- gsm8k_task_score_delta_mean: 0.000000
- triviaqa_task_score_delta_mean: 0.000000
- primary_task_score_delta_sum: 0.000000
- strict_writer_memory_task_count_total: 0
- strict_rank_fraction_mean: 0.096324
- answer_switch_helpfulness_score_mean: 1.171203
- stable_training_task_count_total: 0
- ranking_penalty_count_total: 8

## D1
- depth_label: mid4
- arm_ids: s01, s11
- gsm8k_task_score_delta_mean: 0.000000
- triviaqa_task_score_delta_mean: 0.000000
- primary_task_score_delta_sum: 0.000000
- strict_writer_memory_task_count_total: 1
- strict_rank_fraction_mean: 0.158764
- answer_switch_helpfulness_score_mean: -0.206436
- stable_training_task_count_total: 0
- ranking_penalty_count_total: 6

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
- strict_rank_fraction_mean: 0.127950
- answer_switch_helpfulness_score: 2.232577
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
- strict_writer_memory_task_count: 1
- strict_rank_fraction_mean: 0.242179
- answer_switch_helpfulness_score: -1.065495
- stable_training_task_count: 0
- ranking_penalty_count: 2
- gsm8k.task_score_delta_vs_control: 0.000000
- gsm8k.writer_memory_not_collapsed_strict: True
- gsm8k.primary_usefulness_positive: False
- gsm8k.primary_branch_success: False
- gsm8k.route_live_post_unfreeze: True
- gsm8k.stable_training_v6: False
- gsm8k.projector_manufactured_diversity: False
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
- strict_rank_fraction_mean: 0.064697
- answer_switch_helpfulness_score: 0.109829
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
- strict_rank_fraction_mean: 0.075350
- answer_switch_helpfulness_score: 0.652623
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
