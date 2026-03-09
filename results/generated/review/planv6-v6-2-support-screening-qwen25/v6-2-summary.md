# PLANv6 V6-2 Support Screening Summary

- comparison_conclusion: select_top_two_support_modes
- recommended_next_step: open_v6_3_anti_homogenization_screen
- top_two_support_modes: s3_multi_item_cross_attn_raw, s5_hybrid_pooled_plus_items

## 1. S3 multi_item_cross_attn_raw
- support_mode: multi_item_cross_attn_raw
- stimulus_mix: C0 support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.296875, 2.794325537979603, 5.0]
- FEVER:
  route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.296875
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=0.005747

## 2. S5 hybrid_pooled_plus_items
- support_mode: hybrid_pooled_plus_items
- stimulus_mix: C2 support_and_context_gated
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.046875, 0.8805840536952019, 5.0]
- FEVER:
  route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.046875
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

## 3. S1 pooled_block_gated
- support_mode: pooled_block
- stimulus_mix: C2 support_and_context_gated
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.8800443187355995, 3.0]
- FEVER:
  route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

## 4. S2 structured_support_set
- support_mode: structured_support_set
- stimulus_mix: C1 support_and_context_legacy
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.296875, 3.063650645315647, 4.0]
- FEVER:
  route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.296875
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.006502

## 5. S4 multi_item_cross_attn_encoded
- support_mode: multi_item_cross_attn_encoded
- stimulus_mix: C1 support_and_context_legacy
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.015625, 0.8541292920708656, 5.0]
- FEVER:
  route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.015625
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

## 6. S0 pooled_block_legacy
- support_mode: pooled_block
- stimulus_mix: C1 support_and_context_legacy
- balance_mode: off
- selection_tuple: [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.328125, 4.11899022012949, 4.0]
- FEVER:
  route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=False, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.066826
