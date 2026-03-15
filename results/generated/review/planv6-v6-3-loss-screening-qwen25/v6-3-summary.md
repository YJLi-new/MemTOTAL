# PLANv6 V6-3 Anti-Homogenization Loss Screening Summary

- comparison_conclusion: select_top_auxiliary_families
- recommended_next_step: open_v6_4_mixed_matrix
- top_auxiliary_families: l2_contrastive, l5_orthogonality_coverage, l3_vicreg

## Ranked Combos

### 1. S3 multi_item_cross_attn_raw + L2 contrastive
- combo_id: s3_multi_item_cross_attn_raw__l2_contrastive
- stimulus_mix: C0 support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001862294196267, 1.0, 2.0, 0.046875, 2.1164165660738945]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.046875, writer_task_to_total_grad_ratio_post_unfreeze=1.001862
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=0.009758

### 2. S3 multi_item_cross_attn_raw + L5 orthogonality + coverage
- combo_id: s3_multi_item_cross_attn_raw__l5_orthogonality_coverage
- stimulus_mix: C0 support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0000000091593864, 1.0, 2.0, 0.1875, 2.464398570358753]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.187500, writer_task_to_total_grad_ratio_post_unfreeze=1.000000
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=0.017677

### 3. S3 multi_item_cross_attn_raw + L3 VICReg / VCReg
- combo_id: s3_multi_item_cross_attn_raw__l3_vicreg
- stimulus_mix: C0 support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0000000024582434, 1.0, 2.0, 0.28125, 4.080529414117336]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.281250, writer_task_to_total_grad_ratio_post_unfreeze=1.000000
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=0.003516

### 4. S3 multi_item_cross_attn_raw + L0 task-only
- combo_id: s3_multi_item_cross_attn_raw__l0_task_only
- stimulus_mix: C0 support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.21875, 2.4835936054587364]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.218750, writer_task_to_total_grad_ratio_post_unfreeze=1.000000
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

### 5. S5 hybrid_pooled_plus_items + L5 orthogonality + coverage
- combo_id: s5_hybrid_pooled_plus_items__l5_orthogonality_coverage
- stimulus_mix: C2 support_and_context_gated
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999990329955555, 1.0, 2.0, 0.015625, 0.9896862432360649]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.015625, writer_task_to_total_grad_ratio_post_unfreeze=0.999980
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.068413

### 6. S5 hybrid_pooled_plus_items + L2 contrastive
- combo_id: s5_hybrid_pooled_plus_items__l2_contrastive
- stimulus_mix: C2 support_and_context_gated
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9996417389940634, 1.0, 2.0, 0.359375, 4.116477154195309]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375, writer_task_to_total_grad_ratio_post_unfreeze=0.999642
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

### 7. S5 hybrid_pooled_plus_items + L3 VICReg / VCReg
- combo_id: s5_hybrid_pooled_plus_items__l3_vicreg
- stimulus_mix: C2 support_and_context_gated
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0000096683316109, 0.0, 2.0, 0.328125, 2.912396766245365]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125, writer_task_to_total_grad_ratio_post_unfreeze=1.000000
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

### 8. S5 hybrid_pooled_plus_items + L0 task-only
- combo_id: s5_hybrid_pooled_plus_items__l0_task_only
- stimulus_mix: C2 support_and_context_gated
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.390625, 4.577105976641178]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625, writer_task_to_total_grad_ratio_post_unfreeze=1.000000
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.062885

### 9. S3 multi_item_cross_attn_raw + L1 legacy
- combo_id: s3_multi_item_cross_attn_raw__l1_legacy
- stimulus_mix: C0 support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.00000020870056, 1.0, 2.0, 0.046875, 1.9822446778416634]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.046875, writer_task_to_total_grad_ratio_post_unfreeze=1.000000
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

### 10. S5 hybrid_pooled_plus_items + L1 legacy
- combo_id: s5_hybrid_pooled_plus_items__l1_legacy
- stimulus_mix: C2 support_and_context_gated
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.9999990467819261, 1.0, 2.0, 0.171875, 2.4856837019324303]
- FEVER: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.171875, writer_task_to_total_grad_ratio_post_unfreeze=0.895203
- gsm8k: route_live_post_unfreeze=True, source_not_collapsed=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, source_not_collapsed=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

## Ranked Families

### 1. L2 contrastive
- loss_id: l2_contrastive
- best_combo_id: s3_multi_item_cross_attn_raw__l2_contrastive
- best_support_mode_id: s3_multi_item_cross_attn_raw
- best_selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001862294196267, 1.0, 2.0, 0.046875, 2.1164165660738945]
- any_task_route_live: True
- any_task_source_not_collapsed: True
- any_task_usefulness_positive: True
- legacy_misleading_movement: False

### 2. L5 orthogonality + coverage
- loss_id: l5_orthogonality_coverage
- best_combo_id: s3_multi_item_cross_attn_raw__l5_orthogonality_coverage
- best_support_mode_id: s3_multi_item_cross_attn_raw
- best_selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0000000091593864, 1.0, 2.0, 0.1875, 2.464398570358753]
- any_task_route_live: True
- any_task_source_not_collapsed: True
- any_task_usefulness_positive: True
- legacy_misleading_movement: False

### 3. L3 VICReg / VCReg
- loss_id: l3_vicreg
- best_combo_id: s3_multi_item_cross_attn_raw__l3_vicreg
- best_support_mode_id: s3_multi_item_cross_attn_raw
- best_selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0000000024582434, 1.0, 2.0, 0.28125, 4.080529414117336]
- any_task_route_live: True
- any_task_source_not_collapsed: True
- any_task_usefulness_positive: True
- legacy_misleading_movement: False

### 4. L0 task-only
- loss_id: l0_task_only
- best_combo_id: s3_multi_item_cross_attn_raw__l0_task_only
- best_support_mode_id: s3_multi_item_cross_attn_raw
- best_selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.21875, 2.4835936054587364]
- any_task_route_live: True
- any_task_source_not_collapsed: True
- any_task_usefulness_positive: True
- legacy_misleading_movement: False

### 5. L1 legacy
- loss_id: l1_legacy
- best_combo_id: s3_multi_item_cross_attn_raw__l1_legacy
- best_support_mode_id: s3_multi_item_cross_attn_raw
- best_selection_tuple: [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.00000020870056, 1.0, 2.0, 0.046875, 1.9822446778416634]
- any_task_route_live: True
- any_task_source_not_collapsed: True
- any_task_usefulness_positive: True
- legacy_misleading_movement: False
