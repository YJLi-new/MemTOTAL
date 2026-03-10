# PLANv6 V6-5 Recipe Stabilization Summary

- comparison_conclusion: screen_complete_confirmation_pending
- recommended_next_step: run_v6_5_confirmation_stage
- screen_top_two_recipes: F3__w0__clip_groupwise__plr75e6__acc4__layers_additive, F1__w0__clip_global__plr5e5__acc1__layers_additive
- stabilized_recipes: (none)

## Screen Ranking

### 1. F3__w0__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.468750

### 2. F1__w0__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.437500

### 3. F2__w0__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.437500

### 4. F1__w10__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.437500

### 5. F1__w0__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.437500

### 6. F3__w0__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.437500

### 7. F2__w10__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 8. F1__w10__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 9. F2__w0__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 10. F1__w20__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 11. F2__w10__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 12. F1__w10__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 13. F3__w10__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 14. F2__w10__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.406250

### 15. F2__w20__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.406250

### 16. F1__w0__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.406250

### 17. F2__w0__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.406250

### 18. F1__w0__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.406250

### 19. F3__w20__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.406250

### 20. F2__w20__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 21. F2__w0__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 22. F3__w0__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 23. F3__w20__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 24. F3__w20__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 25. F1__w10__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 26. F2__w20__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 27. F1__w10__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 28. F2__w0__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 29. F2__w0__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 30. F1__w0__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 31. F1__w10__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 32. F1__w20__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 33. F1__w20__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 34. F3__w10__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 35. F3__w20__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 36. F2__w10__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 37. F2__w0__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 38. F1__w0__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 39. F2__w10__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 40. F2__w20__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 41. F2__w10__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 42. F1__w20__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 43. F2__w0__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 44. F1__w0__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 45. F1__w20__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 46. F3__w20__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 47. F3__w20__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 48. F2__w0__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 49. F2__w10__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 50. F2__w20__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 51. F2__w20__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 52. F2__w0__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 53. F1__w0__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 54. F1__w20__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 55. F3__w0__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 56. F3__w0__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 57. F3__w20__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 58. F3__w0__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 59. F3__w10__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 60. F3__w0__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 61. F3__w0__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 62. F1__w20__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 63. F2__w20__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 64. F2__w20__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 65. F1__w0__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 66. F2__w20__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 67. F1__w10__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 68. F1__w0__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 69. F3__w10__clip_groupwise__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 70. F3__w10__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 71. F3__w10__clip_groupwise__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 72. F3__w10__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 73. F1__w10__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 74. F2__w20__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 75. F2__w20__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 76. F2__w10__clip_groupwise__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 77. F1__w20__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 78. F3__w20__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 79. F3__w0__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 80. F3__w10__clip_groupwise__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 81. F3__w0__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.296875

### 82. F1__w0__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.296875

### 83. F3__w0__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.296875

### 84. F1__w20__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.281250

### 85. F2__w10__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.265625

### 86. F1__w20__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.265625

### 87. F1__w0__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.250000

### 88. F3__w10__clip_groupwise__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.234375

### 89. F1__w20__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.125000

### 90. F2__w10__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.078125

### 91. F3__w0__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.062500

### 92. F1__w10__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.046875

### 93. F3__w10__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.015625

### 94. F2__w10__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.015625

### 95. F2__w0__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.015625

### 96. F1__w0__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 97. F3__w20__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 98. F3__w20__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.421875

### 99. F1__w20__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.406250

### 100. F1__w20__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.406250

### 101. F1__w0__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 102. F2__w0__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 103. F1__w10__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 104. F3__w20__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 105. F3__w10__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 106. F3__w10__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625

### 107. F2__w10__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 108. F2__w20__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 109. F1__w10__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 110. F3__w20__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.375000

### 111. F3__w20__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 112. F2__w10__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 113. F2__w10__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 114. F1__w20__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 115. F1__w10__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.359375

### 116. F3__w20__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 117. F2__w20__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 118. F2__w20__clip_global__plr75e6__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 119. F3__w0__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750

### 120. F1__w10__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 121. F2__w0__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125

### 122. F1__w20__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 123. F2__w0__clip_global__plr5e5__acc4__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 124. F3__w0__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500

### 125. F3__w0__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.296875

### 126. F2__w0__clip_global__plr75e6__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.296875

### 127. F3__w10__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.296875

### 128. F2__w10__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.281250

### 129. F3__w10__clip_global__plr75e6__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.281250

### 130. F3__w10__clip_global__plr5e5__acc4__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 4
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.265625

### 131. F2__w20__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.250000

### 132. F1__w0__clip_groupwise__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: groupwise
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.234375

### 133. F3__w0__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.234375

### 134. F3__w20__clip_global__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.218750

### 135. F1__w0__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.171875

### 136. F1__w10__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.140625

### 137. F3__w10__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.140625

### 138. F1__w20__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 20
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.078125

### 139. F2__w0__clip_global__plr75e6__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 0
- clipping_scheme: global
- projector_learning_rate: 7.5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.062500

### 140. F2__w10__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.015625

### 141. F1__w10__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000

### 142. F1__w10__clip_global__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- warmup_steps: 10
- clipping_scheme: global
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000

### 143. F3__w20__clip_groupwise__plr5e5__acc1__layers_additive
- finalist_combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_additive
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000

### 144. F2__w20__clip_groupwise__plr5e5__acc1__layers_base
- finalist_combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- warmup_steps: 20
- clipping_scheme: groupwise
- projector_learning_rate: 5e-05
- accumulation_steps: 1
- layer_variant: layers_base
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.015625

## Confirmation
