# PLANv6 V6-4 Mixed Matrix Summary

- comparison_conclusion: select_finalists
- recommended_next_step: open_v6_5_recipe_stabilization
- finalist_configs: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage, s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg, s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive

## Ranked Combos

### 1. S3 multi_item_cross_attn_raw + C2 support_and_context_gated + L5 orthogonality + coverage
- combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage
- support_mode: multi_item_cross_attn_raw
- stimulus_mode: support_and_context
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.9999925899737295, 0.390625, -0.07238136607841357, 4.108028061687946]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.390625
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

### 2. S3 multi_item_cross_attn_raw + C2 support_and_context_gated + L3 VICReg / VCReg
- combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg
- support_mode: multi_item_cross_attn_raw
- stimulus_mode: support_and_context
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0000005588448722, 0.34375, -0.07238136607841357, 3.2121597304940224]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

### 3. S3 multi_item_cross_attn_raw + C0 support_only + L2 contrastive
- combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive
- support_mode: multi_item_cross_attn_raw
- stimulus_mode: support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9978658627048699, 0.3125, -0.06775173644878393, 3.5503552332520485]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.067752

### 4. S5 hybrid_pooled_plus_items + C2 support_and_context_gated + L5 orthogonality + coverage
- combo_id: s5_hybrid_pooled_plus_items__c2_support_and_context_gated__l5_orthogonality_coverage
- support_mode: hybrid_pooled_plus_items
- stimulus_mode: support_and_context
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.9999860577752556, 0.03125, 0.0002667288314967958, 1.6991052702069283]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.031250
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.000267

### 5. S3 multi_item_cross_attn_raw + C0 support_only + L5 orthogonality + coverage
- combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l5_orthogonality_coverage
- support_mode: multi_item_cross_attn_raw
- stimulus_mode: support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.999999982245719, 0.34375, 0.004712979186579036, 3.2349578961730003]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.343750
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.004713

### 6. S5 hybrid_pooled_plus_items + C2 support_and_context_gated + L3 VICReg / VCReg
- combo_id: s5_hybrid_pooled_plus_items__c2_support_and_context_gated__l3_vicreg
- support_mode: hybrid_pooled_plus_items
- stimulus_mode: support_and_context
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.000000721735108, 0.3125, -0.06370081052285802, 3.7700453475117683]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.312500
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.063701

### 7. S5 hybrid_pooled_plus_items + C0 support_only + L5 orthogonality + coverage
- combo_id: s5_hybrid_pooled_plus_items__c0_support_only__l5_orthogonality_coverage
- support_mode: hybrid_pooled_plus_items
- stimulus_mode: support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.000000041450376, 0.0, -0.06682581052285802, 0.07737631350755692]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.066826

### 8. S3 multi_item_cross_attn_raw + C2 support_and_context_gated + L2 contrastive
- combo_id: s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l2_contrastive
- support_mode: multi_item_cross_attn_raw
- stimulus_mode: support_and_context
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.9993325513858684, 0.09375, -0.07238136607841357, 0.4340813532471657]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.093750
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

### 9. S5 hybrid_pooled_plus_items + C0 support_only + L2 contrastive
- combo_id: s5_hybrid_pooled_plus_items__c0_support_only__l2_contrastive
- support_mode: hybrid_pooled_plus_items
- stimulus_mode: support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.9979347529044235, 0.28125, -0.0684131121101596, 3.7537007555365562]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.281250
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.068413

### 10. S5 hybrid_pooled_plus_items + C2 support_and_context_gated + L2 contrastive
- combo_id: s5_hybrid_pooled_plus_items__c2_support_and_context_gated__l2_contrastive
- support_mode: hybrid_pooled_plus_items
- stimulus_mode: support_and_context
- balance_mode: layernorm_learned_scalar
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.99499025355612, 0.328125, 0.019518186963346673, 3.688073493540287]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.328125
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.019518

### 11. S3 multi_item_cross_attn_raw + C0 support_only + L3 VICReg / VCReg
- combo_id: s3_multi_item_cross_attn_raw__c0_support_only__l3_vicreg
- support_mode: multi_item_cross_attn_raw
- stimulus_mode: support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999999929072901, 0.03125, -0.07238136607841357, 0.5939175561070442]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.031250
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=-0.072381

### 12. S5 hybrid_pooled_plus_items + C0 support_only + L3 VICReg / VCReg
- combo_id: s5_hybrid_pooled_plus_items__c0_support_only__l3_vicreg
- support_mode: hybrid_pooled_plus_items
- stimulus_mode: support_only
- balance_mode: off
- selection_tuple: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999896312780074, 0.1875, 0.017896243960859745, 2.368745632469654]
- fever: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=True, task_score_delta_vs_control=0.187500
- gsm8k: route_live_post_unfreeze=True, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=True, usefulness_positive_v6=True, task_score_delta_vs_control=0.000000
- narrativeqa: route_live_post_unfreeze=False, writer_task_supervision_live=True, source_not_collapsed=True, stable_training_v6=False, usefulness_positive_v6=False, task_score_delta_vs_control=0.017896
