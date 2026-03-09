# TL Writer Value V1 Summary

- comparison_conclusion: failure
- primary_interpretation: writer_architecture_first_matrix_flat
- recommended_arm: control
- move_to_v1_penalties: False
- move_to_v2: True
- stop_after_v1_architecture: True
- failure_reason: common_mode_domination_persists

## Control
- writer_mode: shared_add
- top1_top2_ratio: 70.6796
- common_mode_energy_ratio: 0.9986
- reader_readout_effective_rank: 1.2283
- reader_readout_pairwise_cosine_mean: 0.9993

## Shared Add Scaled
- writer_mode: shared_add_scaled
- shared_state_scale: 0.0200
- top1_top2_ratio: 70.6727
- top1_top2_reduction_factor: 1.0001
- readout_effective_rank: 1.2280
- readout_pairwise_cosine_mean: 0.9993
- collapse_delayed: False
- medium_success: False

## Slot Query Only
- writer_mode: slot_query_only
- top1_top2_ratio: 70.6796
- top1_top2_reduction_factor: 1.0000
- readout_effective_rank: 1.2298
- readout_pairwise_cosine_mean: 0.9993
- collapse_delayed: False
- medium_success: False
- strong_success: False
