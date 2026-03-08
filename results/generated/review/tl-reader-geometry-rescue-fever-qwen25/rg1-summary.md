# TL Reader Geometry RG-1 Summary

- comparison_conclusion: informative
- failure_reason: 
- primary_interpretation: B-1c_linear_fuser
- recommended_control_arm: rg1c_ctxoff_h4_k4_linear
- move_to_rg2: False
- context_overwrite_supported: False
- k_eq_h_supported: False
- linear_fuser_supported: True

## Baseline
- reader_query_entropy_mean: 2.0777
- final_reader_attention_pairwise_cosine_mean: 1.0000
- final_memory_short_effective_rank: 1.2098

## RG-1A CTX-OFF H4-K8
- meaningful_movement: False
- entropy_delta: 0.0017
- pairwise_delta: 0.0000
- short_rank_delta: 0.1739
- final_reader_context_overwrite_ratio: 0.0000

## RG-1B CTX-OFF H4-K4
- meaningful_movement: False
- entropy_delta: 0.0017
- pairwise_delta: 0.0000
- short_rank_delta: -0.0030
- final_reader_context_overwrite_ratio: 0.0000

## RG-1C CTX-OFF H4-K4 Linear
- meaningful_movement: True
- entropy_delta: 0.0017
- pairwise_delta: 0.0000
- short_rank_delta: 2.7756
- final_reader_context_overwrite_ratio: 0.0000
