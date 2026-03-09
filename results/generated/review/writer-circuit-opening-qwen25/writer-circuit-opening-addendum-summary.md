# Writer Circuit Opening Addendum Summary

- comparison_conclusion: move_to_writer_direct_validation
- primary_interpretation: source_stub_route_live_but_not_usefulness_testable
- recommended_substrate: p2a_source_stub_receiver_lora_r2
- p1a_route_live_any_nonfever: True
- p2a_route_live_any_nonfever: True

## gsm8k
- p1a_route_live: True
- p1a_usefulness_positive: False
- p1a_source_grad_norm_steps_1_4_median: 0.241981
- p1a_prefix_attention_nontrivial_layer_count: 2
- p2a_route_live: True
- p2a_usefulness_positive: False
- p2a_source_grad_norm_steps_1_4_median: 0.530186
- p2a_receiver_lora_grad_norm_steps_1_4_median: 36.097210
- p2a_prefix_attention_nontrivial_layer_count: 4
- preferred_route: p2a

## narrativeqa
- p1a_route_live: False
- p1a_usefulness_positive: False
- p1a_source_grad_norm_steps_1_4_median: 0.397231
- p1a_prefix_attention_nontrivial_layer_count: 0
- p2a_route_live: False
- p2a_usefulness_positive: False
- p2a_source_grad_norm_steps_1_4_median: 0.496999
- p2a_receiver_lora_grad_norm_steps_1_4_median: 134.020767
- p2a_prefix_attention_nontrivial_layer_count: 0
- preferred_route: none

## fever
- p1a_route_live: True
- p1a_usefulness_positive: False
- p1a_source_grad_norm_steps_1_4_median: 0.085831
- p1a_prefix_attention_nontrivial_layer_count: 2
- p2a_route_live: True
- p2a_usefulness_positive: False
- p2a_source_grad_norm_steps_1_4_median: 0.381479
- p2a_receiver_lora_grad_norm_steps_1_4_median: 47.329506
- p2a_prefix_attention_nontrivial_layer_count: 3
- preferred_route: p2a
