# Stage C Error Attribution

- near-threshold margin window: [-0.02, 0)

## Backbone Summary

- Qwen2.5-1.5B-Instruct: paired=32, near_threshold=1, improving_unflipped=9, cross_zero=0
- Qwen3-8B: paired=29, near_threshold=1, improving_unflipped=1, cross_zero=0

## Top Near-Threshold Cases

- Qwen2.5-1.5B-Instruct seed=55205 example=2758e7c5-6100-4182-99d3-e7d9e435a394 margin=-0.0141->-0.0132 proxy=0.4965->0.4967 tags=improving_but_unflipped|near_rank_flip|proxy_up_without_flip|story_context_favors_competitor
  gold: A Ned decided to come back for a visit.
  competitor: B Ned was livid the neighborhood celebrated Halloween.
  support_ids: b73dcd4d-e4ad-498d-ac8f-a31a7e4bc969|b98c114b-d657-49d2-9bd1-912438fadd68|ced7d97e-c873-448c-bce0-30274ceac332
- Qwen3-8B seed=55303 example=b73dcd4d-e4ad-498d-ac8f-a31a7e4bc969 margin=-0.0115->-0.0115 proxy=0.4971->0.4971 tags=near_rank_flip|story_context_ambiguous
  gold: A He had to buy new ones.
  competitor: B He went running.
  support_ids: 0faf4829-3b27-42d3-a69f-2d6036b110f3|b98c114b-d657-49d2-9bd1-912438fadd68|f2420b66-049f-4615-b2ba-51a0f6d63c49
