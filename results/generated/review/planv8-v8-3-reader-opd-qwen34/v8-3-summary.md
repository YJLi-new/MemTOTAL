# PLANv8 V8-3 Reader OPD

## Decision
- `comparison_conclusion = reader_opd_flat_open_v8_7_comparators`
- `recommended_next_step = open_v8_7_comparators`
- `best_arm_id = p5_opd_ansplusctx_centered`
- `best_alignment_aux_mode = opd_token_ce_centered`
- `base_for_v8_4_arm_id = p5_opd_ansplusctx_centered`

## Ranking
- `p5_opd_ansplusctx_centered` mode=opd_token_ce_centered score_sum=0.078125 logprob_sum=2.679495 activation=0.000142 opd_pos_frac=0.000000 acceptance=False
- `p3_opd_ansonly_w03` mode=opd_token_ce score_sum=0.046875 logprob_sum=0.073294 activation=0.000143 opd_pos_frac=0.000000 acceptance=False
- `p2_opd_ansonly_w01` mode=opd_token_ce score_sum=0.046875 logprob_sum=-1.233578 activation=0.000142 opd_pos_frac=0.000000 acceptance=False
- `p1_teacher_choice_kl` mode=teacher_choice_kl score_sum=0.031250 logprob_sum=1.380064 activation=0.000142 opd_pos_frac=0.000000 acceptance=False
- `p0_ce_only` mode=off score_sum=0.000000 logprob_sum=0.000000 activation=0.000142 opd_pos_frac=0.000000 acceptance=False
- `p4_opd_ansplusctx_w03` mode=opd_token_ce score_sum=-0.015625 logprob_sum=-1.480470 activation=0.000142 opd_pos_frac=0.000000 acceptance=False

## Control
- `gsm8k` control score = 0.781250
- `triviaqa` control score = 0.015625
- `fever` control score = 0.875000

## Notes
- TriviaQA answer-plus-context arms can legitimately fall back to answer-only when the selected source lacks evidence sentences.
