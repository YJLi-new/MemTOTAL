# M4 Shared Injection Compare

- gate_passed: False
- regressions_vs_base: 9
- flip_gain_vs_shuffle: 0
- flip_gain_vs_zero: 3
- macro_f1_gap_vs_shuffle: 0.0071
- macro_f1_gap_vs_zero: 0.0667

## Arm Summary
- A: task_score=0.2944, macro_f1=0.1516, mean_margin=-2.6675, dominant_label_fraction=1.0000
- I_real: task_score=0.3065, macro_f1=0.2183, mean_margin=-2.1174, dominant_label_fraction=0.8871
- I_shuffle: task_score=0.3065, macro_f1=0.2113, mean_margin=-2.1598, dominant_label_fraction=0.9113
- I_zero: task_score=0.2944, macro_f1=0.1516, mean_margin=-1.8888, dominant_label_fraction=1.0000
- T: task_score=0.5121, macro_f1=0.4680, mean_margin=0.5438, dominant_label_fraction=0.7782

## Pairwise Compare
- A -> T: flip_delta=54, left_wrong_to_right_correct=54, left_correct_to_right_wrong=0, mean_margin_gain=3.2113
- A -> I_real: flip_delta=3, left_wrong_to_right_correct=12, left_correct_to_right_wrong=9, mean_margin_gain=0.5501
- I_shuffle -> I_real: flip_delta=0, left_wrong_to_right_correct=7, left_correct_to_right_wrong=7, mean_margin_gain=0.0424
- I_zero -> I_real: flip_delta=3, left_wrong_to_right_correct=12, left_correct_to_right_wrong=9, mean_margin_gain=-0.2287
