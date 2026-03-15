# M4 Shared Injection Compare

- gate_passed: False
- regressions_vs_base: 15

## Arm Summary
- A: task_score=0.2500, macro_f1=0.2000, mean_margin=-2.5576, dominant_label_fraction=1.0000
- I_real: task_score=0.1094, macro_f1=0.1571, mean_margin=-2.5478, dominant_label_fraction=0.8125
- I_shuffle: task_score=0.4531, macro_f1=0.4618, mean_margin=-0.5860, dominant_label_fraction=0.4375
- I_zero: task_score=0.2500, macro_f1=0.2000, mean_margin=-7.5370, dominant_label_fraction=1.0000
- T: task_score=0.5312, macro_f1=0.5294, mean_margin=0.1045, dominant_label_fraction=0.6875

## Pairwise Compare
- A -> T: flip_delta=18, left_wrong_to_right_correct=19, left_correct_to_right_wrong=1, mean_margin_gain=2.6622
- A -> I_real: flip_delta=-9, left_wrong_to_right_correct=6, left_correct_to_right_wrong=15, mean_margin_gain=0.0099
- I_shuffle -> I_real: flip_delta=-22, left_wrong_to_right_correct=2, left_correct_to_right_wrong=24, mean_margin_gain=-1.9618
- I_zero -> I_real: flip_delta=-9, left_wrong_to_right_correct=6, left_correct_to_right_wrong=15, mean_margin_gain=4.9892
