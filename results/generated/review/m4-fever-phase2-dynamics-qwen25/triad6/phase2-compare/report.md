# M4 Shared Injection Compare

- gate_passed: False
- regressions_vs_base: 16

## Arm Summary
- A: task_score=0.2500, macro_f1=0.2000, mean_margin=-2.5576, dominant_label_fraction=1.0000
- I_real: task_score=0.6875, macro_f1=0.4112, mean_margin=1.1343, dominant_label_fraction=0.9219
- I_shuffle: task_score=0.6875, macro_f1=0.6135, mean_margin=0.4602, dominant_label_fraction=0.6875
- I_zero: task_score=0.2500, macro_f1=0.2000, mean_margin=-7.5370, dominant_label_fraction=1.0000
- T: task_score=0.5781, macro_f1=0.5755, mean_margin=0.4163, dominant_label_fraction=0.6719

## Pairwise Compare
- A -> T: flip_delta=21, left_wrong_to_right_correct=21, left_correct_to_right_wrong=0, mean_margin_gain=2.9740
- A -> I_real: flip_delta=28, left_wrong_to_right_correct=44, left_correct_to_right_wrong=16, mean_margin_gain=3.6919
- I_shuffle -> I_real: flip_delta=0, left_wrong_to_right_correct=11, left_correct_to_right_wrong=11, mean_margin_gain=0.6741
- I_zero -> I_real: flip_delta=28, left_wrong_to_right_correct=44, left_correct_to_right_wrong=16, mean_margin_gain=8.6712
