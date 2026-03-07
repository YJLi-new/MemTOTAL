# M4 Shared Injection Compare

- gate_passed: False
- regressions_vs_base: 0

## Arm Summary
- A: task_score=0.2500, macro_f1=0.2000, mean_margin=-2.5576, dominant_label_fraction=1.0000
- I_real: task_score=0.2500, macro_f1=0.2000, mean_margin=0.0000, dominant_label_fraction=1.0000
- I_shuffle: task_score=0.2500, macro_f1=0.2000, mean_margin=0.0000, dominant_label_fraction=1.0000
- I_zero: task_score=0.2500, macro_f1=0.2000, mean_margin=0.0000, dominant_label_fraction=1.0000
- T: task_score=0.5312, macro_f1=0.5294, mean_margin=0.1045, dominant_label_fraction=0.6875

## Pairwise Compare
- A -> T: flip_delta=18, left_wrong_to_right_correct=19, left_correct_to_right_wrong=1, mean_margin_gain=2.6622
- A -> I_real: flip_delta=0, left_wrong_to_right_correct=0, left_correct_to_right_wrong=0, mean_margin_gain=2.5576
- I_shuffle -> I_real: flip_delta=0, left_wrong_to_right_correct=0, left_correct_to_right_wrong=0, mean_margin_gain=0.0000
- I_zero -> I_real: flip_delta=0, left_wrong_to_right_correct=0, left_correct_to_right_wrong=0, mean_margin_gain=0.0000
