# M4 Shared Injection Compare

- gate_passed: False
- regressions_vs_base: 0

## Arm Summary
- A: task_score=0.3333, macro_f1=0.2500, mean_margin=-0.0001, dominant_label_fraction=1.0000
- I_real: task_score=0.3333, macro_f1=0.2500, mean_margin=0.0000, dominant_label_fraction=1.0000
- I_shuffle: task_score=0.3333, macro_f1=0.2500, mean_margin=0.0000, dominant_label_fraction=1.0000
- I_zero: task_score=0.3333, macro_f1=0.2500, mean_margin=0.0000, dominant_label_fraction=1.0000
- T: task_score=0.3333, macro_f1=0.2500, mean_margin=-0.0016, dominant_label_fraction=1.0000

## Pairwise Compare
- A -> T: flip_delta=0, left_wrong_to_right_correct=0, left_correct_to_right_wrong=0, mean_margin_gain=-0.0014
- A -> I_real: flip_delta=0, left_wrong_to_right_correct=0, left_correct_to_right_wrong=0, mean_margin_gain=0.0001
- I_shuffle -> I_real: flip_delta=0, left_wrong_to_right_correct=0, left_correct_to_right_wrong=0, mean_margin_gain=0.0000
- I_zero -> I_real: flip_delta=0, left_wrong_to_right_correct=0, left_correct_to_right_wrong=0, mean_margin_gain=0.0000
