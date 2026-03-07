# M4 Shared Injection Compare

- gate_passed: False
- regressions_vs_base: 2

## Arm Summary
- A: task_score=0.2500, macro_f1=0.2000, mean_margin=-2.5576, dominant_label_fraction=1.0000
- I_real: task_score=0.3906, macro_f1=0.4061, mean_margin=-0.5086, dominant_label_fraction=0.7188
- I_shuffle: task_score=0.5469, macro_f1=0.5031, mean_margin=0.5708, dominant_label_fraction=0.5469
- I_zero: task_score=0.2500, macro_f1=0.2000, mean_margin=-7.5370, dominant_label_fraction=1.0000
- T: task_score=0.5312, macro_f1=0.5294, mean_margin=0.1045, dominant_label_fraction=0.6875

## Pairwise Compare
- A -> T: flip_delta=18, left_wrong_to_right_correct=19, left_correct_to_right_wrong=1, mean_margin_gain=2.6622
- A -> I_real: flip_delta=9, left_wrong_to_right_correct=11, left_correct_to_right_wrong=2, mean_margin_gain=2.0491
- I_shuffle -> I_real: flip_delta=-10, left_wrong_to_right_correct=11, left_correct_to_right_wrong=21, mean_margin_gain=-1.0794
- I_zero -> I_real: flip_delta=9, left_wrong_to_right_correct=11, left_correct_to_right_wrong=2, mean_margin_gain=7.0284
