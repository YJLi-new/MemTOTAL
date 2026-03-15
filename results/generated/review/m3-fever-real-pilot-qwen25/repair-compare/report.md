# FEVER Real Pilot Compare (Repair Objective)

## Arm Summary
- B_old: score=0.7500, proxy=0.3374, margin=0.0045, best_step=0
- B_new: score=0.7500, proxy=0.3374, margin=0.0045, best_step=0
- R_real: score=0.2500, proxy=0.3352, margin=-0.0060, best_step=0
- R_shuffle: score=0.2500, proxy=0.3352, margin=-0.0060, best_step=0
- R_zero: score=0.2500, proxy=0.3358, margin=-0.0048, best_step=0

## Pairwise Deltas
- B_old -> B_new: flip_count_delta=0, mean_task_gain=0.0000, mean_margin_gain=0.0000
- B_new -> R_real: flip_count_delta=-32, mean_task_gain=-0.5000, mean_margin_gain=-0.0105
- R_shuffle -> R_real: flip_count_delta=0, mean_task_gain=0.0000, mean_margin_gain=0.0000
- R_zero -> R_real: flip_count_delta=0, mean_task_gain=0.0000, mean_margin_gain=-0.0012

## Bucket Summary
- B_old / : task_score=0.7500, proxy=0.3374, margin=0.0045
- B_new / base_correct_control: task_score=0.0000, proxy=0.3354, margin=-0.0083
- B_new / near_threshold_bad: task_score=1.0000, proxy=0.3381, margin=0.0087
- B_new / stubborn_wrong: task_score=1.0000, proxy=0.3379, margin=0.0088
- R_real / base_correct_control: task_score=1.0000, proxy=0.3386, margin=0.0131
- R_real / near_threshold_bad: task_score=0.0000, proxy=0.3342, margin=-0.0123
- R_real / stubborn_wrong: task_score=0.0000, proxy=0.3339, margin=-0.0124
- R_shuffle / base_correct_control: task_score=1.0000, proxy=0.3386, margin=0.0131
- R_shuffle / near_threshold_bad: task_score=0.0000, proxy=0.3342, margin=-0.0123
- R_shuffle / stubborn_wrong: task_score=0.0000, proxy=0.3339, margin=-0.0124
- R_zero / base_correct_control: task_score=1.0000, proxy=0.3386, margin=0.0108
- R_zero / near_threshold_bad: task_score=0.0000, proxy=0.3349, margin=-0.0100
- R_zero / stubborn_wrong: task_score=0.0000, proxy=0.3347, margin=-0.0101

## Repair Bucket Summary
- B_new / anchor_confident_correct: task_score=1.0000, margin=0.0105, repair_active_rate=0.0000
- B_new / anchor_near_threshold: task_score=1.0000, margin=0.0085, repair_active_rate=1.0000
- B_new / anchor_wrong: task_score=0.0000, margin=-0.0083, repair_active_rate=1.0000
- R_real / anchor_confident_correct: task_score=0.0000, margin=-0.0100, repair_active_rate=0.0000
- R_real / anchor_near_threshold: task_score=0.0000, margin=-0.0127, repair_active_rate=1.0000
- R_real / anchor_wrong: task_score=1.0000, margin=0.0131, repair_active_rate=1.0000
- R_shuffle / anchor_confident_correct: task_score=0.0000, margin=-0.0100, repair_active_rate=0.0000
- R_shuffle / anchor_near_threshold: task_score=0.0000, margin=-0.0127, repair_active_rate=1.0000
- R_shuffle / anchor_wrong: task_score=1.0000, margin=0.0131, repair_active_rate=1.0000
- R_zero / anchor_confident_correct: task_score=0.0000, margin=-0.0075, repair_active_rate=0.0000
- R_zero / anchor_near_threshold: task_score=0.0000, margin=-0.0104, repair_active_rate=1.0000
- R_zero / anchor_wrong: task_score=1.0000, margin=0.0108, repair_active_rate=1.0000

## Gate
- gate_passed=False
- real_vs_shuffle_flip_count_delta=0, real_vs_zero_flip_count_delta=0, anchor_vs_real_regression_count=48
- real_vs_shuffle_alignment_rate=0.5625, real_vs_shuffle_weighted_alignment=0.0000
