# FEVER Real Pilot Content Audit

- examples=64
- best_of_BF_flip_gain=2
- best_of_B_plus_content_flip_gain=0
- oracle_per_case_alpha_content_flip_gain=0
- content_alignment_rate_shared_wrong=0.6250
- weighted_content_alignment_shared_wrong=0.0000
- continue_candidate_branch=False

## Summary
- A: task_score=0.2500, proxy=0.3361, margin=-0.0001
- B: task_score=0.7500, proxy=0.3374, margin=0.0045
- F: task_score=0.6719, proxy=0.3370, margin=0.0027
- G: task_score=0.6719, proxy=0.3370, margin=0.0027
- B_plus_content: task_score=0.7500, proxy=0.3374, margin=0.0045
- oracle_best_of_BF: task_score=0.7812, proxy=0.3374, margin=0.0047
- oracle_best_of_B_plus_content: task_score=0.7500, proxy=0.3374, margin=0.0045
- oracle_per_case_alpha_content: task_score=0.7500, proxy=0.3374, margin=0.0048

## By Bucket
- base_correct_control: B=0.0000, F=0.1250, G=0.1250, B_plus_content=0.0000, oracle_alpha=0.0000
- near_threshold_bad: B=1.0000, F=0.8750, G=0.8750, B_plus_content=1.0000, oracle_alpha=1.0000
- stubborn_wrong: B=1.0000, F=0.8125, G=0.8125, B_plus_content=1.0000, oracle_alpha=1.0000
