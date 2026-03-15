# PLANv8 Qwen3-4B: Repair Direct `V8-3 -> V8-7 -> V8-8` Route

## Why

The completed qwen34 `V8-3` summary selected:

- `comparison_conclusion = reader_opd_flat_open_v8_7_comparators`
- `recommended_next_step = open_v8_7_comparators`

The existing qwen34 downstream harness was still wired for the older `V8-6 -> V8-7` promotion path, so no comparator run started after `V8-3` completed.

## Repair Scope

1. Allow `V8-7` comparator promotion to import the best current qwen34 route from either:
   - `V8-3` Reader-only OPD, or
   - `V8-6` Writer-aux, when that phase actually exists.
2. Repoint the qwen34 `V8-7` queue from `V8-6` gating to the completed `V8-3` summary.
3. Make qwen34 `V8-8` selection-manifest generation robust when only the `V8-3` route exists, so the repair path does not require missing `V8-5` or `V8-6` summaries.
4. Clear the inherited localhost proxy in qwen34 `V8-7` to `V8-9` post-publish jobs and in the review-snapshot pusher so successful milestones can actually push.

## Governed Outputs

- `scripts/run_planv8_v8_7_comparators.sh`
- `scripts/run_planv8_v8_7_comparators_qwen34.sh`
- `scripts/update_planv8_v8_7_summary.py`
- `scripts/queue_planv8_qwen34_v8_7_after_v8_3.sh`
- `scripts/planv8_v8_8_selection_manifest.py`
- `scripts/run_planv8_v8_8_multiseed_confirmation.sh`
- `scripts/queue_planv8_qwen34_v8_8_after_v8_7.sh`
- `scripts/queue_planv8_qwen34_v8_9_after_v8_8.sh`
- `scripts/push_github_review_snapshot.sh`
- `scripts/arm_planv8_qwen34_chain.sh`

## Validation

- `python -m py_compile` on the repaired Python scripts
- focused `unittest` coverage for `V8-7` comparator import and `V8-8` selection-manifest logic
- `bash -n` on the repaired shell scripts
- `tests.test_repo_lints`
- `tests.test_repo_contract`
