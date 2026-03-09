# PLANv6 Phase V6-1 - Clean Baseline Rerun Of The Current Writer-Direct Architecture

## Purpose

Run the first clean post-`V6-0` baseline so the repo can measure the legacy `pooled_block` route without the writer-side auxiliary-loss confound.

## Context

- `PLANv6.md` Phase `V6-0` is complete and pushed.
- The next required step is `V6-1`: keep the existing Writer-direct deep-prefix bridge, but rerun it under `L0 task-only` with a short Writer warm-start and the repaired `PLANv6` gates.
- This phase is still about fair measurement, not support-interface innovation yet.

## Canonical Branch

- support mode: `S0 pooled_block_legacy`
- stimulus mix: `C1 support_and_context`
- auxiliary family: `L0 task-only`
- Writer freeze: `F10`
- bridge: current early deep-prefix `[0,1,2,3]` plus early4 receiver micro-LoRA
- recipe repairs carried from `PLANv6`:
  - lower projector LR,
  - group-wise clipping,
  - gradient attribution probes

## Plan Of Work

1. Add a dedicated reproducible runner for the `V6-1` clean baseline.
2. Publish the run under a separate artifact root from the old joint-PEFT branch.
3. Reuse the repaired `PLANv6` summary logic so the phase reports:
   - `route_live_post_unfreeze`
   - `writer_task_supervision_live`
   - `source_not_collapsed`
   - `stable_training_v6`
   - `usefulness_positive_v6`
4. Run the task ladder:
   - FEVER as the primary proof task
   - NarrativeQA and GSM8K as diagnostics

## Concrete Steps

1. Add [`scripts/run_planv6_probe_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv6_probe_qwen25.sh).
2. Sync the new result bundle through [`scripts/publish_review_artifacts.sh`](/root/mydir/MemTOTAL/scripts/publish_review_artifacts.sh).
3. Execute the run into:
   - `runs/verify/planv6-v6-1-clean-baseline-qwen25`
   - `results/generated/planv6-v6-1-clean-baseline-qwen25`
4. Publish the final repo-review bundle under:
   - `runs/review/planv6-v6-1-clean-baseline-qwen25`
   - `results/generated/review/planv6-v6-1-clean-baseline-qwen25`

## Validation And Acceptance

Phase `V6-1` is complete only if all of the following are true:

- the run uses `L0 task-only` with all writer-side auxiliary weights disabled,
- the summary is produced with repaired post-unfreeze gates,
- FEVER, NarrativeQA, and GSM8K artifacts are published,
- the phase yields a real post-unfreeze route verdict plus task-vs-total Writer gradient attribution.

## Progress

- `2026-03-09 22:09 UTC`: opened the `V6-1` exec plan after pushing `V6-0`.
- `2026-03-09 22:12 UTC`: added the dedicated runner and review-bundle sync path for the clean `S0/C1/L0/F10` rerun.
- `2026-03-09 21:50-22:07 UTC`: executed the full `V6-1` task ladder on Qwen2.5 and published the review bundle under `results/generated/review/planv6-v6-1-clean-baseline-qwen25`.
- `2026-03-09 22:11 UTC`: corrected a sparse-probe summarization bug discovered during the live run. Probe medians now use only `gradient_probe_step_active=true` steps, and the `V6-1` summary was regenerated from the finished run without retraining.

## Outcome

- `comparison_conclusion = move_to_writer_usefulness_branch`
- `recommended_next_step = open_writer_usefulness_branch`
- `any_nonfever_route_live = true`
- `any_nonfever_task_supervision_live = true`
- `any_nonfever_source_not_collapsed = true`
- `any_nonfever_stable_training = true`
- `any_nonfever_usefulness_positive = true`

Task notes:

- `gsm8k`: route live, task-supervision live, stable, usefulness-positive, but source collapse still unresolved.
- `narrativeqa`: task-supervision live and source geometry improved, but route/stability/usefulness did not pass.
- `fever`: route live, task-supervision live, stable, usefulness-positive, with strong accuracy and margin gains.

## Decision Log

- Keep `S0/C1/L0/F10` as the first clean rerun before support-interface screening.
- Preserve the existing data materialization path for comparability with the earlier joint-PEFT branch.
- Use the repaired summary script rather than inventing a second incompatible gate implementation.
- The first live `V6-1` run exposed a second-order measurement bug: sparse gradient probes were being summarized across non-probe steps, forcing false zeros. Fix that before interpreting the phase.
