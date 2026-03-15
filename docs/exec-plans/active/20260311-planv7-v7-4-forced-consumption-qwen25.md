# PLANv7 V7-4 Forced Memory Consumption and Minimal Receiver Reopening

## Purpose

Run the `PLANv7.md` `V7-4` matrix to test whether the current bottleneck has moved from writing to consuming after `V7-3` showed that a wide external Writer can be stabilized behind a bounded query-compression bridge without yet moving GSM8K or TriviaQA.

## Context

- `V7-3` published `comparison_conclusion=bridge_stabilizes_wide_writer_tasks_flat_move_to_v7_4`.
- The governed bridge ranking put `b_w3_q16` first, with `D1 / mid4` as the active depth.
- `PLANv7` fixes the `V7-4` matrix to:
  - `F0` control using the best branch from `V7-2` or `V7-3`,
  - `F1_num_mask` on `GSM8K`,
  - `F2_rx_only` on `GSM8K` and `TriviaQA`,
  - `F3_anneal` on `GSM8K`,
  - `F4_dyn_budget` on `GSM8K` and `TriviaQA`.
- The phase acceptance is strict: only actual primary-task score changes count as promotable; logprob-only movement is diagnostic only.

## Plan Of Work

1. Patch the runtime only where `V7-4` truly needs new capability:
   - keep `F1` split prompt routing,
   - keep `F2` staged `receiver_then_joint`,
   - add the missing train-only `F3` starvation annealing schedule,
   - keep `F4` as task-local config materialization rather than a broad runtime rewrite.
2. Add focused tests for the starvation-anneal prompt schedule and the new `V7-4` summary contract.
3. Add the governed `V7-4` runner and summary script.
4. Validate with static checks, focused tests, and the full unit suite.
5. Launch the full `V7-4` matrix in detached `tmux`.
6. Publish the governed summary, commit, push, and refresh the review branch after the phase completes.
7. If the governed summary authorizes it, continue directly into `V7-5`.

## Concrete Steps

1. Read `results/generated/review/planv7-v7-3-bridge-qwen25/v7-3-summary.json` and use its top bridge arm as the default `V7-4` control source; fall back to the recorded direct control only if the bridge ranking is missing.
2. Keep the phase on the `V7-3` winning depth.
3. Materialize configs for:
   - `control` on `gsm8k`, `triviaqa`
   - `f1_num_mask` on `gsm8k`
   - `f2_rx_only` on `gsm8k`, `triviaqa`
   - `f3_anneal` on `gsm8k`
   - `f4_dyn_budget` on `gsm8k`, `triviaqa`
4. Run all suites with the canonical `PLANv7` fixed settings:
   - owner projector LR override,
   - groupwise clipping,
   - gradient accumulation `4`,
   - train steps `300`,
   - active support/context/aux families `S3 + C2 + L5`,
   - `D1 / mid4` depth unless the governed input summary says otherwise.
5. Publish:
   - `v7-4-summary.json`
   - `v7-4-summary.md`
6. Sync review artifacts and push the milestone after publication.

## Validation And Acceptance

`V7-4` is complete only if:

- the matched control finishes on both primary tasks,
- `F1`, `F2`, `F3`, and `F4` finish on their required task subsets,
- the summary distinguishes actual score-changing branches from diagnostic-only branches,
- the promoted `V7-5` base arm is recorded explicitly,
- and the phase is committed, pushed, and republished to the lightweight `review` branch.

Validation commands:

```bash
bash -n \
  scripts/run_planv7_v7_4_forced_consumption_qwen25.sh \
  scripts/publish_review_artifacts.sh
python -m py_compile \
  src/memtotal/training/m4_shared_injection.py \
  scripts/update_planv7_v7_4_forced_consumption_summary.py \
  tests/test_m4_shared_injection.py \
  tests/test_planv7_v7_4_forced_consumption_summary.py
python -m unittest \
  tests.test_m4_shared_injection \
  tests.test_planv7_v7_4_forced_consumption_summary \
  -v
python -m unittest discover -s tests -v
```

Live run command:

```bash
tmux new-session -d -s planv7_v74 \
  "cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_v7_4_forced_consumption_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-v7-4-forced-consumption-qwen25 \
     /root/autodl-tmp/results/generated/planv7-v7-4-forced-consumption-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     300 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
     results/generated/review/planv7-v7-3-bridge-qwen25/v7-3-summary.json \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-v7-4-forced-consumption-qwen25/tmux-session.log"
```

## Progress

- 2026-03-11 22:06 UTC: Re-read `PLANv7.md` `V7-4` and confirmed the phase keeps the bridge winner as the control source unless the bridge ranking is missing.
- 2026-03-11 22:10 UTC: Confirmed that `F1` prompt splitting and `F2` staged `receiver_then_joint` were already live from the `V7-3` overlap work.
- 2026-03-11 22:17 UTC: Added the missing train-only `F3` starvation annealing schedule to `src/memtotal/training/m4_shared_injection.py` and logged the active mask stage/fraction into `train_events.json`.
- 2026-03-11 22:18 UTC: Added focused unit coverage for the starvation schedule and the new prompt-view contract in `tests/test_m4_shared_injection.py`.
- 2026-03-11 22:26 UTC: Added the governed `V7-4` runner and summary script:
  - `scripts/run_planv7_v7_4_forced_consumption_qwen25.sh`
  - `scripts/update_planv7_v7_4_forced_consumption_summary.py`
- 2026-03-11 22:28 UTC: Added the summary-contract test `tests/test_planv7_v7_4_forced_consumption_summary.py`.
- 2026-03-11 22:29 UTC: Updated `scripts/publish_review_artifacts.sh` so `V7-4` results and run bundles publish into the governed review trees.
- 2026-03-11 22:29 UTC: Static checks and focused tests are passing; full-suite validation and the live matrix launch are next.
- 2026-03-11 22:31 UTC: Full validation passed:
  - `python -m unittest tests.test_m4_shared_injection tests.test_planv7_v7_4_forced_consumption_summary -v`
  - `python -m unittest discover -s tests -v`
  - full-suite result: `289 tests, OK`
- 2026-03-11 22:32 UTC: Launched the full `V7-4` matrix in detached `tmux` as `planv7_v74`.
- 2026-03-11 22:50 UTC: The full matrix finished and published the governed summary at `results/generated/planv7-v7-4-forced-consumption-qwen25/v7-4-summary.json`.
- 2026-03-11 22:50 UTC: Final phase outcome:
  - `comparison_conclusion=forced_consumption_diagnostic_only_move_to_v7_5`
  - `recommended_next_step=open_v7_5_targeted_aux_revisit`
  - control source arm: `b_w3_q16`
  - base for `V7-5`: remain on `b_w3_q16` from `V7-3`
- 2026-03-11 22:50 UTC: Governed readout:
  - all four forced-consumption families were route-live on their required tasks,
  - `F2_rx_only` ranked first by diagnostic evidence,
  - `F1_num_mask`, `F2_rx_only`, and `F3_anneal` all raised mean `delta_answer_logprob` on `GSM8K`,
  - but no branch changed actual primary-task score on `GSM8K` or `TriviaQA`,
  - therefore `V7-4` is diagnostic only and does not replace the `V7-3` bridge winner as the mainline base.

## Decision Log

- `F3` is implemented as a train-only schedule rather than a permanent eval-time prompt mask because `PLANv7` defines it as starvation annealing, not a new evaluation prompt format.
- `F4` is kept as task-local materialization:
  - `GSM8K` stays on the wide `64/16` bridge budget,
  - `TriviaQA` uses the smaller `32/8` dynamic budget profile.
- The `V7-4` summary promotes only branches that move actual task score. A logprob-only branch is recorded as diagnostic and can inform `V7-5`, but it is not allowed to replace the mainline on its own.
- The bridge winner remains the canonical base when `V7-4` is usefulness-flat, because `PLANv7` explicitly allows `V7-5` to start from the best branch from `V7-3` or `V7-4`.

## Surprises And Discoveries

- The existing `V7-3` overlap work was enough to make `F1` and `F2` live immediately; only `F3` required new runtime work.
- Because `V7-3` already recorded a top bridge arm while remaining usefulness-flat, `V7-4` can stay tightly scoped to consumption rather than reopening bridge width or projector sweeps.
- `F2_rx_only` produced the strongest diagnostic signal, but that signal stayed entirely in logprob/helpfulness space rather than converting into actual task-score movement.
