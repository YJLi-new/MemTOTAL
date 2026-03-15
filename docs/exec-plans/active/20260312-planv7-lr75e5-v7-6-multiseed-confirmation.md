# PLANv7 (LR updated version) V7-6 Multiseed Confirmation

## Purpose

Queue restarted `V7-6` behind LR-updated `V7-5` so the three-seed confirmation starts automatically if the targeted auxiliary revisit publishes the expected decision-point handoff under projector LR `7.5e-5`.

## Context

- LR-updated `V7-5` will run under:
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25`
- Historical `PLANv7` advanced from `V7-5` into `V7-6` when the revisit reached the decision point.
- The historical `V7-5` governed summary contract is:
  - `comparison_conclusion=aux_revisit_flat_best_branch_for_decision_point`
  - `recommended_next_step=prepare_v7_6_decision_point`
  - `base_for_v7_6_arm_id=a5_barlow`
- The restart wrapper already exists at:
  - [`scripts/run_planv7_lr75e5_v7_6_multiseed_confirmation_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_lr75e5_v7_6_multiseed_confirmation_qwen25.sh)

## Plan Of Work

1. Wait for restarted `V7-5` to emit `v7-5-summary.json`.
2. Wait for the `V7-5` post-publish/push session to finish and for `main` to move off the pre-`V7-5` head.
3. Parse the governed `V7-5` summary and confirm it reaches the `V7-6` decision point.
4. Launch restarted `V7-6` in its own namespace.
5. Arm a detached post-completion publisher for the `V7-6` milestone.

## Concrete Steps

1. Watch:
   - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json`
2. Guard launch on:
   - `planv7_lr75e5_v75_post` exiting
   - GitHub `main` moving off `0a05e04a48a619e6e397d6e9bc39ba99c53504cc`
3. Launch:
   - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25`
   - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25`
4. Pass the direct `V7-5` result-root summary path to the wrapper, plus the LR-updated `V7-0` summary.
5. Arm detached `planv7_lr75e5_v76_post` to publish review artifacts, commit the milestone, push `main`, and refresh `review`.

## Validation & Acceptance

Acceptance for this queued handoff:

- `V7-6` does not start before `V7-5` summary publication,
- `V7-6` does not start before the `V7-5` post-publish push completes,
- the queue parses `recommended_next_step` from the governed `V7-5` summary before launching,
- the queue also requires a non-empty `base_for_v7_6_arm_id`,
- the run/result roots use the `planv7-lr75e5-v7-6-*` namespace,
- a detached `V7-6` post-publisher is armed when launch occurs.

## Progress

- 2026-03-12 UTC: Opened the restart-specific `V7-6` multiseed queue after confirming the historical `V7-5` decision-point contract.
- 2026-03-12 UTC: The queue was armed with pre-`V7-5` sentinel `0a05e04a48a619e6e397d6e9bc39ba99c53504cc`, which is correct for waiting on the upcoming `V7-5` milestone push.
- 2026-03-12 UTC: `V7-5` is now live in `planv7_lr75e5_v75` and the queue session `planv7_lr75e5_v76_queue` remains detached and waiting for:
  - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json`
  - `planv7_lr75e5_v75_post` to exit
  - GitHub `main` to move off `0a05e04a48a619e6e397d6e9bc39ba99c53504cc`
- 2026-03-12 UTC: LR-updated `V7-5` completed with governed handoff:
  - `comparison_conclusion=aux_revisit_flat_best_branch_for_decision_point`
  - `recommended_next_step=prepare_v7_6_decision_point`
  - `base_for_v7_6_arm_id=a5_barlow`
  - `main` moved to `2178f5af0cd2923e352897729701b02b387db3e8`
  - `review` moved to `ddd183f83a353682cc1e23b212ce739aa6cb9a26`
- 2026-03-12 UTC: The queue session was not present after the `V7-5` tmux server exited, so the phase is being launched directly from the validated `V7-5` handoff rather than through the original detached waiter.
- 2026-03-12 UTC: Launched LR-updated `V7-6` directly with:
  - `planv7_lr75e5_v76`
  - `planv7_lr75e5_v76_post`
  - run root `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25`
  - result root `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25`
- 2026-03-12 UTC: Initial live status:
  - the launcher is in `writer_jointpeft_data` materialization
  - the post-publisher is waiting on `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json`
- 2026-03-12 UTC: `V7-6` progressed past setup and into seeded suite execution. Early nested outputs are already present under:
  - `c0_frozen_no_memory/seed_61109/{gsm8k,triviaqa,fever}/suite_metrics.json`
  - `c1_additive_continuity/seed_61109/{gsm8k,triviaqa}/suite_metrics.json`
- 2026-03-12 UTC: Added detached monitoring session `planv7_lr75e5_v76_watch`, which appends five-minute snapshots to:
  - `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/watch.log`
  - first recorded sample: `suites=6`, `gpu=5715 MiB, util=55%`
- 2026-03-13 UTC: `V7-6` completed with all `45/45` nested suites present in the run root and the governed summary written to:
  - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json`
  - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.md`
- 2026-03-13 UTC: Final LR-updated closeout:
  - `comparison_conclusion=path_q_external_writer_unresolved_not_dead`
  - `recommended_next_step=open_stronger_integrated_writer_or_true_highdim_branch`
  - `best_confirmed_variant_id=p1_a5_barlow`
  - `winner_uses_bridge=true`
  - `winning_depth=D1`
- 2026-03-13 UTC: The detached post-publisher completed successfully and pushed:
  - `main`: `73e5a04b0a9f608cd76a31d01742fe5e801e8f06`
  - `review`: `078ec6e49c6f42e766f5d57af62afed1dee397a0`

## Decision Log

- Honor the `V7-5` summary contract literally: the launch gate is `prepare_v7_6_decision_point`, not a guessed `open_v7_6_*` string.
- Require `base_for_v7_6_arm_id` to be non-empty so the multiseed selection manifest can be built on the intended promoted arm.
- Treat the LR-updated `V7-6` outcome as the terminal decision point for this replay line; the next branch requires a successor plan, not a speculative `V7-7`.

## Surprises & Discoveries

- The later-phase restart line shares the same embedded-Python structure as the earlier phases, so proactive harness checks are cheaper than phase-by-phase rediscovery.
- The LR-`7.5e-5` replay did change the final decision materially enough to move the repo from historical `Path R` to replay `Path Q`, even though it still did not unlock consistent primary-task gains.
