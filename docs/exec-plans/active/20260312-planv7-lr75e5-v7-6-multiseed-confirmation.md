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

## Decision Log

- Honor the `V7-5` summary contract literally: the launch gate is `prepare_v7_6_decision_point`, not a guessed `open_v7_6_*` string.
- Require `base_for_v7_6_arm_id` to be non-empty so the multiseed selection manifest can be built on the intended promoted arm.

## Surprises & Discoveries

- The later-phase restart line shares the same embedded-Python structure as the earlier phases, so proactive harness checks are cheaper than phase-by-phase rediscovery.
