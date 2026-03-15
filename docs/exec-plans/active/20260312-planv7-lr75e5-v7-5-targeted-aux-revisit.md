# PLANv7 (LR updated version) V7-5 Targeted Auxiliary Revisit

## Purpose

Queue restarted `V7-5` behind LR-updated `V7-4` so the targeted auxiliary revisit starts automatically if the forced-consumption phase publishes an authorized handoff under projector LR `7.5e-5`.

## Context

- LR-updated `V7-4` is now running under:
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-4-forced-consumption-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-4-forced-consumption-qwen25`
- Historical `PLANv7` advanced from `V7-4` into `V7-5`, but the restart line should only do that if the new `V7-4` summary publishes the same authorization.
- The restart wrapper already exists at:
  - [`scripts/run_planv7_lr75e5_v7_5_targeted_aux_revisit_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_lr75e5_v7_5_targeted_aux_revisit_qwen25.sh)

## Plan Of Work

1. Wait for restarted `V7-4` to emit `v7-4-summary.json`.
2. Wait for the `V7-4` post-publish/push session to finish and for `main` to move off the pre-`V7-4` head.
3. Parse the governed `V7-4` summary and confirm it authorizes `V7-5`.
4. Launch restarted `V7-5` in its own namespace.
5. Arm a detached post-completion publisher for the `V7-5` milestone.

## Concrete Steps

1. Watch:
   - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-4-forced-consumption-qwen25/v7-4-summary.json`
2. Guard launch on:
   - `planv7_lr75e5_v74_post` exiting
   - GitHub `main` moving off `2e3efc4893ca0d33a8fb178886335cddd75ba850`
3. Launch:
   - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25`
   - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25`
4. Pass the direct `V7-4` result-root summary path to the wrapper.
5. Arm detached `planv7_lr75e5_v75_post` to publish review artifacts, commit the milestone, push `main`, and refresh `review`.

## Validation & Acceptance

Acceptance for this queued handoff:

- `V7-5` does not start before `V7-4` summary publication,
- `V7-5` does not start before the `V7-4` post-publish push completes,
- the queue parses `recommended_next_step` from the governed `V7-4` summary before launching,
- the run/result roots use the `planv7-lr75e5-v7-5-*` namespace,
- a detached `V7-5` post-publisher is armed when launch occurs.

## Progress

- 2026-03-12 UTC: Opened the restart-specific `V7-5` targeted-auxiliary queue after LR-updated `V7-4` launched.
- 2026-03-12 UTC: Added the guarded queue helper [`queue_planv7_lr75e5_v7_5_after_v7_4.sh`](/root/mydir/MemTOTAL/scripts/queue_planv7_lr75e5_v7_5_after_v7_4.sh) and validated it with:
  - `bash -n scripts/queue_planv7_lr75e5_v7_5_after_v7_4.sh scripts/run_planv7_lr75e5_v7_5_targeted_aux_revisit_qwen25.sh scripts/publish_review_artifacts.sh`
- 2026-03-12 UTC: Armed detached queue session `planv7_lr75e5_v75_queue`, which will:
  - wait for `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-4-forced-consumption-qwen25/v7-4-summary.json`
  - wait for `planv7_lr75e5_v74_post` to exit
  - require GitHub `main` to move off `2e3efc4893ca0d33a8fb178886335cddd75ba850`
  - parse `recommended_next_step` and only launch `V7-5` if it equals `open_v7_5_targeted_aux_revisit`
  - arm detached `planv7_lr75e5_v75_post` for milestone publication/push
- 2026-03-12 UTC: After `V7-4` pushed, inspection showed the live queue had actually been launched with the wrong predecessor argument (`0a05e04...`, the post-`V7-4` head), so it was waiting forever for `main` to move again. The queue is being re-armed with the correct pre-`V7-4` sentinel `2e3efc4893ca0d33a8fb178886335cddd75ba850`.
- 2026-03-12 UTC: Re-ran the corrected queue helper directly. It passed the existing `V7-4` summary and GitHub-head checks immediately, then launched:
  - `planv7_lr75e5_v75`
  - `planv7_lr75e5_v75_post`
- 2026-03-12 UTC: Current live state:
  - dataset/materialization step is active under `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25`
  - `tmux-session.log` is being written at the run root
  - the post-publisher is waiting on `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json`

## Decision Log

- Keep the restart line phase-accurate: `V7-5` should only open from the governed `V7-4` decision point, not from historical expectation alone.
- Use the current `main` head `2e3efc4893ca0d33a8fb178886335cddd75ba850` as the pre-`V7-4` push sentinel for the guarded queue.
- Carry the relay-script repairs in the `V7-5` milestone commit so the unattended chain is reproducible from the repo state instead of depending on local uncommitted fixes.

## Surprises & Discoveries

- None yet; this document is the queued handoff before `V7-4` closeout.
