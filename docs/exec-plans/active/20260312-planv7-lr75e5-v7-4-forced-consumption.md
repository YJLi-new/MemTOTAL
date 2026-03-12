# PLANv7 (LR updated version) V7-4 Forced-Consumption Stress Test

## Purpose

Queue restarted `V7-4` behind LR-updated `V7-3` so the forced-consumption stress test starts automatically if the bridge phase publishes an authorized handoff under projector LR `7.5e-5`.

## Context

- LR-updated `V7-3` is currently running under:
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-3-bridge-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-3-bridge-qwen25`
- Historical `PLANv7` advanced from `V7-3` into `V7-4`, but the restart line should only do that if the new `V7-3` summary publishes the same authorization.
- The restart wrapper already exists at:
  - [`scripts/run_planv7_lr75e5_v7_4_forced_consumption_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_lr75e5_v7_4_forced_consumption_qwen25.sh)

## Plan Of Work

1. Wait for restarted `V7-3` to emit `v7-3-summary.json`.
2. Wait for the `V7-3` post-publish/push session to finish and for `main` to move off the pre-`V7-3` head.
3. Parse the governed `V7-3` summary and confirm it authorizes `V7-4`.
4. Launch restarted `V7-4` in its own namespace.
5. Arm a detached post-completion publisher for the `V7-4` milestone.

## Concrete Steps

1. Watch:
   - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-3-bridge-qwen25/v7-3-summary.json`
2. Guard launch on:
   - `planv7_lr75e5_v73_post` exiting
   - GitHub `main` moving off `1a9c9ac18babeaff27402cc22eaf40daf714230e`
3. Launch:
   - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-4-forced-consumption-qwen25`
   - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-4-forced-consumption-qwen25`
4. Pass the direct `V7-3` result-root summary path to the wrapper.
5. Arm detached `planv7_lr75e5_v74_post` to publish review artifacts, commit the milestone, push `main`, and refresh `review`.

## Validation & Acceptance

Acceptance for this queued handoff:

- `V7-4` does not start before `V7-3` summary publication,
- `V7-4` does not start before the `V7-3` post-publish push completes,
- the queue parses `recommended_next_step` from the governed `V7-3` summary before launching,
- the run/result roots use the `planv7-lr75e5-v7-4-*` namespace,
- a detached `V7-4` post-publisher is armed when launch occurs.

## Progress

- 2026-03-12 UTC: Opened the restart-specific `V7-4` forced-consumption queue while LR-updated `V7-3` was still running.
- 2026-03-12 UTC: Added the guarded queue helper [`queue_planv7_lr75e5_v7_4_after_v7_3.sh`](/root/mydir/MemTOTAL/scripts/queue_planv7_lr75e5_v7_4_after_v7_3.sh) and validated it with:
  - `bash -n scripts/queue_planv7_lr75e5_v7_4_after_v7_3.sh scripts/run_planv7_lr75e5_v7_4_forced_consumption_qwen25.sh scripts/publish_review_artifacts.sh`
- 2026-03-12 UTC: Armed detached queue session `planv7_lr75e5_v74_queue`, which will:
  - wait for `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-3-bridge-qwen25/v7-3-summary.json`
  - wait for `planv7_lr75e5_v73_post` to exit
  - require GitHub `main` to move off `1a9c9ac18babeaff27402cc22eaf40daf714230e`
  - parse `recommended_next_step` and only launch `V7-4` if it equals `open_v7_4_forced_consumption`
  - arm detached `planv7_lr75e5_v74_post` for milestone publication/push

## Decision Log

- Treat `V7-4` as a guarded queue rather than launching speculatively; the restart line must still obey the `V7-3` decision point.
- Gate the queue on a changed GitHub `main` head so the next phase does not jump ahead of the milestone-push requirement.

## Surprises & Discoveries

- None yet; this document is the queued handoff before `V7-3` closeout.
