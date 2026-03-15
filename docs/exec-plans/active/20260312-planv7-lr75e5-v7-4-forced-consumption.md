# PLANv7 (LR updated version) V7-4 Forced-Consumption Stress Test

## Purpose

Run restarted `V7-4` behind LR-updated `V7-3` so the forced-consumption stress test executes under projector LR `7.5e-5`, while preserving a guarded handoff contract from the governed `V7-3` summary.

## Context

- LR-updated `V7-3` completed and published under:
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-3-bridge-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-3-bridge-qwen25`
- Historical `PLANv7` advanced from `V7-3` into `V7-4`, but the restart line should only do that if the new `V7-3` summary publishes the same authorization.
- The restart wrapper already exists at:
  - [`scripts/run_planv7_lr75e5_v7_4_forced_consumption_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_lr75e5_v7_4_forced_consumption_qwen25.sh)
- The governing `V7-3` summary did authorize this handoff:
  - `comparison_conclusion=bridge_stabilizes_wide_writer_tasks_flat_move_to_v7_4`
  - `recommended_next_step=open_v7_4_forced_consumption`

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
- 2026-03-12 UTC: `V7-3` completed and pushed, but the first `V7-4` launch failed before the first suite with `NameError: name 'os' is not defined` in the base forced-consumption runner's second embedded Python block.
- 2026-03-12 UTC: Patched the missing import in [`run_planv7_v7_4_forced_consumption_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_v7_4_forced_consumption_qwen25.sh) and widened the `V7-4` post-publisher add-set so the repair and the queued `V7-5` relay will be included in the `V7-4` milestone push.
- 2026-03-12 UTC: Re-launched restarted `V7-4` manually on the repaired base script. Current live state:
  - detached sessions alive: `planv7_lr75e5_v74`, `planv7_lr75e5_v74_post`
  - `writer_jointpeft_data` is materializing the `gsm8k,triviaqa` bundle
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-4-forced-consumption-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-4-forced-consumption-qwen25`

## Decision Log

- Treat `V7-4` as a guarded queue rather than launching speculatively; the restart line must still obey the `V7-3` decision point.
- Gate the queue on a changed GitHub `main` head so the next phase does not jump ahead of the milestone-push requirement.
- After the first launch bug, re-use the same namespace and post-publisher rather than forking a second `V7-4` branch; the milestone history should remain single-threaded.

## Surprises & Discoveries

- The base `V7-4` runner repeated the same missing-`import os` pattern that `V7-3` exposed, which confirms this was a shared harness defect rather than a phase-specific anomaly.
