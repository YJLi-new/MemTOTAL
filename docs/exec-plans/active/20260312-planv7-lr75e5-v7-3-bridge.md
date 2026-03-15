# PLANv7 (LR updated version) V7-3 Wide External Writer With Bounded Query Compression

## Purpose

Queue restarted `V7-3` so it can begin as soon as LR-updated `V7-2` publishes its governed summary, using the same bounded query-compression bridge matrix as historical `V7-3` under projector LR `7.5e-5`.

## Context

- Restarted `V7-2` is the direct-bandwidth decision point for the LR-updated line.
- `V7-3` consumes the governed `V7-2` summary to select its matched direct control arm.
- The restart wrapper already exists at:
  - [`scripts/run_planv7_lr75e5_v7_3_bridge_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_lr75e5_v7_3_bridge_qwen25.sh)
- This phase preserves the original `PLANv7` bridge contract:
  - `B_ctrl`
  - `B_W3_q8`
  - `B_W3_q16`
  - `B_W3_q16_s8`
  - `B_W4_q16`

## Plan Of Work

1. Wait for restarted `V7-2` to emit `v7-2-summary.json`.
2. Launch restarted `V7-3` in its own namespace using that summary path explicitly.
3. Wait for `V7-3` summary publication.
4. Publish the governed review bundle, commit the milestone, push `main`, and refresh `review`.

## Concrete Steps

1. Watch:
   - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-2-direct-bandwidth-qwen25/v7-2-summary.json`
2. Launch:
   - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-3-bridge-qwen25`
   - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-3-bridge-qwen25`
3. Pass the direct result-root summary path to the wrapper so launch does not have to wait for review publication.
4. Arm a post-completion publisher for the `V7-3` restart bundle.

## Validation & Acceptance

Acceptance for this queued handoff:

- the launcher waits on the correct `V7-2` summary path,
- `V7-3` starts only after that summary exists,
- the run/result roots use the `planv7-lr75e5-v7-3-*` namespace,
- the post-completion publisher is armed for the `V7-3` milestone.

## Progress

- 2026-03-12 UTC: Opened the restart-specific `V7-3` bridge handoff while LR-updated `V7-2` was running.
- 2026-03-12 UTC: Validation passed:
  - `bash -n scripts/run_planv7_lr75e5_v7_3_bridge_qwen25.sh scripts/publish_review_artifacts.sh`
  - `python -m unittest tests.test_repo_lints tests.test_repo_contract -v`
- 2026-03-12 UTC: Launched restarted `V7-3` in detached `tmux` as `planv7_lr75e5_v73` with:
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-3-bridge-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-3-bridge-qwen25`
  - direct input summary: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-2-direct-bandwidth-qwen25/v7-2-summary.json`
- 2026-03-12 UTC: Armed detached post-completion publisher `planv7_lr75e5_v73_post`, which waits for `v7-3-summary.json`, then refreshes review artifacts, commits the milestone, pushes `main`, and refreshes the lightweight `review` branch.
- 2026-03-12 UTC: The first launch failed before the first suite with `NameError: name 'os' is not defined` inside the base bridge runner's second embedded Python block. The bug was traced to a missing import in [`scripts/run_planv7_v7_3_bridge_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_v7_3_bridge_qwen25.sh).
- 2026-03-12 UTC: Patched the missing import, then revalidated the bridge harness:
  - `bash -n scripts/run_planv7_v7_3_bridge_qwen25.sh scripts/run_planv7_lr75e5_v7_3_bridge_qwen25.sh scripts/publish_review_artifacts.sh`
  - `python -m unittest tests.test_repo_lints tests.test_repo_contract -v`
- 2026-03-12 UTC: Re-launched restarted `V7-3` with the same run/result roots and re-armed `planv7_lr75e5_v73_post`.
- 2026-03-12 UTC: Current live state after relaunch:
  - materialized configs exist for all `gsm8k` and `triviaqa` bridge/control arms
  - the active process is `python scripts/run_m4_selected_shared_injection_suite.py --config .../gsm8k-control.json`
  - `tmux-session.log` currently shows the repaired startup path reaching local-model preparation and first-suite execution
  - both detached sessions are alive.

## Decision Log

- Use the direct result-root summary path from restarted `V7-2` rather than the review mirror to minimize idle time between phases.
- Keep `V7-3` queued but not started until `V7-2` actually produces its governed summary.
- After the failed first launch, preserve the same namespace and rerun in place so the milestone history stays single-threaded and the post-publish automation can remain attached to the real `V7-3` root.

## Surprises & Discoveries

- The base `V7-3` runner had not exercised its second embedded Python block since the LR-override refactor, so the restart surfaced a missing `import os` before any suite output existed.
