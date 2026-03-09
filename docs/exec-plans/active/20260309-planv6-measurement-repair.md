# PLANv6 Phase V6-0 - Measurement Repair And Instrumentation

## Purpose

Implement `PLANv6.md` Phase `V6-0` so the repo can distinguish route failure from measurement artifact before any more meaningful GPU budget is spent.

## Context

- `PLANv6.md` is now the root execution authority.
- The current `writer-deep-prefix-jointpeft` summary misclassifies the route because it evaluates Writer liveness over steps `1-50` even though the Writer is intentionally frozen there.
- The training loop already logs useful gradients and support-state statistics, but it does not yet expose:
  - post-unfreeze route gates,
  - explicit task-vs-aux gradient attribution,
  - support-item attention diagnostics inside `WriterWeaverHead`,
  - group-aware clipping decisions,
  - richer usefulness metrics for classification and generation tasks.

## Plan Of Work

1. Update repo guidance so all future work starts from `PLANv6.md`.
2. Patch the Writer-direct summary logic to use post-unfreeze windows and richer usefulness diagnostics.
3. Extend the writer and training loop with support-attention diagnostics, gradient probes, and group-aware clipping/logging.
4. Add or update tests for the new gate semantics and instrumentation.
5. Run a short validation pass proving the new log fields exist.

## Concrete Steps

1. Update [`AGENTS.md`](/root/mydir/MemTOTAL/AGENTS.md) to point at `PLANv6.md`.
2. Patch [`scripts/update_writer_deep_prefix_jointpeft_summary.py`](/root/mydir/MemTOTAL/scripts/update_writer_deep_prefix_jointpeft_summary.py):
   - add post-unfreeze windows,
   - preserve compatibility fields while adding `route_live_post_unfreeze`,
   - add classification/generative usefulness diagnostics.
3. Patch [`src/memtotal/models/memory.py`](/root/mydir/MemTOTAL/src/memtotal/models/memory.py):
   - expose optional Writer support-attention diagnostics without changing default call behavior.
4. Patch [`src/memtotal/training/m4_shared_injection.py`](/root/mydir/MemTOTAL/src/memtotal/training/m4_shared_injection.py):
   - add gradient probe config parsing and explicit autograd probes,
   - add support-interface diagnostics,
   - add group-aware clipping controls and logging,
   - aggregate the new windows/metrics into `metrics.json`.
5. Add tests in [`tests/test_m4_shared_injection.py`](/root/mydir/MemTOTAL/tests/test_m4_shared_injection.py) and [`tests/test_writer_deep_prefix_jointpeft_summary.py`](/root/mydir/MemTOTAL/tests/test_writer_deep_prefix_jointpeft_summary.py).
6. Run targeted validation plus a short probe/dry-run summary if feasible inside the turn budget.

## Validation And Acceptance

Phase `V6-0` is complete only if all of the following are true:

- the summary no longer treats the frozen `1-50` window as the Writer liveness verdict,
- task-only vs aux-only Writer gradients can be logged on probe steps,
- support-item diagnostics appear for non-pooled support modes,
- tests covering the new logic pass,
- a short validation run or dry-run artifact confirms the new log fields are emitted.

## Progress

- `2026-03-09 21:25 UTC`: `PLANv6.md` downloaded to repo root.
- `2026-03-09 21:26 UTC`: confirmed `PLANv6` Phase `V6-0` is the mandatory first milestone.

## Decision Log

- Keep the first implementation pass focused on `V6-0`; do not jump into support-mode or loss sweeps before the measurement layer is repaired.
- Preserve the current joint-PEFT summary entrypoint, but change its semantics so existing review bundles can be reinterpreted under `PLANv6`.

## Surprises And Discoveries

- The existing repo already contains `structured_support_set` support plumbing; the immediate gap is diagnosis and fair gating, not the total absence of multi-item support infrastructure.
- The current summary script is the direct source of the headline misclassification because it keys `route_live` off the frozen window.
