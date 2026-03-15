# PLANv8 V8-3 Repair: Qwen3-4B OPD Diagnostics Preservation

## Purpose

Repair the qwen34 `PLANv8` `V8-3` Reader-OPD line after the first governed pass crashed before writing:

- `/root/autodl-tmp/results/generated/planv8-v8-3-reader-opd-qwen34/v8-3-summary.json`

This is a `V8-3` repair only. It does not change the governed `V8-4+` plan shape. It restores the stalled qwen34 chain so the existing `V8-4` through `V8-7` queue sessions can continue once `V8-3` completes.

## Failure Surface

The first qwen34 `V8-3` pass advanced through the warm-start control and then crashed on the legacy dense-teacher KL arm:

- active arm at crash: `p1_teacher_choice_kl-triviaqa`
- latest recorded step: `300`
- partial metrics materialized for `5` tasks, but no `v8-3-summary.json`

The runner failed in `run_shared_injection_pilot()` while serializing train-event diagnostics:

```text
KeyError: 'opd_target_score_active'
```

## Diagnosis

`run_shared_injection_pilot()` seeds `alignment_aux_diagnostics` with both:

- legacy dense-teacher metrics
- OPD-specific fields that the train-event payload always records

But in the non-OPD candidate-selection branch, the code replaced that seeded diagnostics dict with the narrower return value from `_alignment_aux_loss()`. That narrower dict contains only dense-teacher keys, so the later train-event serializer crashed when it unconditionally emitted:

- `opd_target_score_active`
- `opd_target_score_base`
- `opd_target_score_teacher`
- and the related OPD token diagnostics

The failure is therefore a diagnostics-merge bug, not an OPD math failure.

## Repair Work

1. Add `_default_alignment_aux_diagnostics()` so the full governed diagnostics surface is defined in one place.
2. Preserve the seeded diagnostics dict in the `teacher_choice_*` and `teacher_margin` path by merging `_alignment_aux_loss()` outputs into it instead of replacing it.
3. Add a focused mocked-pilot regression that runs `pilot_alignment_aux_mode=teacher_choice_kl` through `run_shared_injection_pilot()` and asserts the OPD train-event fields remain present and zeroed.
4. Re-launch qwen34 `V8-3` from the canonical run/result roots after clearing the broken partial outputs.

## Validation

```bash
python -m py_compile \
  src/memtotal/training/m4_shared_injection.py \
  tests/test_m4_shared_injection.py
python -m unittest \
  tests.test_m4_shared_injection.SharedInjectionHelpersTest.test_run_shared_injection_pilot_teacher_choice_aux_preserves_zeroed_opd_metrics \
  tests.test_m4_shared_injection.SharedInjectionHelpersTest.test_alignment_aux_loss_teacher_choice_kl_uses_dense_weight \
  tests.test_opd_alignment \
  tests.test_planv8_v8_3_summary \
  tests.test_repo_lints \
  tests.test_repo_contract -v
```

## Progress

- 2026-03-15 UTC: Confirmed qwen34 `V8-2` finished cleanly and the qwen34 chain was actually stalled in `V8-3`, not `V8-2`.
- 2026-03-15 UTC: Reproduced the live crash directly from `/root/autodl-tmp/runs/verify/planv8-v8-3-reader-opd-qwen34/tmux-session.log`.
- 2026-03-15 UTC: Verified the crash comes from replacing seeded alignment diagnostics in the dense-teacher branch.
- 2026-03-15 UTC: Landed the merge-preserving repair and a one-step regression that fails on the pre-fix code path.
