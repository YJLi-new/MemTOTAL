# PLANv7 V7-5 Targeted Auxiliary Revisit

## Purpose

Run the `PLANv7.md` `V7-5` matrix from the governed `V7-4` outcome to test whether a narrow auxiliary-loss revisit can finally move `GSM8K` or `TriviaQA`, or at least materially improve strict Writer-memory metrics without regressing the other primary task.

## Context

- `V7-4` published `comparison_conclusion=forced_consumption_diagnostic_only_move_to_v7_5`.
- The governed `V7-4` base remained the `V7-3` bridge winner:
  - `base_for_v7_5_arm_id=b_w3_q16`
  - `base_for_v7_5_source_phase=v7_3`
  - depth `D1 / mid4`
- `PLANv7` fixes the `V7-5` matrix to:
  - `A0` = `L5` baseline
  - `A1` = `L5 + reconstruction-lite`
  - `A2` = `L5 + VICReg-lite`
  - `A3` = `L5 + contrastive-lite`
  - `A4` = `L5 + reconstruction-lite + VICReg-lite`
  - `A5` = optional `L5 + Barlow-lite` if the runtime supports it cleanly
- The current runtime already supports reconstruction, VICReg, contrastive, and Barlow individually, but the pre-`V7-5` gate still treated `L5` as mutually exclusive with the other auxiliary families. That must be fixed before the matrix is valid.

## Plan Of Work

1. Patch the auxiliary gate so explicit `V7-5` configs can combine `L5` with `VICReg`, contrastive, or Barlow, while preserving prior behavior for older configs that only rely on `pilot_aux_loss_mode`.
2. Add one focused runtime test locking the new combo contract.
3. Add the governed `V7-5` runner, summary script, summary test, and publish wiring.
4. Validate with static checks, focused tests, and the full unit suite.
5. Launch the full `V7-5` matrix in detached `tmux`.
6. Publish the governed summary, commit, push `main`, and refresh the lightweight `review` branch.
7. If the governed summary authorizes it, continue directly into `V7-6`.

## Concrete Steps

1. Read `results/generated/review/planv7-v7-4-forced-consumption-qwen25/v7-4-summary.json` and use its recorded `base_for_v7_5_*` fields.
2. Keep the phase on the governed `V7-3/V7-4` winning depth.
3. Materialize configs for:
   - `a0_baseline`
   - `a1_reconstruction`
   - `a2_vicreg`
   - `a3_contrastive`
   - `a4_reconstruction_vicreg`
   - `a5_barlow`
4. Run all `V7-5` suites on `gsm8k` and `triviaqa`.
5. Publish:
   - `v7-5-summary.json`
   - `v7-5-summary.md`
6. Sync review artifacts and push once the milestone is complete.

## Validation And Acceptance

`V7-5` is complete only if:

- `A0-A4` finish on both primary tasks,
- `A5` is either cleanly included or omitted with an explicit reason,
- the summary ranks the auxiliary branches against the `A0` baseline,
- the summary distinguishes real primary-task gains from strict-metric-only non-regressive gains,
- and the phase is committed, pushed, and republished to the lightweight `review` branch.

Validation commands:

```bash
bash -n \
  scripts/run_planv7_v7_5_targeted_aux_revisit_qwen25.sh \
  scripts/publish_review_artifacts.sh
python -m py_compile \
  src/memtotal/training/m4_shared_injection.py \
  scripts/update_planv7_v7_5_targeted_aux_summary.py \
  tests/test_m4_shared_injection.py \
  tests/test_planv7_v7_5_targeted_aux_summary.py
python -m unittest \
  tests.test_m4_shared_injection \
  tests.test_planv7_v7_5_targeted_aux_summary \
  -v
python -m unittest discover -s tests -v
```

Live run command:

```bash
tmux new-session -d -s planv7_v75 \
  "cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_v7_5_targeted_aux_revisit_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-v7-5-targeted-aux-revisit-qwen25 \
     /root/autodl-tmp/results/generated/planv7-v7-5-targeted-aux-revisit-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     300 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
     results/generated/review/planv7-v7-4-forced-consumption-qwen25/v7-4-summary.json \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-v7-5-targeted-aux-revisit-qwen25/tmux-session.log"
```

## Progress

- 2026-03-11 23:16 UTC: Re-read `PLANv7.md` `V7-5` and confirmed that the phase is intentionally narrow: auxiliary families only, on top of the best `V7-3/V7-4` base.
- 2026-03-11 23:18 UTC: Identified the missing runtime capability: `pilot_aux_loss_mode` still made `L5` mutually exclusive with `VICReg`, contrastive, and Barlow, which would have made `A2-A5` invalid.
- 2026-03-11 23:22 UTC: Patched `src/memtotal/training/m4_shared_injection.py` so explicitly configured extra auxiliary weights can coexist with `L5`, while older configs that only set `pilot_aux_loss_mode` still keep their old behavior.
- 2026-03-11 23:24 UTC: Added a focused runtime regression test for `L5 + VICReg`.
- 2026-03-11 23:25 UTC: Added the governed `V7-5` runner, summary script, summary test, and publish wiring.
- 2026-03-11 23:38 UTC: `V7-5` validation is green:
  - `bash -n scripts/run_planv7_v7_5_targeted_aux_revisit_qwen25.sh scripts/publish_review_artifacts.sh`
  - `python -m py_compile src/memtotal/training/m4_shared_injection.py scripts/update_planv7_v7_5_targeted_aux_summary.py tests/test_m4_shared_injection.py tests/test_planv7_v7_5_targeted_aux_summary.py`
  - `python -m unittest tests.test_m4_shared_injection.SharedInjectionHelpersTest.test_run_shared_injection_pilot_allows_l5_plus_vicreg_combo tests.test_planv7_v7_5_targeted_aux_summary -v`
  - `python -m unittest discover -s tests -v`
  - full-suite result: `293 tests, OK`
- 2026-03-11 23:39 UTC: The first `tmux` launch failed before the script began because `tee` opened the session log before the run root existed. No suite work was lost because the phase had not started yet.
- 2026-03-11 23:39 UTC: Relaunched the full matrix in detached `tmux` as `planv7_v75` with the run/result roots pre-created.
- 2026-03-11 23:39 UTC: Live roots:
  - run root: `/root/autodl-tmp/runs/verify/planv7-v7-5-targeted-aux-revisit-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-v7-5-targeted-aux-revisit-qwen25`
  - session log: `/root/autodl-tmp/runs/verify/planv7-v7-5-targeted-aux-revisit-qwen25/tmux-session.log`
- 2026-03-11 23:39 UTC: The relaunched job is alive and has begun materializing the governed datasets/configs.
- 2026-03-11 UTC: Mid-run check: completed suites are `a0_baseline/gsm8k`, `a0_baseline/triviaqa`, `a1_reconstruction/gsm8k`, `a1_reconstruction/triviaqa`, and `a2_vicreg/gsm8k`; the active suite is `a2_vicreg/triviaqa`.
- 2026-03-11 UTC: While `V7-5` continues, the next governed milestone surface for `V7-6` has already been scaffolded and validated locally so the three-seed confirmation can start immediately after the `V7-5` push.
- 2026-03-11 16:40 UTC: The full `A0-A5` matrix finished and published `results/generated/review/planv7-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.{json,md}` plus the mirrored `runs/review/planv7-v7-5-targeted-aux-revisit-qwen25` bundle.
- 2026-03-11 16:40 UTC: Governed result:
  - `comparison_conclusion=aux_revisit_flat_best_branch_for_decision_point`
  - `recommended_next_step=prepare_v7_6_decision_point`
  - `base_for_v7_6_arm_id=a5_barlow`
  - `optional_barlow_supported=true`
- 2026-03-11 16:40 UTC: No auxiliary family improved actual `GSM8K` or `TriviaQA` task score versus `A0`, and no family achieved a qualified strict Writer-memory gain. `A5` was still carried forward as the decision-point branch because it had the least-bad strict-rank movement without regression and the summary explicitly selected it for `V7-6`.

## Decision Log

- `A5` is included because Barlow is already implemented cleanly in the runtime, so the optional branch is cheap and governed rather than speculative.
- The `V7-5` baseline is `A0`, not a separate no-memory control. The plan’s question here is whether narrow auxiliary changes can improve the current best injected base, not whether injection still beats a frozen control.
- The summary allows a branch to qualify via strict Writer-memory improvement only when both primary tasks are non-regressive, matching the plan’s “without hurting the other primary benchmark” rule.

## Surprises And Discoveries

- The missing combo gate was small but important: without it, `A2-A5` would have reported clean runs while silently omitting the intended second auxiliary family.
- The governed `V7-4` base staying on `b_w3_q16` keeps `V7-5` simpler than it could have been; there is no need to reopen the task-limited `F1/F3` branches as the live baseline.
- Even with the repaired mixed-aux runtime, all `V7-5` branches stayed in the same regime: route-live and stable on both primary tasks, but still strictly collapsed at the Writer-memory level and flat on actual primary-task score.
