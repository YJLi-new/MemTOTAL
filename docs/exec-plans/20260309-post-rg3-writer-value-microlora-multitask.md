# 20260309 Post-RG3 Writer Value + Micro-LoRA + Multi-Task Bridge

## Purpose

Execute the next hop after the `PLANv2.md` stop-after-RG3 result, now under `PLANv3.md` narrow authority. The live blocker is no longer just local Reader geometry. The next narrow branch must test Writer value diversification, a tiny receiver-side micro-LoRA, and immediate non-FEVER validation.

## Context

- Macro authority remains `PLAN.md`
- Narrow predecessor was `PLANv2.md`
- Active narrow authority is `PLANv3.md`
- `RG-3` is complete and terminal for the local Reader/Fuser rescue line
- Current likely blocker: Writer common-mode domination + receiver gradient starvation
- User-level requirements now adopted:
  - add LoRA, but keep it much lighter than MemGen
  - validate on multiple tasks, not only FEVER

## Plan of Work

1. Add offline common-mode / value-diversity diagnostics.
2. Run FEVER Writer value-diversification matrix under the partition scaffold.
3. Run FEVER micro-LoRA matrix on the best Writer arm.
4. Re-test non-forced Reader behavior only after value diversity improves.
5. Run the best family on NarrativeQA real-smoke and GSM8K real-smoke.
6. Only after multi-task medium success, reopen Qwen3 confirmation and later transfer/CDMI work.

## Concrete Steps

1. Patch logging for raw/centered `M_long` rank, top1/top2 ratio, common-mode energy ratio, value-projected rank, and readout cosine.
2. Implement Writer modes:
   - `shared_add`
   - `shared_add_scaled`
   - `slot_query_only`
   - optional `slot_query_small_shared`
3. Keep `partition / H4 / K4 / linear` as the V1 control scaffold.
4. Add receiver micro-LoRA targeting only `k_proj` / `v_proj` on layers `14/21/27` first, `r=2`.
5. Compare frozen receiver vs micro-LoRA on the best Writer arm.
6. If geometry improves, re-run standard/competitive Reader as V3.
7. Run the best family on:
   - FEVER
   - NarrativeQA real-smoke
   - GSM8K real-smoke

## Validation & Acceptance

- New metrics appear in train/snapshot/final summaries.
- Writer value-diversity summary exists.
- Micro-LoRA summary exists.
- Multi-task summary exists.
- Trainable param counts for LoRA are logged.
- Unit tests pass.

## Progress

- 2026-03-09: Created this active exec-plan from `PLANv3.md`.
- 2026-03-09: Downloaded `PLANv3.md`, moved this exec-plan under `docs/exec-plans/`, and rewrote `AGENTS.md` in English with `PLANv3.md` as the narrow authority reference.
- 2026-03-09: Completed Phase V0 offline forensics patch in `src/memtotal/models/memory.py`, `src/memtotal/training/m4_shared_injection.py`, `src/memtotal/analysis/m4_shared_injection.py`, `scripts/update_m4_run_summary.py`, and `tests/test_m4_shared_injection.py`.
- 2026-03-09: Verified V0 with targeted test coverage (`python -m unittest tests.test_m4_shared_injection tests.test_smoke_components -v`) and an injected dry-run smoke at `/root/autodl-tmp/runs/verify/tl-writer-value-v0-smoke-injected`.
- 2026-03-09: Confirmed the new V0 metrics propagate through `train_events.json`, snapshot metrics, and final `metrics.json`; the smoke classification currently points to Writer-side common-mode domination rather than a Reader-only failure.
- 2026-03-09: Rebased onto upstream `origin/main`, committed Phase V0 as `0cd5375` (`feat: complete planv3 v0 writer value forensics`), and pushed it to `YJLi-new/MemTOTAL`.
- 2026-03-09: Implemented the V1 architecture-first surface: Writer conditioning modes (`shared_add`, `shared_add_scaled`, `slot_query_only`, `slot_query_small_shared`), Writer-side penalties (`writer_slot_energy_balance_loss`, `writer_common_mode_penalty`), FEVER V1 config matrix, the V1 summary comparator, and the new runner scripts.
- 2026-03-09: Verified the V1 code path with targeted validation (`python -m unittest tests.test_m4_shared_injection tests.test_smoke_components tests.test_repo_contract tests.test_repo_lints -v`) and real injected dry-run smokes for control / `shared_add_scaled` / `slot_query_only`.
- 2026-03-09: Ran the new V1 summary script on the three dry-run FEVER arms and observed a flat early picture (`comparison_conclusion=failure`, `failure_reason=common_mode_domination_persists`) at 2 training steps, which is informative but not a substitute for the real Step-2 matrix.
- 2026-03-09: Completed the real Step-2 FEVER Writer matrix at `/root/autodl-tmp/runs/verify/tl-writer-value-fever-qwen25` with review outputs under `/root/autodl-tmp/results/generated/tl-writer-value-fever-qwen25`.
- 2026-03-09: The real V1 matrix was flat across all three Writer arms: `W0/W1/W2` all stayed at `memory_long_top1_top2_ratio≈70.7`, `memory_long_common_mode_energy_ratio≈0.9986`, `reader_readout_effective_rank≈1.23`, and `reader_readout_pairwise_cosine_mean≈0.9993`, with no collapse delay and identical FEVER task scores (`best_adapt_task_score≈0.2951`).
- 2026-03-09: Recorded the real Step-2 decision as a V1 architecture hard stop: `comparison_conclusion=failure`, `failure_reason=common_mode_domination_persists`, `recommended_arm=control`, and under the `PLANv3` hard-stop rule the least-bad Writer arm now carries forward to V2 micro-LoRA without running V1 penalties first.

## Decision Log

- Do not continue RG-4.
- Do not open broad receiver fallback.
- Introduce only tiny diagnostic LoRA inside Workstream B.
- Use repo-supported non-FEVER tasks before onboarding new datasets.
- Treat Phase V0 as complete only after the full repo test suite passes and the milestone is committed/pushed.
- The first active post-V0 training step remains the architecture-first FEVER Writer matrix (`W0/W1/W2`) before any Writer penalties or receiver LoRA.
- Do not read the 2-step V1 smoke as a Step-2 decision; it is only a pipeline validation and an early sanity check on whether the new Writer modes alter immediate geometry.
- Treat the fully flat V1 architecture matrix as a `PLANv3` hard stop for Writer-only architectural tweaks: skip V1 penalty refinement and carry the least-bad Writer arm into V2 micro-LoRA.

## Surprises & Discoveries

- Hard partition proved that attention specialization can be forced without producing semantically diverse readouts.
- The strongest current bottleneck is likely common-mode domination in `M_long`, not merely attention symmetry.
- Receiver gradients became too small for local Reader rescue to be a meaningful next hop.
- The first injected V0 smoke accidentally stayed on `shared_injection_arm=base_only`; once rerun with `shared_injection_arm=injected`, the new metrics became non-zero and internally consistent across train/snapshot/final outputs.
- The injected V0 smoke shows extreme shared Writer energy (`memory_long_common_mode_energy_ratio≈0.999`, `memory_long_top1_top2_ratio≈70.7`) even while centered rank stays healthy, which strengthens the common-mode bottleneck hypothesis and justifies prioritizing V1 Writer diversification before V2 LoRA.
- The first real V1 dry-run smokes execute correctly for `shared_add_scaled` and `slot_query_only`, but at 2 steps both remain almost identical to control (`top1/top2≈70.7`, `readout cosine≈0.999`), so the new modes are wired correctly without yet showing immediate rescue under the inherited warm-start.
- The full FEVER Step-2 matrix matched the dry-run direction almost exactly: both `shared_add_scaled(alpha=0.02)` and `slot_query_only` were effectively indistinguishable from control, which strengthens the view that the next live uncertainty is receiver readability rather than another Writer-only tweak.
