# PLANv6 Phase V6-2 - Support-Interface Screening

## Purpose

Run the first `PLANv6` support-axis screen so the repo can compare `S0/S1/S2/S3/S4/S5` under the same task-only writer-direct bridge and select the top two support modes for `V6-3`.

## Context

- `PLANv6.md` defines `V6-2` as the next authorized phase after `V6-1`.
- `V6-1` already established that the repaired post-unfreeze gates can show real route/task usefulness on the current pooled FEVER/GSM8K path.
- The remaining open question is whether source collapse is primarily driven by the support interface.

## Canonical Matrix

- `S0 + C1 + L0`
- `S1 + C2 + L0`
- `S2 + C1 + L0`
- `S3 + C0 + L0`
- `S4 + C1 + L0`
- `S5 + C2 + L0`

Bridge and recipe held fixed:

- Writer-direct deep prefix with early layers `[0,1,2,3]`
- early4 receiver micro-LoRA on `k_proj/v_proj`
- task-only writer loss family
- group-wise clipping
- lower projector LR
- gradient probes enabled

## Plan Of Work

1. Extend the runtime to expose the full `V6-2` support-mode set.
2. Add the missing support/context balance gate required by `S1/C2`.
3. Add a dedicated screening runner and ranking summary.
4. Execute the short-horizon screen across FEVER, GSM8K, and NarrativeQA.
5. Publish the review bundle and select the top two support modes for `V6-3`.

## Concrete Steps

1. Extend [`src/memtotal/models/memory.py`](/root/mydir/MemTOTAL/src/memtotal/models/memory.py) with a default-off support/context balance gate inside `WriterWeaverHead`.
2. Extend [`src/memtotal/training/m4_shared_injection.py`](/root/mydir/MemTOTAL/src/memtotal/training/m4_shared_injection.py) with explicit support modes:
   - `pooled_block`
   - `structured_support_set`
   - `multi_item_cross_attn_raw`
   - `multi_item_cross_attn_encoded`
   - `hybrid_pooled_plus_items`
3. Add the `V6-2` runner:
   - [`scripts/run_planv6_v6_2_support_screen_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv6_v6_2_support_screen_qwen25.sh)
4. Add the `V6-2` ranking summary:
   - [`scripts/update_planv6_v6_2_support_screening_summary.py`](/root/mydir/MemTOTAL/scripts/update_planv6_v6_2_support_screening_summary.py)
5. Publish to:
   - `runs/verify/planv6-v6-2-support-screening-qwen25`
   - `results/generated/planv6-v6-2-support-screening-qwen25`
   - `runs/review/planv6-v6-2-support-screening-qwen25`
   - `results/generated/review/planv6-v6-2-support-screening-qwen25`

## Validation And Acceptance

Phase `V6-2` is complete only if all of the following are true:

- the new support-mode options are available behind explicit runtime flags,
- the `S1` gated pooled path logs support/context balance metadata,
- the dedicated runner produces per-task artifacts for all screened support modes,
- the summary selects the top two support modes using the repaired `PLANv6` gates,
- the review bundle is published,
- tests cover the new writer balance gate, support-mode dispatch, and support-screen ranking.

## Progress

- `2026-03-09 23:01 UTC`: opened the `V6-2` exec plan after confirming `V6-1` ended in `move_to_writer_usefulness_branch`.
- `2026-03-09 23:08 UTC`: added explicit `V6-2` support-mode dispatch plus the default-off support/context balance gate.
- `2026-03-09 23:15 UTC`: added the dedicated `V6-2` runner and ranking summary.
- `2026-03-09 23:18 UTC`: added targeted tests for the balance gate, support-mode dispatch, and top-two ranking contract.
- `2026-03-09 23:22 UTC`: full local validation passed (`261` tests). First workspace-local detached run failed during checkpoint writes because the repo overlay was nearly full.
- `2026-03-09 23:27 UTC`: moved the partial failed workspace run tree off the overlay and relaunched the full `V6-2` screen in detached `tmux` using `/root/autodl-tmp` for the heavy run/result roots.
- `2026-03-10 00:03 UTC`: the detached rerun finished all `21/21` suites, published [`v6-2-summary.json`](/root/mydir/MemTOTAL/results/generated/review/planv6-v6-2-support-screening-qwen25/v6-2-summary.json), and selected `S3 multi_item_cross_attn_raw` plus `S5 hybrid_pooled_plus_items` as the top two support modes for `V6-3`.
- `2026-03-10 00:09 UTC`: patched `scripts/publish_review_artifacts.sh` so review publication skips empty local result directories and falls through to populated `/root/autodl-tmp` sources; rerunning the publisher filled the previously empty in-repo `results/generated/review/planv6-v6-2-support-screening-qwen25` bundle.
- `2026-03-10 00:07 UTC`: the detached retry finished `21/21` suites and published the summary bundle. `comparison_conclusion=select_top_two_support_modes`, `top_two_support_modes=[s3_multi_item_cross_attn_raw, s5_hybrid_pooled_plus_items]`, and `recommended_next_step=open_v6_3_anti_homogenization_screen`.
- `2026-03-10 00:09 UTC`: refreshed the repo review bundle so the published `V6-2` artifacts now live under `runs/review/planv6-v6-2-support-screening-qwen25` and `results/generated/review/planv6-v6-2-support-screening-qwen25`.

## Decision Log

- Keep the historical `pooled_block` path unchanged and expose all new behavior behind new support-mode and balance-mode flags.
- Treat `S2` and `S4` as separate explicit runtime labels even though the current writer-direct encoded-item path is shared; preserving both labels keeps the screening bundle aligned with `PLANv6`.
- Keep the first `V6-2` screen task-only (`L0`) and short-horizon so support geometry is isolated before `V6-3` anti-homogenization experiments.
- Heavy checkpointed screening runs should not default to repo-local run roots when the overlay filesystem is close to full; use `/root/autodl-tmp` for execution and then publish back into the repo review bundle.
- When publishing review bundles, do not prefer an empty repo-local result directory over a populated off-overlay run root; the publisher now skips empty sources before falling back to later candidates.
- Rank the `V6-2` support modes using the repaired `PLANv6` gates, not raw FEVER delta alone. This kept `S3` and `S5` ahead of legacy pooled variants because they preserved non-FEVER route liveness while still clearing the FEVER gates.

## Surprises And Discoveries

- The current code already had everything needed for raw and hybrid multi-item support except explicit runtime dispatch; only `S1` required new writer behavior.
- The existing `structured_support_set` writer-direct path is already functionally close to the `S4` encoded-item branch described in `PLANv6`, so the main implementation gap was control/measurement clarity rather than missing deep-prefix plumbing.
- The repo overlay can fail during `torch.save()` long before the large `/root/autodl-tmp` volume is pressured, so launch location is now part of the practical harness contract for V6-class screening runs.
- The initial publisher ordering hid the completed `V6-2` result bundle because an empty local `results/generated/planv6-v6-2-support-screening-qwen25` directory short-circuited the fallback chain; future off-overlay phases would have repeated the same failure without the `dir_has_files` guard.
- `S3 multi_item_cross_attn_raw` and `S5 hybrid_pooled_plus_items` both kept FEVER route/task usefulness alive while avoiding the complete pooled collapse seen in the weakest legacy variants, but NarrativeQA still never became route-live post-unfreeze. `V6-3` therefore needs to test whether loss design can make the richer support interface actually matter off FEVER rather than simply ranking support encoders by FEVER fit.
