# PLANv6 Phase V6-3 - Anti-Homogenization Loss Screening

## Purpose

Run the `PLANv6` loss-axis screen on the two `V6-2` winners so the repo can determine whether anti-collapse objectives materially improve Writer source usefulness beyond the `L0` task-only baseline.

## Context

- `PLANv6.md` defines `V6-3` as the next authorized phase after `V6-2`.
- `V6-2` completed with `comparison_conclusion=select_top_two_support_modes`.
- The selected support modes are:
  - `S3 multi_item_cross_attn_raw`
  - `S5 hybrid_pooled_plus_items`
- The next unanswered question is whether richer Writer-side objectives can reduce source homogenization without creating misleading aux-only gradient movement.

## Canonical Matrix

Run the following loss-family screen on both selected support modes:

- `S3 + L0 task-only`
- `S3 + L1 legacy`
- `S3 + L2 contrastive`
- `S3 + L3 VICReg / VCReg`
- `S3 + L5 orthogonality + coverage`
- `S5 + L0 task-only`
- `S5 + L1 legacy`
- `S5 + L2 contrastive`
- `S5 + L3 VICReg / VCReg`
- `S5 + L5 orthogonality + coverage`

Bridge and recipe held fixed unless `PLANv6` requires otherwise:

- Writer-direct deep prefix with early layers `[0,1,2,3]`
- early4 receiver micro-LoRA on `k_proj/v_proj`
- repaired post-unfreeze Writer gates
- group-wise clipping and gradient probes enabled

## Plan Of Work

1. Extend the runtime with explicit `pilot_aux_loss_mode` handling for `L0/L1/L2/L3/L5`.
2. Add the missing anti-homogenization losses and per-step diagnostics on Writer-side representations.
3. Add the dedicated `V6-3` runner and ranking summary.
4. Execute the loss screen across FEVER, GSM8K, and NarrativeQA on `S3` and `S5`.
5. Publish the review bundle and select the top auxiliary families for `V6-4`.

## Concrete Steps

1. Extend [`src/memtotal/training/m4_shared_injection.py`](/root/mydir/MemTOTAL/src/memtotal/training/m4_shared_injection.py) with:
   - `pilot_aux_loss_mode`
   - auxiliary view generation controls
   - `L2/L3/L5` Writer-side loss computation
   - detailed aux diagnostics in train events and metrics
2. Extend [`src/memtotal/models/memory.py`](/root/mydir/MemTOTAL/src/memtotal/models/memory.py) only as needed to support Writer auxiliary projection or view targets.
3. Add the `V6-3` runner:
   - [`scripts/run_planv6_v6_3_loss_screen_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv6_v6_3_loss_screen_qwen25.sh)
4. Add the `V6-3` ranking summary:
   - [`scripts/update_planv6_v6_3_loss_screening_summary.py`](/root/mydir/MemTOTAL/scripts/update_planv6_v6_3_loss_screening_summary.py)
5. Publish to:
   - `runs/verify/planv6-v6-3-loss-screening-qwen25`
   - `results/generated/planv6-v6-3-loss-screening-qwen25`
   - `runs/review/planv6-v6-3-loss-screening-qwen25`
   - `results/generated/review/planv6-v6-3-loss-screening-qwen25`

## Validation And Acceptance

Phase `V6-3` is complete only if all of the following are true:

- explicit runtime flags select `L0/L1/L2/L3/L5` without changing the support-mode plumbing,
- train events and final metrics expose task-only / aux-only gradient norms and `cosine(task, aux)` for the Writer,
- the dedicated runner produces per-task artifacts for all screened support-mode and loss-family pairs,
- the summary ranks the loss families using the repaired `PLANv6` gates and flags aux-only movement failures,
- the review bundle is published,
- tests cover the new aux-loss dispatch, representative loss math, and ranking contract.

## Progress

- `2026-03-10 00:11 UTC`: opened the `V6-3` exec plan after `V6-2` selected `S3` and `S5`.
- `2026-03-10 00:25 UTC`: completed the `V6-3` harness patch for explicit `L0/L1/L2/L3/L5` aux-loss dispatch, repaired the `L5` support-coverage loss so it stays differentiable by reading raw Writer support-attention tensors, and updated the unit test to cover the new path.
- `2026-03-10 00:26 UTC`: validation passed for the phase harness: `bash -n scripts/run_planv6_v6_3_loss_screen_qwen25.sh scripts/publish_review_artifacts.sh`, `python -m py_compile src/memtotal/training/m4_shared_injection.py scripts/update_planv6_v6_3_loss_screening_summary.py`, `python -m unittest tests.test_m4_shared_injection tests.test_planv6_v6_3_loss_screening_summary -v`, and `python -m unittest discover -s tests -v` (`267` tests, `OK`).
- `2026-03-10 00:26 UTC`: launched the full detached matrix in `tmux` as session `planv6_v63`, writing live artifacts under `/root/autodl-tmp/runs/verify/planv6-v6-3-loss-screening-qwen25` and `/root/autodl-tmp/results/generated/planv6-v6-3-loss-screening-qwen25`.
- `2026-03-10 00:27 UTC`: verified the run is alive beyond config materialization. Control suites for `gsm8k`, `narrativeqa`, and `fever` all wrote `suite_metrics.json`, and the first screened arm `gsm8k / s3_multi_item_cross_attn_raw / l0_task_only` holds the active suite lock.
- `2026-03-10 00:29 UTC`: `gsm8k / s3_multi_item_cross_attn_raw / l0_task_only` finished and wrote `suite_metrics.json`; the next arm `gsm8k / s3_multi_item_cross_attn_raw / l1_legacy` started immediately. The first completed non-control run shows live Writer gradients (`train_grad_norm_writer_post_unfreeze_median=15.31`) and deeper-layer prefix mass (`layer 3 ~= 0.019`), but usefulness is still flat (`best_adapt_task_score=0.0`, `delta_answer_logprob=0.0`) and source collapse remains severe (`memory_long_common_mode_energy_ratio ~= 0.99999994`).
- `2026-03-10 00:35 UTC`: the full `33/33` suite matrix finished. The published summary at [`results/generated/review/planv6-v6-3-loss-screening-qwen25/v6-3-summary.json`](/root/mydir/MemTOTAL/results/generated/review/planv6-v6-3-loss-screening-qwen25/v6-3-summary.json) selected the top auxiliary families `L2 contrastive`, `L5 orthogonality + coverage`, and `L3 VICReg / VCReg`, with `comparison_conclusion=select_top_auxiliary_families` and `recommended_next_step=open_v6_4_mixed_matrix`.
- `2026-03-10 00:35 UTC`: the best overall combo was `S3 multi_item_cross_attn_raw + L2 contrastive`. FEVER and GSM8K both cleared the repaired `PLANv6` gates under the best combos, but NarrativeQA remained the limiting task; it stayed below `route_live_post_unfreeze` on every non-`L0` top-ranked branch.
- `2026-03-10 00:38 UTC`: refreshed the governed review bundle into [`runs/review/planv6-v6-3-loss-screening-qwen25`](/root/mydir/MemTOTAL/runs/review/planv6-v6-3-loss-screening-qwen25) and [`results/generated/review/planv6-v6-3-loss-screening-qwen25`](/root/mydir/MemTOTAL/results/generated/review/planv6-v6-3-loss-screening-qwen25). During publish, the repo overlay briefly ran out of space; the fix was to garbage-collect ignored generated checkpoints and tmux logs inside `runs/review/` before rerunning the standard publisher.

## Decision Log

- Preserve `L0 task-only` as a mandatory comparator instead of treating legacy auxiliary losses as an invisible default.
- Use the existing review-artifact publisher instead of inventing a `V6-3`-specific publish path so the new phase stays inside the repo’s established governance contract.
- For `V6-4`, carry forward both top support modes (`S3`, `S5`), the top three auxiliary families (`L2`, `L5`, `L3`), and both top stimulus mixes (`C0`, `C2`) into the mixed matrix rather than collapsing early onto a single best combo.

## Surprises And Discoveries

- The runtime already logs Writer task-vs-aux gradient probes, so `V6-3` mainly needs explicit loss-family control and per-family reporting rather than a new gradient-attribution subsystem.
- The first `L5` implementation mistake was subtle but important: computing coverage loss from `prefix_stats` rewrapped a detached scalar and made the term nondifferentiable. The correct implementation must consume `writer_diagnostics["support_attention_weights_by_layer"]` directly.
- `tmux` itself is healthy in this environment; the earlier failed launch was not a terminal issue. After the harness repair, the detached `V6-3` matrix started normally and began writing run directories immediately.
- The top of the `V6-3` ranking was dominated by `S3` rather than `S5`, but the phase still does not justify dropping `S5`: the `V6-4` question is interaction with stimulus mix and auxiliary family, not single-axis support ranking anymore.
