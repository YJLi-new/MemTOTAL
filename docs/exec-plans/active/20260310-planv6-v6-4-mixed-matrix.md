# PLANv6 Phase V6-4 - Mixed Matrix

## Purpose

Run the `PLANv6` mixed matrix so the repo can test interaction effects across support mode, stimulus mix, and anti-homogenization family, then select `2-3` finalists for `V6-5`.

## Context

- `PLANv6.md` defines `V6-4` as the next authorized phase after `V6-3`.
- `V6-2` selected the top two support modes:
  - `S3 multi_item_cross_attn_raw`
  - `S5 hybrid_pooled_plus_items`
- `V6-3` selected the top three auxiliary families:
  - `L2 contrastive`
  - `L5 orthogonality + coverage`
  - `L3 VICReg / VCReg`
- `PLANv6` explicitly says the mixed matrix should carry the top two stimulus mixes from `{C0, C2}`:
  - `C0 support_only`
  - `C2 support_and_context_gated`
- `V6-3` showed that FEVER and GSM8K can already clear the repaired gates on the best branches, while NarrativeQA remains the main unresolved weakness. `V6-4` should therefore preserve FEVER/GSM8K wins while actively checking whether `C2` helps NarrativeQA without collapsing the source route.

## Canonical Matrix

Run the following `2 x 3 x 2 = 12` combos across FEVER, GSM8K, and NarrativeQA:

- `S3 + C0 + L2`
- `S3 + C0 + L5`
- `S3 + C0 + L3`
- `S3 + C2 + L2`
- `S3 + C2 + L5`
- `S3 + C2 + L3`
- `S5 + C0 + L2`
- `S5 + C0 + L5`
- `S5 + C0 + L3`
- `S5 + C2 + L2`
- `S5 + C2 + L5`
- `S5 + C2 + L3`

Bridge and recipe held fixed unless `PLANv6` requires otherwise:

- Writer-direct deep prefix with early layers `[0,1,2,3]`
- early4 receiver micro-LoRA on `k_proj/v_proj`
- repaired post-unfreeze Writer gates
- group-wise clipping and gradient probes enabled
- same medium deterministic FEVER/GSM8K/NarrativeQA slices used by `V6-3`

## Plan Of Work

1. Add the dedicated `V6-4` mixed-matrix runner with factorized support, stimulus, and loss selections.
2. Add the `V6-4` ranking summary that selects `2-3` finalists using the repaired `PLANv6` gates.
3. Publish the governed review bundle for the matrix.
4. Select `2-3` finalists for `V6-5` recipe stabilization and layer comparison.

## Concrete Steps

1. Add the `V6-4` runner:
   - [`scripts/run_planv6_v6_4_mixed_matrix_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv6_v6_4_mixed_matrix_qwen25.sh)
2. Add the `V6-4` ranking summary:
   - [`scripts/update_planv6_v6_4_mixed_matrix_summary.py`](/root/mydir/MemTOTAL/scripts/update_planv6_v6_4_mixed_matrix_summary.py)
3. Add a summary-contract test:
   - [`tests/test_planv6_v6_4_mixed_matrix_summary.py`](/root/mydir/MemTOTAL/tests/test_planv6_v6_4_mixed_matrix_summary.py)
4. Extend the publisher so it syncs:
   - `runs/verify/planv6-v6-4-mixed-matrix-qwen25`
   - `results/generated/planv6-v6-4-mixed-matrix-qwen25`
   - `runs/review/planv6-v6-4-mixed-matrix-qwen25`
   - `results/generated/review/planv6-v6-4-mixed-matrix-qwen25`

## Validation And Acceptance

Phase `V6-4` is complete only if all of the following are true:

- the runner materializes and executes the full `12` combo matrix plus task controls,
- the summary ranks combos and selects `2-3` finalists using the repaired `PLANv6` gates,
- the report exposes `route_live_post_unfreeze`, `writer_task_supervision_live`, `source_not_collapsed`, `stable_training_v6`, and `usefulness_positive_v6` for every task and combo,
- the governed review bundle is published,
- tests cover the new ranking contract and the runner passes static validation.

## Progress

- `2026-03-10 00:44 UTC`: opened the `V6-4` exec plan after `V6-3` completed with `comparison_conclusion=select_top_auxiliary_families` and `recommended_next_step=open_v6_4_mixed_matrix`.
- `2026-03-10 00:54 UTC`: finished the `V6-4` harness implementation: added the mixed-matrix runner, ranking summary, ranking test, and the publisher sync targets for the new review bundle.
- `2026-03-10 00:56 UTC`: validation passed for the new branch: `bash -n scripts/run_planv6_v6_4_mixed_matrix_qwen25.sh scripts/publish_review_artifacts.sh`, `python -m py_compile scripts/update_planv6_v6_4_mixed_matrix_summary.py`, and `python -m unittest discover -s tests -v` (`268` tests, `OK`).
- `2026-03-10 10:42 UTC`: launched the full `V6-4` matrix in detached `tmux` session `planv6_v64` with run root `/root/autodl-tmp/runs/verify/planv6-v6-4-mixed-matrix-qwen25`, result root `/root/autodl-tmp/results/generated/planv6-v6-4-mixed-matrix-qwen25`, and session log `/root/autodl-tmp/runs/verify/planv6-v6-4-mixed-matrix-qwen25/tmux-session.log`.
- `2026-03-10 10:43 UTC`: confirmed the background job is live: the `tmux` session persists, the `tee` logger is attached, and `python -m memtotal.tasks.writer_jointpeft_data` is actively materializing the deterministic medium-slice inputs before the control suites start.
- `2026-03-10 10:45 UTC`: confirmed the run has entered suite execution: `gsm8k-control/suite_metrics.json` exists, `narrativeqa-control/.suite.lock` is active, and the result root has started populating under `/root/autodl-tmp/results/generated/planv6-v6-4-mixed-matrix-qwen25`.
- `2026-03-10 12:28 UTC`: the full `39/39` suite matrix finished and published. The governed review bundle is now in `results/generated/review/planv6-v6-4-mixed-matrix-qwen25` and `runs/review/planv6-v6-4-mixed-matrix-qwen25`.
- `2026-03-10 12:29 UTC`: `v6-4-summary.json` selected the three `V6-5` finalists:
  - `s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage`
  - `s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg`
  - `s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive`
- `2026-03-10 12:30 UTC`: final ranking readout: `S3 > S5`, `C2 > C0`, and `L5 > L3 > L2`. `S3+C2+L5` was the strongest overall branch, with FEVER and GSM8K both clearing the repaired gates and NarrativeQA becoming route-live and stable for the first time in this `PLANv6` line, though still not usefulness-positive.

## Decision Log

- Keep both `C0` and `C2` in the matrix even though `S3` dominated `V6-3`; `V6-4` is explicitly an interaction test, not a single-axis re-ranking.
- Promote NarrativeQA in the ranking logic as a tie-breaker because it is the remaining weak task under the repaired gates.
- Carry exactly three finalists into `V6-5` because the matrix produced a clear top tier without collapsing into a single dominant recipe, and `PLANv6` explicitly authorizes `2-3` finalists.

## Surprises And Discoveries

- `V6-3` made `S3` look stronger than `S5` under the best auxiliary families, but the ranking still does not justify dropping `S5`; the unresolved question is whether `S5 + C0` or `S3 + C2` unlock combinations that single-axis screening could not see.
- The publish step can fail when ignored generated checkpoints and tmux logs accumulate under `runs/review/`. `V6-4` should retain the standard publisher path but harden it against that space-leak pattern.
- The mixed matrix resolved the `C0` versus `C2` question: `C2 support_and_context_gated` won the top two slots and paired best with both `L5` and `L3`, while `C0` remained useful mainly as the `L2` control-style finalist.
- `L5 orthogonality + coverage` was the only family that made NarrativeQA both route-live and stable inside the top branch, which makes it the strongest recipe-stabilization anchor for `V6-5`.
