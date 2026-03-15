# PLANv7 V7-6 Multi-seed Confirmation

## Purpose

Confirm the best `V7-5` branch across seeds `61109, 61110, 61111` on `GSM8K`, `TriviaQA`, and `FEVER`, while keeping the required control surface explicit:

- frozen no-memory control
- additive continuity baseline
- best earlier direct baseline when the promoted branch is bridge-based

`V7-6` is the paper-facing decision point for the external Writer line under `PLANv7.md`.

## Context

- `V7-5` is the last narrow aux revisit before the multi-seed gate.
- The confirmation logic must not let FEVER or logprob-only movement overrule primary-task score movement.
- The summary must distinguish:
  - real three-seed primary-task movement,
  - strict Writer-memory improvement,
  - projector-only illusion risk,
  - and the final `Path P / Q / R` decision.

## Plan of Work

1. Read the governed `V7-5` summary and automatically select the best one or two promoted branches.
2. Materialize a multi-seed confirmation matrix for:
   - `c0_frozen_no_memory`
   - `c1_additive_continuity`
   - `c2_best_direct` when required
   - `p1_<best_v7_5_arm>`
   - optionally `p2_<second_best_v7_5_arm>`
3. Run the matrix on `GSM8K`, `TriviaQA`, and `FEVER`.
4. Aggregate by seed and task, then decide between:
   - `Path P`: external Writer survives as main thesis
   - `Path Q`: unresolved but not dead
   - `Path R`: architecture pivot required
5. Publish the governed review bundle and relay the final decision back into the repo.

## Concrete Steps

1. Add `scripts/run_planv7_v7_6_multiseed_confirmation_qwen25.sh`.
2. Add `scripts/update_planv7_v7_6_multiseed_confirmation_summary.py`.
3. Add regression coverage in `tests/test_planv7_v7_6_multiseed_confirmation_summary.py`.
4. Update the review publish wiring only once `V7-5` closes cleanly.
5. Launch the `V7-6` tmux run after the `V7-5` milestone is committed and pushed.

## Validation & Acceptance

Static / local:

```bash
bash -n scripts/run_planv7_v7_6_multiseed_confirmation_qwen25.sh
python -m py_compile scripts/update_planv7_v7_6_multiseed_confirmation_summary.py
python -m unittest tests.test_planv7_v7_6_multiseed_confirmation_summary -v
python -m unittest discover -s tests -v
```

Run-time / governed:

```bash
bash scripts/run_planv7_v7_6_multiseed_confirmation_qwen25.sh \
  61109 \
  /root/autodl-tmp/runs/verify/planv7-v7-6-multiseed-confirmation-qwen25 \
  /root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25 \
  runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
  300 \
  /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
  results/generated/review/planv7-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json \
  results/generated/review/planv7-v7-0-metrics-oracle-qwen25/v7-0-summary.json
```

Detached launch:

```bash
tmux new-session -d -s planv7_v76 \
  "cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_v7_6_multiseed_confirmation_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-v7-6-multiseed-confirmation-qwen25 \
     /root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     300 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
     results/generated/review/planv7-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json \
     results/generated/review/planv7-v7-0-metrics-oracle-qwen25/v7-0-summary.json \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-v7-6-multiseed-confirmation-qwen25/tmux-session.log"
```

Acceptance:

- all requested seeds and tasks are present for each required branch/control
- the summary cleanly distinguishes `Path P / Q / R`
- FEVER is reported but never used to overrule the primary-task decision
- the best branch is checked against additive continuity, not only frozen control
- bridge winners are also checked against the earlier direct baseline

## Progress

- 2026-03-11 UTC: Started `V7-6` harness preparation while `V7-5` was still running, to avoid idle wait between milestones.
- 2026-03-11 UTC: Added the new multi-seed runner and summary script skeleton, plus regression coverage for the core `Path P` and `Path Q` decisions.
- 2026-03-11 UTC: Validation passed for the new `V7-6` surface: `bash -n`, `py_compile`, `python -m unittest tests.test_planv7_v7_6_multiseed_confirmation_summary -v`, and `python -m unittest discover -s tests -v` (`293` tests, `OK`).
- 2026-03-11 UTC: Deliberately deferred publish-hook edits and the live `V7-6` launch until the active `V7-5` run is closed and committed.
- 2026-03-11 16:40 UTC: `V7-5` closed with `comparison_conclusion=aux_revisit_flat_best_branch_for_decision_point` and selected `base_for_v7_6_arm_id=a5_barlow`, so the decision gate now uses the Barlow branch as `p1`.
- 2026-03-11 16:42 UTC: After pushing `main` commit `302e515` and republishing the `review` branch from that exact source commit, launched the governed `V7-6` run in detached `tmux` as `planv7_v76`.
- 2026-03-11 16:42 UTC: Live roots:
  - run root: `/root/autodl-tmp/runs/verify/planv7-v7-6-multiseed-confirmation-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25`
  - session log: `/root/autodl-tmp/runs/verify/planv7-v7-6-multiseed-confirmation-qwen25/tmux-session.log`
- 2026-03-11 16:42 UTC: Initial status: the detached job is alive and still in the early config/materialization stage; no confirmation suites have finished yet.
- 2026-03-12 UTC: The first `V7-6` attempt stopped on a data-disk checkpoint write failure while the matrix was nearly complete.
- 2026-03-12 UTC: Preserved partial state audit:
  - completed suites: `43/45`
  - missing suites:
    - `p2_a1_reconstruction/seed_61111/triviaqa`
    - `p2_a1_reconstruction/seed_61111/fever`
- 2026-03-12 UTC: Performed a non-experimental cleanup only. Deleted:
  - old GitHub snapshot temp trees under `/root/autodl-tmp/memtotal-github-review-*`
  - `/root/autodl-tmp/memtotal-review-snapshot-expanded`
  - old dry-run temp trees under `/root/autodl-tmp/tmp/m4-deep-*`
  - unused Hugging Face cache entries for `Qwen3-8B` and `NarrativeQA`
- 2026-03-12 UTC: No experimental raw run tree was deleted in this pass. The deletion manifest is recorded at `/root/runtime-archives/memtotal/deletion-manifests/20260312-v76-space-recovery-first-pass.md`.
- 2026-03-12 UTC: Data-disk headroom after cleanup: approximately `37G` free on `/root/autodl-tmp`.
- 2026-03-12 UTC: Relaunched the same `V7-6` runner in detached `tmux` as `planv7_v76` against the same run/result roots so it can resume from the preserved `43/45` state.
- 2026-03-12 UTC: Post-relaunch health check:
  - `tmux` session is alive
  - a GPU process is active again
  - the run remains targeted at the preserved missing `p2_a1_reconstruction/seed_61111` tail
- 2026-03-12 UTC: Performed approved experimental runtime-data preservation for the cold `V6-5` raw verify tree before deleting it from the data disk:
  - preserved source: `/root/autodl-tmp/runs/verify/planv6-v6-5-recipe-stabilization-qwen25`
  - preserved archive: `/root/runtime-archives/memtotal/experimental-runtime/20260312-planv6-v6-5-recipe-stabilization-qwen25-runtime-only.tar.zst`
  - checksum: `/root/runtime-archives/memtotal/experimental-runtime/20260312-planv6-v6-5-recipe-stabilization-qwen25-runtime-only.sha256`
  - manifest: `/root/runtime-archives/memtotal/experimental-runtime/20260312-planv6-v6-5-recipe-stabilization-qwen25-runtime-only.manifest.md`
  - scope: runtime-only files, explicitly excluding `checkpoint.pt` and `snapshot_evals/`
- 2026-03-12 UTC: The experimental preservation verified cleanly before deletion:
  - archived file count: `1388`
  - archived uncompressed bytes: `855635211`
  - compressed archive bytes: `90881891`
  - checksum status: `OK`
- 2026-03-12 UTC: Deleted the preserved cold raw run tree from the data disk and recorded the action at `/root/runtime-archives/memtotal/deletion-manifests/20260312-v65-runtime-preservation-and-deletion.md`.
- 2026-03-12 UTC: Data-disk headroom after experimental preservation + deletion increased to approximately `208G` free on `/root/autodl-tmp`, while the resumed `V7-6` run remained alive.
- 2026-03-12 UTC: After `V7-6` completed, performed a second approved cold-run cleanup pass to restore durable free-space margin on `/root/autodl-tmp`.
- 2026-03-12 UTC: Preserved runtime-only archives, checksums, and manifests for these deleted cold raw verify trees under `/root/runtime-archives/memtotal/experimental-runtime/` and `/root/runtime-archives/memtotal/deletion-manifests/`:
  - `planv7-v7-5-targeted-aux-revisit-qwen25`
  - `planv7-v7-3-bridge-qwen25`
  - `planv7-v7-4-forced-consumption-qwen25`
  - `planv6-v6-4-mixed-matrix-qwen25`
  - `planv6-v6-3-loss-screening-qwen25`
  - `planv6-v6-2-support-screening-qwen25`
  - `planv7-v7-2-direct-bandwidth-qwen25`
  - `planv7-v7-0-metrics-oracle-qwen25`
  - `planv7-v7-1-width-depth-scout-qwen25`
- 2026-03-12 UTC: Post-cleanup data-disk state reached approximately `575G` free and `46%` used on `/root/autodl-tmp`, which clears the `>=50% free` target.
- 2026-03-12 UTC: `V7-6` closed successfully with all `45/45` suites complete and governed outputs written to:
  - `/root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json`
  - `/root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25/v7-6-summary.md`
- 2026-03-12 UTC: Final governed decision:
  - `comparison_conclusion=path_r_architecture_pivot_required`
  - `recommended_next_step=prepare_backbone_native_writer_pivot`
  - `best_confirmed_variant_id=p1_a5_barlow`
  - `winner_uses_bridge=true`
  - `winning_depth=D1`
- 2026-03-12 UTC: Final readout:
  - no real three-seed primary-task gain on `GSM8K` or `TriviaQA`
  - no strict Writer-memory improvement sustained across the promoted branch set
  - both promoted bridge arms remained route-live and stable, but actual score movement stayed flat
  - this satisfies `PLANv7` `Path R`, so the external bridge line is no longer the active mainline after `V7-6`

## Decision Log

- Use the governed `V7-5` summary as the single selection authority for promoted arms.
- Keep the multi-seed confirmation generic so it can handle either one or two promoted `V7-5` branches.
- Treat additive continuity as the main trained control, with frozen no-memory as context and the earlier direct baseline as the extra bridge sanity check.

## Surprises & Discoveries

- The repo already has all of the required building blocks:
  - phase-to-phase selection through governed summary JSON
  - a stable `run_m4_selected_shared_injection_suite.py` arm contract
  - reusable summary helpers for strict Writer-memory metrics and route/stability checks
- The main missing piece was not model capability but a multi-seed aggregation layer that decides `Path P / Q / R` from governed artifacts instead of ad hoc reading.
