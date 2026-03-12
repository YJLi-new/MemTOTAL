# PLANv7 (LR updated version) Restart

## Purpose

Restart the `PLANv7` sequence with projector LR `7.5e-5` and a non-overwriting artifact namespace, beginning again at `V7-0`.

## Context

- Historical `PLANv7` completed and selected `Path R` under projector LR `7.5e-6`.
- The owner explicitly requested a restart with projector LR `7.5e-5`.
- The restart must preserve the original `PLANv7` bundles and publish under a new namespace.

## Plan Of Work

1. Add a restart overlay doc and distinct artifact namespace.
2. Make the `V7` runner family read projector LR + owner metadata from env-backed defaults.
3. Add a dedicated `V7-0` restart wrapper using LR `7.5e-5`.
4. Add review-publish wiring for the restart `V7-0` bundle.
5. Validate the harness.
6. Launch restarted `V7-0`.
7. When `V7-0` completes, publish and push that milestone before opening `V7-1`.

## Concrete Steps

1. Create [`PLANv7-LR-updated.md`](/root/mydir/MemTOTAL/PLANv7-LR-updated.md).
2. Update [`AGENTS.md`](/root/mydir/MemTOTAL/AGENTS.md) so future agents open the restart doc first.
3. Patch the `V7` runner scripts so `PLANV7_PROJECTOR_LR` and related metadata are configurable without forking all historical scripts.
4. Add [`run_planv7_lr75e5_v7_0_metrics_oracle_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_lr75e5_v7_0_metrics_oracle_qwen25.sh).
5. Update [`publish_review_artifacts.sh`](/root/mydir/MemTOTAL/scripts/publish_review_artifacts.sh) for the restart `V7-0` namespace.
6. Validate with `bash -n` and repo contract/lint tests.
7. Launch restarted `V7-0` in detached `tmux`.

## Validation & Acceptance

Static / local:

```bash
bash -n scripts/run_planv7_v7_0_metrics_oracle_qwen25.sh \
  scripts/run_planv7_v7_1_width_depth_scout_qwen25.sh \
  scripts/run_planv7_v7_2_direct_bandwidth_qwen25.sh \
  scripts/run_planv7_v7_3_bridge_qwen25.sh \
  scripts/run_planv7_v7_4_forced_consumption_qwen25.sh \
  scripts/run_planv7_v7_5_targeted_aux_revisit_qwen25.sh \
  scripts/run_planv7_v7_6_multiseed_confirmation_qwen25.sh \
  scripts/run_planv7_lr75e5_v7_0_metrics_oracle_qwen25.sh \
  scripts/publish_review_artifacts.sh
python -m unittest tests.test_repo_lints tests.test_repo_contract -v
```

Run-time:

```bash
tmux new-session -d -s planv7_lr75e5_v70 \
  "cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_lr75e5_v7_0_metrics_oracle_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-lr75e5-v7-0-metrics-oracle-qwen25 \
     /root/autodl-tmp/results/generated/planv7-lr75e5-v7-0-metrics-oracle-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     200 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-lr75e5-v7-0-metrics-oracle-qwen25/tmux-session.log"
```

Acceptance for this prep slice:

- the restart docs exist,
- the restart runner launches with LR `7.5e-5`,
- the original `PLANv7` artifacts remain untouched,
- the restarted run writes into the `planv7-lr75e5-*` namespace.

## Progress

- 2026-03-12 UTC: Opened the restart after the owner requested `PLANv7 (LR updated version)` with projector LR `7.5e-5`.
- 2026-03-12 UTC: Added the restart overlay doc, updated `AGENTS.md` to point at it, patched the `V7` runner family to read projector LR + owner metadata from env defaults, added the dedicated `V7-0` LR-`7.5e-5` wrapper, and wired restart `V7-0` publication into `publish_review_artifacts.sh`.
- 2026-03-12 UTC: Validation passed:
  - `bash -n` on the touched `V7` runners, the restart wrapper, and `publish_review_artifacts.sh`
  - `python -m unittest tests.test_repo_lints tests.test_repo_contract -v`
- 2026-03-12 UTC: Pushed the restart-prep milestone to GitHub:
  - `main`: `1b6f3233f89ed154da43635f093f87c83bc4fe45`
  - `review`: `9bf1936...` from the same source commit
- 2026-03-12 UTC: Launched restarted `V7-0` in detached `tmux` as `planv7_lr75e5_v70` with:
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-0-metrics-oracle-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-0-metrics-oracle-qwen25`
  - wrapper: `scripts/run_planv7_lr75e5_v7_0_metrics_oracle_qwen25.sh`
- 2026-03-12 UTC: Initial live status: the tmux session is alive and the restart namespace has begun materializing datasets/configs/manifests/sources.
- 2026-03-12 UTC: Restarted `V7-0` completed and published under the governed review namespace:
  - `results/generated/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25/v7-0-summary.json`
  - `results/generated/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25/v7-0-summary.md`
- 2026-03-12 UTC: The LR-updated replay reproduced the historical `V7-0` decision:
  - `comparison_conclusion=oracle_flat_direct_injection_high_risk`
  - `recommended_next_step=open_v7_1_width_depth_scout_keep_bridge_ready`
  - `preferred_depth=indeterminate`
  - recorded metadata now correctly matches the restart override:
    - `owner_locked_projector_lr=7.5e-5`
    - `repo_confirmed_v65_projector_lr_reference=7.5e-5`
    - `owner_override_note=false`
- 2026-03-12 UTC: Restarted `V7-1` completed and was pushed to GitHub:
  - `main`: `6d779f68c07ac2ffcd830c340106f3218583a2b8`
  - `review`: `aaffef178d0319c1d41e1f37ddc033d1bd5aeedf`
- 2026-03-12 UTC: LR-updated `V7-1` again selected `D1 / mid4`:
  - `comparison_conclusion=select_mid4_for_v7_2`
  - `recommended_next_step=open_v7_2_direct_bandwidth_mid4`
  - `winning_depth=D1`
  - owner LR metadata remained aligned with the restart override (`7.5e-5`, `false`)
- 2026-03-12 UTC: Opened LR-updated `V7-2` under the restart namespace with detached run session `planv7_lr75e5_v72` and detached post-completion publisher `planv7_lr75e5_v72_post`.
- 2026-03-12 UTC: Restarted `V7-2` completed and was pushed to GitHub:
  - `main`: `1a9c9ac18babeaff27402cc22eaf40daf714230e`
  - `review`: `94043ec22ece969145170ccc5802b68fe89986c3`
- 2026-03-12 UTC: LR-updated `V7-2` reproduced the historical direct-bandwidth conclusion:
  - `comparison_conclusion=direct_32_slot_noisy_move_to_v7_3_bridge_first`
  - `recommended_next_step=open_v7_3_bridge_first_wide_writer`
  - `promoted_arms=d_w1_shared,d_w2_shared`
- 2026-03-12 UTC: Opened LR-updated `V7-3` under the restart namespace with detached run session `planv7_lr75e5_v73` and detached post-completion publisher `planv7_lr75e5_v73_post`.
- 2026-03-12 UTC: The first `V7-3` launch failed immediately with `NameError: name 'os' is not defined` inside the base bridge runner's config-materialization heredoc. The missing import was added to [`scripts/run_planv7_v7_3_bridge_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_v7_3_bridge_qwen25.sh), and targeted validation passed again:
  - `bash -n scripts/run_planv7_v7_3_bridge_qwen25.sh scripts/run_planv7_lr75e5_v7_3_bridge_qwen25.sh scripts/publish_review_artifacts.sh`
  - `python -m unittest tests.test_repo_lints tests.test_repo_contract -v`
- 2026-03-12 UTC: Re-launched LR-updated `V7-3` cleanly with the same namespace and re-armed the detached post-completion publisher after clearing the stale failed-launch session.
- 2026-03-12 UTC: Current live state:
  - both detached sessions are alive: `planv7_lr75e5_v73`, `planv7_lr75e5_v73_post`
  - configs are materialized under `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-3-bridge-qwen25/materialized-configs`
  - the phase is actively executing `gsm8k-control` via `run_m4_selected_shared_injection_suite.py`
- 2026-03-12 UTC: Added a queued `V7-4` relay so the restart line can continue unattended after `V7-3` milestone push:
  - queue doc: [`20260312-planv7-lr75e5-v7-4-forced-consumption.md`](/root/mydir/MemTOTAL/docs/exec-plans/active/20260312-planv7-lr75e5-v7-4-forced-consumption.md)
  - queue script: [`queue_planv7_lr75e5_v7_4_after_v7_3.sh`](/root/mydir/MemTOTAL/scripts/queue_planv7_lr75e5_v7_4_after_v7_3.sh)
  - detached queue session: `planv7_lr75e5_v74_queue`
  - guardrails: wait for `V7-3` summary, wait for `planv7_lr75e5_v73_post` to finish, require GitHub `main` to move off `1a9c9ac18babeaff27402cc22eaf40daf714230e`, then parse `recommended_next_step` before launching `V7-4`
- 2026-03-12 UTC: LR-updated `V7-3` completed and was pushed to GitHub:
  - `main`: `2e3efc4893ca0d33a8fb178886335cddd75ba850`
  - `review`: `6cb5ed6d72ddf21cc90dbbf6fdd38f31ac90dbba`
- 2026-03-12 UTC: LR-updated `V7-3` again selected the historical handoff:
  - `comparison_conclusion=bridge_stabilizes_wide_writer_tasks_flat_move_to_v7_4`
  - `recommended_next_step=open_v7_4_forced_consumption`
  - `top_bridge_arm_id=b_w3_q16_s8`
- 2026-03-12 UTC: The first `V7-4` launch failed with the same missing-import pattern as `V7-3`: `NameError: name 'os' is not defined` inside the base forced-consumption runner's config-materialization heredoc. The missing import was added to [`scripts/run_planv7_v7_4_forced_consumption_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_v7_4_forced_consumption_qwen25.sh), and the queue/publish harness was widened to carry the repair and the next relay files.
- 2026-03-12 UTC: Re-launched LR-updated `V7-4` cleanly. Current live state:
  - run session: `planv7_lr75e5_v74`
  - post-publisher: `planv7_lr75e5_v74_post`
  - current work is in `writer_jointpeft_data` materialization for `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-4-forced-consumption-qwen25`
- 2026-03-12 UTC: Added a queued `V7-5` relay so the restart line can continue unattended after `V7-4` milestone push:
  - queue doc: [`20260312-planv7-lr75e5-v7-5-targeted-aux-revisit.md`](/root/mydir/MemTOTAL/docs/exec-plans/active/20260312-planv7-lr75e5-v7-5-targeted-aux-revisit.md)
  - queue script: [`queue_planv7_lr75e5_v7_5_after_v7_4.sh`](/root/mydir/MemTOTAL/scripts/queue_planv7_lr75e5_v7_5_after_v7_4.sh)
  - detached queue session: `planv7_lr75e5_v75_queue`
  - guardrails: wait for `V7-4` summary, wait for `planv7_lr75e5_v74_post` to finish, require GitHub `main` to move off `2e3efc4893ca0d33a8fb178886335cddd75ba850`, then parse `recommended_next_step` before launching `V7-5`
- 2026-03-12 UTC: `V7-4` completed and pushed with:
  - `main`: `0a05e04a48a619e6e397d6e9bc39ba99c53504cc`
  - `review`: `446f0421a0df952a9276bfbb405c42c834aebd25`
- 2026-03-12 UTC: The first detached `V7-5` queue did not fire because it was armed with the post-`V7-4` head `0a05e04a48a619e6e397d6e9bc39ba99c53504cc` instead of the required pre-`V7-4` sentinel `2e3efc4893ca0d33a8fb178886335cddd75ba850`. The relay harness is being re-armed with the correct predecessor SHA so the queue can pass immediately once rechecked.
- 2026-03-12 UTC: Re-ran the corrected `V7-5` queue helper directly with pre-`V7-4` head `2e3efc4893ca0d33a8fb178886335cddd75ba850`. It immediately launched:
  - run session: `planv7_lr75e5_v75`
  - post-publisher: `planv7_lr75e5_v75_post`
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25`
- 2026-03-12 UTC: `V7-6` remains queued behind the forthcoming `V7-5` milestone push in detached session `planv7_lr75e5_v76_queue`, still guarded on `recommended_next_step=prepare_v7_6_decision_point`.

## Decision Log

- Preserve `PLANv7.md` as the scientific backbone instead of rewriting it.
- Use a restart overlay doc plus distinct artifact namespace rather than overwriting historical results.
- Make the V7 runners env-configurable so the same harness can execute both historical and restarted lines.
- Treat the LR-updated `V7-0` as a full milestone closeout before opening `V7-1`, because the owner asked for GitHub publication after each completed milestone.
- Continue phase-by-phase under the original `PLANv7` decision rules, even when the LR-updated replay reproduces the historical phase winner exactly.
- Fix the harness in-place when a launch bug is clearly script-local, then relaunch the same milestone and keep the review/publish automation attached to the repaired run rather than opening a side branch.
- For unattended owner-away execution, add guarded queue helpers for the next authorized phase rather than speculative launches; each queue must wait on summary publication, milestone push completion, and an explicit `recommended_next_step` match.
- When a guarded relay waits on GitHub head movement, the sentinel must be the pre-milestone head, not the just-pushed milestone head; otherwise the queue deadlocks after a successful push.

## Surprises & Discoveries

- The original `V7` runner family hardcoded `7.5e-6` directly in each phase script, so a clean restart requires a harness-level LR parameterization rather than only a one-off launch command.
- The base `V7-3` bridge runner had an unexercised missing `import os` in its second embedded Python block; the LR-updated restart surfaced it before the first suite, so the repair was safe and local.
- The base `V7-4` forced-consumption runner had the same missing-`import os` defect in its second embedded Python block, so the LR-updated restart is now surfacing a repeatable harness bug class across phase scripts rather than a one-off `V7-3` mistake.
