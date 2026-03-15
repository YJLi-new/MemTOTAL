# PLANv7 (LR updated version) V7-1 Width Depth Scout

## Purpose

Run the first post-`V7-0` restart phase required by [`PLANv7-LR-updated.md`](/root/mydir/MemTOTAL/PLANv7-LR-updated.md): the four-arm low-cost width × depth scout under projector LR `7.5e-5`, with results published into the `planv7-lr75e5-*` namespace.

## Context

- Restarted `V7-0` completed with `comparison_conclusion=oracle_flat_direct_injection_high_risk`.
- The next authorized step remains `open_v7_1_width_depth_scout_keep_bridge_ready`.
- The restart must preserve the original `planv7-v7-1-*` historical bundle and publish into:
  - `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-1-width-depth-scout-qwen25`
  - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-1-width-depth-scout-qwen25`
  - `runs/review/planv7-lr75e5-v7-1-width-depth-scout-qwen25`
  - `results/generated/review/planv7-lr75e5-v7-1-width-depth-scout-qwen25`

## Plan Of Work

1. Add the remaining LR-updated `V7-1..V7-6` restart wrappers so downstream phases can stay in the new namespace.
2. Extend review publication wiring for all `planv7-lr75e5-v7-*` bundles.
3. Validate the restart harness.
4. Launch `V7-1` in detached `tmux`.
5. Publish the governed summary and select a single winning depth for restarted `V7-2`.
6. Commit/push `main`, then refresh `review`.

## Concrete Steps

1. Use [`scripts/run_planv7_lr75e5_v7_1_width_depth_scout_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_lr75e5_v7_1_width_depth_scout_qwen25.sh).
2. Run the exact `S00/S01/S10/S11` matrix from historical `V7-1`, but with:
   - `pilot_projector_learning_rate=7.5e-5`
   - `owner_locked_projector_lr=7.5e-5`
   - `repo_confirmed_v65_projector_lr_reference=7.5e-5`
   - `owner_override_note=false`
3. Publish the resulting review bundle with [`scripts/publish_review_artifacts.sh`](/root/mydir/MemTOTAL/scripts/publish_review_artifacts.sh).
4. Refresh the lightweight GitHub review branch with [`scripts/push_github_review_snapshot.sh`](/root/mydir/MemTOTAL/scripts/push_github_review_snapshot.sh) after the milestone commit.

## Validation & Acceptance

Static / local:

```bash
bash -n \
  scripts/run_planv7_lr75e5_v7_1_width_depth_scout_qwen25.sh \
  scripts/run_planv7_lr75e5_v7_2_direct_bandwidth_qwen25.sh \
  scripts/run_planv7_lr75e5_v7_3_bridge_qwen25.sh \
  scripts/run_planv7_lr75e5_v7_4_forced_consumption_qwen25.sh \
  scripts/run_planv7_lr75e5_v7_5_targeted_aux_revisit_qwen25.sh \
  scripts/run_planv7_lr75e5_v7_6_multiseed_confirmation_qwen25.sh \
  scripts/publish_review_artifacts.sh
python -m unittest tests.test_repo_lints tests.test_repo_contract -v
```

Run-time:

```bash
tmux new-session -d -s planv7_lr75e5_v71 \
  "mkdir -p /root/autodl-tmp/runs/verify/planv7-lr75e5-v7-1-width-depth-scout-qwen25 \
            /root/autodl-tmp/results/generated/planv7-lr75e5-v7-1-width-depth-scout-qwen25 && \
   cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_lr75e5_v7_1_width_depth_scout_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-lr75e5-v7-1-width-depth-scout-qwen25 \
     /root/autodl-tmp/results/generated/planv7-lr75e5-v7-1-width-depth-scout-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     200 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-lr75e5-v7-1-width-depth-scout-qwen25/tmux-session.log"
```

Acceptance for restarted `V7-1`:

- all four scout arms finish,
- ranking is computed under the strict gates,
- one winning depth is selected for restarted `V7-2`,
- FEVER does not override the primary-task result,
- all artifacts publish into the `planv7-lr75e5-v7-1-*` namespace.

## Progress

- 2026-03-12 UTC: Opened restart-specific `V7-1` after the LR-updated `V7-0` milestone was published on GitHub.
- 2026-03-12 UTC: Added restart wrappers for `V7-1..V7-6` and extended review publication wiring for all `planv7-lr75e5-v7-*` namespaces.
- 2026-03-12 UTC: Validation passed:
  - `bash -n` on the new wrappers and `scripts/publish_review_artifacts.sh`
  - `python -m unittest tests.test_repo_lints tests.test_repo_contract -v`
- 2026-03-12 UTC: First `tmux` launch attempt failed before training start because `tee` opened `tmux-session.log` before the new run root existed.
- 2026-03-12 UTC: Corrected the launch procedure to `mkdir -p` the run/result roots before starting `tee`, then relaunched `planv7_lr75e5_v71`.
- 2026-03-12 UTC: The corrected relaunch completed setup cleanly:
  - materialized datasets, manifests, and all ten configs exist,
  - `prepare_local_qwen25_model.sh` finished successfully,
  - the first completed suite is `gsm8k-control`,
  - GPU-backed execution is live under the `planv7-lr75e5-v7-1-*` namespace.
- 2026-03-12 UTC: The scout moved into the injected matrix and reached `4/10` completed suites while remaining GPU-backed.
- 2026-03-12 UTC: Armed a detached post-completion watcher session `planv7_lr75e5_v71_post` that waits for `v7-1-summary.json`, then:
  - refreshes governed review artifacts,
  - commits the LR-updated `V7-1` milestone,
  - pushes `main`,
  - refreshes the lightweight `review` branch,
  - and logs to `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-1-width-depth-scout-qwen25/postpublish.log`.

## Decision Log

- Reuse the existing `V7-1` matrix and summary logic; only the optimizer override and namespace change in this restart slice.
- Wire all later restart wrappers now so the repo is ready for uninterrupted autonomous continuation if `V7-1` promotes cleanly.

## Surprises & Discoveries

- The original restart prep only published `V7-0`, so downstream restart phases needed their own wrappers and review-sync entries before the LR-updated line could continue safely.
