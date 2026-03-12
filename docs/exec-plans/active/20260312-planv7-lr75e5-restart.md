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

## Decision Log

- Preserve `PLANv7.md` as the scientific backbone instead of rewriting it.
- Use a restart overlay doc plus distinct artifact namespace rather than overwriting historical results.
- Make the V7 runners env-configurable so the same harness can execute both historical and restarted lines.

## Surprises & Discoveries

- The original `V7` runner family hardcoded `7.5e-6` directly in each phase script, so a clean restart requires a harness-level LR parameterization rather than only a one-off launch command.
