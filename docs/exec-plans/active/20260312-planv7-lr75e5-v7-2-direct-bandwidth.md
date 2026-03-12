# PLANv7 (LR updated version) V7-2 Direct Bandwidth Ladder

## Purpose

Run restarted `V7-2` after LR-updated `V7-1` again selected `D1 / mid4`, using the same governed matrix as historical `V7-2` but under projector LR `7.5e-5` and the `planv7-lr75e5-*` namespace.

## Context

- LR-updated `V7-1` completed with:
  - `comparison_conclusion=select_mid4_for_v7_2`
  - `winning_depth=D1`
  - `recommended_next_step=open_v7_2_direct_bandwidth_mid4`
- `PLANv7-LR-updated.md` preserves the original `PLANv7` science contract and changes only:
  - `pilot_projector_learning_rate=7.5e-5`
  - `owner_locked_projector_lr=7.5e-5`
  - `repo_confirmed_v65_projector_lr_reference=7.5e-5`
  - `owner_override_note=false`
- Historical `V7-2` already established the repo/runtime surface for:
  - `D_W1_shared`
  - `D_W2_shared`
  - `D_W2_perlayer`
  - FEVER guardrail after primary ranking

## Plan Of Work

1. Reuse the existing governed `V7-2` runner via the LR-updated wrapper.
2. Publish into the dedicated restart namespace.
3. Monitor primary completion and FEVER guardrail completion.
4. Publish the restarted `V7-2` governed bundle.
5. Commit and push the milestone, then refresh `review`.

## Concrete Steps

1. Use [`scripts/run_planv7_lr75e5_v7_2_direct_bandwidth_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv7_lr75e5_v7_2_direct_bandwidth_qwen25.sh).
2. Launch with explicit `mkdir -p` pre-creation of run/result roots to avoid the known `tee` startup footgun.
3. Keep all historical settings identical except the LR-updated restart override.
4. Arm a detached post-completion watcher that waits for `v7-2-summary.json`, then:
   - refreshes governed review artifacts,
   - commits the milestone,
   - pushes `main`,
   - refreshes the lightweight `review` branch.

## Validation & Acceptance

Static / local:

```bash
bash -n \
  scripts/run_planv7_lr75e5_v7_2_direct_bandwidth_qwen25.sh \
  scripts/publish_review_artifacts.sh
python -m unittest tests.test_repo_lints tests.test_repo_contract -v
```

Run-time:

```bash
tmux new-session -d -s planv7_lr75e5_v72 \
  "mkdir -p /root/autodl-tmp/runs/verify/planv7-lr75e5-v7-2-direct-bandwidth-qwen25 \
            /root/autodl-tmp/results/generated/planv7-lr75e5-v7-2-direct-bandwidth-qwen25 && \
   cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_lr75e5_v7_2_direct_bandwidth_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-lr75e5-v7-2-direct-bandwidth-qwen25 \
     /root/autodl-tmp/results/generated/planv7-lr75e5-v7-2-direct-bandwidth-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     200 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-lr75e5-v7-2-direct-bandwidth-qwen25/tmux-session.log"
```

Acceptance for restarted `V7-2`:

- all three primary arms finish on `GSM8K` and `TriviaQA`,
- FEVER runs only for control plus the promoted top two arms,
- the governed summary answers the `PLANv7` `V7-2` questions,
- artifacts publish into `planv7-lr75e5-v7-2-*`,
- the milestone is committed and pushed after publication.

## Progress

- 2026-03-12 UTC: Opened restart-specific `V7-2` after LR-updated `V7-1` again selected `D1 / mid4`.
- 2026-03-12 UTC: Validation passed:
  - `bash -n scripts/run_planv7_lr75e5_v7_2_direct_bandwidth_qwen25.sh scripts/publish_review_artifacts.sh`
  - `python -m unittest tests.test_repo_lints tests.test_repo_contract -v`
- 2026-03-12 UTC: Launched restarted `V7-2` in detached `tmux` as `planv7_lr75e5_v72` with:
  - run root: `/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-2-direct-bandwidth-qwen25`
  - result root: `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-2-direct-bandwidth-qwen25`
- 2026-03-12 UTC: Armed detached post-completion publisher `planv7_lr75e5_v72_post`, which waits for `v7-2-summary.json`, then:
  - refreshes governed review artifacts,
  - commits the LR-updated `V7-2` milestone,
  - pushes `main`,
  - refreshes the lightweight `review` branch.
- 2026-03-12 UTC: Early live state:
  - both controls (`gsm8k-control`, `triviaqa-control`) completed,
  - active primary arm is `gsm8k-d_w1_shared`,
  - `2/9` suite-level outputs are already present.

## Decision Log

- Reuse the historical `V7-2` runtime surface rather than fork the phase logic again; this restart is an optimizer-controlled replay, not a new scientific branch.
- Keep FEVER as a post-ranking guardrail only, exactly as in the original `PLANv7` contract.

## Surprises & Discoveries

- None yet at phase open.
