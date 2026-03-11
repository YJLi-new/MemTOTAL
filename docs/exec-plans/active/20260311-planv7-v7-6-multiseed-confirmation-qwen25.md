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
