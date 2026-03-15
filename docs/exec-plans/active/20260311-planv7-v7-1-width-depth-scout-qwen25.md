# PLANv7 V7-1 Width Depth Scout

## Purpose

Run the first post-`V7-0` direct-path scout required by `PLANv7.md`: the four-arm low-cost width × depth matrix that selects a single winning depth for `V7-2`.

## Context

- `V7-0` completed with `comparison_conclusion=oracle_flat_direct_injection_high_risk` and `recommended_next_step=open_v7_1_width_depth_scout_keep_bridge_ready`.
- `PLANv7` fixes `V7-1` to `S3 + C2 + L5`, groupwise clipping, accumulation `4`, owner projector LR override, receiver LoRA rank `2`, train steps `200`, and seed `61109`.
- The scout matrix is:
  - `S00 = W0 + D0 + P0`
  - `S01 = W0 + D1 + P0`
  - `S10 = W1 + D0 + P1`
  - `S11 = W1 + D1 + P1`
- Tasks are `GSM8K` and `TriviaQA` only. FEVER must not override the primary-task ranking for this phase.

## Plan Of Work

1. Add the `V7-1` harness surface:
   - `scripts/run_planv7_v7_1_width_depth_scout_qwen25.sh`
   - `scripts/update_planv7_v7_1_width_depth_summary.py`
   - summary unit test
   - publish wiring
2. Validate the harness with static checks and the full test suite.
3. Launch the full matrix in detached `tmux`.
4. Publish the governed summary and promote exactly one depth into `V7-2`.
5. Commit/push `main`, then refresh the lightweight `review` branch.

## Concrete Steps

1. Materialize `gsm8k` and `triviaqa` bundles via `writer_jointpeft_data`.
2. Build control plus four injected configs with the exact `W0/W1`, `D0/D1`, and `P0/P1` settings from `PLANv7`.
3. Run per-task control suites.
4. Run `S00/S01/S10/S11` for both tasks.
5. Copy governed artifacts into the result root.
6. Compute ranking using the strict Writer-memory gates and the V7 ranking order.
7. Publish review artifacts and GitHub-facing review output.

## Validation And Acceptance

`V7-1` is complete only if:

- all four scout arms finish,
- ranking is computed under the strict gates,
- one winning depth is selected for `V7-2`,
- FEVER is not used to override the primary-task result,
- and the summary exposes owner-LR metadata, projector/depth metadata, and bridge-path metadata.

Validation commands:

```bash
bash -n scripts/run_planv7_v7_1_width_depth_scout_qwen25.sh scripts/publish_review_artifacts.sh
python -m py_compile scripts/update_planv7_v7_1_width_depth_summary.py
python -m unittest discover -s tests -v
```

Live run command:

```bash
tmux new-session -d -s planv7_v71 \
  "cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_v7_1_width_depth_scout_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-v7-1-width-depth-scout-qwen25 \
     /root/autodl-tmp/results/generated/planv7-v7-1-width-depth-scout-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     200 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-v7-1-width-depth-scout-qwen25/tmux-session.log"
```

## Progress

- 2026-03-11 00:00 UTC: Opened the phase plan from completed `V7-0` and confirmed the `V7-1` matrix and ranking contract in `PLANv7.md`.
- 2026-03-11 00:00 UTC: Began implementing the new runner, summary script, publish wiring, and summary unit test.
- 2026-03-11 00:00 UTC: Validation passed for `bash -n`, `py_compile`, the new `V7-1` summary test, and `python -m unittest discover -s tests -v` (`279` tests, `OK`).
- 2026-03-11 00:00 UTC: Launched the full matrix in detached `tmux` session `planv7_v71` with live roots at `/root/autodl-tmp/runs/verify/planv7-v7-1-width-depth-scout-qwen25` and `/root/autodl-tmp/results/generated/planv7-v7-1-width-depth-scout-qwen25`.
- 2026-03-11 00:00 UTC: All `10/10` suites completed and the governed summary published at `/root/autodl-tmp/results/generated/planv7-v7-1-width-depth-scout-qwen25/v7-1-summary.json`.
- 2026-03-11 00:00 UTC: `comparison_conclusion=select_mid4_for_v7_2`, `winning_depth=D1`, `recommended_next_step=open_v7_2_direct_bandwidth_mid4`.

## Decision Log

- The phase uses the existing direct-path harness from `V7-0` rather than opening a new runtime surface. The main delta is the width/depth/projector matrix and the ranking contract.
- The summary ranks depths using `GSM8K`, then `TriviaQA`, then strict Writer-memory metrics, then answer-switch helpfulness, then stability, with explicit penalties for manufactured diversity and flat-primary behavior.
- FEVER is excluded from the `V7-1` matrix entirely to make it impossible for FEVER to overrule the primary-task result.
- The published `V7-1` decision keeps `mid4` as the promoted depth because `D1` was tied on primary-task score, clearly better on strict Writer-memory metrics, and incurred fewer penalties than `D0`.

## Surprises And Discoveries

- The existing runtime already exposes enough metadata in `metrics.json` to report projector rank, active depth layers, bridge mode, and memory-path mode without another model-code patch.
- `W1 + D0` on TriviaQA opened the strongest raw route in the scout but still collapsed by the strict Writer-memory gate at suite closeout, which reinforced the need to separate route-liveness from durable Writer-memory structure.
- `W0 + D1` on GSM8K was the only arm to pass `writer_memory_not_collapsed_strict`, which was sufficient to promote `D1` even though all task-score deltas stayed at `0.0`.
