# PLANv7 V7-3 Wide External Writer With Bounded Query Compression

## Purpose

Run the bounded query-compression bridge sweep required by `PLANv7.md` after `V7-2` concluded that direct 32-slot injection was too noisy to keep as the mainline.

## Context

- `V7-2` published `comparison_conclusion=direct_32_slot_noisy_move_to_v7_3_bridge_first`.
- `PLANv7` fixes this phase to:
  - best depth from the prior phases,
  - `S3 + C2 + L5`,
  - groupwise clipping,
  - gradient accumulation `4`,
  - owner projector LR override,
  - seed `61109`,
  - train steps `300`.
- The exact bridge matrix is:
  - `B_ctrl` = best direct arm from `V7-2`, rerun as the direct matched control
  - `B_W3_q8`
  - `B_W3_q16`
  - `B_W3_q16_s8`
  - `B_W4_q16`
- This phase reopens the two-level Reader/Fuser path only as a bounded compression bridge. It is not a broad Reader sweep.

## Plan Of Work

1. Patch the stale runtime contract so `writer_direct + two_level` is explicitly supported for fresh runs.
2. Add a focused runtime test for `writer_direct + two_level + short_slots`.
3. Add the governed `V7-3` runner, summary script, summary test, and publish wiring.
4. Validate the harness with static checks, targeted tests, and the full unit suite.
5. Run the full `V7-3` matrix on `GSM8K` and `TriviaQA`.
6. Publish the governed summary, commit, and push the milestone.
7. Continue to the next authorized `PLANv7` phase based on the governed `V7-3` conclusion.

## Concrete Steps

1. Read `results/generated/review/planv7-v7-2-direct-bandwidth-qwen25/v7-2-summary.json` and use its top-ranked direct arm as the `B_ctrl` selection.
2. Keep the phase depth tied to the `V7-2` winning depth.
3. Materialize configs for:
   - `control` (rerun best direct arm from `V7-2`)
   - `b_w3_q8`
   - `b_w3_q16`
   - `b_w3_q16_s8`
   - `b_w4_q16`
4. Keep bridge arms on:
   - `pilot_memory_path_variant=two_level`
   - `pilot_projector_token_source=short_slots`
   - `pilot_reader_context_mode=prompt_summary`
   - `pilot_bridge_mode=writer_direct`
5. Run all bridge suites on `gsm8k` and `triviaqa`.
6. Publish:
   - `v7-3-summary.json`
   - `v7-3-summary.md`
7. Sync review artifacts and push once the phase is complete.

## Validation And Acceptance

`V7-3` is complete only if:

- the direct matched control finishes on both primary tasks,
- all four bridge arms finish on both primary tasks,
- the governed summary ranks all bridge arms against the matched direct control,
- the summary records direct-vs-two-level metadata and projector-mode metadata,
- and the phase is committed and pushed after publication.

Validation commands:

```bash
bash -n scripts/run_planv7_v7_3_bridge_qwen25.sh scripts/publish_review_artifacts.sh
python -m py_compile \
  scripts/update_planv7_v7_3_bridge_summary.py \
  src/memtotal/training/m4_shared_injection.py
python -m unittest discover -s tests -v
```

Live run command:

```bash
tmux new-session -d -s planv7_v73 \
  "cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_v7_3_bridge_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-v7-3-bridge-qwen25 \
     /root/autodl-tmp/results/generated/planv7-v7-3-bridge-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     300 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
     results/generated/review/planv7-v7-2-direct-bandwidth-qwen25/v7-2-summary.json \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-v7-3-bridge-qwen25/tmux-session.log"
```

## Progress

- 2026-03-11 00:00 UTC: Re-read `PLANv7.md` `V7-3` and confirmed the live branch is the bounded query-compression bridge sweep.
- 2026-03-11 00:00 UTC: Patched the runtime contract so `runtime.pilot_bridge_mode=writer_direct` explicitly allows `pilot_memory_path_variant in {single_level, two_level}`.
- 2026-03-11 00:00 UTC: Added a dedicated unit test covering `writer_direct + two_level + short_slots`.
- 2026-03-11 00:00 UTC: Implemented `scripts/run_planv7_v7_3_bridge_qwen25.sh`.
- 2026-03-11 00:00 UTC: Implemented `scripts/update_planv7_v7_3_bridge_summary.py` with corrected short-run tail handling for the `300`-step bridge phase.
- 2026-03-11 00:00 UTC: Added `tests/test_planv7_v7_3_bridge_summary.py`.
- 2026-03-11 00:00 UTC: Targeted validation passed for `bash -n`, `py_compile`, and the focused `V7-3` tests.
- 2026-03-11 21:05 UTC: Launched the full `V7-3` matrix in detached `tmux` as `planv7_v73`.
- 2026-03-11 21:42 UTC: `V7-3` is still in flight. Completed suites so far:
  - `control / gsm8k`
  - `control / triviaqa`
  - `b_w3_q8 / gsm8k`
  - `b_w3_q8 / triviaqa`
  - `b_w3_q16 / gsm8k`
  The active suite is `gsm8k-b_w3_q16_s8`.
- 2026-03-11 21:42 UTC: While `V7-3` was running, added and validated the next-phase runtime prerequisites inside `src/memtotal/training/m4_shared_injection.py`:
  - split backbone-vs-writer prompt routing via `pilot_backbone_prompt_mask_mode` and `pilot_writer_context_prompt_mode`,
  - explicit `receiver_then_joint` staging via `stage_a_steps` and `stage_b_steps`,
  - lightweight reconstruction auxiliary plumbing via `pilot_reconstruction_aux_*`.
- 2026-03-11 21:42 UTC: Added focused coverage for those runtime hooks in `tests/test_m4_shared_injection.py`; the touched validation set now passes:
  - `python -m py_compile src/memtotal/training/m4_shared_injection.py tests/test_m4_shared_injection.py`
  - `python -m unittest tests.test_m4_shared_injection tests.test_planv7_v7_3_bridge_summary -v`
- 2026-03-11 21:27 UTC: The full matrix finished and published the governed summary at `results/generated/planv7-v7-3-bridge-qwen25/v7-3-summary.json`.
- 2026-03-11 21:27 UTC: Final phase outcome:
  - `comparison_conclusion=bridge_stabilizes_wide_writer_tasks_flat_move_to_v7_4`
  - `recommended_next_step=open_v7_4_forced_consumption`
  - matched direct control: `d_w1_shared`
  - winning depth: `D1 / mid4`
  - top bridge arm: `b_w3_q16`
- 2026-03-11 21:27 UTC: Readout from the governed bridge bundle:
  - all bridge arms were route-live and stable on both `gsm8k` and `triviaqa`,
  - no bridge arm improved primary task score versus the matched direct control,
  - no bridge arm recovered the strict Writer-memory gate,
  - `b_w3_q16` ranked first by the governed bridge ranking, but still remained usefulness-flat.
- 2026-03-11 21:27 UTC: `V7-3` therefore authorizes `V7-4` forced memory consumption rather than a further bridge expansion or an immediate architectural pivot.

## Decision Log

- The matched direct control is not a no-memory baseline. It is the best direct injected arm from `V7-2`, rerun inside the `V7-3` phase budget.
- `V7-3` keeps the bridge search space tiny and fixed: only `B1`, `B2`, `B3`, plus the matched direct control.
- The new summary preserves the strict `V7-1/V7-2` gate semantics, but it corrects the tail-loss interpretation for `300`-step phases by using `train_loss_tail_50_steps_median` when the `451-500` window does not exist.
- Review publishing stays aligned with the governed result roots under `planv7-v7-3-bridge-qwen25`.
- Bridge compression does answer the bounded-bandwidth question: a much wider external Writer can be kept numerically stable behind a small two-level bridge.
- That does not answer the usefulness question. Because the wide bridge arms stayed flat on `gsm8k` and `triviaqa`, the next bottleneck is consumption, not bridge width.

## Surprises And Discoveries

- The old summary helper would silently misread `300`-step bridge runs because it expects a `451-500` tail-loss window. Without a phase-local patch, stable bridge arms would be reported as unstable.
- Fresh two-level writer-direct runs are now supported cleanly, but legacy checkpoints with missing bridge/variant metadata still have strict validation traps. This does not block the canonical fresh `V7-3` path.
- `b_w3_q16` beat the other bridge arms only on “stable wide-writer without regression” style evidence. It did not create a true task win, so the phase remains a bridge-stability result, not a usefulness result.
