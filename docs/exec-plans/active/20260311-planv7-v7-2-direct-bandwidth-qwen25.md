# PLANv7 V7-2 Direct Bandwidth Ladder

## Purpose

Run the first main direct-bandwidth ladder required by `PLANv7.md` after `V7-1` selected `D1 / mid4`.

## Context

- `V7-1` completed with `comparison_conclusion=select_mid4_for_v7_2`, so `V7-2` is fixed to `D1 / mid4`.
- `PLANv7` fixes this phase to `S3 + C2 + L5`, groupwise clipping, accumulation `4`, receiver LoRA rank `2`, seed `61109`, and `200` train steps.
- The main primary-task matrix is:
  - `D_W1_shared`
  - `D_W2_shared`
  - `D_W2_perlayer`
- FEVER is a guardrail only. It must run after the primary ranking on the top two promoted arms plus the no-memory control, and it must not override the primary-task decision.
- This phase required a repo-level capability patch before launch: `pilot_deep_prefix_projector_mode=per_layer_low_rank` and the matching per-layer projector implementation/checkpoint metadata.

## Plan Of Work

1. Add the missing per-layer deep-prefix projector path and runtime metadata.
2. Add the `V7-2` governed runner, summary script, unit test, and publish wiring.
3. Validate the harness with static checks and the full test suite.
4. Run the primary matrix on `GSM8K` and `TriviaQA`.
5. Use the interim governed summary to promote exactly two arms into the FEVER guardrail.
6. Run FEVER on control plus the promoted arms.
7. Publish the final governed summary and push the completed milestone.

## Concrete Steps

1. Materialize `gsm8k`, `triviaqa`, and `fever` bundles via `writer_jointpeft_data`.
2. Build configs for:
   - control
   - `d_w1_shared`
   - `d_w2_shared`
   - `d_w2_perlayer`
3. Keep `D1 / mid4` fixed on all injected arms.
4. Run the primary tasks first.
5. Compute `v7-2-primary-summary.json` and extract `promoted_arms`.
6. Run FEVER only for `control` plus those two promoted arms.
7. Compute the final `v7-2-summary.json`, sync review artifacts, and publish.

## Validation And Acceptance

`V7-2` is complete only if:

- all three main arms finish on `GSM8K` and `TriviaQA`,
- the governed summary ranks the primary matrix and exposes the promoted top two arms,
- FEVER runs only on the promoted arms plus control,
- the summary answers the two `PLANv7` questions with evidence,
- and the phase is committed and pushed after publication.

Validation commands:

```bash
bash -n scripts/run_planv7_v7_2_direct_bandwidth_qwen25.sh scripts/publish_review_artifacts.sh
python -m py_compile \
  scripts/update_planv7_v7_2_direct_bandwidth_summary.py \
  src/memtotal/models/memory.py \
  src/memtotal/training/m4_shared_injection.py
python -m unittest discover -s tests -v
```

Live run command:

```bash
tmux new-session -d -s planv7_v72 \
  "cd /root/mydir/MemTOTAL && \
   bash scripts/run_planv7_v7_2_direct_bandwidth_qwen25.sh \
     61109 \
     /root/autodl-tmp/runs/verify/planv7-v7-2-direct-bandwidth-qwen25 \
     /root/autodl-tmp/results/generated/planv7-v7-2-direct-bandwidth-qwen25 \
     runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b \
     200 \
     /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
   2>&1 | tee /root/autodl-tmp/runs/verify/planv7-v7-2-direct-bandwidth-qwen25/tmux-session.log"
```

## Progress

- 2026-03-11 00:00 UTC: Re-read `PLANv7.md` `V7-2` and confirmed that the repo was missing `per_layer_low_rank` projector support.
- 2026-03-11 00:00 UTC: Added `PerLayerLowRankDeepPrefixProjector`, runtime resolver/metadata, checkpoint inference, and targeted tests.
- 2026-03-11 00:00 UTC: Began implementing the `V7-2` runner, governed summary, publish wiring, and summary unit test.
- 2026-03-11 00:00 UTC: Validation passed for `bash -n`, `py_compile`, targeted `V7-2` tests, and `python -m unittest discover -s tests -v` (`282` tests, `OK`).
- 2026-03-11 00:00 UTC: The first tmux launch exposed a startup bug where `tee` opened before the run directory existed. Restarted the detached session after pre-creating the run/result roots; no phase work was lost.
- 2026-03-11 00:00 UTC: The full ladder completed under `tmux` session `planv7_v72` and published the governed bundle at `/root/autodl-tmp/results/generated/planv7-v7-2-direct-bandwidth-qwen25`.
- 2026-03-11 00:00 UTC: Primary promotion selected `d_w1_shared` and `d_w2_shared` for the FEVER guardrail. `d_w2_perlayer` did not enter FEVER.
- 2026-03-11 00:00 UTC: Final outcome was `comparison_conclusion=direct_32_slot_noisy_move_to_v7_3_bridge_first` and `recommended_next_step=open_v7_3_bridge_first_wide_writer`.

## Decision Log

- `V7-2` is executed as a two-stage ladder: primary tasks first, FEVER second. This makes it impossible for FEVER to override the primary ranking.
- The new summary imports the already-governed `V7-1` replay helpers rather than forking the strict-gate logic again.
- The `W2_shared` arm keeps the Writer width increase while holding the projector to shared rank `64`, so the `W2_perlayer` arm isolates the projector-mode/rank change cleanly.
- The governed primary ranking was strong enough to reject the per-layer projector arm before FEVER: both promoted arms were shared-low-rank variants, and all three primary arms stayed flat on task score with `strict_writer_memory_task_count=0`.
- `V7-2` therefore does not support continuing the direct-bandwidth ladder. The next canonical branch is `V7-3`, and the bridge-first route should be treated as the mainline rather than as a fallback-only curiosity.

## Surprises And Discoveries

- `PLANv7` required a real projector-mode distinction, but the repo still treated every sparse deep-prefix run as shared low-rank before this phase.
- The warm-start/checkpoint fallback path also needed projector-mode inference, otherwise old checkpoints without explicit mode metadata would be misclassified.
- `W2_shared` briefly showed nonzero live-step `delta_answer_logprob` and modest attention mass during training, but the governed closeout stayed flat on actual primary-task outcomes.
- The FEVER guardrail completed successfully but did not change the milestone decision; FEVER remained a guardrail only, exactly as intended by `PLANv7`.
