# PLANv8 V8-1: Reader Interface Scout on Qwen3-4B

## Purpose

Open `PLANv8` phase `V8-1` for the qwen34 restart line after `V8-0` authorizes it.

This milestone asks:

> Which Reader interface gives the strongest first evidence that `Qwen3-4B` can consume external memory when the Writer is frozen / oracle?

## Gate

This phase is gated on the qwen34 `V8-0` summary:

- `/root/autodl-tmp/results/generated/planv8-v8-0-qwen34-baselines-oracles/v8-0-summary.json`
- `recommended_next_step = open_v8_1_reader_interface_scout`

## Scope

1. Reuse the existing `V8-1` reader matrix on the qwen34 backbone.
2. Keep the qwen34 run in a parallel namespace:
   - `/root/autodl-tmp/runs/verify/planv8-v8-1-reader-interface-scout-qwen34`
   - `/root/autodl-tmp/results/generated/planv8-v8-1-reader-interface-scout-qwen34`
3. Preserve the same governed prompt-family selection exported by qwen34 `V8-0`.
4. Publish the qwen34 `V8-1` review surface automatically after completion.

## Notes

- `Qwen3-4B` remains a `36`-layer Qwen3-family backbone, so the current `mid4 / mid8` reader layer geometry from the qwen3 `V8-1` harness is still aligned with the architecture.
- `update_planv8_v8_1_summary.py` now reads the generic `selected_primary_baseline_scores` field from `V8-0`, so the qwen34 line does not need qwen3-specific summary keys.

## Planned Artifacts

- Runner:
  - `scripts/run_planv8_v8_1_reader_interface_scout_qwen34.sh`
- Queue:
  - `scripts/queue_planv8_qwen34_v8_1_after_v8_0.sh`
- Review publication:
  - `scripts/publish_review_artifacts.sh`
