# PLANv8 V8-8: Multi-Seed Confirmation on Qwen3-4B

## Purpose

Open `PLANv8` phase `V8-8` for the qwen34 line after `V8-7` clears the comparator gate.

This phase confirms only the smallest promoted branch set needed for a credible qwen34 decision:

- best Reader-only OPD route
- best current Writer-route continuation
- best bridge-compressed route only if compression previously helped and remains distinct

## Gate

This phase is authorized only when:

- `/root/autodl-tmp/results/generated/planv8-v8-7-comparators-qwen34/v8-7-summary.json`
- reports `recommended_next_step = open_v8_8_multiseed_confirmation`

## Confirmation Policy

The qwen34 harness replays the promoted branch recipes for seeds:

- `61109`
- `61110`
- `61111`

Each replay uses:

- `400` train steps
- the original source-phase materialized config as the recipe authority
- the original source-phase init checkpoint path embedded in that config

This means qwen34 `V8-8` confirms the recipe, not the already-trained branch checkpoint.

## Candidate Selection

`selection-manifest.json` is built from the existing qwen34 summaries:

- `V8-3` contributes the best Reader-only OPD arm
- `V8-6` contributes the best current Writer-route arm
- `V8-5` contributes the best bridge route only when compression was accepted and still adds a distinct branch

## Success Rule

A qwen34 candidate is confirmation-successful only if its 3-seed aggregate:

- improves at least one primary task over the qwen34 `V8-0` floor,
- stays non-regressive on the other primary task,
- keeps FEVER acceptable,
- and still shows a live route on at least one primary task.

If a candidate satisfies that rule, `V8-8` opens qwen34 `V8-9`.

## Governed Outputs

- `scripts/planv8_v8_8_config.py`
- `scripts/run_planv8_v8_8_multiseed_confirmation.sh`
- `scripts/run_planv8_v8_8_multiseed_confirmation_qwen34.sh`
- `scripts/queue_planv8_qwen34_v8_8_after_v8_7.sh`
- `scripts/update_planv8_v8_8_summary.py`
- `tests/test_planv8_v8_8_config.py`
- `tests/test_planv8_v8_8_summary.py`
