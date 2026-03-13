# PLANv8 V8-0 Repair: Qwen3 Prompt Calibration and Summary Key Fixes

## Purpose

Repair the first governed `PLANv8` `V8-0` pass so its acceptance result reflects the real runtime state instead of two harness defects:

1. the summary builder currently reads several stale metric keys and incorrectly marks `RI1` as failed,
2. the `Qwen3-8B` generation baseline is under-calibrated because the harness still uses plain-text prompting plus a decode budget that is too short for the selected prompt families.

This repair remains part of `V8-0`; it does **not** open `V8-1` yet.

## Context

- Existing `V8-0` governed outputs live under:
  - run root: `/root/autodl-tmp/runs/verify/planv8-v8-0-qwen3-baselines-oracles`
  - result root: `/root/autodl-tmp/results/generated/planv8-v8-0-qwen3-baselines-oracles`
- Current summary says:
  - `comparison_conclusion = repair_qwen3_interface_before_v8_1`
  - `ri1_passed_basic_smoke = False`
  - `ri2_passed_basic_smoke = True`
- Direct artifact audit shows `RI1` is materially alive:
  - `pilot_memory_consumer_mode='reader_lora_sequence'` is present in the arm metrics,
  - `prefix_artifact_stats.memory_tokens_count = 8`,
  - prepend-block attention is non-trivial on the sequence arms.
- The main baseline defect is separate:
  - `Qwen3-8B` generation arms use `max_new_tokens=32`,
  - prompt input is sent as plain text instead of the model's chat template,
  - GSM8K exact-match currently compares the full generated string rather than the final numeric answer.

## Plan Of Work

1. Fix the `V8-0` summary builder so it reads the runtime's real key names and prefix-artifact fallbacks.
2. Add an opt-in backbone prompt formatting path for chat-template generation/scoring.
3. Recalibrate the `V8-0` runner for `Qwen3-8B`:
   - enable chat-template prompting,
   - disable native Qwen3 thinking blocks for governed calibration arms,
   - increase decode budgets for generation tasks,
   - use a final-answer GSM8K normalizer in this phase.
4. Add focused tests for:
   - chat-template prompt routing,
   - `gsm8k_final_answer` normalization,
   - summary fallback behavior on real `V8-0` metric shapes.
5. Validate locally, then relaunch `V8-0` in a fresh repair namespace.

## Concrete Steps

1. Patch:
   - `src/memtotal/models/backbone.py`
   - `src/memtotal/tasks/evaluator.py`
   - `src/memtotal/training/m4_shared_injection.py`
   - `scripts/run_planv8_v8_0_qwen3_baselines_oracles.sh`
   - `scripts/update_planv8_v8_0_summary.py`
2. Extend tests:
   - `tests/test_backbone_real_mode.py`
   - `tests/test_task_registry.py`
   - `tests/test_planv8_v8_0_summary.py`
3. Run focused validation.
4. Launch a repaired run namespace:
   - run root: `/root/autodl-tmp/runs/verify/planv8-v8-0-qwen3-baselines-oracles-r1`
   - result root: `/root/autodl-tmp/results/generated/planv8-v8-0-qwen3-baselines-oracles-r1`
5. Refresh review artifacts, commit, and push.

## Validation & Acceptance

Local:

```bash
python -m py_compile \
  src/memtotal/models/backbone.py \
  src/memtotal/tasks/evaluator.py \
  src/memtotal/training/m4_shared_injection.py \
  scripts/update_planv8_v8_0_summary.py
python -m unittest \
  tests.test_backbone_real_mode \
  tests.test_task_registry \
  tests.test_planv8_v8_0_summary -v
```

Governed repair acceptance:

- `RI1` summary status matches the actual sequence-memory metrics.
- `Qwen3-8B` prompt calibration no longer returns implausible all-zero primary baselines solely from truncation / prompt-format mismatch.
- the repaired summary makes the real next-step decision for `V8-0`.

## Progress

- 2026-03-13 UTC: Audited the first `V8-0` run and confirmed the current failure is mixed:
  - summary misread on `RI1`,
  - genuine `Qwen3` baseline under-calibration on generation tasks.
- 2026-03-13 UTC: Local direct probes against `/root/autodl-tmp/models/Qwen3-8B` confirmed:
  - `max_new_tokens=32` truncates both GSM8K and TriviaQA generations,
  - the tokenizer chat template exposes `enable_thinking=False`,
  - `Qwen3` TriviaQA short-answer behavior becomes usable under the chat template with thinking disabled,
  - GSM8K reaches the correct final answer once the decode budget is increased, but exact-match still needs final-answer normalization.

## Decision Log

- Treat this as a `V8-0` repair, not as permission to open `V8-1`.
- Keep chat-template prompting opt-in so existing non-`PLANv8` paths do not shift silently.
- Use a targeted `gsm8k_final_answer` normalizer in the governed calibration flow instead of broadening exact-match semantics project-wide.

## Surprises & Discoveries

- The current `V8-0` runtime already writes the useful `RI1` evidence; the summary simply was not reading it.
- `Qwen3-8B` local tokenizer supports `enable_thinking=False` directly in the template, which is a cleaner governed calibration switch than prompt hacks.
- On GSM8K, decode length and answer normalization are both required; either repair alone is insufficient.
