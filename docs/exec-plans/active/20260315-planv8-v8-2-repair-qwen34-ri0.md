# PLANv8 V8-2 Repair: Qwen3-4B `ri0` Sweep Contract

## Purpose

Repair the qwen34 `V8-2` reader sweep after the first live launch failed in the selected `ri0_legacy_prefix` branch.

## Root Cause

The qwen34 `V8-1` winner was:

- `best_arm_id = i0_prefix_legacy_r2`
- `selected_interface_family_for_v8_2 = ri0_legacy_prefix`

The original `V8-2` harness incorrectly mapped the governed sweep matrix directly onto both:

- deep-prefix layers/rank, and
- receiver micro-LoRA layers/rank

That produced illegal configs such as:

- `writer_direct`
- `legacy_prefix`
- receiver micro-LoRA target layers in `mid8` or `mid12`
- receiver micro-LoRA rank `16-64`

But the runtime contract for `writer_direct + legacy_prefix` only allows tiny receiver micro-LoRA sets:

- at most `5` target layers
- rank in `[1, 4]`
- alpha in `(0, 8]`

So the first qwen34 `V8-2` run stopped immediately at runtime validation before the first sweep arm.

## Repair

The repaired `V8-2` harness keeps the receiver micro-LoRA pinned to the legal tiny base selected by qwen34 `V8-1` and applies the governed `V8-2` sweep only to the deep-prefix projector path for `ri0_legacy_prefix`.

For qwen34 that means:

- fixed receiver micro-LoRA base:
  - target layers: `[16, 17, 18, 19]`
  - rank: `2`
  - alpha: `4.0`
- swept variables:
  - deep-prefix layer band
  - deep-prefix rank
  - learning rate

`ri1_prepend_block` and `ri2_cross_attn` remain unchanged.

## Code Surface

- Added reusable config materializer:
  - `scripts/planv8_v8_2_config.py`
- Rewired runner:
  - `scripts/run_planv8_v8_2_reader_sweep.sh`
- Added focused config coverage:
  - `tests/test_planv8_v8_2_config.py`
- Updated qwen34 `V8-2` milestone note:
  - `docs/exec-plans/active/20260315-planv8-v8-2-reader-sweep-qwen34.md`

## Validation

- `python -m py_compile scripts/planv8_v8_2_config.py`
- `bash -n scripts/run_planv8_v8_2_reader_sweep.sh`
- `python -m unittest tests.test_planv8_v8_2_config tests.test_planv8_v8_2_summary -v`
- targeted runtime dry-run on the repaired qwen34 `ri0` arm using the live materialized datasets and qwen34 `V8-1` summary

## Run Plan

Re-arm qwen34 `V8-2` on the existing governed namespace after the repair lands:

- run session: `planv8_v82_q34`
- watcher: `planv8_v82_q34_watch`
- post tail: `planv8_v82_q34_post`

The existing qwen34 result root can be reused because only the control pilots completed before the failure.
