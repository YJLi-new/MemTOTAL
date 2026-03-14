# PLANv8 V8-2: Reader Sweep on Qwen3-4B

## Purpose

Open `PLANv8` phase `V8-2` for the qwen34 line after the completed `V8-1` reader-interface scout.

This milestone asks:

> Given the best qwen34 `V8-1` interface family, does a governed sweep over layer band, rank, and learning rate produce real consumer-side signal before `V8-3`?

## Gate

This phase is gated on the qwen34 `V8-1` summary:

- `/root/autodl-tmp/results/generated/planv8-v8-1-reader-interface-scout-qwen34/v8-1-summary.json`
- `recommended_next_step in {open_v8_2_reader_sweep, open_v8_2_reader_sweep_last_chance}`

For the first qwen34 `V8-2` launch, the governing `V8-1` result is:

- `best_arm_id = i0_prefix_legacy_r2`
- `selected_interface_family_for_v8_2 = ri0_legacy_prefix`
- `recommended_next_step = open_v8_2_reader_sweep_last_chance`

So this qwen34 branch is explicitly the last governed Reader-only sweep before `V8-3`.

## Scope

1. Implement the generic `V8-2` sweep harness from `PLANv8.md` section `14`.
2. Use the explicit layer-band definitions from the plan:
   - `mid8 = [14,15,16,17,18,19,20,21]`
   - `mid12 = [12,13,14,15,16,17,18,19,20,21,22,23]`
   - `late8 = [20,21,22,23,24,25,26,27]`
3. Keep the qwen34 run in a separate governed namespace:
   - `/root/autodl-tmp/runs/verify/planv8-v8-2-reader-sweep-qwen34`
   - `/root/autodl-tmp/results/generated/planv8-v8-2-reader-sweep-qwen34`
4. Reuse the prompt-family selection already locked by qwen34 `V8-0`.
5. Publish the qwen34 `V8-2` review surface automatically after completion.

## Sweep Matrix

The runner implements the governed `V8-2` arm matrix:

- `r0_mid8_r32_lr5e5`
- `r1_mid8_r64_lr1e4`
- `r2_mid12_r64_lr1e4`
- `r3_mid12_r64_lr2e4`
- `r4_late8_r32_lr1e4`
- `r5_mid8_r16_lr5e5`

The qwen34 harness applies that matrix to the interface family selected by `V8-1`:

- `ri0_legacy_prefix`: sweep deep-prefix layer band, deep-prefix rank, and learning rate while keeping the receiver micro-LoRA pinned to the legal tiny base selected by `V8-1`
- `ri1_prepend_block`: sweep reader-LoRA layer band, rank, and learning rate
- `ri2_cross_attn`: sweep adapter layer band plus an FF-hidden proxy dimension that tracks the governed rank values

## Repair Note

The first qwen34 launch exposed a real contract mismatch for the `ri0_legacy_prefix` branch: `writer_direct + legacy_prefix` only permits tiny receiver micro-LoRA sets, so blindly sweeping `mid8`/`mid12` receiver-LoRA targets at ranks `16-64` is invalid.

The repaired harness therefore keeps the receiver micro-LoRA fixed to the legal tiny `V8-1` base arm and applies the governed `V8-2` layer-band/rank sweep to the deep-prefix projector path instead.

## Planned Artifacts

- Runner:
  - `scripts/run_planv8_v8_2_reader_sweep.sh`
  - `scripts/run_planv8_v8_2_reader_sweep_qwen34.sh`
- Queue:
  - `scripts/queue_planv8_qwen34_v8_2_after_v8_1.sh`
- Summary:
  - `scripts/update_planv8_v8_2_summary.py`
- Tests:
  - `tests/test_planv8_v8_2_summary.py`
