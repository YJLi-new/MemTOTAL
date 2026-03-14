# PLANv8 V8-5: Bridge / Compression Revisit on Qwen3-4B

## Purpose

Open `PLANv8` phase `V8-5` for the qwen34 line after a successful qwen34 `V8-4` external-Writer result.

This milestone asks:

> Can the qwen34 `V8-4` long-memory route be compressed through a two-level bridge without destroying the newly established consumer gains?

## Gate

This phase is gated on the qwen34 `V8-4` summary:

- `/root/autodl-tmp/results/generated/planv8-v8-4-external-writer-qwen34/v8-4-summary.json`
- `recommended_next_step == open_v8_5_bridge`

The qwen34 `V8-5` harness consumes:

- `base_for_v8_5_arm_id = best qwen34 V8-4 arm`
- `selected_interface_family_for_v8_5 = best qwen34 V8-4 interface family`

## Runtime Note

The qwen34 `V8-4` winner is a single-level route. The qwen34 `V8-5` bridge arms reopen the existing two-level path while preserving the same consumer interface family from `V8-4`.

Warm-start policy:

- `b0-b3` keep the `V8-4` Writer geometry and use a full warm start from the selected qwen34 `V8-4` checkpoint.
- `b4_q48_s16_x96` is the dedicated `x96` bridge arm. If the qwen34 `V8-4` winner already used `96` long slots, it also warm-starts. If the winner used `64` slots, this arm is materialized as a cold-start bridge-expansion arm and records that status explicitly in runtime metadata.

## Arm Matrix

The qwen34 harness implements the governed `V8-5` arm set:

- `b0_no_bridge`
- `b1_q16_s16`
- `b2_q32_s16`
- `b3_q32_s8`
- `b4_q48_s16_x96`

Interpretation:

- `b0_no_bridge` is the within-phase no-compression control
- `b1-b4` reopen the two-level `Reader/Fuser` bridge while keeping the qwen34 `V8-4` consumer interface family fixed

## Schedule

Default qwen34 `V8-5` schedule:

- `pilot_train_steps = 300`
- `pilot_trainable_variant = reader_only`
- `pilot_reader_fuser_bootstrap_steps = 0`

This keeps the Writer frozen while the bridge and consumer-side route adapt.

## Planned Artifacts

- Config helper:
  - `scripts/planv8_v8_5_config.py`
- Runner:
  - `scripts/run_planv8_v8_5_bridge_revisit.sh`
  - `scripts/run_planv8_v8_5_bridge_revisit_qwen34.sh`
- Queue:
  - `scripts/queue_planv8_qwen34_v8_5_after_v8_4.sh`
- Summary:
  - `scripts/update_planv8_v8_5_summary.py`
- Tests:
  - `tests/test_planv8_v8_5_config.py`
  - `tests/test_planv8_v8_5_summary.py`
