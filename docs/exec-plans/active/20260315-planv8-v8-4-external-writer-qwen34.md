# PLANv8 V8-4: External Writer Reintroduction on Qwen3-4B

## Purpose

Open `PLANv8` phase `V8-4` for the qwen34 line after a successful qwen34 `V8-3` Reader-OPD result.

This milestone asks:

> Once the qwen34 Reader path is shown to consume memory, can a trainable external Writer beat the qwen34 oracle/frozen Writer control?

## Gate

This phase is gated on the qwen34 `V8-3` summary:

- `/root/autodl-tmp/results/generated/planv8-v8-3-reader-opd-qwen34/v8-3-summary.json`
- `recommended_next_step == open_v8_4_external_writer`

The qwen34 `V8-4` harness consumes:

- `base_for_v8_4_arm_id = best qwen34 V8-3 arm`
- `selected_interface_family_for_v8_4 = best qwen34 V8-3 interface family`

## Runtime Note

The qwen34 `V8-3` checkpoint geometry is Reader-first and may not match the larger Writer slot counts in `V8-4`. The runtime therefore exposes:

- `pilot_trainable_variant=writer_then_joint`
- `pilot_init_checkpoint_mode=consumer_only`

That partial warm-start mode carries forward the compatible consumer-side state from qwen34 `V8-3` while allowing the Writer and projector geometry to change for the `V8-4` Writer sweep.

## Arm Matrix

The qwen34 harness implements the governed `V8-4` arm set:

- `w0_oracle64`
- `w1_ext2layer64_lr2e5`
- `w2_ext3layer64_lr2e5`
- `w3_ext3layer96_lr1e5`
- `w4_ext3layer64_lr5e5`

Interpretation:

- `w0_oracle64` is the within-phase oracle/frozen Writer control
- `w1-w4` reopen the trainable Writer with an `80`-step Writer/exposure warm start followed by `220` joint steps

## Schedule

Default qwen34 `V8-4` schedule:

- `pilot_train_steps = 300`
- `stage_a_steps = 80`
- `stage_b_steps = 220`
- `pilot_trainable_variant = writer_then_joint`

The qwen34 control arm keeps the Reader path active while leaving the Writer frozen.

## Planned Artifacts

- Config helper:
  - `scripts/planv8_v8_4_config.py`
- Runner:
  - `scripts/run_planv8_v8_4_external_writer.sh`
  - `scripts/run_planv8_v8_4_external_writer_qwen34.sh`
- Queue:
  - `scripts/queue_planv8_qwen34_v8_4_after_v8_3.sh`
- Summary:
  - `scripts/update_planv8_v8_4_summary.py`
- Tests:
  - `tests/test_planv8_v8_4_config.py`
  - `tests/test_planv8_v8_4_summary.py`
