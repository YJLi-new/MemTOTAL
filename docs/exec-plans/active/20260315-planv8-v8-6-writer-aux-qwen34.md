# PLANv8 V8-6: Writer-Side Auxiliary Revisit on Qwen3-4B

## Purpose

Open `PLANv8` phase `V8-6` for the qwen34 line after the qwen34 `V8-5` bridge decision.

This milestone asks:

> Which narrowly targeted auxiliary signals help the qwen34 Writer encode more useful, less collapsed memory once the base route is already known to work?

## Gate

This phase is gated on the qwen34 `V8-5` summary:

- `/root/autodl-tmp/results/generated/planv8-v8-5-bridge-revisit-qwen34/v8-5-summary.json`
- `recommended_next_step in {open_v8_6_writer_aux, open_v8_6_writer_aux_full_route}`

The qwen34 `V8-6` harness consumes:

- `base_for_v8_6_arm_id = best qwen34 V8-5 route`
- `selected_interface_family_for_v8_6 = fixed qwen34 consumer interface family`
- `selected_bridge_family_for_v8_6 = fixed qwen34 route bridge family`

## Runtime Note

`V8-6` is not another route search. It restores qwen34 Writer training on top of the chosen `V8-5` route and varies only the auxiliary family.

Warm-start policy:

- every qwen34 `V8-6` arm uses a full checkpoint warm start from the chosen qwen34 `V8-5` route
- the route geometry, interface family, and bridge family remain fixed
- the Writer is reopened with the established `writer_then_joint` schedule

## Arm Matrix

The qwen34 harness implements the governed `V8-6` arm set:

- `a0_none`
- `a1_barlow`
- `a2_recon_bow`
- `a3_writer_opd_ans`
- `a4_writer_opd_ansctx`
- `a5_writer_opd_plus_recon`

Interpretation:

- `a0_none` is the within-phase no-aux control
- `a1_barlow` keeps only the lightweight Barlow family from the earlier V7 signal
- `a2_recon_bow` adds hashed bag-of-tokens reconstruction
- `a3_writer_opd_ans` uses the existing OPD path with answer-only hints
- `a4_writer_opd_ansctx` uses the same OPD path with richer answer-plus-context hints
- `a5_writer_opd_plus_recon` combines the contextual OPD hint with reconstruction

## Schedule

Default qwen34 `V8-6` schedule:

- `pilot_train_steps = 300`
- `stage_a_steps = 80`
- `stage_b_steps = 220`
- `pilot_trainable_variant = writer_then_joint`

Default auxiliary settings:

- `pilot_barlow_loss_weight = 0.02`
- `pilot_reconstruction_aux_weight = 0.02`
- `pilot_opd_weight_max = 0.1`
- `pilot_opd_start_step = 80`
- `pilot_opd_ramp_steps = 80`

This preserves the fixed qwen34 route while asking whether a small Writer-side auxiliary can improve primary score or Writer health without reopening a large loss sweep.

## Planned Artifacts

- Config helper:
  - `scripts/planv8_v8_6_config.py`
- Runner:
  - `scripts/run_planv8_v8_6_writer_aux.sh`
  - `scripts/run_planv8_v8_6_writer_aux_qwen34.sh`
- Queue:
  - `scripts/queue_planv8_qwen34_v8_6_after_v8_5.sh`
- Summary:
  - `scripts/update_planv8_v8_6_summary.py`
- Tests:
  - `tests/test_planv8_v8_6_config.py`
  - `tests/test_planv8_v8_6_summary.py`
