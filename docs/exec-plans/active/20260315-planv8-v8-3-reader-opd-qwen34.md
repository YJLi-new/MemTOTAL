# PLANv8 V8-3: Reader OPD on Qwen3-4B

## Purpose

Open `PLANv8` phase `V8-3` for the qwen34 line after the governed `V8-2` reader sweep.

This milestone asks:

> Can a warm-started Reader-only OPD sweep turn qwen34 `V8-2` diagnostic movement into actual primary-task movement before any trainable external-Writer phase is reopened?

## Gate

This phase is gated on the qwen34 `V8-2` summary:

- `/root/autodl-tmp/results/generated/planv8-v8-2-reader-sweep-qwen34/v8-2-summary.json`
- `recommended_next_step in {open_v8_3_reader_opd, open_v8_3_reader_opd_last_consumer_attempt}`

The qwen34 line is explicitly using the governed `V8-2` base arm:

- `base_for_v8_3_arm_id = best V8-2 arm`
- `selected_interface_family_for_v8_3 = best V8-2 interface family`
- each `V8-3` task arm warm-starts from `/root/autodl-tmp/runs/verify/planv8-v8-2-reader-sweep-qwen34/<base-arm>-<task>/checkpoint.pt`

## Scope

1. Reuse the exact qwen34 `V8-2` base arm and checkpoint as the fixed Reader interface for `V8-3`.
2. Keep Writer behavior frozen in the governed qwen34 Reader-only path.
3. Compare a CE-only warm-start control against legacy dense-teacher KL and OPD hint variants.
4. Publish the qwen34 `V8-3` review surface automatically after completion.

## Arm Matrix

The qwen34 harness implements the governed `V8-3` arm set from `PLANv8` section `15`:

- `p0_ce_only`
- `p1_teacher_choice_kl`
- `p2_opd_ansonly_w01`
- `p3_opd_ansonly_w03`
- `p4_opd_ansplusctx_w03`
- `p5_opd_ansplusctx_centered`

All arms:

- warm-start from the per-task qwen34 `V8-2` checkpoint
- keep `pilot_trainable_variant=reader_only`
- run `300` train steps with snapshots at `[0,10,25,50,100,150,200,250,300]`

## OPD Policy

The runtime exposes the qwen34 `V8-3` Reader OPD controls directly in `m4_shared_injection.py`:

- `pilot_alignment_aux_mode in {off, teacher_choice_kl, opd_token_ce, opd_token_ce_centered}`
- `pilot_opd_scope=reader_only`
- `pilot_opd_teacher_force_gold=true`
- `pilot_opd_mask_mode=target_only`
- `pilot_opd_advantage_clip=5.0`
- `pilot_opd_center=0.0`
- `pilot_opd_scale=0.5`

Task hint policy:

- `gsm8k`: answer-only or answer plus truncated gold rationale
- `triviaqa`: answer-only or answer plus evidence sentence(s)
- `fever`: label-only or label plus evidence

## Data Note

The active qwen34 TriviaQA source remains `rc.wikipedia.nocontext`. In this source, the available `entity_pages` and `search_results` context arrays are empty in the current materialization, so `answer_plus_evidence` can legitimately collapse to answer-only at runtime for qwen34 `V8-3`.

This is intentional and is reflected in the OPD diagnostics rather than treated as a harness failure.

## Planned Artifacts

- Runner:
  - `scripts/run_planv8_v8_3_reader_opd.sh`
  - `scripts/run_planv8_v8_3_reader_opd_qwen34.sh`
- Queue:
  - `scripts/queue_planv8_qwen34_v8_3_after_v8_2.sh`
- Config helper:
  - `scripts/planv8_v8_3_config.py`
- Summary:
  - `scripts/update_planv8_v8_3_summary.py`
- Tests:
  - `tests/test_opd_alignment.py`
  - `tests/test_planv8_v8_3_summary.py`
