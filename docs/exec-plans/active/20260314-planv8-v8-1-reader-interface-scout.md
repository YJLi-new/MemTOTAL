# PLANv8 V8-1: Reader Interface Scout on Qwen3

## Purpose

Open `PLANv8` phase `V8-1` after the repaired `V8-0` calibration pass.

This milestone asks:

> Which Reader interface gives the strongest first evidence that `Qwen3-8B` can consume external memory when the Writer is frozen / oracle?

## Context

- Repaired `V8-0` decision surface:
  - `/root/autodl-tmp/results/generated/planv8-v8-0-qwen3-baselines-oracles-r1/v8-0-summary.json`
  - `comparison_conclusion = qwen3_calibrated_interfaces_alive_open_v8_1`
  - `recommended_next_step = open_v8_1_reader_interface_scout`
- Selected prompt modes frozen from `V8-0`:
  - `gsm8k -> q3_gsm8k_nonthink`
  - `triviaqa -> q3_trivia_think`
  - `fever -> answer_slot_labels`

## Scope

1. Add the missing runtime support for `pilot_trainable_variant=reader_only`.
2. Add a governed `V8-1` runner that:
   - keeps the Writer on `EW0` oracle hidden-state slots,
   - runs one control per task plus the six `V8-1` reader arms,
   - preserves the repaired `Qwen3` chat-template calibration from `V8-0`.
3. Add a `V8-1` summary builder that ranks reader arms by:
   - `GSM8K` task-score delta,
   - `TriviaQA` task-score delta,
   - primary-task delta sum,
   - primary answer-logprob movement,
   - memory-consumption diagnostics,
   - FEVER non-regression.
4. Extend the review publisher to surface `V8-1`.
5. Launch a watched tmux run with an auto-publish tail.

## Implementation Notes

- The current repo already supports:
  - `RI0` legacy deep-prefix,
  - `RI1` prepend-block sequence memory with receiver micro-LoRA,
  - `RI2` reader cross-attention adapters.
- The repo does **not** yet implement the dedicated `memory_exposure_projector` surface described in `PLANv8` section `10.2.3`.
- For this `V8-1` harness, the rank column is therefore mapped to the nearest implemented consumer-capacity knob:
  - `RI0`: deep-prefix projector bottleneck rank, plus tiny legacy receiver micro-LoRA,
  - `RI1`: receiver micro-LoRA rank,
  - `RI2`: cross-attention FF hidden width proxy (`r32 -> 2048`, `r64 -> 4096`).

This is an explicit approximation, not a silent claim that the full `10.2.3` surface already exists.

## Planned Artifacts

- Runner:
  - `scripts/run_planv8_v8_1_reader_interface_scout.sh`
- Summary:
  - `scripts/update_planv8_v8_1_summary.py`
- Tests:
  - `tests/test_m4_shared_injection.py`
  - `tests/test_planv8_v8_1_summary.py`
- Review publication:
  - `scripts/publish_review_artifacts.sh`

## Validation

Local:

```bash
python -m py_compile \
  src/memtotal/training/m4_shared_injection.py \
  scripts/update_planv8_v8_1_summary.py
python -m unittest \
  tests.test_m4_shared_injection \
  tests.test_planv8_v8_1_summary -v
bash -n \
  scripts/run_planv8_v8_1_reader_interface_scout.sh \
  scripts/publish_review_artifacts.sh
```

Governed:

- `v8-1-summary.json` and `v8-1-summary.md` are produced.
- One arm is selected as `base_for_v8_2_arm_id`.
- Review artifacts sync into the repo review surface.

## Planned Run Namespace

- run root:
  - `/root/autodl-tmp/runs/verify/planv8-v8-1-reader-interface-scout`
- result root:
  - `/root/autodl-tmp/results/generated/planv8-v8-1-reader-interface-scout`

## Progress

- 2026-03-14 UTC: Opened `V8-1` from the repaired `V8-0` baseline.
- 2026-03-14 UTC: Confirmed the missing runtime blocker is `reader_only` support; existing runtime only supported `full / projector_only / writer_adapter_only / receiver_then_joint`.
- 2026-03-14 UTC: Confirmed the repo lacks the dedicated `memory_exposure_projector` surface from `PLANv8`; documented the temporary capacity-proxy mapping used for `V8-1`.
