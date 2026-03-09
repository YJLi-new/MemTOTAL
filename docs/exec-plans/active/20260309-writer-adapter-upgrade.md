# 20260309 Writer Adapter Upgrade

## Purpose
Execute `PLANv4.md` fallback F1b after F1a ended at `move_to_f1b`: keep the bigger writer scaffold fixed, freeze the base writer, and test whether a tiny writer-side micro-LoRA on writer projection layers can create geometry signal without reopening receiver-side PEFT.

## Context
- Macro authority remains `PLAN.md`.
- Narrow authority is `PLANv4.md` Section 18.1 fallback F1b.
- W0 proved the writer-direct path is wired but only reached `plumbing_only`.
- F1a increased writer capacity and conditioning depth, then stopped at `move_to_f1b`; usefulness stayed flat and geometry regressed badly on `top1_top2_ratio`.
- `PLANv4` explicitly allows only a small writer-side adapter here, not a broader receiver fallback.

## Plan of Work
1. Add a tiny writer-only micro-LoRA path on addressable `WriterWeaverHead` projection layers.
2. Keep the rest of the direct bridge fixed: same frozen reasoner, same projector, same Tier-1 slices, same usefulness and anti-common-mode losses.
3. Run one F1b arm per Tier-1 task and compare it directly against W0 control, W0 support+context, and F1a.
4. Use the result to decide between reopening W2, opening F2 GRPO-lite, or stopping after F1b.

## Concrete Steps
1. Extend `WriterWeaverHead` with writer-local micro-LoRA support on conditioning and encoder projection layers.
2. Plumb writer-adapter config, trainability control, checkpoint metadata, and gradient logging through `m4_shared_injection.py`.
3. Add F1b config templates for GSM8K, NarrativeQA, and FEVER-label-gen with:
   - the same larger F1a writer shape,
   - tiny writer-only adapters,
   - `pilot_trainable_variant=writer_adapter_only`.
4. Add an F1b runner that:
   - reuses W0 and F1a review artifacts,
   - runs one adapter-only arm per Tier-1 task,
   - publishes a structured review bundle.
5. Add an F1b summary script and regression test for the phase gate.

## Validation & Acceptance
- Writer micro-LoRA wiring passes targeted unit tests.
- F1b runner completes on all Tier-1 tasks and publishes review artifacts.
- The F1b summary reports exactly one of:
  - `move_to_w2`
  - `move_to_f2`
  - `stop_after_f1b`
- No receiver-side adapter path is enabled.

## Progress
- 2026-03-09: Created from `PLANv4.md` fallback F1 after F1a resolved to `move_to_f1b`.
- 2026-03-09: Added writer-side micro-LoRA support on `WriterWeaverHead` conditioning and encoder projection layers, plus runtime metadata, trainability control, and regression coverage.
- 2026-03-09: Added F1b configs, the `run_writer_weaver_f1b_qwen25.sh` runner, the `update_writer_weaver_f1b_summary.py` comparator, and a dedicated F1b summary regression test.
- 2026-03-09: Completed the F1b Qwen2.5 run on the W0 Tier-1 slices and published the review bundle under `runs/review/writer-weaver-qwen25-f1b` and `results/generated/review/writer-weaver-qwen25-f1b`.
- 2026-03-09: F1b result is `comparison_conclusion=stop_after_f1b`, `move_to_w2=false`, `move_to_f2=false`, `stop_after_f1b=true`; the writer-side adapter reduced F1a’s worst geometry blow-up but never produced non-FEVER usefulness or a geometry win over W0.

## Decision Log
- Keep the larger F1a writer shape fixed so F1b measures adapter effect, not another architecture change.
- Freeze the base writer during F1b and train only the writer-side micro-LoRA plus the projector.
- Target only addressable writer projection layers; do not reopen the receiver micro-LoRA stack.
- Do not open F2 or W2 from this branch; `PLANv4` does not authorize that after an F1b no-signal result.

## Surprises & Discoveries
- The adapter path is mechanically wired, but it did not become a live learning surface in practice on these runs: `train_grad_norm_writer_adapter_steps_1_4_median` stayed `0.0` on GSM8K, NarrativeQA, and FEVER.
- F1b substantially reduced F1a’s catastrophic `memory_long_top1_top2_ratio` from roughly `2.17e7` / `2.42e7` / `2.68e7` down to roughly `2288.6` / `2454.8` / `2289.6`, but that still failed to beat the W0 support+context reference in the hundreds range.
- `memory_long_common_mode_energy_ratio` remained effectively pinned at `0.999999+` and `delta_answer_logprob` stayed exactly `0.0` on every Tier-1 task, so the branch never crossed from geometry cleanup into usefulness.
