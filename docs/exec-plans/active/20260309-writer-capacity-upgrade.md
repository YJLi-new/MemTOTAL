# 20260309 Writer Capacity Upgrade

## Purpose
Execute `PLANv4.md` fallback F1a after W0 finished as `plumbing_only`: increase WriterWeaver capacity before reopening any receiver-side idea, and test whether extra conditioning depth plus a less bottlenecked writer can finally move usefulness.

## Context
- Macro authority remains `PLAN.md`.
- Narrow authority is `PLANv4.md` Section 18.1 fallback F1.
- W0 already proved the writer-direct scaffold is real, published, and reproducible.
- W0 did not authorize W1/W2: `comparison_conclusion=plumbing_only`, `move_to_w1=false`, `stop_after_w0=true`.
- The likely remaining problem is not wiring but Writer capacity: W0 support+context changed geometry, yet `common_mode_energy_ratio` stayed pinned near `1.0` and non-FEVER `delta_answer_logprob` stayed `0.0`.

## Plan of Work
1. Increase WriterWeaver conditioning capacity without changing the frozen reasoner contract.
2. Keep the same direct-bridge losses and the same small Tier-1 slices used by W0.
3. Compare the new bigger-writer arm directly against the published W0 support+context baseline and W0 control.
4. Use the result to decide between opening W2 or moving deeper into fallback F1b.

## Concrete Steps
1. Extend `WriterWeaverHead` so it can stack multiple context/support conditioning passes.
2. Plumb the new writer-capacity knob through `m4_shared_injection.py` metadata and checkpoint payloads.
3. Add F1a config templates for GSM8K, NarrativeQA, and FEVER-label-gen with:
   - same latent token budget,
   - larger feed-forward hidden size,
   - deeper writer encoder,
   - deeper conditioning stack.
4. Add an F1a runner that:
   - reuses W0 review datasets and reference metrics,
   - runs one bigger support+context arm per Tier-1 task,
   - publishes a structured review bundle.
5. Add an F1a summary script and a unit test for its gate logic.

## Validation & Acceptance
- New writer-capacity code passes targeted unit tests.
- The F1a runner completes on all Tier-1 tasks and publishes review artifacts.
- The F1a summary reports one of:
  - `move_to_w2`
  - `move_to_f1b`
- No receiver-side PEFT is reintroduced.

## Progress
- 2026-03-09: Created from `PLANv4.md` fallback F1 after W0 stopped at `plumbing_only`.
- 2026-03-09: Extended `WriterWeaverHead` with stacked conditioning layers so F1a can increase cross-attention depth without increasing the latent token budget.
- 2026-03-09: Added F1a configs, the `run_writer_weaver_f1a_qwen25.sh` runner, the `update_writer_weaver_f1a_summary.py` comparator, and a dedicated summary-script unit test.
- 2026-03-09: Completed the F1a Qwen2.5 run on the W0 Tier-1 slices and published the review bundle under `runs/review/writer-weaver-qwen25-f1a` and `results/generated/review/writer-weaver-qwen25-f1a`.
- 2026-03-09: F1a result is `comparison_conclusion=move_to_f1b`, `move_to_w2=false`, `move_to_f1b=true`; the bigger writer stayed flat on usefulness and did not satisfy any weak geometry threshold.

## Decision Log
- Keep the frozen reasoner, injector, and usefulness losses unchanged.
- Reuse W0 control and support+context review artifacts as the explicit reference family.
- Keep the latent token budget fixed; increase writer capacity through depth and hidden width instead of more injected tokens.
- Do not open W2 from F1a; the next live branch is fallback F1b, a lightweight writer-side adapter.

## Surprises & Discoveries
- The real issue was not a hidden F1a wiring bug. The bigger writer ran cleanly on all three tasks with `pilot_writer_conditioning_layers=3`.
- Capacity alone did not help. Non-FEVER `delta_answer_logprob` stayed exactly `0.0`, task scores stayed flat, and the weak threshold gate remained closed everywhere.
- The larger writer pushed `memory_long_top1_top2_ratio` in the wrong direction, from the W0 support+context hundreds range into roughly `2.17e7` on GSM8K, `2.42e7` on NarrativeQA, and `2.68e7` on FEVER.
- `memory_long_common_mode_energy_ratio` also remained effectively maxed out at `1.0`, so the added conditioning depth widened the representation while preserving the same unusable common-mode collapse.
