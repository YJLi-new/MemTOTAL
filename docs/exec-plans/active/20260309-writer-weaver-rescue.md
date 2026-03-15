# 20260309 Writer-Weaver Rescue

## Purpose
Execute the first narrow branch authorized by `PLANv4.md`: train the Writer first, bypass Reader/Fuser, and test whether a MemGen-inspired weaver-style direct bridge can break the current common-mode collapse.

## Context
- Macro authority remains `PLAN.md`.
- `PLANv3.md` terminated at V4 with no actionable non-FEVER signal.
- Current blocker is Writer common-mode domination, not Reader-local geometry.
- User hard requirement: train Writer first; Reader is not the next concern.

## Plan of Work
1. Add a Writer-direct bridge mode.
2. Feed the Writer both support-side evidence and pre-answer reasoner hidden state.
3. Train the Writer under answer-span CE + no-memory delta gain + anti-common-mode losses.
4. Run Tier-1 smokes on GSM8K, NarrativeQA, FEVER-label-gen.
5. Only after positive geometry signal, open real Qwen2.5 multi-task pilots.

## Concrete Steps
1. Patch `memory.py` with `WriterWeaverHead` / stimulus modes.
2. Patch `backbone.py` to expose prompt-boundary hidden-state extraction.
3. Patch `m4_shared_injection.py` with:
   - `bridge_mode=writer_direct`
   - answer-span-only CE
   - paired no-memory forward
   - Writer geometry losses/logging
4. Add configs and runners for:
   - smoke
   - multitask pilot
5. Add tests for the direct bridge path and summary generation.

## Validation & Acceptance
- all new metrics appear in train/snapshot/final outputs;
- Reader/Fuser bypass does not break publishing;
- per-task smokes complete;
- at least one Writer-direct arm changes geometry versus control.

## Progress
- 2026-03-09: Created from `PLANv4.md`.
- 2026-03-09: Implemented the writer-direct W0 scaffold in `memory.py`, `backbone.py`, and `m4_shared_injection.py`, including `WriterWeaverHead`, prompt-boundary hidden-state extraction, answer-logprob delta logging, and Writer geometry metrics.
- 2026-03-09: Added W0 smoke configs, the `run_writer_weaver_smoke_qwen25.sh` runner, the `update_writer_weaver_summary.py` comparator, and a summary-script regression test.
- 2026-03-09: Fixed a writer-direct evaluation bug where snapshot/final eval tried to build prompt-independent prefixes; writer-direct now uses a representative eval prompt whenever checkpoint/save paths need a standalone prefix artifact.
- 2026-03-09: Completed the Tier-1 Qwen2.5 W0 smoke bundle on GSM8K, NarrativeQA, and FEVER-label-gen and published the review artifacts under `runs/review/writer-weaver-qwen25-smoke` and `results/generated/review/writer-weaver-qwen25-smoke`.
- 2026-03-09: W0 result is `comparison_conclusion=plumbing_only`, `move_to_w1=false`, `stop_after_w0=true`; the scaffold is live and context wiring is verified, but no non-FEVER `Delta(answer logprob)` signal appeared.

## Decision Log
- Do not reopen Reader rescue.
- Do not broaden receiver fallback.
- Do not inherit failed common-mode Writer checkpoints.
- Do not train on FEVER only.
- W1 is not authorized from this smoke result; the next live branch is `PLANv4` Section 18.1 fallback F1 (Writer-capacity upgrade before any receiver reopening).

## Surprises & Discoveries
- The initial W0 smoke exposed a real scaffold bug rather than a plan result: writer-direct snapshot/final eval still assumed prompt-independent prefixes and crashed when `support_and_context` required prompt context.
- After fixing that path, support-only and support+context both ran end-to-end on all Tier-1 tasks.
- Support+context clearly changes Writer geometry relative to support-only, especially `memory_long_top1_top2_ratio` and centered rank, but the common-mode ratio remains effectively pinned near `1.0` and `Delta(answer logprob)` stays exactly `0.0` on GSM8K and NarrativeQA.
- FEVER-label-gen remains a weak regression check here: both injected arms matched the control score (`0.5`) while still showing the same no-gain delta behavior.
