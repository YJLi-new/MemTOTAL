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

## Decision Log
- Do not reopen Reader rescue.
- Do not broaden receiver fallback.
- Do not inherit failed common-mode Writer checkpoints.
- Do not train on FEVER only.

## Surprises & Discoveries
- pending
