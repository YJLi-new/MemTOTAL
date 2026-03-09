# 20260309 Independent Writer Deep-Prefix Circuit

## Purpose
Execute the first live branch authorized by `PLANv5.md`: open the attention -> deep-prefix -> source gradient circuit for the independent Writer before any substantial Writer usefulness training.

## Context
- Macro authority remains `PLAN.md`.
- `PLANv4.md` terminated its active branch at `W0 -> F1a -> F1b`.
- The latest evidence closes `shallow_prefix + frozen receiver (+ writer-only fallback)` as a useful substrate.
- The project has **not** yet tested the intended contract:
  - deep prefix / per-layer KV injection,
  - independent Writer with per-layer projectors,
  - and minimal receiver LoRA only if needed to open routing.

## Immediate Goals
1. Enable `writer_direct + sparse_deep_prefix + optional receiver micro-LoRA`.
2. Add a `source_stub` calibration route for circuit opening.
3. Add per-layer projector initialization via KV-statistics / anchors.
4. Run:
   - P1a: source-stub deep-prefix, no LoRA
   - P2a: source-stub deep-prefix, tiny early4 receiver LoRA
5. Only after a live circuit appears, open Writer-direct training on the winning substrate.

## Canonical First Runs
### Run A
- source stub
- deep prefix
- selected layers `[0,1,2,3]`
- no receiver LoRA
- KV-stat init
- tasks: GSM8K, NarrativeQA, FEVER-label-gen

### Run B
- source stub
- deep prefix
- selected layers `[0,1,2,3]`
- receiver micro-LoRA on `k_proj/v_proj`
- rank `2`, alpha `4`
- KV-stat init
- same tasks

### Run C
- real Writer
- deep prefix
- W0-sized Writer
- support+context stimulus
- chosen substrate from A or B

## Required Code Changes
1. `src/memtotal/training/m4_shared_injection.py`
   - allow `writer_direct + receiver micro-LoRA` when `pilot_injection_mode=sparse_deep_prefix`;
   - add `source_stub` mode;
   - add per-layer prefix attention diagnostics and circuit metrics.

2. `src/memtotal/models/backbone.py`
   - expose calibration utilities for per-layer KV statistics / anchors.

3. `src/memtotal/models/memory.py`
   - add per-layer deep-prefix projector family;
   - add `kv_stat_match`, `semantic_anchor`, `hidden_state_anchor` init modes.

4. Configs / scripts / tests
   - add P1/P2 templates, runners, summary scripts, and contract tests.

## Gates
### Circuit-open gate
Pass only if at least one non-FEVER task shows:
- nonzero positive or at least non-flat `delta_answer_logprob`,
- nontrivial source or Writer gradient,
- nontrivial prefix attention mass on at least one selected layer.

### Stop rule
If both P1a and P2a fail to open the circuit, do **not** run bigger Writer or writer-only adapter branches again; escalate to the fallback architecture decision in `PLANv5.md`.

## Progress
- 2026-03-09: Created from `PLANv5.md`. No implementation has been started yet in this active exec plan.
