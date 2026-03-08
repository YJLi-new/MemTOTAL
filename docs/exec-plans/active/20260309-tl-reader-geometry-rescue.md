# 20260309 TL Reader Geometry Rescue

## Purpose

Execute `PLANv2.md` for the current live blocker in Workstream B, starting with RG-0 instrumentation so the repo can distinguish query-conditioning collapse, attention symmetry collapse, fuser duplication, and residual writer weakness without guessing.

## Context

- Active macro/narrow plan: `PLANv2.md`
- Live blocker: two-level FEVER bridge geometry collapse in the Reader/Fuser path
- Immediate requirement from `PLANv2.md`: finish RG-0 before any new full rescue run

## Plan Of Work

1. Add RG-0 diagnostics to the two-level Reader/Fuser path and thread them through train events, snapshot evals, top-level metrics, and dynamics recovery outputs.
2. Add or extend tests so the new diagnostics are enforced by the repo.
3. Run targeted validation plus one smoke-style diagnostic check.
4. Commit and push the completed RG-0 phase before starting RG-1 work.

## Concrete Steps

1. Extend `src/memtotal/models/memory.py` so the Reader exposes base queries, conditioned queries, context shift, and pre-softmax QK logits.
2. Extend `src/memtotal/training/m4_shared_injection.py` so `_prefix_stats` computes RG-0 reader/fuser/writer geometry metrics and propagates them to train/snapshot/final outputs.
3. Extend `src/memtotal/analysis/m4_shared_injection.py` so recovery CSVs and summary helpers surface the new scalar diagnostics.
4. Add tests in `tests/test_smoke_components.py` and `tests/test_m4_shared_injection.py`.
5. Run targeted tests and a direct two-level smoke diagnostic.

## Validation & Acceptance

- New RG-0 metrics appear in:
  - `train_events.json`
  - `snapshot_evals/*/metrics.json`
  - top-level `metrics.json`
  - dynamics recovery CSV outputs
- Targeted tests pass.
- A smoke-style two-level runtime check shows non-empty, finite diagnostics.

## Progress

- 2026-03-08 18:28 UTC: Created the active exec-plan file for the TL Reader geometry rescue.
- 2026-03-08 18:28 UTC: Started RG-0 instrumentation implementation in the Reader, shared prefix-stat pipeline, and recovery analysis path.

## Decision Log

- Keep RG-0 narrow: add observability first, do not start RG-1 probe runs yet.
- Reuse `_prefix_stats` and `_prefix_scalar_summary` as the single serialization path so train/snapshot/final/recovery outputs stay aligned.
- Keep the Fuser forward API stable for now; derive RG-0 fuser geometry metrics from readouts, `M_short`, and the existing module state.

## Surprises & Discoveries

- The repo already had partial two-level diagnostics wired through `_prefix_stats`, which makes RG-0 much cheaper than a fresh instrumentation pass.
- The move of `AGENTS.md` to repo root left stale hardcoded test references that were fixed before starting RG-0 work.
