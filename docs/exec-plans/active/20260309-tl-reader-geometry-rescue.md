# 20260309 TL Reader Geometry Rescue

## Purpose

Execute `PLANv2.md` for the current live blocker in Workstream B, starting with RG-0 instrumentation so the repo can distinguish query-conditioning collapse, attention symmetry collapse, fuser duplication, and residual writer weakness without guessing.

## Context

- Active macro/narrow plan: `PLANv2.md`
- Live blocker: two-level FEVER bridge geometry collapse in the Reader/Fuser path
- RG-0 status: complete and merged
- RG-1 status: complete with published review artifacts under `results/generated/review/tl-reader-geometry-rescue-fever-qwen25/`
- Immediate next requirement from `PLANv2.md`: use the winning RG-1 control substrate to start RG-2 Reader specialization work

## Plan Of Work

1. Finish RG-0 diagnostics and validation so the repo can classify the collapse mode.
2. Materialize and execute the three RG-1 control probes from `PLANv2.md`.
3. Write the RG-1 decision into the repo with a concrete recommendation for RG-2.
4. Commit and push the completed RG-1 phase before editing the Reader for RG-2.

## Concrete Steps

1. Keep the RG-0 instrumentation path as the single source of truth for Reader/Fuser geometry serialization.
2. Add RG-1 config materialization, execution, and summary helpers for `CTX-OFF / H4-K8`, `CTX-OFF / H4-K4`, and `CTX-OFF / H4-K4 / linear`.
3. Publish the resulting review bundle under the new TL Reader geometry rescue directory.
4. Record which hypothesis currently wins among B-1a, B-1b, and B-1c before opening RG-2.

## Validation & Acceptance

- RG-0 metrics appear in:
  - `train_events.json`
  - `snapshot_evals/*/metrics.json`
  - top-level `metrics.json`
  - dynamics recovery CSV outputs
- RG-1 review artifacts exist for all three arms plus a top-level summary.
- Targeted tests pass.
- The repo has an explicit RG-2 control recommendation grounded in the RG-1 outputs.

## Progress

- 2026-03-08 18:28 UTC: Created the active exec-plan file for the TL Reader geometry rescue.
- 2026-03-08 18:28 UTC: Started RG-0 instrumentation implementation in the Reader, shared prefix-stat pipeline, and recovery analysis path.
- 2026-03-09 02:38 UTC: Started the RG-1 probe runner for `rg1a-ctxoff-h4-k8`, `rg1b-ctxoff-h4-k4`, and `rg1c-ctxoff-h4-k4-linear`.
- 2026-03-09 02:48 UTC: RG-1 probe runner finished, refreshed the review bundle, and wrote `rg1-summary.json` / `rg1-summary.md`.
- 2026-03-09 02:48 UTC: RG-1 selected `rg1c_ctxoff_h4_k4_linear` as the control substrate for RG-2.

## Decision Log

- Keep RG-0 narrow: add observability first, do not start RG-1 probe runs yet.
- Reuse `_prefix_stats` and `_prefix_scalar_summary` as the single serialization path so train/snapshot/final/recovery outputs stay aligned.
- Keep the Fuser forward API stable for now; derive RG-0 fuser geometry metrics from readouts, `M_short`, and the existing module state.
- RG-1A (`CTX-OFF / H4-K8`) did not materially move Reader specialization or selection.
- RG-1B (`CTX-OFF / H4-K4`) did not materially move Reader specialization or selection.
- RG-1C (`CTX-OFF / H4-K4 / linear`) materially increased `M_short` and fuser rank without making selection pass, so the current winner is `B-1c_linear_fuser`.
- RG-2 should use `CTX-OFF / H4-K4 / linear` as the control arm rather than repeating more RG-1 probes.

## Surprises & Discoveries

- The repo already had partial two-level diagnostics wired through `_prefix_stats`, which makes RG-0 much cheaper than a fresh instrumentation pass.
- The move of `AGENTS.md` to repo root left stale hardcoded test references that were fixed before starting RG-0 work.
- The RG-1 context-overwrite metric stayed at `0.0` in the baseline and all three probes, so the live failure is not behaving like B-1a.
- The Reader attention pairwise cosine stayed pinned at `1.0` across RG-1, yet the linear fuser still raised final `M_short` effective rank from `1.2098` to `3.9854`, which isolates the healthiest near-term lever to the Fuser side.
