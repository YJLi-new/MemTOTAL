# 20260309 TL Reader Geometry Rescue

## Purpose

Execute `PLANv2.md` for the current live blocker in Workstream B, starting with RG-0 instrumentation so the repo can distinguish query-conditioning collapse, attention symmetry collapse, fuser duplication, and residual writer weakness without guessing.

## Context

- Active macro/narrow plan: `PLANv2.md`
- Live blocker: two-level FEVER bridge geometry collapse in the Reader/Fuser path
- RG-0 status: complete and merged
- RG-1 status: complete with published review artifacts under `results/generated/review/tl-reader-geometry-rescue-fever-qwen25/`
- RG-2 status: complete with published review artifacts under `results/generated/review/tl-reader-symmetry-break-fever-qwen25/`
- RG-3 status: complete with published review artifacts under `results/generated/review/tl-reader-local-bootstrap-fever-qwen25/`
- Immediate next requirement from `PLANv2.md`: stop the local Reader/Fuser rescue line at RG-3, publish the failure classification, and do not open RG-4 because no bootstrap arm produced geometry gain

## Plan Of Work

1. Finish RG-0 diagnostics and validation so the repo can classify the collapse mode.
2. Materialize and execute the three RG-1 control probes from `PLANv2.md`.
3. Write the RG-1 decision into the repo with a concrete recommendation for RG-2.
4. Commit and push the completed RG-1 phase before editing the Reader for RG-2.
5. Run the minimum RG-2 control / competitive / partition matrix and record the next stop-rule decision.
6. Run the minimum RG-3 bootstrap-only / bootstrap+reconstruction matrix and write the explicit stop-or-advance decision back into the repo.

## Concrete Steps

1. Keep the RG-0 instrumentation path as the single source of truth for Reader/Fuser geometry serialization.
2. Add RG-1 config materialization, execution, and summary helpers for `CTX-OFF / H4-K8`, `CTX-OFF / H4-K4`, and `CTX-OFF / H4-K4 / linear`.
3. Publish the resulting review bundle under the new TL Reader geometry rescue directory.
4. Record which hypothesis currently wins among B-1a, B-1b, and B-1c before opening RG-2.
5. Add the RG-2 Reader conditioning / attention modes, run the explicit symmetry-break matrix, and classify whether the bridge is now alive, only partially improved, or still dead.
6. Add the narrow RG-3 local bootstrap controls, compare them directly against the RG-2 partition control, and stop the rescue line if they do not improve collapse timing or bridge geometry.

## Validation & Acceptance

- RG-0 metrics appear in:
  - `train_events.json`
  - `snapshot_evals/*/metrics.json`
  - top-level `metrics.json`
  - dynamics recovery CSV outputs
- RG-1 review artifacts exist for all three arms plus a top-level summary.
- RG-2 review artifacts exist for control / competitive / partition plus a top-level summary.
- RG-3 review artifacts exist for bootstrap-only / bootstrap+reconstruction plus a top-level summary.
- Targeted tests pass.
- The repo has an explicit RG-2 control recommendation grounded in the RG-1 outputs.
- The repo has an explicit RG-3 go / no-go decision grounded in the RG-2 outputs.
- The repo has an explicit post-RG-3 stop decision grounded in the published comparison outputs.

## Progress

- 2026-03-08 18:28 UTC: Created the active exec-plan file for the TL Reader geometry rescue.
- 2026-03-08 18:28 UTC: Started RG-0 instrumentation implementation in the Reader, shared prefix-stat pipeline, and recovery analysis path.
- 2026-03-09 02:38 UTC: Started the RG-1 probe runner for `rg1a-ctxoff-h4-k8`, `rg1b-ctxoff-h4-k4`, and `rg1c-ctxoff-h4-k4-linear`.
- 2026-03-09 02:48 UTC: RG-1 probe runner finished, refreshed the review bundle, and wrote `rg1-summary.json` / `rg1-summary.md`.
- 2026-03-09 02:48 UTC: RG-1 selected `rg1c_ctxoff_h4_k4_linear` as the control substrate for RG-2.
- 2026-03-09 03:00 UTC: Started RG-2 implementation for Reader conditioning modes, Reader attention modes, the RG-2 summary helper, and the symmetry-break runner/config family.
- 2026-03-09 03:06 UTC: Started the RG-2 sweep from the `rg1c` checkpoint using the linear control, competitive Reader, and masked-partition Reader arms.
- 2026-03-09 03:14 UTC: RG-2 sweep finished and published `rg2-summary.json` / `rg2-summary.md` under `results/generated/review/tl-reader-symmetry-break-fever-qwen25/`.
- 2026-03-09 03:14 UTC: RG-2 concluded with partition-only partial gain and an explicit `move_to_rg3: true` handoff.
- 2026-03-09: Started RG-3 implementation for local Reader/Fuser bootstrap, the optional short reconstruction auxiliary, the RG-3 summary helper, and the RG-3 config family.
- 2026-03-09: RG-3 bootstrap-only preserved the forced specialization geometry but left `dominant_label_collapse_onset_step` at `0`, so it did not clear the RG-3 stop rule.
- 2026-03-09: RG-3 bootstrap+reconstruction also left `dominant_label_collapse_onset_step` at `0` and slightly regressed readout / short-rank metrics relative to the RG-2 partition control.
- 2026-03-09: Published `rg3-summary.json` / `rg3-summary.md` under `results/generated/review/tl-reader-local-bootstrap-fever-qwen25/` and closed RG-3 with `move_to_rg4: false`, `stop_after_rg3: true`, and final classification `B-1_local_bootstrap_failed`.

## Decision Log

- Keep RG-0 narrow: add observability first, do not start RG-1 probe runs yet.
- Reuse `_prefix_stats` and `_prefix_scalar_summary` as the single serialization path so train/snapshot/final/recovery outputs stay aligned.
- Keep the Fuser forward API stable for now; derive RG-0 fuser geometry metrics from readouts, `M_short`, and the existing module state.
- RG-1A (`CTX-OFF / H4-K8`) did not materially move Reader specialization or selection.
- RG-1B (`CTX-OFF / H4-K4`) did not materially move Reader specialization or selection.
- RG-1C (`CTX-OFF / H4-K4 / linear`) materially increased `M_short` and fuser rank without making selection pass, so the current winner is `B-1c_linear_fuser`.
- RG-2 should use `CTX-OFF / H4-K4 / linear` as the control arm rather than repeating more RG-1 probes.
- Preserve Reader checkpoint compatibility while adding `conditioning_mode`: `none` must still be able to warm-start from checkpoints that contain `context_proj`.
- RG-2 control confirmed the RG-1 conclusion: the linear fuser already keeps `M_short` effectively full-rank, so the remaining blocker is Reader specialization rather than short-slot capacity.
- RG-2 competitive attention did not improve specialization or task behavior relative to the linear control.
- RG-2 masked partition forced clean Reader specialization but still failed the geometry-alive gate because dominant-label collapse happened too early, so the next step is RG-3 local bootstrap rather than more RG-2 variants.
- RG-3 bootstrap-only respected the temporary projector freeze and preserved the RG-2 partition geometry, but it did not delay dominant-label collapse or improve readout rank relative to the RG-2 control.
- RG-3 bootstrap+reconstruction also failed to delay collapse and slightly reduced the best readout / short-rank metrics, so the auxiliary did not rescue the local bridge.
- Because neither RG-3 arm produced geometry gain, `PLANv2.md` does not authorize RG-4; the correct repository state is to stop stacking local Reader losses and record this branch as `B-1_local_bootstrap_failed`.

## Surprises & Discoveries

- The repo already had partial two-level diagnostics wired through `_prefix_stats`, which makes RG-0 much cheaper than a fresh instrumentation pass.
- The move of `AGENTS.md` to repo root left stale hardcoded test references that were fixed before starting RG-0 work.
- The RG-1 context-overwrite metric stayed at `0.0` in the baseline and all three probes, so the live failure is not behaving like B-1a.
- The Reader attention pairwise cosine stayed pinned at `1.0` across RG-1, yet the linear fuser still raised final `M_short` effective rank from `1.2098` to `3.9854`, which isolates the healthiest near-term lever to the Fuser side.
- Under RG-2, the control and competitive arms both kept `M_short` rank near `3.98` while Reader pairwise cosine stayed at `~1.0`, so the bridge is no longer bottlenecked by short-slot collapse.
- The hard partition arm dropped Reader entropy to `0.6931` and pairwise cosine to `0.0` while preserving high `M_short` rank, which cleanly demonstrates that forced specialization works structurally but still collapses too early to pass G3.
- A stale duplicate RG-2 shell outside the intended run root had to be killed twice during execution; the published review bundle under `tl-reader-symmetry-break-fever-qwen25` is the authoritative RG-2 record.
- RG-3 confirmed that early local bootstrap pressure can preserve the partitioned Reader geometry while still failing at step-0 dominant-label collapse, which keeps the failure classified as a local bridge problem rather than a width-restoration problem.
- A duplicate RG-3 recovery invocation repeatedly reinitialized the arm-2 output directory until the analysis step was rerun under `flock`; the published review bundle under `tl-reader-local-bootstrap-fever-qwen25` is the authoritative RG-3 record.
