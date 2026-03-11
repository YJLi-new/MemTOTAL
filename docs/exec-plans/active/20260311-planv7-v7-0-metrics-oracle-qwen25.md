# PLANv7 Phase V7-0 - Metrics Repair, Continuity Replay, And Oracle Gate

## Purpose

Implement the `PLANv7.md` `V7-0` slice so the repo can repair its measurement layer, add TriviaQA to the writer-direct bundle, replay the continuity baselines under the new strict gates, and run the first oracle gate before any new bandwidth sweep opens.

## Context

- `PLANv7.md` is now the root execution authority.
- `PLANv7` explicitly requires `V7-0` to run before `V7-1`, `V7-2`, or any W2/W3 bandwidth experiment.
- The mandatory first-wave code delta is limited to:
  - TriviaQA support in `writer_jointpeft_data.py`
  - mid4 method presets
  - strict Writer-memory metrics and summary gates
  - oracle context/support echo paths
  - owner-LR discrepancy metadata logging
- The repo entry-point docs still contain stale `PLANv6` references, so governance has to be frozen first.

## Plan Of Work

1. Freeze governance so the repo itself points future work at `PLANv7`.
2. Audit the existing writer-direct bundle, training runtime, projector/injection stack, and summary pipeline for the exact `V7-0` gaps.
3. Implement only the `V7-0` code/config/script delta.
4. Add targeted tests for the new bundle contract, strict gates, mid4 presets, and oracle modes.
5. Validate the new harness and prepare the governed `V7-0` runner/summary entrypoints.

## Concrete Steps

1. Update [`AGENTS.md`](/root/mydir/MemTOTAL/AGENTS.md), [`README.md`](/root/mydir/MemTOTAL/README.md), and related guidance so `PLANv7.md` is the advertised authority.
2. Audit:
   - [`src/memtotal/tasks/writer_jointpeft_data.py`](/root/mydir/MemTOTAL/src/memtotal/tasks/writer_jointpeft_data.py)
   - [`src/memtotal/tasks/registry.py`](/root/mydir/MemTOTAL/src/memtotal/tasks/registry.py)
   - [`src/memtotal/tasks/sources.py`](/root/mydir/MemTOTAL/src/memtotal/tasks/sources.py)
   - [`src/memtotal/models/memory.py`](/root/mydir/MemTOTAL/src/memtotal/models/memory.py)
   - [`src/memtotal/models/backbone.py`](/root/mydir/MemTOTAL/src/memtotal/models/backbone.py)
   - [`src/memtotal/training/m4_shared_injection.py`](/root/mydir/MemTOTAL/src/memtotal/training/m4_shared_injection.py)
   - existing writer summary scripts and `PLANv6` runner patterns.
3. Implement the dataset-bundle generalization:
   - benchmark list CLI support,
   - split-plan JSON support,
   - first-class TriviaQA materialization,
   - deterministic split preservation.
4. Implement the runtime/config layer needed for `V7-0`:
   - mid4 method presets,
   - additive7 continuity preset,
   - oracle prefix-source modes,
   - owner-LR discrepancy metadata,
   - strict Writer-memory gates and diagnostics.
5. Add `V7-0` runner and summary entrypoints.
6. Validate with targeted unit tests, static checks, and dry-run or small-run harness validation if feasible within the turn budget.

## Validation And Acceptance

`V7-0` preparation is complete only if all of the following are true:

- repo governance points future work at `PLANv7.md`,
- TriviaQA can be materialized through the writer-direct bundle interface,
- mid4 presets exist and are usable,
- strict Writer-memory gates and metadata fields are emitted by the new summary path,
- oracle context/support echo modes exist in configuration and runtime plumbing,
- targeted tests for the new contracts pass.

## Progress

- `2026-03-11 00:00 UTC`: `PLANv7.md` downloaded to the repo root from the GitHub `review` branch.
- `2026-03-11 00:02 UTC`: confirmed that `PLANv7` requires `V7-0` as the mandatory first slice.
- `2026-03-11 00:05 UTC`: began governance freeze and code-gap audit for the `V7-0` delta.
- `2026-03-11 01:05 UTC`: completed the first implementation wave:
  - repo-map updates now advertise `PLANv7.md`,
  - `writer_jointpeft_data.py` now defaults to `gsm8k + triviaqa + fever`, supports benchmark selection and split-plan overrides, and persists split plans in the manifest,
  - mid4 and additive7 method presets were added,
  - `SharedInjectionPilotRuntime` now supports `oracle_context_echo` and `oracle_support_echo`,
  - V7 owner-LR discrepancy metadata is emitted into runtime metrics,
  - the governed `V7-0` runner and summary entrypoints now exist.
- `2026-03-11 01:12 UTC`: validation passed:
  - `python -m py_compile src/memtotal/tasks/writer_jointpeft_data.py src/memtotal/training/m4_shared_injection.py scripts/update_planv7_v7_0_metrics_oracle_summary.py`
  - `bash -n scripts/run_planv7_v7_0_metrics_oracle_qwen25.sh scripts/publish_review_artifacts.sh`
  - `python -m unittest discover -s tests -v` (`278` tests, `OK`)
- `2026-03-11 01:15 UTC`: no active `tmux` jobs or GPU compute processes remained from prior phases, so the full governed `V7-0` matrix became safe to launch.
- `2026-03-11 01:16 UTC`: launched the full governed `V7-0` matrix in detached `tmux` session `planv7_v70` with roots:
  - `/root/autodl-tmp/runs/verify/planv7-v7-0-metrics-oracle-qwen25`
  - `/root/autodl-tmp/results/generated/planv7-v7-0-metrics-oracle-qwen25`
- `2026-03-11 01:18 UTC`: confirmed the detached phase is alive and still in the initial bundle materialization step (`python -m memtotal.tasks.writer_jointpeft_data ... --benchmarks gsm8k,triviaqa,fever`).

## Decision Log

- Keep the first implementation wave strictly bounded to `V7-0`; do not start `V7-1` scouting or any W2/W3 bandwidth branch before the `V7-0` bundle publishes.
- Preserve `V6-5` as a stabilization reference, but do not let FEVER-only wins redefine the main architecture anchor.
- Update the repo map now so future agents do not accidentally continue from `PLANv6` out of stale docs.

## Surprises And Discoveries

- The repo entry-point docs were only partially updated: `AGENTS.md` mostly pointed to `PLANv7`, but [`README.md`](/root/mydir/MemTOTAL/README.md), [`docs/GITHUB_REVIEW_EXPORT.md`](/root/mydir/MemTOTAL/docs/GITHUB_REVIEW_EXPORT.md), and the lint expectation still advertised `PLANv6`.
- [`src/memtotal/tasks/writer_jointpeft_data.py`](/root/mydir/MemTOTAL/src/memtotal/tasks/writer_jointpeft_data.py) is still hardcoded to `gsm8k`, `narrativeqa`, and `fever`, so TriviaQA bundle support is a real `V7-0` code delta rather than a pure config change.
- The old permissive collapse gate and the new strict Writer-memory gate can disagree strongly on the same branch, so `V7-0` needs both rendered side-by-side in the governed summary rather than replacing the old field silently.
- Oracle prefix echo can be implemented cleanly with the existing backbone hidden-state slice helpers; the missing piece was runtime routing, not model capability.
