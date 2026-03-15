# 20260309 Independent Writer Deep-Prefix Circuit

## Status
- Historical route-opening record only.
- The live continuation of `PLANv5.md` is now tracked in `docs/exec-plans/active/20260309-planv5-addendum-writer-direct-validation.md`.
- The old `stop_after_p2a` conclusion below is preserved for provenance, but it is no longer the governing decision rule for the project.

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

## Validation & Acceptance
- Code-path validation:
  - `python -m unittest tests.test_m4_shared_injection tests.test_backbone_real_mode -v`
  - `python -m py_compile src/memtotal/models/backbone.py src/memtotal/models/memory.py src/memtotal/training/m4_shared_injection.py scripts/update_writer_circuit_opening_summary.py`
  - `bash -n scripts/run_writer_circuit_opening_qwen25.sh`
  - `bash -n scripts/publish_review_artifacts.sh`
- Experiment entry point:
  - `./scripts/run_writer_circuit_opening_qwen25.sh`
- Published review artifacts:
  - `results/generated/review/writer-circuit-opening-qwen25/`
  - `runs/review/writer-circuit-opening-qwen25/`

## Progress
- 2026-03-09: Created from `PLANv5.md`.
- 2026-03-09: Implemented the `PLANv5` P0 scaffold:
  - added `SourceStubMemory`;
  - moved the deep-prefix projector into `src/memtotal/models/memory.py` with `kv_stat_match`, `semantic_anchor`, and `hidden_state_anchor` init modes;
  - added backbone calibration helpers for prompt/hidden/KV statistics;
  - enabled `writer_direct + sparse_deep_prefix + optional receiver micro-LoRA`;
  - added source-stub optimizer plumbing, circuit metrics, and prefix-routing diagnostics;
  - added PLANv5 configs, runner, summary script, and contract tests.
- 2026-03-09: First live P1a/P2a attempt failed at runtime because the Qwen HF path was still using `sdpa`, which does not return attentions for the new diagnostics. Patched `BackboneWrapper.score_continuations(...)` to temporarily force eager attention only for diagnostic forwards, then added a regression test in `tests/test_backbone_real_mode.py`.
- 2026-03-09: Discovered a publication gap: `scripts/publish_review_artifacts.sh` did not mirror the new writer-circuit-opening bundle into repo review directories. Added sync rules for both `runs/review/writer-circuit-opening-qwen25` and `results/generated/review/writer-circuit-opening-qwen25`, then re-ran the publisher after the suite completed.
- 2026-03-09: Completed the canonical opening bundle:
  - `P1a`: source-stub, deep prefix, early4, no receiver LoRA;
  - `P2a`: source-stub, deep prefix, early4 receiver micro-LoRA (`k_proj/v_proj`, rank 2, alpha 4);
  - tasks: GSM8K, NarrativeQA, FEVER-label-gen.
- 2026-03-09: Final gate result is terminal for this branch. `results/generated/review/writer-circuit-opening-qwen25/writer-circuit-opening-summary.json` records:
  - `comparison_conclusion=stop_after_p2a`
  - `primary_interpretation=no_nonfever_circuit_open`
  - `move_to_p1b=false`
  - `move_to_p2b=false`
  - `stop_after_p2a=true`
- 2026-03-09: Non-FEVER interpretation:
  - GSM8K `P1a`: source grad median `0.24198`, nontrivial prefix-attention layers `2`, but `delta_answer_logprob=0.0`.
  - GSM8K `P2a`: source grad median `0.53019`, receiver-LoRA grad median `36.09721`, nontrivial prefix-attention layers `4`, but `delta_answer_logprob=0.0`.
  - NarrativeQA `P1a`: source grad median `0.39723`, but no layer crossed the nontrivial prefix-attention threshold and `delta_answer_logprob=0.0`.
  - NarrativeQA `P2a`: source grad median `0.49700`, receiver-LoRA grad median `134.02077`, but no layer crossed the nontrivial prefix-attention threshold and `delta_answer_logprob=0.0`.
- 2026-03-09: FEVER stayed flat rather than rescuing the branch:
  - control accuracy `0.5`
  - `P1a` accuracy `0.5`
  - `P2a` accuracy `0.5`
  - both FEVER arms also had `delta_answer_logprob=0.0`.
- 2026-03-09: Corrected interpretation recorded in the new addendum exec plan:
  - the `source_stub` branch is retained as a route-liveness probe;
  - `delta_answer_logprob > 0` is no longer treated as a valid usefulness gate for a static prefix source;
  - the next live branch is the stable `writer_direct + sparse_deep_prefix + early4 receiver micro-LoRA` validation recipe.

## Decision Log
- The branch did show mechanical liveness:
  - source gradients were nontrivial in both P1a and P2a on all three tasks;
  - receiver-LoRA gradients were strongly nonzero in P2a;
  - GSM8K and FEVER showed nontrivial prefix-attention mass on selected layers.
- That was still insufficient for the `PLANv5` gate because the required usefulness signal never appeared on any non-FEVER task: `delta_answer_logprob` stayed exactly `0.0` for GSM8K and NarrativeQA in both P1a and P2a.
- Historical note: the original branch write-up treated the zero-mean `delta_answer_logprob` result as terminal. That specific interpretation is now revoked because `source_stub` is static and was never a valid usefulness substrate.
- The next authorized move is defined by the addendum exec plan, not by the old `stop_after_p2a` line here.

## Surprises & Discoveries
- The new diagnostics changed the backbone runtime contract: when prefix-attention auditing is requested, the HF Qwen path must provide attentions, which requires eager attention rather than `sdpa`.
- Even under stronger bridge physics than `PLANv4`, the opening runs still separated into:
  - clearly live gradients;
  - some measurable prefix routing;
  - and zero downstream usefulness movement on the non-FEVER tasks.
- That makes the failure mode sharper than “no gradient”: the route is mechanically open enough to train, but still not open enough to alter answer preference in a useful way.
