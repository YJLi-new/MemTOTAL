# PLANv7 (LR updated version) Path Q Handoff

## Purpose

Record the final decision boundary reached by `PLANv7 (LR updated version)` after `V7-6` and hand off the repo in a state where the next executor can open the narrower follow-on branch that `PLANv7.md` explicitly authorizes for `Path Q`.

## Trigger

`PLANv7.md` section `17.4` says `Path Q` is justified if:

- Writer metrics improve,
- but primary-task scores still fail to move consistently.

The LR-updated `V7-6` closeout now satisfies that condition.

## Final Governed Outcome

- phase: `V7-6`
- summary:
  - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json`
  - `/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.md`
- repo review mirror:
  - `results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/`
  - `runs/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/`
- decision:
  - `comparison_conclusion=path_q_external_writer_unresolved_not_dead`
  - `recommended_next_step=open_stronger_integrated_writer_or_true_highdim_branch`
  - `best_confirmed_variant_id=p1_a5_barlow`
  - `best_confirmed_promoted_arm_id=a5_barlow`
  - `winner_uses_bridge=true`
  - `winning_depth=D1`

## Why Path Q Was Reached

- The LR-updated replay did materially change the final closeout relative to the historical `PLANv7` line: the repo moved from `Path R` to `Path Q`.
- The best confirmed branch remained a bridge branch (`p1_a5_barlow`) at `D1 / mid4`, which means the route was not dismissed as a simple projector-only artifact.
- The current external-Writer line still did not produce consistent primary-task score movement on `GSM8K` and `TriviaQA` across the three-seed confirmation gate.
- Therefore the repo should not jump straight to the historical backbone-native / integrated-weaver pivot. The narrower next step is still external-Writer-adjacent.

## Repo State For The Next Phase

- Keep `PLANv7-LR-updated.md` as the completed authority for the LR-`7.5e-5` replay line.
- Treat `V7-6` as the terminal decision point for that replay line.
- Do not reopen another direct-injection / bridge-only / FEVER-only sweep under the same `PLANv7 (LR updated version)` namespace.
- The next plan should explicitly define one of the two branches authorized by `Path Q`:
  1. a stronger integrated Writer branch, or
  2. a true high-dimensional `M_long` extension branch.

## Relationship To Historical PLANv7

- Historical `PLANv7` without the LR update still closed at `Path R` and remains archived in:
  - [`20260312-planv7-path-r-pivot-handoff.md`](/root/mydir/MemTOTAL/docs/exec-plans/active/20260312-planv7-path-r-pivot-handoff.md)
- The repo should preserve both conclusions:
  - historical `PLANv7`: `Path R`
  - LR-updated replay: `Path Q`

## Immediate Next Human / Agent Task

Write the successor plan that specifies:

1. whether the repo will open the stronger integrated Writer branch or the true high-dimensional `M_long` branch first,
2. which controls from `PLANv7` are preserved unchanged,
3. the first low-cost smoke/oracle gates for that branch,
4. the publication and storage policy inherited from the current repo state.
