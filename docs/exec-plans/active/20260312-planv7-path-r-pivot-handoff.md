# PLANv7 Path R Pivot Handoff

## Purpose

Record the final `PLANv7` decision boundary after `V7-6` and hand off the repo in a state where the next executor can open the justified backbone-native / integrated-weaver pivot without re-reading the entire run history.

## Trigger

`PLANv7.md` section `17.4` says `Path R` is justified only if all of the following are true:

- the oracle gate is weak,
- wide Writer + bridge + forced consumption + reconstruction still fail,
- three-seed confirmation finds no real primary-task gain.

`V7-6` now satisfies that condition.

## Final Governed Outcome

- phase: `V7-6`
- summary:
  - `/root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json`
  - `/root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25/v7-6-summary.md`
- repo review mirror:
  - `results/generated/review/planv7-v7-6-multiseed-confirmation-qwen25/`
  - `runs/review/planv7-v7-6-multiseed-confirmation-qwen25/`
- decision:
  - `comparison_conclusion=path_r_architecture_pivot_required`
  - `recommended_next_step=prepare_backbone_native_writer_pivot`
  - `best_confirmed_variant_id=p1_a5_barlow`
  - `winner_uses_bridge=true`
  - `winning_depth=D1`

## Why Path R Was Reached

- `GSM8K` and `TriviaQA` showed no real three-seed score gain in either promoted branch.
- Strict Writer-memory improvement did not survive the confirmation gate.
- The bridge path remained route-live and stable, so the failure is not a simple dead-route issue.
- FEVER stayed calibration-flat and did not overrule the primary-task outcome.

## Repo State For The Next Phase

- Keep `PLANv7.md` as the completed authority for the external-Writer line.
- Treat `V7-6` as the terminal decision point for that line.
- Do not reopen another bridge-only or FEVER-only sweep under `PLANv7`.
- The next plan should define a backbone-native Writer / integrated-weaver pivot explicitly before more live training begins.

## Storage / Runtime Notes

- `/root/autodl-tmp` has been restored to a safe margin after preserving and deleting cold raw run trees.
- Runtime-only preservation manifests are under `/root/runtime-archives/memtotal/deletion-manifests/`.
- Runtime-only archives are under `/root/runtime-archives/memtotal/experimental-runtime/`.

## Immediate Next Human / Agent Task

Write the successor plan that specifies:

1. the backbone-native / integrated-weaver architecture to test,
2. the exact control surface preserved from `PLANv7`,
3. the first low-cost smoke and oracle gates for the pivot,
4. the publication and storage policy inherited from the current repo state.
