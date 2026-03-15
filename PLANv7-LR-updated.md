# PLANv7 (LR updated version)

**Version:** 2026-03-12  
**Status:** completed restart of the `PLANv7` program with a single intentional optimizer override  
**Backbone authority:** [`PLANv7.md`](PLANv7.md) remains the full phase-definition backbone unless this document explicitly overrides it

## 0. Purpose

Restart the `PLANv7` execution sequence with the projector learning rate set to `7.5e-5` instead of `7.5e-6`, because the prior `PLANv7` run exhausted its own decision logic under the low-LR override and the owner has explicitly requested a fresh try with the higher projector LR.

This document is not a new scientific plan. It is a governed restart overlay on top of `PLANv7.md`.

## 1. Exact Overrides Relative To PLANv7

For all restarted `V7-*` phases in this line:

- `pilot_projector_learning_rate = 7.5e-5`
- `owner_locked_projector_lr = 7.5e-5`
- `repo_confirmed_v65_projector_lr_reference = 7.5e-5`
- `owner_override_note = false`

Everything else inherits from `PLANv7.md` unless explicitly changed in a later restart-specific doc or exec plan.

## 2. Namespace And Preservation Rules

Do not overwrite the historical `PLANv7` bundles. The restart must publish to its own namespace.

Use these prefixes:

- run roots: `planv7-lr75e5-v7-*`
- result roots: `planv7-lr75e5-v7-*`
- repo review mirrors:
  - `runs/review/planv7-lr75e5-v7-*`
  - `results/generated/review/planv7-lr75e5-v7-*`

The original `PLANv7` bundles remain historical and must stay intact.

## 3. Phase Order

The phase order is unchanged and remains inherited from `PLANv7.md`:

1. `V7-0`
2. `V7-1`
3. `V7-2`
4. `V7-3`
5. `V7-4`
6. `V7-5`
7. `V7-6`

Do not skip ahead. The restart must re-earn each gate in order.

## 4. Current State

The restart is now complete through `V7-6`.

Final governed outcome:

- `comparison_conclusion = path_q_external_writer_unresolved_not_dead`
- `recommended_next_step = open_stronger_integrated_writer_or_true_highdim_branch`
- `best_confirmed_variant_id = p1_a5_barlow`
- `winner_uses_bridge = true`
- `winning_depth = D1`

Primary closeout artifacts:

- [`20260312-planv7-lr75e5-restart.md`](docs/exec-plans/active/20260312-planv7-lr75e5-restart.md)
- [`20260312-planv7-lr75e5-v7-6-multiseed-confirmation.md`](docs/exec-plans/active/20260312-planv7-lr75e5-v7-6-multiseed-confirmation.md)
- [`20260313-planv7-lr75e5-path-q-handoff.md`](docs/exec-plans/active/20260313-planv7-lr75e5-path-q-handoff.md)

No further `V7-*` phases are authorized under this restart overlay. The next live work requires a successor plan that explicitly opens the stronger integrated Writer branch or the true high-dimensional `M_long` branch.

## 5. Decision Rules

Acceptance criteria, milestone gates, and `Path P / Q / R` logic are inherited unchanged from `PLANv7.md`.

This restart exists to answer one narrower question:

> Does rerunning the same governed `PLANv7` sequence with projector LR `7.5e-5` materially change the primary-task outcome?

Answer from the completed restart:

> Yes, materially enough to move the final decision from historical `Path R` to `Path Q`, but not enough to keep the current external-Writer bridge line as a solved main thesis.

## 6. Documentation Rule

When reporting this restart, refer to it as:

> `PLANv7 (LR updated version)`
