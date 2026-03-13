# MemTOTAL

MemTOTAL studies an internal-memory route for frozen reasoning LLMs: write a high-capacity latent memory `M_long`, read and compress it into `M_short`, and inject only the short memory into the next reasoning context.

This repository is now at the `PLANv7 (LR updated version)` decision closeout. The external-Writer bridge line was rerun end-to-end with projector LR `7.5e-5` through oracle checks, width/depth scouting, bandwidth expansion, bounded query compression, forced-consumption reopening, targeted auxiliary revisit, and final three-seed confirmation.

## Current Status

- `PLANv7 (LR updated version)` is complete.
- Latest governed decision: `Path Q`.
- `comparison_conclusion = path_q_external_writer_unresolved_not_dead`
- `recommended_next_step = open_stronger_integrated_writer_or_true_highdim_branch`

The current repo conclusion is not “the route is dead,” and it is no longer the older `Path R` closeout. Under the LR-updated restart, the bridge line remained live enough that the justified next branch is still external-Writer-adjacent: either a stronger integrated Writer or a true high-dimensional `M_long` extension. The current line still failed to produce consistent three-seed primary-task movement on `GSM8K` or `TriviaQA`, so it is unresolved rather than vindicated.

## Latest LR-Updated Report

- `PLANv7 (LR updated version)` runbook: [PLANv7-LR-updated.md](PLANv7-LR-updated.md)
- inherited backbone runbook: [PLANv7.md](PLANv7.md)
- Final multi-seed confirmation report:
  - [v7-6-summary.md](results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.md)
  - [v7-6-summary.json](results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json)
- LR-updated Path Q handoff:
  - [20260313-planv7-lr75e5-path-q-handoff.md](docs/exec-plans/active/20260313-planv7-lr75e5-path-q-handoff.md)
- Full LR-updated restart relay:
  - [20260312-planv7-lr75e5-restart.md](docs/exec-plans/active/20260312-planv7-lr75e5-restart.md)
  - [20260312-planv7-lr75e5-v7-6-multiseed-confirmation.md](docs/exec-plans/active/20260312-planv7-lr75e5-v7-6-multiseed-confirmation.md)

## Latest Readout From V7-6

- Best confirmed branch: `p1_a5_barlow`
- Best promoted arm: `a5_barlow`
- Winner uses bridge: `true`
- Winning depth: `D1` / `mid4`
- Seeds: `61109, 61110, 61111`

Primary-task outcome:

- `GSM8K`: no real gain across the confirmed branch set
- `TriviaQA`: no real gain across the confirmed branch set
- `FEVER`: calibration-flat and not allowed to overrule the primary decision

Key evidence from the governed summary:

- `oracle_gate_weak = true`
- `any_real_primary_gain_across_three_seeds = false`
- `any_strict_writer_metric_improvement_across_three_seeds = false`
- `best_branch_not_projector_only = true`

Interpretation:

- the current bridge route is real,
- the route is not merely a projector-only illusion,
- but the current external-Writer line is still unresolved on primary tasks,
- so the next justified research step is a stronger integrated Writer branch or a true high-dimensional `M_long` branch.

## Historical Contrast

Historical `PLANv7` without the LR update is preserved and still important:

- historical final report:
  - [v7-6-summary.md](results/generated/review/planv7-v7-6-multiseed-confirmation-qwen25/v7-6-summary.md)
  - [v7-6-summary.json](results/generated/review/planv7-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json)
- historical closeout:
  - [20260312-planv7-path-r-pivot-handoff.md](docs/exec-plans/active/20260312-planv7-path-r-pivot-handoff.md)

The important repo-level fact is that the LR-updated rerun changed the final conclusion from historical `Path R` to current `Path Q`.

## Most Useful Supporting LR-Updated Reports

- `V7-5 targeted auxiliary revisit`:
  - [v7-5-summary.md](results/generated/review/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.md)
  - [v7-5-summary.json](results/generated/review/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json)
- `V7-4 forced consumption`:
  - [v7-4-summary.md](results/generated/review/planv7-lr75e5-v7-4-forced-consumption-qwen25/v7-4-summary.md)
  - [v7-4-summary.json](results/generated/review/planv7-lr75e5-v7-4-forced-consumption-qwen25/v7-4-summary.json)
- `V7-3 bridge`:
  - [v7-3-summary.md](results/generated/review/planv7-lr75e5-v7-3-bridge-qwen25/v7-3-summary.md)
  - [v7-3-summary.json](results/generated/review/planv7-lr75e5-v7-3-bridge-qwen25/v7-3-summary.json)
- `V7-0 metrics + oracle`:
  - [v7-0-summary.md](results/generated/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25/v7-0-summary.md)
  - [v7-0-summary.json](results/generated/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25/v7-0-summary.json)

## What To Read First

- [PLANv7-LR-updated.md](PLANv7-LR-updated.md): the completed restart overlay and current closeout entry point
- [PLANv7.md](PLANv7.md): the completed backbone execution authority for the external-Writer line
- [docs/MAIN_IDEA.md](docs/MAIN_IDEA.md): method definition and paper-story constraints
- [docs/EXPERIMENTS_INFO.md](docs/EXPERIMENTS_INFO.md): experiment protocol and reporting rules
- [AGENTS.md](AGENTS.md): repo entry-point map for agents

## Review-Branch Policy

GitHub’s default downloadable branch is the lightweight `review` branch. It is maintained for external review and lightweight reproduction, and it stays under the repository zip-size budget while keeping the latest governed reports and the minimal code/docs surface.
