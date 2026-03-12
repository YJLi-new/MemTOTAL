# MemTOTAL

MemTOTAL studies an internal-memory route for frozen reasoning LLMs: write a high-capacity latent memory `M_long`, read and compress it into `M_short`, and inject only the short memory into the next reasoning context.

This repository is now at the `PLANv7` decision closeout. The external-Writer bridge line was tested through oracle checks, width/depth scouting, bandwidth expansion, bounded query compression, forced-consumption reopening, targeted auxiliary revisit, and final three-seed confirmation.

## Current Status

- `PLANv7` is complete.
- Final governed decision: `Path R`.
- `comparison_conclusion = path_r_architecture_pivot_required`
- `recommended_next_step = prepare_backbone_native_writer_pivot`

The current repo conclusion is not “the route is dead.” The bridge path remained route-live and stable. The problem is that after the full `PLANv7` sequence, it still produced no real three-seed primary-task gain on `GSM8K` or `TriviaQA`, and strict Writer-memory improvement did not survive confirmation.

## Final PLANv7 Report

- `PLANv7` runbook: [PLANv7.md](PLANv7.md)
- Final multi-seed confirmation report:
  - [v7-6-summary.md](results/generated/review/planv7-v7-6-multiseed-confirmation-qwen25/v7-6-summary.md)
  - [v7-6-summary.json](results/generated/review/planv7-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json)
- Pivot handoff:
  - [20260312-planv7-path-r-pivot-handoff.md](docs/exec-plans/active/20260312-planv7-path-r-pivot-handoff.md)
- Full `V7-6` execution log:
  - [20260311-planv7-v7-6-multiseed-confirmation-qwen25.md](docs/exec-plans/active/20260311-planv7-v7-6-multiseed-confirmation-qwen25.md)

## Final Readout From V7-6

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
- but the external-Writer line is no longer the active mainline after `PLANv7`,
- so the next justified research step is a backbone-native / integrated-weaver pivot.

## Most Useful Supporting Reports

- `V7-5 targeted auxiliary revisit`:
  - [v7-5-summary.md](results/generated/review/planv7-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.md)
  - [v7-5-summary.json](results/generated/review/planv7-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json)
- `V7-4 forced consumption`:
  - [v7-4-summary.md](results/generated/review/planv7-v7-4-forced-consumption-qwen25/v7-4-summary.md)
  - [v7-4-summary.json](results/generated/review/planv7-v7-4-forced-consumption-qwen25/v7-4-summary.json)
- `V7-3 bridge`:
  - [v7-3-summary.md](results/generated/review/planv7-v7-3-bridge-qwen25/v7-3-summary.md)
  - [v7-3-summary.json](results/generated/review/planv7-v7-3-bridge-qwen25/v7-3-summary.json)
- `V7-0 metrics + oracle`:
  - [v7-0-summary.md](results/generated/review/planv7-v7-0-metrics-oracle-qwen25/v7-0-summary.md)
  - [v7-0-summary.json](results/generated/review/planv7-v7-0-metrics-oracle-qwen25/v7-0-summary.json)

## What To Read First

- [PLANv7.md](PLANv7.md): the completed execution authority for the external-Writer line
- [docs/MAIN_IDEA.md](docs/MAIN_IDEA.md): method definition and paper-story constraints
- [docs/EXPERIMENTS_INFO.md](docs/EXPERIMENTS_INFO.md): experiment protocol and reporting rules
- [AGENTS.md](AGENTS.md): repo entry-point map for agents

## Review-Branch Policy

GitHub’s default downloadable branch is the lightweight `review` branch. It is maintained for external review and lightweight reproduction, and it stays under the repository zip-size budget while keeping the latest governed reports and the minimal code/docs surface.
