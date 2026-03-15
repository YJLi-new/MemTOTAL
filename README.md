# MemTOTAL

MemTOTAL studies an internal-memory route for frozen reasoning LLMs: write a high-capacity latent memory `M_long`, read and compress it into `M_short`, and inject only the short memory into the next reasoning context.

The repository is now governed by the qwen34 `PLANv8` line. The latest completed decision is the qwen34 `V8-7` comparator gate, and the current governed outcome is a hold before `V8-8`.

## Current Status

- Active governed line: `PLANv8` on `Qwen3-4B`
- Latest completed milestone: `V8-7 comparators`
- `comparison_conclusion = comparators_do_not_support_v8_8`
- `recommended_next_step = hold_v8_8_comparator_review`

What that means:

- qwen34 `V8-0` through `V8-3` completed end-to-end
- the best qwen34 route entering the comparator gate came from `V8-3`, arm `p5_opd_ansplusctx_centered`
- that route was strong enough to justify a direct `V8-3 -> V8-7` comparator check
- the comparator gate did not clear the bar required to open qwen34 `V8-8` multi-seed confirmation

The current qwen34 line is therefore paused by governed evidence, not by an infrastructure failure.

## Latest Qwen34 Readout

Latest qwen34 milestones:

- `V8-0 baselines + oracles`
  - [v8-0-summary.json](results/generated/review/planv8-v8-0-qwen34-baselines-oracles/v8-0-summary.json)
- `V8-1 reader interface scout`
  - [v8-1-summary.json](results/generated/review/planv8-v8-1-reader-interface-scout-qwen34/v8-1-summary.json)
- `V8-2 reader sweep`
  - [v8-2-summary.json](results/generated/review/planv8-v8-2-reader-sweep-qwen34/v8-2-summary.json)
- `V8-3 reader OPD`
  - [v8-3-summary.json](results/generated/review/planv8-v8-3-reader-opd-qwen34/v8-3-summary.json)
  - [v8-3-summary.md](results/generated/review/planv8-v8-3-reader-opd-qwen34/v8-3-summary.md)
- `V8-7 comparators`
  - [v8-7-summary.json](results/generated/review/planv8-v8-7-comparators-qwen34/v8-7-summary.json)
  - [v8-7-summary.md](results/generated/review/planv8-v8-7-comparators-qwen34/v8-7-summary.md)

Important qwen34 relay docs:

- [PLANv8.md](PLANv8.md)
- [20260314-planv8-qwen34-restart.md](docs/exec-plans/active/20260314-planv8-qwen34-restart.md)
- [20260315-planv8-v8-3-reader-opd-qwen34.md](docs/exec-plans/active/20260315-planv8-v8-3-reader-opd-qwen34.md)
- [20260315-planv8-v8-7-comparators-qwen34.md](docs/exec-plans/active/20260315-planv8-v8-7-comparators-qwen34.md)
- [20260315-planv8-v8-7-v8-8-repair-qwen34-v83-route.md](docs/exec-plans/active/20260315-planv8-v8-7-v8-8-repair-qwen34-v83-route.md)

## Repo-Level Interpretation

The most recent repo conclusion is narrower and more concrete than the older `PLANv7` closeout:

- qwen34 Reader-side work produced a real promoted route through `V8-3`
- the route did not beat the comparator bar strongly enough to justify `V8-8`
- the repository therefore preserves the qwen34 `PLANv8` artifacts as a governed hold, not a claimed win

Historical qwen25 work is still important context, especially the LR-updated `PLANv7` closeout:

- [PLANv7-LR-updated.md](PLANv7-LR-updated.md)
- [v7-6-summary.json](results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json)

## Where To Read First

- [PLANv8.md](PLANv8.md): current qwen34 execution authority
- [docs/MAIN_IDEA.md](docs/MAIN_IDEA.md): core method definition
- [docs/EXPERIMENTS_INFO.md](docs/EXPERIMENTS_INFO.md): reporting and experiment protocol
- [docs/exec-plans/active](docs/exec-plans/active): milestone-by-milestone execution notes
- [AGENTS.md](AGENTS.md): repo map and local operating rules

## Repository Layout

- `src/memtotal`: model, training, task, and baseline code
- `scripts`: governed runners, summary builders, publishers, and queue helpers
- `configs`: experiment, method, and task configuration surfaces
- `results/generated/review`: canonical published review artifacts
- `runs/review`: canonical run-side review artifacts
- `tests`: focused milestone logic tests plus repo governance checks

## Minimal Validation

Useful local checks:

```bash
python -m unittest tests.test_repo_lints tests.test_repo_contract -v
python -m unittest tests.test_planv8_v8_7_summary tests.test_planv8_v8_8_summary -v
```

For shell-script changes, also run:

```bash
bash -n scripts/push_github_review_snapshot.sh
```

## Review Branch Policy

GitHub's lightweight downloadable branch remains `review`. It is maintained as the external review surface while `main` carries the full governed code and milestone history.
