# MemTOTAL

MemTOTAL studies internal memory routes for frozen reasoning LLMs and agents: write a high-capacity latent memory `M_long`, read and compress it into `M_short`, and inject only the short actionable memory into the next reasoning context.

The repository is now governed by `PLANv9.md`. `PLANv8` remains the frozen predecessor evidence base, and the first active `PLANv9` milestone is `V9-0`, the qwen34 FlashMem discrepancy discrimination test.

## Current Status

- Active governed line: `PLANv9` on `Qwen3-4B`
- Latest completed predecessor decision: qwen34 `V8-7 comparators`
- Frozen predecessor conclusion: `comparison_conclusion = comparators_do_not_support_v8_8`
- Current active milestone: `V9-0 flashmem discrimination`

What that means:

- qwen34 `V8-0` through `V8-3` completed end-to-end
- the best qwen34 route entering the comparator gate came from `V8-3`, arm `p5_opd_ansplusctx_centered`
- that route was strong enough to justify a direct `V8-3 -> V8-7` comparator check
- the comparator gate did not clear the bar required to open qwen34 `V8-8` multi-seed confirmation
- `PLANv9` treats that hold as a real scientific decision, not as an excuse to keep sweeping single-turn latent-memory variants
- the immediate governed next step is the qwen34 `V9-0` four-arm discriminator that tests whether FlashMem-style cache-prefill is actually safer than the destructive in-sequence route on this stack

The current repo therefore keeps the qwen34 `PLANv8` results as preserved evidence while shifting active execution authority to `PLANv9`.

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

Current governing docs:

- [PLANv9.md](PLANv9.md)
- [PLANv8.md](PLANv8.md)
- [20260315-planv9-v9-0-flashmem-discrimination-qwen34.md](docs/exec-plans/active/20260315-planv9-v9-0-flashmem-discrimination-qwen34.md)
- [20260314-planv8-qwen34-restart.md](docs/exec-plans/active/20260314-planv8-qwen34-restart.md)
- [20260315-planv8-v8-3-reader-opd-qwen34.md](docs/exec-plans/active/20260315-planv8-v8-3-reader-opd-qwen34.md)
- [20260315-planv8-v8-7-comparators-qwen34.md](docs/exec-plans/active/20260315-planv8-v8-7-comparators-qwen34.md)
- [20260315-planv8-v8-7-v8-8-repair-qwen34-v83-route.md](docs/exec-plans/active/20260315-planv8-v8-7-v8-8-repair-qwen34-v83-route.md)

## Repo-Level Interpretation

The most recent repo conclusion is narrower and more concrete than the older `PLANv7` closeout:

- qwen34 Reader-side work produced a real promoted route through `V8-3`
- the route did not beat the comparator bar strongly enough to justify `V8-8`
- the repository therefore preserves the qwen34 `PLANv8` artifacts as a governed hold, not a claimed win
- `PLANv9` now reopens the project only through a narrower contradiction test: `V9-0` must determine whether FlashMem-style cache-prefill is non-destructive before the repo spends more budget on long-horizon memory architecture work

Historical qwen25 work is still important context, especially the LR-updated `PLANv7` closeout:

- [PLANv7-LR-updated.md](PLANv7-LR-updated.md)
- [v7-6-summary.json](results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json)

## Where To Read First

- [PLANv9.md](PLANv9.md): current execution authority
- [PLANv8.md](PLANv8.md): frozen predecessor evidence line
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
python -m unittest tests.test_planv9_v9_0_config tests.test_planv9_v9_0_summary -v
```

For shell-script changes, also run:

```bash
bash -n scripts/push_github_review_snapshot.sh
```

## Review Branch Policy

GitHub's lightweight downloadable branch remains `review`. It is maintained as the external review surface while `main` carries the full governed code and milestone history.
