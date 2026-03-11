# MemTOTAL

MemTOTAL studies an internal-memory route for frozen reasoning LLMs: write a high-capacity latent memory `M_long`, read and compress it into a short memory `M_short`, and inject only the short memory into the next reasoning context.

This repository is in active research mode. The current authoritative runbook is [PLANv7.md](PLANv7.md).

## Current Status

As of March 11, 2026:

- The active program is `PLANv7`, focused on external-Writer bandwidth expansion, pure mid-layer injection, projector co-scaling, and bounded query compression.
- The mandatory first governed slice is `V7-0`: measurement repair, continuity replay, and the oracle gate.
- `GSM8K` and `TriviaQA` are now the primary decision pair. `FEVER` is retained only as a calibration guardrail.
- The preserved architecture anchor is `S3 + C2 + L5`.
- The preserved stabilization defaults come from the completed `V6-5` branch, but `PLANv7` does not let FEVER-only wins define the mainline architecture.

## Latest Useful Reports

 - `PLANv7 runbook`:
  [plan](PLANv7.md)
  - Outcome: current root execution authority
  - Readout: `V7-0` must run first, with TriviaQA bundle support, mid4 presets, strict Writer-memory gates, oracle context/support echo paths, and continuity replay before any new W2/W3 bandwidth sweep.

- `V6-5 recipe stabilization`:
  [summary](results/generated/review/planv6-v6-5-recipe-stabilization-qwen25/v6-5-summary.md),
  [json](results/generated/review/planv6-v6-5-recipe-stabilization-qwen25/v6-5-summary.json)
  - Outcome: `comparison_conclusion=select_stabilized_recipe`
  - Readout: the best stabilized FEVER recipe was `F3__w0__clip_groupwise__plr75e6__acc4__layers_additive`, with `F1__w0__clip_global__plr5e5__acc1__layers_additive` as the runner-up; both passed all `3/3` confirmation seeds and now serve as stabilization references rather than architecture winners.

- `V6-4 mixed matrix`:
  [summary](results/generated/review/planv6-v6-4-mixed-matrix-qwen25/v6-4-summary.md),
  [json](results/generated/review/planv6-v6-4-mixed-matrix-qwen25/v6-4-summary.json)
  - Outcome: `comparison_conclusion=select_finalists`
  - Readout: `S3 + C2 + L5` became the `PLANv7` architecture anchor.

- `V6-3 anti-homogenization screening`:
  [summary](results/generated/review/planv6-v6-3-loss-screening-qwen25/v6-3-summary.md),
  [json](results/generated/review/planv6-v6-3-loss-screening-qwen25/v6-3-summary.json)
  - Outcome: `comparison_conclusion=select_top_auxiliary_families`
  - Readout: the best auxiliary families were `l2_contrastive`, `l5_orthogonality_coverage`, and `l3_vicreg`.

- `V6-2 support screening`:
  [summary](results/generated/review/planv6-v6-2-support-screening-qwen25/v6-2-summary.md),
  [json](results/generated/review/planv6-v6-2-support-screening-qwen25/v6-2-summary.json)
  - Outcome: `comparison_conclusion=select_top_two_support_modes`
  - Readout: the best support interfaces were `s3_multi_item_cross_attn_raw` and `s5_hybrid_pooled_plus_items`.

- `V6-5 execution log`:
  [exec plan](docs/exec-plans/active/20260310-planv6-v6-5-recipe-stabilization.md)
  - Scope: FEVER-first recipe stabilization across warmup, clipping, projector LR, gradient accumulation, and additive layer expansion.
  - Final status: completed with `recommended_next_step=open_v6_7_reader_reopening`.

- `Historical route reset before PLANv6`:
  [writer-direct summary](results/generated/review/writer-deep-prefix-jointpeft-qwen25/writer-deep-prefix-jointpeft-summary.md),
  [json](results/generated/review/writer-deep-prefix-jointpeft-qwen25/writer-deep-prefix-jointpeft-summary.json),
  [PLANv5 addendum summary](results/generated/review/writer-circuit-opening-qwen25/writer-circuit-opening-addendum-summary.json)
  - Why it matters: this is the route-correction bundle that separated route-liveness from usefulness and directly motivated `PLANv6` and the stricter `PLANv7` continuation.

## What To Read First

- [PLANv7.md](PLANv7.md): current execution order, acceptance criteria, and milestone gating.
- [docs/MAIN_IDEA.md](docs/MAIN_IDEA.md): method definition, Writer/Reader/Fuser roles, and paper-story constraints.
- [docs/EXPERIMENTS_INFO.md](docs/EXPERIMENTS_INFO.md): experiment protocol, baselines, artifact rules, and paper-facing reporting standards.
- [AGENTS.md](AGENTS.md): repo entry-point map for coding agents.

## Repo Map

- `src/`: core models, training runtime, task adapters, and analysis logic.
- `scripts/`: single-command runners, summary builders, and artifact publishers.
- `docs/exec-plans/active/`: active multi-hour execution plans and relay logs.
- `runs/review/`: governed raw run bundles copied from live run roots.
- `results/generated/review/`: governed summaries, tables, JSON reports, and review-ready artifacts.
- [docs_review_bundle.zip](docs_review_bundle.zip): continuously refreshed documentation bundle for external review.

## Supported Backbones

The current codebase supports two backbone families:

- `Qwen2.5-1.5B-Instruct`
- `Qwen3-8B`

## Current Bottom Line

The repo is past the old "is the route physically connected?" question. Under `PLANv7`, the current evidence says:

- structured multi-item support is better than pooled-only support,
- anti-homogenization losses materially help,
- FEVER can be stabilized, but FEVER alone no longer decides the architecture,
- the current Writer is still under-provisioned relative to the hypothesis being tested,
- pure mid-layer `[12,13,14,15]` remains the most important unscreened injection variable,
- the next authorized step is `V7-0`: metrics repair, continuity replay, and oracle gating on GSM8K plus TriviaQA.
