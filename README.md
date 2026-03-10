# MemTOTAL

MemTOTAL studies an internal-memory route for frozen reasoning LLMs: write a high-capacity latent memory `M_long`, read and compress it into a short memory `M_short`, and inject only the short memory into the next reasoning context.

This repository is in active research mode. The current authoritative runbook is [PLANv6.md](PLANv6.md).

## Current Status

As of March 10, 2026:

- The active program is `PLANv6`, focused on post-gate-repair independent-Writer recovery, structured support validation, and anti-homogenization training.
- `V6-5` is complete and succeeded on its own gates, so `PLANv6` now authorizes `V6-7` rather than the `V6-6` fallback.
- The latest completed governed batch is `V6-2 -> V6-5`. That batch shows the repaired Writer-direct route can be stabilized into multi-seed FEVER gains under additive deep-prefix expansion.
- `V6-5` selected these two stabilized recipes:
  - `F3__w0__clip_groupwise__plr75e6__acc4__layers_additive`
  - `F1__w0__clip_global__plr5e5__acc1__layers_additive`
- The top stabilized recipe, `F3__w0__clip_groupwise__plr75e6__acc4__layers_additive`, delivered stable FEVER improvement across all `3/3` confirmation seeds.

## Latest Useful Reports

- `V6-5 recipe stabilization`:
  [summary](results/generated/review/planv6-v6-5-recipe-stabilization-qwen25/v6-5-summary.md),
  [json](results/generated/review/planv6-v6-5-recipe-stabilization-qwen25/v6-5-summary.json)
  - Outcome: `comparison_conclusion=select_stabilized_recipe`
  - Readout: the best stabilized FEVER recipe was `F3__w0__clip_groupwise__plr75e6__acc4__layers_additive`, with `F1__w0__clip_global__plr5e5__acc1__layers_additive` as the runner-up; both passed all `3/3` confirmation seeds and send `PLANv6` to `V6-7`.

- `V6-4 mixed matrix`:
  [summary](results/generated/review/planv6-v6-4-mixed-matrix-qwen25/v6-4-summary.md),
  [json](results/generated/review/planv6-v6-4-mixed-matrix-qwen25/v6-4-summary.json)
  - Outcome: `comparison_conclusion=select_finalists`
  - Readout: `S3` support with the gated context branch dominated the top of the ranking; FEVER improvement was real, GSM8K stayed route-live, and NarrativeQA remained non-robust.

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
  - Why it matters: this is the route-correction bundle that separated route-liveness from usefulness and directly motivated `PLANv6`.

## What To Read First

- [PLANv6.md](PLANv6.md): current execution order, acceptance criteria, and milestone gating.
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

The repo is past the old "is the route physically connected?" question. Under `PLANv6`, the current evidence says:

- structured multi-item support is better than pooled-only support,
- anti-homogenization losses materially help,
- FEVER and GSM8K can both be route-live under the repaired gates,
- NarrativeQA is still the main failure mode,
- `V6-5` converted the best `V6-4` finalists into stable multi-seed FEVER wins,
- the next authorized step is `V6-7`: Reader reopening, portability, and usefulness scaling.
