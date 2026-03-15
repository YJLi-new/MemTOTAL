# 2026-03-15 — PLANv9 V9-1 Long-Horizon Baselines (qwen34)

## Goal

Promote the repo's long-horizon benchmark scaffolds into governed qwen34 pilot baselines after the `V9-0` hard fail on `A2 precache_latent`.

`V9-0` outcome:

- `outcome_id = O2`
- `comparison_conclusion = flashmem_precache_collapse_matches_sequence_replay`
- `recommended_next_step = hard_fail_a2_shift_mainline_consumer_to_c0_or_c2`

That means `V9-1` must harden the benchmark layer under the `C0/C2-only` constraint before any new latent-memory consumer work opens.

## Scope

This milestone adds:

- governed pilot materialization for:
  - `MemoryAgentBench`
  - `LongMemEval`
  - `ALFWorld`
- qwen34 baseline runner + summary surface for:
  - `B0 short-window`
  - `B1 full-history`
  - `B2 text-summary`
  - `B3 text-RAG`
- unattended tmux arm/watch/post/supervisor scripts with a `600s` restart loop
- review publication wiring for:
  - `planv9-v9-1-longhorizon-baselines-qwen34`

## Pilot policy

### MemoryAgentBench

- governed size: `25` examples per competency, `100` total
- `B0` keeps a bounded short-window control
- `B1` removes the old `512`-token smoke truncation
- `B2` uses deterministic text summaries from keypoints / prior events
- `B3` retrieves a bounded, evenly spread per-episode context bank only

### LongMemEval

- official cleaned source: `longmemeval_s_cleaned.json`
- governed primary pilot: `100` questions across five plan-authorized question types
- auxiliary holdout:
  - `knowledge-update`
- `B3` retrieves from the question's own session bank only

### ALFWorld

- governed size: `20` episodes x `6` task families = `120` total
- source priority:
  - `valid_seen`
  - `valid_unseen`
  - `valid_train`
- evaluation mode:
  - one expert bootstrap action
  - then qwen34 controls the episode
- tracked metrics:
  - `success_rate`
  - `mean_steps_executed`
  - `mean_invalid_resolution_count`

## Acceptance gate

`V9-1` is considered complete only if:

- all four baselines produce stable outputs for:
  - `MemoryAgentBench`
  - `LongMemEval`
  - `ALFWorld`
- the result summary writes:
  - `v9-1-summary.json`
  - `v9-1-summary.md`
- the review namespace publishes successfully

If those conditions hold, the governed next step is:

- `open_v9_2_withinsession_sharedkv_scout_c0_c2`
