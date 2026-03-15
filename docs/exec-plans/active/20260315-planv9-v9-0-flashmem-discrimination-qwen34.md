# PLANv9 V9-0 FlashMem Discrimination for Qwen3-4B

## Purpose

Open the first governed `PLANv9` milestone by implementing and running the mandatory qwen34 four-arm discriminator:

- `A0` no-memory control
- `A1` legacy-prefix oracle replay
- `A2` FlashMem-style `precache_latent` cache-prefill oracle
- `A3` in-sequence sequence-replay oracle

This phase resolves the remaining contradiction between the destructive qwen34 `PLANv8` in-sequence route and the external FlashMem claim that a backbone-processed in-stream route can stay safe on `Qwen3-4B`.

## Scope

Deliver the governed `V9-0` surface required by [PLANv9.md](/root/mydir/MemTOTAL/PLANv9.md):

- backbone support for `memory_consumer_mode = precache_latent`
- qwen34 `V9-0` config builder
- qwen34 `V9-0` runner
- `V9-0` summary builder
- focused unit tests
- review-publisher wiring for:
  - `runs/review/planv9-v9-0-flashmem-discrimination-qwen34`
  - `results/generated/review/planv9-v9-0-flashmem-discrimination-qwen34`

## Design

The phase stays narrow on purpose:

- reuse the published qwen34 `V8-0` `GSM8K` 64-example eval split
- reuse the selected qwen34 `V8-0` prompt mode from:
  - `results/generated/review/planv8-v8-0-qwen34-baselines-oracles/selected-prompt-modes.json`
- keep all four arms on the same qwen34 decoding settings
- do not add training sweeps or long-horizon benchmark work here

The new `A2 precache_latent` route consumes oracle hidden-state slots by:

1. chunk-pooling support hidden states into `8 x H` latent vectors
2. RMS-rescaling them to the live content-embedding norm
3. running a memory-only `inputs_embeds` pre-pass with `use_cache=True`
4. feeding the real task prompt against the prefixed KV cache

## Files

- [PLANv9.md](/root/mydir/MemTOTAL/PLANv9.md)
- [AGENTS.md](/root/mydir/MemTOTAL/AGENTS.md)
- [README.md](/root/mydir/MemTOTAL/README.md)
- [backbone.py](/root/mydir/MemTOTAL/src/memtotal/models/backbone.py)
- [m4_shared_injection.py](/root/mydir/MemTOTAL/src/memtotal/training/m4_shared_injection.py)
- [planv9_v9_0_config.py](/root/mydir/MemTOTAL/scripts/planv9_v9_0_config.py)
- [run_planv9_v9_0_flashmem_discrimination.sh](/root/mydir/MemTOTAL/scripts/run_planv9_v9_0_flashmem_discrimination.sh)
- [run_planv9_v9_0_flashmem_discrimination_qwen34.sh](/root/mydir/MemTOTAL/scripts/run_planv9_v9_0_flashmem_discrimination_qwen34.sh)
- [update_planv9_v9_0_summary.py](/root/mydir/MemTOTAL/scripts/update_planv9_v9_0_summary.py)
- [publish_review_artifacts.sh](/root/mydir/MemTOTAL/scripts/publish_review_artifacts.sh)
- [test_backbone_real_mode.py](/root/mydir/MemTOTAL/tests/test_backbone_real_mode.py)
- [test_planv9_v9_0_config.py](/root/mydir/MemTOTAL/tests/test_planv9_v9_0_config.py)
- [test_planv9_v9_0_summary.py](/root/mydir/MemTOTAL/tests/test_planv9_v9_0_summary.py)

## Validation

Required local validation for this milestone:

- `python -m py_compile src/memtotal/models/backbone.py src/memtotal/training/m4_shared_injection.py scripts/planv9_v9_0_config.py scripts/update_planv9_v9_0_summary.py`
- `bash -n scripts/run_planv9_v9_0_flashmem_discrimination.sh scripts/run_planv9_v9_0_flashmem_discrimination_qwen34.sh scripts/publish_review_artifacts.sh`
- `python -m unittest tests.test_backbone_real_mode tests.test_planv9_v9_0_config tests.test_planv9_v9_0_summary -v`
- `python -m unittest tests.test_repo_lints tests.test_repo_contract -v`

## Acceptance

This milestone is complete when:

1. `PLANv9` is the active repo authority in [AGENTS.md](/root/mydir/MemTOTAL/AGENTS.md) and [README.md](/root/mydir/MemTOTAL/README.md)
2. qwen34 `V9-0` configs can materialize all four arms reproducibly
3. `precache_latent` works in backbone score/generate paths and is unit-tested
4. the runner can complete `A0/A1/A2/A3` on the published qwen34 `V8-0` GSM8K split
5. the summary emits:
   - `v9-0-summary.json`
   - `v9-0-summary.md`
   - governed `O0/O1/O2/O3` routing
6. the review publisher can export the new `planv9-v9-0-flashmem-discrimination-qwen34` namespace

## Progress

- 2026-03-15 UTC: adopted `PLANv9.md` into the repo root and switched repo guidance from `PLANv8` to `PLANv9`.
- 2026-03-15 UTC: started the qwen34 `V9-0` milestone with the exact published `V8-0` GSM8K split and prompt mode as the continuity anchor.
