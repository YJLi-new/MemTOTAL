# PLANv8 Restart: Qwen3-4B Parallel Track

## Purpose

Restart `PLANv8` on `Qwen3-4B` without overwriting the finished `Qwen3-8B` governed record.

This is a backbone change, so it must open a new governed track rather than pretending the old `Qwen3-8B` calibration still applies.

## Decision

- Preserve the completed `Qwen3-8B` `V8-0` and `V8-1` review surfaces.
- Open a new parallel `Qwen3-4B` namespace.
- Re-enter `PLANv8` at `V8-0`.

## Why Restart From V8-0

`V8-1` depends on `V8-0` outputs for:

- prompt-family selection,
- calibrated no-memory baselines,
- oracle/interface sanity checks,
- the decision to open the reader-interface scout.

Changing the backbone invalidates those `Qwen3-8B` calibration results, so `Qwen3-4B` must first complete its own `V8-0` pass.

## Scope

1. Add shared backbone support for `Qwen3-4B`.
2. Reuse the generalized `V8-0` summary builder so the new track emits `qwen34_*` decisions without mutating the finished qwen3 outputs.
3. Parameterize the governed `V8-0` runner, then add a thin qwen34 wrapper that writes to:
   - `/root/autodl-tmp/runs/verify/planv8-v8-0-qwen34-baselines-oracles`
   - `/root/autodl-tmp/results/generated/planv8-v8-0-qwen34-baselines-oracles`
4. Extend the review publisher to surface the qwen34 `V8-0` track.
5. Launch watched tmux sessions and auto-publish the `V8-0` result when the summary lands.

## Notes

- The repo already exposes the governed `q3_*` prompt variants; there is no separate `q34_*` prompt-family runtime surface yet, so `V8-0` reuses the existing governed prompt ids during calibration instead of inventing unsupported names.
- `Qwen3-4B` remains in the same Qwen3 family and the official model card still reports `36` layers, so the `V8-0` mid-band oracle geometry can stay aligned with the existing qwen3 harness.

## Validation

```bash
python -m py_compile \
  src/memtotal/models/backbone.py \
  src/memtotal/utils/repro.py \
  scripts/update_planv8_v8_0_summary.py \
  scripts/update_planv8_v8_1_summary.py
python -m unittest \
  tests.test_backbone_real_mode \
  tests.test_planv8_v8_0_summary \
  tests.test_planv8_v8_1_summary \
  tests.test_repo_lints \
  tests.test_repo_contract -v
bash -n \
  scripts/prepare_local_qwen3_model.sh \
  scripts/prepare_local_qwen34_model.sh \
  scripts/run_planv8_v8_0_qwen3_baselines_oracles.sh \
  scripts/run_planv8_v8_0_qwen34_baselines_oracles.sh \
  scripts/publish_review_artifacts.sh
```

## Progress

- 2026-03-14 UTC: Confirmed that swapping the backbone from `Qwen3-8B` to `Qwen3-4B` reopens the governed restart point at `V8-0`.
- 2026-03-14 UTC: Confirmed the official `Qwen3-4B` model card keeps the same `36`-layer depth, so the existing `V8-0` middle-band oracle layer placement remains consistent.
- 2026-03-14 UTC: Hardened qwen34 local staging after the first live run stalled on `model-00002-of-00003.safetensors`:
  - `prepare_local_qwen3_model.sh` now keeps small-file `hf_hub_download()` retries but resolves signed shard URLs explicitly and downloads large weights with resumable `wget -c`
  - the relaunched `planv8_v80_q34` session resumed forward progress immediately from the partial local shard set under `/root/autodl-tmp/models/Qwen3-4B`
- 2026-03-14 UTC: Added an idempotent chain-arm helper, [`scripts/arm_planv8_qwen34_chain.sh`](/root/mydir/MemTOTAL/scripts/arm_planv8_qwen34_chain.sh), so the qwen34 unattended stack can be re-armed without reconstructing ad hoc tmux commands:
  - restores the `V8-0` runner, watcher, local shard watcher, post-publish tail, `V8-1` queue, and chain superwatch if any session drops
  - upgrades the live shard watcher to report the real staged files in `/root/autodl-tmp/models/Qwen3-4B` instead of the stale Hugging Face cache artifact path
