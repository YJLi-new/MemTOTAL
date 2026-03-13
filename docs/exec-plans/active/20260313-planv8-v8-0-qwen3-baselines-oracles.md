# PLANv8 V8-0 Qwen3 Baselines and Oracle / Interface Sanity

## Purpose

Implement and launch the first governed `PLANv8` phase, [`V8-0`](../../PLANv8.md), using the existing MemTOTAL governed harness rather than a sidecar experiment stack.

The concrete goal is to finish the minimum pre-live engineering required for `V8-0`, then run the phase in a new non-overwriting namespace and publish the governed summary bundle.

## Context

- `PLANv8.md` is now the next execution authority.
- Historical `PLANv7` and `PLANv7 (LR updated version)` are closed and already published.
- The repo already has:
  - `Qwen3-8B` backbone naming in [`src/memtotal/models/backbone.py`](/root/mydir/MemTOTAL/src/memtotal/models/backbone.py)
  - the governed `run_m4_selected_shared_injection_suite.py` execution backend
  - `V7-0` shell/summary/publish patterns that closely match the required `V8-0` scaffold
- The main missing capabilities are:
  - a real sequence-memory path in the HF backbone path
  - a deterministic oracle hidden-state Writer path (`EW0`)
  - a backbone-adjacent Reader cross-attention adapter smoke path (`RI2`)
  - `PLANv8`-specific runner / summary / review publication wiring

## Plan Of Work

1. Add a `PLANv8` active exec plan and lock the implementation scope around `V8-0`.
2. Extend the backbone for:
   - sequence-memory token prepending in real HF mode
   - deterministic chunk-pooled hidden-state slot extraction
   - optional backbone-side Reader cross-attention adapters
3. Extend the shared-injection runtime so `V8-0` can use:
   - `EW0` oracle hidden-state slots
   - `RI1` sequence-memory consumption
   - `RI2` cross-attention adapter smoke
4. Add the governed `V8-0` runner, summary builder, tests, and publish wiring.
5. Validate locally with static checks plus focused/unit tests.
6. Launch `V8-0` in detached `tmux`, add a watcher, and arm milestone publish/push flow.

## Concrete Steps

1. Create the active `PLANv8` exec plan file.
2. Patch [`src/memtotal/models/backbone.py`](/root/mydir/MemTOTAL/src/memtotal/models/backbone.py) to add:
   - sequence-memory `memory_tokens` support for `score_continuations()` and `generate()`
   - chunk-pooled hidden-state slot extraction for oracle Writer materialization
   - backbone-side Reader cross-attention adapters with trainable parameter exposure
3. Patch [`src/memtotal/training/m4_shared_injection.py`](/root/mydir/MemTOTAL/src/memtotal/training/m4_shared_injection.py) to add:
   - `PLANv8` consumer mode routing
   - `EW0` oracle slot construction
   - sequence-memory / cross-attn artifact propagation
   - new optimizer groups / metrics needed for `V8-0`
4. Add the `V8-0` governed shell runner and summary builder:
   - [`scripts/run_planv8_v8_0_qwen3_baselines_oracles.sh`](/root/mydir/MemTOTAL/scripts/run_planv8_v8_0_qwen3_baselines_oracles.sh)
   - [`scripts/update_planv8_v8_0_summary.py`](/root/mydir/MemTOTAL/scripts/update_planv8_v8_0_summary.py)
5. Add the matching tests:
   - backbone real-mode sequence-memory / cross-attn smoke
   - shared-injection oracle hidden-state slot path
   - `V8-0` summary contract
6. Extend [`scripts/publish_review_artifacts.sh`](/root/mydir/MemTOTAL/scripts/publish_review_artifacts.sh) for the `planv8-v8-0-*` namespace.
7. Launch `V8-0` in detached `tmux`, then arm:
   - a watcher session
   - a post-completion publish / commit / push session

## Validation & Acceptance

Static / local:

```bash
bash -n scripts/run_planv8_v8_0_qwen3_baselines_oracles.sh scripts/publish_review_artifacts.sh
python -m py_compile \
  src/memtotal/models/backbone.py \
  src/memtotal/training/m4_shared_injection.py \
  scripts/update_planv8_v8_0_summary.py
python -m unittest \
  tests.test_backbone_real_mode \
  tests.test_m4_shared_injection \
  tests.test_planv8_v8_0_summary \
  tests.test_repo_lints \
  tests.test_repo_contract -v
```

Run-time acceptance for `V8-0`:

- real-HF `Qwen3-8B` load succeeds
- all `b0..b4` calibration arms finish
- all `o0..o4` sanity arms finish
- the phase summary selects one prompt mode per primary task
- the summary records whether `RI1` and `RI2` passed basic smoke
- the governed review bundle publishes under the `planv8-v8-0-*` namespace

## Progress

- 2026-03-13 UTC: Opened the `PLANv8` `V8-0` exec plan after pulling `PLANv8.md` and confirming it is now the next run authority.
- 2026-03-13 UTC: Audited the current harness. Reusable assets confirmed:
  - `Qwen3-8B` backbone naming already exists
  - `prefix_embeddings` already prepend true `inputs_embeds`
  - `V7-0` runner/summary/publish files are the closest scaffold
  - `receiver_lora` plumbing already exists and can be extended
- 2026-03-13 UTC: Gap analysis confirmed the main new work is not dataset or publisher plumbing; it is the missing `PLANv8` consumer route:
  - sequence-memory in HF scoring/generation
  - deterministic oracle hidden-state slots
  - backbone-side cross-attention Reader adapter smoke
- 2026-03-13 UTC: Completed the `V8-0` runtime surface in the governed harness:
  - HF sequence-memory prepend path
  - chunk-pooled hidden-state slot extraction
  - backbone-side Reader cross-attention adapters
  - shared-injection support for `oracle_hidden_state_slots`, `RI1`, and `RI2`
- 2026-03-13 UTC: Added the governed `V8-0` shell runner, summary builder, Qwen3 local-model prep script, summary contract test, publish wiring, and refreshed repo lint authority to `PLANv8.md`.
- 2026-03-13 UTC: Local validation passed:
  - `bash -n scripts/prepare_local_qwen3_model.sh scripts/run_planv8_v8_0_qwen3_baselines_oracles.sh scripts/publish_review_artifacts.sh`
  - `python -m py_compile src/memtotal/models/backbone.py src/memtotal/training/m4_shared_injection.py scripts/update_planv8_v8_0_summary.py tests/test_planv8_v8_0_summary.py tests/test_backbone_real_mode.py`
  - `python -m unittest tests.test_backbone_real_mode tests.test_m4_shared_injection tests.test_planv8_v8_0_summary -v`
  - `python -m unittest tests.test_repo_lints tests.test_repo_contract -v`
- 2026-03-13 UTC: Live detached `V8-0` launch is now armed:
  - run session: `planv8_v80`
  - post-publish / commit / push session: `planv8_v80_post`
  - watcher session: `planv8_v80_watch`
  - run root: `/root/autodl-tmp/runs/verify/planv8-v8-0-qwen3-baselines-oracles`
  - result root: `/root/autodl-tmp/results/generated/planv8-v8-0-qwen3-baselines-oracles`
- 2026-03-13 UTC: Startup repair applied immediately after first launch:
  - first attempt died before model staging because `prepare_local_qwen3_model.sh` was invoked as `./scripts/...` without execute permissions
  - repaired by switching the runner to `bash scripts/prepare_local_qwen3_model.sh`
  - relaunch succeeded and advanced into `Qwen3-8B` local model staging
- 2026-03-13 UTC: Second startup repair applied during live staging:
  - the original `Qwen3` prep path used the default Xet transport and remained abnormally slow at `0/15` arms with no GPU work opened
  - `prepare_local_qwen3_model.sh` now:
    - validates sharded completeness before early exit instead of treating `config.json + model.safetensors.index.json` as sufficient
    - clears stale local Hugging Face `.lock` files before resume
    - defaults to `HF_HUB_DISABLE_XET=1` and `HF_HUB_ENABLE_HF_TRANSFER=0` so resumed sharded downloads use resumable parallel HTTP instead of the slow Xet path observed here
  - the main `planv8_v80` session was restarted against the same roots after the patch
  - the resumed `model-00001/00002/00003` incomplete shard files began growing again on the patched path, confirming forward progress
- 2026-03-13 UTC: Third startup repair applied after the patched HTTP path still aborted on a broken stream:
  - `snapshot_download()` was still too brittle because one interrupted shard transfer terminated the full staging pass
  - `prepare_local_qwen3_model.sh` now downloads required files individually with `hf_hub_download()` plus retry/backoff, preserving resume behavior for partial shard files
  - the main `planv8_v80` session was relaunched again on the same run/result roots
  - post-relaunch validation confirmed the final missing shard (`model-00001-of-00005`) resumed and grew again under the new retry-based path
- 2026-03-13 UTC: Fourth runtime repair applied after the first oracle arm failed immediately:
  - root cause: `selected-prompt-modes.json` was built from `metrics["pilot_prompt_variant"]`, but governed metrics expose the field as `metrics["prompt_variant"]`
  - effect: the selected prompt variants for `gsm8k` and `triviaqa` were written as empty strings, which later materialized oracle configs with `pilot_prompt_variant=""`
  - repair:
    - `scripts/run_planv8_v8_0_qwen3_baselines_oracles.sh` now reads `prompt_variant` first and falls back to `pilot_prompt_variant`
    - `scripts/update_planv8_v8_0_summary.py` uses the same fallback so the final governed summary reflects the real selected baseline prompts
  - baseline arms remain valid and reusable; the main session can relaunch from the completed baseline state without recomputing them
- 2026-03-13 UTC: Fifth runtime repair applied after the next oracle arm failed at snapshot step `0`:
  - failing branch: `o1_q3_prefix_oracle_mid4_*`
  - root cause: `oracle_hidden_state_slots + legacy_prefix` correctly built per-layer prefix tensors, but `SharedInjectionPilotRuntime.score_example()` and `generate_text()` were still forwarding the internal `memory_tokens` tensor into the HF backbone
  - the HF real-mode scorer/generator only accepts `memory_tokens` for sequence-memory consumption (`prepend_block`) or backbone-side cross-attention; `legacy_prefix` should pass only `layer_prefix_hidden_by_layer`
  - repair:
    - `score_example()` and `generate_text()` now suppress `memory_tokens` when `pilot_memory_consumer_mode=legacy_prefix`
    - added regression coverage in `tests/test_m4_shared_injection.py` to lock this routing behavior
  - baseline and completed oracle arms remain reusable; relaunch continues from the first unfinished oracle config
- 2026-03-13 UTC: Current live state:
  - dataset materialization completed
  - `Qwen3-8B` local staging is in progress under `/root/autodl-tmp/models/Qwen3-8B`
  - watcher confirms `0/15` arms complete so far, which is expected before the first baseline arm opens

## Decision Log

- Reuse the governed `M4` / `V7-0` harness instead of writing a new ad hoc experiment runner.
- Treat existing `prefix_embeddings` support as the nearest ancestor of `RI1`, but do not overload the old `prefix_embeddings` path silently; add an explicit sequence-memory path.
- Keep the legacy deep-prefix path intact for `o0/o1` continuity control rather than mutating it into the new mainline.
- Implement only the minimum `RI2` smoke needed for `V8-0`; broader consumer sweeps belong to `V8-1/V8-2`.

## Surprises & Discoveries

- `BackboneWrapper.generate(memory_tokens=...)` already exposes the right public argument, but the HF branch currently ignores it completely.
- The current repo already has enough oracle extraction helpers and optimizer-group machinery that `V8-0` does not need a separate training stack.
- There was no existing `prepare_local_qwen3_model.sh`; `V8-0` now adds one via `huggingface_hub.snapshot_download` instead of relying on transient cache state.
- `snapshot_download(..., local_dir=...)` stages metadata into the target directory first and can remain silent while shards are still downloading, so a quiet runner plus zero GPU use is expected during the earliest `Qwen3` setup window.
- `snapshot_download(..., local_dir=...)` can leave a partially staged directory that looks superficially complete because `config.json` and `model.safetensors.index.json` appear before the shard set is complete; the prep script now validates the full shard map before declaring success.
- For this environment and repo state, the default Xet path was materially slower than resumable parallel HTTP for `Qwen3-8B` local staging.
- For this environment, large-shard `snapshot_download()` was also too fragile as a phase gate because a single `ChunkedEncodingError` aborted the whole stage; per-file resumable retries are safer for governed long-run setup.
- The governed runtime writes prompt provenance to `metrics["prompt_variant"]`; runner-side prompt-selection logic must read that field instead of assuming `pilot_prompt_variant` is mirrored into the final metrics payload.
- `legacy_prefix` is a deep-prefix projection path, not a sequence-memory path; any internal oracle slot tokens used to construct layer-wise prefixes must be stripped before the final HF scoring/generation call.
