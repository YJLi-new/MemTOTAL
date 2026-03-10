# PLANv6 Phase V6-5 - Recipe Stabilization And Layer Comparison

## Purpose

Run the `PLANv6` `V6-5` stabilization branch so the repo can convert the strongest `V6-4` finalists into stable FEVER wins before deciding whether `V6-6` is necessary.

## Context

- `PLANv6.md` defines `V6-5` as the next authorized phase after `V6-4`.
- `V6-4` selected the following finalists:
  - `s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage`
  - `s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg`
  - `s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive`
- `PLANv6` requires the following sweep variables:
  - warmup length `0 / 10 / 20`
  - clipping scheme `global / group-aware`
  - projector LR `5e-5 / 7.5e-5`
  - accumulation `1 / 4`
  - layer set `base early [0,1,2,3] / additive [0,1,2,3,4,8,14]`
- `PLANv6` also says additive layer expansion must not remove the early layers.

## Assumptions

- `V6-5` is implemented as a FEVER-first stabilization phase. The screen uses FEVER only, then the top two screen recipes are confirmed across three FEVER seeds. This keeps the phase aligned with the `PLANv6` exit criterion of stable FEVER improvement while controlling runtime.
- The additive layer recipe expands deep-prefix injection to `[0,1,2,3,4,8,14]`, but receiver micro-LoRA remains tiny and therefore uses `[0,1,2,3,4]`. This matches the runtime guardrail that writer-direct receiver micro-LoRA may not target more than five layers.
- Publisher wiring for the new review bundle may be handled outside this harness if another in-flight change owns `scripts/publish_review_artifacts.sh`.

## Canonical Sweep

For each of the three `V6-4` finalists, screen the full `3 x 2 x 2 x 2 x 2 = 48` recipe combinations on FEVER:

- warmup `0 / 10 / 20`
- clipping `global / groupwise`
- projector LR `5e-5 / 7.5e-5`
- accumulation `1 / 4`
- layers `base / additive`

Total FEVER screen size: `144` injected runs plus one FEVER control.

After the screen:

- rank all recipes from the FEVER-only summary,
- select the top two recipes,
- confirm both across FEVER seeds `{base, base+1, base+2}` with paired FEVER controls.

## Plan Of Work

1. Add a dedicated `V6-5` runner that materializes FEVER data, builds configs from the `V6-4` finalists, and executes the stabilization sweep.
2. Add a `V6-5` summary script that ranks the screen recipes and aggregates the multi-seed confirmation stage.
3. Add a summary-contract unit test for the new report.
4. Publish the governed FEVER-first review bundle once the runtime-side accumulation support is available.

## Concrete Steps

1. Add the runner:
   - [`scripts/run_planv6_v6_5_recipe_stabilization_qwen25.sh`](/root/mydir/MemTOTAL/scripts/run_planv6_v6_5_recipe_stabilization_qwen25.sh)
2. Add the summary script:
   - [`scripts/update_planv6_v6_5_recipe_stabilization_summary.py`](/root/mydir/MemTOTAL/scripts/update_planv6_v6_5_recipe_stabilization_summary.py)
3. Add the summary test:
   - [`tests/test_planv6_v6_5_recipe_stabilization_summary.py`](/root/mydir/MemTOTAL/tests/test_planv6_v6_5_recipe_stabilization_summary.py)
4. Materialize FEVER data via `python -m memtotal.tasks.writer_jointpeft_data`.
5. Run screen stage at:
   - `/root/autodl-tmp/runs/verify/planv6-v6-5-recipe-stabilization-qwen25`
   - `/root/autodl-tmp/results/generated/planv6-v6-5-recipe-stabilization-qwen25`
6. After screen completion, rerank and confirm the top two recipes across three FEVER seeds.

## Validation And Acceptance

Phase `V6-5` is complete only if all of the following are true:

- the runner executes the FEVER control plus the full `144` recipe screen,
- the summary ranks the screen and selects the top two recipes for confirmation,
- the confirmation stage runs both selected recipes across three FEVER seeds with paired FEVER controls,
- the final summary exposes stable FEVER improvement, source-collapse diagnostics, and Writer task-supervision diagnostics for each confirmed recipe,
- the new runner passes static validation and the summary contract test passes.

## Progress

- `2026-03-10 12:40 UTC`: opened the `V6-5` exec plan after `V6-4` published `comparison_conclusion=select_finalists` and `recommended_next_step=open_v6_5_recipe_stabilization`.
- `2026-03-10 12:44 UTC`: decided on a FEVER-first stabilization branch for `V6-5` because the formal `PLANv6` exit criterion is FEVER stability, while the full three-task Cartesian sweep would be disproportionately large for a recipe-only phase.
- `2026-03-10 12:46 UTC`: documented the additive-layer receiver-LoRA compromise: deep-prefix expansion is additive, but receiver micro-LoRA stays tiny at five layers maximum.
- `2026-03-10 13:10 UTC`: landed runtime-side gradient accumulation support in `m4_shared_injection.py`, including microstep averaging for train-event diagnostics and explicit `pilot_gradient_accumulation_steps` / `pilot_effective_train_examples` metrics.
- `2026-03-10 13:18 UTC`: completed static validation for the new runner and summary (`bash -n`, `py_compile`, targeted unit tests) and confirmed the full repo suite passes at `271` tests before launch.
- `2026-03-10 13:20 UTC`: patched the runner to republish review artifacts after the final confirmation summary so the governed `results/generated/review/planv6-v6-5-recipe-stabilization-qwen25` bundle cannot lag the finished phase output.
- `2026-03-10 13:55 UTC`: launched the full `V6-5` FEVER-first stabilization sweep in detached `tmux` as session `planv6_v65`. Live run root: `/root/autodl-tmp/runs/verify/planv6-v6-5-recipe-stabilization-qwen25`. Live result root: `/root/autodl-tmp/results/generated/planv6-v6-5-recipe-stabilization-qwen25`. Live session log: `/root/autodl-tmp/runs/verify/planv6-v6-5-recipe-stabilization-qwen25/tmux-session.log`.
- `2026-03-10 14:00 UTC`: config materialization completed (`145` config artifacts: `144` screen recipes plus the FEVER control), the FEVER control already wrote `metrics.json` / `suite_metrics.json`, and the first injected screen arm `F1__w0__clip_global__plr5e5__acc1__layers_base` is running under `run_m4_selected_shared_injection_suite.py`.
- `2026-03-10 15:20 UTC`: the detached run did not finish normally. The shell exited because a child Python process raised `FileNotFoundError: [Errno 2] No usable temporary directory found`, which was triggered after the root filesystem filled and `tempfile.gettempdir()` could no longer resolve a writable temp path.
- `2026-03-10 15:22 UTC`: audited the partial phase state. FEVER control completed and `20/144` FEVER screen recipes completed before the crash; no `F2` or `F3` screen arms ran, and the confirmation stage never started.
- `2026-03-10 15:25 UTC`: patched the `V6-5` runner to force `TMPDIR` / `TEMP` / `TMP` onto `/root/autodl-tmp/tmp` so resumed execution is isolated from root-disk pressure. The recovery path is to resume in place from the existing run/result roots so the completed FEVER screen arms are reused rather than rerun.
- `2026-03-10 15:51 UTC`: relaunched `V6-5` in detached `tmux` as session `planv6_v65` with `TMPDIR=/root/autodl-tmp/tmp`, reusing the existing run/result roots.
- `2026-03-10 15:55 UTC`: the resumed run passed the old tempdir failure point and entered the first previously missing FEVER screen arm, `F1__w10__clip_global__plr75e6__acc1__layers_base`, under `run_m4_selected_shared_injection_suite.py`.
- `2026-03-10 16:00 UTC`: updated the runner policy so future `V6-5` launches prefer `/tmp/memtotal-tmp` on the system disk for read speed when root has enough free space, with `/root/autodl-tmp/tmp` kept as the fallback path when the free-space guardrail is violated.
- `2026-03-10 16:08 UTC`: moved cold historical `runs/verify/*` trees that are not part of the live `V6-5` resume path onto `/root/autodl-tmp/system-relocated/...` and replaced them with symlinks. This restored roughly `21G` free on `/`, which is enough to keep the system-temp-first policy safe without touching the in-flight `planv6_v65` session.

## Decision Log

- Use all three `V6-4` finalists in the screen because `PLANv6` authorizes `2-3` finalists and the `V6-4` ranking produced a clear top tier of three recipes.
- Keep accumulation in the sweep even though the current runtime patch may still be in flight, because `PLANv6` explicitly lists `1 / 4` accumulation as a stabilization variable.
- Confirm only the top two recipes after the screen to keep the multi-seed stage focused and tractable.

## Surprises And Discoveries

- The runtime already supports the additive deep-prefix layer set, but not large receiver micro-LoRA layer sets; the additive recipe therefore needs a split definition rather than a naive mirrored layer list.
- The `V6-4` winners already suggest the direction of travel: `S3 + C2 + L5` is the main stabilization anchor, `S3 + C2 + L3` is the softer regularized comparator, and `S3 + C0 + L2` is the control-style finalist that tests whether the gains depend on context gating.
- The summary/publish contract for long phases needs explicit end-of-phase republishing, not just mid-phase publishing, otherwise the review tree can preserve an intermediate screen-only state even when confirmation has completed successfully.
- The live run emits standard PyTorch / Transformers determinism warnings about `CUBLAS_WORKSPACE_CONFIG` under CUDA 10.2+ while deterministic algorithms are enabled; these warnings are noisy but non-fatal and do not indicate a harness regression for `V6-5`.
- Root-disk exhaustion can kill long-running Python launches indirectly through `tempfile` resolution even when the model run itself is healthy; the better policy is system-temp-first with a free-space guardrail and cold-data offloading, not unconditional `/tmp` usage and not unconditional data-disk temp usage.
- The better steady-state policy is system-temp-first with an explicit free-space guardrail and a data-disk fallback, not unconditional data-disk temp usage.
