# 20260309 PLANv5 Addendum: Stable Writer-Direct Validation

## Purpose
Correct the PLANv5 `P1a/P2a` stop logic and continue the active branch with the first stable real-Writer validation recipe.

## Context
- `PLANv5.md` remains the governing top-level plan.
- The historical `P1a/P2a` source-stub bundle proved route liveness but used an invalid usefulness gate for a static prefix source.
- `source_stub` is a fixed learnable prefix source. It can show that gradients and prefix routing are physically live, but it cannot be the decisive non-FEVER usefulness test for the independent Writer thesis.
- The addendum keeps the existing deep-prefix infrastructure, early-layer receiver micro-LoRA, and KV-stat initialization. It changes the decision rule and the training recipe.

## Governing Corrections
1. Reclassify `P1a/P2a` as historical route-liveness probes.
2. Revoke `stop_after_p2a` as a live decision rule.
3. Split evaluation into three distinct questions:
   - `route_live`
   - `stable_training`
   - `usefulness_positive`
4. Keep `source_stub` only as a short health check after the recipe fix.
5. Move the real decision point to `writer_direct + sparse_deep_prefix + early4 receiver micro-LoRA`.

## Canonical Live Branch
### Source-stub health check
- task: GSM8K only
- source: `source_stub`
- injection: `sparse_deep_prefix`
- layers: `[0, 1, 2, 3]`
- receiver LoRA: early4 `k_proj/v_proj`, rank `2`, alpha `4`, dropout `0.0`
- train steps: `32`
- scheduler: `constant_with_linear_warmup`
- warmup: `8`
- source-stub LR: `1e-4`
- projector LR: `1.5e-4`
- receiver-LoRA LR: `5e-5`
- gradient clip: `1.0`
- purpose: verify that the unstable high-LR source-stub recipe is fixed before the real Writer branch runs

### Writer-direct canonical run
- tasks:
  - GSM8K medium slice: support `8`, train `80`, eval `40`
  - NarrativeQA medium slice: support `8`, train `32`, eval `24`
  - FEVER medium slice: support `8`, train `64`, eval `64`
- source: real Writer
- bridge mode: `writer_direct`
- injection: `sparse_deep_prefix`
- selected layers: `[0, 1, 2, 3]`
- receiver LoRA: early4 `k_proj/v_proj`, rank `2`, alpha `4`, dropout `0.0`
- Writer shape: current W0-sized `WriterWeaverHead`
- support encoder: `pooled_block`
- Writer stimulus: `support_and_context`
- Writer context tokens: `8`
- train steps: `500`
- scheduler: `constant_with_linear_warmup`
- warmup: `50`
- writer freeze phase: steps `1-50` via `pilot_projector_warmup_steps=50`
- writer LR: `1e-4`
- projector LR: `1.5e-4`
- receiver-LoRA LR: `5e-5`
- gradient clip: `1.0`
- Writer regularizers:
  - `pilot_writer_gain_margin_weight=0.25`
  - `pilot_writer_common_mode_penalty_weight=0.1`
  - `pilot_writer_covariance_diversity_weight=0.05`
  - `pilot_writer_slot_energy_balance_weight=0.01`
- snapshot steps: `[0, 10, 25, 50, 100, 200, 350, 500]`

## Acceptance Criteria
### Route live
Pass if GSM8K or NarrativeQA shows:
- Writer grad median over steps `1-50` `> 1e-4`
- projector grad median over steps `1-50` `> 1e-3`
- at least one selected layer with nontrivial prefix attention
- all logged metrics finite

### Stable training
Pass if:
- `train_loss_steps_451_500_median < train_loss_steps_1_50_median`
- no NaN or Inf appears
- `delta_answer_logprob` is not identically zero across all eval rows on at least one non-FEVER task

### Usefulness positive
Pass if:
- mean `delta_answer_logprob > 0` on GSM8K or NarrativeQA
- the task score is non-regressive versus control

## Fallback Order
1. If `route_live` fails:
   - treat it as implementation or recipe failure
   - stay on the same architecture
2. If `route_live` passes but `stable_training` fails:
   - stay on the same architecture
   - adjust recipe only
3. If `stable_training` passes but non-FEVER prefix attention remains weak:
   - open the deeper-layer comparator `[0, 4, 8, 14]`
4. Only if the deeper-layer comparator still yields live and stable training without usefulness:
   - open the Prefix-Tuning+-style attention-independent branch

## Required Harness Changes
1. Add true LR scheduling with warmup to shared-injection training.
2. Record per-group learning rates, clip flags, and head/tail metric windows.
3. Materialize deterministic medium slices for GSM8K, NarrativeQA, and FEVER.
4. Publish a corrected addendum summary beside the historical writer-circuit-opening bundle.
5. Add the writer-direct G2 templates, runner, and summary script.

## Validation & Acceptance
- Static checks:
  - `python -m py_compile src/memtotal/training/m4_shared_injection.py src/memtotal/tasks/writer_jointpeft_data.py scripts/update_writer_circuit_opening_addendum_summary.py scripts/update_writer_deep_prefix_jointpeft_summary.py`
  - `bash -n scripts/run_writer_deep_prefix_jointpeft_qwen25.sh`
  - `bash -n scripts/publish_review_artifacts.sh`
- Unit tests:
  - `python -m unittest tests.test_writer_deep_prefix_jointpeft_summary tests.test_writer_jointpeft_data -v`
- Live entry point:
  - `./scripts/run_writer_deep_prefix_jointpeft_qwen25.sh`

## References
- Prefix-Tuning+: `https://arxiv.org/abs/2506.13674`
- MemGen: `https://arxiv.org/abs/2509.24704`
- When Do Prompting and Prefix-Tuning Work?: `https://arxiv.org/abs/2310.19698`
- P-Tuning v2: `https://arxiv.org/abs/2110.07602`

## Progress
- 2026-03-09: Opened as the live continuation of `PLANv5.md` after correcting the historical source-stub interpretation.
- 2026-03-09: Scope fixed to stable writer-direct validation before any deeper architecture pivot.
