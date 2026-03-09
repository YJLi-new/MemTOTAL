# PLANv3.md — Post-RG3 Recovery Plan for MemTOTAL

## 0. What this document is, and what it is not

This document is **not** a replacement for `PLAN.md`.

`PLAN.md` still has the correct macro order:

1. stabilize the active bridge and stop dead-end loops,
2. make the true two-level path actually alive,
3. refresh transfer / adaptation evidence,
4. run CDMI and broader benchmark evidence,
5. confirm on the stronger backbone,
6. only then broaden fallback logic if still needed.

What changed is the **next hop after RG-3**.

`PLANv2.md` correctly narrowed the current blocker to Workstream B / Failure Mode B-1 and executed the Reader/Fuser rescue line through RG-3. That line has now reached a legitimate stop condition:

- `RG-3` finished,
- `comparison_conclusion = failure`,
- `move_to_rg4 = false`,
- `stop_after_rg3 = true`,
- final classification = `B-1_local_bootstrap_failed`.

Therefore this document does three things:

1. **preserves** the macro sequence from `PLAN.md`,
2. **supersedes** `PLANv2.md` only for the next hop after RG-3,
3. introduces two changes that are now justified and required:
   - **mandatory tiny receiver-side LoRA** (much lighter than MemGen's LoRA usage),
   - **mandatory multi-task validation** (not FEVER-only anymore).

The central claim of this plan is:

> The project should remain on the main two-level memory track, but the immediate blocker is no longer "Reader geometry alone." It is now a **compound blocker**: **writer-side value homogenization** plus **receiver-side gradient starvation**. The next phase must address both, while also checking that any gain is not FEVER-specific.

---

## 1. Inputs reviewed for this plan

This plan is based on the following reviewed inputs:

### 1.1 Repo code paths reviewed

- `src/memtotal/models/memory.py`
- `src/memtotal/models/backbone.py`
- `src/memtotal/training/m4_shared_injection.py`
- `src/memtotal/analysis/m4_shared_injection.py`
- `src/memtotal/baselines/adapters.py`
- active scripts and summaries under `scripts/` and `results/generated/review/`

### 1.2 Docs reviewed

- `PLAN.md`
- `PLANv2.md`
- `README.md`
- `docs/TODO_LIST.md`
- `docs/ARCHITECTURE.md`
- `docs/EXPERIMENTS_INFO.md`
- `docs/exec-plans/active/20260309-tl-reader-geometry-rescue.md`
- uploaded review/docs bundle

### 1.3 Result artifacts reviewed

At minimum:

- `results/generated/review/tl-slot-basis-rescue-fever-qwen25/slot-basis-summary.json`
- `results/generated/review/tl-reader-symmetry-break-fever-qwen25/rg2-summary.json`
- `results/generated/review/tl-reader-local-bootstrap-fever-qwen25/rg3-summary.json`
- corresponding `train_events.json` files under `runs/review/...`

### 1.4 External reference classes used to sharpen this plan

- Prefix/deep prompt tuning stability and task dependence
- LoRA / PEFT / IA3
- Slot-style competitive binding and latent resampling
- Perceiver-style fixed-query readout
- feature-distortion under continued fine-tuning
- MemGen's LoRA-based trigger/weaver design as a comparison point
- benchmark/task papers for FEVER / NarrativeQA / GSM8K / HotpotQA / BoolQ

A reference list with URLs is included at the end of this document.

---

## 2. Locked current state

Before proposing anything new, the current state must be stated in a way that leaves no ambiguity.

### 2.1 What is already settled enough to stop debating

#### Settled A — Workstream A is over for now

The single-level FEVER objective line has already been taken far enough:

- M5.2 writer objective rewrite: negative but informative
- M5.3 dense teacher: truly engaged, still negative
- conclusion: do **not** continue looping on single-level FEVER objective tweaks

#### Settled B — The two-level path is really wired into the active harness

`TL-PoC`, `TL bridge rescue`, `TL slot-basis rescue`, and `RG-0` through `RG-3` have all been executed in the live FEVER harness. The current issue is not "the two-level architecture was never truly tested."

#### Settled C — Reader-local rescue reached a valid stop

`PLANv2` was correct to stop after `RG-3`. The repo now explicitly records that RG-4 is **not authorized** after the no-gain local bootstrap outcome.

#### Settled D — FEVER alone is no longer enough

At this stage, any next-hop architecture decision validated only on FEVER is too brittle to guide the full project. The paper-critical deliverables already require broader evidence; continuing FEVER-only would create a false sense of progress.

### 2.2 What the newest results jointly establish

The important points are now extremely concrete.

#### Fact 1 — Slot-basis rescue improved write-side geometry, but not capability

From `slot-basis-summary.json`:

- `tl_slot_basis_final_memory_long_effective_rank ≈ 1.6126`
- `tl_slot_basis_final_writer_slot_basis_pairwise_cosine_mean ≈ 0`
- but `selection_passed = false`
- and `screen248_test_gate_passed = false`

So the system is no longer stuck at the older trivial explanation of "`M_long` is rank-1 because Writer has no basis at all." Some write-side geometric structure has been restored.

#### Fact 2 — RG-2 partition proved that attention specialization can be forced

From `rg2-summary.json` for the partition arm:

- `reader_attention_entropy_mean ≈ 0.6931 = ln(2)`
- `reader_attention_pairwise_cosine_mean = 0.0`
- `reader_readout_effective_rank ≈ 1.2297`
- `memory_short_effective_rank ≈ 3.9853`
- still `geometry_alive = false`

This is a very strong attribution result.

It means:

- the problem is **not merely** that the Reader cannot be made to specialize in attention space,
- because hard partition **did** impose clean specialization,
- but the resulting readouts were still nearly the same in value space,
- and the bridge still did not become semantically alive.

#### Fact 3 — RG-3 proved that local Reader/Fuser bootstrap is not enough

From `rg3-summary.json`:

- bootstrap-only: no geometry gain, no collapse delay
- bootstrap + reconstruction: also no gain, slight regression on readout/short-rank metrics
- final decision: `stop_after_rg3 = true`

This means the next hop should **not** be "more Reader-only regularization." That branch has already been tested far enough.

---

## 3. Independent diagnosis: what is really going wrong now

This section is the core of the plan. It deliberately goes beyond repeating the repo's own conclusion.

### 3.1 The current blocker is no longer plain B-1; it is a compound blocker

The practical live blocker is now best described as:

> **B-1d (writer-side value homogenization)** + **early receiver gradient starvation**.

In plain terms:

- the two-level bridge now contains some structural diversity in `M_long`,
- but the actual value content seen by the Reader is dominated by one huge shared component,
- and the frozen receiver is not returning enough useful gradient to teach the Reader/Fuser how to exploit the tiny residual diversity.

So if the team keeps treating this as only "Reader geometry," it will waste time.

### 3.2 Smoking gun #1 — `M_long` is still dominated by one giant common direction

The strongest numerical clue is not attention entropy. It is the singular-value structure of `M_long` once the RG-2 partition instrumentation exists.

For the partition control path, the logged values are approximately:

- `memory_long_singular_value_top1 = 117.698`
- `memory_long_singular_value_top2 = 1.665`
- `memory_long_singular_value_top3 = 1.665`
- `memory_long_slot_norm_mean = 41.642`
- `memory_long_slot_norm_std = 0.045`

So the leading singular value is about **70.7x** the second.

That is not a healthy multi-facet memory matrix. It is a matrix whose energy is overwhelmingly concentrated in one direction.

The slot norms are also nearly identical, which means the slots are not differentiating by magnitude either. The most plausible interpretation is:

> each slot is effectively "the same large vector" plus a very small orthogonal residue.

This explains why improving raw slot-basis orthogonality did **not** automatically fix the readout problem.

### 3.3 Smoking gun #2 — the Writer code itself currently encourages this common-mode domination

The key Writer logic in `MemoryWriter.write(..., input_schema='support_set')` is:

```python
pooled_support_state = state.mean(dim=1)
conditioned_slots = slots + self.state_proj(pooled_support_state).unsqueeze(1)
attended_slots, _ = self.support_cross_attn(
    query=conditioned_slots,
    key=state,
    value=state,
    need_weights=False,
)
```

This means the same projected pooled-support vector is added to **every** slot before support cross-attention.

That is a strong architectural prior toward a shared common mode.

If the additive shared vector is much larger than the slot-specific basis component, then each slot enters support cross-attention from almost the same position in representation space. Under a tiny budget and weak downstream supervision, the system has no reason to split these slots cleanly.

So my independent verdict is:

> the current Writer architecture is not merely under-trained; it is **structurally biased toward slot value homogenization**.

### 3.4 Smoking gun #3 — partition fixes "where to look" but not "what is there"

The RG-2 partition arm is extremely informative because it forcibly destroys the usual attention symmetry problem.

Under partition:

- attention entropy falls to `ln(2)`
- query attention cosine falls to `0.0`
- but readout cosine remains about `0.9993`
- and readout effective rank remains only about `1.23`

This means the four query heads were made to look at different slot groups, but they still extracted almost the same content.

That is only possible if the **value space itself** is nearly identical across those slots.

This is why I do **not** believe the current main explanation should remain "Reader specialization failed." The more accurate explanation is:

> Reader attention specialization can be imposed, but there is too little genuinely distinct value content available to read.

### 3.5 Smoking gun #4 — high `M_short` rank from the linear Fuser is mostly pseudo-diversity

The linear Fuser raises `memory_short_effective_rank` to about `3.98`, which at first looks excellent.

But this does **not** mean four distinct semantic factors are reaching the reasoner.

Why?

Because `reader_readout_effective_rank` remains around `1.23` while `M_short` rises to nearly full rank. Therefore the Fuser is creating diversity by projecting nearly identical readouts through different linear subspaces.

That can help numerical rank, but it does not solve the semantic bottleneck.

So `M_short` rank must no longer be interpreted in isolation.

From now on, any claim that the short bridge is healthy must require both:

- **readout diversity before Fuser**, and
- **short-slot diversity after Fuser**.

### 3.6 Smoking gun #5 — after RG-2 / RG-3, Reader/Fuser gradients are effectively starved

The RG-2 and RG-3 `train_events` indicate that local Reader/Fuser learning is almost dead once the partition line is in place.

Representative values:

- RG-2 partition step 1:
  - `grad_norm_reader ≈ 1.27e-05`
  - `grad_norm_fuser ≈ 5.40e-05`
  - `grad_norm_writer ≈ 3.08e-07`
- RG-2 partition step 32:
  - `grad_norm_reader ≈ 5.50e-10`
  - `grad_norm_fuser ≈ 2.56e-09`
  - `grad_norm_writer ≈ 4.22e-08`
- RG-3 bootstrap step 1:
  - `grad_norm_reader ≈ 1.00e-10`
  - `grad_norm_fuser ≈ 4.64e-10`
  - `grad_norm_writer = 0.0`
- RG-3 bootstrap step 32:
  - `grad_norm_reader ≈ 2.95e-10`
  - `grad_norm_fuser ≈ 1.37e-09`
  - `grad_norm_writer ≈ 2.63e-08`

So the local bootstrap branch was trying to improve a module family that, by that point, was receiving effectively no useful gradient.

That does **not** mean the Reader idea is wrong in principle.

It means the project has reached the point where a completely frozen receiver is likely starving the two-level bridge of the gradient it needs.

### 3.7 Why this changes the earlier anti-LoRA stance

Earlier in the project, it was correct to resist receiver adaptation, because the main question was whether the memory-side route itself could do anything.

That stage is over.

We now know:

- Writer-side basis can be improved,
- attention symmetry can be forced,
- `M_short` rank can be inflated,
- yet the bridge still collapses,
- and Reader/Fuser gradients become nearly zero.

At this point, a **tiny** receiver-side PEFT component is no longer an overreaction. It is now a justified **diagnostic co-intervention**.

Important nuance:

> This does **not** mean "give up and just fine-tune the receiver." It means: introduce the smallest possible receiver-side LoRA that can test whether the frozen receiver is currently the gradient bottleneck.

That is a different claim, and it is the correct next move.

### 3.8 Why FEVER-only is now actively risky

FEVER was valuable as a controlled substrate.

But after this many iterations, FEVER-only evaluation is risky for two reasons:

1. it can reward brittle fixes that do not generalize,
2. it delays the real research objective, which is not "a FEVER-only bridge," but a broader two-level memory interface with later transfer and CDMI relevance.

So the next plan must include at least one reasoning-heavy non-FEVER task and one longer-context or integrative reading task that already exists in the repo.

---

## 4. Architecture viability verdicts

This section answers the question: which ideas are still live, which are not, and which must now be reframed.

### 4.1 What remains viable

#### Viable A — the full MemTOTAL method track remains viable

The repository still contains the right long-term components:

- structured support encoder
- Writer (`M_long`)
- Reader (query-based readout)
- Fuser (`M_short`)
- injector into frozen/mostly-frozen reasoner
- later Stage B/C transfer logic
- later CDMI experiments

The method track is not dead.

#### Viable B — the two-level architecture is still the right scientific target

Nothing in RG-3 invalidates the two-level idea itself.

What RG-3 invalidates is the assumption that **local Reader/Fuser rescue alone** can revive the bridge under the current writer and receiver constraints.

#### Viable C — a very small receiver-side LoRA is now methodologically acceptable

A tiny receiver-side LoRA can be introduced **inside Workstream B** without collapsing the scientific attribution, provided it is designed as follows:

- very low rank,
- very few layers,
- only attention projections nearest the injection path,
- no broad all-layer adaptation,
- no extra trigger model,
- no separate second LoRA system like MemGen's trigger + weaver split.

### 4.2 What is no longer viable as the next step

#### Not viable A — more Reader-only rescue in the RG-2 / RG-3 style

That branch has already produced its decisive information.

#### Not viable B — another single-level FEVER objective rewrite loop

The project has already learned what it can from M5.2 and M5.3.

#### Not viable C — keeping receiver PEFT as a purely distant fallback

That was justifiable before RG-3. It is no longer the right default now that gradient starvation has become explicit.

### 4.3 What must be reframed

#### Reframe A — receiver LoRA is no longer "Workstream E only"

For this plan, tiny receiver LoRA is **not** treated as a broad fallback after all memory-side options fail.

Instead it is treated as:

> a narrow, diagnostic, bridge-unlocking intervention inside Workstream B,

whose role is to answer whether the receiver is currently preventing the memory bridge from ever becoming readable.

#### Reframe B — multi-task evidence begins now, but in smoke form

This is **not yet** the full transfer/CDMI stage.

But from now on, any architecture claim must be checked beyond FEVER.

That means: small, repo-supported, smoke-scale multi-task validation begins immediately.

---

## 5. Strategic decision summary

### 5.1 What the project should do next

The next narrow program should be:

> **Writer Value Diversification + Micro-Receiver LoRA + Multi-Task Bridge Validation**

and it should be executed in that order.

### 5.2 What the project should not do next

Do **not** do any of the following as the immediate next hop:

- RG-4 style width restoration before the value bottleneck is fixed
- another round of local Reader/Fuser bootstrap losses
- broad receiver LoRA across many layers/modules
- FEVER-only continuation without at least one non-FEVER cross-check
- full-token KL as the main next move

### 5.3 Guiding principle for this phase

One major question per phase:

1. Can the Writer stop producing value-homogenized slots?
2. If yes, does a tiny receiver LoRA unlock useful gradient and capability?
3. If yes, does the bridge work on at least one non-FEVER task?
4. If yes, then and only then reopen broader transfer / CDMI work.

---

## 6. New phase structure for the next hop

This plan introduces a new narrow execution sequence under Workstream B.

### Phase V0 — Offline forensics patch (no new training)
### Phase V1 — Writer Value Diversification (WVD) on FEVER
### Phase V2 — Mandatory micro-receiver LoRA on FEVER
### Phase V3 — Restore non-forced Reader behavior once values diversify
### Phase V4 — Multi-task bridge validation (repo-supported non-FEVER tasks)
### Phase V5 — Qwen3-8B confirmation on the stabilized family
### Phase V6 — Re-open Stage B/C refresh and CDMI only after multi-task medium success

---

# Phase V0 — Offline forensics patch (must happen first)

## 7. Objective

Before running another real training job, the repo needs one more diagnostic layer that directly measures the common-mode/value bottleneck rather than inferring it indirectly from attention patterns.

## 8. Required new logging

These metrics must be added to the same serialization path used by the existing train/snapshot/final summaries.

### 8.1 `M_long` common-mode diagnostics

For every logged step and every selected evaluation checkpoint:

1. `memory_long_common_mode_vector_l2`
   - L2 norm of the slot-wise mean vector
2. `memory_long_common_mode_energy_ratio`
   - ratio of common-mode energy to total energy
3. `memory_long_centered_effective_rank`
   - effective rank after subtracting the slot-wise mean vector
4. `memory_long_top1_top2_ratio`
   - top singular value divided by second singular value
5. `memory_long_centered_top1_top2_ratio`
   - same after centering

Why this matters:

- if raw rank is low but centered rank is healthy, the bottleneck is dominated by a removable shared component,
- if centered rank is also poor, the Writer is not producing enough distinct content even after common-mode removal.

### 8.2 Reader value-projection diagnostics

The repo already logs Reader attention and some readout metrics. Extend this with:

1. `reader_value_projected_effective_rank`
   - rank of the value-projected slot matrix before attention weighting
2. `reader_value_projected_pairwise_cosine_mean`
   - mean pairwise cosine across the value-projected slots
3. `reader_readout_pairwise_cosine_mean`
   - ensure this remains first-class and not optional
4. `reader_readout_centered_effective_rank`
   - rank of mean-centered readouts

Why this matters:

Partition already proved that changing attention topology does not fix the bottleneck. These metrics make that failure mode measurable without guesswork.

### 8.3 Fuser pseudo-diversity diagnostics

1. `fuser_rank_gain_over_readout`
   - `rank(M_short) - rank(readouts)`
2. `fuser_diversity_without_semantic_gain_flag`
   - boolean summary flag when `M_short` rank is high but readout rank remains low and readout cosine remains near 1

Why this matters:

The project must stop reading high `M_short` rank as evidence of a healthy bridge unless pre-Fuser readouts are also diverse.

### 8.4 Gradient-routing diagnostics

Log:

- `grad_norm_support_encoder`
- `grad_norm_writer`
- `grad_norm_reader`
- `grad_norm_fuser`
- `grad_norm_projector`
- later also `grad_norm_receiver_lora`

And add summary ratios:

- `reader_to_support_grad_ratio`
- `fuser_to_support_grad_ratio`
- `receiver_lora_to_reader_grad_ratio`

These should be aggregated at least over:

- steps 1-4,
- steps 5-8,
- final step.

## 9. Files to modify in V0

### 9.1 `src/memtotal/analysis/m4_shared_injection.py`

Add the summary computations and new review fields.

### 9.2 `src/memtotal/training/m4_shared_injection.py`

Expose the intermediate tensors needed for the metrics.

### 9.3 `src/memtotal/models/memory.py`

Provide helper outputs from Writer/Reader/Fuser if necessary.

## 10. Acceptance for V0

V0 is complete only when:

- the new metrics appear in `train_events.json`,
- they also appear in snapshot/final summaries,
- the summary script can classify whether the live bottleneck is mostly common-mode domination, value-projected homogenization, or receiver starvation.

---

# Phase V1 — Writer Value Diversification (WVD)

## 11. Why Phase V1 is the real next hop

RG-2 and RG-3 already showed that the bridge is not rescued by manipulating Reader behavior alone.

The strongest remaining memory-side hypothesis is:

> the Writer currently emits slots whose values are still dominated by a large shared pooled-state component, so the Reader never receives sufficiently distinct content to read.

Therefore the next real training phase must target the Writer architecture directly.

## 12. Canonical hypothesis for V1

If the Writer stops adding the same large pooled-support vector to all slots, then:

- `M_long` common-mode dominance should fall,
- value-projected slot diversity should rise,
- partitioned Reader readouts should become less collinear,
- and only then does it become meaningful to ask whether the receiver can consume them.

## 13. Invariants for V1

To isolate the Writer variable, keep the rest as stable as possible.

### 13.1 Fixed items

- backbone: `Qwen2.5-1.5B-Instruct`
- task: FEVER
- control substrate: `partition / H=4 / K=4 / linear fuser`
- current conservative training budget: same family as RG-2 / RG-3
- current injector family: sparse deep prefix
- deep prefix layers: keep the current layer schedule
- selection / gate logic: unchanged

### 13.2 Why partition remains the V1 scaffold

Partition is **not** the final desired Reader mode.

It is the best diagnostic scaffold because it removes the usual attention-symmetry excuse. If a Writer variant still produces identical readouts under partition, then the Writer variant has failed.

## 14. Required Writer variants

### Arm W0 — Baseline control

Keep the current Writer behavior as the reference arm:

- `slot_conditioning_mode = shared_add`

This is needed to measure improvement honestly.

### Arm W1 — Scaled shared additive conditioning

Modify the current Writer to:

```python
conditioned_slots = slots + alpha * self.state_proj(pooled_support_state).unsqueeze(1)
```

with a small `alpha`.

Recommended initial values:

- `alpha = 0.02` (canonical)
- optional audit `alpha = 0.05`

Interpretation:

- if this alone sharply reduces top1/top2 ratio and readout cosine, the current bottleneck is mainly the oversized shared additive state;
- if it barely changes anything, the architecture needs a stronger change.

### Arm W2 — Pure slot-query Writer (recommended canonical arm)

Remove the shared additive pooled-state term entirely and let slot identity drive support retrieval:

```python
conditioned_slots = slots
attended_slots, _ = self.support_cross_attn(
    query=conditioned_slots,
    key=state,
    value=state,
    need_weights=False,
)
```

Then continue with the existing encoder/output path.

This is the cleanest architectural test of whether slot-specific query directions can pull different content from the support set.

### Arm W3 — Slot-query plus small shared bias (fallback)

If W2 is too unstable, use:

```python
conditioned_slots = slots + alpha * self.state_proj(pooled_support_state).unsqueeze(1)
```

with very small `alpha` (same range as W1), but keep the interpretation as "slot-driven first, shared summary second."

This is a compromise arm, not the conceptual default.

### Arm W4 — Optional common-mode subtraction audit

After Writer output, optionally subtract the slot mean:

```python
encoded_slots = encoded_slots - encoded_slots.mean(dim=1, keepdim=True)
```

Do **not** make this canonical immediately. Use it only as a diagnostic audit if W1/W2/W3 still leave top1/top2 badly high.

Why optional only:

- it can reveal whether the shared component is the real blocker,
- but it also changes the representational contract aggressively,
- so it is better as a late audit than as the first default.

## 15. Required Writer-side losses in V1

### 15.1 Keep existing slot-basis orthogonality support

Do **not** throw away the useful part of slot-basis rescue.

Keep the existing slot-basis warm-start and orthogonality machinery unless it conflicts with a specific arm.

### 15.2 Add slot energy balance loss

The repo now needs a loss that controls not just direction, but contribution magnitude.

Define a small penalty encouraging per-slot energy balance, for example by penalizing the coefficient of variation of slot norms or per-slot variance.

Purpose:

- orthogonal slots can still be semantically useless if 98% of the energy lives in one shared component,
- this loss encourages the Writer to actually use multiple slots.

### 15.3 Add common-mode penalty

Add a penalty on the mean slot vector norm relative to total slot energy.

Purpose:

- directly discourages "same large vector + tiny residuals"
- aligns with the actual failure observed in the logs

### 15.4 Loss rollout rule

Do **not** activate all new losses at once in the first run matrix.

Recommended sequence:

- first run W0 / W1 / W2 with the smallest extra change set,
- only if top1/top2 remains very high, activate `slot_energy_balance_loss` and `common_mode_penalty` on the best architecture arm.

This preserves attribution.

## 16. V1 success criteria

### 16.1 Diagnostic success

Any Writer arm counts as diagnostically successful if all of the following improve relative to W0:

- `memory_long_top1_top2_ratio` falls by at least **3x**
- `reader_readout_pairwise_cosine_mean` decreases meaningfully
- `reader_readout_effective_rank` rises above the RG-2 / RG-3 control level

### 16.2 Medium success

A Writer arm reaches medium success if:

- `memory_long_top1_top2_ratio < 15`
- `reader_readout_pairwise_cosine_mean < 0.95`
- `reader_readout_effective_rank > 2.0`
- early collapse is delayed relative to RG-2 partition control

### 16.3 Strong success

A Writer arm reaches strong success if:

- `memory_long_top1_top2_ratio < 10`
- `memory_long_centered_effective_rank >= 3`
- `reader_readout_pairwise_cosine_mean < 0.90`
- `reader_readout_effective_rank > 2.5`
- FEVER selection becomes alive or nearly alive

### 16.4 Hard stop for V1

If all Writer arms keep:

- `memory_long_top1_top2_ratio > 30`
- `reader_readout_pairwise_cosine_mean > 0.98`

then the Writer remains the primary blocker and Phase V2 should be run only with the best least-bad Writer arm, without pretending the Writer problem is solved.

---

# Phase V2 — Mandatory micro-receiver LoRA

## 17. Why LoRA must enter now

This is the biggest planning change relative to `PLANv2.md`.

The reason is not fashion. It is attribution.

RG-3 has already shown that after the local Reader/Fuser rescue line stalls, the bridge is receiving too little useful gradient. If the receiver never learns to slightly reinterpret the injected deep-prefix states, then the memory-side modules may remain trapped in an unreadable geometry even after the Writer improves.

So the next question is no longer:

> "Should receiver PEFT ever be allowed?"

It is:

> "What is the smallest possible receiver-side LoRA that can test whether unreadability of the frozen receiver is the remaining bottleneck?"

## 18. MemGen comparison and the design constraint

The user requirement is clear: LoRA must be added, but it must be **lighter than MemGen's LoRA usage**.

That is a good constraint.

MemGen uses LoRA more broadly as part of its memory trigger and memory weaver formulation, with a separate trigger/weaver structure and LoRA-based memory synthesis attached to the frozen reasoner. This plan will not do that.

### 18.1 Our micro-LoRA must be lighter than MemGen in four ways

1. **No separate trigger module**
2. **No second LoRA-based memory weaver attached as a full subsystem**
3. **No broad all-layer receiver adaptation**
4. **No rank sweep beyond what is needed to prove or disprove the bottleneck**

### 18.2 Operational definition of “lighter than MemGen” for this project

Canonical micro-LoRA in this plan means:

- only `k_proj` and `v_proj`
- only on a very small subset of decoder layers nearest the injection path
- default rank `r = 2`
- no `q_proj`, no `o_proj`, no MLP by default
- no trigger learning path
- no RL stage
- no second policy module

This is intentionally minimal.

## 19. Canonical micro-LoRA design

### 19.1 Target layers

The current sparse deep-prefix injection uses five layers:

- `0`, `7`, `14`, `21`, `27`

Canonical micro-LoRA should begin even narrower:

- **primary arm**: target only `14`, `21`, `27`
- **expanded arm** (only if needed): target all five injection layers

Rationale:

- the later injection-adjacent layers are the most plausible place where a tiny receiver adaptation can improve how injected prefix states are interpreted,
- this keeps the parameter budget extremely low.

### 19.2 Target modules

Default target modules:

- `self_attn.k_proj`
- `self_attn.v_proj`

Do not target initially:

- `q_proj`
- `o_proj`
- MLP projections

Reason:

- `k_proj`/`v_proj` are the most direct interface for how injected deep-prefix states become readable through attention/cache dynamics,
- this is the smallest meaningful lever.

### 19.3 Rank and scale

Canonical start:

- `rank = 2`
- `alpha = 4`
- `dropout = 0.0`

Fallback only if there is evidence of under-capacity:

- `rank = 4`
- same target modules

Do not begin with `rank >= 8`.

### 19.4 Initialization

- standard LoRA zero-init on the up projection
- no warm-start
- receiver backbone weights remain frozen
- only the injected LoRA matrices are trainable on the receiver side

## 20. Required V2 run matrix

At minimum, on FEVER, run the following.

### Arm L0 — Best Writer arm, frozen receiver

This is the frozen control.

### Arm L1 — Best Writer arm + micro-LoRA (`r=2`, layers 14/21/27, `k_proj` + `v_proj`)

This is the canonical new arm.

### Arm L2 — Best Writer arm + expanded micro-LoRA (`r=2`, layers 0/7/14/21/27, `k_proj` + `v_proj`)

Run this only if L1 increases gradient flow but still underperforms in capability.

### Arm L3 — Best Writer arm + micro-LoRA (`r=4`, same layer scope as the best of L1/L2)

Run only if `r=2` shows real signal but appears under-capacity.

### Arm L4 — Optional ablation: old Writer + canonical micro-LoRA

This is useful if the team needs to separate:

- "LoRA alone helped"

from

- "Writer fix + LoRA together helped."

This arm is informative but not mandatory in the first batch.

## 21. Optimizer policy for V2

Micro-LoRA must be kept subordinate to the memory path.

Recommended default:

- support encoder / Writer / Reader / Fuser LR: keep the current diagnostic family scale
- projector LR: keep current scale
- receiver LoRA LR: **lower** than the projector LR
- receiver LoRA weight decay: `0.0`

Reason:

- the LoRA should open a narrow channel for readability,
- not overpower the memory-side search.

## 22. V2 success criteria

### 22.1 Diagnostic success

V2 is diagnostically successful if, relative to L0:

- median `grad_norm_reader` and `grad_norm_fuser` over early steps increase by at least **10x**, or
- collapse onset is delayed, or
- selection behavior becomes materially more alive.

### 22.2 Medium success

V2 reaches medium success if:

- geometry metrics from V1 remain improved,
- LoRA arm outperforms the frozen control on FEVER selection / collapse timing,
- and the gain is not accompanied by obvious degeneration into label collapse.

### 22.3 Strong success

V2 reaches strong success if:

- the best LoRA arm passes FEVER selection / gate where the frozen control does not,
- or the best LoRA arm is the first two-level path that clearly outperforms the current `SL-8` control on a fair gate.

### 22.4 Hard stop for V2

If:

- Writer-side value metrics improve substantially,
- but none of L1/L2/L3 improves gradient flow or capability,

then the problem is no longer "just frozen receiver unreadability." At that point the team should suspect either:

- task misfit,
- budget misfit,
- or a deeper mismatch in how the deep-prefix interface is currently realized.

That would justify moving to V4 earlier and potentially revisiting the task harness before broadening receiver PEFT.

---

# Phase V3 — Restore non-forced Reader behavior

## 23. Why this phase comes after V1/V2, not before

Under `PLANv2`, the project tried to rescue Reader geometry first. That was the correct thing to test then.

But after RG-3, the order has changed:

- first fix Writer value diversity,
- then test minimal receiver readability,
- only then check whether a non-forced Reader can become healthy again.

## 24. Canonical V3 question

> If the Writer now emits more distinct slot values, and a tiny receiver LoRA opens the gradient path, does standard or competitive Reader behavior begin to specialize without hard partition?

That is the real test of whether the bridge is alive rather than merely scaffolded.

## 25. V3 matrix

Run this only on the best configuration emerging from V1/V2.

### R0 — masked partition control

Keep the diagnostic scaffold for comparison.

### R1 — standard Reader

Re-enable standard per-query softmax over slots.

### R2 — competitive Reader

Re-test slot-competition only **after** value diversity has improved.

Why re-test competitive now:

- earlier it failed because attention competition cannot create value diversity that does not exist,
- after V1/V2 it may become useful rather than empty.

## 26. V3 success criteria

### Geometry success

Any non-forced Reader arm is geometry-alive if:

- `reader_attention_pairwise_cosine_mean < 0.8`
- `reader_attention_entropy_mean` is materially below the uniform baseline
- `reader_readout_pairwise_cosine_mean < 0.9`
- `reader_readout_effective_rank > 2.0`

### Capability success

A non-forced Reader arm should also show no earlier collapse than the partition control and ideally better selection behavior.

### Stop rule

If partition remains strictly necessary even after V1/V2, that is acceptable for the moment. It means the bridge is not yet self-specializing, but it may still be functionally useful.

Do not block the entire project on fully spontaneous Reader specialization if the partitioned bridge becomes functionally alive across tasks.

---

# Phase V4 — Multi-task bridge validation (mandatory from this plan onward)

## 27. Why this phase is now mandatory

The user requirement is explicit: not FEVER-only.

More importantly, the project has reached the point where FEVER-only results can no longer reliably tell whether the architecture is truly improving or merely overfitting to a narrow substrate.

So from this plan onward, every serious architecture candidate must be checked on at least one non-FEVER task already supported in the repo.

## 28. Canonical task set for the immediate next stage

The practical rule here is: prefer tasks that are **already wired in the repo**, so the team does not lose another week to dataset onboarding.

### 28.1 FEVER — keep as the continuity/control substrate

Role:

- cheapest continuity with all active diagnostics
- easiest place to compare against the historical line
- not sufficient on its own

### 28.2 NarrativeQA real-smoke — make this the first non-FEVER task

Reason:

- it already exists in the repo,
- it stresses integrative narrative comprehension rather than shallow label classification,
- it is more aligned with the eventual long-context / compressed-memory story than FEVER alone.

### 28.3 GSM8K real-smoke — make this the second non-FEVER task

Reason:

- it already exists in the repo,
- it covers the math side of the later CDMI story,
- it checks whether the bridge has any value for reasoning-like tasks beyond fact verification.

### 28.4 Story Cloze — keep as a stress-only task, not a primary gate

Reason:

- useful historically, but too fragile to become the architecture-deciding benchmark at this stage.

### 28.5 BoolQ / HotpotQA — optional future intake, not immediate blockers

If the team later wants a cheaper entailment-like short-context control or a more explicit multi-hop QA task, these are good candidates.

But they should not delay the immediate next hop because they are not yet part of the current low-friction active harness.

## 29. V4 execution order

### V4A — FEVER

Run the best V1/V2 configuration first, because it is cheapest and the current diagnostics are richest.

### V4B — NarrativeQA real-smoke

Run the same architecture family without opening a giant sweep.

Goal:

- check whether the bridge shows any semantically meaningful improvement on an integrative reading task.

### V4C — GSM8K real-smoke

Run the same architecture family again, keeping settings as close as practical.

Goal:

- check whether the bridge has any sign of carrying structured information useful beyond FEVER.

## 30. V4 success criteria

### 30.1 Weak success

- FEVER improves clearly,
- and at least one of NarrativeQA or GSM8K shows nontrivial gain over its zero-memory / frozen control.

### 30.2 Medium success

- FEVER reaches at least medium success,
- and **both** NarrativeQA and GSM8K show at least directional positive evidence,
- with no gross collapse pattern.

### 30.3 Strong success

- FEVER passes a real gate,
- NarrativeQA shows meaningful improvement,
- GSM8K also shows improvement,
- and the best configuration is consistent enough to justify 3-seed confirmation.

### 30.4 What does *not* count as success

- FEVER-only gain with no non-FEVER signal
- one-off non-FEVER noise without bridge metrics
- better `M_short` rank but no task-level gain anywhere

---

# Phase V5 — Qwen3-8B confirmation

## 31. When Qwen3-8B becomes active again

Do **not** jump to Qwen3-8B immediately.

Qwen3-8B should re-enter only after the Qwen2.5 line has at least **medium success** across FEVER and at least one non-FEVER task.

Reason:

- Qwen3 is needed for the final paper,
- but it is not the right place to discover whether the bridge logic is broken.

## 32. Minimal Qwen3 confirmation matrix

Use only the best stabilized family from V1-V4.

Recommended minimum:

- best frozen-receiver arm
- best micro-LoRA arm
- on NarrativeQA real-smoke and GSM8K real-smoke
- optionally FEVER as a continuity anchor

The purpose is not full ablation. It is to confirm that the stabilized family is not a Qwen2.5-only artifact.

---

# Phase V6 — Re-open Stage B/C refresh and CDMI only after multi-task medium success

## 33. Why later workstreams remain locked for now

The repo's macro plan is still correct:

- do not reopen transfer evidence refresh too early,
- do not claim CDMI before the bridge is alive on more than FEVER,
- do not treat broad receiver adaptation as the whole answer.

So V6 only opens after V4 reaches at least medium success.

## 34. What V6 unlocks

### 34.1 Transfer / Stage B/C refresh

At this point the team can honestly revisit:

- fixed Writer + query adaptation
- same-task vs cross-task adaptation curves
- `q_only / w_only / w_plus_q` style refresh if still relevant in the current harness

### 34.2 CDMI experiments

The canonical later target remains:

- Math + Code vs Math + Narrative

and the repo already treats CDMI as paper-critical.

### 34.3 Efficiency / cost table

Once micro-LoRA is introduced, the final paper should include a proper performance-cost comparison against:

- memory-only path,
- micro-LoRA-enhanced path,
- baseline PEFT baselines.

---

## 35. Exact file-by-file implementation plan

### 35.1 `src/memtotal/models/memory.py`

#### Writer changes

Add a new configuration field:

- `slot_conditioning_mode`
  - `shared_add`
  - `shared_add_scaled`
  - `slot_query_only`
  - `slot_query_small_shared`
  - optionally `post_write_mean_subtract`

Add a new scalar field:

- `shared_state_scale`

Add helper outputs/metrics support for:

- common-mode vector
- centered slots
- per-slot energies / norms

Preserve checkpoint compatibility where possible:

- old checkpoints without `shared_state_scale` should still load with sensible defaults
- new modes should degrade gracefully when loading older states

#### Reader changes

No major new architectural work in the first step, beyond exposing any tensors needed for value-projection diagnostics.

#### Fuser changes

No major new architecture change in the first step. The current linear Fuser stays as the diagnostic scaffold.

### 35.2 `src/memtotal/models/backbone.py`

Add minimal in-repo receiver-side LoRA support.

Recommended implementation style:

- create a tiny wrapper around target `nn.Linear` modules,
- do not add a heavyweight external dependency if avoidable,
- allow targeting by module path, layer index, and projection type.

Required features:

- target only selected layers
- target only selected projection names
- zero-init standard LoRA behavior
- exact trainable parameter count logging
- easy freeze/unfreeze
- save/load compatibility in checkpoints

### 35.3 `src/memtotal/training/m4_shared_injection.py`

Required additions:

- new Writer mode plumbing
- new Writer penalties
  - `writer_slot_energy_balance_loss`
  - `writer_common_mode_penalty`
- receiver LoRA construction / registration
- optimizer param-group support for receiver LoRA
- gradient logging for receiver LoRA
- updated summary outputs for V1/V2/V3/V4

### 35.4 `src/memtotal/analysis/m4_shared_injection.py`

Add:

- common-mode summary fields
- top1/top2 ratio summary fields
- centered-rank summary fields
- value-projection diversity summaries
- pseudo-diversity flags
- micro-LoRA parameter budget summaries

### 35.5 Config files

Add method configs such as:

- `configs/method/tl_writer_shared_scaled.yaml`
- `configs/method/tl_writer_slot_query_only.yaml`
- `configs/method/tl_writer_slot_query_small_shared.yaml`
- `configs/method/tl_receiver_microlora_kv_r2_late3.yaml`
- `configs/method/tl_receiver_microlora_kv_r2_all5.yaml`
- `configs/method/tl_receiver_microlora_kv_r4.yaml`

Add experiment configs for:

- FEVER V1/V2
- NarrativeQA real-smoke
- GSM8K real-smoke
- Qwen3 confirmation

### 35.6 Scripts

Add at least:

- `scripts/run_tl_writer_value_diversification_fever_qwen25.sh`
- `scripts/update_tl_writer_value_summary.py`
- `scripts/run_tl_micro_lora_bridge_fever_qwen25.sh`
- `scripts/update_tl_micro_lora_summary.py`
- `scripts/run_tl_bridge_multitask_qwen25.sh`
- `scripts/update_tl_multitask_summary.py`
- later: `scripts/run_tl_bridge_qwen3_confirmation.sh`

### 35.7 Tests

Add tests for:

- Writer mode shape/output contracts
- old checkpoint compatibility with new Writer fields
- common-mode metric calculations
- trainable param counts for micro-LoRA
- exact layer/module targeting for LoRA
- frozen backbone invariants outside targeted LoRA modules
- smoke run for one FEVER V1 arm and one V2 arm

---

## 36. Exact experimental order and stop rules

This is the concrete run order. Follow it strictly.

### Step 1 — Implement V0 instrumentation and validate it

Acceptance:

- new metrics appear in train logs and review summaries
- unit tests pass

### Step 2 — Run V1 architecture-first Writer matrix on FEVER

Minimum matrix:

- W0: `shared_add`
- W1: `shared_add_scaled(alpha=0.02)`
- W2: `slot_query_only`

All under:

- partitioned Reader
- H=4, K=4
- linear Fuser
- frozen receiver

Decision after Step 2:

- if W2 or W1 clearly improves value diversity, choose the best arm and continue
- if none helps enough, add Writer penalties and rerun only the best candidate architecture

### Step 3 — If needed, run V1 penalty refinement on the best Writer arm

Add:

- common-mode penalty
- slot energy balance loss

Do not run a giant grid.

### Step 4 — Run V2 micro-LoRA FEVER matrix

Minimum:

- L0 frozen receiver
- L1 micro-LoRA rank 2 on layers 14/21/27, `k_proj` + `v_proj`

Only expand to L2/L3 if L1 produces partial evidence.

Decision after Step 4:

- if micro-LoRA clearly improves gradient routing and capability, it becomes part of the canonical next family
- if not, do not broaden LoRA blindly yet; first continue to V4 to test task sensitivity

### Step 5 — Run V3 non-forced Reader restoration on the best family

Minimum:

- partition control
- standard Reader
- competitive Reader

Decision after Step 5:

- if standard or competitive becomes healthy, prefer the simplest one
- if partition remains necessary, keep partition for now and do not block later validation

### Step 6 — Run V4 multi-task validation on Qwen2.5

Order:

1. FEVER
2. NarrativeQA real-smoke
3. GSM8K real-smoke

Decision after Step 6:

- if medium success across FEVER + non-FEVER exists, proceed to Qwen3 confirmation
- if only FEVER improves, do not reopen transfer/CDMI yet

### Step 7 — Run V5 Qwen3 confirmation

Keep this minimal and hypothesis-driven.

### Step 8 — Only after V4/V5 medium success, reopen Stage B/C refresh and CDMI

---

## 37. Concrete gate definitions

The project needs more precise gates than before.

### 37.1 Value-Diversity Gate (VDG)

A candidate passes VDG if all of the following hold on FEVER under the partition scaffold:

- `memory_long_top1_top2_ratio < 15`
- `memory_long_centered_effective_rank >= 3`
- `reader_readout_pairwise_cosine_mean < 0.95`
- `reader_readout_effective_rank > 2.0`

### 37.2 Gradient-Live Gate (GLG)

A candidate passes GLG if, relative to its frozen-receiver matched control:

- early-step median `grad_norm_reader` improves by at least `10x`, or
- early-step median `grad_norm_fuser` improves by at least `10x`, or
- the LoRA arm materially delays collapse / improves selection.

### 37.3 Bridge-Alive Gate (BAG)

A candidate is considered bridge-alive if:

- it passes VDG,
- it passes GLG or shows equivalent capability improvement,
- and collapse is no longer immediate (`step0/2` style collapse is not acceptable).

### 37.4 Multi-Task Medium Success

A configuration reaches multi-task medium success if:

- FEVER reaches at least bridge-alive or near-gate behavior,
- NarrativeQA shows positive task-native movement,
- GSM8K shows positive task-native movement,
- and none of those gains depends on obviously broken geometry.

### 37.5 Multi-Task Strong Success

A configuration reaches strong success if:

- it passes FEVER strongly,
- and shows meaningful positive results on both NarrativeQA and GSM8K,
- and survives 3-seed confirmation on Qwen2.5.

---

## 38. Recommended default hyperparameters for the next phase

These are defaults, not a sweep invitation.

### 38.1 Writer defaults

- keep current slot-basis warm-start machinery
- `shared_state_scale = 0.02` for `shared_add_scaled`
- keep `support_query_residual_scale` conservative
- if adding new penalties:
  - start them small
  - do not exceed the main task loss scale

### 38.2 Receiver micro-LoRA defaults

- layers: `14, 21, 27`
- target modules: `k_proj`, `v_proj`
- rank: `2`
- alpha: `4`
- dropout: `0.0`
- no weight decay
- lower LR than projector

### 38.3 Budget defaults

- keep the same base training-step family initially for comparability
- allow one `64-step` audit only **after** geometry improves; do not use longer steps to hide a broken bridge

### 38.4 Seed policy

- V0/V1/V2/V3 diagnostics: 1 seed is acceptable
- first multi-task confirmation: at least 1 seed per task
- once a family reaches medium success: rerun **3 seeds** on Qwen2.5
- final paper curves/tables follow the repo's stricter seed policy

---

## 39. Risks and anti-patterns specific to this phase

### 39.1 Anti-pattern: broad receiver LoRA immediately

Do not jump to all-layer LoRA, high-rank LoRA, or MLP LoRA.

That would destroy attribution and make the paper story weaker.

### 39.2 Anti-pattern: confusing projection diversity with semantic diversity

High `M_short` rank alone is not evidence of success.

The readout metrics must improve too.

### 39.3 Anti-pattern: repeating Reader-local loss stacking

The repo has already learned enough from RG-2/RG-3.

### 39.4 Anti-pattern: FEVER-only claim inflation

A FEVER-only rescue from now on is not a project-level rescue.

### 39.5 Anti-pattern: adding a brand-new dataset before using the repo-supported ones

Do not delay the next hop by onboarding a new benchmark if NarrativeQA and GSM8K already exist in the repo.

### 39.6 Anti-pattern: using LoRA to bypass the memory question entirely

Micro-LoRA is here to test readability and unlock gradients, not to replace the memory system.

---

## 40. Deliverables required from PLANv3

### 40.1 Code deliverables

- Writer conditioning modes
- Writer value/common-mode losses
- receiver micro-LoRA support
- expanded geometry logging
- multitask runner support for the selected repo-supported tasks

### 40.2 Artifact deliverables

At minimum:

- V0 summary
- V1 Writer-value summary
- V2 micro-LoRA summary
- V3 Reader-restoration summary
- V4 multitask summary
- V5 Qwen3 confirmation summary

### 40.3 Documentation deliverables

Update:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/TODO_LIST.md`
- `docs/tech-debt-tracker.md`
- active exec-plan under `docs/exec-plans/active/`
- refreshed review bundle

### 40.4 Validation deliverables

- targeted unit tests
- smoke commands for new Writer/LoRA paths
- trainable-parameter budget logs
- clean worktree after each phase

---

## 41. Recommended immediate command-level work sequence

The exact filenames can change, but the work order should not.

```bash
# 1) instrumentation first
python -m unittest discover -s tests -v

# 2) Writer value diversification on FEVER
bash scripts/run_tl_writer_value_diversification_fever_qwen25.sh
python scripts/update_tl_writer_value_summary.py

# 3) micro-LoRA on FEVER
bash scripts/run_tl_micro_lora_bridge_fever_qwen25.sh
python scripts/update_tl_micro_lora_summary.py

# 4) non-forced Reader restoration on the best family
bash scripts/run_tl_reader_restore_fever_qwen25.sh
python scripts/update_tl_reader_restore_summary.py

# 5) multitask validation
bash scripts/run_tl_bridge_multitask_qwen25.sh
python scripts/update_tl_multitask_summary.py

# 6) only after success, Qwen3 confirmation
bash scripts/run_tl_bridge_qwen3_confirmation.sh
python scripts/update_tl_qwen3_confirmation_summary.py
```

---

## 42. Final decision table

### Case A — Writer improves, frozen receiver still fails, micro-LoRA succeeds

Interpretation:

- Writer homogenization was real,
- receiver unreadability was also real,
- canonical next family should include micro-LoRA.

Action:

- proceed to multi-task validation with the micro-LoRA family.

### Case B — Writer improves and frozen receiver already works

Interpretation:

- the dominant blocker was Writer value homogenization,
- micro-LoRA is optional or only a confirmation ablation.

Action:

- proceed to multi-task validation with frozen receiver as the canonical path; keep micro-LoRA as a helpful comparison.

### Case C — Writer improves, micro-LoRA improves gradients, but tasks remain dead everywhere

Interpretation:

- the bridge is still not semantically useful under the current task/budget mix.

Action:

- do not broaden LoRA blindly,
- move to multi-task smoke checks to see whether FEVER is misleading,
- if all tasks remain dead, revisit budget/task selection before further architecture branching.

### Case D — Writer does not improve at all

Interpretation:

- the main bottleneck remains the Writer architecture.

Action:

- stay memory-side,
- do not use receiver LoRA success as an excuse to ignore the Writer problem.

### Case E — Only FEVER improves

Interpretation:

- architecture is still too task-specific.

Action:

- do not reopen transfer / CDMI yet,
- continue non-FEVER bridge validation first.

---

## 43. Blunt project-level assessment

The project is **not** in a hopeless state.

But it is at a point where another round of FEVER-only, frozen-receiver, local Reader rescue would very likely be wasted effort.

The most likely route to real progress now is:

1. fix the Writer's value-space common-mode problem,
2. allow the smallest possible receiver LoRA to test whether the receiver was starving the bridge,
3. prove the resulting family is not FEVER-only by running on repo-supported non-FEVER tasks,
4. only then return to the broader paper program.

This keeps the project honest, keeps attribution relatively clean, and satisfies both scientific and practical constraints.

---

## 44. Reference notes that informed this plan

These references are not here as decoration. Each one supports a specific design decision.

### R1. Prefix-Tuning

Li, Xiang Lisa, and Percy Liang. 2021.
**Prefix-Tuning: Optimizing Continuous Prompts for Generation.**
https://aclanthology.org/2021.acl-long.353/

Use here:

- supports the idea that frozen backbones can consume learned prefix-like states,
- but also reminds us that such routes can be fragile and task-dependent.

### R2. P-Tuning v2

Liu et al. 2022.
**P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks.**
https://arxiv.org/abs/2110.07602

Use here:

- relevant to deep prompt/prefix behavior on NLU-style tasks,
- supports the view that prompt-side adaptation can work, but the optimization/interface matter greatly.

### R3. LoRA

Hu et al. 2021.
**LoRA: Low-Rank Adaptation of Large Language Models.**
https://arxiv.org/abs/2106.09685

Use here:

- basis for the mandatory tiny receiver-side LoRA,
- justifies keeping receiver adaptation extremely parameter-efficient.

### R4. IA3 / T-Few

Liu et al. 2022.
**Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning.**
https://arxiv.org/abs/2205.05638

Use here:

- useful as a later optional comparison because it is even lighter than LoRA,
- but LoRA remains mandatory in this plan because of the user's requirement.

### R5. Slot Attention

Locatello et al. 2020.
**Object-Centric Learning with Slot Attention.**
https://arxiv.org/abs/2006.15055

Use here:

- explains why competitive binding can break attention symmetry,
- but current results also show its limit: competition cannot invent value diversity that is not present.

### R6. Perceiver IO

Jaegle et al. 2022.
**Perceiver IO: A General Architecture for Structured Inputs & Outputs.**
https://arxiv.org/abs/2107.14795

Use here:

- supports query-based readout from a latent array,
- useful for interpreting Reader/Fuser design and why fixed queries need meaningful latent content to read.

### R7. Flamingo / Perceiver Resampler

Alayrac et al. 2022.
**Flamingo: a Visual Language Model for Few-Shot Learning.**
https://arxiv.org/abs/2204.14198

Use here:

- relevant to the idea of compressing richer external state into a fixed number of latents for a mostly frozen backbone,
- supports the broader architectural legitimacy of a learned resampling bridge.

### R8. Fine-Tuning can Distort Pretrained Features

Kumar et al. 2022.
**Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.**
https://arxiv.org/abs/2202.10054

Use here:

- supports the earlier observation that naive continuation can destroy a useful warm-start manifold,
- reinforces the caution against brute-force objective loops.

### R9. MemGen

**MemGen: Weaving Generative Latent Memory for Self-Evolving Agents.**
https://arxiv.org/abs/2509.24704

Use here:

- relevant comparison point for LoRA-based memory mechanisms,
- motivates the constraint that our receiver LoRA should remain much lighter than MemGen's trigger/weaver formulation.

### R10. NarrativeQA

Kočiský et al. 2017.
**The NarrativeQA Reading Comprehension Challenge.**
https://arxiv.org/abs/1712.07040

Use here:

- supports choosing a non-FEVER integrative narrative task that actually stresses document-level understanding.

### R11. GSM8K

Cobbe et al. 2021.
**Training Verifiers to Solve Math Word Problems.**
https://arxiv.org/abs/2110.14168

Use here:

- supports choosing a math reasoning benchmark already aligned with the later CDMI math side.

### R12. BoolQ

Clark et al. 2019.
**BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions.**
https://arxiv.org/abs/1905.10044

Use here:

- good optional future near-domain control task if the team later wants a simpler non-FEVER NLU benchmark.

### R13. HotpotQA

Yang et al. 2018.
**HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.**
https://arxiv.org/abs/1809.09600

Use here:

- useful optional future intake if the team later wants an explicit multi-hop QA task.

---

## 45. Final operational statement

The next phase should **not** be described as:

> "more Reader rescue"

and it should **not** be described as:

> "full receiver fallback"

The correct description is:

> **post-RG3 Workstream B continuation through Writer Value Diversification, tiny diagnostic micro-LoRA on the receiver, and immediate multi-task bridge validation.**

That is the narrowest plan that is still logically complete, technically justified, and likely to move the project forward from its real current bottleneck.
