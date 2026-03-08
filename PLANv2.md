# PLANv2.md — Narrow Execution Plan for Workstream B / Failure Mode B-1

**Project:** MemTOTAL  
**Date:** 2026-03-08  
**Status:** Active narrow plan  
**Language:** English  
**Role of this file:** operational companion to `PLAN.md`, specialized for the current next hop

---

## 0. Purpose and relationship to the existing plan

This document does **not** replace the existing top-level `PLAN.md`.

`PLAN.md` remains the macro-order authority for the whole project:

1. stabilize the FEVER shared-injection substrate,
2. bring the true two-level path alive,
3. refresh Stage B/C transfer evidence,
4. run CDMI experiments,
5. only then consider receiver-side fallback if diagnostics justify it.

That macro order is still correct.

What changed is that the project has now completed the following real stages:

- Workstream A / single-level objective rescue has been pushed far enough to yield a **negative but informative** answer.
- Workstream B / TL-PoC has already been implemented and executed in the active FEVER harness.
- Two follow-up rescues (`TL bridge rescue`, `TL slot-basis rescue`) have already narrowed the blocker much more precisely than the original `PLAN.md` could.

Therefore the repository now needs a **narrower, deeper, directly executable plan** for the current live blocker:

> **How to rescue Reader/Fuser query-side readout geometry in the two-level FEVER bridge, so that H=4 actually specializes, Reader stops reading the 8 long slots almost uniformly, and M_short stops collapsing to a low-rank near-average.**

This file is that plan.

---

## 1. Scope boundary: what this plan covers and what it does not

### 1.1 Covered by this plan

This plan covers exactly the current next-hop work:

- **Workstream B remains active**.
- The immediate target is **Failure mode B-1**.
- The live problem is now **Reader/Fuser readout geometry**, not Writer existence, not single-level objective design, and not receiver adaptation.
- The primary benchmark remains the current **FEVER active harness** because that is where the two-level bridge is already fully instrumented.

### 1.2 Explicitly not covered by this plan

This plan does **not** authorize the following yet:

- Step 6 — refresh Stage B/C transfer evidence
- Step 7 — CDMI experiments
- Step 8 — receiver fallback
- a new global single-level objective cycle (`M5.4`, `M5.5`, etc.)
- a broad backbone switch
- a large benchmark expansion
- paper-writing-level narrative reshaping

### 1.3 Why the boundary is so strict

The current evidence is already strong enough to impose discipline:

- the project no longer needs to ask whether the two-level path exists in code,
- no longer needs to ask whether `M_long` is always rank-1 noise,
- and no longer needs to ask whether the single-level dense-teacher objective simply “needs one more try.”

The correct current question is narrower and more structural:

> **Why does a healthier `M_long` still get read into an almost uniform, weakly specialized, low-rank `M_short`?**

If that question is not answered first, every downstream step becomes ambiguous.

---

## 2. Inputs reviewed for this plan

This plan is based on a direct cross-read of the current repository, the uploaded repo bundle, the uploaded docs bundle, the existing `PLAN.md`, and the newest result artifacts.

### 2.1 Repo/code paths reviewed

The following code paths are load-bearing for the present plan:

- `src/memtotal/models/memory.py`
- `src/memtotal/training/m4_shared_injection.py`
- `src/memtotal/pipeline.py`
- `src/memtotal/training/m3.py`
- `configs/method/*.yaml` relevant to the two-level FEVER path
- `configs/exp/tl_*_fever_qwen25*.yaml`
- `scripts/run_tl_poc_fever_qwen25.sh`
- `scripts/run_tl_bridge_rescue_fever_qwen25.sh`
- `scripts/run_tl_slot_basis_rescue_fever_qwen25.sh`
- `scripts/update_tl_poc_summary.py`
- `scripts/update_tl_bridge_rescue_summary.py`
- `scripts/update_tl_slot_basis_summary.py`

### 2.2 Docs reviewed

The following docs were explicitly cross-checked:

- `PLAN.md`
- `docs/MAIN_IDEA.md`
- `docs/TODO_LIST.md`
- `docs/ARCHITECTURE.md`
- `docs/EXPERIMENTS_INFO.md`
- `docs/briefs/20260307-m4-shared-injection-brief.md`
- `docs/tech-debt-tracker.md`
- `docs/exec-plans/active/20260306-initial-bootstrap.md`

### 2.3 Result artifacts reviewed

The following artifact families drive this plan:

- `results/generated/review/m5-fever-dense-teacher-qwen25/`
- `results/generated/review/tl-poc-fever-qwen25/`
- `results/generated/review/tl-bridge-rescue-fever-qwen25/`
- `results/generated/review/tl-slot-basis-rescue-fever-qwen25/`
- the corresponding `runs/review/.../train_events.json` payloads
- the corresponding `reader_query_diagnostics.csv` payloads
- `selection.json`, `run-summary.json`, and top-level summary JSON/MD files

---

## 3. Locked current state

This section freezes the exact project state that this plan assumes.

### 3.1 Workstream A is effectively settled

The repo has already run the current best single-level objective family far enough to justify stopping.

#### Settled facts

- `M5.3 dense teacher` was not a fake or dormant experiment.
- Dense teacher actually activated during training.
- Yet canonical dense teacher still did **not** beat safe-hinge control.
- Therefore the project should **not** continue iterating on single-level FEVER objective repair as the default next move.

### 3.2 Workstream B has already started and already produced real narrowing

The repo has already executed the true two-level FEVER path in the active harness.

#### Completed real runs

- `TL-PoC`
- `TL bridge rescue`
- `TL slot-basis rescue`

#### What these runs now jointly establish

1. the two-level path is genuinely wired into the active FEVER harness;
2. `M_long` write-side geometry was indeed a problem;
3. direct writer-side basis/factorization rescue can improve `M_long`;
4. but the bridge still does not pass selection or primary gate;
5. therefore the live blocker has narrowed from “can the two-level path exist?” to “why is the two-level readout still almost uniform and low-specialization?”

### 3.3 The new blocker is now specific

The current live blocker is no longer “Writer rank-1 collapse” in the broad sense.

It is now:

> **Failure mode B-1, narrowed to query-side readout geometry:**
> a healthier `M_long` is still being converted by `Reader/Fuser` into an almost uniform, weakly specialized, low-rank `M_short`.

That is the sole focus of this plan.

---

## 4. Independent diagnosis of the current failure

This section gives the current best bottom-up explanation of the newest results.

It intentionally goes beyond merely restating the existing repo docs.

---

### 4.1 What the newest results really prove

The newest results prove **three things at once**.

#### A. Writer-side geometry is no longer the only explanation

`TL slot-basis rescue` meaningfully improved `M_long` geometry:

- final `memory_long_effective_rank` rose to about **1.61**,
- writer slot basis pairwise cosine was pushed to approximately **0**,
- and this did not require touching the receiver.

So the hypothesis “the two-level path fails only because Writer never escapes rank-1” is no longer sufficient.

#### B. Reader specialization is still absent from the first moment that matters

Even after the writer-side rescue:

- `reader_attention_pairwise_cosine_mean` remains effectively **1.0**,
- `reader_attention_entropy_mean` remains near **ln(8)**,
- `memory_short_effective_rank` remains only around **1.2**,
- and collapse still appears extremely early.

This means the project is **not** one tiny regularization tweak away from success.

The geometric bottleneck has moved downstream from the Writer into the **Reader/Fuser interface**.

#### C. The current failure begins too early to be explained only by late training drift

From the current `train_events.json` traces, the relevant gradients and geometry change extremely fast:

- the Reader/Fuser side sees meaningful gradients at the beginning,
- but those gradients collapse sharply within the first few steps,
- while attention stays almost uniform,
- meaning the system never really establishes a specialized readout regime before the task dynamics flatten it.

This strongly suggests the problem is not merely “the right signal arrives later but gets washed out.”

Instead, the system likely starts from a **badly symmetric readout geometry** and never escapes it.

---

### 4.2 Root cause 1 — prompt-summary conditioning likely overwrites query identity

This is the most important new diagnosis produced by the code review.

In the current `MemoryReader.read(...)` implementation, conditioned queries are formed as:

```text
queries = base_queries
        + context_proj(prompt_summary)
        + optional candidate_state
```

Critically, `context_proj(prompt_summary)` is the **same vector for every query** in the batch.

So the current design does this:

- start with `H` small learned query vectors,
- add the **same context vector** to all of them,
- then run standard cross-attention.

In the current FEVER path, that is dangerous for a very specific reason:

1. the base query vectors are small learned parameters,
2. the prompt summary comes from backbone hidden states whose norm is much larger,
3. the same shared context offset is added to every query,
4. therefore the conditioned queries can become almost parallel before attention even starts.

If this is happening, then **uniform/near-identical attention is not just a training failure**.
It is partly a consequence of the current architecture.

This is consistent with the observed facts:

- query pairwise behavior looks collapsed from the earliest snapshots,
- attention entropy is near uniform,
- H=4 behaves only marginally differently from H=1,
- diversity regularization on *attention outputs* does not rescue the situation.

### Why this matters

If this diagnosis is correct, then the current next move is **not** “more regularization.”

The current next move is:

> **stop allowing shared prompt conditioning to erase query identity.**

This is why the first rescue stage in this plan is centered on **context-off / context-gated** readout probes.

---

### 4.3 Root cause 2 — standard per-query softmax over slots has a bad symmetric fixed point here

The current Reader uses standard cross-attention:

- each query independently scores all slots,
- softmax is normalized **over slots**,
- all queries can simultaneously attend to all slots,
- there is no built-in competition that forces different queries to claim different subsets.

Under strong supervision and rich gradients, that can still specialize.

Under the present setting, however, the system has three properties:

1. frozen Qwen receiver,
2. narrow injected channel,
3. tiny-data / short-horizon FEVER pilot dynamics.

In this regime, standard cross-attention has an easy symmetric equilibrium:

- all queries read roughly the same mixture,
- attention stays high-entropy,
- downstream task loss is satisfied via label bias instead of informative readout structure.

This explains why:

- `reader_attention_pairwise_cosine_mean` sticks near the worst-case boundary,
- `reader_attention_entropy` stays at the uniform limit,
- and simply penalizing similar attention maps does not do much.

The present evidence therefore supports the following interpretation:

> **The current Reader does not merely need a stronger loss; it needs an explicit symmetry-breaking mechanism in the forward path.**

That is why this plan treats **competitive readout** as a mainline rescue, not a bonus idea.

---

### 4.4 Root cause 3 — the current H=4 → K=8 fuser is probably duplicating readouts

The current canonical two-level bridge variant is `TL-H4-K8`.

That means:

- Reader produces **4** readouts,
- Fuser then resamples them into **8** short slots.

This creates an underappreciated structural problem:

> The Fuser is being asked to *expand* 4 query outputs into 8 injected slots.

When the 4 readouts are already similar, the easiest solution is to produce 8 similar outputs.

This is exactly what the current metrics look like:

- `memory_short_pairwise_cosine_mean` stays extremely close to **1.0**,
- `memory_short_effective_rank` remains low,
- and the bridge does not become semantically alive.

This does **not** mean the Fuser is badly implemented.
It means the present canonical bridge is still confounded by an **upsampling regime**.

Therefore the immediate geometry rescue should first prefer:

- `K = H` (especially `H=4, K=4`),
- and a simpler Fuser (`linear` or identity-style grouped mapping),
- before returning to `H=4 -> K=8` bridge widening.

In other words:

> **We should first learn to read 4 distinct things, then worry about how to expand or bridge them into 8 injected slots.**

---

### 4.5 Root cause 4 — the gradients that could teach specialization vanish too early

The current train-event traces show a very important dynamic:

- Reader/Fuser gradients are non-trivial at the very beginning,
- but they collapse dramatically by around the first few steps,
- while attention remains near-uniform and M_short remains near-duplicated.

This matters because it changes the correct intervention style.

If the gradients that could teach specialization vanish by step 2–4, then the system cannot rely on “eventually learning good readout geometry” from the downstream task alone.

Instead, the system needs one or both of the following:

1. **architectural asymmetry from step 0**, and/or
2. **an early local bootstrap objective** that shapes Reader/Fuser before the task objective collapses the channel.

This is why this plan explicitly includes:

- forward-path symmetry breaking,
- a possibility of early reader bootstrap,
- and local readout-side auxiliary losses only as a second-line option.

---

### 4.6 Why the existing diversity regularization did not solve it

The current negative result for `TL bridge rescue` makes sense under the diagnosis above.

The diversity regularizers were applied to:

- `memory_long`,
- `memory_short`,
- Reader attention maps.

But the real collapse likely happens **before** these observables become useful levers:

- conditioned queries are already too similar,
- the attention operator itself has a symmetric equilibrium,
- the Fuser is already in an upsampling regime,
- and downstream gradients dry up quickly.

So the regularizers were not useless.
They were simply attached **too late in the causal chain**.

The right rescue should therefore move the intervention upstream to:

- query conditioning,
- attention competition,
- and fuser shape.

---

## 5. Adjudication of the critical objections

This section explicitly addresses the criticisms you surfaced and decides which ones now matter.

---

### 5.1 “You should use full token KL” — current verdict: not the next move

This criticism is important in spirit but wrong as the immediate prescription.

#### What is correct in the criticism

It is correct that an output-only objective can be too weak or too indirect when the real problem is internal geometry.

#### What is not correct for the current stage

At the current stage, the live bottleneck is no longer the single-level objective family.

The project already tested a denser teacher signal in Workstream A and got a negative answer good enough to stop there.
The current blocker is now **upstream of the reasoner decision**, inside Reader/Fuser geometry.

So using full token KL right now would have three drawbacks:

1. it adds a much costlier objective to the wrong stage,
2. it still supervises the system only indirectly through the end model,
3. it muddies the current failure analysis instead of sharpening it.

#### Final decision

**No full token KL in the current B-1 rescue.**

If teacher signals are revisited later, they should only come back **after** the two-level bridge is geometry-alive, and even then a local hidden-state or readout-space target would be more justified than a large token-level distillation objective.

---

### 5.2 “Plugging Reader/Fuser into the harness was too optimistic” — verdict: correct

This criticism is now strongly supported by the real data.

The two-level path was not wrong to try.
But the idea that the current Reader/Fuser would self-specialize simply by being plugged into the FEVER task objective was too optimistic.

That optimism has now been corrected by evidence.

The current plan therefore does **not** assume that “Reader/Fuser will learn their roles naturally if left alone.”

Instead, the plan treats Reader/Fuser as needing **explicit geometry rescue**.

---

### 5.3 “You should not have gone to two-level before fixing single-level” — verdict: partly wrong, now no longer useful

In hindsight, the project got something valuable from moving to the two-level path:

- it proved the bridge can be instrumented end-to-end,
- it separated Writer-side geometry from readout-side geometry,
- it narrowed the live blocker more precisely than more single-level objective work would have.

So the move to Workstream B was not wasted.

However, from now on the project must be stricter:

- no more broad structural changes without clear diagnostic purpose,
- one bottleneck family at a time,
- and no receiver move until memory-side evidence truly warrants it.

---

### 5.4 “Receiver freezing is too dogmatic” — verdict: still freeze for now

This criticism becomes valid only after the memory-side channel becomes healthy enough that the receiver is the dominant suspect.

That is **not yet the current state**.

At present we still have:

- no real H=4 specialization,
- near-uniform Reader attention,
- low-rank M_short,
- and no stable selection-passing two-level FEVER bridge.

So moving to receiver adaptation now would weaken causal interpretation.

#### Final decision

Keep the receiver frozen for the current plan.

Receiver fallback remains valid **later**, but only after B-1 is either solved or cleanly transformed into B-2.

---

## 6. Strategic decision for the current phase

### 6.1 The project stays in Workstream B

This is non-negotiable.

The project does **not** move to Stage B/C refresh, CDMI, or receiver adaptation until the two-level FEVER bridge reaches at least the level that `PLAN.md` called **medium success**.

### 6.2 The current main target is no longer writer rank

The next phase is **not** about:

- another writer objective rewrite,
- more writer-side diversity regularization,
- or revisiting dense teacher under single-level FEVER.

The current main target is:

> **query-side readout geometry and fuser-side anti-duplication.**

### 6.3 The current default substrate objective remains conservative

For comparability and stability, the default task objective for the current narrow plan should remain the least-collapsed substrate obtained from Workstream A and Workstream B:

- keep the current FEVER harness,
- keep the current safe-hinge control substrate,
- keep the current writer slot-basis rescue as the write-side base,
- and only mutate Reader/Fuser plus the smallest necessary local auxiliaries.

This prevents unnecessary confounding.

---

## 7. New fine-grained failure taxonomy inside B-1

The old `B-1` bucket is now too coarse.

For the next phase, split it into four submodes.

### B-1a — query-conditioning collapse

**Definition:** shared prompt-summary conditioning overwrites query individuality before attention.

**Primary signal:** conditioned queries become nearly parallel; turning off context significantly improves query diversity metrics.

### B-1b — attention symmetry collapse

**Definition:** even with preserved query identity, standard cross-attention still converges to a near-uniform multi-query readout.

**Primary signal:** `reader_attention_pairwise_cosine_mean` remains high and entropy remains near uniform despite context rescue.

### B-1c — fuser duplication collapse

**Definition:** Reader specialization appears, but the Fuser duplicates or re-averages the readouts into low-rank `M_short`.

**Primary signal:** readout diversity improves but `memory_short_pairwise_cosine_mean` remains near 1 and `memory_short_effective_rank` stays low.

### B-1d — residual write-side insufficiency

**Definition:** even after slot-basis rescue, `M_long` still does not contain enough usable multi-facet information for H=4 specialization.

**Primary signal:** hard partition or competitive Reader fails to raise `M_short` rank or downstream behavior at all.

This subtyping is essential because each submode requires a different fix.

---

## 8. Invariants for the current rescue cycle

To keep this rescue interpretable, the following must remain fixed in the initial rounds unless explicitly stated otherwise.

### 8.1 Backbone and task invariants

Keep fixed initially:

- backbone: `Qwen2.5-1.5B-Instruct`
- task: current FEVER active harness
- structured support-set encoder path
- shared injection path
- frozen receiver
- train/eval splits and selection/gate protocol
- the same run artifact contract

### 8.2 Budget invariants

Keep fixed initially:

- 32 train steps for direct comparability,
- current selection snapshots,
- current prompt/support serialization unless the probe explicitly targets that variable.

Only the winning geometry rescue variant may later be extended to 48 or 64 steps.

### 8.3 Writer-side invariants

Use as the initial two-level base:

- current slot-basis rescue writer family,
- current writer slot-basis orthogonalization if already part of the run,
- current writer-side settings unless the run is explicitly a B-1d check.

### 8.4 What not to vary initially

Do **not** vary all of the following at once:

- Writer architecture,
- Reader attention mode,
- prompt conditioning mode,
- Fuser architecture,
- auxiliary losses,
- train steps,
- selection policy,
- receiver adaptation.

The whole point of this plan is to narrow the causal chain, not re-expand it.

---

## 9. Success criteria for the current narrow plan

The project now needs a two-level success definition that is stricter than “something changed” and more informative than only checking the final FEVER gate.

### 9.1 Geometry-alive gate (new internal gate)

Before talking about FEVER capability gates, the two-level path must satisfy a new **geometry-alive** condition.

A run counts as geometry-alive only if it simultaneously achieves all three:

#### G1. Query specialization
At some snapshot before collapse:

- `reader_attention_pairwise_cosine_mean <= 0.90`, and
- `reader_attention_entropy_mean <= 1.95` for 8 long slots,
- or equivalently another pre-registered query-specificity metric that demonstrates non-uniform multi-query reading.

#### G2. Non-collapsed `M_short`
At that same or later snapshot:

- `memory_short_effective_rank >= 1.80` for `K=4`, or
- `memory_short_effective_rank >= 2.00` for `K=8`, and
- `memory_short_pairwise_cosine_mean <= 0.98`.

#### G3. Non-instant task collapse
At minimum:

- no dominant-label collapse before step 8, and
- a meaningful snapshot remains evaluable without immediate all-one-label degeneration.

If G1–G3 are not reached, the bridge is **not alive**, even if one isolated metric twitches.

### 9.2 Diagnostic success

Counts as **diagnostic success** if:

- the run does **not** pass selection,
- but the geometry-alive gate is passed,
- and the failure mode has been cleanly reclassified from B-1 to B-2 or to a narrower B-1 submode.

This is not enough to unlock later workstreams, but it is enough to justify one more narrow bridge iteration.

### 9.3 Medium success

Counts as **medium success** if:

- the geometry-alive gate is passed,
- at least one two-level variant passes `screen248-val` selection,
- and the winning two-level variant is clearly more internally healthy than the current slot-basis rescue base,
- even if `screen248-test` is not yet passed.

This is the minimum level at which the project may consider reopening the next macro step.

### 9.4 Strong success

Counts as **strong success** if:

- a two-level variant passes the geometry-alive gate,
- passes `screen248-val` selection,
- and passes `screen248-test`.

Strong success fully unlocks the downstream sequence from `PLAN.md`.

---

## 10. Work packages and execution order

This section is the operational core.

The order matters.

---

# Phase RG-0 — Instrumentation patch (must happen first)

## 10.1 Objective

Before adding another rescue idea, add the observability that can distinguish B-1a, B-1b, B-1c, and B-1d.

## 10.2 Why this phase is mandatory

Current metrics tell us attention is uniform.
They do **not** yet tell us whether the uniformity originates mainly from:

- context-overwrite at the query embedding level,
- score collapse before softmax,
- readout collapse after attention,
- or fuser duplication after otherwise healthy readouts.

That is too little observability for stable progress.

## 10.3 Required new logging

Add the following diagnostics to the two-level FEVER path.

### Reader-side pre-attention diagnostics

For each training event and each snapshot, log:

- `reader_base_query_pairwise_cosine_mean`
- `reader_conditioned_query_pairwise_cosine_mean`
- `reader_base_query_norm_mean`
- `reader_conditioned_query_norm_mean`
- `reader_context_shift_norm_mean`
- `reader_context_overwrite_ratio = ||context_shift|| / mean||base_query||`
- `reader_qk_logit_mean`
- `reader_qk_logit_std`
- `reader_qk_logit_range`
- `reader_qk_logit_pairwise_cosine_mean`

### Reader-side post-attention diagnostics

In addition to the existing metrics, add:

- `reader_argmax_mass_mean`
- `reader_argmax_mass_std`
- `reader_readout_pairwise_cosine_mean`
- `reader_readout_effective_rank`
- `reader_query_to_slot_top1_agreement_rate`
- `reader_query_to_slot_top2_coverage_fraction`

### Fuser-side diagnostics

Log:

- `fuser_input_pairwise_cosine_mean`
- `fuser_input_effective_rank`
- `fuser_output_pairwise_cosine_mean`
- `fuser_output_effective_rank`
- for resampler fuser: `fuser_short_query_pairwise_cosine_mean`
- for linear fuser: singular-value or rank-style summary of the fused short slots

### Writer residual diagnostics

Even though Writer is no longer the primary suspect, still log:

- per-slot energy / norm histogram summary
- `memory_long_slot_energy_cv`
- `memory_long_slot_variance_cv`
- top singular values of `M_long`

These are needed to rule out B-1d later.

## 10.4 Required code changes

### Files

- `src/memtotal/models/memory.py`
- `src/memtotal/training/m4_shared_injection.py`
- `tests/test_m4_shared_injection.py`
- `tests` for any new summary/diagnostic helper

### Minimum implementation contract

The new metrics must appear in:

- train events,
- snapshot evaluation payloads,
- dynamics recovery CSV/summary,
- review summary CSV/MD,
- and top-level summary JSON where appropriate.

## 10.5 Acceptance for RG-0

RG-0 is complete only if:

- the new metrics are emitted for current two-level runs,
- existing run summaries still generate successfully,
- tests pass,
- and a single smoke run proves the diagnostics are non-empty and numerically sane.

---

# Phase RG-1 — Lowest-cost decisive probes (config-first, minimal-code)

This phase is intentionally conservative.

Its purpose is to answer the biggest remaining structural question **without** immediately rewriting the Reader.

> Is the current collapse driven primarily by shared prompt-summary conditioning and/or by H=4→K=8 fuser expansion?

## 11.1 Canonical hypothesis for RG-1

The current strongest hypothesis is:

1. prompt-summary conditioning is overwriting query individuality,
2. and H=4→K=8 resampling is compounding the duplication.

So RG-1 tests those two causes with the cheapest possible probes.

## 11.2 Required RG-1 runs

Run exactly the following first.

### RG-1A — `CTX-OFF / H4-K8 / slot-basis base`

Same as the current slot-basis rescue base, except:

- `pilot_reader_context_mode = none`
- keep `H=4, K=8`
- keep Writer slot-basis rescue
- keep safe-hinge substrate
- keep Fuser as currently configured unless the probe explicitly changes it

**Purpose:** isolate whether shared prompt-summary conditioning is the main reason all queries look the same.

### RG-1B — `CTX-OFF / H4-K4 / slot-basis base`

Same as RG-1A, except:

- `K=4` instead of `K=8`

**Purpose:** remove the fuser upsampling confound and test whether a square readout (`H=K`) is already healthier.

### RG-1C — `CTX-OFF / H4-K4 / linear fuser`

Same as RG-1B, except:

- `fuser.arch = linear`

**Purpose:** remove the second attention stage and test whether the resampler itself is re-averaging otherwise useful readouts.

## 11.3 Exact interpretation rules for RG-1

### If RG-1A already improves query metrics substantially

Interpretation:

- B-1a is real and load-bearing,
- prompt-summary conditioning is a primary culprit,
- future canonical Reader rescue should not use raw shared additive prompt-summary conditioning.

### If RG-1A is weak but RG-1B improves

Interpretation:

- context overwrite is not the whole story,
- but H=4→K=8 expansion is clearly harmful,
- canonical rescue should temporarily standardize on `K=H`.

### If RG-1B is weak but RG-1C improves

Interpretation:

- the resampler attention stage is re-averaging or duplicating readouts,
- canonical rescue should use `linear` (or grouped identity-style) fuser for the next phase.

### If none of RG-1A/B/C materially improves geometry

Interpretation:

- the bottleneck is deeper than simple context overwrite or resampler duplication,
- move immediately to Phase RG-2 competitive Reader rescue.

## 11.4 RG-1 acceptance threshold

At least one RG-1 run must show **meaningful metric movement** to count as informative.

Use the following minimal threshold for “meaningful movement”:

- drop in `reader_attention_entropy_mean` by at least **0.05** relative to the current slot-basis base, or
- drop in `reader_attention_pairwise_cosine_mean` by at least **0.05**, or
- increase in `memory_short_effective_rank` by at least **0.30**, or
- selection becoming alive where it was previously absent.

Smaller changes should be treated as noise unless accompanied by a clean causal explanation from the new diagnostics.

---

# Phase RG-2 — Mainline Reader rescue through explicit symmetry breaking

If RG-1 does not already solve the problem, the project must stop expecting standard cross-attention to self-specialize.

This phase makes the forward path itself more specialization-friendly.

## 12.1 Canonical design principle

The Reader must stop relying on downstream task gradients alone to discover specialization.

The forward path should itself provide one of the following:

- competition,
- partition,
- or identity-preserving conditioning.

## 12.2 New Reader features to add

### Feature A — conditioning mode options

Extend `MemoryReader` to support:

- `conditioning_mode = add` (current behavior)
- `conditioning_mode = gated_add`
- `conditioning_mode = none`
- optional later: `conditioning_mode = film`

#### Required semantics

- `add`: current behavior, for backward compatibility only
- `gated_add`: add context only through a learned or fixed small scale, e.g. `queries + alpha * context_shift`, where `alpha` is initialized small
- `none`: no context addition
- `film` (optional later): modulate queries through scale/bias without overwriting base directions

### Why this is needed

This directly addresses B-1a.

The project should **not** keep a design in which the same unscaled context vector is added to every query if that vector is swamping query individuality.

### Feature B — attention mode options

Extend `MemoryReader` to support:

- `attention_mode = standard` (current behavior)
- `attention_mode = competitive_slots`
- `attention_mode = masked_partition`

#### `competitive_slots`

Implement a Slot-Attention-like competitive assignment where normalization happens across queries for each slot (or an equivalent competition mechanism), so that slots must be claimed by different queries instead of every query independently smoothing over all slots.

#### `masked_partition`

Implement a hard diagnostic mode where each query may only read a pre-specified slot subset.

### Why this is needed

This directly addresses B-1b.

The point is not elegance.
The point is to force a clean answer to the question:

> If we make specialization structurally easier, does the bridge become geometry-alive?

## 12.3 Canonical RG-2 run matrix

Do **not** run a huge sweep.
Run the minimum matrix with maximal causal value.

### Arm 1 — best RG-1 control

Whichever of RG-1A/B/C is the healthiest yet simplest becomes the control arm.

Most likely candidates:

- `CTX-OFF / H4-K4 / resampler`, or
- `CTX-OFF / H4-K4 / linear`

### Arm 2 — competitive Reader canonical

Recommended first canonical version:

- `H=4, K=4`
- Writer slot-basis rescue retained
- `conditioning_mode = gated_add` **or** `none` depending on RG-1 outcome
- `attention_mode = competitive_slots`
- `fuser.arch = linear`
- safe-hinge substrate retained

### Arm 3 — hard partition diagnostic

Recommended diagnostic version:

- `H=4, K=4`
- `conditioning_mode = none`
- `attention_mode = masked_partition`
- `fuser.arch = linear`

**Purpose:** determine whether forced specialization alone is enough to move `M_short` rank and downstream behavior.

## 12.4 Why `H=4, K=4` should be canonical in RG-2

At this stage the project is not trying to maximize final bridge width.
It is trying to make the bridge **alive**.

`H=4, K=4` is therefore the right canonical geometry because:

- it preserves a one-to-one interpretive story,
- it removes upsampling confounds,
- and it lets the project judge whether query specialization exists before worrying about expansion to `K=8`.

Only after `H=4, K=4` becomes healthy should the plan return to `H=4, K=8`.

## 12.5 RG-2 acceptance

RG-2 is successful if at least one arm reaches the geometry-alive gate.

If the hard partition arm improves `M_short` rank or selection while the competitive arm does not, then the project has still learned something useful:

- specialization is indeed the right target,
- but the chosen competitive mechanism is not yet sufficient.

That would still justify a narrow follow-up within Workstream B.

---

# Phase RG-3 — Local Reader/Fuser bootstrap (only if RG-2 is not enough)

This phase is conditional.

Use it only if RG-2 produces partial geometry gains but still fails to make the bridge alive.

## 13.1 Why RG-3 exists

If the Reader begins to specialize but still collapses too early, then the issue is likely not “no specialization signal exists at all.”

The issue is then more specifically:

- the geometry starts to form,
- but the task objective still crushes it too quickly,
- or the Fuser still re-averages it.

That is the right moment for local auxiliary shaping.

## 13.2 Allowed RG-3 interventions

Use at most one or two together.
Do not pile on everything.

### Intervention A — conditioned-query orthogonality loss

Add a loss on the **conditioned queries**, not only on attention maps:

- encourage pairwise cosine separation among conditioned queries,
- apply at low weight,
- only if conditioning is still used.

### Why this is better than attention diversity alone

Because it operates **before** softmax attention and directly counters B-1a/B-1b.

### Intervention B — short reconstruction auxiliary

Add a lightweight auxiliary loss that asks `M_short` to reconstruct:

- either the Reader readouts,
- or a compressed target derived from `M_long`.

This should be stop-gradient on the target side and low weight.

### Purpose

Give the Reader/Fuser a local pressure not to collapse to duplicated short slots.

### Intervention C — early bootstrap schedule

For the first few steps only, use a temporary schedule such as:

- freeze Writer and projector,
- train only Reader/Fuser geometry,
- then unfreeze full memory path for the remaining steps.

This is justified by the observed fact that Reader/Fuser gradients collapse very early.

## 13.3 What is not allowed in RG-3

Still do **not** do any of the following here:

- receiver LoRA,
- receiver IA3,
- full token KL,
- large new benchmark changes,
- wholesale objective redesign.

RG-3 is still a memory-side rescue stage.

---

# Phase RG-4 — Restore bridge width only after geometry is alive

This phase only opens if `H=4, K=4` reaches at least medium success.

## 14.1 Objective

Once the square readout is alive, return carefully to the true bridge question:

> Can the system widen from 4 specialized readouts to 8 injected short slots **without** re-collapsing?

## 14.2 Recommended order

### RG-4A — `H4-K8` with best Reader rescue but simple Fuser

If possible, first widen only one component at a time.

### RG-4B — grouped expansion instead of unconstrained resampling

Prefer a structured widening path over an unconstrained attention resampler.

Possible design:

- each query controls a subset of short slots,
- expansion is grouped or blockwise,
- not a free attention over all readouts.

### Why

If the project immediately returns to a symmetric resampler, it risks recreating B-1c.

---

## 15. Concrete implementation plan by file

This section maps the conceptual plan to actual repo work.

---

### 15.1 New active exec-plan file

Create a new self-contained active execution document:

- `docs/exec-plans/active/20260309-tl-reader-geometry-rescue.md`

This is important because the current bootstrap exec-plan is now too broad for the live problem.

That new exec-plan should mirror the operational content of this `PLANv2.md`, but in shorter checkpoint-friendly form.

---

### 15.2 Code changes in `src/memtotal/models/memory.py`

#### Reader

Add the following reader capabilities:

- `conditioning_mode`
- optional `context_scale` / `context_gate`
- `attention_mode`
- optional partition mask support
- optional output of pre-attention diagnostics

#### Fuser

Keep support for:

- `resampler`
- `linear`

Add, if needed:

- `identity_grouped` or equivalent when `K=H`
- grouped expansion if returning to `K>H`

#### Output contract

`MemoryReader.read(...)` should return enough structured fields to compute all diagnostics without duplicated logic in the trainer.

---

### 15.3 Code changes in `src/memtotal/training/m4_shared_injection.py`

Add runtime/config support for:

- `pilot_reader_conditioning_mode`
- `pilot_reader_context_scale_init` (or equivalent)
- `pilot_reader_attention_mode`
- `pilot_reader_partition_mode`
- `pilot_conditioned_query_diversity_weight`
- `pilot_memory_short_reconstruction_weight`
- optional bootstrap schedule knobs

Add aggregation/logging for all new diagnostics.

Add explicit top-level interpretation fields to the next summary JSON, such as:

- `context_overwrite_supported`
- `competitive_reader_supported`
- `linear_fuser_supported`
- `geometry_alive`
- `bridge_failure_submode`

---

### 15.4 New config family

Create a new config family rather than mutating old ones in place.

#### Suggested method configs

- `configs/method/memory_bootstrap_transformer_real_twolevel_h4_k8_slot_basis_ctxnone.yaml`
- `configs/method/memory_bootstrap_transformer_real_twolevel_h4_k4_slot_basis_ctxnone.yaml`
- `configs/method/memory_bootstrap_transformer_real_twolevel_h4_k4_slot_basis_ctxnone_linear.yaml`
- `configs/method/memory_bootstrap_transformer_real_twolevel_h4_k4_reader_competitive.yaml`
- `configs/method/memory_bootstrap_transformer_real_twolevel_h4_k4_reader_partition.yaml`

#### Suggested experiment configs

- `configs/exp/tl_reader_geometry_rescue_fever_qwen25_ctxnone_h4_k8.yaml`
- `configs/exp/tl_reader_geometry_rescue_fever_qwen25_ctxnone_h4_k4.yaml`
- `configs/exp/tl_reader_geometry_rescue_fever_qwen25_ctxnone_linear_h4_k4.yaml`
- `configs/exp/tl_reader_geometry_rescue_fever_qwen25_competitive_h4_k4.yaml`
- `configs/exp/tl_reader_geometry_rescue_fever_qwen25_partition_h4_k4.yaml`

Create matching screen/gate variants only for the winning arm(s), not for every exploratory config.

---

### 15.5 New runner and summary script

Create:

- `scripts/run_tl_reader_geometry_rescue_fever_qwen25.sh`
- `scripts/update_tl_reader_geometry_summary.py`

The runner should enforce staged execution:

1. RG-0 smoke / diagnostics sanity
2. RG-1 probe matrix
3. stop and summarize
4. only then RG-2 if needed
5. only then RG-3 if needed

Do not script a giant all-at-once sweep.

---

### 15.6 Tests

Add or extend tests for:

- new Reader conditioning modes
- new Reader attention modes
- partition-mask correctness
- linear vs resampler fuser contracts
- logging of new diagnostics
- summary-script aggregation logic
- backward compatibility of existing configs

Passing tests remain mandatory before the main run.

---

## 16. Exact experimental order and stop rules

This section is the practical decision tree.

---

### Step 1 — instrumentation patch + smoke

Do not start a new full run before RG-0 metrics exist.

**Stop rule:** if the new diagnostics are missing or obviously nonsensical, fix harness first.

---

### Step 2 — run RG-1A / RG-1B / RG-1C

These are the first full probes.

**Interpretation priority:**

1. Does turning off prompt-summary conditioning help?
2. Does `K=H` help?
3. Does linear fuser help?

**Stop rule:**
- If none of the three meaningfully improves geometry, do not repeat them with different seeds first. Move to RG-2.
- If one clearly improves geometry, adopt it as the control substrate for RG-2.

---

### Step 3 — run RG-2 control + competitive + partition

This is the main B-1 rescue stage.

**Stop rule:**
- If competitive or partitioned Reader makes the bridge geometry-alive, continue within Workstream B using the winning arm.
- If both fail and diagnostics say `M_long` is still too weak, classify as B-1d and return to writer-side energy balancing only.
- If both fail but Reader/Fuser geometry visibly improves and downstream still flattens, reclassify as B-2 candidate, but do **not** open receiver yet until a summary says so explicitly.

---

### Step 4 — only if needed, run RG-3

Use RG-3 only if RG-2 yields partial but not sufficient gains.

**Stop rule:**
- If RG-3 still cannot make the bridge geometry-alive, do not keep stacking local losses indefinitely.
- Instead, write a clean summary declaring whether the failure remains B-1 or has become B-2.

---

### Step 5 — if geometry is alive, restore bridge width

Only after `H=4, K=4` reaches medium success should the plan reopen `H=4, K=8` bridge widening.

**Stop rule:**
- If widening reintroduces collapse, the widening mechanism itself is now the blocker. Do not blame the Reader generically.

---

## 17. Deliverables required from this narrow plan

This section defines what must exist in the repo when this plan is considered executed.

### 17.1 Code deliverables

- Reader conditioning modes implemented
- Reader attention modes implemented
- linear-fuser path validated as an active FEVER option
- geometry diagnostics fully wired through train and snapshot outputs
- new runner and summary script

### 17.2 Artifact deliverables

Under a new review directory, e.g.

- `results/generated/review/tl-reader-geometry-rescue-fever-qwen25/`

publish at minimum:

- top-level summary JSON
- top-level summary MD
- per-arm run summaries
- reader geometry diagnostics CSV
- fuser geometry diagnostics CSV
- context-overwrite diagnostics CSV
- selection/gate artifacts for winning arms
- updated docs bundle

### 17.3 Documentation deliverables

Update at minimum:

- `README.md`
- `docs/TODO_LIST.md`
- `docs/ARCHITECTURE.md`
- `docs/briefs/20260307-m4-shared-injection-brief.md`
- `docs/tech-debt-tracker.md`
- `docs/exec-plans/active/20260309-tl-reader-geometry-rescue.md`

### 17.4 Validation deliverables

- full unit test pass
- explicit smoke run for new config family
- explicit top-level statement of whether B-1a, B-1b, B-1c, or B-1d is the current winner

---

## 18. What unlocks later workstreams

This section is intentionally strict.

### 18.1 Unlocking Step 6 — Stage B/C refresh

Allowed only if the two-level FEVER path reaches at least **medium success**.

That means:

- geometry-alive,
- at least one two-level variant passes selection,
- and the winning variant is not just a diagnostic trick with no stable internal improvement.

### 18.2 Unlocking Step 7 — CDMI experiments

Allowed only after Stage B/C refresh is re-established under the stabilized two-level family.

### 18.3 Unlocking Step 8 — receiver fallback

Allowed only if the summary clearly says the project has moved from B-1 to B-2:

- `M_long` is healthy enough,
- H=4 specialization exists,
- `M_short` is non-trivial,
- yet the frozen receiver still flattens the signal downstream.

Without that evidence, receiver fallback remains blocked.

---

## 19. What this plan deliberately avoids

To keep the project stable, this plan deliberately avoids several tempting but currently harmful directions.

### Avoid 1 — another single-level FEVER objective cycle

That question has already been answered sufficiently for now.

### Avoid 2 — more writer-side regularization as the default medicine

Writer-side geometry is no longer the narrowest current bottleneck.

### Avoid 3 — early receiver adaptation

This would reduce interpretability before the memory-side case is exhausted.

### Avoid 4 — a giant sweep over H/K/L and every auxiliary loss

The project needs sharper causal inference, not more hyperparameter fog.

### Avoid 5 — pretending a tiny entropy change is a rescue

The current system is extremely close to the worst-case uniform boundary. Tiny deltas are not enough.
Use the geometry-alive thresholds.

---

## 20. Blunt decision summary

This section is the shortest correct executive summary.

1. **Stay in Workstream B.**
2. **Do not move to transfer refresh, CDMI, or receiver fallback yet.**
3. **Treat the current blocker as query-side readout geometry, not writer existence.**
4. **The first thing to test is whether prompt-summary conditioning is erasing query identity.**
5. **The next canonical rescue should prefer `H=4, K=4` and a simple fuser before returning to `H=4, K=8`.**
6. **If standard cross-attention still stays uniform, move to competitive or partitioned Reader designs.**
7. **Only after the bridge is geometry-alive should you reopen later workstreams.**

---

## 21. Recommended immediate next command-level work sequence

If someone were to start execution immediately, the order should be:

1. create `docs/exec-plans/active/20260309-tl-reader-geometry-rescue.md`
2. implement RG-0 instrumentation
3. run a smoke test for the new diagnostics
4. materialize RG-1 configs
5. run `CTX-OFF / H4-K8`
6. run `CTX-OFF / H4-K4`
7. run `CTX-OFF / H4-K4 / linear`
8. summarize and classify the dominant submode
9. only then implement competitive/partitioned Reader if still needed

That is the correct current next hop.

---

## 22. Reference notes that informed the plan

These are not being used as decorative citations; they are the main external concepts that shaped the decisions above.

### R1. Prefix-Tuning
Li, X. L., & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* ACL 2021.  
https://aclanthology.org/2021.acl-long.353/

Used here for the narrow claim that frozen-backbone prefix-style adaptation is possible, but fragile and optimization-sensitive.

### R2. P-Tuning v2
Liu, X. et al. (2022). *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks.* ACL 2022.  
https://aclanthology.org/2022.acl-short.8/  
https://arxiv.org/abs/2110.07602

Used here for the lesson that deep prompt tuning can work on hard NLU, but optimization and architecture choices matter substantially, especially outside very large fully adapted settings.

### R3. Fine-Tuning can Distort Pretrained Features
Kumar, A. et al. (2022). *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR 2022.  
https://openreview.net/forum?id=UYneFzXSJWh

Used here for the broader insight that continuation training can distort useful existing geometry instead of stabilizing it.

### R4. Slot Attention
Locatello, F. et al. (2020). *Object-Centric Learning with Slot Attention.* NeurIPS 2020.  
https://arxiv.org/abs/2006.15055

Used here as the main justification for competition-based slot/query assignment to break symmetry, rather than hoping standard attention will specialize under weak supervision.

### R5. Perceiver IO
Jaegle, A. et al. (2022). *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* ICLR 2022.  
https://arxiv.org/abs/2107.14795

Used here as architectural precedent for query-to-latent readout interfaces and structured output queries.

### R6. ANIL
Raghu, A. et al. (2020). *Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML.* ICLR 2020.  
https://openreview.net/forum?id=rkgMkCEtPB

Used here to support the broader MemTOTAL claim that adaptation can be concentrated on a small readout/head-like interface rather than the whole system.

### R7. MiniLLM
Gu, Y. et al. (2024). *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR 2024.  
https://openreview.net/forum?id=5h0qf7IBZZ

Used here only to clarify why generative-model distillation arguments do not automatically imply that full token KL is the right current move for this narrow readout-geometry problem.

### R8. LoRA
Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.  
https://openreview.net/forum?id=nZeVKeeFYf9

Used here only as the later receiver-side fallback reference, not as the current plan.

### R9. (IA)^3 / T-Few
Liu, H. et al. (2022). *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning.* NeurIPS 2022.  
https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html

Used here as the later receiver-side fallback reference, especially for very small activation-scaling adaptations.

---

## 23. Final operational statement

The most important correction to the project’s next move is this:

> The project does **not** currently need a new grand theory, a new backbone, or a new receiver-side adaptation story. It needs a disciplined rescue of Reader/Fuser geometry inside the already active two-level FEVER bridge.

The next phase should therefore proceed in the following conceptual order:

1. **measure the right geometry,**
2. **remove context-overwrite if it exists,**
3. **force or induce real query specialization,**
4. **stop the Fuser from duplicating readouts,**
5. **only then ask whether the two-level bridge is alive enough to reopen the rest of the roadmap.**

That is the correct current next plan.
