# PLANv6 — Post-Gate-Repair Independent Writer Recovery, Structured Support Validation, and Anti-Homogenization Program

**Date:** 2026-03-09  
**Repository audited:** `YJLi-new/MemTOTAL` (`main` as inspected on 2026-03-09)  
**Prepared after cross-reading:** current repo code, historical `PLAN.md`–`PLANv5.md`, active exec-plans, generated review summaries, raw training traces in the attached archive, and relevant PEFT / latent-memory / prefix-tuning literature and open-source implementations.

---

## 0. Scope, intent, and blunt conclusion

This plan is the direct continuation of the unfinished parts of the previous PLANs, but with a stricter methodology.

My bottom-line recommendation is:

1. **Do not treat the current Writer-direct deep-prefix result as an architectural falsification.**
2. **Do not reopen Reader/Fuser yet.**
3. **Do not scale Writer capacity first.**
4. **Do not move to RL / GRPO / DPO yet.**
5. **Do not judge the route with the current 1–50-step gate.**
6. **The next canonical work is a corrected Writer-direct validation program built around four things:**
   - gate repair,
   - task-vs-regularizer gradient attribution,
   - support-interface reconstruction (especially non-`pooled_block` paths and true multi-item conditioning),
   - explicit anti-homogenization objectives with controlled mixed experiments.

In one sentence:

> **The route is not dead; the route was mis-measured, the Writer input is still information-starved, the writer-side gradients are still attribution-ambiguous, and the optimization recipe still lets the projector dominate the learning dynamics.**

So the right move is **not a broad pivot**, but a **disciplined PLANv6 recovery program** that fairly tests the independent Writer thesis under structured support, richer supervision, corrected gating, and targeted anti-collapse losses.

---

## 1. What I audited

I audited the following materials together, not in isolation:

### 1.1 Core code paths

- `src/memtotal/models/backbone.py`
- `src/memtotal/models/memory.py`
- `src/memtotal/training/m4_shared_injection.py`
- `src/memtotal/tasks/writer_jointpeft_data.py`
- `src/memtotal/tasks/sources.py`
- current scripts and result-summary generators

### 1.2 Planning / architecture docs

- `PLAN.md`
- `PLANv2.md`
- `PLANv3.md`
- `PLANv4.md`
- `PLANv5.md`
- `docs/ARCHITECTURE.md`
- active exec-plans under `docs/exec-plans/active/`

### 1.3 Latest experimental artifacts

- `results/generated/review/writer-deep-prefix-jointpeft-qwen25/writer-deep-prefix-jointpeft-summary.json`
- corresponding `writer-deep-prefix-jointpeft-summary.md`
- raw `train_events.json` traces for the three tasks
- prior review bundles and addenda relevant to Writer-first / Reader rescue / capacity fallback / joint-PEFT

### 1.4 External references checked for alignment

- Prefix-Tuning
- P-Tuning v2
- LoRA / Hugging Face PEFT
- theory on prompting / prefix-tuning limitations
- Prefix-Tuning+
- SoftCoT
- MemGen
- collapse-prevention / representation-regularization lines such as VICReg, Barlow Twins, VCReg, VIB, and orthogonality regularization

---

## 2. Ground-truth findings from the current repo and latest run

This section is intentionally concrete.

### 2.1 What the current code already proved

#### A. The current deep-prefix path is real

The current branch is no longer “fake deep prompting” or mere embedding concat. The backbone actually:

- accepts `layer_prefix_hidden_by_layer`,
- normalizes them per selected layer,
- runs each layer’s frozen `input_layernorm`,
- projects them through that layer’s own frozen `k_proj` / `v_proj`,
- applies rotary positional embedding,
- writes them into `past_key_values`,
- and then scores continuations through the real Qwen attention stack.

That means the project has already crossed one major threshold:

> **the deep-prefix physical injection route is real.**

This matters because many earlier conclusions in the project history were made while the bridge itself was still under suspicion.

#### B. Tiny receiver LoRA is also mechanically live

The receiver micro-LoRA plumbing exists and is trainable on Qwen-side `k_proj/v_proj` targets. This means the current route is no longer “Writer alone vs a hard-frozen receiver” in the old sense; it is already a minimal joint-PEFT route.

#### C. Structured support plumbing already exists in code, but was not used in the canonical run

The repo already contains both:

- `pooled_block`
- `structured_support_set`

That is crucial. It means the next high-value experiment does **not** require inventing the whole support-interface stack from scratch. The infrastructure to stop compressing all support into one vector is already partially there.

---

### 2.2 What the latest canonical run actually did

The active addendum’s canonical Writer-direct joint-PEFT run used:

- `bridge_mode = writer_direct`
- `injection = sparse_deep_prefix`
- selected layers `[0, 1, 2, 3]`
- early4 receiver LoRA on `k_proj/v_proj`
- W0-sized `WriterWeaverHead`
- `support encoder = pooled_block`
- `Writer stimulus = support_and_context`
- `Writer context tokens = 8`
- `train_steps = 500`
- `warmup = 50`
- `writer freeze phase = steps 1–50`
- plus several writer-side regularizers

That choice matters because it means the latest run did **not** test:

> “Can a well-conditioned independent Writer produce useful deep prefixes?”

It tested something narrower:

> “Can a small Writer conditioned on a single pooled support vector plus short context learn useful deep prefixes under a projector-dominant, clipped, jointly regularized recipe?”

Those are not equivalent questions.

---

### 2.3 The official `route_live=false` verdict is invalid as an architectural verdict

This is the single most important measurement correction.

The same canonical plan:

- freezes the Writer during steps 1–50,
- and simultaneously defines `route_live` using Writer gradient median over steps 1–50.

That gate is impossible to satisfy by construction.

This does **not** mean the run was good. It means the **headline verdict is not trustworthy** as a route verdict.

---

### 2.4 Raw-trace facts that matter more than the top-line verdict

From the raw logs and summary, the following facts are the real story:

#### A. Post-unfreeze Writer gradients are nonzero on all three tasks

Medians from the actual run windows show:

| Task | Writer grad 1–50 | Writer grad 51–100 | Writer grad 451–500 |
|---|---:|---:|---:|
| GSM8K | 0.0 | 10.81 | 1.136 |
| NarrativeQA | 0.0 | 10.62 | 0.543 |
| FEVER | 0.0 | 19.45 | 0.021 |

So the route is **not** a zero-gradient dead wire after unfreeze.

#### B. The projector and receiver LoRA are highly active

The projector gradients are much larger than the Writer gradients, and receiver LoRA gradients are also clearly nonzero. So the downstream bridge is active.

#### C. Prefix attention is nontrivial on multiple injected layers

The prefixes are being attended to in the backbone. On GSM8K, layer 2 was strongest in the latest run; on FEVER and NarrativeQA, multiple early layers also carried nontrivial attention mass.

#### D. The source-side latent remains almost maximally collapsed

The latest summary shows:

- GSM8K `memory_long_common_mode_energy_ratio ≈ 0.99999973`
- NarrativeQA `≈ 0.99999969`
- FEVER `≈ 0.99999988`

This is still catastrophic source homogenization.

#### E. The projector outputs are high-rank even though the Writer source is low-rank

Projected memory effective rank is already around `16–18`, while source-side slot rank is near `1.0`.

That means the projector is generating **geometric diversity downstream of a collapsed source**.

#### F. The support interface used in the canonical run is literally rank-1 in practice

From the raw logs in this run:

- `writer_support_state_count = 1.0` throughout
- `support_state_effective_rank = 1.0` throughout
- `memory_slot_effective_rank` drifts toward `~1.01`

So the current Writer is truly being fed an effectively single support state.

#### G. Context may be numerically swamping support

In the post-unfreeze `51–100` window, the median Writer-side hidden norms were roughly:

- GSM8K: context/support ≈ `3.17x`
- NarrativeQA: context/support ≈ `3.60x`
- FEVER: context/support ≈ `3.67x`

That means even before the Writer learns anything, its conditioning may already be biased toward prompt-context magnitude over support evidence.

#### H. Optimization is still unstable and projector-dominated

Important symptoms:

- GSM8K was clipped on essentially **100%** of steps.
- NarrativeQA was clipped on ~**87%** of steps.
- FEVER was clipped on ~**78%** of steps.
- prefix L2 norms grew substantially over training.
- the ratio `(writer + support) grad / projector grad` is small and tends to shrink late in training.

This suggests the projector is carrying too much of the update energy.

---

## 3. My independent mechanistic diagnosis

This section is the most important part of the plan.

---

### 3.1 The project has already crossed one true milestone

A real milestone has been achieved:

> **the independent Writer → projector → deep-prefix → frozen Qwen(+tiny LoRA) circuit is physically alive.**

That is not trivial. It means the project is no longer blocked by the old question “can gradients and attention even move through this route?”

They can.

The main failure mode has shifted.

---

### 3.2 The main bottleneck is now source usefulness, not pure route disconnection

The best current explanation is:

1. the Writer produces highly common-mode memory,
2. the projector transforms that nearly constant source into high-rank prefixes,
3. those prefixes influence the receiver enough to help FEVER somewhat,
4. but the signal is too weakly input-specific to help harder generative reasoning tasks.

In plain language:

> **the route is alive, but the content entering the route is still low-information.**

That is why I do **not** read the latest result as “deep prefix failed.”
I read it as:

> **the current Writer conditioning contract failed to produce differentiated evidence-conditioned memory.**

---

### 3.3 `pooled_block` is not a benign simplification; it is a likely upstream collapse generator

This is where the latest branch repeats a deeper historical pattern.

In `PLANv2`, the project already discovered that **shared conditioning can erase identity**. That diagnosis was made on the Reader side: adding the same prompt-summary vector to all queries made them nearly parallel and encouraged symmetric collapse.

The same structural error is now reappearing upstream in the Writer path.

`pooled_block` does the following:

- collapse multiple support items into one summarized block,
- summarize that block into one hidden state,
- pass that as support conditioning to the Writer.

In the current run, the support side is therefore effectively:

- one support token,
- rank 1,
- much smaller in norm than context,
- and shared across all memory slots.

That strongly encourages the Writer to emit a common-mode latent.

This is **not** just a training failure. It is a bad information interface.

So the next step must explicitly treat `pooled_block` as a variable and not as a fixed default.

---

### 3.4 But changing support structure alone is not enough; the loss landscape is also lazy

This is where I agree with the strongest criticism in the external reviews.

If the downstream task loss does not strongly reward differentiated Writer content, a richer support interface alone may still collapse into lazy averages.

That is why PLANv6 must not only change the input interface.
It must also add explicit anti-homogenization experiments, including:

- contrastive objectives,
- VICReg / VCReg-style variance-covariance regularization,
- Barlow-style redundancy reduction,
- cosine orthogonality penalties,
- information-bottleneck variants,
- dropout- and masking-based stochastic views.

The correct view is:

> **support structure fixes the information supply problem; auxiliary losses fix the optimization incentive problem.**

We need both.

---

### 3.5 The current Writer gradients are still attribution-ambiguous

The current run has nonzero Writer gradients after unfreeze, but that does **not** prove task supervision is effectively training the Writer.

Why?

Because the run also includes writer-side regularizers such as:

- gain margin,
- common-mode penalty,
- covariance diversity,
- slot-energy balance.

Those losses can generate Writer gradients even if the task loss contributes little or nothing useful.

So the next branch must answer the unresolved question directly:

> **How much Writer gradient comes from the task, how much comes from writer-side auxiliary losses, and are those gradients aligned or conflicting?**

This is a first-class instrumentation requirement, not a nice-to-have.

---

### 3.6 There is a second hidden bottleneck: projector-dominated global gradient clipping may be starving the Writer

This is a key diagnosis that I do not think should be ignored.

The current training loop clips **all trainable parameters together** with a single global norm cap.
At the same time:

- projector gradients are much larger than Writer gradients,
- clipping fires on most steps,
- and the writer-to-projector grad ratio is small.

This creates a likely side effect:

> when the projector dominates the total gradient norm, global clipping scales **down every trainable group together**, including the Writer.

So the Writer may be receiving two kinds of starvation at once:

1. weak task supervision,
2. clip-induced attenuation because the projector explodes first.

This is important because it means the current recipe may be **systematically suppressing the very gradients we need to test**.

PLANv6 must therefore replace global undifferentiated clipping with **group-aware clipping or at least group-aware logging and comparison**.

---

### 3.7 Why FEVER moved while `delta_answer_logprob` stayed flat

The FEVER result is real enough to take seriously, but not in the way the current reporting frame suggests.

My interpretation is:

- FEVER is a classification-style task,
- deep-prefix injection is more naturally suited to biasing existing internal computations than to creating entirely new multi-step reasoning trajectories,
- the projector can learn a useful quasi-static bias even from weakly differentiated Writer inputs,
- this can improve label selection or margin behavior without showing up as a strong mean `delta_answer_logprob` signal.

So FEVER’s improvement should be read as:

> **the current bridge can already bias the frozen reasoner usefully on easier decision-style tasks, even though Writer-side input-specific memory is not yet solved.**

That is encouraging, not decisive.

---

### 3.8 Why GSM8K is the wrong first hard gate

GSM8K is a poor first proof target here.

The theory on prompting / prefix-tuning limitations says these methods are much better understood as **biasing existing model computations** than as arbitrarily rewriting the full internal reasoning program.

That means the first convincing positive signal is more likely to appear on:

- FEVER,
- Story Cloze / similar classification,
- shallow evidence selection,
- short-form QA,

than on multi-step generative math reasoning.

So PLANv6 will use a **task ladder**, not a GSM8K-first go/no-go rule.

---

## 4. Architecture feasibility cross-examination

---

### 4.1 What is still viable and should stay

#### Keep A — Independent Writer

The independent Writer thesis is still alive.

The current evidence says the route is physically live but poorly conditioned. That does **not** falsify the independence idea.

#### Keep B — Per-layer deep prefix projection

This is the right injection contract to keep testing before any larger pivot.

#### Keep C — Tiny early receiver LoRA

The tiny early `k_proj/v_proj` LoRA remains the right minimal receiver-side assist.
It is small enough to preserve the independent Writer story while still giving the route a learnable interface.

#### Keep D — FEVER-first / shallow-task-first validation

This is the correct way to get the first clean positive proof.

---

### 4.2 What should stay off for now

#### Delay A — Reader / Fuser reopening

Reader/Fuser should remain off until the Writer can produce genuinely differentiated, task-useful source memory.

Reopening Reader now would only add confounds.

#### Delay B — large Writer scaling

Earlier capacity-upgrade work already showed that bigger Writer capacity can worsen collapse instead of fixing it.

#### Delay C — RL / DPO / GRPO

Supervised usefulness is not stably established yet.
There is no reason to add higher-variance training until the deterministic branch is alive and useful.

#### Delay D — Qwen3 transfer / portability claims

Portability remains one of the most interesting long-horizon contributions, but it is not the next proof point.

---

### 4.3 What becomes the first architectural fallback if deep prefix still underdelivers

If the corrected Writer-direct program still shows:

- real post-unfreeze task gradients,
- non-collapsed Writer-side source states,
- but weak usefulness due to prefix competition inside attention,

then the first architecture fallback should be:

> **Prefix-Tuning+-style attention-independent injection**

not MemGen-style reasoner-coupled weaving.

Why?

Because that fallback preserves the independent Writer thesis while changing only the insertion contract.

That is much cleaner scientifically.

---

## 5. Hard requirements explicitly incorporated into PLANv6

You asked that the plan explicitly include the following. I agree with all four and treat them as non-optional.

### 5.1 `pooled_block` must be isolated as its own experimental variable

Yes. `pooled_block` will no longer be treated as an invisible default. It becomes a first-class axis in the experiment matrix.

### 5.2 Anti-homogenization losses and measures must be tested directly

Yes. PLANv6 includes dedicated experiments for:

- Contrastive loss
- VICReg / VCReg-style variance-covariance regularization
- Barlow Twins-style redundancy reduction
- cosine orthogonality penalties
- information bottleneck variants
- dropout / masking / view-noising injections

### 5.3 Multi-item cross-attention must be explicitly tested

Yes. PLANv6 includes support interfaces where the Writer cross-attends over multiple support items rather than a pooled summary.

### 5.4 Mixed experiments must be included

Yes. PLANv6 includes a staged mixed-matrix program so that:

- support interface,
- anti-collapse loss family,
- and conditioning mix

can be combined in a controlled, not ad hoc, way.

---

## 6. PLANv6 operating principles

These are the non-negotiable rules for the next phase.

1. **No more frozen-window route gates.**
2. **No top-line architecture verdict without gradient attribution.**
3. **No support-interface claim without support-rank / item-coverage diagnostics.**
4. **No usefulness verdict from mean `delta_answer_logprob` alone.**
5. **No Reader reopening until source-side usefulness is proven.**
6. **No capacity scaling before conditioning and loss design are fairly tested.**
7. **No broad factorial explosion at the start; use a staged design-of-experiments process.**
8. **Every comparison must hold the rest of the bridge constant.**
9. **Anti-collapse losses must hit Writer-side representations first, not only projected prefixes.**
10. **Group-wise gradient health must be measured because projector domination is now a known risk.**

---

## 7. The corrected PLANv6 hypothesis

### 7.1 Main hypothesis

If the project:

- repairs the invalid route gate,
- logs task-only vs auxiliary-only Writer gradients and their cosine alignment,
- replaces or complements `pooled_block` with structured multi-item Writer conditioning,
- prevents projector-dominated clipping from starving the Writer,
- and adds targeted anti-homogenization objectives on Writer-side representations,

then the independent Writer + deep-prefix + tiny receiver LoRA architecture will show **stable, input-conditioned usefulness** first on FEVER-class tasks and later on harder tasks.

### 7.2 Negative hypothesis to falsify

If after all of the above:

- the Writer still receives near-zero task gradients,
- source-side support rank and Writer slot rank remain near 1,
- common-mode energy remains effectively maxed,
- and classification tasks still do not improve,

then the limitation is likely the **injection contract**, not merely the input interface or auxiliary loss design.
At that point the next move is Prefix-Tuning+-style attention-independent injection.

---

## 8. New gates, metrics, and stop criteria

This section replaces the broken logic from the current summary framework.

---

### 8.1 New gate A — `route_live_post_unfreeze`

For Writer-direct runs, define route liveness on **post-unfreeze** windows.

#### Weak pass

Pass if on at least one non-control task:

- `writer_grad_norm_steps_51_100_median > 1e-4`
- `projector_grad_norm_steps_51_100_median > 1e-3`
- `receiver_lora_grad_norm_steps_51_100_median > 1e-4`
- `prefix_attention_nontrivial_layer_count >= 2`
- all metrics finite

#### Medium pass

Additionally require:

- same conditions still hold in steps `451–500`
- no systematic collapse to zero after the early window

#### Strong pass

Additionally require:

- task-only Writer gradient is nontrivial (see Section 8.2)

This is the first gate that should determine whether the route is alive.

---

### 8.2 New gate B — `writer_task_supervision_live`

This is the gate missing from all previous plans.

For logged diagnostic steps, compute:

- `grad_writer_task_only`
- `grad_writer_aux_only`
- `grad_writer_total`
- `cos(task, aux)`
- `cos(task, total)`

#### Weak pass

- `median(norm(task_only)) / median(norm(total)) >= 0.10`
- `median(cos(task, total)) > 0`

#### Medium pass

- ratio `>= 0.20`
- `median(cos(task, total)) >= 0.30`
- `median(cos(task, aux)) > -0.20`

#### Failure interpretation

- **near-zero task norm + high aux norm**: Writer movement is mostly auxiliary-driven
- **task norm nonzero but `cos(task, aux) << 0`**: regularizers are fighting the task
- **task norm nonzero and aligned**: Writer is genuinely being trained by downstream objective

This gate is mandatory before any optimistic conclusion about Writer usefulness.

---

### 8.3 New gate C — `source_not_collapsed`

Track collapse where it actually matters: on the support states and Writer slots.

Required metrics:

- `writer_support_state_count`
- `support_state_effective_rank`
- `writer_memory_slot_effective_rank`
- `memory_long_common_mode_energy_ratio`
- pairwise slot cosine statistics
- per-slot / per-item attention specialization metrics

#### Weak pass

At least one of:

- `support_state_effective_rank > 1.2`
- `writer_memory_slot_effective_rank > 1.5`
- residual energy `(1 - common_mode_ratio)` improves by `>= 100x` vs pooled baseline

#### Medium pass

At least two of:

- `support_state_effective_rank > 2.0`
- `writer_memory_slot_effective_rank > 2.0`
- `common_mode_ratio < 0.999`
- slot pairwise cosine median is materially below the pooled baseline

#### Strong pass

- `common_mode_ratio < 0.99`
- clear slot/item specialization signatures

This gate is not the same as task success. It checks whether the source branch is still degenerating.

---

### 8.4 New gate D — `stable_training_v6`

Training is stable only if all of the following are true:

- `train_loss_steps_451_500_median < train_loss_steps_1_50_median`
- no NaN / Inf
- clipping fraction is not near 100% for all groups for the full run
- prefix norm growth is bounded and not just exploding upward
- step-to-step loss oscillation is not pathological

New stability metrics to log:

- `was_grad_clipped_writer`
- `was_grad_clipped_projector`
- `was_grad_clipped_receiver_lora`
- group update/parameter norm ratios
- prefix L2 growth ratio relative to step 0 / step 10
- median absolute deviation of loss in rolling windows

---

### 8.5 New gate E — `usefulness_positive_v6`

#### For classification-style tasks (FEVER, Story Cloze if added)

Track:

- accuracy
- macro-F1
- gold-vs-best-competitor logit margin delta
- fraction of examples with positive margin shift
- argmax flip count toward the correct label
- answer-switch rate under greedy decoding

A run counts as usefulness-positive if:

- task score is non-regressive vs control,
- and at least one of margin/flip diagnostics is positive,
- even if mean `delta_answer_logprob` is small.

#### For generative tasks (GSM8K, NarrativeQA)

Track:

- exact match / F1
- mean `delta_answer_logprob`
- median `delta_answer_logprob`
- positive-delta fraction
- nonzero-delta case count
- first-answer-token delta
- greedy answer-switch rate

A run counts as usefulness-positive if:

- task score is non-regressive,
- and at least one of median delta / positive fraction / first-token delta improves,
- not only the mean.

---

## 9. Required instrumentation work before spending more GPU budget

This is Phase 0 and it is not optional.

---

### 9.1 Patch the broken summary gate

**File:** `scripts/update_writer_deep_prefix_jointpeft_summary.py`

#### Required changes

- remove route verdict dependence on steps `1–50` for Writer-direct runs
- define configurable `head_window`, `post_unfreeze_window`, `tail_window`
- mark `route_live_post_unfreeze` separately from any historical compatibility field
- do not let `stable_training` be implicitly forced false just because the frozen head-window Writer grad is zero
- add classification usefulness diagnostics beyond mean delta

#### Deliverable

A summary JSON/MD that can no longer mislabel a deliberately frozen Writer as a dead route.

---

### 9.2 Add gradient attribution probes

**Primary file:** `src/memtotal/training/m4_shared_injection.py`

#### Add optional diagnostic mode

New runtime knobs:

- `pilot_gradient_probe_interval`
- `pilot_gradient_probe_max_steps`
- `pilot_gradient_probe_modules = {writer, support_encoder, projector, receiver_lora}`
- `pilot_gradient_probe_enabled`

#### Computation method

On probe steps only, on the same batch:

1. compute task loss only,
2. compute auxiliary loss only,
3. compute total,
4. collect `autograd.grad(...)` for Writer params without polluting the main optimizer state,
5. log:
   - norm(task)
   - norm(aux)
   - norm(total)
   - cosine(task, aux)
   - cosine(task, total)
   - cosine(aux, total)

#### Important note

Do **not** rely on backward+zero-grad bookkeeping alone. Use explicit gradient extraction for correctness.

---

### 9.3 Add support-interface diagnostics

The Writer currently does not expose enough about how it uses support items.

#### Required additions

In `WriterWeaverHead` and related support code, add optional diagnostics for:

- support attention weights per conditioning layer
- per-slot attention entropy over support items
- per-item coverage aggregated over slots
- top-attended item count per slot
- pairwise cosine of support item states
- rank of raw support item states before any pooling / set encoding

#### Why this matters

If multi-item support is added but the Writer still attends uniformly to all items, then we have changed the interface without actually changing the information geometry.

---

### 9.4 Add group-aware gradient clipping / at least group-aware logging

Current global clipping is too coarse.

#### Minimum acceptable change

Log per-group unclipped norms and per-group clip decisions.

#### Recommended change

Use per-group clipping such as:

- Writer (+support encoder): clip `1.0`
- projector: clip `0.5`
- receiver LoRA: clip `0.5` or `1.0`

The exact values can be tuned, but the key principle is:

> **do not let projector explosions scale down Writer gradients by default.**

---

### 9.5 Add a Writer-side auxiliary projection head for anti-collapse losses

For contrastive / VICReg / Barlow experiments, add a small auxiliary head:

- input: Writer slot matrix or pooled Writer latent
- output: normalized auxiliary embedding for auxiliary losses

This head should be used for diagnostic/auxiliary regularization only, not for the main deep-prefix bridge.

Why:

- it prevents the auxiliary losses from acting only on the projected Qwen-side prefix geometry,
- and lets us target the actual source collapse directly.

---

## 10. Support-interface reconstruction workstream

This is the first main experiment axis.

---

## 10.1 Objective

Test whether the current common-mode collapse is primarily caused by the support encoding contract.

---

## 10.2 Support interface variants to implement and test

### S0 — `pooled_block_legacy`

Current canonical baseline:

- all support rows collapsed to one support block,
- summarized once,
- fed as one support state.

Purpose:

- historical baseline,
- must remain in the matrix for fair comparison.

---

### S1 — `pooled_block_gated`

Same single pooled block, but add an explicit balance control so support is not numerically drowned by context.

Implementation options:

- LayerNorm both support and context before Writer conditioning,
- learned scalar gates for context and support streams,
- or fixed re-scaling so support and context norms are comparable.

Purpose:

- isolate whether the failure is “one token only” or partly “support too weak relative to context.”

---

### S2 — `structured_support_set`

Use the existing structured support-set path:

- summarize each support row independently,
- pass them through `StructuredSupportSetEncoder`,
- feed the resulting item sequence to the Writer.

Purpose:

- first fair test of non-pooled support under the same bridge.

---

### S3 — `multi_item_cross_attn_raw`

New path:

- summarize each support row independently,
- do **not** immediately compress them into one state,
- feed the raw item states directly to the Writer’s support cross-attention.

Purpose:

- isolate the effect of item-wise Writer access without an intermediate set encoder.

---

### S4 — `multi_item_cross_attn_encoded`

New path:

- summarize each support row independently,
- pass through `StructuredSupportSetEncoder`,
- feed encoded item states to Writer cross-attention.

Purpose:

- compare raw multi-item vs encoded multi-item.

---

### S5 — `hybrid_pooled_plus_items`

New path:

- include one pooled global token,
- plus the item-wise support states,
- let Writer attend to both.

Purpose:

- preserve global summary while restoring local distinctions.

This is the most plausible hybrid if pure item-wise conditioning proves too noisy.

---

## 10.3 Stimulus-mix variants to test with the support interface

Treat support interface and Writer stimulus as separate factors.

### C0 — `support_only`

Purpose:

- pure evidence-writing probe,
- tests whether context is drowning support.

### C1 — `support_and_context_legacy`

Purpose:

- compatibility baseline.

### C2 — `support_and_context_gated`

Purpose:

- keep context available while preventing magnitude dominance.

Recommended default gate form:

- independent learned scalar gates or low-rank gates for context and support streams,
- initialized to favor support slightly in the early phase.

---

## 10.4 Support-interface acceptance targets

A support-interface change is promising only if it improves **source-side** diagnostics, not just downstream rank.

Look for:

- `writer_support_state_count > 1`
- `support_state_effective_rank > 1`
- lower slot pairwise cosine
- lower common-mode ratio
- better item coverage / slot specialization
- and ideally nonzero task-only Writer gradients

If projected rank improves but source-side rank stays at ~1, the interface change did not solve the actual bottleneck.

---

## 11. Anti-homogenization loss workstream

This is the second major experiment axis and explicitly answers your hard requirements.

---

## 11.1 Objective

Force the Writer to stop taking the lazy common-mode solution when task loss alone is too weak.

---

## 11.2 General design rules for all auxiliary losses

1. Apply the losses primarily to **Writer-side representations**:
   - support item states,
   - Writer slot outputs,
   - auxiliary projection head outputs.
2. Do **not** only regularize the projected Qwen-side prefix, because that would attack the wrong level.
3. Start with one auxiliary family at a time.
4. Only later test mixed auxiliary families.
5. Keep weights small and scheduled; over-regularization can create a new fake source of Writer gradients.

---

## 11.3 Auxiliary family L0 — task-only clean baseline

This is mandatory.

- Turn off legacy writer-side regularizers.
- Keep only task loss, basic safety caps, and standard optimization.

Purpose:

- measure real task-to-Writer gradient without contamination,
- establish whether structured support alone already helps.

Without this baseline, every future conclusion remains ambiguous.

---

## 11.4 Auxiliary family L1 — legacy writer regularizers baseline

Keep the current family as its own explicit baseline:

- gain margin
- common-mode penalty
- covariance diversity
- slot-energy balance

Purpose:

- compare the old auxiliary bundle fairly against new ones,
- rather than treating it as invisible default behavior.

---

## 11.5 Auxiliary family L2 — contrastive Writer loss

### Idea

Construct two stochastic views of the same support set and require Writer representations for the same example to be closer than representations from different examples.

### Candidate positive-pair construction

Generate two views via:

- support-row dropout,
- support-row masking,
- token masking inside support rows,
- mild row order perturbation (only when order is not semantically essential),
- context token dropout.

### Candidate representation target

Use either:

- pooled Writer slot representation,
- flattened slot representation passed through aux head,
- or slot-to-item aggregated representation.

### Candidate loss

- InfoNCE / NT-Xent style contrastive loss.

### Why it is valuable here

It explicitly punishes constant representations across different examples while preserving same-example consistency under mild perturbation.

---

## 11.6 Auxiliary family L3 — VICReg / VCReg-style variance-covariance regularization

### Idea

On Writer-side auxiliary embeddings, add:

- variance floor term,
- covariance penalty on off-diagonal entries,
- optional invariance term across two views.

### Why it is valuable here

This directly targets collapse and redundancy without requiring large negative sets.

### Recommended application points

Primary:

- Writer slot outputs
- auxiliary projection head outputs

Optional secondary:

- support item states

### Recommended notes

- If using only variance+covariance in supervised setting, a VCReg-style variant is acceptable.
- If using two views, VICReg-style invariance can also be added.

---

## 11.7 Auxiliary family L4 — Barlow-style redundancy reduction

### Idea

Create two stochastic views of the same support-conditioned example and push the cross-correlation matrix of their auxiliary embeddings toward identity.

### Why it is valuable here

This encourages:

- invariance across same-example views,
- decorrelation across dimensions,
- and collapse resistance.

### Caveat

Use a small auxiliary head so Barlow regularization shapes Writer-side semantics rather than distorting the main projector directly.

---

## 11.8 Auxiliary family L5 — cosine orthogonality / slot orthogonality

### Idea

Penalize excessive cosine similarity among Writer slots.

Example form:

`L_slot_ortho = mean_{i != j} max(0, cos(slot_i, slot_j) - tau)^2`

with a small tolerance `tau`.

### Why it is valuable here

It directly targets one observed pathology: slot representations drifting toward the same direction.

### Important caveat

Orthogonality alone is insufficient.
It can force arbitrary geometric separation without meaningful evidence specialization.

So this family should usually be paired with one of:

- item-coverage term,
- task loss,
- or contrastive/VICReg view loss.

---

## 11.9 Auxiliary family L6 — information bottleneck (VIB-lite)

### Idea

Introduce a stochastic bottleneck on Writer-side latent states:

- Writer predicts `mu, logvar`,
- sample latent via reparameterization,
- add small KL penalty to a prior.

### Why it is valuable here

It can discourage the Writer from memorizing arbitrary nuisance content while keeping task-relevant information.

### Major warning

A strong bottleneck can worsen collapse if applied too early or too strongly.

### Recommendation

- use only as a late auxiliary family,
- with small `beta`,
- only after task-only gradient is proven nonzero.

---

## 11.10 Auxiliary family L7 — dropout / masking injection

### Idea

Inject stochasticity into:

- support rows,
- support tokens,
- context tokens,
- Writer conditioning streams,
- auxiliary head.

### Why it is valuable here

This is both:

- an ordinary regularizer,
- and a view generator for contrastive / VICReg / Barlow objectives.

### Recommended knobs

- support-row dropout `0.1–0.2`
- context-token dropout `0.05–0.15`
- projector hidden dropout `0.0–0.1`

---

## 11.11 Auxiliary family L8 — attention specialization + coverage

This is highly relevant once multi-item support exists.

### Idea

When the Writer attends over multiple support items, add two complementary pressures:

1. **slot specialization:** each slot should not spread uniformly over all items,
2. **global coverage:** all slots together should not collapse onto the same one or two items.

### Example terms

- minimize per-slot attention entropy,
- minimize KL between aggregate item coverage and a broad target distribution,
- or use soft top-k coverage constraints.

### Why it is valuable here

This is the first auxiliary family that directly encourages “different slots look at different evidence.”

---

## 11.12 Recommended order for anti-collapse loss testing

1. `L0 task-only`
2. `L1 legacy`
3. `L2 contrastive`
4. `L3 VICReg / VCReg`
5. `L5 orthogonality + coverage`
6. `L4 Barlow`
7. `L7 dropout-only`
8. `L6 VIB-lite`
9. mixed families only after single-family screening

Reason:

- first establish real task gradient,
- then test the least confounded anti-collapse families,
- keep the more delicate or expensive ones later.

---

## 12. Mixed experiment program

This section explicitly answers your request to test combinations of support mode + anti-collapse measures + multi-item conditioning.

The key idea is **staged mixing**, not brute-force explosion.

---

## 12.1 Stage M0 — support-axis screening (single-factor)

Hold recipe constant and compare support interfaces under one baseline loss.

### Runs

- `S0 + C1 + L0`
- `S1 + C2 + L0`
- `S2 + C1 + L0`
- `S3 + C0 + L0`
- `S4 + C1 + L0`
- `S5 + C2 + L0`

Purpose:

- identify which support interfaces materially improve source rank / item coverage / Writer task gradients.

---

## 12.2 Stage M1 — anti-collapse screening on top support modes

Pick the best **two** support interfaces from M0 and test auxiliary families.

### Runs (example)

For top support modes `Sa` and `Sb`:

- `Sa + best stimulus mix + L0`
- `Sa + best stimulus mix + L1`
- `Sa + best stimulus mix + L2`
- `Sa + best stimulus mix + L3`
- `Sa + best stimulus mix + L5`
- `Sb + best stimulus mix + L0`
- `Sb + best stimulus mix + L1`
- `Sb + best stimulus mix + L2`
- `Sb + best stimulus mix + L3`
- `Sb + best stimulus mix + L5`

Additionally keep **one pooled baseline** in the matrix:

- `S0 + C1 + best anti-collapse family`

Purpose:

- test whether the best new loss can rescue `pooled_block`,
- versus whether non-pooled support is fundamentally necessary.

This directly addresses the concern that “loss may matter more than structure” while still testing the structure hypothesis fairly.

---

## 12.3 Stage M2 — small mixed matrix

Take the strongest support modes and strongest auxiliary families into a mixed matrix.

### Example 2x3x2 matrix

- support interfaces: top 2 of `{S2, S3, S4, S5}`
- auxiliary families: top 3 of `{L2, L3, L5, L8}`
- stimulus mix: top 2 of `{C0, C2}`

Total = `12` configs, manageable.

Purpose:

- detect synergy,
- confirm that improvements are not brittle one-off effects.

---

## 12.4 Stage M3 — recipe stabilization on finalists

Take the best `2–3` configs from M2 and sweep recipe variables:

- warmup length
- clipping scheme
- projector LR
- accumulation steps
- optional layer expansion

Purpose:

- move from “interesting” to “stable.”

---

## 12.5 Stage M4 — canonical confirmation runs

For finalists only:

- run longer horizon (`500–800` steps depending on task)
- multiple seeds (`>=3`)
- paired evaluation against no-memory control
- full summary bundle publication

This is where a real project-level claim can be made.

---

## 13. Recommended training recipe revisions

These are recipe-level changes I recommend before the next canonical run.

---

### 13.1 Shorten or remove the long Writer freeze

The current `50`-step freeze is no longer the right default.

Recommended variants:

- `F0`: no Writer freeze
- `F10`: 10-step warm-start
- `F20`: 20-step warm-start
- keep `F50` only as legacy comparator

Why:

- a long freeze encourages projector/receiver adaptation around a static Writer,
- which can make later Writer updates fight an already stabilized downstream bridge.

---

### 13.2 Reduce projector dominance

Recommended changes:

- lower projector LR from the current canonical setting
- keep Writer LR moderate
- keep receiver LoRA LR modest
- optionally add mild weight decay to projector only

Suggested starting point:

- Writer LR: `1e-4`
- support encoder LR: `7.5e-5`
- projector LR: `5e-5`
- receiver LoRA LR: `5e-5`

These are starting values, not doctrine.

---

### 13.3 Replace global clipping with group-aware clipping

Recommended default:

- Writer + support encoder: `clip 1.0`
- projector: `clip 0.5`
- receiver LoRA: `clip 0.5` or `1.0`

The exact values can be adjusted after diagnostics.

The key is decoupling Writer survival from projector explosions.

---

### 13.4 Use gradient accumulation if per-step variance is high

If effective batch size is tiny, use accumulation (e.g. `4`) for screening and canonical runs.

Why:

- smooths training,
- reduces single-example oscillation,
- especially useful on NarrativeQA / generative tasks.

---

### 13.5 Add prefix norm monitoring and soft restraint

The current prefix norm growth suggests norm-scaling compensation.

Add:

- prefix norm growth ratio to logs,
- optional soft penalty if prefix norm exceeds a configured multiple of initialization,
- but avoid overly hard norm clamps that kill learning.

---

### 13.6 Lower-confound baseline should disable writer-side auxiliary losses initially

The first new canonical rerun should not inherit the full legacy regularizer stack by default.

Reason:

- we need a clean read on task gradient,
- and we need the anti-collapse families to compete fairly.

---

## 14. Task ladder and evaluation program

---

## 14.1 Near-term task ladder

### Primary proof task

- **FEVER**

Reason:

- already materialized,
- already showed positive movement,
- theoretically more compatible with prefix-style biasing.

### Existing secondary diagnostics

- **NarrativeQA**: medium-stress generative diagnostic
- **GSM8K**: stretch diagnostic, not the first gate

### Recommended additional classification-style task

If you want a second proof task with minimal engineering,
consider extending the bundle with a classification-style source already scaffolded in `tasks/sources.py`, such as **Story Cloze**.

This is not required for the first PLANv6 steps, but it is a good medium-term improvement because it would give a second FEVER-like evaluation without needing to invent a whole new data source layer from scratch.

---

## 14.2 Evaluation rules

1. Keep deterministic medium slices for comparability.
2. Use paired with-memory / without-memory evaluation.
3. Report medians over seeds, not only best checkpoints.
4. Publish source-side and receiver-side metrics together.
5. Never report only `delta_answer_logprob`.

---

## 14.3 What counts as a real win

A real PLANv6 win on a task means **all** of the following:

- route is alive post-unfreeze,
- task-only Writer gradient is materially nonzero,
- source-side collapse metrics improve,
- task metrics improve or at least show a consistent positive trend,
- training is not just projector-dominated chaos.

---

## 15. Detailed PLANv6 execution order

This is the most operational section.

---

## Phase V6-0 — Measurement repair and instrumentation

### Goal

Make the repo capable of telling truth from measurement artifact.

### Mandatory code changes

- patch summary gate logic
- add gradient attribution probe
- add cosine similarity diagnostics
- add support-item attention diagnostics
- add group-aware clipping/logging
- add richer usefulness metrics

### Deliverables

- corrected summary script(s)
- unit tests
- one short dry-run summary proving the new logs exist

### Exit criteria

- summary no longer calls the frozen 1–50 Writer “dead”
- task-only vs aux-only Writer grads can be logged on probe steps
- support-item diagnostics appear for non-pooled support modes

---

## Phase V6-1 — Clean baseline rerun of current architecture

### Goal

Measure the current `pooled_block` route fairly before changing architecture.

### Canonical config

- support mode: `S0 pooled_block_legacy`
- stimulus: `C1` and optionally `C2`
- auxiliary family: `L0 task-only`
- freeze length: `F10` or `F0`
- same deep-prefix early4 + early receiver micro-LoRA bridge

### Why

This tells us whether the old support interface is doomed even without auxiliary confounds.

### Exit criteria

- obtain true post-unfreeze route verdict
- obtain first clean task-gradient attribution

---

## Phase V6-2 — Support-interface screening

### Goal

Test whether non-pooled support materially changes source collapse.

### Recommended configs

- `S0 + C1 + L0`
- `S1 + C2 + L0`
- `S2 + C1 + L0`
- `S3 + C0 + L0`
- `S4 + C1 + L0`
- `S5 + C2 + L0`

### Suggested run length

- short screening horizon first (e.g. `150–300` steps)
- 2 seeds if budget permits

### Primary readouts

- support state rank
- Writer slot rank
- task-only Writer gradient
- common-mode ratio
- item specialization / coverage
- FEVER usefulness metrics

### Exit criteria

Pick the top two support modes.

---

## Phase V6-3 — Anti-homogenization loss screening

### Goal

Test whether richer losses can make the Writer actually use the richer interface.

### On each top support mode, run

- `L0 task-only`
- `L1 legacy`
- `L2 contrastive`
- `L3 VICReg / VCReg`
- `L5 orthogonality + coverage`
- optionally `L4 Barlow`

### Required diagnostics

- task-only / aux-only gradient norms
- cosine(task, aux)
- collapse metrics
- usefulness metrics

### Interpretation logic

- if `L0` already works: loss redesign is optional, not mandatory
- if `L0` fails but `L2/L3/L5` help: loss inertia was real
- if only `L1 legacy` produces Writer grad but not task progress: old regularizers are likely creating misleading movement

---

## Phase V6-4 — Mixed matrix

### Goal

Test combined effects of support mode + anti-collapse family + stimulus mix.

### Suggested 12-run matrix

- top 2 support modes
- top 3 auxiliary families
- top 2 stimulus mixes

### Required reporting

For every run:

- route_live_post_unfreeze
- writer_task_supervision_live
- source_not_collapsed
- stable_training_v6
- usefulness_positive_v6

### Exit criteria

Select `2–3` finalists.

---

## Phase V6-5 — Recipe stabilization and layer comparison

### Goal

Convert interesting finalists into stable ones.

### Sweep variables

- warmup length: `0 / 10 / 20`
- clipping scheme: global vs group-aware
- projector LR: `5e-5 / 7.5e-5`
- accumulation: `1 / 4`
- layer set: base early `[0,1,2,3]` vs additive expansion `[0,1,2,3,4,8,14]`

### Important note

Layer expansion should be **additive**, not substitutive, because the current traffic is already strongest in the early stack.

### Exit criteria

- stable FEVER improvement across multiple seeds
- non-collapsed source diagnostics
- healthy task-to-Writer gradient attribution

---

## Phase V6-6 — Fallback architecture only if needed

### Trigger

Only enter this phase if all of the following are true:

- source-side collapse is materially improved,
- task-only Writer gradients are real,
- but usefulness remains weak and prefix-attention competition still looks limiting.

### Fallback

Implement **Prefix-Tuning+-style attention-independent injection** while preserving:

- independent Writer
- per-layer projector abstraction
- tiny receiver PEFT philosophy

### Why this is the right fallback

It changes the insertion physics without abandoning the main research thesis.

---

## Phase V6-7 — Only after success: Reader reopening, portability, usefulness scaling

Only after V6-5 or V6-6 succeeds should you reopen the macro roadmap:

- Reader/Fuser reintroduction as adaptive readout / compression layer
- usefulness-focused SFT branch
- preference-based fine-tuning only if deterministic usefulness already exists
- cross-reasoner portability experiment (same Writer, retrain projector + tiny receiver PEFT only)
- larger benchmark suite and CDMI-style macro goals

---

## 16. File-by-file implementation plan

---

### 16.1 `src/memtotal/training/m4_shared_injection.py`

#### Add

- gradient attribution probe
- cosine similarity logging
- group-wise clipping support
- new support interface modes
- new auxiliary loss modes
- support/context gating options
- richer eval metrics

#### New config flags (suggested)

- `pilot_gradient_probe_enabled`
- `pilot_gradient_probe_interval`
- `pilot_aux_loss_mode`
- `pilot_support_encoder_mode` extended values
- `pilot_writer_stimulus_mode`
- `pilot_context_support_balance_mode`
- `pilot_groupwise_grad_clip`
- `pilot_projector_grad_clip_norm`
- `pilot_writer_grad_clip_norm`
- `pilot_receiver_grad_clip_norm`
- `pilot_support_row_dropout`
- `pilot_context_token_dropout`
- `pilot_aux_projection_dim`
- `pilot_vib_beta`
- `pilot_contrastive_temperature`

---

### 16.2 `src/memtotal/models/memory.py`

#### Extend `WriterWeaverHead`

- optional return of support attention weights
- optional gating between context and support streams
- optional direct raw multi-item support path
- optional hybrid pooled+item path
- optional auxiliary projection head

#### New helper modules (suggested)

- `WriterAuxProjectionHead`
- `SupportContextBalanceGate`
- `HybridSupportFusion` (if `S5` is implemented as its own module)

---

### 16.3 `src/memtotal/models/backbone.py`

#### For now

No major architecture rewrite required for the mainline.

#### Later fallback hook

Prepare an abstraction point so that the same `WriterDeepPrefixProjector` output can be routed either to:

- current deep prefix cache injection,
- or future attention-independent injection.

This will make Phase V6-6 cleaner.

---

### 16.4 `src/memtotal/tasks/writer_jointpeft_data.py`

#### Near-term

Keep FEVER / GSM8K / NarrativeQA bundle support.

#### Medium-term optional improvement

Add one classification-style benchmark split builder (e.g. Story Cloze if desired), reusing existing source materialization support.

---

### 16.5 New scripts

Suggested additions:

- `scripts/run_planv6_probe_qwen25.sh`
- `scripts/run_planv6_support_sweep_qwen25.sh`
- `scripts/run_planv6_loss_sweep_qwen25.sh`
- `scripts/run_planv6_mix_qwen25.sh`
- `scripts/update_planv6_probe_summary.py`
- `scripts/update_planv6_support_summary.py`
- `scripts/update_planv6_loss_summary.py`
- `scripts/update_planv6_mix_summary.py`

---

### 16.6 Tests

Add tests for:

- post-unfreeze gate logic
- gradient attribution math
- support interface tensor shapes
- support attention diagnostics
- group-wise clipping path
- auxiliary loss forward pass / serialization
- no-memory control compatibility

---

## 17. Decision table for interpreting outcomes

This table is critical. It prevents wishful reading.

### Case A

- task-only Writer grad ≈ zero
- aux-only Writer grad large
- source collapse unchanged

**Interpretation:** Writer is not being meaningfully supervised by the task. Fix loss/recipe first.

**Next move:** reduce or remove legacy regularizers; improve support interface; retest.

---

### Case B

- task-only Writer grad nonzero
- source collapse unchanged
- projected rank high
- usefulness weak

**Interpretation:** task signal exists, but Writer still finds common-mode shortcut.

**Next move:** anti-homogenization losses, especially contrastive / VICReg / orthogonality+coverage.

---

### Case C

- task-only Writer grad nonzero
- source collapse improves
- FEVER improves
- GSM8K still weak

**Interpretation:** architecture is working in its expected regime.

**Next move:** continue FEVER-first ladder; do not declare failure because GSM8K is still hard.

---

### Case D

- source collapse improves
- task grad improves
- usefulness still weak
- prefix attention seems to compete poorly

**Interpretation:** insertion contract may now be the bottleneck.

**Next move:** Prefix-Tuning+ style attention-independent branch.

---

### Case E

- pooled_block + best loss still fails
- non-pooled support + same loss succeeds

**Interpretation:** support structure is the real critical variable.

**Next move:** retire pooled_block from canonical use.

---

### Case F

- pooled_block + best loss succeeds almost as well as non-pooled support

**Interpretation:** loss design was more limiting than support structure.

**Next move:** keep pooled path as cheap baseline but prefer the clearer non-pooled scientific story if its diagnostics are healthier.

---

## 18. What not to do now

1. **Do not call the current architecture falsified.**
2. **Do not reopen Reader/Fuser yet.**
3. **Do not increase Writer width/depth as the first move.**
4. **Do not jump to GRPO / DPO / RL.**
5. **Do not judge by mean `delta_answer_logprob` alone.**
6. **Do not keep `pooled_block` as invisible default.**
7. **Do not keep all legacy writer regularizers as fixed background assumptions.**
8. **Do not keep global clipping without checking whether projector domination is starving Writer gradients.**
9. **Do not use GSM8K as the first hard gate for architectural survival.**
10. **Do not open MemGen-style reasoner-coupled writing before fairly testing structured-support independent Writer.**

---

## 19. Recommended default canonical configuration for the next serious run

This is the best single next canonical run after Phase V6-0 instrumentation is done.

### Canonical candidate: `V6-CANON-1`

- Writer architecture: current W0 `WriterWeaverHead`
- support mode: `S2 structured_support_set` **or** `S3 multi_item_cross_attn_raw` (run both if budget permits; choose one canonical and one comparator)
- stimulus mix: `C2 support_and_context_gated`
- auxiliary family: `L0 task-only` first, then `L2` or `L3` as comparator
- deep prefix layers: `[0,1,2,3]`
- receiver LoRA: early4 `k_proj/v_proj`, rank 2, alpha 4
- Writer freeze: `10` steps or none
- steps: `500–600`
- group-wise clip: enabled
- projector LR: reduced relative to current canonical run
- FEVER as primary target
- NarrativeQA / GSM8K as diagnostics, not fatal gates

### Why this is the right next canonical run

It changes the smallest number of things necessary to test the most likely bottlenecks fairly:

- support structure,
- gradient attribution clarity,
- and recipe stability.

It does **not** contaminate the verdict by reopening Reader or changing the overall research thesis.

---

## 20. Long-horizon research story if PLANv6 succeeds

The strongest eventual paper-level novelty is not simply “another deep prompt method.”

It is:

> **independent Writer + reasoner-specific projector interface + optional later Reader adapter, with the long-term goal of cross-reasoner portability.**

That differentiates the project from tighter reasoner-coupled latent memory systems.

If PLANv6 succeeds, the most valuable later experiment is:

- keep the Writer fixed,
- switch reasoner backbone,
- retrain only projector + tiny receiver PEFT,
- test whether usefulness transfers.

That would be much more novel than merely getting another single-model latent-memory result.

---

## 21. Final recommendation in plain language

Here is the blunt version.

The current project is **not** stuck because the deep-prefix route is impossible.
It is stuck because the last canonical run combined:

- an invalid route gate,
- a support interface that collapses multiple evidence items into one state,
- a context stream that may be numerically overpowering support,
- Writer-side regularizers that muddy gradient interpretation,
- and projector-dominated clipping that may be attenuating the very gradients the project needs.

So the next move is very clear:

> **repair the measurement, isolate `pooled_block` as a true variable, test real multi-item support conditioning, add direct anti-homogenization experiments, and run structured mixed experiments before any larger pivot.**

That is the fairest, most scientifically defensible, and most execution-stable next route.

---

## 22. Condensed execution checklist

### Code / instrumentation

- [ ] patch post-unfreeze route gate
- [ ] add gradient attribution probe
- [ ] log gradient cosine similarity
- [ ] add support attention diagnostics
- [ ] add group-wise clipping or at least group-wise clip logs
- [ ] add richer classification and generation usefulness metrics

### Support axis

- [ ] keep `pooled_block` baseline
- [ ] add `pooled_block_gated`
- [ ] run `structured_support_set`
- [ ] run `multi_item_cross_attn_raw`
- [ ] run `multi_item_cross_attn_encoded`
- [ ] run `hybrid_pooled_plus_items`

### Loss axis

- [ ] clean `task-only` baseline
- [ ] legacy writer regularizers as explicit baseline
- [ ] contrastive loss
- [ ] VICReg / VCReg-style loss
- [ ] orthogonality + coverage loss
- [ ] Barlow-style loss
- [ ] VIB-lite (later)
- [ ] dropout / masking view generator

### Mixed experiments

- [ ] support-axis screening
- [ ] loss-axis screening on top support modes
- [ ] small mixed matrix
- [ ] recipe stabilization on finalists
- [ ] canonical multi-seed confirmation

### Hard stop rules

- [ ] stop using frozen-window Writer grad as liveness gate
- [ ] stop using mean `delta_answer_logprob` as sole usefulness criterion
- [ ] stop widening Writer before support/loss/program is fairly tested
- [ ] stop reopening Reader before Writer usefulness is real

---

## 23. Selected references and relevant open-source implementations

Below are the external references I recommend explicitly citing or keeping nearby while implementing PLANv6.

### Prefix / PEFT / injection literature

1. **Prefix-Tuning: Optimizing Continuous Prompts for Generation**  
   Paper: https://arxiv.org/abs/2101.00190  
   Code: https://github.com/XiangLi1999/PrefixTuning

2. **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**  
   Paper: https://arxiv.org/abs/2110.07602  
   Code: https://github.com/THUDM/P-tuning-v2

3. **LoRA: Low-Rank Adaptation of Large Language Models**  
   Paper: https://arxiv.org/abs/2106.09685  
   Code: https://github.com/microsoft/LoRA

4. **Hugging Face PEFT**  
   Docs: https://huggingface.co/docs/peft/index  
   Code: https://github.com/huggingface/peft

5. **When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations**  
   Paper: https://arxiv.org/abs/2310.19698

6. **Prefix-Tuning+: Modernizing Prefix-Tuning by Decoupling the Prefix from Attention**  
   Paper: https://arxiv.org/abs/2506.13674

### Latent-memory / latent-reasoning adjacent work

7. **SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs**  
   Paper: https://aclanthology.org/2025.acl-long.1137/  
   Code: https://github.com/xuyige/SoftCoT

8. **MemGen: Weaving Generative Latent Memory for Self-Evolving Agents**  
   Paper / OpenReview entry: https://openreview.net/forum?id=8tr13s8mG5  
   Code: https://github.com/VectorSpaceLab/MemGen

9. **LightThinker: Thinking Step-by-Step Compression**  
   Paper: https://arxiv.org/abs/2502.15589  
   Code: https://github.com/zjunlp/LightThinker

10. **CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation**  
    Paper: https://arxiv.org/abs/2502.21074

### Anti-collapse / representation regularization references

11. **VICReg: Variance-Invariance-Covariance Regularization**  
    Paper: https://arxiv.org/abs/2105.04906  
    Code: https://github.com/facebookresearch/vicreg

12. **Barlow Twins: Self-Supervised Learning via Redundancy Reduction**  
    Paper: https://arxiv.org/abs/2103.03230  
    Code: https://github.com/facebookresearch/barlowtwins

13. **Variance-Covariance Regularization Improves Representation Learning (VCReg)**  
    Paper: https://arxiv.org/abs/2306.13292  
    Example code: https://github.com/Algomancer/VCReg

14. **Deep Variational Information Bottleneck**  
    Paper: https://arxiv.org/abs/1612.00410  
    Demo code: https://github.com/alexalemi/vib_demo

15. **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**  
    Paper: https://arxiv.org/abs/2002.05709  
    Google Research repo: https://github.com/google-research/simclr

16. **Representation Learning with Contrastive Predictive Coding (InfoNCE / CPC)**  
    Paper: https://arxiv.org/abs/1807.03748

17. **Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?**  
    Paper: https://arxiv.org/abs/1810.09102

18. **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**  
    Paper: https://jmlr.org/papers/v15/srivastava14a.html

---

## 24. Final one-paragraph verdict

**Do not pivot away from the independent Writer. Fix the invalid route gate, prove task-gradient-to-Writer explicitly, treat `pooled_block` as a real variable instead of an invisible default, move the Writer to structured multi-item conditioning, run dedicated anti-homogenization loss experiments, and then test mixed combinations in a staged matrix. If that corrected branch still underdelivers after source collapse is materially reduced and task-only Writer gradients are real, then the next architectural fallback is Prefix-Tuning+-style attention-independent injection — not immediate Reader reopening, not Writer scaling, and not MemGen imitation.**
