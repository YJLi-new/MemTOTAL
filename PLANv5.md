# PLANv5 — Gradient-First Independent Writer Recovery
**Subtitle:** Deep-Prefix, Per-Layer Projectors, Circuit Opening, and Minimal Joint PEFT  
**Date:** 2026-03-09  
**Language:** English  
**Status:** Active replacement for the narrow next-step branch after `PLANv4.md` reached its own terminal gates on the `W0 -> F1a -> F1b` path.

---

## 0. Scope, Authority, and Intent

This document is **not** a replacement for the original macro roadmap in `PLAN.md`. The macro order remains valid:

- the project still has to cover **Qwen2.5-1.5B-Instruct** and later **Qwen3-8B**,
- the project still has to refresh **transfer evidence**,
- and **CDMI** remains a **P0 deliverable**.

What changes here is the **narrow active branch**.

`PLANv4.md` correctly moved the project away from Reader rescue and toward **Writer-first, Weaver-inspired** training. However, the actual `W0 / F1a / F1b` execution chain produced a stronger conclusion than “Writer-first failed”:

> **The current Writer-first branch only disproved `independent Writer + shallow embedding-level prefix + frozen receiver (+ writer-only fallback)` as a useful substrate. It did _not_ yet test the user’s intended contract: deep prefix, gradient circuit opening first, and joint PEFT via LoRA while keeping Qwen base weights frozen.**

This document therefore proposes a **PLANv5 reboot** that keeps the **independent Writer** idea alive, while changing the training physics of the bridge so that the following can become true **before serious Writer training begins**:

1. the prefix is a **deep prompt** that directly participates in selected Qwen layers’ key/value computations,
2. the bridge has a **real gradient circuit** from Qwen attention back into the prefix source,
3. and Qwen is allowed a **minimal LoRA-only adaptation** (base weights frozen) to learn to route useful attention to the prefix when needed.

This plan also incorporates the additional user-provided route:

- do **not** throw away the independent-Writer thesis too early,
- use **per-layer projectors** and **deep KV injection** to satisfy the same core conditions that make deep prompt methods work,
- and treat a **Qwen-native/MemGen-style Writer** as a **fallback**, not the primary route.

---

## 1. Executive Summary

### 1.1 My independent bottom-line diagnosis

The project’s current blocker is **not** best described as “Reader geometry,” and it is also **not** accurately described as “there is absolutely no gradient to the Writer in every branch.”

The more precise diagnosis is:

> **The active Writer-first branch is suffering from _gradient usefulness starvation_: under shallow embedding-level prefix injection and a frozen receiver, optimization can be mechanically live, but the easiest loss-reducing behavior is to push the projector and/or Writer toward a safe, high-common-mode, near-collinear prefix regime that Qwen can largely ignore. In F1b, the writer-side adapter becomes effectively dead because that same contract is too upstream and too weak to create an actionable learning surface.**

So the real issue is not simply **“no learning”**. It is:

- sometimes **nonzero but useless** gradients (W0 / F1a),
- sometimes **dead adapter gradients** (F1b),
- with **no non-FEVER usefulness signal** in either case.

That means the next step should **not** be:
- more shallow-prefix Writer capacity,
- more writer-only adapter cleanup,
- more Reader/Fuser rescue,
- or immediate MemGen-style full route change.

The next step should be:

> **Open the gradient circuit under the intended contract: independent Writer -> per-layer deep prefix -> selected Qwen layers -> loss, with minimal joint LoRA if needed, before asking the Writer to learn usefulness.**

### 1.2 PLANv5 in one sentence

**PLANv5 reboots the active branch around an independent Writer with per-layer deep projectors, uses multi-layer KV injection instead of shallow concat, explicitly calibrates the Qwen attention-to-prefix gradient loop _before_ Writer training, and only then reopens Writer usefulness optimization (with tiny receiver LoRA as a narrow, earlier tool rather than a last-ditch fallback).**

### 1.3 Why this is the right next move

This is the narrowest plan that simultaneously respects all of the following:

- your original macro research story,
- the negative evidence from `PLANv3` and `PLANv4`,
- your hard requirement that **prefix means real deep prompt / KV injection**,
- your hard requirement that **LoRA may be used while Qwen base weights remain frozen**,
- and your latest request to keep the **independent Writer** path alive rather than collapsing immediately into MemGen.

---

## 2. Ground Truth from the Current Repo and Docs

This section is the reality check: what the code and artifacts actually say right now.

### 2.1 What the latest narrow branch actually proved

#### `W0`
Published summary: `comparison_conclusion = plumbing_only`, `move_to_w1 = false`, `stop_after_w0 = true`.

Interpretation:
- support/context wiring is real,
- Writer-direct geometry changes are real,
- but non-FEVER `delta_answer_logprob` remained `0.0`.

#### `F1a`
Published summary: `comparison_conclusion = move_to_f1b`, `move_to_w2 = false`, `move_to_f1b = true`.

Interpretation:
- making the Writer bigger did **not** create usefulness,
- geometry did **not** improve meaningfully,
- common-mode collapse got dramatically worse.

#### `F1b`
Published summary: `comparison_conclusion = stop_after_f1b`, `move_to_w2 = false`, `move_to_f2 = false`, `stop_after_f1b = true`.

Interpretation:
- the writer-side adapter path was mechanically wired,
- but never became a live learning surface,
- common-mode energy stayed pinned near `1.0`,
- and non-FEVER `delta_answer_logprob` remained exactly `0.0`.

#### `V4`
Published summary: `comparison_conclusion = failure`, `primary_interpretation = no_multitask_signal`, `move_to_v5 = false`, `stop_after_v4 = true`.

Interpretation:
- there was no bridge rescue on NarrativeQA or GSM8K,
- FEVER did not generalize into broader evidence,
- and the old branch correctly closed itself.

### 2.2 What the code currently does in the active Writer-first path

From the repo state described in the latest branch:

- `WriterWeaverHead` exists,
- `writer_direct` bridge mode exists,
- `backbone.py` already contains support for **deep prefix / per-layer hidden injection**,
- but the active `W0 / F1a / F1b` branch was run with **`shallow_prefix`** templates,
- and `m4_shared_injection.py` currently contains a runtime guard that prevents `writer_direct` from using receiver micro-LoRA.

That is crucial. It means the branch that just “failed” **did not** test the intended contract:

- not deep prompt,
- not joint minimal receiver LoRA,
- not “gradient circuit first.”

### 2.3 Why this matters

The current `PLANv4` branch therefore answered a narrower question than the one we actually care about.

It answered:

> Can an independent Writer become useful under shallow embedding-level prefix injection, a frozen receiver, and shallow writer-only remediation?

The answer is currently: **no**.

But the next question—the one that matters now—is different:

> Can an independent Writer become useful if its outputs are translated into Qwen’s **per-layer KV space**, with **multiple direct gradient channels**, and with a **minimal receiver-side LoRA** available when needed to open attention routing?

That question is still open.

---

## 3. Independent Mechanistic Diagnosis

This is my own bottom-up explanation, not just a paraphrase of prior summaries.

## 3.1 The project is not “failing to learn” in a generic sense

The wrong story is:

> “The system gets no gradients, so nothing is learning.”

That is too coarse.

A more accurate story is:

- in **W0** and **F1a**, the system can be mechanically trainable,
- but the optimization signal is being absorbed into a **safe common-mode projection regime**,
- while in **F1b**, the writer-side adapter is too upstream and too weak under the current injection contract, so it never becomes live.

So the true problem is:

> **not “no optimization,” but “optimization is routed into the wrong degrees of freedom.”**

This is why some branches show loss movement or geometry movement without usefulness movement.

## 3.2 The key physical bottleneck is the injection contract

Under **shallow embedding-level prefix concat**, the prefix is introduced only once, at the input embedding interface.

That creates two problems:

### Problem A — OOD entry point
The Writer/projector emits synthetic vectors that Qwen did not pretrain to interpret as prefix tokens.

### Problem B — long gradient path
If the loss wants the prefix to matter, the backprop signal has to survive:
- the attention routing decisions,
- many residual blocks,
- many nonlinear transforms,
- and the entire depth from the top loss back to the input concat point.

This is precisely where the current bridge becomes fragile.

## 3.3 Why the current branch converges toward a “silent prefix”

If Qwen can reduce training loss more easily by treating the prefix as something harmless than by genuinely learning to use it, the training dynamics tend to settle into a regime where:

- the prefix has high common-mode structure,
- token-to-token diversity is weak,
- projected memory is near-collinear,
- and the answer distribution barely changes.

This is exactly the regime the current branch appears to enter.

In plain terms:

> **The projector learns how to make the prefix non-disruptive faster than the system learns how to make it useful.**

That explains why “plumbing works” but utility does not appear.

## 3.4 Why “Writer-first” was still the right instinct, but under the wrong physics

You were right to deprioritize Reader/Fuser after the previous failure chain. The current evidence is consistent with:

- garbage in -> garbage out at the memory source,
- so Reader/Fuser was unlikely to save the system.

But “Writer-first” only works if the Writer can receive **actionable** gradients.

Under the shallow/frozen contract, the Writer is not being trained in the regime that best matches the independent memory thesis.

So the right correction is **not** “abandon Writer-first.” It is:

> **Keep Writer-first, but switch to a bridge whose gradient geometry matches the problem.**

## 3.5 The most important clarification: it is not enough to say “add LoRA”

Adding LoRA is directionally reasonable, but “LoRA” by itself is not a plan.

The real questions are:

- **where** to put it,
- **when** to turn it on,
- **what** it is supposed to fix,
- and **how** to prevent it from masking a dead Writer rather than opening a useful circuit.

My answer is:

- use **tiny** receiver LoRA,
- on **selected early or injection-adjacent layers**,
- only after a **circuit-audit baseline** is in place,
- and only inside a **deep-prefix** bridge, not the old shallow concat contract.

---

## 4. The Key Theoretical Correction: What Deep Prefix Fixes

This section incorporates the additional route you provided and makes it central.

## 4.1 The shallow-prefix gradient problem in attention form

At one layer of self-attention, if prefix keys/values are concatenated with content keys/values:

```text
Keys   = [P_k ; X_k]
Values = [P_v ; X_v]
α      = softmax(Q [P_k ; X_k]^T / sqrt(d))
output = α [P_v ; X_v]
```

Then:

- the gradient to `P_v` is multiplied by the attention mass placed on the prefix,
- and the gradient to `P_k` depends on the softmax Jacobian and query direction.

If prefix attention is tiny, then value gradients become tiny immediately. Key gradients are not mathematically impossible, but under a long shallow path they can become numerically negligible by the time they reach the prefix source.

This is exactly why shallow-prefix branches so often collapse into “prefix present but ignored.”

## 4.2 What P-Tuning v2 changes

The key advantage of deep prompt methods is not merely “more prompt capacity.”

It is:

> **each injection layer creates its own local gradient channel.**

Instead of relying on one injection point at the embedding layer, a deep prefix provides per-layer learnable hidden states (or equivalent per-layer K/V injections). Then each selected layer can produce its own:

- forward attention route to prefix,
- and backward signal back to the prefix source.

That is a much healthier learning geometry for medium-size models and harder NLU/generative tasks.

## 4.3 How to preserve an independent Writer without becoming MemGen

There are two broad ways to solve the distribution-gap problem:

### Route A — MemGen-style
Make the Writer part of Qwen itself (e.g. a LoRA-augmented internal weaver).

Advantage:
- machine-native hidden states automatically.

Disadvantage:
- loses Writer independence,
- weakens cross-backbone portability story,
- moves the project closer to MemGen’s architectural lane.

### Route B — Independent Writer + per-layer projectors
Keep the Writer external and independent, but translate its outputs into Qwen’s **layer-local KV space** using **per-layer projectors**.

Advantage:
- preserves Writer independence,
- preserves “generic memory + backbone-specific adapters” research story,
- matches your original project’s broader ambition.

This plan chooses **Route B as primary**, with **Route A only as a fallback**.

## 4.4 The missing ingredient: initialization that lands inside Qwen’s usable KV manifold

Even with deep prefix, randomly initialized per-layer projectors may still emit strongly OOD hidden states.

So PLANv5 introduces a missing piece that earlier plans did not make central enough:

> **native-KV initialization / semantic anchoring.**

We will estimate real Qwen per-layer key/value statistics on calibration data and initialize per-layer projectors so their outputs begin near the mean/scale of the real model’s KV manifold.

This can be done in one of two ways:

### Method 1 — KV-statistics initialization
For each selected layer:
- run Qwen on calibration prompts,
- collect real token K/V statistics,
- initialize projector bias and scale to match those statistics.

### Method 2 — Semantic anchor initialization
Use Qwen-native anchor tokens or anchor hidden states (e.g. “Answer:”, “Reasoning:”, prompt-boundary states) and initialize projector outputs toward those per-layer states.

The important point is:
- **no Qwen base weights change**, and
- the deep prefix starts inside a more plausible local manifold.

---

## 5. Architecture Feasibility Cross-Examination

This section is the “hard questioning” part.

## 5.1 What is no longer viable

The following should be treated as closed or near-closed within the current branch family:

1. **Shallow embedding-level prefix as the main Writer contract.**  
   This is no longer a plausible primary path.

2. **Writer-only capacity increases without circuit repair.**  
   F1a already made clear that bigger Writer capacity can worsen collapse.

3. **Writer-only micro-LoRA under shallow/frozen conditions.**  
   F1b already showed this can remain dead.

4. **Reader/Fuser rescue as the next thing.**  
   This is still the wrong priority.

## 5.2 What remains clearly viable

### Viable A — Independent Writer + deep prefix + per-layer projectors
This is the most faithful continuation of MemTOTAL’s original spirit.

### Viable B — Tiny receiver LoRA with Qwen base frozen
This is completely compatible with “not touching Qwen base weights.”  
LoRA is exactly a method that keeps base weights frozen while adding trainable low-rank adapters.

### Viable C — Multi-task validation from the start
We should not go back to FEVER-only narrow loops.

The Tier-1 set should remain:
- GSM8K,
- NarrativeQA,
- FEVER-label-gen.

Optionally add a Tier-2 task later if and only if the bridge becomes useful.

## 5.3 What must become a hard precondition

Before any substantial Writer training begins, the active branch must prove the following on at least one non-FEVER task:

1. some selected Qwen layer actually routes measurable attention to the deep prefix,
2. there is nontrivial gradient from answer loss back into the prefix source,
3. and the prefix changes answer log-probability in a non-degenerate way.

This is the **Circuit-Open Precondition**.  
Nothing beyond that is authorized until it is passed.

---

## 6. PLANv5 Strategy Overview

PLANv5 is a **concentrated-route** plan, not a sprawling tree.

It deliberately keeps two tightly related routes alive:

### Route I — Independent Writer, no receiver LoRA first
**Purpose:** test whether deep prefix + per-layer projectors + good initialization are already enough to open the circuit while preserving maximum architectural purity.

### Route II — Independent Writer + tiny receiver LoRA
**Purpose:** if Route I is too weak, add the narrowest possible receiver-side help without giving up the independent Writer story.

### Shared rules
Both routes share the same constraints:

- no Reader/Fuser yet,
- no return to shallow prefix,
- no bigger Writer experiments before circuit opening,
- no FEVER-only loops,
- no Qwen3 reopen yet,
- no CDMI reopen yet.

---

## 7. The Canonical PLANv5 Hypothesis

The central hypothesis of PLANv5 is:

> **The external Writer hypothesis is still alive if and only if the Writer is allowed to write into Qwen through per-layer deep-prefix translators that start inside Qwen’s layer-local KV manifold, with minimal early-layer receiver LoRA available only if necessary to open attention routing.**

Equivalent formulation:

- The current failure does **not** force a MemGen pivot yet.
- It forces a **bridge-physics correction**.

---

## 8. Workstream P0 — Circuit Audit and Code Unblocking

## 8.1 Purpose

Make the codebase capable of testing the intended contract.

## 8.2 Required code changes

### `src/memtotal/training/m4_shared_injection.py`
1. Remove or relax the current guard that rejects:
   - `bridge_mode = writer_direct`
   - together with `receiver_lora_enabled = true`.

New rule:
- `writer_direct + receiver micro-LoRA` is allowed **only when**
  - `pilot_injection_mode = sparse_deep_prefix`,
  - and target modules/layers are from an approved tiny receiver set.

2. Add an explicit pre-flight validation block for:
- `writer_direct`,
- `sparse_deep_prefix`,
- selected per-layer projectors,
- and optional receiver LoRA.

3. Add a `source_stub` mode for pre-Writer circuit calibration:
- a tiny trainable latent source or minimal source mapper,
- used to prove the deep-prefix route is alive before Writer training.

4. Add new diagnostics:
- per-layer prefix attention mass,
- per-layer prefix K/V output norm,
- per-layer prefix-vs-content attention ratio,
- source grad norm,
- writer grad norm,
- projector grad norm,
- receiver LoRA grad norm,
- no-memory vs memory answer-logprob delta by task.

### `src/memtotal/models/backbone.py`
1. Reuse the existing deep-prefix interface as the default active path for PLANv5.
2. Add lightweight utilities for:
- collecting per-layer key/value statistics,
- extracting anchor hidden states / anchor token KV states,
- and optionally logging layer-local prefix attention diagnostics on selected eval steps.

### `src/memtotal/models/memory.py`
1. Keep `WriterWeaverHead`, but do **not** increase capacity further yet.
2. Add a **per-layer projector family**:
   - `WriterDeepPrefixProjector`,
   - mapping Writer outputs to selected layers’ deep-prefix hidden states.

3. Add projector initialization modes:
- `random`,
- `kv_stat_match`,
- `semantic_anchor`,
- `hidden_state_anchor`.

4. Add optional per-layer norm caps / scale parameters to avoid catastrophic layer explosions.

## 8.3 New config families

### Method configs
- `configs/method/writer_direct_deep_prefix_no_lora_early4.yaml`
- `configs/method/writer_direct_deep_prefix_receiver_lora_r2_early4.yaml`
- `configs/method/writer_direct_deep_prefix_receiver_lora_r4_early4.yaml`
- `configs/method/writer_direct_deep_prefix_receiver_lora_r2_all5.yaml`

### Exp templates
- `configs/exp/writer_circuit_g1_prefix_only_gsm8k_template.yaml`
- `configs/exp/writer_circuit_g1_prefix_only_narrativeqa_template.yaml`
- `configs/exp/writer_circuit_g1_prefix_only_fever_template.yaml`
- `configs/exp/writer_circuit_g1_source_stub_*`
- `configs/exp/writer_circuit_g2_writer_direct_*`
- `configs/exp/writer_circuit_g3_anchor_*`

## 8.4 New scripts

- `scripts/run_writer_circuit_opening_qwen25.sh`
- `scripts/update_writer_circuit_opening_summary.py`
- `scripts/run_writer_deep_prefix_jointpeft_qwen25.sh`
- `scripts/update_writer_deep_prefix_jointpeft_summary.py`
- `scripts/run_writer_hidden_state_anchor_qwen25.sh`
- `scripts/update_writer_hidden_state_anchor_summary.py`

## 8.5 Acceptance

P0 is complete only if:
- the runtime supports `writer_direct + sparse_deep_prefix + optional receiver_lora`,
- the new summary scripts run,
- and tests cover these contracts.

---

## 9. Workstream P1 — Route I: Circuit Opening Without Receiver LoRA

This stage respects the strongest independent-Writer purity.

## 9.1 Purpose

Test whether an independent Writer-style source can create a live deep-prefix circuit **without** receiver LoRA, provided that:
- prefix is deep,
- projectors are per-layer,
- and projector initialization is Qwen-aware.

## 9.2 Substage P1a — Prefix-only calibration

### Idea
Do **not** start with the full Writer.  
Start with the smallest deep-prefix source that can prove the physics.

Use one of:
- free trainable deep-prefix tensors,
- or a tiny source stub mapped into per-layer projectors.

### Why
This isolates whether:
- deep prefix itself can change answers,
- and the selected layers can produce usable gradients,
before the Writer is blamed.

### Canonical setting
- backbone: `Qwen2.5-1.5B-Instruct`
- tasks: GSM8K, NarrativeQA, FEVER-label-gen
- injection: `sparse_deep_prefix`
- selected layers: **[0, 1, 2, 3]** as canonical early-stack
- no receiver LoRA
- source: prefix-only or source-stub
- projector init: `kv_stat_match`
- comparator init: `semantic_anchor`

### Why early4
The purpose of this stage is **circuit opening**.  
If the model is going to learn to route attention to the prefix, that is most plausibly decided in the early routing layers.

### Loss
- answer-span CE or task-native generative CE,
- plus paired no-memory answer-logprob delta metric,
- optional mild prefix norm regularization.

### Gate
P1a passes only if **at least one non-FEVER task** shows:
- nonzero positive `delta_answer_logprob`,
- source grad norm above floor,
- and nontrivial prefix attention mass in at least one selected layer.

Suggested floor:
- `source_grad_norm_steps_1_4_median > 1e-6`
- `delta_answer_logprob != 0` on at least one non-FEVER eval slice

No need to require final task-score improvement yet.  
P1a is a circuit test, not a full usefulness claim.

## 9.3 Substage P1b — Writer-direct, still no receiver LoRA

If P1a passes, replace the source stub with the real `WriterWeaverHead`, but keep:
- the same deep-prefix route,
- the same per-layer projector,
- the same selected layers,
- and the same initialization.

### Canonical setting
- keep Writer shape close to W0, not F1a bigger-writer
- keep support+context stimulus
- keep latent token budget fixed
- do not open Reader/Fuser
- do not add receiver LoRA yet

### Additional losses
On top of task CE:
- no-memory delta-gain margin,
- anti-common-mode penalty,
- token energy balance penalty,
- layerwise projector distribution-anchor loss (decayed over time).

### Why W0-shaped Writer
F1a already showed that bigger capacity can magnify collapse when the bridge contract is wrong.

### Gate
P1b passes weakly if:
- at least one non-FEVER task gets positive `delta_answer_logprob`,
- Writer grad norms remain nontrivial,
- common-mode energy is not worse than W0,
- and projected memory is not fully rank-1.

P1b passes medium if:
- one non-FEVER task improves actual task score versus no-memory control,
- while the others remain non-regressive.

### If P1b fails
Do **not** go back to bigger Writer or writer-only adapter.
Open Route II.

---

## 10. Workstream P2 — Route II: Tiny Receiver LoRA Inside the Deep-Prefix Contract

This stage implements your hard requirement that LoRA be used to jointly fine-tune Qwen while keeping base weights frozen.

## 10.1 Purpose

If deep-prefix + per-layer projectors still do not open a sufficient circuit, use the smallest reasonable receiver-side LoRA to teach Qwen to route attention to prefix.

## 10.2 Design principles

1. **Qwen base weights remain frozen.**
2. Use **LoRA only**, no full-weight tuning.
3. Keep the LoRA footprint **smaller than MemGen-style broad reasoning adapters**.
4. Target only:
   - `k_proj`,
   - `v_proj`,
   - on a tiny set of selected layers.

## 10.3 Canonical target layers

### Primary canonical
- layers `[0, 1, 2, 3]`

### Comparator
- sparse all-depth `[0, 7, 14, 21, 27]`

### Why not late3 as the canonical setting
The old late-layer setting was not a decisive test of receiver PEFT.  
By late layers, routing may already have committed away from the prefix.

## 10.4 Canonical LoRA hyperparameters

### Primary
- rank `r = 2`
- alpha `= 4`
- dropout `= 0.0` or minimal
- target modules: `k_proj`, `v_proj`

### Comparator
- rank `r = 4`
- alpha `= 8`

This keeps the intervention tiny.

## 10.5 Substage P2a — Source-stub + receiver LoRA

Before reintroducing the Writer, confirm that:
- deep prefix + early receiver LoRA
can create a live and useful route.

If P2a cannot produce nonzero usefulness signal, then:
- the branch is not ready for Writer training,
- and the next step should be a more radical bridge rethink.

## 10.6 Substage P2b — Writer-direct + receiver LoRA

If P2a passes:
- turn on the W0-shaped Writer,
- keep per-layer projectors,
- keep deep prefix,
- keep support+context stimulus,
- and jointly optimize:
  - Writer,
  - per-layer projectors,
  - tiny receiver LoRA.

### Training schedule
**Important:** before Writer training begins in earnest, do a short receiver/projector warm-start.

#### Phase 1 — circuit warm-start
- freeze Writer,
- train projector + receiver LoRA only for a tiny number of steps,
- goal: get the prefix route out of the totally silent regime.

#### Phase 2 — Writer unlock
- unfreeze Writer,
- keep the same tasks and losses,
- train jointly.

This directly implements your hard rule:
> “Before any Writer training starts, first open the Qwen attention -> prefix -> Writer gradient loop.”

## 10.7 Loss design for P2b

### Main task loss
- task-native generative CE on answer span / label string.

### Paired usefulness term
- no-memory paired forward
- encourage positive `delta_answer_logprob`.

### Geometry regularizers
- anti-common-mode penalty on Writer outputs,
- pairwise cosine diversity penalty,
- token energy balance,
- mild centered-rank surrogate if cheap enough.

### Distribution bridge term
- layerwise KV-stat alignment / anchor loss during early training,
- then decay it after the circuit becomes live.

### What to avoid initially
- do not jump immediately to RL/GRPO,
- do not jump immediately to DPO,
- do not reopen Reader/Fuser.

## 10.8 Gate

P2b weak success:
- at least one non-FEVER task:
  - positive `delta_answer_logprob`,
  - nonzero utility beyond trivial numeric noise,
  - no worse task score than control.

P2b medium success:
- at least one non-FEVER task:
  - measurable task-score gain,
  - plus stable positive `delta_answer_logprob`,
  - plus reduced common-mode collapse relative to W0/F1b.

P2b strong success:
- both GSM8K and NarrativeQA show actionable positive signal,
- FEVER-label-gen is at least stable,
- and the deep-prefix joint-PEFT family clearly outperforms shallow-prefix baselines.

If P2b reaches medium success, then and only then:
- open the next Writer usefulness branch,
- later reopen compression/Reader,
- and restore broader roadmap items.

---

## 11. Workstream P3 — Native-State Anchoring (If P1/P2 Are Live but Still Weak)

This stage keeps the independent Writer but improves the bridge further.

## 11.1 Purpose

Reduce the OOD gap between Writer outputs and Qwen’s local hidden/KV space without turning the Writer into Qwen itself.

## 11.2 Methods

### P3a — Hidden-state anchor initialization
Use prompt-boundary hidden states from Qwen as initialization anchors for selected layers’ deep-prefix hidden states.

### P3b — Layerwise anchor loss
During early training:
- keep projected deep-prefix states close to anchor manifold statistics,
- then anneal the loss.

### P3c — Hybrid anchor mix
Mix:
- KV-stat initialization,
- semantic token anchor bias,
- and hidden-state anchor regularization.

## 11.3 Why this is still not MemGen

The Writer remains independent.  
Only the translators are anchored.

That preserves the research story:
- generic Writer,
- backbone-local bridge modules.

## 11.4 Gate

P3 is only worth running if:
- P1 or P2 already opened the circuit,
- but usefulness remains weak.

If the circuit is still dead, P3 is not the right move.

---

## 12. Workstream P4 — Writer Usefulness Training (Only After Circuit Opening)

Only run this once P1 or P2 has produced a live substrate.

## 12.1 Purpose

Now that the bridge is physically viable, train the Writer to produce more useful memory rather than merely non-harmful prefix.

## 12.2 Recommended sequence

### Stage U1 — SFT-only usefulness training
Use the best P1/P2/P3 substrate and train with:
- CE,
- no-memory delta gain,
- and geometry regularizers.

### Stage U2 — DPO-lite (if you can construct memory preferences)
Construct pairs such as:
- same question with better vs worse memory-conditioned completions,
- or same completion with better vs worse `delta_answer_logprob`.

Use DPO because it is simpler and more stable than jumping straight into RL.

### Stage U3 — GRPO-lite (only if there is already a positive signal)
If and only if:
- the model already shows positive task-sensitive movement,
- and you want to sharpen memory usefulness with sample-based improvement,
then a lightweight GRPO-like variant may be tested.

## 12.3 Why not GRPO first

If the bridge is still dead or silent, RL will optimize noise or degenerate behaviors.

GRPO is appropriate only after:
- the prefix route is live,
- the Writer affects answers,
- and a reward/usefulness signal exists.

---

## 13. Workstream P5 — Compression and Reader Reopening (Delayed)

Reader/Fuser is still not the next move.

It may only be reopened after the following are true:

1. deep-prefix Writer-direct branch reaches **medium success**,
2. at least one non-FEVER task shows real usefulness,
3. the memory source is no longer clearly common-mode dominated.

Only then is it reasonable to ask again:
- whether compression helps,
- whether multi-query readout helps,
- and whether the two-level `M_long -> M_short` story should re-enter the active harness.

Until then, Reader remains out of scope.

---

## 14. Workstream P6 — Restoring the Macro Roadmap

Only after P4/P5 succeed do we reopen:

1. Qwen2.5 broader Tier-2 validation,
2. Qwen3-8B confirmation,
3. Stage B/C transfer evidence refresh,
4. CDMI experiments.

This preserves `PLAN.md` and `TODO_LIST.md` as macro authority.

---

## 15. Concentrated Route Matrix

This is the concise operational matrix.

## Route I — Independent Writer purity route
### I-0
- source stub
- deep prefix
- per-layer projectors
- no LoRA
- KV-stat init

### I-1
- real Writer
- deep prefix
- per-layer projectors
- no LoRA
- KV-stat init

### I-2
- real Writer
- deep prefix
- per-layer projectors
- no LoRA
- hidden-state anchor

## Route II — Minimal joint PEFT route
### II-0
- source stub
- deep prefix
- early4 receiver LoRA
- KV-stat init

### II-1 (canonical)
- real Writer
- deep prefix
- early4 receiver LoRA
- KV-stat init

### II-2 (comparator)
- real Writer
- deep prefix
- all5 sparse receiver LoRA
- KV-stat init

### II-3
- real Writer
- deep prefix
- early4 receiver LoRA
- hidden-state anchor

## Not authorized
- shallow-prefix Writer-first reruns,
- bigger Writer before circuit-open success,
- writer-only adapter reruns before deep-prefix circuit testing,
- Reader rescue,
- Qwen3 reopen,
- CDMI reopen.

---

## 16. Recommended Default Hyperparameters

These are starting points, not sacred constants.

## 16.1 Writer
- Start from the smaller W0-style Writer, not F1a.
- Keep latent token budget fixed.
- Keep support+context stimulus mode.

## 16.2 Deep prefix
- number of prefix tokens: same as current Writer output budget unless evidence demands more
- selected layers canonical: `[0, 1, 2, 3]`
- comparator layers: `[0, 7, 14, 21, 27]`

## 16.3 Receiver LoRA
- canonical rank: `2`
- canonical alpha: `4`
- comparator rank: `4`
- comparator alpha: `8`
- target modules: `k_proj`, `v_proj`

## 16.4 Initialization
- primary: `kv_stat_match`
- secondary: `semantic_anchor`
- tertiary: `hidden_state_anchor`

## 16.5 Learning rate schedule
### Phase 1 warm-start
- projector LR > receiver LoRA LR
- Writer frozen

### Phase 2 joint
- Writer LR moderate
- projector LR reduced
- receiver LoRA LR tiny but live

---

## 17. Metrics, Gates, and Stop Criteria

## 17.1 Core metrics

### Utility
- `delta_answer_logprob`
- task-native score (`exact_match`, `qa_f1`, label accuracy)

### Gradient circuit
- `grad_norm_source_stub`
- `grad_norm_writer`
- `grad_norm_prefix_projector`
- `grad_norm_receiver_lora`

### Geometry
- `memory_long_top1_top2_ratio`
- `memory_long_common_mode_energy_ratio`
- `projected_memory_effective_rank`
- pairwise cosine metrics

### Prefix routing
- per-layer prefix attention mass
- per-layer prefix-to-content attention ratio
- layerwise K/V norm stats

## 17.2 Circuit-open gate

Passes if all are true on at least one non-FEVER task:
1. source or Writer grad median over steps 1–4 is above floor,
2. receiver LoRA grad is live if receiver LoRA is enabled,
3. prefix attention mass is nontrivial on at least one selected layer,
4. `delta_answer_logprob` is not identically zero.

This is the first true gate.

## 17.3 Weak success
- one non-FEVER task:
  - positive `delta_answer_logprob`,
  - non-regressive task score,
  - geometry not worse than W0.

## 17.4 Medium success
- one non-FEVER task:
  - actual task improvement over no-memory control,
  - stable positive `delta_answer_logprob`,
  - circuit remains live,
  - common-mode indicators improve or at least stop worsening.

## 17.5 Strong success
- both non-FEVER tasks show useful signal,
- FEVER is stable,
- geometry is materially healthier,
- and the branch is clearly better than shallow-prefix baselines.

## 17.6 Hard stop rules

### Stop Rule A
If **P1a and P2a both fail** to open a live circuit on any non-FEVER task, then:
- stop external Writer-direct projector work,
- do not run more Writer capacity or adapter variants,
- and pivot to the fallback architecture in Section 20.

### Stop Rule B
If the circuit opens but utility never appears after P3 and P4 weak attempts, then:
- the problem is not pure routing anymore,
- and the branch should be reassessed for task choice or Writer formulation.

### Stop Rule C
Do not reopen:
- Reader/Fuser,
- Qwen3,
- transfer refresh,
- CDMI,
until medium success is achieved here.

---

## 18. Implementation Checklist

## 18.1 Code changes
- [ ] allow `writer_direct + receiver micro-LoRA` under deep-prefix-only rule
- [ ] add per-layer projector class
- [ ] add `source_stub` route
- [ ] add KV-stat collection utility
- [ ] add semantic/hidden-state anchor init
- [ ] add per-layer prefix attention diagnostics
- [ ] add summary scripts for P1/P2/P3

## 18.2 Config changes
- [ ] canonical early4 no-LoRA templates
- [ ] canonical early4 LoRA templates
- [ ] comparator all5 LoRA templates
- [ ] anchor init variants

## 18.3 Test changes
- [ ] unit test: writer_direct deep-prefix path builds correctly
- [ ] unit test: writer_direct + receiver LoRA no longer errors under allowed config
- [ ] unit test: per-layer projectors serialize / checkpoint correctly
- [ ] unit test: summary scripts gate correctly
- [ ] regression test: no-memory paired evaluation remains valid

## 18.4 Documentation changes
- [ ] update active exec plan
- [ ] update README narrow-status paragraph
- [ ] update `TODO_LIST.md` narrow branch pointer
- [ ] publish review bundles for each phase

---

## 19. Why PLANv5 Is Better Than Simply “Doing MemGen”

This needs to be said clearly.

A pure MemGen pivot would likely be a stronger engineering shortcut, because its Weaver is already reasoner-native.

But PLANv5 keeps alive something that is still scientifically valuable and distinct:

> **the idea that memory can be produced by an independent Writer and then translated into a backbone’s local computation space through lightweight bridge modules.**

That idea is more reusable across:
- backbones,
- transfer settings,
- and later CDMI-style formulations.

So PLANv5 does **not** reject MemGen’s insight.  
It borrows the correct lesson:

- machine-native memory matters,
- usefulness must be trained under the real reasoning route,
- and bridge physics must be respected.

But it does so while preserving the more ambitious MemTOTAL story.

---

## 20. Fallback If PLANv5 Route I/II Both Fail

If both independent-Writer concentrated routes fail, the correct pivot is:

> **Qwen-native Writer / Weaver fallback**

This means:
- the Writer ceases to be a fully external transformer,
- memory tokens are synthesized from Qwen hidden states or Qwen+LoRA hidden dynamics,
- and the system moves closer to a MemGen-like internal Writer.

This fallback is authorized only if:
- the independent Writer cannot be made to open a live deep-prefix circuit,
- or the circuit opens but utility never materializes after reasonable P3/P4 attempts.

This is the right fallback because it directly removes the external-to-internal distribution bridge.

It should **not** be the first move, because doing so would prematurely abandon a central scientific claim of MemTOTAL.

---

## 21. Final Recommended Execution Order

This is the exact order I recommend.

### Step 1 — P0
Unblock code and diagnostics.

### Step 2 — P1a
Deep-prefix prefix-only or source-stub calibration, **no receiver LoRA**.

### Step 3 — P2a
If P1a is weak/dead, add **tiny early4 receiver LoRA** and rerun the calibration.

### Step 4 — Choose substrate
- if P1a works, prefer Route I purity
- otherwise, if P2a works, Route II becomes canonical

### Step 5 — P1b or P2b
Real Writer-direct branch on the chosen substrate.

### Step 6 — P3 if needed
Native-state / KV anchor strengthening.

### Step 7 — P4 only after circuit-open success
Usefulness training:
- SFT first,
- DPO-lite second,
- GRPO-lite only if clearly warranted.

### Step 8 — Only then
Reopen broader roadmap items.

---

## 22. The Most Important Changes Relative to Earlier Plans

1. **Shallow prefix is no longer accepted as a decisive Writer-first substrate.**
2. **Receiver LoRA moves earlier, but only inside the correct deep-prefix contract.**
3. **The first hard gate is now “gradient circuit open,” not “task score moved.”**
4. **Independent Writer remains primary; MemGen-style internal Writer is fallback.**
5. **Reader stays closed until Writer usefulness is proven.**

---

## 23. Blunt Conclusion

The project should not keep iterating on the old narrow branch.

The correct reading of the evidence is not:
- “the Writer idea is dead,”
and not:
- “just make the Writer bigger,”
and not:
- “go fix Reader again.”

The correct reading is:

> **You have not yet tested the independent Writer under the bridge contract that the theory actually requires. The next branch must make the prefix a real deep prompt, give the Writer per-layer translators into Qwen’s KV space, initialize those translators near Qwen-native statistics, and—if needed—use the smallest possible early-layer LoRA to open attention routing before Writer usefulness training begins.**

That is what PLANv5 authorizes.

---

## 24. References That Inform This Plan

1. **Prefix-Tuning: Optimizing Continuous Prompts for Generation**  
   Xiang Lisa Li, Percy Liang, ACL 2021.  
   Key idea: frozen LM + continuous prefix vectors.

2. **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-Tuning Universally Across Scales and Tasks**  
   Xiao Liu et al., ACL 2022.  
   Key idea: deep prompt tuning gives much stronger and more universal behavior on NLU tasks.

3. **LoRA: Low-Rank Adaptation of Large Language Models**  
   Edward J. Hu et al., ICLR 2022.  
   Key idea: freeze base model and inject trainable low-rank adapters.

4. **Weaving Generative Latent Memory for Self-Evolving Agents (MemGen)**  
   Gaozhuo Zhang et al.  
   Key idea: memory trigger + memory weaver generate machine-native latent memory in the reasoning stream.

5. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**  
   Rafael Rafailov et al., NeurIPS 2023.  
   Key idea: lightweight preference optimization without full RLHF pipeline.

6. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**  
   Zhihong Shao et al., 2024.  
   Key idea: GRPO as a lighter RL alternative once a meaningful task signal already exists.

7. **Fine-Tuning Can Distort Pretrained Features and Underperform Out-of-Distribution**  
   Ananya Kumar et al., ICLR 2022.  
   Key idea: if the transfer route is badly aligned, more adaptation can distort rather than help.

---

# Appendix A — Active Exec Plan to Create Next

Recommended active exec-plan filename:

`docs/exec-plans/active/20260309-independent-writer-deep-prefix-circuit.md`

Recommended first line:

> Execute PLANv5 P0/P1: open the deep-prefix gradient circuit for the independent Writer before any substantial Writer usefulness training.

---

# Appendix B — Canonical First Three Runs

## Run A
- source stub
- deep prefix
- layers `[0,1,2,3]`
- no LoRA
- KV-stat init
- tasks: GSM8K, NarrativeQA, FEVER-label-gen

## Run B
- source stub
- deep prefix
- layers `[0,1,2,3]`
- receiver LoRA `r=2`, `alpha=4`
- KV-stat init
- same tasks

## Run C
- real Writer
- deep prefix
- layers `[0,1,2,3]`
- use whichever substrate (A or B) first opened the circuit
- keep W0-size Writer
- support+context stimulus
- same tasks

These three runs should happen before any new capacity, adapter, Reader, or Qwen3 work.

