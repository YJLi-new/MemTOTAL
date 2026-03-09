# PLANv4.md — Writer-First, Weaver-Inspired Recovery Plan for MemTOTAL

**Version:** 2026-03-09  
**Scope:** narrow-but-comprehensive next-hop execution plan after `PLANv3` closed at `V4`  
**Audience:** project owner / core contributors / future agents operating inside this repo  
**Primary language:** English

---

## 0. What this document is, and what it is not

This document is **not** a replacement for `PLAN.md`.

The macro order in `PLAN.md` remains correct:

1. stabilize the active path and stop dead-end loops,
2. make the real memory bridge actually alive,
3. refresh transfer evidence,
4. run CDMI and broader benchmark evidence,
5. confirm on the stronger backbone,
6. only then expand fallback logic.

What changed is the **next narrow authority after `PLANv3` terminated at V4**.

`PLANv3` did the correct thing operationally:

- it shifted the diagnosis from “Reader geometry only” to **Writer common-mode domination + receiver gradient starvation**;
- it tested architecture-first Writer tweaks (`V1`), then **tiny diagnostic receiver micro-LoRA** (`V2`), then **non-FEVER validation** (`V4`);
- it stopped when the branch had clearly failed to produce any actionable positive signal.

This document therefore does **three** things:

1. **preserves** the macro order of `PLAN.md`,
2. **accepts** the terminal evidence from `PLANv3`, and
3. **redefines the next hop** as a **Writer-first, Weaver-inspired recovery line**.

The governing idea is simple:

> The next problem is no longer “how to rescue Reader/Fuser” and no longer “whether tiny receiver LoRA is mechanically live.”  
> The next problem is: **how to train a Writer that actually produces reasoner-useful, non-common-mode latent memory before we ask any Reader to compress it.**

---

## 1. Inputs this plan assumes have been reviewed

This plan assumes the following have been read or inspected together:

1. the current GitHub repository state,
2. the uploaded repo/docs bundles,
3. existing repo documentation, especially:
   - `docs/MAIN_IDEA.md`
   - `docs/TODO_LIST.md`
   - `docs/EXPERIMENTS_INFO.md`
   - `docs/ARCHITECTURE.md`
   - `docs/briefs/20260307-m4-shared-injection-brief.md`
   - `docs/exec-plans/active/20260309-tl-reader-geometry-rescue.md`
   - `docs/exec-plans/20260309-post-rg3-writer-value-microlora-multitask.md`
4. active code paths, especially:
   - `src/memtotal/models/memory.py`
   - `src/memtotal/models/backbone.py`
   - `src/memtotal/pipeline.py`
   - `src/memtotal/training/m3.py`
   - `src/memtotal/training/m4_shared_injection.py`
5. recent result artifacts and run summaries from:
   - `tl-poc-fever-qwen25`
   - `tl-bridge-rescue-fever-qwen25`
   - `tl-slot-basis-rescue-fever-qwen25`
   - `tl-reader-symmetry-break-fever-qwen25`
   - `tl-reader-local-bootstrap-fever-qwen25`
   - `tl-writer-value-fever-qwen25`
   - `tl-micro-lora-fever-qwen25`
   - `tl-bridge-multitask-qwen25`

This document is therefore **not** speculative in a vacuum. It is built around the actual repo, the actual failure chain, and the actual public state of the related literature.

---

## 2. One non-negotiable correction before any new work starts

### 2.1 The current blocker is not “Reader first” anymore

That interpretation was reasonable during `RG-0 → RG-3`, but it is no longer the best reading after `PLANv3`.

The failure chain is now:

- `RG-2` showed that Reader specialization can be **forced structurally** (e.g. partition) without producing semantically different readouts.
- `RG-3` showed that local Reader bootstrap did **not** create meaningful geometry gain.
- `V1` showed that simple Writer architecture tweaks (`shared_add_scaled`, `slot_query_only`) did **not** move the dominant geometry at all.
- `V2` showed that receiver micro-LoRA is mechanically trainable but does **not** change capability or collapse behavior.
- `V4` showed that even when non-FEVER tasks create much stronger optimization signals, the same Writer geometry persists and answer quality still does not improve.

So the blocker is now better stated as:

> **Writer value homogenization is the main failure source; Reader collapse is a downstream symptom.**

### 2.2 The next hop is therefore not “more Reader rescue” and not “broader receiver fallback”

From this point onward:

- **do not** continue the RG line,
- **do not** broaden the V2 receiver-LoRA line,
- **do not** reopen Reader/Fuser restoration before Writer usefulness is established,
- **do not** interpret “non-zero gradients” as evidence that the bridge is learning the right thing.

The next hop must be:

> **Train the Writer first, directly, under a shorter and clearer usefulness path, with Reader temporarily bypassed or frozen.**

That matches both the current evidence and the user’s current hard requirement.

---

## 3. Current state after V4: the exact project position

This section freezes the current state so future work cannot drift or reinterpret history.

### 3.1 What `PLANv3` already established

#### V1 — Writer-only architecture tweaks were flat

The real `V1` FEVER Writer matrix reported that the tested Writer arms remained effectively indistinguishable from control:

- `memory_long_top1_top2_ratio ≈ 70.7`
- `memory_long_common_mode_energy_ratio ≈ 0.9986`
- `reader_readout_effective_rank ≈ 1.23`
- `reader_readout_pairwise_cosine_mean ≈ 0.9993`
- no collapse delay
- essentially identical FEVER task score (`best_adapt_task_score ≈ 0.2951`)

Interpretation:

- the current common-mode failure is **not** fixed by merely scaling or removing one shared conditioning path inside the inherited scaffold;
- the warm-start family likely traps the system in a basin where slotwise perturbations are too weak to matter.

#### V2 — Tiny receiver micro-LoRA was live but unhelpful

`V2` proved the receiver LoRA path is mechanically real:

- the harness logs real trainable LoRA parameters,
- the path receives non-zero gradients,
- the implementation is not a dead code path.

But the real FEVER result still stayed flat:

- `best_adapt_task_score ≈ 0.2951`
- `best_adapt_macro_f1 ≈ 0.1519`
- `dominant_label_collapse_onset_step = 0`
- `memory_long_top1_top2_ratio ≈ 70.7`
- `reader_readout_pairwise_cosine_mean ≈ 0.9993`

Interpretation:

- receiver-side PEFT can **exist** without solving the bridge,
- the current bottleneck is upstream enough that tiny receiver adaptation has nothing useful to amplify.

#### V4 — The failure is not FEVER-only

The non-FEVER validation did **not** reopen the bridge.

- NarrativeQA control reached `task_score ≈ 0.2857` (`qa_f1`) on the small eval slice.
- The carried-forward micro-LoRA family regressed to `0.0` on NarrativeQA.
- GSM8K stayed at `0.0` exact match for both control and bridge.
- Across both tasks, the bridge kept the same core geometry:
  - `memory_long_top1_top2_ratio ≈ 70.7`
  - `reader_readout_effective_rank ≈ 1.22`
- This happened **despite** larger early Reader/Fuser/receiver-LoRA gradients on the non-FEVER smokes.

Interpretation:

- the failure is **not** just “FEVER is the wrong task”;
- the failure is **not** simply “there was no optimization signal”;
- the failure is **not** solved by “trying a second or third task” inside the same geometry.

### 3.2 What is already settled enough to stop debating

The project has already established enough evidence to stop revisiting these claims:

1. **Frozen Qwen is not prefix-blind.** Earlier shared-injection evidence already showed non-zero main-chain consumption.
2. **The bridge can be wired end-to-end.** The issue is not missing implementation of training/eval/review logic.
3. **Reader specialization can be forced syntactically.** The problem is that forced specialization did not create semantically diverse readouts.
4. **Tiny receiver LoRA can be wired mechanically.** The issue is not a dead adapter surface.
5. **The current Writer family is not “empty” in the probe sense, but it is still unusable in the bridge sense.**

### 3.3 The updated one-line diagnosis

The most accurate short diagnosis now is:

> The system is receiving optimization signal, but the current Writer family converts that signal into a nearly shared, low-information latent mode. That common mode is easy to preserve, hard for the frozen reasoner to exploit, and robust to many superficial architecture or adapter tweaks.

---

## 4. Independent deep diagnosis: why the system is “learning” without improving

This section gives the bottom-up diagnosis, not merely a restatement of run summaries.

### 4.1 The real failure is a **responsibility mismatch**

The current bridge asks too many modules to solve too many hidden problems at once:

```text
support selection / support encoder
-> writer
-> reader
-> fuser
-> injector/projector
-> frozen reasoner
-> task metric
```

If the end metric is poor, the gradient must travel through **five** learned interfaces before it reaches the Writer.

That means the Writer is not being trained under a direct question like:

> “Did this memory help the reasoner produce the correct answer?”

Instead it is trained under a much murkier question:

> “After several transformations, did the final model behave slightly better?”

This is an extremely weak and underdetermined supervision channel.

### 4.2 The current Writer found a “safe but useless” local strategy

The observed geometry suggests the Writer has discovered a stable compromise:

- produce slotwise outputs that are **not numerically identical** after centering,
- but keep almost all energy in a strong **shared/common component**,
- thereby avoiding catastrophic disruption of the frozen reasoner,
- while still leaving enough tiny variation for probes or regularizers to register “some structure.”

In other words, the Writer is not truly solving the task. It is solving a lower-level survival problem:

> “How can I change the latent channel a little, without causing the frozen model to become even more unstable?”

That is why the system can display:

- non-zero gradients,
- parameter updates,
- finite adapter stats,
- some local geometry movement,

and still produce **no useful answer improvement**.

### 4.3 Why the gradients can be large and still be non-actionable

The current “illusion of learning” can be understood as follows:

1. The frozen reasoner sees a latent perturbation it does not know how to use.
2. Task loss becomes large, so gradients become non-zero and can even be large in magnitude.
3. But the gradient direction that reaches the Writer is still too indirect and poorly assigned.
4. The easiest response is not to build richer latent factors; it is to stay near a common-mode perturbation that minimally destabilizes the base model.

This is not “zero gradient.” It is **high-magnitude, low-quality gradient**.

### 4.4 Why Reader rescue failed for principled reasons

Reader rescue failed not because the implementation was careless, but because the repair happened at the wrong level of the stack.

If the upstream latent slots are dominated by one common component, then:

- masked partition can change **where** attention looks,
- competitive attention can change **which slot wins**,
- local bootstrap can change **how queries separate**,

but none of these creates new semantic content in the slot values themselves.

So Reader rescue was diagnosing the right symptom, but not attacking the causal source.

### 4.5 Why micro-LoRA also failed for principled reasons

The receiver micro-LoRA result is also interpretable, not mysterious.

Receiver LoRA can help only if there is some stable latent factor worth decoding.
If the incoming latent memory is nearly common-mode and nearly collinear, the receiver has two bad options:

- learn to ignore it, or
- overfit to brittle nuisance patterns.

The V2/V4 results look exactly like that regime.

So the correct interpretation of V2 is **not** “receiver PEFT is useless.”
It is:

> Receiver PEFT cannot rescue a Writer that has not yet learned to emit differentiated, useful memory.

### 4.6 Why the user’s new direction is correct

The user’s hard suggestion—**train the Writer first, do not worry about the Reader yet, and borrow heavily from MemGen’s weaver training philosophy**—is directionally correct.

I agree with that direction **with one important refinement**:

> We should borrow the **weaver-first training logic** from MemGen, but not blindly clone the whole MemGen agent stack.

That means:

- yes to **train Writer in isolation first**,
- yes to **freeze the reasoner during Writer training**,
- yes to **fixed insertion schedule before any trigger/selector logic**,
- yes to **multi-task training rather than FEVER-only**,
- yes to **SFT first, RL later if needed**,
- no to **reopening Reader now**,
- no to **broad receiver PEFT now**,
- no to **assuming MemGen’s exact multi-turn GRPO stack can be copied directly into current MemTOTAL tasks without adaptation**.

---

## 5. Architecture cross-examination: what survives scrutiny and what does not

### 5.1 What does **not** survive scrutiny anymore

#### A. “Reader is the primary next hop”

No longer credible.

The RG line already exhausted the most plausible Reader-local rescue paths, and later V1/V2/V4 evidence made Writer common-mode domination more central than Reader symmetry.

#### B. “Tiny receiver LoRA should be expanded immediately”

No longer justified.

It was correct to test it once. It is **not** correct to let it dominate the next branch after V2/V4 stayed flat.

#### C. “Another FEVER-only objective tweak may still solve it”

No longer efficient.

Workstream A already answered that line negatively, and V4 showed the deeper geometry does not magically improve just because the task changes.

### 5.2 What *does* survive scrutiny

#### A. The global macro plan still survives

The repository still needs:

- Qwen2.5 + Qwen3 confirmation,
- transfer evidence,
- CDMI,
- cross-domain adaptation,
- final paper-quality baselines.

Those deliverables remain locked.

#### B. A Writer-first branch is now the most rational next hop

The evidence now points to the Writer as the bottleneck with the highest causal leverage.

#### C. MemGen is the correct external reference point—but mainly for **training structure**

MemGen is relevant because it demonstrates a workable separation:

- train a memory weaver **with the reasoner fixed**,
- insert latent memory under a fixed schedule **before** learning the trigger,
- use either SFT or GRPO for the weaver,
- let memory generation become directly responsible for downstream behavior.

That is exactly the kind of decoupling the current MemTOTAL branch lacks.

### 5.3 The key refinement: “Weaver logic” is more important than “MemGen surface imitation”

The next MemTOTAL branch should imitate the **logic** of MemGen’s weaver, not necessarily its full agent machinery.

What matters most is:

1. **Writer trained in isolation**,
2. **frozen reasoner**,
3. **direct memory usefulness supervision**,
4. **fixed insertion schedule before any trigger/selector complexity**,
5. **multi-task, generative settings that provide richer supervision than FEVER-only classification**.

That is the correct abstraction to borrow.

---

## 6. Core strategic decision of PLANv4

### 6.1 New narrow authority

`PLANv4` becomes the new narrow authority after `PLANv3`.

### 6.2 New live branch name

The next branch is:

# **Workstream W — Writer-First Weaver-Style Rescue**

### 6.3 Main hypothesis

> If the Writer is trained more like a MemGen-style weaver—i.e. directly responsible for producing a small set of latent memory tokens that improve answer generation under a frozen reasoner, without Reader/Fuser in the loop—then the common-mode collapse can be broken earlier and more cleanly than by continuing Reader rescue or receiver-side PEFT.

### 6.4 Immediate non-goals

Until Workstream W reaches at least **medium success**, do **not**:

- reopen Reader/Fuser optimization,
- reopen broad receiver fallback,
- reopen Qwen3 confirmation,
- reopen Stage B/C transfer refresh,
- reopen CDMI experiments.

The reason is not bureaucracy. It is causal hygiene.

---

## 7. The new target architecture: **Writer-as-Weaver direct bridge**

This is the central design change.

### 7.1 Replace the active training question

The current active question is too indirect:

> Can a support-conditioned latent survive Writer → Reader → Fuser → Injector and still help the frozen reasoner?

The new question must be shorter and cleaner:

> Can a Writer, conditioned on support and current prompt state, generate a small latent memory block that directly improves the frozen reasoner’s answer generation?

### 7.2 Core structural move

Temporarily bypass or freeze Reader/Fuser.

The active path becomes:

```text
prompt / task input
+ structured support set
-> frozen reasoner prompt-state extraction
-> Writer (as Weaver)
-> direct latent memory block (K tokens)
-> projector / injector
-> frozen reasoner answer generation
```

### 7.3 What the Writer now sees

The Writer should no longer rely only on pooled support state.

Its stimulus should include **both**:

1. **support-side evidence** (structured support set or retrieved exemplars), and
2. **reasoner-side prompt state** right before answer generation.

That makes the Writer much closer to MemGen’s weaver, which conditions memory generation on the current cognitive state of the reasoner.

### 7.4 Why this matters

Support-only conditioning answers:

> “What evidence exists?”

But not:

> “What does the frozen reasoner need *right now* to answer this prompt correctly?”

The pre-answer hidden state supplies that missing signal.

### 7.5 Minimum concrete architecture

Use a new Writer variant, provisionally called **`WriterWeaverHead`**, with this flow:

1. Build prompt up to a fixed insertion boundary (usually just before answer tokens).
2. Run the frozen reasoner on the prompt prefix and extract a hidden-state slice `H_ctx`.
3. Encode the support set into `S_sup` using the current structured support encoder.
4. Start from learned slot seeds `Q0`.
5. Cross-attend `Q0` to `H_ctx` first (context need).
6. Cross-attend the updated queries to `S_sup` second (support retrieval).
7. Refine through a small transformer block.
8. Emit `K` latent memory tokens `M_writer`.
9. Project/inject them directly into the existing shared-injection path.

This preserves the current harness and injector, but changes the Writer from a weak pooled-state mapper into a true stimulus-conditioned latent generator.

### 7.6 Reader policy during Workstream W

During Workstream W:

- Reader = bypassed or frozen,
- Fuser = bypassed or frozen,
- active `M_short` = the Writer’s own emitted memory block,
- compression is **not** the problem we are solving yet.

This is deliberate. We are teaching the system to write useful memory before we ask it to compress memory.

### 7.7 Injection policy during Workstream W

Keep the current injector family as stable as possible.

Specifically:

- preserve the current shared-injection harness,
- preserve the frozen reasoner,
- preserve the current projector contract,
- do **not** broaden receiver LoRA in this phase.

If any PEFT is introduced here, it should be **writer-side or writer-adjacent**, not receiver-first.

---

## 8. MemGen borrowing rules: what to borrow, what not to borrow

### 8.1 What to borrow from MemGen

#### Borrow 1 — Train the Writer/Weaver in isolation first

MemGen explicitly separates Weaver training from Trigger training.
The current public recipe trains the weaver first and only then the trigger.
This is exactly the right lesson for MemTOTAL now.

#### Borrow 2 — Keep the reasoner frozen during Writer training

This is the cleanest way to make Writer usefulness measurable and to avoid washing the problem into the backbone.

#### Borrow 3 — Use a fixed insertion schedule before learning any selector

MemGen’s paper describes training the weaver before the trigger is available by using a fixed/random insertion policy.  
MemTOTAL should do the analogous thing: use a **fixed insertion boundary** (and later, if needed, random delimiter insertion on rationale-rich tasks) before reintroducing any Reader/selector logic.

#### Borrow 4 — Use SFT first, RL later if needed

MemGen is explicitly compatible with SFT and GRPO for the weaver. That fits our situation well:

- first teach Writer to produce useful latent memory under a dense supervised objective,
- only later, if needed, refine with a group-relative reward.

#### Borrow 5 — Multi-task validation is required

MemGen’s claim is not FEVER-specific, and the user already required “not only FEVER.”  
So Workstream W must start with at least **two** non-identical tasks, not another FEVER-only loop.

### 8.2 What *not* to borrow blindly from MemGen

#### Do not borrow the full Trigger/Weaver dual-stack immediately

Trigger is not the next problem.
The current problem is that the Writer is not producing useful latent memory even under fixed insertion.

#### Do not jump straight to GRPO as the first live training stage

The public MemGen repo prominently exposes Weaver-SFT workflows and notes that some GRPO artifacts are still being staged separately. More importantly, our current issue is geometry collapse **before** any positive direct bridge signal exists.

So the correct order is:

1. Writer-SFT / behavior cloning style training,
2. optional Writer-RL only after Writer usefulness is visible.

#### Do not copy MemGen’s LoRA breadth by default

MemGen’s official configs use LoRA on `q_proj` / `v_proj` with `r=16`, `alpha=32` for the Weaver/Trigger stack.  
That is a valid baseline reference, but it should **not** automatically become the default MemTOTAL setting.

For Workstream W, the first objective is not “match MemGen parameter count.”
It is “make Writer usefulness real without opening too many new degrees of freedom.”

So any LoRA added in this plan must remain **small, targeted, and writer-side first**.

---

## 9. New workstream definition

# Workstream W — Writer-First Weaver-Style Rescue

## 9.1 Primary goal

Make the Writer produce **non-common-mode, reasoner-useful latent memory** under a direct bridge, before reintroducing Reader/Fuser compression.

## 9.2 Secondary goals

1. prove positive signal on **at least one non-FEVER task**;
2. establish a stable writer-training recipe that can later feed the true two-level path;
3. keep the macro paper plan alive without reopening irrelevant fallback paths too early.

## 9.3 Main hypothesis family

Workstream W tests the following hypothesis family, in order:

### H-W1 — Objective mismatch hypothesis

The current Writer fails mainly because it is trained through an overly long and indirect path.
If we train it directly against answer-generation behavior, geometry will improve.

### H-W2 — Stimulus mismatch hypothesis

The current Writer sees support but not the right notion of current reasoning need.
If we add pre-answer reasoner-state stimulus, usefulness will improve.

### H-W3 — Common-mode attractor hypothesis

The current Writer defaults to common-mode because there is no explicit penalty against it and no explicit reward for memory usefulness over a no-memory baseline.
If we add direct usefulness and anti-common-mode objectives, collapse will relax.

### H-W4 — RL refinement hypothesis

If Writer-SFT gets the bridge partially alive but not yet strong, a small MemGen-style group-relative refinement can help—but only once Writer usefulness is already visible.

---

## 10. Exact architecture proposal for Workstream W

## 10.1 Variant names

We will use the following naming convention.

### Controls

- **C0** — frozen no-memory control
- **C1** — current direct-bridge baseline without Reader (existing writer, support-only, no new loss)

### Writer-as-Weaver variants

- **W1** — support-only direct Writer, fresh init, Reader bypassed
- **W2** — support + pre-answer prompt-state Writer (`WriterWeaverHead`)
- **W3** — W2 + direct usefulness / anti-common-mode training losses
- **W4** — optional W3 + lightweight writer-side adapter (only if W3 geometry improves but capability still lags)
- **W5** — optional W3/W4 + GRPO-lite refinement (only if SFT stage shows partial success)

## 10.2 Frozen vs trainable modules by phase

### Default frozen

- frozen reasoner backbone
- receiver-side adapters disabled by default
- Reader
- Fuser

### Default trainable in W1–W3

- new Writer / WeaverHead
- projector / injector
- optionally support encoder **only after** initial Writer-only pilots

### Important rule

For the first real Writer-first pilots:

> **freeze the support encoder and train only Writer + projector.**

Reason:

- current evidence suggests gradients can be absorbed upstream without forcing the Writer itself to solve memory synthesis;
- MemGen’s weaver logic updates the weaver while the frozen reasoner supplies the environment.

Only if Writer geometry improves but task gain remains weak should support encoder unfreezing be reopened.

## 10.3 Fresh initialization rule

Do **not** inherit the failed common-mode Writer checkpoints as the starting point for Workstream W.

Start from:

- fresh Writer parameters,
- orthogonal slot/seed initialization,
- fresh projector if practical,
- frozen backbone,
- frozen Reader/Fuser.

Rationale:

The V1 architecture matrix already showed that under the inherited warm-start family, even architecture changes remained trapped in the same geometry basin.

## 10.4 Direct bridge contract

The active Writer-first bridge is:

```text
(task prompt up to insertion boundary)
-> frozen reasoner hidden slice H_ctx
(task support / exemplars)
-> frozen or fixed support encoder S_sup
(H_ctx, S_sup)
-> WriterWeaverHead
-> K latent tokens
-> existing projector/injector
-> frozen reasoner answer generation
```

## 10.5 Fixed insertion policy

### Initial policy (required)

Insert memory **once**, at a deterministic boundary just before answer generation.

This is the cleanest and shortest supervision path.

### Later optional policy (only after direct bridge is alive)

On rationale-rich tasks (e.g. GSM8K solutions or other teacher-forced intermediate traces), allow random delimiter/punctuation insertion during Writer training, borrowing the MemGen “fixed/random insertion before trigger learning” intuition.

But this is **not** day-one work.

---

## 11. Training objectives: the Writer must be optimized for usefulness, not mere existence

The current project needs a new objective family.

## 11.1 Main supervised objective: answer-token CE

Let:

- `x` = task prompt,
- `S` = support set,
- `b` = insertion boundary,
- `H_ctx` = frozen reasoner hidden states just before `b`,
- `M = W_phi(H_ctx, S)` = Writer memory tokens,
- `P = Projector(M)` = injected latent tokens/prefix,
- `y*` = gold answer string.

Primary loss:

```text
L_answer = - log p_theta(y* | x, P)
```

where `theta` (the reasoner) is frozen and gradients flow only through `phi` (Writer) and trainable bridge-side parameters.

### Important implementation detail

Loss should be computed on the **answer span only**, not on the entire prompt.

Reason:

- the project is not trying to teach the Writer to imitate the prompt,
- it is trying to teach the Writer to make the frozen reasoner answer better.

## 11.2 Mandatory direct usefulness objective: no-memory delta gain

The current system needs an objective that explicitly asks:

> “Did memory improve the probability of the correct answer compared with no memory?”

So compute a no-memory baseline on the same batch:

```text
Delta(x) = log p_theta(y* | x, P) - log p_theta(y* | x, no_memory)
```

Then use either:

### Margin version

```text
L_gain = max(0, m - Delta(x))
```

or

### Reward-style version

```text
L_gain = - Delta(x)
```

with clipping or normalization as needed.

This is essential. Without it, the Writer can still “optimize” answer CE by finding degenerate perturbations that do not create truly useful memory.

## 11.3 Mandatory anti-common-mode objective

The Writer must be directly penalized for collapsing most slot energy into a shared component.

Use a differentiable common-mode penalty. One practical form:

```text
mu = mean_k M_k
E_common = ||mu||^2
E_total = (1/K) * sum_k ||M_k||^2
L_common = E_common / (E_total + eps)
```

This need not perfectly equal the reporting metric; it only needs to push against the actual failure mode.

## 11.4 Mandatory diversity / covariance objective

Common-mode penalty alone is not enough. We also need a term that rewards usable slot diversity in the centered memory.

Let:

```text
M_centered = M - mean_k M_k
C = covariance(M_centered)
```

Use one of:

- negative log-determinant regularizer,
- off-diagonal cosine penalty,
- top1/top2 singular-value ratio penalty.

Recommended default:

```text
L_cov = - log det(C + eps I)
```

or, if numerically simpler,

```text
L_offdiag = mean_{i != j} cos(M_centered_i, M_centered_j)^2
```

## 11.5 Slot energy balance objective

The slot-basis rescue already showed that orthogonality alone is not enough. We also need to prevent all information from hiding in one or two large slots.

Use:

```text
L_balance = Var_k(||M_k||)
```

This is a mild regularizer, not a main objective.

## 11.6 Optional but recommended: slot-dropout consistency

To force the Writer to distribute information rather than betting everything on one hidden slot, train with random slot dropout.

Procedure:

1. compute answer loss with full `M`,
2. randomly drop a subset of Writer slots,
3. compute answer loss again,
4. penalize excessive degradation.

This is a practical way to discourage “single hidden carrier + cosmetic residuals.”

## 11.7 Recommended total SFT objective

A good initial combined objective is:

```text
L_total = L_answer
        + lambda_gain * L_gain
        + lambda_common * L_common
        + lambda_cov * L_cov
        + lambda_balance * L_balance
        + lambda_drop * L_slotdrop
```

### Initial weighting guidance

Start conservatively:

- `lambda_gain`: moderate
- `lambda_common`: moderate
- `lambda_cov`: small-to-moderate
- `lambda_balance`: small
- `lambda_drop`: small

Reason:

- `L_answer` must remain the anchor objective,
- auxiliary losses are there to break the specific geometry trap, not to replace task learning.

## 11.8 What should *not* be used initially

Do **not** begin this workstream with:

- full-token teacher KL,
- broad receiver-side PEFT,
- Reader/Fuser auxiliary losses,
- trigger learning,
- multi-step RL,
- inherited common-mode warm-start.

These all add complexity before the Writer itself becomes useful.

---

## 12. Multi-task plan: do not train on FEVER alone

This is now a hard requirement.

## 12.1 Why FEVER-only is insufficient for Workstream W

FEVER has been useful diagnostically, but it is a poor sole training ground for a Writer-as-Weaver branch because:

- it is label-space narrow,
- it is not naturally a generative memory benchmark,
- it does not reward rich latent memory composition in the way longer-form generative tasks do.

So FEVER should become:

- a regression/sanity benchmark,
- not the only live optimization surface.

## 12.2 Required task mix for initial Workstream W

### Tier-1 tasks (required from the start)

Use **at least two** of the following, with a strong preference for the first two because they are already near the active pilot surface:

1. **GSM8K** — math / exact match
2. **NarrativeQA** — narrative QA / QA-F1
3. **FEVER (label generation form)** — regression/sanity auxiliary, not the main teacher

### Tier-2 tasks (open only after Tier-1 positive signal)

4. **TriviaQA** — open-domain QA / EM-F1
5. **Story Cloze or RocStories** — narrative continuation / multiple choice or generative label form
6. **KodCode** — code benchmark, to reconnect with the later CDMI path

## 12.3 Curriculum rule

The initial writer-first live run should be:

- **multi-task from day one**, but
- task-balanced and conservative.

Recommended first curriculum:

- 40% GSM8K
- 40% NarrativeQA
- 20% FEVER-label-gen

Only after positive signal should Tier-2 tasks be added.

## 12.4 Why this curriculum makes sense

- GSM8K gives clean exact-answer supervision.
- NarrativeQA gives long-form answer grounding and richer lexical supervision.
- FEVER remains a useful regression check because it is already deeply instrumented in the repo.

This combination provides broader signal than FEVER-only without opening too many tasks at once.

---

## 13. Workstream W execution phases

# Phase W0 — Harness refactor and diagnostic freeze

## Purpose

Create the Writer-direct scaffold cleanly and freeze the meaning of the new branch.

## Concrete work

1. Add a **Writer-direct mode** to the active shared-injection pilot.
2. Add prompt-boundary hidden-state extraction for the frozen reasoner.
3. Add explicit support+context Writer stimulus plumbing.
4. Add answer-span-only CE.
5. Add no-memory delta computation on the same batch.
6. Add direct Writer geometry logging:
   - common-mode energy ratio,
   - top1/top2 ratio,
   - centered effective rank,
   - slot norm mean/std,
   - direct Writer-token pairwise cosine.
7. Add trainable/frozen manifest logging so every run records exactly which modules moved.

## Acceptance

- one injected smoke run completes on each Tier-1 task,
- metrics propagate into `train_events`, snapshot summaries, and final summaries,
- Reader/Fuser can be completely bypassed without breaking analysis or review publishing.

## Hard rule

Do not start any real writer-training sweep before W0 is complete.

---

# Phase W1 — Single-task Writer-direct SFT smokes

## Purpose

Validate the new Writer-direct scaffold and choose between support-only and support+context Writer variants.

## Arms

- `C0`: no-memory frozen control
- `C1`: current direct Writer baseline (support only, fresh init)
- `W1`: support-only Writer-direct
- `W2`: support+context Writer-direct (`WriterWeaverHead`)
- `W3`: W2 + gain/common-mode losses

## Tasks

Run small fixed smokes on:

- GSM8K
- NarrativeQA
- FEVER-label-gen

## Budget

- single seed
- tiny fixed subset per task
- 4–8 steps for pure plumbing smoke
- then 16–32 steps for first directional smoke

## What to look for

The goal of W1 is **not** final accuracy.
It is to answer:

1. does support+context conditioning move geometry earlier than support-only?
2. does the direct usefulness loss move `Delta(answer logprob)` above zero?
3. does common-mode ratio start dropping before capability emerges?

## Advance criterion

Proceed to W2 only if **at least one** Writer-direct arm beats the simple direct baseline on **both**:

- a geometry metric, and
- `Delta(answer logprob)`

for at least one non-FEVER task.

If none do, first inspect implementation and metric correctness. If the scaffold is correct and still flat, proceed to the W1-fallback described in Section 18.

---

# Phase W2 — Qwen2.5 Writer-direct real pilot

## Purpose

Run the first real Writer-first branch under meaningful budget, but still before RL.

## Fixed choices

- backbone: `Qwen2.5-1.5B-Instruct`
- reasoner: frozen
- Reader/Fuser: bypassed or frozen
- receiver-side LoRA: disabled
- Writer init: fresh, orthogonal slot seeds
- support encoder: frozen initially

## Primary arms

- `P0`: no-memory frozen control
- `P1`: support-only Writer-direct
- `P2`: support+context Writer-direct
- `P3`: P2 + usefulness/anti-common-mode losses

## Optional arm

- `P4`: P3 + support encoder unfrozen

Only open `P4` if `P3` shows geometry improvement but task signal remains weak.

## Budget recommendation

- 1 seed for pilot
- 64–128 steps
- fixed Tier-1 training mixture
- stable small validation slices per task

## Success gates

### Weak success

- `memory_long_common_mode_energy_ratio` drops materially from the V1/V4 level,
- `top1/top2` drops materially from ~70.7,
- `Delta(answer logprob)` becomes positive on at least one non-FEVER task,
- no catastrophic regression vs no-memory control on all tasks.

### Medium success

- geometry improves materially **and**
- at least one non-FEVER task beats frozen control on actual answer metric.

### Strong success

- two or more Tier-1 tasks beat frozen control,
- geometry metrics clear the thresholds defined in Section 16,
- the improvement is not isolated to one lucky checkpoint.

## Decision

- If **medium or strong success**: proceed to W3 multi-task confirmation.
- If only **weak success**: proceed to W4 Writer-RL refinement.
- If **no success**: go to the Writer-capacity fallback path in Section 18.

---

# Phase W3 — Multi-task Writer-direct confirmation

## Purpose

Confirm that the Writer-direct recipe is not a one-task artifact.

## Work

1. keep the best W2 Writer family fixed,
2. run 3 seeds,
3. retain Tier-1 tasks,
4. optionally add one Tier-2 task only if harness is stable.

## Required outputs

- per-task curves,
- aggregate summary,
- geometry summary,
- no-memory delta summary,
- seed-wise confidence intervals.

## Promotion criterion

Only after W3 shows **at least medium success across seeds** may the repo reopen:

- Reader/Fuser reintroduction,
- Qwen3 confirmation,
- later transfer/CDMI work.

---

# Phase W4 — Optional Writer-RL refinement (GRPO-lite, not day one)

## Purpose

Refine a partially successful Writer if SFT created geometry but not enough answer gain.

## When to open

Open W4 only if W2/W3 show:

- Writer geometry improved,
- some positive `Delta(answer logprob)` signal exists,
- but actual answer metrics remain below desired threshold.

If W2/W3 are completely flat, W4 is **not** the next move.

## Why RL might help

Once the Writer is already somewhat useful, group-relative optimization can:

- sharpen answer-level preferences,
- reward useful memory generation more directly than plain CE,
- help the Writer discover higher-value latent memory modes.

## But plain outcome-only GRPO is not enough

Vanilla GRPO uses sparse final rewards. That is dangerous for memory modules, because it does not tell the system **which part of the memory helped**.

So if W4 is opened, the reward must include a **memory-usefulness term**, not only final task correctness.

### Recommended reward

For each sampled completion group:

```text
R_total = R_task
        + alpha * Delta(answer logprob vs no memory)
        - beta  * common_mode_penalty
        - gamma * slot_collapse_penalty
```

This is much closer to the kind of direct memory-level credit assignment that recent memory-RL work argues for.

## Important constraint

Keep W4 single-turn and fixed-insertion.
Do **not** import trigger learning here.

---

# Phase W5 — Reintroduce Reader/Fuser compression only after Writer usefulness exists

## Purpose

Once the Writer produces useful `M_long`, ask whether a Reader/Fuser can compress it without destroying usefulness.

## Rules

1. freeze the successful Writer family first,
2. train Reader/Fuser as a compression bridge,
3. compare direct `K=8` Writer memory vs compressed `K=4` / `K=2` memory,
4. preserve the same task mix and evaluation slices.

## Why this order is correct

A Reader cannot compress useful information that does not yet exist.  
Compression is a **later** question than usefulness.

## Success criterion

Compressed memory should preserve most of the task gain while reducing token budget and keeping geometry sane.

Only then is the repo back on the true two-level path.

---

# Phase W6 — Reopen Qwen3 and later macro milestones

## Gate to reopen Qwen3

Do **not** reopen Qwen3 until Qwen2.5 Writer-first branch shows at least **medium success** on non-FEVER tasks.

## After Qwen3 reopens

Resume the original macro plan:

1. confirm on Qwen3-8B,
2. refresh transfer/adaptation evidence,
3. reopen CDMI,
4. finish the locked paper deliverables in `docs/TODO_LIST.md`.

---

## 14. Detailed task protocol for Workstream W

## 14.1 FEVER protocol change

FEVER should be converted into a **label-generation** task for this branch.

Example targets:

- `SUPPORTS`
- `REFUTES`
- `NOT ENOUGH INFO`

Reason:

This keeps the output surface consistent with the Writer-direct answer-generation recipe.

## 14.2 GSM8K protocol

- answer-only loss on the final boxed/normalized answer string,
- optional rationale-aware insertion only in later phases,
- primary metric: exact match,
- secondary metric: answer log-prob delta vs no-memory control.

## 14.3 NarrativeQA protocol

- answer-only loss on the gold short answer,
- primary metric: QA-F1,
- secondary metric: answer log-prob delta.

## 14.4 Multi-task batching rule

Use balanced or temperature-smoothed sampling.
Do not let FEVER dominate token count or update count.

## 14.5 Support-set rule

For the first live Writer-first branch, **do not redesign support retrieval at the same time**.

Use the current repo-supported support-source protocol per task as-is.

Reason:

- retrieval redesign is a separate variable,
- Workstream W is about Writer usefulness, not retrieval policy.

---

## 15. Implementation plan by file/module

This section is intentionally operational.

## 15.1 `src/memtotal/models/memory.py`

Add or extend:

- `WriterWeaverHead`
- `stimulus_mode`:
  - `support_only`
  - `context_only`
  - `support_plus_context`
- fresh orthogonal seed initialization
- centered-slot helpers
- common-mode helper
- covariance/off-diagonal helper
- optional slot-dropout forward path

## 15.2 `src/memtotal/models/backbone.py`

Add:

- prompt-prefix hidden-state extraction at a named insertion boundary,
- answer-span indexing utilities,
- optional writer-side adapter support only if later needed.

Do **not** broaden receiver adapter logic here unless later explicitly authorized.

## 15.3 `src/memtotal/training/m4_shared_injection.py`

Add:

- `bridge_mode = writer_direct`
- answer-span-only CE
- no-memory paired forward pass for `Delta(answer logprob)`
- Writer-only trainable manifest
- new losses:
  - `writer_gain_margin_loss`
  - `writer_common_mode_penalty`
  - `writer_covariance_diversity_loss`
  - `writer_slot_energy_balance_loss`
  - `writer_slot_dropout_consistency_loss`
- fixed insertion boundary logic
- multi-task task-native answer normalization/scoring

## 15.4 `src/memtotal/analysis/m4_shared_injection.py`

Add summaries for:

- Writer-direct geometry
- no-memory delta
- per-task answer metrics
- phase decision JSON
- success classification

## 15.5 New scripts

Add at minimum:

- `scripts/run_writer_weaver_smoke_qwen25.sh`
- `scripts/run_writer_weaver_multitask_qwen25.sh`
- `scripts/update_writer_weaver_summary.py`
- later if needed: `scripts/run_writer_weaver_grpo_qwen25.sh`

## 15.6 Configs

Add:

- `configs/exp/writer_weaver_qwen25_smoke_*.yaml`
- `configs/exp/writer_weaver_qwen25_multitask_*.yaml`
- later if needed: `configs/exp/writer_weaver_qwen25_grpo_*.yaml`

## 15.7 Tests

Add tests for:

- Writer-direct mode wiring
- answer-span loss slicing
- no-memory paired forward correctness
- fresh-init vs warm-start contract
- Reader bypass invariants
- multi-task score aggregation
- review bundle generation

---

## 16. Metrics and gates (explicit)

These are the authoritative gates for Workstream W.

## 16.1 Geometry metrics

Track at least:

- `memory_long_common_mode_energy_ratio`
- `memory_long_top1_top2_ratio`
- `memory_long_centered_effective_rank`
- `memory_long_slot_norm_mean`
- `memory_long_slot_norm_std`
- `writer_token_pairwise_cosine_mean`
- `projected_memory_effective_rank`

## 16.2 Usefulness metrics

Track:

- `answer_logprob_with_memory`
- `answer_logprob_without_memory`
- `delta_answer_logprob`
- `delta_answer_logprob_by_task`

## 16.3 Task metrics

Track:

- FEVER label accuracy / macro-F1
- GSM8K exact match
- NarrativeQA QA-F1
- later task-native metrics as added

## 16.4 Weak / medium / strong thresholds

### Weak geometry success

Any Writer arm achieves all of:

- `common_mode_energy_ratio <= 0.97`
- `top1_top2_ratio <= 30`
- `centered_effective_rank >= 3.0`
- `delta_answer_logprob > 0` on at least one non-FEVER task

### Medium geometry+usefulness success

Any Writer arm achieves all of:

- `common_mode_energy_ratio <= 0.93`
- `top1_top2_ratio <= 15`
- `centered_effective_rank >= 4.0`
- positive actual task gain vs frozen control on at least one non-FEVER task

### Strong success

Any Writer family achieves all of:

- `common_mode_energy_ratio <= 0.85`
- `top1_top2_ratio <= 8`
- `centered_effective_rank >= 5.0`
- positive task gain on at least two tasks,
- stable direction across seeds.

## 16.5 Failure definition

Workstream W is classified as **failure at the current stage** if:

- geometry stays essentially flat,
- `delta_answer_logprob` stays non-positive or noisy,
- and no non-FEVER task improves over control.

That is the exact pattern that closed `PLANv3` at V4, and it must not be tolerated again without changing the actual Writer training recipe.

---

## 17. Acceptance criteria and stop rules

## 17.1 Accept W0 only if

- Writer-direct mode runs end-to-end,
- metrics appear in all required outputs,
- tests pass.

## 17.2 Accept W1 only if

- one Writer-direct arm shows directional geometry movement,
- paired no-memory delta behaves sensibly,
- support+context can be distinguished from support-only in outputs.

## 17.3 Accept W2 only if

- real pilot run completes with structured review bundle,
- one arm reaches at least weak success.

## 17.4 Accept W3 only if

- multi-task improvement is not isolated to FEVER,
- at least medium success is reproduced across seeds.

## 17.5 Stop rules

### Stop rule A

If W1/W2 show **zero** geometry movement under fresh Writer init and direct usefulness loss, do not reopen receiver LoRA.
Move to the Writer-capacity fallback.

### Stop rule B

If W2/W3 show geometry movement but still no meaningful task gain, open W4 Writer-RL refinement.

### Stop rule C

If W3 shows medium or strong success, do not waste time on more writer-only variations. Reopen Reader compression.

---

## 18. Explicit fallback logic if the first Writer-first branch fails

This section is important because the project should not get trapped in another endless loop.

## 18.1 Fallback F1 — Writer capacity upgrade before receiver reopening

If fresh-init Writer-direct training is still flat, the most likely next issue is not the Reader and not the receiver. It is that the **current small Writer architecture is itself too weak or too constrained**.

Then open a **Writer-capacity upgrade**, still without Reader:

### F1a — bigger WriterWeaverHead

- deeper Writer block,
- more cross-attn depth,
- larger hidden size,
- same frozen reasoner,
- same direct bridge,
- same usefulness losses.

### F1b — lightweight writer-side adapter

If needed, add a **small writer-side adapter** only.

Allowed options:

- tiny LoRA on writer-side projection layers,
- IA3-style writer-side gating,
- small rank only.

This remains within the user’s current “train Writer first” request.

## 18.2 Fallback F2 — GRPO-lite only after SFT has moved geometry

If geometry improves but answer metrics lag, then RL refinement is justified.
If geometry never improves, RL is premature.

## 18.3 Fallback F3 — only after Writer usefulness exists, reconsider receiver-side PEFT

Receiver-side PEFT may still matter later, but after V2/V4 it is no longer the right immediate next hop.

Only reopen it if:

- Writer-direct memory becomes useful,
- but frozen reasoner readability remains the limiting factor.

---

## 19. What this plan intentionally postpones

To protect causal clarity, `PLANv4` explicitly postpones:

- trigger learning,
- Reader/Fuser rescue,
- broad receiver fallback,
- Qwen3 confirmation,
- Stage B/C transfer refresh,
- CDMI experiments,
- large baseline sweeps unrelated to Writer usefulness.

These are not abandoned. They are gated behind the Writer-first branch.

---

## 20. Why this plan is the most stable next step

This plan is stable because it removes the three biggest sources of confusion from the previous branch:

### Stability gain 1 — It shortens the responsibility chain

The Writer is now directly responsible for answer generation quality, not indirectly responsible through Reader/Fuser.

### Stability gain 2 — It changes the supervision, not just the shape

The plan adds a **usefulness delta** objective and anti-common-mode geometry losses, rather than merely trying more architectural nudges inside the old supervision path.

### Stability gain 3 — It matches the strongest external precedent at the right abstraction level

MemGen’s most relevant lesson is not “copy our trigger.”  
It is:

> train the memory generator first, with the reasoner fixed, under direct downstream usefulness.

That is exactly what Workstream W does.

---

## 21. Immediate next commands (the first concrete execution slice)

The first execution slice after accepting this document should be:

### Step 1

Create a new active exec-plan under:

```text
docs/exec-plans/active/20260309-writer-weaver-rescue.md
```

### Step 2

Implement W0 only:

- Writer-direct mode
- prompt-boundary hidden-state extraction
- answer-span-only CE
- no-memory delta computation
- Writer-direct geometry logging

### Step 3

Run one smoke per Tier-1 task:

- GSM8K
- NarrativeQA
- FEVER-label-gen

### Step 4

Publish one structured summary:

- whether support+context signal is wired correctly,
- whether no-memory delta behaves sensibly,
- whether geometry changes at all under fresh Writer init.

Only after those four steps are complete should the project open W1 real smokes.

---

## 22. Final blunt summary

The previous branch did not fail because the repo lacked effort. It failed because the project kept trying to rescue downstream geometry before establishing upstream memory usefulness.

The next rational move is therefore:

> **Stop trying to save the Reader. Stop broadening receiver fallback. Train the Writer first, directly, like a MemGen-style weaver under a frozen reasoner, on multiple tasks, with direct usefulness supervision and explicit anti-common-mode objectives.**

If that branch produces positive non-FEVER signal, the rest of the MemTOTAL macro plan can reopen.  
If it does not, the project will at least have tested the correct bottleneck directly instead of continuing to orbit it.

---

## 23. References informing this plan

These references are included because `PLANv4` intentionally draws from them at the level of mechanism and training logic.

1. **MemGen: Weaving Generative Latent Memory for Self-Evolving Agents** — Guibin Zhang, Muxin Fu, Shuicheng Yan, ICLR 2026 / arXiv 2025.  
   Relevant here for: frozen-reasoner + weaver-first training, SFT/GRPO-compatible weaver, trigger trained later, fixed/random insertion before trigger learning.

2. **MemGen official public repository / README** — confirms the current public two-stage Weaver/Trigger training split and exposes the public weaver-SFT surface; also notes some GRPO training artifacts are staged separately.

3. **Prefix-Tuning: Optimizing Continuous Prompts for Generation** — Li & Liang, ACL 2021.  
   Relevant here for: frozen backbone + learned continuous prefix is viable in principle, so the present failure is not proof that latent prefixing is impossible.

4. **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks** — Liu et al., ACL 2022.  
   Relevant here for: deep/continuous prompt methods can work broadly, but require the right optimization setup.

5. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR 2022.  
   Relevant here for: PEFT is useful, but low-rank adapters only help if the signal they are meant to decode is meaningful.

6. **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning** — Liu et al., NeurIPS 2022.  
   Relevant here for: IA3/PEFT as lightweight alternatives, but again only after the correct causal bottleneck is identified.

7. **Perceiver IO** and **Flamingo / Perceiver Resampler** — Jaegle et al. (2022), Alayrac et al. (2022).  
   Relevant here for: fixed-query latent compression remains a valid later-stage mechanism, but only after the source memory is semantically rich enough to compress.

8. **Self-Memory Policy Optimization (MemPO)** — arXiv 2026.  
   Relevant here for: outcome-only GRPO is too sparse for memory modules; memory-specific reward/advantage terms are valuable if RL refinement is opened later.

