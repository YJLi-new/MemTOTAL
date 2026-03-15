
# PLANv8.md — External-Writer Continuation via Qwen3-8B, Reader-LoRA Consumption, and OPD-Guided Training for MemTOTAL

**Version:** 2026-03-13  
**Primary language:** English  
**Scope:** successor plan after `PLANv7 (LR updated version)` closed at `Path Q`  
**Audience:** project owner, core contributors, future coding agents, experiment operators  
**Primary controlling intent:** preserve the external-Writer thesis, but stop asking the old KV-prefix route to prove more than it can  
**Primary backbone for this plan:** `Qwen3-8B`  
**Continuity / regression backbone:** `Qwen2.5-1.5B-Instruct`  
**Primary benchmarks:** `GSM8K`, `TriviaQA`  
**Calibration benchmark only:** `FEVER`  
**Optional late-stage transfer / stress benchmarks:** `NarrativeQA`, `Story Cloze`, `MemGen comparator tasks`  
**Primary hardware assumption:** single `NVIDIA RTX PRO 6000 96GB`  
**Harness stance:** preserve the governed `PLANv6/PLANv7` infrastructure and extend it, not rewrite it

---

## 0. What this document is, and what it is not

This document is **not** a philosophical reset and **not** a rejection of the project’s core idea.

It is a **successor execution authority** written after the following facts became unavoidable:

1. `PLANv7 (LR updated version)` did **not** vindicate the old external-Writer + deep-prefix route on the primary tasks.
2. The LR correction from `7.5e-6` to `7.5e-5` changed the final governance closeout from `Path R` to `Path Q`, which means the route is **not completely dead**, but it is still **unresolved** on the tasks that matter.
3. The `V7-0` oracle remained flat on `GSM8K` and `TriviaQA`, which means the original **KV-cache deep-prefix mechanism** is still the main architectural suspect.
4. The codebase already contains enough `Qwen3-8B`, teacher-alignment, and comparator plumbing that a stronger plan can be executed **inside the current repo**, rather than as an unrelated rewrite.

This document therefore does five things simultaneously:

1. **preserves** the external-Writer thesis rather than abandoning it,
2. **reclassifies** the main trainable surface from “Writer first at all costs” to “Reader-consumption first, Writer reintroduction second,”
3. **promotes** `Qwen3-8B` from optional portability target to the main experimental backbone,
4. **adapts** `OpenClaw-RL`’s OPD idea into a benchmark-supervised, token-level memory-consumption training signal,
5. **keeps** the repo’s governed review / publish / summary machinery as a first-class asset.

The one-sentence thesis of `PLANv8` is:

> **Do not give up on the external Writer; instead, preserve it as the high-bandwidth memory producer, move the main trainable burden to a Reader-LoRA / Reader-adapter that teaches the base LLM how to consume memory, upgrade the backbone to Qwen3-8B so the primary tasks are actually learnable, and use OPD-style token-level teacher advantages to break the gradient-starvation regime that destroyed the V6/V7 line.**

---

## 1. Inputs this plan assumes have been reviewed

This plan assumes the following have been reviewed together:

### 1.1 Repo and review snapshot inputs

1. Current uploaded review snapshot (`MemTOTAL-review (3).zip`)
2. Uploaded prior plans:
   - `PLANv4.md`
   - `PLANv6dot1.md`
   - `PLANv7.md`
3. Current repo review branch state
4. Current public GitHub repo state
5. Current governed summaries and exec logs for:
   - `V7-0`
   - `V7-1`
   - `V7-2`
   - `V7-3`
   - `V7-4`
   - `V7-5`
   - `V7-6`

### 1.2 Code paths that matter for this plan

This plan was written with the following code paths in mind:

- `src/memtotal/models/backbone.py`
- `src/memtotal/models/memory.py`
- `src/memtotal/training/m4_shared_injection.py`
- `src/memtotal/analysis/m4_shared_injection.py`
- `src/memtotal/tasks/writer_jointpeft_data.py`
- `scripts/run_planv7_*`
- `scripts/run_m4_selected_shared_injection_suite.py`

### 1.3 Literature and external references that matter here

This plan explicitly uses the following reference families:

- prefix / prompt tuning and their limitations
- LoRA / QLoRA
- context compression / gist / ICAE
- cross-attention latent bottlenecks
- MemGen-style latent in-model memory
- OPD-style token-level teacher supervision

The goal is not to imitate any single paper. The goal is to use the literature to design a route that is actually compatible with the repo’s observed failure modes.

---

## 2. Source-of-truth precedence and one important conflict that must be frozen

Before any more work begins, the project needs a clear rule for conflicting summaries.

### 2.1 Source-of-truth precedence for `PLANv8`

When two repo artifacts disagree, use this precedence:

1. **phase summary JSON**
2. **phase summary MD**
3. **phase runbook / exec plan**
4. **branch README rollup**
5. **public `main` branch README**

This is mandatory because `PLANv8` was written after observing that rollup surfaces can lag or simplify the phase-level truth.

### 2.2 Critical discrepancy discovered during review

The review branch README states:

- `any_strict_writer_metric_improvement_across_three_seeds = false`

but the `V7-6` phase summary reports:

- `any_strict_writer_metric_improvement_across_three_seeds = true`

while still reporting:

- `any_real_primary_gain_across_three_seeds = false`

`PLANv8` treats the **phase summary** as authoritative. That means the correct interpretation is:

> There **was** some phase-level strict writer metric movement in the LR-updated replay, but it still failed to convert into actual primary-task gains.

This matters because the difference between “absolutely no movement” and “small but non-converting movement” is exactly the difference between `Path R` and `Path Q`.

### 2.3 Public-main vs review-branch mismatch

The public `main` branch README still presents the repo as `PLANv6`-authoritative, while the review branch and uploaded snapshot clearly place the project at `PLANv7 (LR updated version)` closeout. `PLANv8` therefore treats the uploaded review snapshot + review branch as the scientific authority for next-step planning.

---

## 3. Non-negotiable owner constraints

The following are **hard constraints**, not preferences.

### 3.1 Strategic hard constraints

1. **Do not abandon the external Writer yet.**
2. **Try `Qwen3-8B` seriously, not as a token portability appendix.**
3. **Reopen the Reader as a foundational component of memory consumption.**
4. **Consider making the Reader the main LoRA-adapted surface rather than the Writer.**
5. **Use `OpenClaw-RL`’s OPD idea as a candidate solution to gradient starvation.**
6. **Try more than one learning rate / optimizer setting rather than silently inheriting one point estimate.**
7. **Keep `GSM8K` and `TriviaQA` as the primary decision pair.**
8. **Write the plan at operational detail comparable to `PLANv4` and more detailed than a generic research memo.**

### 3.2 Operational translation of those constraints

`PLANv8` translates them into these concrete rules:

- The external Writer stays in the architecture and in the paper story.
- `Qwen3-8B` becomes the **main** live backbone for the next branch.
- `Qwen2.5-1.5B-Instruct` becomes a continuity / smoke / cheap regression backbone only.
- The main trainable surface moves to a **Reader-LoRA / Reader-adapter** that teaches the base model how to use memory.
- The Writer is first reintroduced as:
  - an **oracle / frozen support encoder**,
  - then a **high-dimensional external writer**,
  - and only later a **trainable external writer**.
- OPD is introduced **first on the Reader**, not the Writer.
- Learning-rate sweeps are small, local, and deliberately targeted.

---

## 4. Executive bottom line

The shortest honest summary of the current project state is:

> `PLANv7 (LR updated version)` established that the old route is not a pure projector illusion, but it did **not** establish a paper-ready external-Writer system for `GSM8K` and `TriviaQA`.

The next plan should therefore **not** do any of the following:

- not another FEVER-first sweep,
- not another pure KV-prefix bandwidth sweep on `Qwen2.5-1.5B`,
- not another broad Reader-geometry philosophical detour,
- not another vague “stronger Writer” proposal that still feeds the same dead channel,
- not another architecture-blind search over losses.

Instead, `PLANv8` will pursue this disciplined sequence:

1. **revalidate the benchmarked backbone on `Qwen3-8B`**
2. **retest memory injection with a Reader-centric interface rather than old deep-prefix alone**
3. **train the Reader first, with the Writer frozen / oracle**
4. **apply OPD to the Reader to fix gradient starvation**
5. **only then reintroduce a trainable external Writer**
6. **only then reopen compression / bridge decisions**
7. **only after positive primary movement run multi-seed confirmation and CDMI / comparator work**

---

## 5. Independent deep diagnosis of the current project state

This section is the real reason `PLANv8` is different.

### 5.1 What the LR-updated replay actually changed

The projector LR restart from `7.5e-6` to `7.5e-5` changed one important thing:

- it restored enough route activity that the governed closeout moved from `Path R` to `Path Q`.

That is real and important.

What it did **not** change:

- `GSM8K` still showed no real gain across the confirmed branch set,
- `TriviaQA` still showed no real gain across the confirmed branch set,
- `FEVER` remained calibration-only,
- strict Writer non-collapse was still not robustly established as a useful state.

So the correct reading is:

> the restart softened the negative verdict, but it did not rescue the primary scientific claim.

### 5.2 The oracle result remains the binding constraint

The most constraining result in the entire project is still `V7-0`:

- early vs mid baselines were flat on primary tasks,
- context-echo oracle was flat,
- support-echo oracle was flat,
- prefix attention mass remained low,
- `all_oracles_flat_on_primary_tasks = true`.

This means:

> even when the model is given better-than-trainable Writer content through the old route, it still does not improve on `GSM8K` and `TriviaQA`.

That is not a “Writer capacity” result.
That is an **injection mechanism** result.

### 5.3 Why the old route likely fails mechanistically

The current code and results point to three coupled bottlenecks.

#### 5.3.1 Prefix/KV injection is a weak control surface for these tasks

The deep-prefix path in `backbone.py` constructs a per-layer KV cache and, for non-target layers, fills the prefix with a zero hidden template passed through the model’s own layernorm and `k_proj` / `v_proj`. Even when this is not catastrophic numerically, it means the experiment is asking the model to learn through a highly indirect route.

Theoretical work on prompting / prefix tuning is consistent with this failure mode:

- context-based fine-tuning can be strictly less expressive than full fine-tuning,
- it cannot arbitrarily rewrite the model’s relative attention pattern over content,
- it may work best when eliciting capabilities already present, not when forcing a new use of memory.

That fits the empirical result almost too well:
the route is enough to move FEVER-like calibration, but not enough to unlock robust generative use on the main tasks.

#### 5.3.2 The old Writer is trained through too long a gradient chain

Under the V6/V7 route the Writer influences the loss only after passing through:

- support encoder / support representation,
- Writer,
- projector,
- optional bridge,
- injection interface,
- frozen backbone,
- next-token loss.

That means the Writer is answering a weak question:

> “after many transformations, did the final output get marginally better?”

This is the textbook setup for diffuse supervision and collapse to an easy common mode.

#### 5.3.3 The backbone scale confound is real

At `Qwen2.5-1.5B`, the project repeatedly ran into a hard interpretability problem:

- if the base model is near-zero on a reasoning task,
- a memory system is being asked to create capability, not merely augment it.

That is an unfair test for almost any memory interface.
A stronger backbone is not a luxury here; it is needed for the experiment to be interpretable.

### 5.4 What `Path Q` means scientifically

`Path Q` means:

- the external-Writer line is **not purely fake**,
- the bridge line is **not merely projector-only**,
- some metric movement exists,
- but the current route is **not yet useful where it counts**.

So `Path Q` is **not** a justification for more old-route optimization.
It is a justification for a **nearby architectural continuation** that preserves the Writer thesis while changing the consumption interface.

### 5.5 Why the Reader should now be the main adapted surface

The user’s idea—“what if the Reader is the LoRA, not the Writer?”—is the right next question.

The reason is straightforward:

- the Reader sits closest to the base model’s actual token prediction;
- it receives the shortest and clearest gradient path from task loss;
- it is the place where “use the memory” can be learned directly;
- it lets the Writer remain external and high-bandwidth without demanding that the Writer discover, by itself, the exact internal semantics the frozen model wants.

This is not abandoning the external Writer.
It is making the external Writer **consumable**.

### 5.6 Why OPD is relevant here

The user’s OPD proposal is not an unrelated RL idea.
It maps directly onto the observed failure mode.

The key failure in V6/V7 was not “no gradients.”
It was “the gradients reaching the important upstream surface were too weak, too delayed, and too nonspecific.”

OPD-style token-level teacher advantages do exactly what this project is missing:

- they turn “your final answer was wrong” into
- “at these tokens, conditioned on better hindsight, the model would have preferred a different continuation.”

That is an information-rich signal.
And if applied to the Reader, it lands very near the actual memory-consumption interface.

---

## 6. `PLANv8` strategic thesis

The strategic thesis of `PLANv8` is:

> **Preserve the external Writer as the high-capacity memory producer, but stop treating its direct projection into a frozen LM as the central learning problem. Instead, use a stronger backbone (`Qwen3-8B`), teach a Reader-LoRA / Reader-adapter to consume memory through a more expressive interface than the old KV-prefix path, apply OPD to the Reader first, and only then reopen trainable external-Writer scaling.**

This thesis has four consequences.

### 6.1 The Writer is preserved, but not forced to do everything first

The Writer remains part of the core architecture and part of the long-term paper story.

But the immediate phases do **not** ask the Writer to solve the whole problem alone.

The plan sequence becomes:

1. **oracle / frozen Writer**
2. **Reader consumption learning**
3. **Reader OPD**
4. **trainable external Writer**
5. **compression / bridge optimization**
6. **Writer-side OPD / auxiliary refinement**

### 6.2 `Qwen3-8B` becomes the main scientific backbone

This is the right move because:

- the repo already supports `Qwen3-8B`,
- the code already maps this backbone name to `Qwen/Qwen3-8B`,
- the model card states that `Qwen3-8B` is an 8.2B-parameter causal LM with 36 layers, GQA (`32` Q heads / `8` KV heads), and native 32k context,
- the same model family is explicitly advertised as stronger than prior Qwen2.5 instruct variants on reasoning-heavy tasks,
- the model card also documents task-specific prompting and `enable_thinking` behavior that the current harness has not yet used as a first-class experimental variable.

### 6.3 The Reader must be allowed back in, but in a new role

The Reader in `PLANv8` is **not** the old broad “geometry rescue” project from earlier plans.

It is a narrower, more grounded component:

- either a **Reader-LoRA** operating on the backbone’s own attention projections,
- or a **small cross-attention Reader-adapter** at selected layers,
- with the explicit job of making memory consumption learnable.

This is a fundamentally different reopening than the old “Reader/Fuser rescue” line.

### 6.4 OPD is introduced as a local teacher signal, not a grand RL rewrite

The plan does **not** turn the project into a generic online RL system.

Instead it adapts the OPD idea in a benchmark-supervised way:

- teacher context = prompt + memory + hindsight hint
- student context = prompt + memory
- advantage = token-level difference in teacher vs student preference
- loss applies first to Reader parameters

This is conceptually aligned with `OpenClaw-RL`, but engineered for the current benchmark stack and the current training harness.

---

## 7. Architectural design for `PLANv8`

This section defines the actual new system.

## 7.1 High-level architecture

`PLANv8` standardizes the following abstraction:

```text
support material
-> support encoder / support hidden states
-> external Writer produces M_long
-> optional bridge compresses M_long to M_short
-> Reader interface injects / exposes memory to the backbone
-> Reader-adapted backbone consumes memory
-> task loss + OPD loss train the consumer first, then the producer
```

### 7.1.1 What remains external

The following remain external to the base LLM:

- support bank
- external Writer
- optional bridge/compressor
- memory bookkeeping and diagnostics

### 7.1.2 What becomes backbone-adjacent

The following become the main adapted consumer surfaces:

- Reader-LoRA on `q_proj/k_proj/v_proj/o_proj`
- optional Reader cross-attention block
- memory-to-embedding projection / exposure layer
- OPD teacher/student scoring head logic

---

## 7.2 Writer families in `PLANv8`

The external Writer is preserved, but with a better escalation ladder.

### 7.2.1 `EW0` — oracle hidden-state Writer

This is the first Writer used in live `PLANv8` experiments.

**Definition**

- run the frozen backbone on the support material,
- extract hidden states from a fixed layer,
- convert them into memory slots by chunk pooling.

**Default settings**

- backbone: `Qwen3-8B`
- extraction layer: `18`
- pooling window: `16` tokens
- stride: `16`
- slot cap:
  - `16` in the first consumer scout
  - `32` in the reader sweep
  - `64` in the high-dimensional Writer phases
- slot dimension: `4096` (same as backbone hidden size)
- pre-slot normalization: `LayerNorm`
- post-pool projection: off by default, optional `2-layer MLP 4096->4096->4096`

**Why it exists**

This is the cleanest way to preserve the external-Writer idea while removing “the Writer failed to encode anything” as the first confound.

### 7.2.2 `EW1` — lightweight external Writer

This is the first trainable external Writer family.

**Definition**

- input = support hidden-state sequence from `EW0`’s frozen encoder
- latent slot queries cross-attend to those support states
- output = `M_long`

**Default architecture**

- slot count: `64`
- slot dim: `4096`
- latent slot parameters: learned
- cross-attention blocks: `2`
- feed-forward width: `16384`
- attention heads: `16`
- residual + pre-norm: yes
- output norm: yes

**Purpose**

- test whether trainable external memory production beats frozen-oracle chunk pooling once the Reader already knows how to consume memory.

### 7.2.3 `EW2` — stronger integrated external Writer

This is the “stronger integrated Writer” interpretation that remains external-Writer-adjacent.

**Definition**

Same as `EW1`, but with one more level of structure:

- `3` cross-attention / transformer blocks
- learned slot basis
- explicit slot energy balancing / orthogonality regularization hooks
- optional support-context dual input (`support_only` vs `support_and_context`)

**Why**

This satisfies the current repo’s `Path Q` handoff recommendation without falling back into the old tiny low-rank Writer regime.

### 7.2.4 `EW3` — true high-dimensional `M_long`

This is the “true high-dimensional branch” interpretation.

**Definition**

- keep `M_long` large and full-dimensional
- do **not** immediately compress to tiny short slots
- allow `64` or `96` slots at full hidden size
- only compress later if memory budget or results demand it

**Purpose**

To respect the user’s hard constraint that the Writer can be very wide because later queries / Reader compression can reduce it.

---

## 7.3 Reader interface families in `PLANv8`

This is the most important architectural change.

### 7.3.1 `RI0` — legacy deep-prefix KV route (continuity control only)

This is kept only as a continuity control.

**Definition**

- same style as V7 deep-prefix KV injection
- mid-layer emphasis retained where needed
- tiny receiver LoRA continuity preserved

**Role**

- historical control only
- never the main promotion target in `PLANv8`

### 7.3.2 `RI1` — sequence-memory Reader-LoRA (mainline)

This is the default mainline Reader family.

**Definition**

- memory slots are exposed to the backbone as actual **sequence positions** processed through the full forward pass,
- not as a per-layer synthetic KV cache only,
- Reader-LoRA teaches the model how to use them.

**Default exposure mode**

- `prepend_block`
- memory segment placed after system / task preamble and before the task question
- all memory tokens participate in the normal causal stack

**Optional exposure mode**

- `interleave_every_k`
- insert memory tokens every `k` prompt tokens
- used only as an ablation if `prepend_block` is flat

**Reader-LoRA default**

- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- default layer bands:
  - `mid8 = [14,15,16,17,18,19,20,21]`
  - `mid12 = [12,13,14,15,16,17,18,19,20,21,22,23]`
  - `late8 = [20,21,22,23,24,25,26,27]`
- default rank candidates: `32`, `64`
- default alpha: `2 * rank`
- default dropout: `0.05`

**Memory-to-embedding exposure layer**

- `LayerNorm`
- `MLP 4096 -> 4096 -> 4096`
- GELU
- residual enabled
- optional learned segment embedding

**Why this is the mainline**

It keeps the Writer external, but it stops asking the model to read memory through the old high-risk KV-only route.

### 7.3.3 `RI2` — mid-layer cross-attention Reader-adapter

This is the secondary consumer family.

**Definition**

- add a small gated cross-attention block at selected backbone layers,
- backbone hidden states act as queries,
- memory slots act as keys/values,
- output is added back through a learned residual gate.

**Default settings**

- insertion layers:
  - primary: `[16,17,18,19]`
  - secondary sweep: `[18,19,20,21]`
- cross-attn heads: `16`
- adapter width: `4096`
- ff width: `16384`
- residual gate init: `0.0`
- per-layer gate: yes

**Why it exists**

This is the strongest alternative if sequence-memory Reader-LoRA is still too prompt-like.

---

## 7.4 Bridge / compression families in `PLANv8`

The bridge is allowed back in, but only after the consumer is proven.

### 7.4.1 `BR0` — no compression

- `M_long` goes directly to the Reader interface

### 7.4.2 `BR1` — light learned query compression

- learned queries = `16`
- output short slots = `16`

### 7.4.3 `BR2` — aggressive query compression

- learned queries = `32`
- output short slots = `16`

### 7.4.4 `BR3` — aggressive compression with hard short budget

- learned queries = `32`
- output short slots = `8`

**Rule**

`PLANv8` does **not** reopen broad Reader/Fuser rescue first.
Compression is reopened only after:

- the consumer interface is real,
- at least one primary task moves,
- or memory budget forces the issue.

---

## 7.5 OPD adaptation in `PLANv8`

### 7.5.1 Main idea

For each supervised training example:

- student sees: `prompt + memory`
- teacher sees: `prompt + memory + hindsight hint`
- compute teacher vs student preference over target tokens
- use the difference as token-level directional supervision

### 7.5.2 Default supervised OPD variant

`PLANv8` uses a benchmark-friendly adaptation of OPD first.

**Default student target tokens**

- use gold target tokens under teacher forcing
- do not start with free-running sampled trajectories

**Reason**

- simpler
- more stable
- easier to integrate with existing benchmark code
- easier to reuse current alignment-aux infrastructure

### 7.5.3 Default advantage definition

For token `t` on the gold continuation:

```text
A_t = max(0, log p_teacher(y_t | prompt, memory, hint, y_<t)
             - log p_student(y_t | prompt, memory, y_<t))
```

Optional centered variant:

```text
A_t = sigmoid((teacher_logprob - student_logprob - center) / scale)
```

### 7.5.4 Default OPD loss

For the Reader-first phases:

```text
L_total = L_task_CE + λ_opd * mean_t[ stopgrad(A_t) * CE_t(student, gold_t) ]
```

Alternative KL version:

```text
L_total = L_task_CE + λ_opd * KL( teacher || student ) over the masked target span
```

### 7.5.5 Hint construction policy

Hints are **training-only**.

**GSM8K**

- `H1`: final numeric answer only
- `H2`: final answer + short rationale sketch
- `H3`: gold rationale prefix capped to `64` tokens

**TriviaQA**

- `H1`: gold answer string only
- `H2`: answer string + one evidence sentence containing the answer
- `H3`: answer string + two shortest evidence sentences

**FEVER**

- `H1`: label only
- `H2`: label + one evidence sentence

### 7.5.6 Where OPD is applied

Priority order:

1. **Reader only**
2. **Reader + exposure projector**
3. **Writer only**
4. **Writer + Reader joint**

The plan starts with (1), because that is where the gradient path is shortest.

---

## 8. Global defaults for `PLANv8`

This section locks the defaults.

### 8.1 Global backbone defaults

#### 8.1.1 Main backbone

```yaml
backbone:
  name: Qwen3-8B
  model_id: Qwen/Qwen3-8B
  load_mode: hf_causal_lm
  dtype: bf16
  attn_implementation: flash_attention_2
  gradient_checkpointing: true
```

#### 8.1.2 Continuity backbone

```yaml
backbone:
  name: Qwen2.5-1.5B-Instruct
  model_id: Qwen/Qwen2.5-1.5B-Instruct
  load_mode: hf_causal_lm
  dtype: bf16
  attn_implementation: flash_attention_2
  gradient_checkpointing: true
```

### 8.2 Software / environment defaults

- `transformers >= 4.51.0` for `Qwen3`
- `peft` current repo-compatible latest stable
- `accelerate` current repo-compatible latest stable
- `torch` version pinned to the already working CUDA stack
- `flash-attn` only if already stable in the environment; otherwise fall back cleanly

### 8.3 Training budget defaults

```yaml
runtime:
  device: cuda
  train_batch_size_per_device: 2
  eval_batch_size_per_device: 4
  gradient_accumulation_steps: 8
  train_steps_scout: 200
  train_steps_main: 300
  train_steps_confirm: 400
  warmup_steps: 20
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1e-8
  lr_schedule: cosine
  min_lr_ratio: 0.1
```

### 8.4 Clipping defaults

`PLANv8` keeps the useful stabilization lessons from the V6/V7 governed line.

```yaml
runtime:
  clip_mode: groupwise
  group_clip_norm:
    reader: 1.0
    writer: 1.0
    bridge: 1.0
    projector: 1.0
  global_clip_fallback: 1.0
```

### 8.5 Sequence and memory length defaults

```yaml
runtime:
  max_support_tokens: 768
  max_prompt_tokens: 1024
  max_target_tokens_gsm8k: 256
  max_target_tokens_triviaqa: 96
  max_target_tokens_fever: 32
  max_memory_slots_small: 16
  max_memory_slots_medium: 32
  max_memory_slots_large: 64
  max_memory_slots_xlarge: 96
```

### 8.6 Seed policy

- scout seed: `61109`
- confirmation seeds: `61109`, `61110`, `61111`

### 8.7 Benchmark data defaults

The main decision pair remains `GSM8K` and `TriviaQA`.

#### 8.7.1 Phase-scout default sizes

| Task | source pool | support count | train | eval | role |
|---|---:|---:|---:|---:|---|
| GSM8K | 160 | 8 | 96 | 64 | primary |
| TriviaQA | 160 | 8 | 96 | 64 | primary |
| FEVER | 128 | 8 | 80 | 48 | calibration |

#### 8.7.2 Confirmation default sizes

| Task | train | eval |
|---|---:|---:|
| GSM8K | 128 | 128 |
| TriviaQA | 128 | 128 |
| FEVER | 96 | 96 |

**Rule:** `FEVER` is never allowed to override the primary decision.

### 8.8 Task prompt defaults

`Qwen3` prompt calibration is treated as a first-class experimental variable in `V8-0`.

Default initial guess:

- `GSM8K`: `enable_thinking=True`, explicit step-by-step instruction, boxed final answer
- `TriviaQA`: `enable_thinking=False` first, with concise answer format
- `FEVER`: `enable_thinking=False`, short label format

The calibration phase may revise these defaults.

### 8.9 Artifact namespace defaults

All new artifacts must use a non-overwriting namespace.

Examples:

- `planv8-v8-0-qwen3-baselines-oracles`
- `planv8-v8-1-reader-interface-scout`
- `planv8-v8-2-reader-sweep`
- `planv8-v8-3-reader-opd`
- `planv8-v8-4-external-writer`
- `planv8-v8-5-bridge`
- `planv8-v8-6-writer-aux-opd`
- `planv8-v8-7-comparators`
- `planv8-v8-8-confirmation`
- `planv8-v8-9-cdmi`

---

## 9. Metrics and acceptance gates

`PLANv8` needs better success definitions than “route alive”.

### 9.1 Primary task metrics

**GSM8K**

- exact match
- normalized numeric extraction exact match
- answer logprob delta vs no-memory control

**TriviaQA**

- exact match
- token-level F1
- answer logprob delta vs no-memory control

**FEVER**

- accuracy
- macro F1
- calibration sanity only

### 9.2 Route / consumer activation metrics

New Reader-side metrics must be added and logged:

- `memory_token_attention_mass_mean`
- `memory_token_attention_mass_by_layer`
- `reader_to_memory_grad_norm`
- `reader_lora_grad_norm_by_layer`
- `memory_projection_output_l2`
- `cross_attn_gate_open_fraction` (for `RI2`)
- `opd_positive_token_fraction`
- `opd_mean_advantage`
- `hint_teacher_margin_minus_student_margin`

### 9.3 Writer metrics retained from V7

Keep:

- `writer_rank_fraction`
- `memory_long_common_mode_energy_ratio`
- `writer_memory_not_collapsed_strict`
- `projector_manufactured_diversity` where applicable

### 9.4 Strict acceptance tiers

#### Tier A — diagnostic alive

A branch is diagnostic-alive if:

- route runs end-to-end,
- train loss is stable,
- memory surface receives gradients,
- memory-consumption diagnostics are nontrivial.

#### Tier B — primary-usefulness signal

A branch is primary-useful if:

- at least one primary task shows `task_score_delta_vs_no_memory > 0`,
- the other primary task is non-regressive within noise,
- FEVER does not catastrophically regress.

#### Tier C — confirmation-ready

A branch is confirmation-ready if:

- it shows primary-task movement in scout,
- it keeps the route stable,
- it is not a pure metric illusion,
- and its main gain is not only on FEVER.

---

## 10. Engineering work required before live `PLANv8` phases

This section exists so the next executor does not underestimate the implementation delta.

## 10.1 Backbone / input pipeline work

### 10.1.1 Add a sequence-memory input path

New path needed in `backbone.py` or adjacent runtime utilities:

- accept memory embeddings / memory token block as true sequence positions
- support `prepend_block`
- support optional `interleave_every_k`
- produce correct `attention_mask`, `position_ids`, and score slicing
- support both `input_ids`-only and `inputs_embeds` modes cleanly

### 10.1.2 Add `Qwen3` real-HF smoke coverage

The repo already names `Qwen3-8B`, but `PLANv8` requires:

- real HF load smoke, not only stub smoke
- prompt calibration smoke
- GQA/KV cache path smoke
- gradient checkpointing smoke

### 10.1.3 Keep deep-prefix path intact as legacy control

Do not delete or silently mutate the V7 path.
`RI0` must remain available for continuity comparisons.

---

## 10.2 Reader adaptation implementation work

### 10.2.1 Add Reader-LoRA config family

New method surface:

```yaml
method:
  memory_consumer_mode: reader_lora_sequence
  reader_lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
  reader_lora_layers: [...]
  reader_lora_rank: 64
  reader_lora_alpha: 128
  reader_lora_dropout: 0.05
```

### 10.2.2 Add cross-attention Reader-adapter family

New method surface:

```yaml
method:
  memory_consumer_mode: reader_cross_attn
  reader_cross_attn_layers: [16,17,18,19]
  reader_cross_attn_heads: 16
  reader_cross_attn_gate_init: 0.0
  reader_cross_attn_ff_mult: 4
```

### 10.2.3 Add memory exposure projector

```yaml
method:
  memory_exposure_projector:
    type: mlp
    hidden_dim: 4096
    out_dim: 4096
    layers: 2
    activation: gelu
    pre_norm: true
    residual: true
```

---

## 10.3 OPD implementation work

The current training code already includes teacher-alignment primitives (`teacher_margin`, `teacher_choice_kl`, `teacher_choice_js`). `PLANv8` should extend this surface rather than invent a parallel stack.

### 10.3.1 New config keys

```yaml
runtime:
  pilot_alignment_aux_mode: opd_token_ce
  pilot_opd_scope: reader_only
  pilot_opd_weight_max: 0.3
  pilot_opd_start_step: 60
  pilot_opd_ramp_steps: 80
  pilot_opd_hint_mode_gsm8k: answer_plus_rationale
  pilot_opd_hint_mode_triviaqa: answer_plus_evidence
  pilot_opd_teacher_force_gold: true
  pilot_opd_mask_mode: target_only
  pilot_opd_advantage_clip: 5.0
  pilot_opd_center: 0.0
  pilot_opd_scale: 0.5
```

### 10.3.2 New helper modules

- hint constructor for `GSM8K`
- hint constructor for `TriviaQA`
- hint constructor for `FEVER`
- token-level teacher scoring
- token-level advantage computation
- token mask construction
- scope-based optimizer routing (`reader_only`, `writer_only`, `joint`)

### 10.3.3 Required tests

- advantage monotonicity
- hint masking correctness
- no hint leakage into eval
- scope routing correctness
- token-mask correctness for target-only loss
- reproducibility with fixed seeds

---

## 10.4 Writer implementation work

### 10.4.1 Add oracle hidden-state Writer family

This should be simple and deterministic:

- support encoding pass
- chunk pooling
- slot cap
- optional cache-to-disk

### 10.4.2 Add lightweight external Writer family

- learned latent slots
- cross-attention over support hidden states
- high-dimensional output
- no low-rank bottleneck by default

### 10.4.3 Add high-dimensional Writer cache

Because `Qwen3-8B` is expensive, add the option to cache:

- support hidden states
- oracle memory slots
- retrieval metadata
- tokenization manifests

This is mandatory for runtime sanity.

---

## 10.5 Summary / publish / governance work

New summary builders must report:

- Reader activation metrics
- OPD metrics
- interface family label
- Writer family label
- whether the gain came before or after Writer training
- whether the gain depended on compression
- whether the gain survived FEVER not overruling the primary tasks

---

## 11. `PLANv8` phase structure

`PLANv8` is organized into nine governed phases:

1. `V8-0` — Qwen3 baseline calibration + oracle/interface sanity
2. `V8-1` — Reader interface scout (CE only, Writer frozen/oracle)
3. `V8-2` — Reader sweep (rank / layers / LR)
4. `V8-3` — Reader OPD sweep
5. `V8-4` — external Writer reintroduction
6. `V8-5` — bridge / compression revisit
7. `V8-6` — Writer-side OPD and targeted auxiliary revisit
8. `V8-7` — paper-facing comparators
9. `V8-8` — multi-seed confirmation
10. `V8-9` — CDMI / cross-domain interference and paper closeout

`V8-9` is only authorized if a confirmation-ready branch exists.

---

# 12. Phase `V8-0` — Qwen3 baseline calibration and oracle/interface sanity

## 12.1 Purpose

This phase answers four questions:

1. Is `Qwen3-8B` actually producing a reasonable baseline on the current benchmark harness?
2. Are the current prompt templates underusing `Qwen3`’s thinking / formatting capabilities?
3. Does the old deep-prefix oracle remain flat on `Qwen3`?
4. Do sequence-memory and/or Reader-adapter interfaces at least pass basic sanity as candidate consumer routes?

This phase must be completed before any serious `Qwen3` training branch opens.

## 12.2 Required implementation before launch

- real-HF `Qwen3-8B` load smoke
- prompt calibration hooks
- sequence-memory input path
- oracle hidden-state Writer path
- summary fields for Qwen3 prompt mode

## 12.3 Arm list

### 12.3.1 Baseline calibration arms

| Arm ID | Task scope | thinking mode | prompt format | train? | purpose |
|---|---|---|---|---:|---|
| `b0_q3_gsm8k_nonthink` | GSM8K | off | concise final numeric answer | no | control |
| `b1_q3_gsm8k_think_boxed` | GSM8K | on | step-by-step + `\boxed{}` final | no | math baseline |
| `b2_q3_trivia_nonthink` | TriviaQA | off | concise answer only | no | QA baseline |
| `b3_q3_trivia_think` | TriviaQA | on | concise answer only | no | reasoning QA baseline |
| `b4_q3_fever_nonthink` | FEVER | off | label only | no | calibration |

### 12.3.2 Oracle and interface sanity arms

| Arm ID | Writer | Interface | Reader trainable | steps | purpose |
|---|---|---|---|---:|---|
| `o0_q25_prefix_replay` | legacy | legacy deep-prefix | no | 0 | continuity replay |
| `o1_q3_prefix_oracle_mid4` | `EW0` | `RI0` | no | 0 | oracle ceiling replay |
| `o2_q3_seq_oracle16` | `EW0` 16 slots | `RI1 prepend_block` | no | 0 | no-train sequence sanity |
| `o3_q3_seq_oracle32` | `EW0` 32 slots | `RI1 prepend_block` | no | 0 | no-train bigger memory sanity |
| `o4_q3_xattn_oracle_smoke` | `EW0` 16 slots | `RI2` | yes | 50 | adapter wiring smoke |

## 12.4 Default configs

### 12.4.1 Qwen3 baseline config

```yaml
experiment:
  stage: V8-0
  family: qwen3_baseline_calibration

backbone:
  name: Qwen3-8B
  model_id: Qwen/Qwen3-8B
  load_mode: hf_causal_lm
  dtype: bf16

runtime:
  eval_only: true
  seed: 61109
  eval_examples_gsm8k: 64
  eval_examples_triviaqa: 64
  eval_examples_fever: 48
```

### 12.4.2 Sequence-memory oracle config

```yaml
method:
  writer_family: EW0
  memory_consumer_mode: reader_lora_sequence
  memory_segment_mode: prepend_block
  reader_lora_enabled: false
  writer_extract_layer: 18
  writer_slot_pool_window: 16
  writer_slot_cap: 16   # 32 for o3
```

### 12.4.3 Cross-attention smoke config

```yaml
method:
  writer_family: EW0
  memory_consumer_mode: reader_cross_attn
  reader_cross_attn_layers: [16,17,18,19]
  reader_cross_attn_heads: 16
  reader_cross_attn_gate_init: 0.0
  trainable_variant: reader_only
runtime:
  train_steps: 50
  learning_rate: 5e-5
```

## 12.5 Acceptance criteria

This phase passes if all of the following are true:

1. `Qwen3-8B` real-HF load works end-to-end.
2. One calibrated prompt mode is selected per primary task and frozen for later phases.
3. `Qwen3` baseline on at least one primary task is meaningfully above the `Qwen2.5-1.5B` baseline.
4. The legacy prefix oracle result is reproduced or clearly bounded on `Qwen3`.
5. Sequence-memory and cross-attention interfaces both pass basic smoke without shape or masking bugs.

## 12.6 Kill / repair criteria

Stop and repair before `V8-1` if:

- `Qwen3-8B` baseline is implausibly low due to prompt or extraction mismatch,
- real-HF load fails,
- sequence-memory input path corrupts score slicing,
- GQA path fails or gives shape errors,
- cross-attention adapter explodes numerically.

---

# 13. Phase `V8-1` — Reader interface scout (CE only, Writer frozen/oracle)

## 13.1 Purpose

Now that `Qwen3` is calibrated, `V8-1` asks:

> Which Reader interface gives the best first evidence that the backbone can actually consume external memory?

The Writer is intentionally kept frozen / oracle here.

## 13.2 Promotion rule

A branch may promote even without immediate task-score gain if it shows:

- strong memory-consumption diagnostics,
- positive answer-logprob movement,
- route stability,
- and non-regression.

But no branch may promote purely from FEVER.

## 13.3 Arm matrix

| Arm ID | Interface | memory slots | reader layers | rank | LR | steps |
|---|---|---:|---|---:|---:|---:|
| `i0_prefix_legacy_r2` | `RI0` | 16 | `[16,17,18,19]` tiny legacy | 2 | `7.5e-5` | 200 |
| `i1_seq16_r32_mid8` | `RI1 prepend_block` | 16 | `mid8` | 32 | `5e-5` | 200 |
| `i2_seq16_r64_mid8` | `RI1 prepend_block` | 16 | `mid8` | 64 | `1e-4` | 200 |
| `i3_seq32_r64_mid8` | `RI1 prepend_block` | 32 | `mid8` | 64 | `1e-4` | 200 |
| `i4_xattn16_mid4_r32` | `RI2` | 16 | `[16,17,18,19]` | 32 | `5e-5` | 200 |
| `i5_xattn16_mid4_r64` | `RI2` | 16 | `[16,17,18,19]` | 64 | `1e-4` | 200 |

## 13.4 Global config block

```yaml
runtime:
  train_steps: 200
  warmup_steps: 20
  train_batch_size_per_device: 2
  gradient_accumulation_steps: 8
  clip_mode: groupwise
  clip_norm: 1.0
  weight_decay: 0.01
  seed: 61109

method:
  writer_family: EW0
  trainable_variant: reader_only
  bridge_family: BR0
  opd_enabled: false
```

## 13.5 What gets compared

Primary comparison order:

1. `GSM8K` task score delta
2. `TriviaQA` task score delta
3. sum of primary deltas
4. answer-logprob delta on primary tasks
5. memory-consumption diagnostics
6. FEVER only as non-regression sanity

## 13.6 Expected outcomes and interpretation

### Best-case

One sequence or cross-attn arm shows positive primary movement already.

### Acceptable-case

No score movement yet, but one arm shows:

- clearly better answer-logprob movement,
- clearly better memory attention / gate activation,
- stable training.

In this case `V8-2` is still authorized.

### Bad-case

Everything is flat and the old prefix arm is indistinguishable from the new arms.

If so, the consumer change did not help enough yet, and `V8-2` must become a hard last chance before major reconsideration.

---

# 14. Phase `V8-2` — Reader sweep (rank / layer band / LR)

## 14.1 Purpose

`V8-2` assumes one Reader interface family is better than the others.
It then optimizes only the most important consumer variables:

- layer band
- LoRA rank
- learning rate

Do **not** reopen Writer training yet.

## 14.2 Arm matrix

Assume the winner from `V8-1` is the mainline interface. Then run:

| Arm ID | layer band | rank | LR | steps |
|---|---|---:|---:|---:|
| `r0_mid8_r32_lr5e5` | `mid8` | 32 | `5e-5` | 300 |
| `r1_mid8_r64_lr1e4` | `mid8` | 64 | `1e-4` | 300 |
| `r2_mid12_r64_lr1e4` | `mid12` | 64 | `1e-4` | 300 |
| `r3_mid12_r64_lr2e4` | `mid12` | 64 | `2e-4` | 300 |
| `r4_late8_r32_lr1e4` | `late8` | 32 | `1e-4` | 300 |
| `r5_mid8_r16_lr5e5` | `mid8` | 16 | `5e-5` | 300 |

## 14.3 Default shared settings

```yaml
method:
  writer_family: EW0
  trainable_variant: reader_only
  bridge_family: BR0
  opd_enabled: false
runtime:
  train_steps: 300
  warmup_steps: 30
  eval_every: 50
```

## 14.4 What this phase is trying to learn

- Does the Reader need broader layer coverage than the old mid4 intuition?
- Is `rank=64` actually necessary on `Qwen3-8B`?
- Is the consumer underfitting at `5e-5` and waking up at `1e-4` or `2e-4`?

## 14.5 Acceptance criteria

Promote the top Reader config if any of the following hold:

1. primary-task score moves positively on either `GSM8K` or `TriviaQA`,
2. or answer-logprob movement is consistently positive on both primary tasks with strong reader activation,
3. and training remains stable.

## 14.6 Hard gate before `V8-3`

If after `V8-2` there is still absolutely no consumer-side signal—not even in answer logprob or memory-attention metrics—then the project should **not** open trainable Writer work yet. In that case the next action is to compare with a comparator route earlier than planned.

---

# 15. Phase `V8-3` — Reader OPD sweep

## 15.1 Purpose

This is the phase where `PLANv8` explicitly attacks gradient starvation.

The question is:

> If the Reader is the main trainable consumer, can OPD-style hint-enhanced token supervision turn “diagnostic movement only” into actual primary-task movement?

## 15.2 Fixed base for this phase

- best `V8-2` Reader interface
- Writer remains `EW0` (oracle / frozen)
- no bridge by default
- only Reader and exposure projector trainable

## 15.3 Arm matrix

| Arm ID | OPD mode | hint strength | λ_opd_max | start | ramp | scope |
|---|---|---|---:|---:|---:|---|
| `p0_ce_only` | off | none | 0.0 | — | — | reader |
| `p1_teacher_choice_kl` | legacy dense teacher KL | task default | 0.1 | 60 | 80 | reader |
| `p2_opd_ansonly_w01` | token CE | answer only | 0.1 | 60 | 80 | reader |
| `p3_opd_ansonly_w03` | token CE | answer only | 0.3 | 60 | 80 | reader |
| `p4_opd_ansplusctx_w03` | token CE | answer + rationale/evidence | 0.3 | 60 | 80 | reader |
| `p5_opd_ansplusctx_centered` | token CE centered | answer + rationale/evidence | 0.3 | 60 | 80 | reader |

## 15.4 Default OPD settings

```yaml
runtime:
  train_steps: 300
  warmup_steps: 30
  pilot_alignment_aux_mode: opd_token_ce
  pilot_opd_scope: reader_only
  pilot_opd_teacher_force_gold: true
  pilot_opd_mask_mode: target_only
  pilot_opd_advantage_clip: 5.0
  pilot_opd_center: 0.0
  pilot_opd_scale: 0.5
```

## 15.5 Task-specific hints

### GSM8K default hint progression

- `answer_only`: final numeric answer
- `answer_plus_rationale`: final answer + first `48` tokens of gold rationale
- `answer_plus_rationale64`: final answer + first `64` tokens of gold rationale

### TriviaQA default hint progression

- `answer_only`: gold answer string
- `answer_plus_evidence`: answer + one evidence sentence
- `answer_plus_two_evidence`: answer + two shortest evidence sentences

### FEVER default hint progression

- `label_only`
- `label_plus_evidence`

## 15.6 What success means in this phase

This is the first phase where **actual primary-task score movement** is the expected bar.

A phase-3 success is:

- positive `GSM8K` or `TriviaQA` task score delta vs no-memory baseline,
- non-regression on the other primary task,
- and reader activation consistent with real memory use.

## 15.7 Kill criterion after `V8-3`

If `V8-3` produces **no actual primary-task movement** across all OPD arms, then `PLANv8` does **not** authorize a full trainable external-Writer branch as the mainline.

In that scenario the project should still preserve the external Writer as a thesis, but it must move comparator work earlier and consider a stronger architectural shift.

---

# 16. Phase `V8-4` — trainable external Writer reintroduction

## 16.1 Purpose

Only after the Reader is shown to consume memory should the project reopen the trainable external Writer.

This phase asks:

> Once the consumer is competent, can a trainable external Writer outperform the frozen/oracle Writer?

## 16.2 Fixed base for this phase

- best Reader config from `V8-3`
- OPD retained if helpful
- bridge off by default (`BR0`)
- Writer becomes the new experimental variable

## 16.3 Arm matrix

| Arm ID | Writer family | slots | blocks | writer LR | stage schedule |
|---|---|---:|---:|---:|---|
| `w0_oracle64` | `EW0` | 64 | 0 | — | control |
| `w1_ext2layer64_lr2e5` | `EW1` | 64 | 2 | `2e-5` | 80 writer-only + joint |
| `w2_ext3layer64_lr2e5` | `EW2` | 64 | 3 | `2e-5` | 80 writer-only + joint |
| `w3_ext3layer96_lr1e5` | `EW3` | 96 | 3 | `1e-5` | 80 writer-only + joint |
| `w4_ext3layer64_lr5e5` | `EW2` | 64 | 3 | `5e-5` | 80 writer-only + joint |

## 16.4 Default training schedule

```yaml
runtime:
  train_steps: 300
  stage_a_steps: 80        # writer/exposure only
  stage_b_steps: 220       # joint with reader
method:
  trainable_variant: writer_then_joint
```

## 16.5 Why this schedule exists

The Writer still has a harder gradient path than the Reader.
Giving it a short isolated warm start is justified once the consumer is already known to work.

## 16.6 Promotion logic

Promote only if a trainable Writer beats the `EW0` control on actual primary score or on very strong non-score evidence that is likely to convert at confirmation scale.

If the oracle/frozen Writer remains best, that is **not** a failure.
It is a scientifically meaningful result:

> external memory may be more valuable as a frozen high-dimensional support encoder than as a fully trainable latent producer under the current budget.

---

# 17. Phase `V8-5` — bridge / compression revisit

## 17.1 Purpose

This phase is only authorized if either:

- the trainable external Writer helps,
- or the no-bridge route is clearly too expensive.

The question is:

> Can `M_long` be compressed without destroying the new Reader-consumption gains?

## 17.2 Arm matrix

| Arm ID | bridge family | input slots | queries | short slots |
|---|---|---:|---:|---:|
| `b0_no_bridge` | `BR0` | 64 | — | 64 |
| `b1_q16_s16` | `BR1` | 64 | 16 | 16 |
| `b2_q32_s16` | `BR2` | 64 | 32 | 16 |
| `b3_q32_s8` | `BR3` | 64 | 32 | 8 |
| `b4_q48_s16_x96` | `BR2` | 96 | 48 | 16 |

## 17.3 Default bridge settings

```yaml
method:
  bridge_mode: learned_query_compressor
  bridge_query_dim: 4096
  bridge_pre_norm: true
  bridge_output_norm: true
  bridge_ff_mult: 2
  bridge_dropout: 0.05
```

## 17.4 Promotion criteria

Compression promotes only if it:

- preserves or improves primary-task score,
- reduces compute or memory cost materially,
- and does not turn the route into another collapse illusion.

If compression hurts, the correct conclusion is:
keep the high-dimensional route longer.

---

# 18. Phase `V8-6` — Writer-side OPD and targeted auxiliary revisit

## 18.1 Purpose

Now that the consumer is working and the Writer has been reintroduced, this phase asks:

> Which targeted auxiliary signals help the Writer encode more useful, less collapsed memory without reopening another huge loss zoo?

## 18.2 Principles

Only revisit auxiliary losses that are mechanistically motivated by V7.

That means keeping:

- `Barlow` because it showed the best helpfulness signal in `V7-5`
- reconstruction because content-specific supervision is still plausible
- Writer-side OPD because it directly answers the owner’s proposal

Do **not** reopen a giant undirected sweep.

## 18.3 Arm matrix

| Arm ID | Auxiliary family | description |
|---|---|---|
| `a0_none` | none | control |
| `a1_barlow` | Barlow-lite | preserve V7’s strongest helpfulness family |
| `a2_recon_bow` | reconstruction-lite | hashed bag-of-entities / tokens |
| `a3_writer_opd_ans` | Writer OPD | answer-only hint |
| `a4_writer_opd_ansctx` | Writer OPD | answer + rationale/evidence hint |
| `a5_writer_opd_plus_recon` | hybrid | Writer OPD + reconstruction |

## 18.4 Default weights

```yaml
runtime:
  barlow_weight: 0.02
  reconstruction_weight: 0.02
  writer_opd_weight_max: 0.1
  writer_opd_start_step: 80
  writer_opd_ramp_steps: 80
```

## 18.5 Acceptance criteria

Promote only if the auxiliary helps actual primary-task score or clearly improves the Writer without hurting the best route.

---

# 19. Phase `V8-7` — paper-facing comparators

## 19.1 Purpose

`PLANv8` should not wait until the very end to think about comparators.
This phase runs the minimum paper-facing baselines needed to interpret the result.

## 19.2 Comparator set

| Comparator ID | Description | Why included |
|---|---|---|
| `m0_nomemory_qwen3` | best prompt-calibrated `Qwen3-8B` no-memory baseline | floor |
| `m1_text_rag_qwen3` | text retrieval baseline | external non-latent memory comparator |
| `m2_memgen_qwen3` | `MemGen` comparator where feasible | in-model latent memory comparator |
| `m3_legacy_planv7_qwen25` | best V7 branch | historical negative/continuity comparator |
| `m4_best_v8` | best current route | candidate final system |

## 19.3 Important note

The repo already contains `Qwen3-8B` references and `MemGen`-related configs.
Use that existing work rather than reopening the comparator question from zero.

## 19.4 What this phase must answer

- Is the new route better than plain no-memory?
- Is it better than text RAG?
- Is it competitive with a MemGen-like route?
- Is its contribution still externally meaningful if MemGen wins?

---

# 20. Phase `V8-8` — multi-seed confirmation

## 20.1 Purpose

This is the governed confirmation phase for the best `1–3` branches only.

## 20.2 Confirmation candidates

Promote at most:

- best Reader-only OPD branch
- best trainable external Writer branch
- best bridge-compressed branch (only if compression helped)

## 20.3 Confirmation settings

```yaml
runtime:
  seeds: [61109, 61110, 61111]
  train_steps: 400
  eval_examples_gsm8k: 128
  eval_examples_triviaqa: 128
  eval_examples_fever: 96
```

## 20.4 Confirmation success

A branch is confirmation-successful if:

- it shows positive movement across at least one primary task in the 3-seed aggregate,
- it stays non-regressive on the other primary task,
- FEVER remains acceptable,
- and the route is still clearly real.

---

# 21. Phase `V8-9` — CDMI / cross-domain interference and paper closeout

This phase is only authorized if `V8-8` succeeds.

## 21.1 Purpose

This is where the project returns to the broader MemTOTAL paper story:

- write-long / read-short memory
- cross-domain memory use
- CDMI / interference measurement

## 21.2 Minimal CDMI experiment design

Train the best branch jointly on `GSM8K` + `TriviaQA`.

Then evaluate:

1. math-trained memory on math
2. trivia-trained memory on trivia
3. joint-trained memory on both
4. math-focused support exposed during trivia inference
5. trivia-focused support exposed during math inference

Measure:

- score deltas
- memory attention allocation
- negative transfer rate
- domain-conditioned gate / consumer shift
- whether compression worsens cross-domain leakage

## 21.3 Why CDMI is late

Because cross-domain claims are meaningless unless the base route actually works on one domain first.

---

## 22. Learning-rate policy for `PLANv8`

The user explicitly requested more LR variation.
`PLANv8` therefore uses **small, local LR sweeps** rather than pretending one inherited setting is sufficient.

### 22.1 Reader LR sweep family

- `5e-5`
- `1e-4`
- `2e-4`

### 22.2 Writer LR sweep family

- `1e-5`
- `2e-5`
- `5e-5`

### 22.3 Bridge / exposure projector LR sweep family

- `5e-5`
- `7.5e-5`
- `1e-4`

### 22.4 OPD weight sweep family

- `0.1`
- `0.3`
- `0.5` only if the lower two are stable

### 22.5 What not to do

Do not reopen giant full-factorial sweeps.
Every LR sweep must remain local to one phase.

---

## 23. Detailed prompt and target policies

Because `Qwen3` is a different backbone family, prompt policy matters.

### 23.1 GSM8K policy

Default evaluation prompt:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

Training target policy:

- keep rationale + final answer if token budget allows
- cap target at `256` tokens
- extract final numeric answer robustly

### 23.2 TriviaQA policy

Default evaluation prompt:

```text
Answer the question as concisely as possible. Output only the final answer.
```

Thinking mode is calibrated in `V8-0`, but default remains non-thinking unless calibration clearly prefers thinking.

### 23.3 FEVER policy

Default evaluation prompt:

```text
Output one label only: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO.
```

---

## 24. Unit tests and validation checklist

`PLANv8` must not repeat the phase-script fragility that surfaced during the LR-updated replay.

### 24.1 Static validation

Run at minimum:

```bash
bash -n scripts/run_planv8_*.sh
python -m unittest tests.test_repo_lints tests.test_repo_contract -v
python -m unittest \
  tests.test_qwen3_sequence_memory \
  tests.test_reader_lora_targets \
  tests.test_opd_alignment \
  tests.test_writer_oracle_slots \
  tests.test_planv8_summaries -v
```

### 24.2 Required new tests

1. `Qwen3` real-HF load smoke
2. sequence-memory input ids / embeds alignment
3. score slicing with prepended memory block
4. LoRA target mapping on `Qwen3` `q_proj/k_proj/v_proj/o_proj`
5. cross-attention adapter gate initialization
6. OPD hint constructor correctness
7. no hint leakage into eval
8. writer cache reproducibility
9. summary-builder correctness for new metrics

---

## 25. Concrete script and config deliverables

The next coding agent should create, at minimum:

### 25.1 Scripts

- `scripts/run_planv8_v8_0_qwen3_baselines_oracles.sh`
- `scripts/run_planv8_v8_1_reader_interface_scout.sh`
- `scripts/run_planv8_v8_2_reader_sweep.sh`
- `scripts/run_planv8_v8_3_reader_opd.sh`
- `scripts/run_planv8_v8_4_external_writer.sh`
- `scripts/run_planv8_v8_5_bridge.sh`
- `scripts/run_planv8_v8_6_writer_aux_opd.sh`
- `scripts/run_planv8_v8_7_comparators.sh`
- `scripts/run_planv8_v8_8_multiseed_confirmation.sh`
- `scripts/run_planv8_v8_9_cdmi.sh`

### 25.2 Config families

- `configs/method/reader_lora_sequence_qwen3_*.yaml`
- `configs/method/reader_cross_attn_qwen3_*.yaml`
- `configs/method/external_writer_oracle_qwen3_*.yaml`
- `configs/method/external_writer_learned_qwen3_*.yaml`
- `configs/method/bridge_query_compressor_qwen3_*.yaml`
- `configs/exp/benchmark_gsm8k_qwen3_real_*.yaml`
- `configs/exp/benchmark_triviaqa_qwen3_real_*.yaml`
- `configs/exp/benchmark_fever_qwen3_real_*.yaml`
- `configs/exp/memgen_gsm8k_qwen3_eval.yaml` integration patches if needed

### 25.3 Summary builders

- `src/memtotal/analysis/planv8_reader_summary.py`
- `src/memtotal/analysis/planv8_opd_summary.py`
- `src/memtotal/analysis/planv8_comparator_summary.py`

---

## 26. Decision tree for `PLANv8`

This section prevents drift.

### 26.1 If `V8-0` fails

Repair environment / prompt calibration / `Qwen3` support first.
Do not continue.

### 26.2 If `V8-1` shows a better consumer interface

Open `V8-2`.

### 26.3 If `V8-2` still shows no signal

Open `V8-3` as the last consumer-first attempt.
Do not open trainable Writer yet.

### 26.4 If `V8-3` still shows no actual primary-task movement

Do not treat “Path Q” as a license to keep optimizing the same idea blindly.
Open comparator work and prepare a more serious architectural pivot memo while preserving the external-Writer thesis as unresolved.

### 26.5 If `V8-3` succeeds

Open `V8-4` and reintroduce the trainable external Writer.

### 26.6 If `V8-4` shows oracle Writer remains best

That is acceptable.
Keep the external Writer in the scientific story, but report that the current budget favors a frozen/oracle external producer.

### 26.7 If `V8-5` shows compression hurts

Keep full `M_long` longer.
Do not force `M_short` prematurely just because it was the original architectural ideal.

### 26.8 If `V8-8` succeeds

Open `V8-9` and start paper assembly.

---

## 27. Why this plan is scientifically stronger than another V7-style continuation

`PLANv8` is stronger than another V7-style continuation for seven reasons.

1. It preserves the external Writer **without** forcing the old KV-prefix channel to carry the whole idea.
2. It resolves the 1.5B confound by making `Qwen3-8B` the main backbone.
3. It accepts the oracle result instead of arguing with it.
4. It moves the trainable burden to the Reader, where the gradient is strongest.
5. It introduces OPD exactly where the project needs more directional signal.
6. It uses the existing harness instead of starting over.
7. It keeps comparator and CDMI work in view, so the project can become paper-ready instead of endless-optimization-ready.

---

## 28. Honest risks and mitigations

### 28.1 Risk: `Qwen3` baseline is lower than expected because the harness prompt is wrong

**Mitigation:** treat prompt calibration as mandatory `V8-0`, not optional setup.

### 28.2 Risk: sequence-memory path behaves like another soft-prompt and still underperforms

**Mitigation:** keep `RI2` cross-attention Reader-adapter as the secondary mainline consumer family.

### 28.3 Risk: Reader-LoRA destabilizes the model

**Mitigation:** start with:
- rank `32`
- mid8 only
- LR `5e-5`
- weight decay `0.01`
- groupwise clip `1.0`

### 28.4 Risk: OPD hint design leaks too much or becomes task-specific overfitting

**Mitigation:**
- keep hints strictly training-only,
- start with answer-only,
- then move to answer+context,
- log hint type in all artifacts.

### 28.5 Risk: trainable external Writer never beats oracle/frozen Writer

**Mitigation:** accept that outcome honestly.
It is still a publishable and useful result if the Reader route works.

### 28.6 Risk: bridge / compression hurts everything

**Mitigation:** do not force compression early.
A full-dimensional external memory can still be the right answer under a 96GB budget.

### 28.7 Risk: comparator work shows MemGen dominates

**Mitigation:** that is scientifically valuable.
It would narrow the contribution to:
- external vs in-model latent memory,
- consumer-learning vs producer-learning,
- CDMI measurement,
- and negative-result rigor around prefix/KV injection.

---

## 29. Minimal paper story if `PLANv8` succeeds

If the route works, the paper story becomes:

1. **Negative result:** KV-prefix external memory on the old route stays flat on primary generative tasks even under oracle content.
2. **Mechanistic diagnosis:** the problem is not pure route deadness; it is weak consumption + gradient starvation.
3. **Positive result:** external Writer becomes useful when the consumer is a Reader-LoRA / Reader-adapter on a stronger backbone.
4. **Training result:** OPD-style hint-enhanced teacher advantages improve memory consumption learning.
5. **Architectural result:** high-dimensional `M_long` can remain external and only be compressed if the consumer is already competent.
6. **Broader result:** cross-domain interference can then be measured honestly.

That is a substantially stronger story than “we tried more projector variants.”

---

## 30. Final blunt conclusion

The project should **not** close the external-Writer thesis here.

But it should also stop pretending that one more old-style width sweep on `Qwen2.5-1.5B` will settle the issue.

The right successor branch is:

> **Qwen3-8B as the main backbone, Reader-first consumption learning, OPD on the Reader, trainable external Writer second, bridge/compression third, confirmation and CDMI last.**

That is the shortest route that:

- respects the owner’s hard constraints,
- respects the evidence from `Path Q`,
- addresses the actual bottlenecks,
- and still keeps the long-term MemTOTAL idea intact.

---

## 31. References

1. Qwen Team. **Qwen3 Technical Report.** arXiv:2505.09388, 2025.  
2. Qwen Team. **Qwen3-8B model card.** Hugging Face model card, 2025.  
3. Hu, E. J. et al. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685 / ICLR 2022.  
4. Dettmers, T. et al. **QLoRA: Efficient Finetuning of Quantized LLMs.** arXiv:2305.14314 / NeurIPS 2023.  
5. Li, X. L. & Liang, P. **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** arXiv:2101.00190 / ACL 2021.  
6. Petrov, A., Torr, P. H. S., & Bibi, A. **When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations.** arXiv:2310.19698 / ICLR 2024.  
7. Mu, J., Li, X. S., & Goodman, N. **Learning to Compress Prompts with Gist Tokens.** arXiv:2304.08467 / NeurIPS 2023.  
8. Ge, T. et al. **In-context Autoencoder for Context Compression in a Large Language Model.** arXiv:2307.06945 / ICLR 2024.  
9. Alayrac, J.-B. et al. **Flamingo: a Visual Language Model for Few-Shot Learning.** arXiv:2204.14198 / NeurIPS 2022.  
10. Jaegle, A. et al. **Perceiver: General Perception with Iterative Attention.** arXiv:2103.03206 / ICML 2021.  
11. Zhang, G., Fu, M., & Yan, S. **MemGen: Weaving Generative Latent Memory for Self-Evolving Agents.** arXiv:2509.24704, 2025.  
12. Wang, Y. et al. **OpenClaw-RL: Train Any Agent Simply by Talking.** arXiv:2603.10165, 2026.  
