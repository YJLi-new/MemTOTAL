# PLANv9.md — Resolve the FlashMem Discrepancy First, Then Pivot MemTOTAL to Long-Horizon Agent Memory

**Version:** 2026-03-15  
**Scope:** successor execution authority after the qwen34 `PLANv8` comparator hold  
**Audience:** project owner / core contributors / future agents operating inside this repository  
**Primary language:** English  
**Status intent:** actionable research authority, not a lightweight memo

---

## 0. What this document is, and what it is not

This document is **not** a retroactive rewrite of `PLANv7.md`, `PLANv7-LR-updated.md`, or `PLANv8.md`.

Those documents remain historically important and scientifically binding:

- `PLANv7-LR-updated` closed the qwen25 replay at **Path Q**:
  - `comparison_conclusion = path_q_external_writer_unresolved_not_dead`
  - `recommended_next_step = open_stronger_integrated_writer_or_true_highdim_branch`
- The current repo is explicitly governed by the qwen34 `PLANv8` line, and that line is paused at the comparator gate:
  - `comparison_conclusion = comparators_do_not_support_v8_8`
  - `recommended_next_step = hold_v8_8_comparator_review`

This document therefore does **five** things at once:

1. **Preserves** the scientific meaning of V6–V8 instead of erasing it.
2. **Cross-examines** the latest code and result artifacts, not just prior plan text.
3. **Resolves** the single highest-value remaining discrepancy: the FlashMem vs. V8 in-stream-memory contradiction.
4. **Pivots** the main evaluation regime away from over-investment in single-turn tasks and toward long-horizon agent memory.
5. **Transplants** the unfinished, still-valuable pieces of earlier plans—especially high-dimensional memory, read-compress logic, and CDMI—into a better-matched benchmark and architecture regime.

The core strategic move of PLANv9 is simple:

> **Do not reopen broad single-turn latent-memory sweeps.**  
> First resolve whether a FlashMem-style backbone-processed latent memory route is actually non-destructive on your stack.  
> Then, if it survives that gate, move the mainline to long-horizon agent memory, where latent memory is theoretically justified and empirically more likely to matter.

That ordering follows the logic of the repository evidence **and** your newest hard constraint: prioritize long-duration agent tasks, use FlashMem’s Shared-KV consolidator as the conceptual anchor, but follow the narrow discrimination test in text **【5】** before making a large engineering bet.

---

## 1. Inputs this plan assumes have been reviewed

This plan assumes the following were reviewed together and cross-compared:

### 1.1 Repository state

- latest public repository state on GitHub
- current README / review-facing summaries
- latest execution-plan documents
- active training/evaluation scripts
- active model/training/task code paths

### 1.2 Uploaded and bundled documents

- `PLANv4.md`
- `PLANv6dot1.md`
- `PLANv7.md`
- `PLANv8.md`
- the latest uploaded review snapshot zip (`MemTOTAL-review (4).zip`)
- the historical review bundles where useful for consistency checks

### 1.3 Local repo documents inside the review bundle

Especially:

- `README.md`
- `docs/MAIN_IDEA.md`
- `docs/ARCHITECTURE.md`
- `docs/EXPERIMENTS_INFO.md`
- `docs/TODO_LIST.md`
- `docs/tech-debt-tracker.md`
- `docs/exec-plans/active/20260315-planv8-v8-8-multiseed-confirmation-qwen34.md`
- `docs/exec-plans/active/20260315-planv8-v8-9-cdmi-qwen34.md`

### 1.4 Local code paths inspected

Especially:

- `src/memtotal/models/backbone.py`
- `src/memtotal/models/memory.py`
- `src/memtotal/training/m4_shared_injection.py`
- `src/memtotal/tasks/registry.py`

### 1.5 Result artifacts reviewed

Especially:

- `results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json`
- `results/generated/review/planv8-v8-0-qwen34-baselines-oracles/v8-0-summary.json`
- `results/generated/review/planv8-v8-1-reader-interface-scout-qwen34/v8-1-summary.json`
- `results/generated/review/planv8-v8-2-reader-sweep-qwen34/v8-2-summary.json`
- `results/generated/review/planv8-v8-3-reader-opd-qwen34/v8-3-summary.json`
- `results/generated/review/planv8-v8-7-comparators-qwen34/v8-7-summary.json`

### 1.6 Literature that materially affects the next step

At minimum:

- Petrov, Torr, Bibi (2024): theoretical limits of prompting/prefix tuning
- PrefixMemory-Tuning / Prefix-Tuning+ (2025/2026): decoupling prefix from attention
- FlashMem (2026): Shared-KV consolidation + soft injection
- LongMemEval (2025): indexing / retrieval / reading decomposition for long-term memory
- LoCoMo (2024): very long-term conversational memory evaluation
- MemoryAgentBench (2025/2026): four core competencies for memory agents
- LongBench v2 (2025): realistic long-context tasks including code-repository understanding
- LongMem (2023): frozen-backbone memory encoder + adaptive residual retriever/reader
- A-MEM (2025), RMM (2025), EMG-RAG (2024), AgentFly (2025) as structured or policy-level memory alternatives

---

## 2. The current project position, frozen in one place

This section deliberately freezes the current state so future work cannot drift, soften, or selectively reinterpret the evidence.

## 2.1 Repo-level status

The public repo README now states that the project is governed by the qwen34 `PLANv8` line, the latest completed milestone is `V8-7 comparators`, and the current outcome is a governed hold before `V8-8`, not an infra failure or an unpublished partial state. That means PLANv9 must be a **successor authority**, not a hidden continuation of V8 under a different name.  
Source anchor in repo: `README.md`

## 2.2 What qwen25 `PLANv7-LR-updated` settled

The qwen25 replay did **not** vindicate the external Writer route, but it did refine the conclusion:

- the route was **not merely projector-only**
- the route was **not dead in the trivial sense**
- but the best confirmed variant still showed no real three-seed primary-task gain on GSM8K or TriviaQA

The correct reading is:

> qwen25 Path Q downgraded the conclusion from “dead” to “unresolved,” but did not establish task-level viability.

That matters because it means the old branch still had some live internals worth transplanting—especially the idea of a stronger integrated Writer and a true high-dimensional `M_long`—but it did **not** earn the right to keep consuming single-turn benchmark budget in the same form.

## 2.3 What qwen34 `PLANv8` settled

V8 is the decisive line because it moved the backbone from “too weak to reason” into “competent enough to be harmed.”

### Baseline state from `V8-0`

Selected qwen34 baseline scores:

- GSM8K: **0.8125**
- TriviaQA: **0.0312**
- FEVER: **0.7917**

This is the first stage of the project where the backbone had meaningful task capability on GSM8K.

### Oracle findings from `V8-0`

- Legacy prefix oracle:
  - GSM8K: **0.7812**
  - TriviaQA: **0.0312**
- Sequence oracle:
  - GSM8K: **0.2656**
  - TriviaQA: **0.0000**
- Cross-attention oracle smoke:
  - GSM8K: **0.0000**
  - TriviaQA: **0.0000**

Interpretation:

- legacy prefix became **non-destructive but flat**
- in-sequence memory became **catastrophically destructive**
- cross-attention became **even more destructive**

### Interface-scout findings from `V8-1`

The qwen34 interface scout reproduced the oracle pattern after training.

| Arm | Interface family | GSM8K Δ vs control | TriviaQA Δ vs control | FEVER Δ vs control | Primary activation score | Cross-attn gate |
|---|---|---:|---:|---:|---:|---:|
| `i0_prefix_legacy_r2` | ri0_legacy_prefix | +0.0000 | +0.0000 | +0.0417 | 0.000123 | 0.000 |
| `i1_seq16_r32_mid8` | ri1_prepend_block | -0.4844 | -0.0312 | -0.1250 | 0.099490 | 0.000 |
| `i2_seq16_r64_mid8` | ri1_prepend_block | -0.5625 | -0.0312 | -0.0625 | 0.097032 | 0.000 |
| `i3_seq32_r64_mid8` | ri1_prepend_block | -0.5000 | -0.0312 | -0.0833 | 0.099973 | 0.000 |
| `i4_xattn16_mid4_r32` | ri2_cross_attn | -0.7344 | +0.0000 | +0.1458 | 0.754893 | 0.500 |
| `i5_xattn16_mid4_r64` | ri2_cross_attn | -0.7812 | -0.0312 | +0.0833 | 1.010141 | 0.497 |

This is one of the most important tables in the entire project history. It shows that:

- legacy prefix is the **only** route that does not crash the competent backbone
- sequence and cross-attention routes are **not ignored**
- they are actively consumed (`primary_activation_score_mean` is large relative to legacy prefix)
- yet they damage reasoning instead of helping it

### Reader sweep from `V8-2`

Every reader-only rescue attempt within the legacy prefix family remained flat-to-negative.

| Arm | Layer band | Rank label | LR tag | GSM8K Δ vs control | TriviaQA Δ vs control | Primary activation score |
|---|---|---|---|---:|---:|---:|
| `r0_mid8_r32_lr5e5` | mid8 | r32 | lr5e5 | -0.0312 | -0.0156 | 0.000098 |
| `r1_mid8_r64_lr1e4` | mid8 | r64 | lr1e4 | -0.0469 | -0.0312 | 0.000100 |
| `r2_mid12_r64_lr1e4` | mid12 | r64 | lr1e4 | -0.0312 | +0.0000 | 0.000141 |
| `r3_mid12_r64_lr2e4` | mid12 | r64 | lr2e4 | -0.0156 | -0.0312 | 0.000143 |
| `r4_late8_r32_lr1e4` | late8 | r32 | lr1e4 | -0.0469 | -0.0312 | 0.010415 |
| `r5_mid8_r16_lr5e5` | mid8 | r16 | lr5e5 | -0.0469 | -0.0312 | 0.000096 |

### OPD from `V8-3`

The OPD phase did not produce the mechanism it was supposed to produce on primary tasks.

| Arm | Interface family | GSM8K Δ vs control | TriviaQA Δ vs control | GSM8K OPD positive-token fraction | TriviaQA OPD positive-token fraction |
|---|---|---:|---:|---:|---:|
| `p0_ce_only` | ri0_legacy_prefix | +0.0000 | +0.0000 | 0.000 | 0.000 |
| `p1_teacher_choice_kl` | ri0_legacy_prefix | +0.0000 | +0.0312 | 0.000 | 0.000 |
| `p2_opd_ansonly_w01` | ri0_legacy_prefix | +0.0312 | +0.0156 | 0.000 | 0.000 |
| `p3_opd_ansonly_w03` | ri0_legacy_prefix | +0.0469 | +0.0000 | 0.000 | 0.000 |
| `p4_opd_ansplusctx_w03` | ri0_legacy_prefix | -0.0156 | +0.0000 | 0.000 | 0.000 |
| `p5_opd_ansplusctx_centered` | ri0_legacy_prefix | +0.0781 | +0.0000 | 0.000 | 0.000 |

Even where the final score sum moved slightly, the actual intended OPD signal on primary tasks remained effectively absent.

### Comparator verdict from `V8-7`

Comparator scores:

- No memory:
  - GSM8K: **0.8125**
  - TriviaQA: **0.0312**
- Text RAG:
  - GSM8K: **0.6406**
  - TriviaQA: **0.1250**
- Best V8 route:
  - GSM8K: **0.8594**
  - TriviaQA: **0.0156**

The governed conclusion was correct:

- `best_v8_beats_floor = false`
- `best_v8_beats_text_rag = false`
- `comparison_conclusion = comparators_do_not_support_v8_8`

The single-turn latent-memory line therefore did **not** clear the bar for mechanical continuation into `V8-8` and `V8-9`.

## 2.4 What remains unfinished from prior plans—and what to do with it

The project has unfinished work, but not all unfinished work deserves resurrection.

### Unfinished pieces worth **transplanting**

1. **High-dimensional `M_long`**
   - still useful as a persistent archive concept
   - should be repurposed for long-horizon agent memory, not reopened on single-turn GSM8K/TriviaQA sweeps

2. **Reader/query compression**
   - still scientifically valuable
   - should be used to compress retrieved persistent memory into a short actionable representation for the current turn/session
   - this is the most natural continuation of your C2 idea

3. **CDMI closeout (`V8-9`)**
   - still important
   - but the original qwen34 `V8-9` design (`GSM8K + TriviaQA`) is no longer the right target
   - the CDMI machinery should be transplanted to **agent domains and cross-session persistence**

4. **Governed execution infrastructure**
   - summary builders
   - milestone runners
   - review publication
   - queue helpers
   - unit tests
   - these are valuable and must be preserved

### Unfinished pieces that should **not** be reopened as-is

1. Broad reader-only rescue sweeps on single-turn tasks
2. New OPD variants on the same qwen34 single-turn route
3. New sequence-injection or cross-attention consumer sweeps for frozen single-turn reasoning
4. Large new GSM8K/TriviaQA/FEVER screening matrices as the mainline

---

## 3. One non-negotiable correction before any new work starts

There are four corrections that PLANv9 treats as binding.

### 3.1 Do not call FlashMem “already safe” on this stack

It is true that FlashMem’s **memory construction** is much closer to the backbone manifold than the old external Writer route: its Shared-KV consolidator attends directly to the backbone’s frozen KV cache and synthesizes latent memory from those states.

But its **memory consumption** is **not** KV-only.

Its own paper describes:

- synthesized memory being **soft-injected back into the backbone input stream**
- the continuous embeddings being fed through the backbone to generate their KV pairs
- those KV pairs then being appended to the active cache
- the backbone subsequently attending back to the injected latent positions during generation

So FlashMem does **not** automatically bypass the entire V8-1 failure surface. It is promising, but not exempt from replication/falsification.

### 3.2 Do not reopen single-turn latent-memory screening as the mainline

Single-turn tasks now have a new role:

- **sanity gate**
- **mechanism discriminator**
- **catastrophic-regression detector**

They are no longer the main proving ground for the research thesis.

### 3.3 Do not claim the central blocker is still “Writer bandwidth” alone

That was a reasonable focus during parts of V6 and V7, but V8 changed the diagnosis.

The main blocker is now better stated as:

> the dominant failure on a capable frozen backbone is not only poor memory content generation, but the inability to introduce new, highly attended latent positions without destabilizing the model’s existing reasoning circuit.

### 3.4 Do not let the comparator hold be misread as “nothing worked”

Something did work scientifically:

- the repo built a real governed qwen34 route
- the qwen34 route established a sharp mechanistic contrast between safe-flat prefix and destructive attended latent positions
- the repo now has smoke-level support for several long-horizon benchmark families

The correct response is **not** “give up.”  
The correct response is **change the proving ground and resolve the remaining architectural discrepancy with the cheapest decisive test first.**

---

## 4. Independent deep diagnosis: what is actually going on

This section is the independent bottom-up diagnosis, not merely a restatement of prior summaries.

## 4.1 The project now spans three distinct memory regimes

The evidence across V6–V8 reveals three regimes, not one.

### Regime A — weak backbone, zero-capability regime

Observed on qwen25-era GSM8K:

- the base model could not solve the task
- injected memory did not create capability
- therefore all upstream memory engineering was bottlenecked by backbone competence

Interpretation:

> memory cannot teach a frozen model a skill it fundamentally does not have.

### Regime B — capable backbone, ignored-safe prefix regime

Observed on qwen34 legacy prefix:

- the model remains close to baseline
- prefix attention mass is tiny
- task scores remain flat or nearly flat
- the route is safe because it is weakly load-bearing

Interpretation:

> the memory route behaves more like a tiny bias term than a genuine information-bearing memory channel.

This is exactly the type of regime Petrov/Torr/Bibi predict for context-based tuning when the mechanism can only bias outputs in limited directions without altering relative attention structure.

### Regime C — capable backbone, destructive-attended-memory regime

Observed on qwen34 sequence and cross-attention routes:

- the model attends to the injected route
- logits move strongly
- task scores collapse

Interpretation:

> the injected route is now strong enough to matter, but not aligned enough to help.  
> It perturbs an already competent reasoning circuit more than it supplies useful task information.

This is the most important mechanistic insight in the current project.

## 4.2 Why legacy prefix is “not bad, not useful”

The legacy prefix route on qwen34 is **not** helping because the backbone barely grants it meaningful causal responsibility.

That is consistent with two lines of explanation:

1. **Petrov/Torr/Bibi limitation**  
   Prefix-like context methods cannot freely rewrite relative attention patterns; they bias outputs within a constrained representational envelope.

2. **PrefixMemory-Tuning tradeoff**  
   Standard prefix-tuning on modern LLMs faces an inherent tradeoff between prefix significance and input significance inside the attention head.  
   If the prefix matters too much, it can overwhelm input specificity.  
   If it matters too little, it becomes ineffectual.

Your qwen34 legacy prefix route is living on the “too little significance” side of that tradeoff.

This is why it is safe-flat instead of destructive.

## 4.3 Why sequence and cross-attention injection are catastrophic

The usual shallow explanation is “out-of-distribution memory tokens.” That explanation is incomplete.

The stronger diagnosis is:

1. **The qwen34 backbone already has a functioning reasoning circuit on GSM8K.**
2. **Any newly introduced attended positions become high-leverage control points inside that circuit.**
3. **If those positions are not perfectly aligned to the model’s internal reasoning geometry, they do not merely fail to help—they distort the trajectory.**

This is why:

- activation becomes high
- answer log-prob shifts become large
- task scores get worse, not merely unchanged

In other words:

> once the model is competent, the main risk is no longer “memory ignored.”  
> The main risk becomes **memory mis-steering**.

## 4.4 Why OPD failed to fire

The V8-3 OPD failure is not just an implementation footnote.

It tells you something about signal geometry.

On primary tasks, the positive-token fraction stayed at or near zero. That means the hint-enhanced teacher did not systematically make the desirable answer tokens more probable than the student.

Why?

### Reason 1: on GSM8K, the student was already competent

With qwen34 at ~81% GSM8K on the chosen split, many examples already have strong token trajectories under the base model. The hint then changes little, leaving weak or zero advantage signal.

### Reason 2: on TriviaQA, the memory route still did not encode usable factual information

So even if the task is more “memory-shaped,” the student/teacher discrepancy still does not isolate a strong consumption signal if the memory channel itself is not informative.

### Reason 3: OPD is downstream of the consumer interface

If the consumer interface is either ignored (legacy prefix) or destructive (sequence / cross-attn), improving the token-level supervision signal does not automatically fix the route.

So the correct demotion is:

> OPD is not the first-line fix anymore.  
> The consumer mechanism must first be proven non-destructive and potentially useful in the correct task regime.

## 4.5 Why text RAG beat latent memory on TriviaQA

This is a critical clue, not an embarrassment.

Text RAG helped on the task where external information matters because:

- the evidence remained in natural-language form
- the pretrained model already knows how to consume text
- retrieval matched the true information need of the task

By contrast, the latent memory route:

- did not demonstrably encode the relevant facts
- did not present them in the model’s most natural consumption format
- and sometimes distorted the reasoning path instead of informing it

This means that on single-turn factual recall, the latent route was trying to solve the wrong problem.

## 4.6 Why long-horizon agent tasks are the correct next regime

Long-horizon agent tasks differ from GSM8K/TriviaQA/FEVER in three crucial ways:

1. **The memory need is endogenous.**  
   The model needs to remember its own prior tool actions, retrieved evidence, constraints, user preferences, intermediate variables, and file states.

2. **The baseline memory gap is real even for strong models.**  
   LongMemEval, LoCoMo, and MemoryAgentBench all exist because even capable models still lose important information over extended interaction histories.

3. **Compression and persistence become the actual research problem.**  
   The question becomes:
   - what to keep,
   - when to consolidate,
   - how to retrieve,
   - how to avoid interference,
   - how to remain non-destructive,
   not “can latent memory beat plain text evidence on a single standalone question?”

That is why your first hard request in this message is correct: single-turn tasks should be **downshifted**, not remain the center of gravity.

---


## 4.7 Code-grounded implementation diagnosis

The diagnosis above is not just benchmark-level; it is supported by the current code.

### Finding A — sequence memory is literally prepended into the attended input stream

In `src/memtotal/models/backbone.py`, `_prepare_prefixed_hf_inputs(...)` handles `memory_tokens` by normalizing them, matching their norm to content embeddings, and then **concatenating them to `inputs_embeds` and `attention_mask`** before the real prompt tokens. That means the `reader_lora_sequence` route is not an abstract side channel; it creates genuine attended positions in the causal sequence. V8-1's catastrophic regressions should therefore be interpreted as failures of a real in-stream intervention, not a logging artifact.

### Finding B — deep prefix is generated as per-layer KV cache entries, with zero templates at non-target layers

In the same file, `_prepare_deep_prefixed_hf_inputs(...)` expands the layer-prefix inputs into a `DynamicCache`. For layers that are not explicitly targeted, it uses a `zero_hidden_template`, runs it through `input_layernorm`, `k_proj`, `v_proj`, and RoPE, and still populates prefix cache entries. This means the legacy prefix route is not “absence of prefix”; it is “prefix everywhere, meaningful on target layers, near-zero elsewhere.” That helps explain why the route can remain non-destructive yet weakly load-bearing.

### Finding C — cross-attention is a real residual intervention, not a read-only probe

`ReaderCrossAttentionAdapter` in `backbone.py` computes queries from the current hidden states, keys/values from `memory_tokens`, and adds a **gated residual update** back into the hidden stream. V8-1's high activation and catastrophic score drops are therefore consistent with a direct hidden-state intervention: the adapter is not merely exposing memory for inspection; it is perturbing the backbone’s internal trajectory.

### Finding D — the old “C2 gate” is global balancing, not instance-conditional routing

`SupportContextBalanceGate` in `src/memtotal/models/memory.py` is a LayerNorm plus learned log-scale for context and support streams. It is useful for magnitude balancing, but it is not a per-sample or per-turn router deciding when to trust memory. This should continue to be interpreted as a calibration device, not evidence of dynamic memory utilization.

### Finding E — the old deep-prefix projector still contains a hard shared bottleneck

`WriterDeepPrefixProjector` in `memory.py` applies a shared `down_proj(hidden_size -> bottleneck_rank)` and `up_proj(bottleneck_rank -> num_layers * hidden_size)` with default `bottleneck_rank = 32`. That architecture can easily manufacture geometric diversity from a narrow source while still constraining semantic throughput. The repo already contains `PerLayerLowRankDeepPrefixProjector`, which remains useful as a transplantable idea, but the current single-turn line never established that these projectors create task-meaningful memory on the frozen qwen34 backbone.

### Finding F — the repo already has the right task hooks for a benchmark pivot

`src/memtotal/tasks/registry.py` already registers `alfworld` and `memoryagentbench`, and the docs explicitly state that these are currently smoke-level or truncated versions rather than formal long-context reproductions. PLANv9 should therefore **promote** these scaffolds instead of starting from zero.


## 5. Architecture cross-examination: what survives scrutiny and what does not

## 5.1 What survives from the current MemTOTAL codebase

The repo already contains several assets that should absolutely be kept.

### Surviving asset A — governed research infrastructure

Keep:

- milestone config builders
- controlled runners
- summary updaters
- review publication
- queue helpers
- unit-test style governance

This is a strategic advantage. PLANv9 should **reuse** it, not replace it.

### Surviving asset B — long-memory concepts from `MAIN_IDEA.md`

The repo’s locked conceptual contributions still make sense:

- **C1:** hierarchical memory (`write/read/compress`)
- **C2:** adaptation concentrated in read/query/compress logic
- **C3:** explicit CDMI measurement instead of hand-waving about transfer

But these ideas need a better regime than single-turn QA.

### Surviving asset C — task scaffolds for agent-like or long-context work

The repo already has smoke-level task registration for:

- `alfworld`
- `memoryagentbench`
- `narrativeqa`
- `kodcode`
- long-context / story tasks

The important caveat is that current support is still partly smoke/proxy, not yet formal benchmark reproduction. That is good news, not bad news: the foundation is present, and PLANv9 can promote it.

### Surviving asset D — old bridge and memory metrics

Keep these diagnostics:

- memory effective rank
- common-mode energy
- activation scores
- attention mass diagnostics
- route-live indicators
- interference / leakage metrics

But apply them to the new architecture where appropriate.

## 5.2 What does **not** survive as a mainline design

### Retired as mainline

1. External Writer → projection → in-sequence prepend for frozen single-turn reasoning
2. Reader cross-attention rescue for the same regime
3. Broad OPD reopening before consumer safety is established
4. Mechanical continuation of V8-8 and V8-9 in their original single-turn form

### Downgraded to sidecars only

1. Further qwen34 prefix-only tuning sweeps
2. Reproduction of broad MemGen-style RL training on a single GPU before simpler discrepancy tests are resolved
3. Qwen3-8B as the immediate first move

## 5.3 What is still uncertain and therefore must be re-validated

There is one unresolved architectural question that dominates all others:

> Is FlashMem-style soft-append/backbone-preprocessed latent memory genuinely different enough from your failed V8-1 in-sequence route to justify further investment?

This is **the** Phase-0 question of PLANv9.

---

## 6. Core strategic decision of PLANv9

The central strategic decision is:

> **Resolve the FlashMem discrepancy first with the smallest possible controlled test.**  
> If the route is at least non-destructive, pivot the mainline to long-horizon agent memory.  
> If the route collapses, do not keep spending on latent in-stream consumers for frozen single-turn reasoning; pivot to safer consumers and to long-horizon external/structured memory baselines, while preserving the negative-result publication track.

This gives PLANv9 two linked branches:

### Mainline scientific branch

1. **V9-0:** FlashMem discrepancy discrimination
2. **V9-1:** long-horizon benchmark hardening
3. **V9-2+:** Shared-KV consolidator on within-session agent memory
4. **V9-4+:** cross-session persistence and CDMI

### Parallel publication branch

- package V6–V8 as a rigorous negative result on latent memory injection for frozen single-turn reasoning/factual recall

Both branches should run in parallel, but only the mainline branch should consume engineering budget on new architecture.

---

## 7. The new target architecture: Shared-KV Persistent Memory with Safe Consumption

For clarity, PLANv9 names the new architecture family:

# **SKPM-SC**  
**S**hared-**K**V **P**ersistent **M**emory with **S**afe **C**onsumption

This is not a marketing rename. It encodes the actual design constraints learned from V8.

## 7.1 Design principle 1 — Memory construction should occur in the backbone’s own state manifold

The old external Writer route failed in part because it asked a detached module to invent useful memory representations and then forced the frozen backbone to consume them.

The new default is:

- use the backbone’s own trajectory states and/or KV cache as the **raw material**
- let a lightweight Shared-KV consolidator synthesize event/session memory from those states
- keep the backbone frozen unless a small, carefully isolated read/query module is explicitly being trained

This addresses the strongest remaining part of the “integrated Writer” idea from Path Q.

## 7.2 Design principle 2 — Memory consumption must be treated as a risk surface, not an afterthought

Because V8 showed that newly attended latent positions can be destructive, PLANv9 treats the consumer as a first-class object with multiple controlled variants.

### Consumer families allowed in PLANv9

#### C0 — Safe legacy cache-prefix control
- purpose: continuity, non-destructive reference
- expectation: safe but probably flat

#### C1 — Flash-style soft-append-to-cache
- latent embeddings are fed through the backbone to produce KV pairs
- closest to FlashMem consumption
- **must** pass V9-0 before becoming mainline

#### C2 — Decoupled-prefix sidecar (PrefixMemory-Tuning-inspired)
- prefix information is moved out of the standard attention-head competition
- intended as the lowest-risk orthogonal attempt to break safe-flat prefix behavior
- used only if C1 is flat or harmful and C0 remains the only safe route

### Consumer families retired from mainline use

#### X1 — sequence prepend of projected memory tokens
#### X2 — general reader cross-attention as primary consumer for frozen single-turn tasks

These are now retirement-class routes for the current problem setting.

## 7.3 Design principle 3 — Persistence must be hierarchical

The old `M_long -> M_short` idea is not wrong. It was just used in the wrong place.

In PLANv9, the hierarchy becomes:

- **event memory** (`M_event_long`)
- **session memory** (`M_session_long`)
- **retrieved active memory** (`M_active_long`)
- **compressed active memory** (`M_active_short`)

That is the correct place to reuse the bridge/query compression logic.

## 7.4 Design principle 4 — Persistent memory must be dual-format, not latent-only

Every persistent memory item should have:

1. a **latent representation** (for model-side retrieval/compression)
2. a **textual / structured audit record** (for debugging, fallback retrieval, ablation, and contamination analysis)

This is extremely important.

It gives you:

- inspectability
- safer debugging
- a clean text-memory baseline
- easier CDMI analysis
- a direct bridge to structured-memory baselines if the latent route underperforms

## 7.5 Design principle 5 — Adaptation should live in the read/compress/retrieve logic, not in a detached “Writer”

This is the best continuation of your original C2 intuition.

Under PLANv9:

- “Writer” is no longer an external encoder by default
- the **consolidator** becomes the integrated writer-like memory constructor
- the **Reader/Bridge** becomes the main adaptive query/compress module
- the backbone remains frozen unless a tiny carefully bounded sidecar is justified

---

## 8. Exact architecture proposal for SKPM-SC

## 8.1 Module list

### Module A — Frozen backbone

Default continuity backbone for V9 mainline:

- `Qwen3-4B-Instruct` (continuity with V8, manageable on 96 GB)
- bf16 preferred
- gradient checkpointing on whenever training additional modules

Optional later confirmation backbone:

- `Qwen3-8B-Instruct`
- **not** first-line in V9; only opened after V9 demonstrates a real long-horizon route on qwen34

### Module B — Event boundary / uncertainty monitor

Default trigger policy is **hybrid**, not pure entropy:

An event should be eligible for consolidation when **any** of the following fire:

1. tool call finished
2. file edit / patch proposal finished
3. session ended
4. retrieval step finished
5. uncertainty spike:
   - mean token entropy over the last `W=8` generated tokens exceeds threshold `τ`
   - threshold policy:
     - pilot default: rolling mean + `1.0` std
     - alternate pilot: fixed percentile `p90`

Why hybrid and not entropy-only?

Because agent tasks contain many memory-worthy states that are not high-entropy:
- final tool outputs
- discovered file paths
- user preference updates
- error messages
- accepted constraints

### Module C — Shared-KV consolidator

Default pilot consolidator:

- 1 decoder-style cross-attention block
- input query bank:
  - `Q = 8` learnable queries for scout
  - `Q = 16` as promoted wider arm
- source:
  - selected backbone KV slices from `mid4` or `mid8`
- output:
  - `M_event_long ∈ R^(Q × H)` where `H = backbone_hidden_dim`

Default training surface:
- consolidator parameters only
- optional tiny read/query bridge parameters
- backbone frozen

### Module D — Memory archive

Each persistent item stores:

```text
memory_id
session_id
turn_start
turn_end
timestamp
domain_tag
task_tag
tool_tag
file_paths (optional)
latent_tensor_path
text_note
retrieval_key_text
retrieval_key_latent (optional)
evidence_span_refs (optional)
confidence / trigger score
```

Storage contract:

- latent tensors: `safetensors`
- metadata index: `jsonl` or SQLite-backed index if volume grows
- one archive file per session plus global manifest

### Module E — Query reader / bridge

This is where older MemTOTAL ideas return.

Responsibilities:

1. retrieve top-k event/session memories
2. compress them into a short actionable representation
3. keep domain adaptation concentrated in read/query logic

Default bridge arms:

- `B0`: no compression, direct top-k event/session use
- `B1`: learned query compression to `4` active tokens
- `B2`: learned query compression to `8` active tokens
- `B3`: two-stage compression:
  - top-k event memories -> session summary
  - session summary -> active short memory

### Module F — Consumer

Chosen by V9-0 gate and later agent pilots:

- `C0 safe legacy cache-prefix`
- `C1 Flash-style soft-append-to-cache`
- `C2 decoupled-prefix sidecar`

### Module G — Audit and safety layer

Every run should emit:

- latent norm statistics
- memory attention mass
- no-memory KL divergence on short-context sanity turns
- trigger count
- retrieval hit rate
- stale-memory rate
- contradiction / contamination rate

This is not optional.

---

## 9. Training objectives: what to optimize, and what not to optimize

## 9.1 Primary objective in PLANv9

The main objective is **task success on long-horizon agent memory tasks**, not single-turn score squeezing.

That means:

- tool success
- cross-session recall accuracy
- code-repo dependency resolution
- long-range understanding
- knowledge updates
- temporal reasoning
- selective forgetting

## 9.2 Default loss recipe for early V9 training

For within-session agent pilots, use:

### `L_total = L_task + λ_distill L_teacher + λ_safe L_nondestructive + λ_budget L_trigger_budget + λ_div L_query_div`

Where:

#### `L_task`
- standard supervised CE / imitation loss on target outputs
- for tool tasks: action token loss / successful trajectory imitation
- for QA/chat tasks: answer token loss

#### `L_teacher`
- optional full-history or richer-context teacher distillation
- only used when a stronger teacher is available without changing the main question
- preferred over OPD in early V9

#### `L_nondestructive`
- KL or logit distance to the no-memory baseline **on turns where long-term memory should not matter**
- purpose: punish the route for needlessly perturbing local reasoning

This is a PLANv9-specific safeguard directly motivated by V8.

#### `L_trigger_budget`
- regularize the number of consolidation events per episode/session
- prevent memory spam
- default target:
  - scout: average ≤ 1 consolidation per 32 generated tokens
  - agent tasks: average ≤ 1 consolidation per event boundary, plus uncertainty-triggered exceptions

#### `L_query_div`
- diversity penalty on query bank outputs
- prevents all queries from collapsing to one event or one slot

## 9.3 What is explicitly demoted

### OPD is demoted, not banned

OPD should **not** be the first-line optimization signal in V9.

It may return later, but only after:

1. a non-destructive consumer is proven,
2. a long-horizon task route is alive,
3. teacher/student discrepancy on memory-relevant turns is nontrivial.

### Broad external-Writer auxiliary sweeps are demoted

Earlier auxiliary work is not deleted from history, but it is no longer the first place to invest.

## 9.4 Preferred teacher signals in V9

Use the following, in order of priority:

1. **full-history teacher**  
   Student sees compressed memory, teacher sees full history.

2. **successful trajectory imitation**  
   Especially for ALFWorld / tool-use tasks.

3. **retrieval-oracle teacher**  
   For LongMemEval-style tasks, teacher gets oracle relevant sessions or textual notes.

4. **policy-level retrieval fine-tuning**  
   Only after the route is alive.

---

## 10. Benchmark strategy: single-turn becomes a gate, long-horizon becomes the mainline

## 10.1 Single-turn tasks in PLANv9

Single-turn tasks are now:

- mechanism discrimination tools
- sanity checks
- catastrophic-regression detectors

They are **not** the main selection target.

### Allowed single-turn uses

1. `GSM8K 64-sample split` for V9-0 discrimination
2. minimal `TriviaQA` and `FEVER` sanity checks after a new consumer is introduced
3. regression guardrails for catastrophic collapse

### Forbidden single-turn use

Do **not** reopen large architecture sweeps on:

- GSM8K
- TriviaQA
- FEVER

as the mainline justification for the research direction.

## 10.2 Primary long-horizon benchmark families in PLANv9

PLANv9 prioritizes the following benchmark families, in this order.

### Family A — MemoryAgentBench

Reason:

- directly built for memory agents
- four competencies:
  - accurate retrieval
  - test-time learning
  - long-range understanding
  - selective forgetting
- especially appropriate because the repo already has a smoke-level integration scaffold

Role in V9:

- first promoted multi-turn benchmark after V9-0
- best near-term “bridge” between current repo capability and real long-horizon evaluation

### Family B — LongMemEval

Reason:

- explicitly decomposes long-term memory into indexing / retrieval / reading
- multi-session long-term chat memory
- aligned with cross-session persistence and C1/C3

Role in V9:

- primary benchmark for cross-session persistent memory
- especially useful for memory organization and retrieval ablations

### Family C — ALFWorld

Reason:

- already smoke-scaffolded in repo
- multi-step tool-like interaction
- useful to test whether memory improves state tracking over action sequences

Role in V9:

- first tool-use benchmark
- cheaper than full web-agent stacks

### Family D — LongBench v2 (code-repository understanding subset)

Reason:

- realistic long-context tasks
- includes code repository understanding
- good bridge from static long context to cross-file code work

Role in V9:

- first formal code-oriented long-context evaluation
- can be run before a custom cross-file patch-generation benchmark is mature

### Family E — LoCoMo

Reason:

- very long-term multi-session dialogue benchmark
- useful for strong cross-session persistence validation

Role in V9:

- secondary / confirmation benchmark after LongMemEval pipeline is stable

## 10.3 Secondary or optional benchmarks

### WebShop
- optional multi-step web interaction benchmark
- open only after ALFWorld or MemoryAgentBench route is alive

### Custom cross-file code benchmark
- build only after LongBench v2 code subset and repo-level code understanding pilots are stable

## 10.4 Benchmark priority matrix

| Benchmark | Memory need | Repo readiness | Cost | V9 role |
|---|---|---:|---:|---|
| GSM8K 64-sample | low / mechanism only | high | low | V9-0 discriminator |
| MemoryAgentBench | high | medium (smoke exists) | medium | first promoted agent-memory benchmark |
| ALFWorld | medium-high | medium (smoke exists) | medium | tool-use pilot |
| LongMemEval | high | low-medium | medium | primary cross-session benchmark |
| LongBench v2 (code subset) | medium-high | low | medium | code long-context pilot |
| LoCoMo | high | low | medium-high | later validation |
| WebShop | medium-high | low | high | optional |

---

## 11. PLANv9 execution phases

The phases are sequential in authority, but some writing / baseline preparation can happen in parallel.

---

## Phase V9-0 — The FlashMem discrepancy discrimination test

## Purpose

Resolve the single most important unresolved contradiction in the project:

> FlashMem reports a gain on Qwen 3 4B via an in-stream latent-memory route, while your V8 sequence oracle on the same backbone family shows catastrophic collapse.

This must be resolved **before** serious investment in FlashMem-style engineering or in large benchmark pivots.

## Why this phase is mandatory

Because it is the cheapest decisive experiment left in the project:

- 4 arms
- 1 task
- no broad sweep
- reuses the exact qwen34 stack and split family already in repo
- directly determines whether FlashMem-style consumption is plausible on your system

## Exact experiment: four-arm 2×2 discrimination test

### Dataset / split

- `GSM8K`
- use the **existing V8 64-sample qwen34 eval split**
- do not change prompt style for this phase
- use the same qwen34 baseline prompt mode used in V8

### Backbone

- `Qwen3-4B-Instruct`
- same loading / decoding settings as V8-0 continuity runs

### Arms

#### Arm A0 — no memory control
- identical to current V8 qwen34 control
- expected ≈ **0.8125**

#### Arm A1 — legacy prefix oracle
- same as `o1_q34_prefix_oracle_mid4_gsm8k`
- backbone hidden states injected as deep-prefix cache entries
- no in-sequence latent positions
- expected ≈ **0.7812**

#### Arm A2 — FlashMem-style soft-append, untrained
- take backbone layer-16 hidden states from the support/history material
- contiguous chunk-pool into `8` vectors of size `H`
- **do not** project through a learned memory→embedding adapter
- normalize RMS to match input embedding scale
- feed these `8` continuous vectors through the backbone as `inputs_embeds` in a dedicated memory pre-pass
- obtain their KV pairs from the backbone forward
- append those KV pairs to the active cache
- then run the actual task prompt against the extended cache

This is the closest cheap approximation of the FlashMem consumption route that resolves Hypotheses A/B from text **【5】**.

#### Arm A3 — V8 sequence oracle replay
- same hidden-state source and chunking as A2
- use the V8-1 sequence route
- prepend as sequence memory positions in the model input
- expected ≈ **0.2656**

## Required instrumentation

Log all of the following:

- task accuracy
- answer log-prob delta vs A0
- generation length
- invalid / malformed answer rate
- latent token norm statistics
- prefix or memory attention mass
- number of attention layers with nontrivial attention to latent positions
- cache growth
- decoding latency overhead

## Implementation notes

### Code changes

Add a new consumer path to `backbone.py`:

- name suggestion: `memory_consumer_mode = "precache_latent"`
- behavior:
  1. accept continuous latent embeddings `[B, M, H]`
  2. run a memory-only forward pass with `inputs_embeds`
  3. collect produced KV cache
  4. append to active cache before task generation

Do **not** reuse the old sequence prepend path for A2.

### Deliverables

- `scripts/planv9_v9_0_config.py`
- `scripts/run_planv9_v9_0_flashmem_discrimination_qwen34.sh`
- `scripts/update_planv9_v9_0_summary.py`
- focused unit tests:
  - config build
  - cache append correctness
  - summary logic
- review artifact namespace:
  - `planv9-v9-0-flashmem-discrimination-qwen34`

## Acceptance logic

Define the following interpretation rules.

### Outcome O0 — A2 ≈ A1, both near baseline
Meaning:
- FlashMem-style backbone-preprocessed latent memory is **non-destructive**
- but untrained memory is still not useful
- proceed to trained Shared-KV consolidator on long-horizon tasks

### Outcome O1 — A2 > A0
Meaning:
- first true positive sign for this route
- immediately justify trained Flash-style branch
- promote Flash-style consumer to mainline consumer candidate

### Outcome O2 — A2 ≈ A3 (both collapse)
Meaning:
- in-stream latent-position failure is robust on your stack
- FlashMem-like route is **not** safe to assume viable
- do **not** invest in full FlashMem replication yet
- shift mainline consumer to safe-prefix or decoupled-prefix branch

### Outcome O3 — A2 between A1 and A3
Meaning:
- backbone pre-processing reduces but does not remove the destructive effect
- there is probably a two-part failure:
  - projection/manifold mismatch
  - added attended-position disruption
- proceed only with strong safety regularization and long-horizon agent tasks

## Hard rule

No LongMemEval/LoCoMo/large new agent-memory implementation should begin **before** V9-0 is complete and published.

Small benchmark-hardening prep work may happen in parallel, but no architectural commitment should be made until V9-0 resolves.

---

## Phase V9-0.5 — PrefixMemory-Tuning sidecar (conditional, low-cost)

## Purpose

If V9-0 shows:

- legacy prefix is safe-flat
- Flash-style consumption is unsafe or only partially safe

then the best orthogonal architectural sidecar is to test a decoupled-prefix consumer inspired by PrefixMemory-Tuning / Prefix-Tuning+.

## Why this is conditional

Because this is **not** the mainline before V9-0.  
It is the best low-cost fallback if:

- `C1 Flash-style soft-append-to-cache` does not clear the safety bar
- `C0 legacy prefix` remains the only non-destructive route

## Minimal scout

### Arms
- `P0`: legacy prefix continuity control
- `P1`: decoupled-prefix side module, same memory source, no train
- `P2`: decoupled-prefix side module, trainable read/query sidecar for 100 steps

### Task
- GSM8K 64-sample sanity + 32-sample TriviaQA sanity

### Promotion rule
Only promote if:
- no catastrophic regression on GSM8K
- at least weak positive delta on one long-horizon pilot later

This phase is optional, but it is the cleanest way to exploit the new prefix literature without reopening the old full prefix route.

---

## Phase V9-1 — Long-horizon benchmark hardening

## Purpose

Promote the repo’s existing smoke scaffolds into real pilot benchmarks usable for scientific selection.

This is where the “unfinished benchmark pivot” finally becomes real.

## Benchmarks to harden in this phase

### B1 — MemoryAgentBench pilot
Current repo state:
- smoke support exists
- real-source subsets exist
- context is still truncated for local stub convenience

Required upgrade:
- remove 512-token truncation for official-ish pilot runs
- preserve capability-wise scoring:
  - AR
  - TTL
  - LRU
  - CR

### B2 — ALFWorld pilot
Current repo state:
- text transition-style smoke exists

Required upgrade:
- define a fixed governed subset
- stable success-rate metric
- stable step-budget metric
- explicit action formatting contract

### B3 — LongMemEval pilot
Required build:
- materializer
- question-type balanced pilot split
- indexing / retrieval / reading aware reporting

### B4 — LongBench v2 code-repository pilot
Required build:
- code-repository subset materializer
- MCQ evaluator
- length-aware runtime budget

### B5 — LoCoMo pilot (optional in V9-1, mandatory by V9-4)
Required build:
- question-answer pilot
- summarization pilot
- session metadata handling

## Concrete work

### 1. Turn smoke subsets into governed pilot subsets

For each benchmark define:

- `support` / `train` / `eval` split or task-specific equivalent
- exact subset size
- seed
- maximum context budget
- latency budget
- official-ish metric if available
- fallback local proxy metric if not

### 2. Define minimal pilot sizes

Use the following pilot sizes to keep the first cycle tractable:

- MemoryAgentBench:
  - `25` episodes per competency
  - total `100` episodes
- ALFWorld:
  - `20` episodes × `6` task templates
  - total `120` episodes
- LongMemEval:
  - `20` questions × `5` abilities
  - total `100` questions
- LongBench v2 code subset:
  - `50` items
- LoCoMo:
  - `50` QA + `20` summarization items (deferred if needed)

### 3. Establish benchmark baselines

For every hardened benchmark, run at least:

- `B0`: no memory / short-window baseline
- `B1`: full-history baseline if context fits
- `B2`: text-summary memory baseline
- `B3`: text RAG memory baseline

Do **not** introduce latent memory before these baselines exist.

## Deliverables

- benchmark config builders
- benchmark runners
- benchmark summary updaters
- stable pilot subset manifests
- review artifact namespace:
  - `planv9-v9-1-longhorizon-baselines-qwen34`

## Acceptance

V9-1 passes only if:

1. all selected benchmarks run end-to-end with stable outputs
2. baseline metrics are reproducible
3. runtime budgets are documented
4. at least MemoryAgentBench + ALFWorld + LongMemEval are ready for model-side experiments

---

## Phase V9-2 — Within-session Shared-KV consolidator scout

## Purpose

Test whether memory constructed from the backbone’s own KV/history states can improve **within-session** agent-memory performance without destabilizing the backbone.

This is the first real mainline phase after the FlashMem discrepancy gate.

## Precondition

Open only if:

- V9-0 does **not** yield Outcome O2 (catastrophic robust collapse)
- V9-1 benchmark hardening is complete for:
  - MemoryAgentBench pilot
  - ALFWorld pilot
  - LongMemEval pilot

## Architecture under test

### Fixed
- frozen qwen34 backbone
- Shared-KV consolidator
- hybrid trigger policy
- dual-format archive logging (latent + text audit)

### Sweep axes

#### Source layer band
- `L0 = mid4`
- `L1 = mid8`

#### Query count
- `Q0 = 8`
- `Q1 = 16`

#### Trigger mode
- `T0 = event_only`
- `T1 = event_plus_entropy`

#### Consumer family
- `C0 = safe legacy cache-prefix`
- `C1 = Flash-style soft-append-to-cache` (only if V9-0 safe enough)
- `C2 = decoupled-prefix sidecar` (only if V9-0 recommends)

## Scout matrix

Maximum scout size for first pass: `8` arms, not a combinatorial explosion.

Recommended first-pass arms:

1. `mid4_q8_event_C0`
2. `mid4_q8_event+entropy_C0`
3. `mid8_q8_event_C0`
4. `mid8_q16_event+entropy_C0`
5. `mid4_q8_event_C1` (conditional)
6. `mid8_q8_event+entropy_C1` (conditional)
7. `mid4_q8_event_C2` (conditional)
8. `mid8_q16_event+entropy_C2` (conditional)

## Training recipe

### Default optimizer
- AdamW
- `lr = 1e-4` for consolidator/query modules
- `weight_decay = 0.01`
- `betas = (0.9, 0.95)`

### Stability settings
- grad clip `1.0`
- bf16
- gradient checkpointing on
- dropout `0.0` initially
- effective batch via gradient accumulation

### Step budget
- scout:
  - `200` train steps
- promotion confirmation:
  - `400` train steps

### Seeds
- scout: `61109`
- confirmation: `61109`, `61110`, `61111`

## Tasks

Primary scout tasks:
- MemoryAgentBench pilot
- ALFWorld pilot

Secondary observer task:
- LongMemEval pilot (can be eval-first if training cost is high)

## Metrics

### Primary
- benchmark score improvement vs no memory
- benchmark score improvement vs text-summary baseline
- benchmark score improvement vs text RAG baseline (where applicable)

### Mechanistic
- consolidations per episode
- retrieval hit precision
- memory attention mass
- no-memory KL on short-range turns
- archive growth rate

### Safety
- regression on short-range questions inside the same benchmark
- catastrophic local-answer corruption rate
- runtime / VRAM overhead

## Acceptance

Promote an arm only if all of the following hold:

1. improves at least one primary long-horizon benchmark over `B0`
2. does not catastrophically regress any other primary long-horizon benchmark
3. keeps `L_nondestructive` under control on short-range turns
4. does not increase average latency by more than `2.0×` relative to the baseline for the pilot subset

## Hard rule

Do not use GSM8K/TriviaQA aggregate improvement to select V9-2 winners.  
At most, use them as catastrophic-regression checks.

---

## Phase V9-3 — Consumer sweep and safe-consumption refinement

## Purpose

Once the within-session Shared-KV route is alive, determine the best consumer family under agent-memory conditions.

## Why this phase exists

V8 taught you that memory construction is not enough.  
Consumption is the risk surface.

So V9-3 explicitly focuses on the consumer after a live within-session route exists.

## Consumer families to compare

### C0 — safe legacy cache-prefix
- likely safest
- may remain flat

### C1 — Flash-style soft-append-to-cache
- likely strongest if it can be made safe
- only if V9-0/V9-2 indicate viability

### C2 — decoupled-prefix sidecar
- best orthogonal low-risk consumer if prefix remains safe-flat

## Additional safety regularizers to test

### S0 — none
### S1 — no-memory KL on short-range turns
### S2 — no-memory KL + consolidation budget penalty
### S3 — no-memory KL + consolidation budget + memory-use sparsity

## Scout grid

Limit first scout to `6` arms:

1. `C0_S1`
2. `C0_S2`
3. `C1_S1`
4. `C1_S2`
5. `C2_S1`
6. `C2_S2`

## Tasks

- MemoryAgentBench pilot
- LongMemEval pilot
- one of:
  - ALFWorld
  - LongBench v2 code subset

## Acceptance

Promote only if:
- one consumer beats all other consumers on at least one benchmark family
- it remains non-catastrophic on the rest
- its safety regularizer reduces destructive drift without erasing gains

---

## Phase V9-4 — Cross-session persistence v1 (C1 realization)

## Purpose

Turn the within-session memory route into a persistent memory system across sessions.

This is the real beginning of your C1 contribution in the proper task regime.

## Architecture additions

### Session summarizer
- collects event memories within a session
- compresses them to one or more session-memory items

### Cross-session index
- retrieves relevant sessions by:
  - time
  - domain
  - tool/file metadata
  - textual key expansion
  - optional latent key similarity

### Active-memory assembly
- selected sessions -> retrieved event/session memories -> compressed active memory for current turn

## Retrieval variants

### R0 — flat retrieval
- no session decomposition

### R1 — session decomposition
- retrieve at session granularity first, then event granularity

### R2 — session decomposition + time-aware query expansion
- inspired by LongMemEval

### R3 — session decomposition + time-aware query expansion + file/tool tag filtering
- recommended for code and tool tasks

## Tasks

Primary:
- LongMemEval
- LoCoMo

Secondary:
- custom repo-memory pilot or LongBench v2 code subset
- MemoryAgentBench multi-turn slices

## Metrics

- multi-session reasoning accuracy
- temporal reasoning accuracy
- knowledge-update consistency
- stale-memory rate
- wrong-session retrieval rate
- archive retrieval latency
- evidence citation precision (if available)

## Acceptance

A route is live only if it beats the no-persistence version on at least one of:

- LongMemEval total score
- LoCoMo QA score
- LoCoMo summarization score

without materially increasing contamination or stale recall.

---

## Phase V9-5 — Cross-file code memory and multi-step tool-use expansion

## Purpose

Satisfy the practical target you explicitly requested:
- multi-step tool use
- cross-file code generation / understanding
- extended dialogue

## Track A — cross-file code memory

### Stage A1 — standardized benchmark
- LongBench v2 code repository subset

### Stage A2 — repo-native pilot
Build a small governed benchmark, tentatively `RepoPatchBench-pilot`, with tasks that require:

1. inspecting one file
2. remembering a constraint from another file or earlier session
3. generating or selecting a patch / fix / code action

Recommended pilot size:
- `30` episodes
- `3` repos
- `10` tasks per repo

Metrics:
- exact answer / patch correctness where possible
- compile/test pass proxy where available
- dependency recall correctness
- wrong-file edit rate

## Track B — multi-step tool use

### Stage B1 — ALFWorld promoted subset
- success rate
- steps to success
- memory-hit rate

### Stage B2 — WebShop optional
Open only if ALFWorld route is alive and infrastructure budget remains.

## Track C — extended dialogue

### Stage C1 — LongMemEval promoted split
### Stage C2 — LoCoMo promoted split

## Acceptance

At least one track among A/B/C must show a real memory benefit for the route to continue.

Do **not** keep the entire long-horizon line alive solely on one benchmark family if the others strongly contradict it.

---

## Phase V9-6 — CDMI on agent domains (C3 realization)

## Purpose

Finish the unfinished C3 idea in the correct regime.

The original V8-9 CDMI closeout was specified for `GSM8K + TriviaQA`.  
PLANv9 redefines CDMI over **agent-memory domains**, where interference is actually meaningful.

## Domain pairs to test

### Pair 1 — dialogue vs code
- LongMemEval/LoCoMo ↔ LongBench v2 code / repo pilot

### Pair 2 — dialogue vs tool use
- LongMemEval/LoCoMo ↔ ALFWorld

### Pair 3 — code vs tool use
- code pilot ↔ ALFWorld

## Condition template per pair

For domain pair `A/B`, run:

- `self_A`
- `self_B`
- `joint_A_eval_A`
- `joint_B_eval_B`
- `A_memory_on_B_eval`
- `B_memory_on_A_eval`

## Metrics

- self vs joint delta
- foreign-memory contamination rate
- wrong-domain retrieval rate
- stale-memory carryover
- negative transfer rate
- retrieval precision@k by domain tag
- latent/text audit disagreement rate

## Acceptance

CDMI is acceptable only if:

- joint training does not erase the self-domain gains
- cross-domain foreign-memory contamination remains below a pre-registered threshold
- no benchmark family shows severe negative transfer while others improve

Suggested initial thresholds:
- contamination rate `< 5%`
- wrong-domain retrieval among top-4 `< 10%`
- joint-vs-self primary-score drop `< 3 absolute points` on any retained domain

---

## Phase V9-7 — Multi-seed confirmation and optional Qwen3-8B sidecar

## Purpose

Confirm only after a **real long-horizon route** exists.

This phase restores the spirit of `V8-8`, but in the correct regime.

## Confirmation policy

### Required
- three seeds:
  - `61109`
  - `61110`
  - `61111`
- `400` train steps or promoted-equivalent if earlier phases used more
- re-materialized configs, not copied checkpoints

### Optional scale sidecar
Open qwen3-8B only if:
1. qwen34 route is already positive on at least one long-horizon benchmark family
2. the route is non-catastrophic
3. compute budget remains acceptable

Do **not** use qwen3-8B as a blind first move.

## Acceptance

A route is confirmation-successful only if across three seeds:

- it improves at least one primary long-horizon benchmark over the best non-latent baseline
- it is non-regressive or minimally regressive on the other retained benchmark families
- safety metrics remain within threshold

---

## Phase V9-P — Parallel negative-result paper branch

## Purpose

Preserve publication value from V6–V8 regardless of what V9 ultimately proves.

## Thesis of the paper

A strong negative-result framing is now available:

> On frozen LLMs, latent memory injection for single-turn factual recall and reasoning faces a capability-utility dilemma:  
> weak backbones cannot use the memory at all, while capable backbones either ignore safe prefix-style memory or are corrupted by newly attended latent positions.

## Required paper backbone

### Hypothesis
Latent memory injection can improve frozen LLM performance on reasoning/factual recall tasks.

### Test
Oracle memory routes on both weak and capable backbones.

### Result
- qwen25: no meaningful utility ceiling
- qwen34:
  - prefix: safe-flat
  - sequence: catastrophic
  - cross-attn: catastrophic
  - best route does not beat text RAG on factual recall

### Mechanism
- prefix is too weak / ignored
- in-sequence latent positions perturb reasoning
- cross-attn side channels destabilize the frozen computation
- OPD does not rescue the route when the consumer is wrong

### Implication
Single-turn reasoning/factual tasks are the wrong proving ground for this class of latent memory on frozen LLMs.

## Why keep this paper branch live now

Because regardless of V9 outcome:

- V6–V8 remain scientifically valuable
- the paper can be written in parallel
- if V9 succeeds on agent tasks, the paper becomes even stronger:
  - “single-turn latent injection fails, but long-horizon agent memory can work under stricter architectural conditions”

---

## 12. Exact config defaults for PLANv9

This section exists to make the plan directly executable.

## 12.1 Global defaults

### Backbone
- continuity default: `Qwen3-4B-Instruct`
- optional confirmation backbone: `Qwen3-8B-Instruct`

### Precision / memory
- `bf16 = true`
- gradient checkpointing `= true`
- flash attention if already supported/stable in current environment
- tokenizer parallelism disabled in scripts

### Optimizer
- AdamW
- `betas = (0.9, 0.95)`
- `eps = 1e-8`
- `weight_decay = 0.01`

### LR defaults
- new consolidator / query / bridge params:
  - default `1e-4`
  - scout grid: `(5e-05, 0.0001, 0.0002)`
- safety sidecar / decoupled prefix params:
  - default `1e-4`
- do **not** carry over old single-turn projector LR as a silent default for this line

### Gradient clipping
- global norm clip `1.0`

### Batching
Because the effective budget is one RTX PRO 6000 96GB:

- training batch size:
  - `1` or `2` episodes / prompts per step
- gradient accumulation:
  - `8` to `16` as needed
- eval batch size:
  - keep small, prioritize determinism and logging correctness

### Seeds
- scout:
  - `61109`
- confirmation:
  - `61109`, `61110`, `61111`

## 12.2 Default memory sizes

### Event memory
- scout default:
  - `Q = 8`
- promoted wider arm:
  - `Q = 16`

### Active compressed memory
- `M_active_short = 4` or `8` tokens / slots

### Retrieval top-k
- first-stage session retrieval:
  - `k_session = 4`
- second-stage event retrieval:
  - `k_event = 8`

### Archive caps
- `max_events_per_session = 32` before compression/clustering
- `max_session_summaries_per_user = configurable`, default `128`

## 12.3 Trigger defaults

### Event boundaries
Always consolidate on:
- tool completion
- file patch completion
- session end

### Entropy trigger
- rolling window `W = 8`
- threshold:
  - pilot default `mean + 1.0 * std`
- cooldown:
  - no new entropy-triggered consolidation within the next `16` generated tokens unless event boundary fires

## 12.4 Safety defaults

### Non-destructive KL
Apply on:
- short-range questions/turns
- tool steps where all necessary state is already within active local context

### Budget penalty
Default:
- target ≤ `1` uncertainty-triggered consolidation per `32` generated tokens
- event-boundary consolidations exempt

### Abort conditions
Abort or flag an arm immediately if:

- GSM8K sanity check drops by `> 10` absolute points vs continuity baseline
- LongMemEval / MemoryAgentBench pilot drops by `> 10` absolute points vs no-memory baseline after `50` training steps
- malformed output rate exceeds `2×` baseline
- latency exceeds `3×` baseline during scout

---

## 13. Deliverables by phase

## 13.1 V9-0 deliverables

- `scripts/planv9_v9_0_config.py`
- `scripts/run_planv9_v9_0_flashmem_discrimination_qwen34.sh`
- `scripts/update_planv9_v9_0_summary.py`
- `tests/test_planv9_v9_0_config.py`
- `tests/test_planv9_v9_0_summary.py`
- review namespace:
  - `planv9-v9-0-flashmem-discrimination-qwen34`

## 13.2 V9-1 deliverables

- benchmark config builders / materializers
- pilot subset manifests
- benchmark summary logic
- review namespace:
  - `planv9-v9-1-longhorizon-baselines-qwen34`

## 13.3 V9-2 deliverables

- Shared-KV consolidator module
- trigger policy module
- within-session training runner
- summary logic for memory-use/safety metrics
- review namespace:
  - `planv9-v9-2-withinsession-sharedkv-qwen34`

## 13.4 V9-3 deliverables

- consumer-ablation runner
- safety-regularizer logic
- review namespace:
  - `planv9-v9-3-consumer-safety-qwen34`

## 13.5 V9-4 deliverables

- persistent archive index
- retrieval pipeline
- cross-session runner
- review namespace:
  - `planv9-v9-4-persistence-qwen34`

## 13.6 V9-6 deliverables

- pairwise CDMI config builder
- CDMI summary logic
- review namespace:
  - `planv9-v9-6-agent-cdmi-qwen34`

## 13.7 V9-7 deliverables

- multiseed confirmation runner
- optional qwen3-8B wrapper
- review namespace:
  - `planv9-v9-7-confirmation-qwen34`
  - optional `planv9-v9-7-confirmation-qwen38`

---

## 14. What earlier-plan components are explicitly reused

## 14.1 From PLANv7 / Path Q

### Reused
- stronger integrated memory-construction idea
- true high-dimensional `M_long`
- bridge as a real compression object, not mere scaffolding

### Transformed
- external Writer becomes integrated Shared-KV consolidator
- `M_long` becomes event/session archive memory

## 14.2 From PLANv8

### Reused
- qwen34 continuity backbone
- comparator discipline
- consumer diagnostics
- OPD conclusion as a demotion signal
- multiseed confirmation logic (`V8-8` spirit)
- CDMI closeout spirit (`V8-9` spirit)

### Rejected as-is
- reopening V8-8/V8-9 in single-turn form
- more reader-only rescue on legacy prefix for the same regime

## 14.3 From the original MemTOTAL idea

### Preserved
- two-level memory
- read-long / read-short distinction
- explicit interference testing

### Corrected
- the proving ground is now long-horizon agent memory, not single-turn factual QA

---

## 15. Hard research rules for future agents working under PLANv9

1. **Do not skip V9-0.**  
   The FlashMem discrepancy test is mandatory.

2. **Do not treat FlashMem as already validated on this stack.**  
   Its published mechanism remains close enough to the V8 failure surface that replication/falsification is mandatory.

3. **Do not reopen broad single-turn sweeps.**  
   Single-turn tasks are guardrails, not the mainline.

4. **Do not continue a route just because it is “not dead.”**  
   Path Q on qwen25 was informative, not sufficient.

5. **Do not spend major compute on qwen3-8B before a qwen34 long-horizon route is alive.**

6. **Do not allow latent-only persistent memory without an inspectable textual/structured audit trail.**

7. **Do not run CDMI only on math vs trivia anymore.**  
   Use agent domains.

8. **Do not interpret positive activation alone as success.**  
   V8 already proved high activation can coexist with catastrophic degradation.

9. **Do not interpret FEVER as decisive for this line.**  
   FEVER can remain a calibration metric, not an architectural governor.

10. **Do not let negative-result writing block new experiments, and do not let new experiments delay negative-result writing indefinitely.**

---

## 16. Minimal immediate action list (the first 7 days)

This section is intentionally operational.

## Day 1–2: close the contradiction

### Run V9-0
- implement `precache_latent`
- execute A0/A1/A2/A3
- publish `v9-0-summary.json` and `v9-0-summary.md`

### In parallel
- finish reading PrefixMemory-Tuning paper carefully
- start drafting the negative-result paper skeleton from V6–V8

## Day 3–4: prepare the real proving ground

If V9-0 does **not** hard-fail A2:

- promote MemoryAgentBench pilot from smoke to governed pilot
- promote ALFWorld pilot
- build LongMemEval pilot subset/materializer

If V9-0 **does** hard-fail A2:

- keep the same benchmark-hardening work
- but switch the mainline consumer candidate to `C0/C2` instead of `C1`

## Day 5–7: open V9-2 scout

- launch within-session Shared-KV consolidator scout
- benchmark against:
  - no memory
  - text summary memory
  - text RAG memory

No more than `8` scout arms.

---

## 17. Final recommendation in one paragraph

The project should **not** mechanically continue the old single-turn latent-memory line, and it should **not** blindly assume FlashMem solves the V8 failure. The correct next move is to run the exact four-arm discrimination test from text **【5】** on the current qwen34 stack, because that is the cheapest decisive experiment left. Then, assuming the route is at least non-destructive, the mainline should pivot to long-horizon agent-memory benchmarks—starting with MemoryAgentBench, ALFWorld, and LongMemEval—using a Shared-KV consolidator for in-manifold memory construction, a guarded consumer family (`Flash-style` only if earned, otherwise safe/decoupled prefix), and a hierarchical persistent memory archive that finally realizes your original C1/C3 goals in the regime where they make sense.

---

## 18. References and external reading list

The following references matter directly for PLANv9 design and interpretation.

1. **MemTOTAL repository README (current public state).**  
   GitHub raw README.  
   URL: `https://raw.githubusercontent.com/YJLi-new/MemTOTAL/main/README.md`

2. **Petrov, A., Torr, P., & Bibi, A. (2024).**  
   *When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations.*  
   ICLR 2024.  
   URL: `https://openreview.net/forum?id=JewzobRhay`

3. **Wang, H. et al. (2025/2026).**  
   *PrefixMemory-Tuning / Prefix-Tuning+: Modernizing Prefix-Tuning by Decoupling the Prefix from Attention.*  
   OpenReview / arXiv.  
   URL: `https://openreview.net/forum?id=LvUMpZE44r`  
   URL: `https://arxiv.org/abs/2506.13674`

4. **Hou, Y. et al. (2026).**  
   *FlashMem: Distilling Intrinsic Latent Memory via Computation Reuse.*  
   arXiv.  
   URL: `https://arxiv.org/abs/2601.05505`

5. **Wu, D. et al. (2025).**  
   *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.*  
   ICLR 2025.  
   URL: `https://openreview.net/forum?id=pZiyCaVuti`

6. **Maharana, A. et al. (2024).**  
   *Evaluating Very Long-Term Conversational Memory of LLM Agents.*  
   ACL 2024 (LoCoMo).  
   URL: `https://aclanthology.org/2024.acl-long.747/`

7. **Hu, Y. et al. (2025/2026).**  
   *Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions.*  
   MemoryAgentBench.  
   URL: `https://arxiv.org/abs/2507.05257`

8. **Bai, Y. et al. (2025).**  
   *LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks.*  
   ACL 2025.  
   URL: `https://aclanthology.org/2025.acl-long.183/`

9. **Wang, W. et al. (2023).**  
   *Augmenting Language Models with Long-Term Memory.*  
   LongMem.  
   URL: `https://arxiv.org/abs/2306.07174`

10. **Xu, W. et al. (2025).**  
    *A-MEM: Agentic Memory for LLM Agents.*  
    URL: `https://arxiv.org/abs/2502.12110`

11. **Tan, Z. et al. (2025).**  
    *In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents.*  
    ACL 2025.  
    URL: `https://aclanthology.org/2025.acl-long.413/`

12. **Wang, Z. et al. (2024).**  
    *Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs.*  
    EMG-RAG, EMNLP 2024.  
    URL: `https://aclanthology.org/2024.emnlp-main.281/`

13. **Shridhar, M. et al. (2021).**  
    *ALFWorld: Aligning Text and Embodied Environments for Interactive Learning.*  
    ICLR 2021.  
    URL: `https://openreview.net/forum?id=0IOX0YcCdTn`

14. **Yao, S. et al. (2022).**  
    *WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents.*  
    NeurIPS 2022.  
    URL: `https://arxiv.org/abs/2207.01206`

15. **Zhou, H. et al. (2025).**  
    *AgentFly: Fine-tuning LLM Agents without Fine-tuning LLMs.*  
    URL: `https://arxiv.org/abs/2508.16153`

16. **Zhang, G., Fu, M., & Yan, S. (2025).**  
    *MemGen: Weaving Generative Latent Memory for Self-Evolving Agents.*  
    URL: `https://arxiv.org/abs/2509.24704`

---

## 19. Closing note

The correct reading of the current project is not “latent memory failed, therefore memory is over.”  
The correct reading is:

- latent memory injection for **frozen single-turn reasoning/factual recall** has now been stress-tested hard enough to reveal its limits,
- the FlashMem discrepancy remains the one unresolved mechanism question worth a fast decisive test,
- and the scientifically appropriate next battleground is **long-horizon agent memory with persistence, retrieval, compression, and interference control**.

That is what PLANv9 is for.
