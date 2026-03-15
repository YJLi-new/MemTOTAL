# PLANv7 — PLANv6.1-Derived External-Writer Bandwidth Expansion, Mid-Layer Injection, and Query-Compression Recovery Plan for MemTOTAL

**Version:** 2026-03-11  
**Primary language:** English  
**Primary controlling intent:** optimize and supersede the forward-looking content of `PLANv6.1` while preserving its variable priority and its overall scientific honesty  
**Style target:** at least the operational density and explicitness of `PLANv4`  
**Primary hardware assumption:** single NVIDIA RTX PRO 6000 96GB  
**Primary backbone target:** `Qwen2.5-1.5B-Instruct`  
**Primary benchmarks for this phase:** `GSM8K` and `TriviaQA`  
**Calibration benchmark only:** `FEVER`  
**Tertiary / delayed stress benchmark:** `NarrativeQA`  

---

## 0. Scope, intent, and blunt conclusion

This document is **not** a fresh philosophical restart. It is an **optimized, stricter, more operational continuation of `PLANv6.1`**, written under four explicit constraints:

1. `PLANv6.1` is the **main backbone** and should remain the main backbone.
2. The next plan must incorporate the newest repo state and the completed `V6-5` stabilization evidence.
3. The owner’s new hard proposals are **not optional** and must be translated into operational defaults.
4. The result must be detailed enough that a future agent can execute it with minimal guesswork.

The short verdict is:

> **Do not abandon the external Writer. The current repo has proven route liveness and task-gradient reachability, but it has not yet performed a fair test of a high-bandwidth external Writer under mid-layer injection with a co-scaled projector and a bounded query-compression bridge.**

The most important correction is this:

> **The project’s next question is no longer “can FEVER be moved by a small early-prefix Writer?” The next question is “can a wider external Writer produce contentful, non-collapsed `M_long` that improves GSM8K and TriviaQA when injected at the network middle, either directly or via a minimal query-compression bridge?”**

That means the next phase must be organized around the following priority order:

1. **strict measurement repair and continuity replay**
2. **oracle test of the injection mechanism**
3. **low-cost width × depth scout**
4. **main bandwidth expansion at mid layers**
5. **projector co-scaling and per-layer projection**
6. **wide-Writer query compression (`M_long -> M_short`)**
7. **forced memory consumption / receiver reopening**
8. **targeted auxiliary revisit, especially reconstruction-style supervision**
9. **multi-seed confirmation on GSM8K + TriviaQA**

The single most important planning stance is:

> **`PLANv7` keeps the external Writer, widens it aggressively, prioritizes `[12,13,14,15]`, preserves `S3 + C2 + L5` as the architecture anchor, absorbs `V6-5` stabilization settings as defaults, and treats the Reader only as a bounded query-compressor when wide external memory must be shortened before injection.**

---

## 1. Non-negotiable owner-locked constraints

This section exists so the next executor cannot silently drift away from the owner’s instructions.

### 1.1 Locked strategic constraints

The following are **hard constraints**, not soft preferences:

1. **Do not abandon the external Writer yet.**
2. **Writer bandwidth should be increased substantially**, including more slots and more internal capacity.
3. **GSM8K and TriviaQA are the main evaluation pair.**
4. **Layer priority is `[12,13,14,15]`** on `Qwen2.5-1.5B-Instruct`.
5. **The main architecture anchor remains the V6-4 winner:** `S3 + C2 + L5`.
6. **The V6-5 stabilization findings are absorbed into the defaults** rather than re-screened from scratch.
7. **The plan must remain very detailed** and must specify experimental settings rather than relying on hand-wavy future judgment.

### 1.2 Locked operational interpretation of those constraints

`PLANv7` therefore enforces the following translation:

- **Architecture anchor:** `S3 + C2 + L5`
- **Primary injection site:** pure mid-layer `[12,13,14,15]`
- **Continuity regression control:** additive layer recipe `[0,1,2,3,4,8,14]`
- **Default optimizer stabilizers:** groupwise clipping + accumulation `4` + gradient clip `1.0`
- **Default tiny receiver adaptation:** micro-LoRA on the active injection strip, rank `2`, `k_proj/v_proj`
- **Primary benchmark ranking:** `GSM8K` and `TriviaQA`
- **FEVER role:** calibration only, not the main decision axis

### 1.3 Mandatory governance note on the projector-LR discrepancy

There is one critical mismatch that must be recorded explicitly.

The repo-side `V6-5` artifacts record the screened projector learning-rate values as **`5e-5` and `7.5e-5`**, with the stabilized top recipe using `7.5e-5` in the best FEVER-confirmed `F3` branch and several strong `F1`/`F2` branches also using `7.5e-5`. However, the owner’s new hard instruction writes the inherited default as **`7.5e-6`**.

`PLANv7` will therefore use the following rule:

- **Operational default for all new V7 phases:** `pilot_projector_learning_rate = 7.5e-6`
- **But every new config and every new summary must record:**
  - `owner_locked_projector_lr = 7.5e-6`
  - `repo_confirmed_v65_projector_lr_reference = 7.5e-5`
  - `owner_override_note = true`

This prevents silent drift and makes the discrepancy explicit in the record.

### 1.4 The key separation that prevents confusion

The owner’s constraints create an important distinction that `PLANv7` keeps explicit:

- **Architecture anchor** = `S3 + C2 + L5`
- **Stabilization defaults** = groupwise clipping + accumulation `4` + additive continuity baseline + inherited tiny receiver-LoRA pattern

These are **not the same thing**.

The repo’s top FEVER-stable recipe after `V6-5` is `F3` (`S3 + C0 + L2`) under additive layers, but that result is **not** allowed to redefine the architecture anchor for hard-task research. The repo’s `V6-5` outcome is used for stability defaults and continuity controls; it is **not** allowed to overrule the `PLANv6.1` judgment that `S3 + C2 + L5` is the correct mainline architectural starting point.

---

## 2. Materials audited

This plan was prepared by cross-reading the project **jointly** rather than piecemeal.

### 2.1 Primary local planning documents

- `PLANv6dot1.md` — treated as the main backbone to optimize
- `PLANv4.md` — treated as the target style for detail density and execution explicitness
- `PLAN.md`, `PLANv2.md`, `PLANv3.md`, `PLANv5.md`, `PLANv6.md`

### 2.2 Repo-side method and implementation documents

- `docs/MAIN_IDEA.md`
- `docs/ARCHITECTURE.md`
- `docs/EXPERIMENTS_INFO.md`
- active exec plans under `docs/exec-plans/active/`

### 2.3 Core code paths cross-read against the planning claims

- `src/memtotal/models/backbone.py`
- `src/memtotal/models/memory.py`
- `src/memtotal/training/m4_shared_injection.py`
- `src/memtotal/tasks/writer_jointpeft_data.py`
- `src/memtotal/tasks/registry.py`
- `src/memtotal/tasks/sources.py`
- summary scripts for `PLANv6` review and writer-direct review bundles

### 2.4 Latest governed result bundles checked

- `V6-2` support screening
- `V6-3` loss screening
- `V6-4` mixed matrix
- `V6-5` stabilization screen and confirmation

### 2.5 External references intentionally used to shape this plan

- Prefix-Tuning
- P-Tuning v2
- theory of prompting / prefix limitations
- Prefix-Tuning+
- LoRA / PEFT
- Perceiver
- Gist Tokens
- ICAE
- AutoCompressors
- VICReg
- Barlow Twins
- MemGen

---

## 3. Verified facts from code and governed results

This section is deliberately concrete. It records the parts of the current state that should **not** be re-litigated informally.

### 3.1 The deep-prefix path is real

The current backbone path is no longer a fake route or an embedding-concat proxy. The code now constructs **layer-specific prefix hidden states**, transforms them with each decoder layer’s own `k_proj` / `v_proj`, aligns them with RoPE, and writes them into the cache before scoring / generation. This is a real sparse deep-prefix bridge.

### 3.2 The current Writer is still very narrow by the standards of the hypothesis being tested

The current writer-direct baseline uses:

- `memory_slots = 8`
- `hidden_dim = 128`
- `num_heads = 4`
- `transformer_layers = 2`
- `conditioning_layers = 1` (implicit/default)

The output slot dimensionality remains tied to the backbone hidden size, but the Writer’s **internal feed-forward and conditioning capacity** is tiny relative to the problem being asked of it.

### 3.3 `C2` does not mean what a casual reading suggests

The repo’s `SupportContextBalanceGate(mode="layernorm_learned_scalar")` is a **global LayerNorm + learned scalar rescaling of context and support**, not a true per-sample semantic trust gate.

Therefore:

- `C2 > C0` means **magnitude balancing helps**
- it does **not** yet mean the model learned example-wise memory trust
- it does **not** prove intelligent memory acceptance

### 3.4 The current summary logic still conflates three different things

The repo’s current `source_not_collapsed` logic can return `true` if **any** of the following is sufficient:

- support-state rank is above a floor
- writer memory rank is above a floor
- common-mode ratio drops below a threshold
- slot pairwise cosine becomes less degenerate

That means the current governed summaries can say “source not collapsed” while the **Writer slots themselves remain near rank-1**.

### 3.5 The latest architecture winner and the latest stabilization winner are not the same thing

The repo evidence now says two different but compatible things:

- `V6-4` selected the **main architecture anchor** as `S3 + C2 + L5`
- `V6-5` selected the **top FEVER-stabilized recipe** as an additive-layer `F3` branch and the runner-up as an additive-layer `F1` branch

The correct reading is:

- use `V6-4` to define the architecture anchor
- use `V6-5` to define the stabilization defaults and regression controls
- do **not** let a FEVER-only stabilization phase decide the hard-task architecture

### 3.6 Additive layers matter, but the interpretation must be careful

Both FEVER-confirmed `V6-5` finalists used the additive layer set:

- deep-prefix layers: `[0,1,2,3,4,8,14]`
- receiver LoRA layers: `[0,1,2,3,4]`

This is important evidence that layer placement matters and that touching layer `14` is helpful on FEVER. But it is **not** evidence that pure mid-layer `[12,13,14,15]` has already been validated. The current additive result still includes the original early strip.

### 3.7 Task gradients do reach the Writer

The latest governed line is no longer in the “maybe the Writer only sees auxiliary gradients” regime.

The top `V6-4` branches show:

- strong task-to-total Writer gradient ratios
- positive task/total cosine
- relatively minor aux contribution in the top `C2 + L5` branch

This means the route is live enough for meaningful architecture testing.

### 3.8 The Writer itself remains near collapsed even when the projector is active

Across the strongest `V6-4` branches, the pattern is consistent:

- `support_state_effective_rank` is meaningfully above `1`
- `projected_memory_effective_rank` is high
- `writer_memory_slot_effective_rank` remains near `1`
- `memory_long_common_mode_energy_ratio` remains extremely close to `1`

So the projector is producing diverse projected geometry from a Writer source that is still almost entirely common-mode.

### 3.9 The projector is currently a hard bottleneck candidate

The deep-prefix projector uses a **LayerNorm -> down-proj -> bottleneck -> up-proj** structure with a default bottleneck rank of `32`, shared across all target layers.

This is tolerable for the current tiny Writer, but it is an obvious bottleneck risk once Writer bandwidth is expanded.

### 3.10 The current writer-direct data bundle is still locked to the old benchmark trio

`writer_jointpeft_data.py` currently materializes only:

- `gsm8k`
- `narrativeqa`
- `fever`

`TriviaQA` exists in the task registry and source scaffolding, but it is not yet a first-class member of the writer-direct bundle.

### 3.11 There is a subtle implementation risk in deep-prefix zero-padding at non-target layers

The current deep-prefix preparation logic populates non-target layers using a `zero_hidden_template`. That means the model still sees prefix positions at every layer even when only a subset of layers carries meaningful injected content.

This is potentially harmless, but it can also create a weak attention sink / interference pattern, especially when trying to test pure mid-layer injection.

### 3.12 The repo already contains evidence that wider Writer templates were at least contemplated

The bundle still contains wider writer-style templates from earlier branches (for example a `3072`-width writer-weaver template), which confirms that the current `8 × 128 × 2` writer was never meant to be the final scientific answer to the external-Writer question.

---

## 4. Independent mechanistic diagnosis

This section is the most important interpretive section. It is where `PLANv7` goes beyond simply repeating prior conclusions.

### 4.1 The primary blocker is no longer route disconnection

That argument is no longer the best reading of the evidence.

The route now demonstrably does all of the following:

- creates real deep prefixes
- influences attention
- delivers task gradients to the Writer
- improves FEVER stably under some recipes

Therefore the project must stop behaving as if it is still debugging mere physical wiring.

### 4.2 The dominant failure is semantic bandwidth mismatch, not mere numerical instability

The current system is in the following regime:

- the support interface is alive enough to create non-trivial geometry
- the projector can generate active prefix representations
- the receiver can be nudged by tiny LoRA
- FEVER can improve
- but the Writer’s own memory slots remain almost rank-1/common-mode
- and the hard tasks do not move

My interpretation is:

> **The system currently has enough route activity to create low-dimensional bias effects, but not enough contentful, instance-specific Writer bandwidth to move hard reasoning or factual QA outcomes.**

### 4.3 The current FEVER wins are compatible with a weak-memory explanation

FEVER is not useless, but FEVER is too easy to move with the wrong kind of mechanism.

A branch can improve FEVER by any combination of:

- low-dimensional margin bias
- attention sink effects
- global magnitude balancing
- shallow classification steering
- modest lexical prior shifts

None of those, by themselves, prove that the Writer is encoding rich external memory.

So `PLANv7` treats FEVER as a **necessary calibration guardrail** but **not a sufficient proof benchmark**.

### 4.4 `C2` helps because scale balance matters, not because semantic memory routing is solved

The current `C2` result is best explained as a scale-balancing effect:

- the support stream is no longer numerically swamped by the context stream
- Writer conditioning is better balanced
- small prefix signals survive more reliably

That is useful. But it should be read as **conditioning hygiene**, not as proof of intelligent decoding.

### 4.5 The projector is currently fabricating diversity from a weak source

When Writer rank is approximately `1` but projected-memory rank is large, the system is in a “projection manufactures geometric diversity” regime.

That regime can sometimes help FEVER, but it is unlikely to be enough for:

- arithmetic reasoning where the right numbers and constraints must be preserved
- short factual QA where the right entity/relation details must be retained

That is why projector co-scaling and later query compression are central to `PLANv7`.

### 4.6 The credit-assignment problem is probably larger on GSM8K than on TriviaQA

`GSM8K` is hard because a memory intervention may need to affect the **beginning of a multi-step reasoning chain** in order to change the **final answer token**. That is a long and diffuse credit path.

`TriviaQA` is different:

- the answer is shorter
- the relevant information is often more directly factual
- the path from memory to answer is shorter

Therefore the likely first real success is:

- **TriviaQA moves first**
- **GSM8K moves later**

However, because the owner explicitly requires GSM8K + TriviaQA as the main pair, `PLANv7` treats both as primary benchmarks while still recognizing that a `TriviaQA-only` first movement is **partial success, not failure**.

### 4.7 Pure mid-layer injection remains under-tested and is the most important unscreened variable

The additive-layer result strongly suggests that touching the middle of the network helps. But because the current additive recipe still includes early layers, the repo still does **not** know whether the real gain is coming from:

- early layers only,
- layer `14` as a bonus,
- or true mid-network semantic integration.

The only honest answer is to test pure `[12,13,14,15]` directly.

### 4.8 The zero-prefix issue may matter more at mid layers than at early layers

If the model receives meaningless prefix positions at layers that are supposed to be inactive, those positions can still shape attention allocation. This matters most when the real content is delayed until the network middle.

Therefore a sparse-cache or non-target masking diagnostic is worth adding even if it is not the first code change.

### 4.9 The current direct path cannot be the final answer for very wide external memory

Once the Writer becomes much wider, directly injecting all Writer slots as prefix tokens becomes undesirable because it increases:

- prefix length
- interference risk
- compute and memory cost
- mismatch with the long-term paper claim (`write long, read short`)

Therefore the correct next move is **not** “abandon the external Writer.” The correct next move is:

> **let the Writer stay wide, but compress it with a minimal learned query interface before injection.**

### 4.10 A truly fair test of the external Writer has not happened yet

A fair test requires all of the following at once:

- non-tiny Writer bandwidth
- mid-layer injection
- a co-scaled projector
- strict Writer-collapse measurement
- GSM8K + TriviaQA as the decision pair
- a bounded compression bridge for very wide Writer output

The repo has not yet run that test. `PLANv7` is that test.

---

## 5. Architecture stance: what survives scrutiny and what does not

### 5.1 What survives scrutiny

The following remain correct and should remain central:

1. **Keep the external Writer.**
2. **Keep the paper-level endgame in mind:** universal Writer, adaptable query-based reading, write-long/read-short, later CDMI evidence.
3. **Treat `S3 + C2 + L5` as the best architecture anchor currently available.**
4. **Treat mid-layer `[12,13,14,15]` as the main priority.**
5. **Allow a minimal Reader-like query compressor only when it serves wide external memory compression.**

### 5.2 What does not survive scrutiny

The following should **not** control the next stage:

1. **Another FEVER-first recipe sweep**
2. **Another pooled-support or shallow support-mode reopening**
3. **A full Reader/Fuser rescue sweep before bandwidth and mid-layer testing**
4. **Broad receiver fallback before proving that the Writer became more contentful**
5. **Declaring success from logprob deltas without actual score movement on GSM8K or TriviaQA**

### 5.3 The central distinction in `PLANv7`

`PLANv7` distinguishes four different concepts that earlier documents occasionally blurred:

1. **route liveness**
2. **support-interface liveness**
3. **true Writer-memory non-collapse**
4. **downstream usefulness on primary tasks**

Only the fourth item is real success.

### 5.4 The Reader is allowed back in only as a bounded compressor

This is the key compromise between the owner’s instruction and the project’s long-term method story.

`PLANv7` does **not** reopen the old broad Reader rescue debate.

It allows only a **minimal query-compression bridge** under these rules:

- the Writer remains external and primary
- the bridge exists only to compress wide `M_long` into bounded `M_short`
- the bridge is not allowed to become a giant confound
- the bridge search space remains tiny and fixed

### 5.5 The additive-layer result becomes a continuity control, not the main decision rule

Because the owner wants to preserve the V6-5 stabilized recipe and because the repo already proved it is stable on FEVER, `PLANv7` keeps the additive layer recipe as a mandatory continuity regression arm.

But the architecture decision is made by:

- **pure mid4** vs **early4** on GSM8K + TriviaQA,
- not by “whatever won FEVER with additive layers.”

---

## 6. Research questions and success conditions

`PLANv7` is organized around the following explicit questions.

### RQ1. Was the current failure mainly caused by under-provisioned Writer bandwidth?

This asks whether increasing:

- slot count,
- internal Writer width,
- conditioning depth,
- and projector capacity

causes strict Writer-memory metrics and primary-task scores to improve.

### RQ2. Is pure mid-layer `[12,13,14,15]` actually better than early `[0,1,2,3]` for the external Writer?

This is the most important unscreened architecture variable.

### RQ3. Is the projector currently hiding Writer failure by manufacturing rank?

This asks whether a per-layer or higher-rank projector changes the relationship between Writer rank and usefulness.

### RQ4. Do non-target-layer prefix placeholders interfere with the true mid-layer test?

This is the sparse-cache / non-target masking diagnostic.

### RQ5. What is the smallest query-compression bridge that makes a wide Writer viable?

This asks whether `M_long -> M_short` via fixed queries solves the bandwidth problem without reopening the full old Reader debate.

### RQ6. If the Writer becomes better but the tasks remain flat, is the receiver still under-consuming memory?

This motivates receiver-only adaptation and controlled starvation.

### RQ7. Does the Writer need a shorter local supervision path that says what to encode, not just “be different”?

This motivates reconstruction-style auxiliary losses.

### RQ8. What counts as success in this phase?

`PLANv7` defines three levels of success.

#### Partial success

At least one of `GSM8K` or `TriviaQA` improves over control on actual task score, and strict Writer-collapse diagnostics improve.

#### Medium success

One primary task improves across three seeds and the other is at least non-regressive with better Writer diagnostics; FEVER remains non-regressive.

#### Strong success

Both GSM8K and TriviaQA improve over control across three seeds, strict Writer-collapse diagnostics pass, FEVER remains healthy, and the best branch is not relying on projector-manufactured diversity from a near rank-1 Writer.

---

## 7. Global defaults for all V7 phases

This section is intentionally explicit so future executors can materialize configs without guesswork.

## 7.1 Global backbone defaults

```yaml
backbone:
  name: Qwen2.5-1.5B-Instruct
  load_mode: hf_causal_lm
  dtype: bfloat16
  freeze_all_backbone_weights: true
  max_new_tokens:
    gsm8k: 64
    triviaqa: 32
    fever: 8
```

## 7.2 Global task bundle defaults

Unless a phase explicitly says otherwise:

```yaml
tasks:
  primary:
    - gsm8k
    - triviaqa
  calibration:
    - fever
  delayed_stress:
    - narrativeqa
```

### 7.2.1 Default split plans

`PLANv7` uses the following split plans unless a phase explicitly overrides them:

| Benchmark | source_examples | support_examples | train_examples | eval_examples | Status |
|---|---:|---:|---:|---:|---|
| GSM8K | 128 | 8 | 80 | 40 | primary |
| TriviaQA | 128 | 8 | 80 | 40 | primary |
| FEVER | 256 | 8 | 64 | 64 | calibration only |
| NarrativeQA | 64 | 8 | 32 | 24 | delayed stress only |

### 7.2.2 Support policy

`PLANv7` does **not** reopen support retrieval redesign. It keeps support count and serialization fixed unless a phase explicitly states otherwise.

```yaml
runtime:
  pilot_support_examples: 8
  pilot_train_support_mode: static_support_rows
  pilot_support_serialization: example_blocks_raw8
  pilot_support_encoder_mode: multi_item_cross_attn_raw
```

## 7.3 Global writer-conditioning defaults

The mainline `S3 + C2 + L5` family means:

```yaml
runtime:
  pilot_writer_stimulus_mode: support_and_context
  pilot_writer_context_tokens: 8
  pilot_context_support_balance_mode: layernorm_learned_scalar
  pilot_context_balance_scale_init: 1.0
  pilot_support_balance_scale_init: 1.0
  pilot_alignment_aux_mode: off
```

### 7.3.1 Default residual scales

Unless a writer family overrides them:

```yaml
runtime:
  pilot_writer_context_query_residual_scale: 1.0
method:
  writer:
    support_query_residual_scale: 1.0
    output_slot_basis_scale: 0.0
    dropout: 0.0
```

## 7.4 Global optimizer defaults

These defaults apply to all V7 phases unless a phase explicitly overrides them.

```yaml
runtime:
  pilot_lr_schedule: constant_with_linear_warmup
  pilot_lr_warmup_steps: 0
  pilot_projector_warmup_steps: 0
  pilot_writer_learning_rate: 1.0e-4
  pilot_projector_learning_rate: 7.5e-6   # owner-locked override
  repo_confirmed_v65_projector_lr_reference: 7.5e-5
  pilot_receiver_lora_learning_rate: 5.0e-5
  pilot_writer_weight_decay: 0.0
  pilot_projector_weight_decay: 0.0
  pilot_receiver_lora_weight_decay: 0.0
  pilot_gradient_accumulation_steps: 4
  pilot_groupwise_grad_clip: true
  pilot_gradient_clip_norm: 1.0
```

### 7.4.1 Default train-step budgets

| Phase type | train_steps | snapshot_steps |
|---|---:|---|
| smoke / scout | 200 | `[0,10,25,50,100,150,200]` |
| bridge / forced-consumption | 300 | `[0,10,25,50,100,150,200,250,300]` |
| multi-seed confirmation | 300 | `[0,10,25,50,100,150,200,250,300]` |

### 7.4.2 Default seeds

| Purpose | Seeds |
|---|---|
| single-seed screens | `61109` |
| three-seed confirmation | `61109, 61110, 61111` |

## 7.5 Global receiver-LoRA defaults

Unless a phase explicitly widens the receiver reopening:

```yaml
method:
  receiver_lora:
    enabled: true
    target_modules: [k_proj, v_proj]
    rank: 2
    alpha: 4.0
    dropout: 0.0
```

Receiver target layers follow the active injection strip.

- early4 experiments: `[0,1,2,3]`
- mid4 experiments: `[12,13,14,15]`
- additive continuity baseline: `[0,1,2,3,4]`

## 7.6 Global injection defaults

### 7.6.1 Direct phases (`V7-0` through `V7-2`)

```yaml
runtime:
  pilot_bridge_mode: writer_direct
  pilot_memory_path_variant: single_level
  pilot_injection_mode: sparse_deep_prefix
  pilot_projector_token_source: writer_slots
  pilot_prefix_source_mode: writer
  pilot_deep_prefix_init_mode: kv_stat_match
```

### 7.6.2 Bridge phases (`V7-3` onward when enabled)

```yaml
runtime:
  pilot_bridge_mode: writer_direct          # Writer remains external
  pilot_memory_path_variant: two_level
  pilot_injection_mode: sparse_deep_prefix
  pilot_projector_token_source: short_slots
  pilot_prefix_source_mode: writer
  pilot_deep_prefix_init_mode: kv_stat_match
  pilot_reader_context_mode: prompt_summary
```

## 7.7 Global FEVER policy

FEVER is not removed; it is demoted.

- use FEVER in `V7-0` continuity replay
- use FEVER only on promoted finalists in later phases
- do not rank the architecture by FEVER alone

---

## 8. New metrics, gates, and ranking doctrine

The summary logic must be repaired before the next science phase can be trusted.

## 8.1 New metric definitions

`PLANv7` requires the following explicit metrics in all new summaries.

### 8.1.1 Support-interface metrics

- `support_state_effective_rank`
- `support_attention_entropy_mean`
- `support_attention_coverage`
- `context_balance_scale`
- `support_balance_scale`

### 8.1.2 Writer-memory metrics

- `writer_memory_slot_effective_rank`
- `writer_rank_fraction = writer_memory_slot_effective_rank / memory_slots`
- `memory_long_common_mode_energy_ratio`
- `writer_slot_pairwise_cosine_mean`
- `writer_slot_pairwise_cosine_std`
- `writer_slot_norm_mean`
- `writer_slot_norm_std`

### 8.1.3 Projection metrics

- `projected_memory_effective_rank`
- `projector_rank_gain_factor = projected_memory_effective_rank / max(writer_memory_slot_effective_rank, 1e-6)`
- `prefix_attention_mass_mean`
- `prefix_attention_mass_mean_by_layer`
- `prefix_attention_nontrivial_layer_count`

### 8.1.4 Task-causality metrics

- `task_score_delta_vs_control`
- `delta_answer_logprob_mean`
- `first_answer_token_delta`
- `answer_switch_rate`
- `correct_flip_count`
- `positive_delta_fraction`

### 8.1.5 Gradient-attribution metrics

- `writer_task_to_total_grad_ratio_post_unfreeze`
- `writer_task_total_cosine_post_unfreeze_median`
- `writer_task_aux_cosine_post_unfreeze_median`
- `writer_grad_norm_post_unfreeze_median`
- `projector_grad_norm_post_unfreeze_median`
- `receiver_lora_grad_norm_post_unfreeze_median`

## 8.2 New boolean gates

### 8.2.1 `support_interface_alive`

A branch passes `support_interface_alive` if either of the following is true:

- `support_state_effective_rank >= 1.75`
- or `support_attention_entropy_mean` and coverage diagnostics show non-trivial item usage

This gate only says the support interface is alive.

### 8.2.2 `writer_memory_not_collapsed_strict`

This is the crucial new gate.

A branch passes `writer_memory_not_collapsed_strict` **only if all conditions below are true**:

| Writer family | rank floor | rank-fraction floor | common-mode ceiling | pairwise cosine ceiling |
|---|---:|---:|---:|---:|
| W0 / W1 | 2.0 | 0.125 | 0.995 | 0.90 |
| W2 | 4.0 | 0.125 | 0.990 | 0.85 |
| W3 / W4 | 6.0 | 0.125 | 0.985 | 0.80 |

Additionally:

- metrics must be finite
- slot norms must not have degenerated to near-zero variance

### 8.2.3 `projector_manufactured_diversity`

This flag becomes `true` when:

- `projected_memory_effective_rank` is high,
- but `writer_memory_slot_effective_rank` fails the strict gate,
- and `projector_rank_gain_factor > 4.0`

This is a warning flag, not a success flag.

### 8.2.4 `primary_usefulness_positive`

A branch passes `primary_usefulness_positive` only if:

- `task_score_delta_vs_control > 0` on GSM8K **or** TriviaQA,
- and the improvement is not solely FEVER-only.

### 8.2.5 `primary_branch_success`

A branch counts as a real mainline success only if all of the following are true:

1. `route_live_post_unfreeze = true`
2. `stable_training_v6 = true`
3. `writer_task_supervision_live_medium = true`
4. `writer_memory_not_collapsed_strict = true`
5. `primary_usefulness_positive = true`

## 8.3 Ranking rules

### 8.3.1 Screen ranking order

All V7 screens use the following ranking order:

1. actual `GSM8K` score delta
2. actual `TriviaQA` score delta
3. strict Writer-memory improvement
4. first-answer-token delta / answer-switch helpfulness
5. FEVER guardrail (when available)
6. stability

### 8.3.2 Automatic penalties

Apply explicit ranking penalties if:

- `projector_manufactured_diversity = true`
- FEVER improves but both primary tasks remain flat
- Writer rank remains near `1.0`
- common-mode ratio remains near `1.0`
- task/aux cosine becomes strongly negative

### 8.3.3 What no longer counts as success

The following are no longer sufficient:

- FEVER-only wins
- route-live only
- positive logprob delta without answer changes
- `source_not_collapsed = true` under the old permissive logic

---

## 9. Mandatory code work before the first real V7 screen

`PLANv7` requires a small but specific code-preparation phase before the main experiments.

## 9.1 Generalize `writer_jointpeft_data.py`

### Required change

Replace the current hardcoded benchmark bundle with a general interface.

### New CLI contract

```bash
python -m memtotal.tasks.writer_jointpeft_data \
  --output_root ... \
  --source_output_root ... \
  --manifest_root ... \
  --seed 61109 \
  --benchmarks gsm8k,triviaqa,fever \
  --split_plan_json path/to/split_plan.json
```

### Required behavior

- any benchmark in `registry.py` that supports materialization can be selected
- `TriviaQA` becomes first-class
- aliases and normalization metadata must survive materialization
- deterministic split behavior must remain stable

## 9.2 Add mid-layer method presets

Add at least the following configs:

- `configs/method/writer_direct_deep_prefix_receiver_lora_r2_mid4.yaml`
- `configs/method/writer_direct_deep_prefix_receiver_lora_r4_mid4.yaml`
- `configs/method/writer_direct_deep_prefix_no_lora_mid4.yaml`
- `configs/method/writer_direct_deep_prefix_receiver_lora_r2_additive7.yaml` (continuity control)

### Exact layer definitions

```yaml
mid4:
  pilot_deep_prefix_layers: [12,13,14,15]
  receiver_lora.target_layers: [12,13,14,15]

additive7:
  pilot_deep_prefix_layers: [0,1,2,3,4,8,14]
  receiver_lora.target_layers: [0,1,2,3,4]
```

## 9.3 Add explicit Writer bandwidth families

Add separate config fragments for the Writer families defined in Section 10.

At minimum:

- `W0`
- `W1`
- `W2`
- `W3`
- `W4` (optional extreme)

## 9.4 Add projector mode and rank scaling

### Required new config keys

```yaml
runtime:
  pilot_deep_prefix_projector_mode: shared_low_rank | per_layer_low_rank
  pilot_deep_prefix_rank: <int>
```

### Required new implementation

Add `PerLayerLowRankDeepPrefixProjector`, or a functionally equivalent module, so that W2+ is not forced through one shared low-rank up-projection across all target layers.

## 9.5 Add sparse-cache / non-target-layer diagnostic

### Required new config key

```yaml
runtime:
  pilot_non_target_prefix_strategy: dense_zero_cache | sparse_cache
```

### Required behavior

- `dense_zero_cache` = current behavior
- `sparse_cache` = best-effort implementation that avoids meaningful prefix entries at non-target layers

If the Hugging Face cache contract makes `sparse_cache` invasive, that is acceptable, but the implementation must then log `sparse_cache_supported=false` and keep the branch as a deferred diagnostic rather than silently pretending it exists.

## 9.6 Add oracle path(s)

### Required new config keys

```yaml
runtime:
  pilot_prefix_source_mode: writer | source_stub | oracle_context_echo | oracle_support_echo
```

### Required behavior

- `oracle_context_echo`: extract a fixed number of hidden states from the current prompt at the answer boundary and re-inject them as layer-wise prefixes
- `oracle_support_echo`: extract a fixed number of hidden states from the serialized support block and re-inject them as layer-wise prefixes

Both must support early4 and mid4.

## 9.7 Add the bounded query-compression bridge

Reuse the existing two-level Reader/Fuser path where possible, but constrain it to a minimal compression role.

### Required new config families

- `query_compressor_q8_s8`
- `query_compressor_q16_s16`
- optional `query_compressor_q16_s8`

### Required rule

The bridge search space must remain tiny. Do not reopen broad Reader gating or partition sweeps.

## 9.8 Add split prompt support for controlled starvation

The runtime needs a new capability:

- backbone scoring prompt may be masked / starved
- Writer context extraction may still use the full unmasked prompt

### Required new config keys

```yaml
runtime:
  pilot_backbone_prompt_mask_mode: none | gsm8k_numbers | triviaqa_entities
  pilot_writer_context_prompt_mode: same_as_backbone | full_unmasked_prompt
```

## 9.9 Add reconstruction auxiliary plumbing

Add a lightweight local content objective that tells the Writer what to encode.

### Required new config keys

```yaml
runtime:
  pilot_reconstruction_aux_mode: off | hashed_bow | task_keyphrases
  pilot_reconstruction_aux_weight: 0.0
  pilot_reconstruction_vocab_size: 1024
  pilot_reconstruction_hidden_dim: 1024
  pilot_reconstruction_weight_schedule: constant | three_stage_decay
```

## 9.10 Repair and extend the summary contracts

All V7 summaries must include:

- old gates for backward compatibility
- new strict gates for mainline ranking
- owner-LR discrepancy metadata
- projector-mode metadata
- active depth family metadata
- direct vs two-level bridge metadata

---

## 10. Canonical config families for PLANv7

This section is the central operational reference.

## 10.1 Canonical support / stimulus families

### Mainline family: `S3 + C2 + L5`

```yaml
runtime:
  pilot_support_encoder_mode: multi_item_cross_attn_raw
  pilot_support_serialization: example_blocks_raw8
  pilot_writer_stimulus_mode: support_and_context
  pilot_context_support_balance_mode: layernorm_learned_scalar
  pilot_aux_loss_mode: orthogonality_coverage
  pilot_writer_slot_orthogonality_weight: 0.05
  pilot_writer_support_coverage_weight: 0.05
```

### FEVER sentinel family: `S3 + C0 + L2`

This is not the mainline architecture. It is preserved only as a FEVER continuity sentinel because the repo’s top stabilized `V6-5` recipe came from that family.

```yaml
runtime:
  pilot_support_encoder_mode: multi_item_cross_attn_raw
  pilot_support_serialization: example_blocks_raw8
  pilot_writer_stimulus_mode: support_only
  pilot_context_support_balance_mode: off
  pilot_aux_loss_mode: contrastive
  pilot_contrastive_loss_weight: 0.05
  pilot_contrastive_temperature: 0.10
  pilot_contrastive_queue_size: 64
```

## 10.2 Writer families

### Table: canonical Writer families

| Writer ID | memory_slots | hidden_dim | num_heads | transformer_layers | conditioning_layers | default bridge usage | projector mode | deep-prefix rank |
|---|---:|---:|---:|---:|---:|---|---|---:|
| W0 | 8 | 128 | 4 | 2 | 1 | direct | shared | 32 |
| W1 | 16 | 512 | 4 | 2 | 2 | direct | shared | 64 |
| W2 | 32 | 1536 | 8 | 4 | 2 | direct | per-layer preferred | 128 |
| W3 | 64 | 3072 | 8 | 4 | 3 | bridge preferred | per-layer | 128 |
| W4 | 96 | 3072 | 8 | 4 | 3 | bridge only | per-layer | 256 |

### 10.2.1 W0 — current repo baseline

```yaml
method:
  writer:
    arch: transformer
    memory_slots: 8
    hidden_dim: 128
    num_heads: 4
    transformer_layers: 2
    conditioning_layers: 1
    dropout: 0.0
```

### 10.2.2 W1 — low-risk bandwidth expansion

```yaml
method:
  writer:
    arch: transformer
    memory_slots: 16
    hidden_dim: 512
    num_heads: 4
    transformer_layers: 2
    conditioning_layers: 2
    dropout: 0.0
```

### 10.2.3 W2 — main direct Writer candidate

```yaml
method:
  writer:
    arch: transformer
    memory_slots: 32
    hidden_dim: 1536
    num_heads: 8
    transformer_layers: 4
    conditioning_layers: 2
    dropout: 0.0
```

### 10.2.4 W3 — main wide external Writer candidate

```yaml
method:
  writer:
    arch: transformer
    memory_slots: 64
    hidden_dim: 3072
    num_heads: 8
    transformer_layers: 4
    conditioning_layers: 3
    dropout: 0.0
```

### 10.2.5 W4 — optional extreme branch

Open only if W3 is promising and the direct/bridge path is stable.

```yaml
method:
  writer:
    arch: transformer
    memory_slots: 96
    hidden_dim: 3072
    num_heads: 8
    transformer_layers: 4
    conditioning_layers: 3
    dropout: 0.0
```

### 10.2.6 Optional advanced extension: true high-dimensional `M_long`

Only open if W3/W4 still look under-expressive **after** the bounded query-compression bridge.

This optional extension introduces a Writer whose internal `M_long` channel dimensionality exceeds the backbone hidden size and is then compressed down to backbone-sized `M_short` before projection.

This extension is intentionally **not** part of the first V7 sweep because it requires a larger code delta.

## 10.3 Depth families

| Depth ID | deep-prefix layers | receiver LoRA layers | role |
|---|---|---|---|
| D_add | `[0,1,2,3,4,8,14]` | `[0,1,2,3,4]` | continuity regression control |
| D0 | `[0,1,2,3]` | `[0,1,2,3]` | early control |
| D1 | `[12,13,14,15]` | `[12,13,14,15]` | main priority |
| D2 | `[10,11,12,13,14,15]` | `[12,13,14,15]` | wider-mid ablation |
| D3 | `[0,1,2,3,12,13,14,15]` | `[12,13,14,15]` | hybrid ablation |

## 10.4 Projector families

| Projector ID | mode | bottleneck rank | use |
|---|---|---:|---|
| P0 | shared_low_rank | 32 | W0 only |
| P1 | shared_low_rank | 64 | W1 direct control |
| P2 | per_layer_low_rank | 128 | W2 / W3 default |
| P3 | per_layer_low_rank | 256 | W4 or optional advanced branch |

## 10.5 Query-compression bridge families

| Bridge ID | memory_path_variant | reader queries | short_slots | reader context | use |
|---|---|---:|---:|---|---|
| B0 | single_level | — | — | — | direct control |
| B1 | two_level | 8 | 8 | prompt_summary | smallest bridge |
| B2 | two_level | 16 | 16 | prompt_summary | main bridge |
| B3 | two_level | 16 | 8 | prompt_summary | strong compression stress test |

### 10.5.1 Canonical B1 config

```yaml
runtime:
  pilot_memory_path_variant: two_level
  pilot_projector_token_source: short_slots
  pilot_reader_context_mode: prompt_summary
  pilot_reader_num_queries: 8
  pilot_fuser_short_slots: 8
method:
  reader:
    num_queries: 8
    use_query_gating: false
    condition_on_context: true
    conditioning_mode: add
    attention_mode: standard
    dropout: 0.0
    query_residual_scale: 0.0
  fuser:
    arch: resampler
    hidden_dim: 1536
    num_heads: 8
    short_slots: 8
    dropout: 0.0
```

### 10.5.2 Canonical B2 config

```yaml
runtime:
  pilot_memory_path_variant: two_level
  pilot_projector_token_source: short_slots
  pilot_reader_context_mode: prompt_summary
  pilot_reader_num_queries: 16
  pilot_fuser_short_slots: 16
method:
  reader:
    num_queries: 16
    use_query_gating: false
    condition_on_context: true
    conditioning_mode: add
    attention_mode: standard
    dropout: 0.0
    query_residual_scale: 0.0
  fuser:
    arch: resampler
    hidden_dim: 1536
    num_heads: 8
    short_slots: 16
    dropout: 0.0
```

## 10.6 Forced-consumption families

| Family | Description | Main benchmark |
|---|---|---|
| F0 | none / control | both |
| F1 | writer sees full prompt, backbone sees masked prompt | GSM8K first |
| F2 | receiver-only adaptation stage, then joint unfreeze | both |
| F3 | starvation annealing | GSM8K first |
| F4 | dynamic short-slot budget by task | both |

### 10.6.1 F1 — GSM8K number masking

```yaml
runtime:
  pilot_backbone_prompt_mask_mode: gsm8k_numbers
  pilot_writer_context_prompt_mode: full_unmasked_prompt
```

Behavior:

- replace all numeric spans in the backbone prompt with ordered placeholders `<NUM1>`, `<NUM2>`, ...
- preserve the original prompt for Writer context extraction
- evaluate whether the memory path can restore task performance

### 10.6.2 F2 — receiver-only adaptation schedule

```yaml
stage_a:
  steps: 75
  trainable:
    receiver_lora: true
    writer: false
    projector: false
    query_compressor: false

stage_b:
  steps: 125
  trainable:
    receiver_lora: true
    writer: true
    projector: true
    query_compressor: true
```

Use `receiver_lora.rank = 4`, `alpha = 8.0` for this phase only.

### 10.6.3 F3 — starvation annealing

For GSM8K:

- steps `0-50`: all numbers masked
- steps `51-100`: 50% of number spans masked
- steps `101-200` or `101-300`: original prompt restored

### 10.6.4 F4 — dynamic short-slot budget by task

```yaml
triviaqa:
  short_slots: 8
  long_slots: 32

gsm8k:
  short_slots: 16
  long_slots: 64
```

## 10.7 Auxiliary families

| Aux ID | Description | Exact default |
|---|---|---|
| A0 | L5 baseline | orthogonality `0.05`, coverage `0.05` |
| A1 | L5 + VICReg-lite | `vicreg_loss_weight = 0.02` |
| A2 | L5 + contrastive-lite | `contrastive_loss_weight = 0.02`, queue `64`, temp `0.10` |
| A3 | L5 + Barlow-lite | `barlow_loss_weight = 0.02` |
| A4 | L5 + reconstruction-lite | `reconstruction_aux_weight = 0.02` |
| A5 | L5 + reconstruction-lite + VICReg-lite | `0.02 + 0.02` |

### 10.7.1 Reconstruction auxiliary specification

The default reconstruction auxiliary in `PLANv7` is deliberately lightweight and local.

#### Mode

`pilot_reconstruction_aux_mode = hashed_bow`

#### Feature extraction

From the **full source text seen by the Writer**:

- keep all normalized numbers
- keep answer aliases when available
- keep capitalized spans / probable named entities
- keep lowercased content words with length `>= 4`
- remove stopwords
- hash into `1024` bins

#### Prediction head

- input: mean-pooled `M_long` (or mean-pooled `M_short` in bridge phases as an ablation)
- MLP: `hidden_size -> 1024 -> 1024`
- loss: multi-label BCE

#### Weight schedule

Use `three_stage_decay`:

- first `40%` of steps: full weight
- next `30%` of steps: `0.5 ×` weight
- final `30%` of steps: `0.25 ×` weight

This ensures the local signal helps open the slot space without permanently dominating the task loss.

---

## 11. Phase V7-0 — Measurement repair, continuity replay, and oracle gate

## 11.1 Purpose

Before spending serious GPU budget on wide Writers, `PLANv7` first repairs the diagnostics, adds TriviaQA to the bundle, replays the continuity baselines under the new metrics, and runs the oracle gate that tests the injection mechanism directly.

## 11.2 Deliverables

1. strict Writer-memory metrics and gates
2. TriviaQA in the writer-direct bundle
3. mid4 method presets
4. oracle context-echo and support-echo paths
5. continuity replay under additive / early / mid baselines
6. governed `V7-0` review bundle

## 11.3 Exact experiments

### Baseline replay arms

| Arm ID | Writer | Depth | Task set | Notes |
|---|---|---|---|---|
| C0 | no memory control | — | GSM8K, TriviaQA, FEVER | frozen control |
| C_add | W0 + `S3+C2+L5` | D_add | GSM8K, TriviaQA, FEVER | continuity regression baseline |
| C_early | W0 + `S3+C2+L5` | D0 | GSM8K, TriviaQA, FEVER | early control |
| C_mid | W0 + `S3+C2+L5` | D1 | GSM8K, TriviaQA, FEVER | main new baseline |

### Oracle arms

| Arm ID | Oracle source | Depth | Tasks |
|---|---|---|---|
| O_ctx_early | context-echo | D0 | GSM8K, TriviaQA |
| O_ctx_mid | context-echo | D1 | GSM8K, TriviaQA |
| O_sup_early | support-echo | D0 | GSM8K, TriviaQA |
| O_sup_mid | support-echo | D1 | GSM8K, TriviaQA |

## 11.4 Exact config settings

### 11.4.1 Replay arms

Use global defaults plus:

```yaml
runtime:
  pilot_train_steps: 200
  pilot_snapshot_steps: [0,10,25,50,100,150,200]
  pilot_groupwise_grad_clip: true
  pilot_gradient_accumulation_steps: 4
```

### 11.4.2 Oracle arms

Oracle arms are **eval-only** and do not train the Writer.

For both oracle modes:

- `prefix_tokens = 8`
- capture the final `8` prompt-side or support-side hidden states
- preserve per-layer alignment for the target depth strip
- compare directly against the no-memory control on the same examples

## 11.5 Required summary outputs

`V7-0` must report:

- old vs new collapse gates side-by-side
- early vs mid baseline comparison
- early vs mid oracle comparison
- additive continuity replay result
- owner-LR discrepancy metadata

## 11.6 Acceptance criteria

`V7-0` is complete only if all of the following are true:

1. TriviaQA materialization works end-to-end.
2. Mid4 configs run end-to-end.
3. All new strict metrics are present.
4. The additive continuity baseline is replayed on GSM8K, TriviaQA, and FEVER.
5. At least one oracle mode is reported for both early4 and mid4 on GSM8K and TriviaQA.

## 11.7 Decision rule after V7-0

### If `O_ctx_mid` or `O_sup_mid` beats the corresponding early oracle

Then mid4 remains the mainline depth for all later phases.

### If all oracles are flat on both primary tasks

Do **not** kill the external Writer program yet. Instead:

- mark the direct injection route as **high risk**
- still run the low-cost width × depth scout
- and prepare to move to the compression bridge earlier if needed

### If additive continuity beats both early and mid by a wide margin at W0

Do **not** let additive win the architecture by default. Record it as a regression control and still continue the pure mid4 test at W1/W2.

---

## 12. Phase V7-1 — Low-cost width × depth scout

## 12.1 Purpose

This phase merges the low-end bandwidth test with the essential early-vs-mid test, exactly to avoid wasting a full bandwidth ladder on the wrong depth.

## 12.2 Fixed settings

All arms use:

- `S3 + C2 + L5`
- groupwise clipping
- accumulation `4`
- owner projector LR override
- receiver LoRA rank `2`
- train steps `200`
- seed `61109`

## 12.3 Scout matrix

| Arm ID | Writer | Depth | Projector | Tasks |
|---|---|---|---|---|
| S00 | W0 | D0 | P0 | GSM8K, TriviaQA |
| S01 | W0 | D1 | P0 | GSM8K, TriviaQA |
| S10 | W1 | D0 | P1 | GSM8K, TriviaQA |
| S11 | W1 | D1 | P1 | GSM8K, TriviaQA |

### Optional continuity re-check

Do **not** rerun `C_add` unless `V7-0` produced suspicious baselines or the summary contract changed.

## 12.4 Promotion policy

Promote the winning depth into `V7-2` using the following rule:

1. actual score delta on GSM8K + TriviaQA
2. strict Writer-memory metrics
3. answer-switch helpfulness
4. stability

If mid4 is better or tied on primary tasks and clearly better on Writer metrics, mid4 wins.

If early4 clearly wins both primary tasks and Writer metrics, document the upset and let early4 temporarily control `V7-2`.

If the results split by task, prefer the depth that helps `TriviaQA` first **only if** the Writer diagnostics are materially better there.

## 12.5 Acceptance criteria

`V7-1` is complete only if:

- all four scout arms finish,
- ranking is computed under the new strict gates,
- a single winning depth is selected for `V7-2`,
- and FEVER is not used to override the primary-task result.

---

## 13. Phase V7-2 — Main direct-bandwidth ladder at the winning depth

## 13.1 Purpose

This is the first real test of the owner’s bandwidth thesis in the direct path.

## 13.2 Fixed settings

- winning depth from `V7-1` (expected: D1 / mid4)
- `S3 + C2 + L5`
- groupwise clipping
- accumulation `4`
- receiver LoRA rank `2`
- seed `61109`
- train steps `200`

## 13.3 Main matrix

| Arm ID | Writer | Projector | Tasks |
|---|---|---|---|
| D_W1_shared | W1 | P1 | GSM8K, TriviaQA |
| D_W2_shared | W2 | shared rank-64 control | GSM8K, TriviaQA |
| D_W2_perlayer | W2 | P2 | GSM8K, TriviaQA |

### FEVER guardrail

After the main ranking, evaluate FEVER only for the top two promoted arms and the no-memory control.

## 13.4 Exact W2 config

```yaml
method:
  writer:
    memory_slots: 32
    hidden_dim: 1536
    num_heads: 8
    transformer_layers: 4
    conditioning_layers: 2
runtime:
  pilot_deep_prefix_projector_mode: per_layer_low_rank
  pilot_deep_prefix_rank: 128
```

## 13.5 Promotion rules

### Best-case outcome

If `D_W2_perlayer`:

- passes strict Writer-collapse gates,
- and improves GSM8K or TriviaQA actual score,

then promote it directly into multi-seed confirmation **and** into the bridge phase as the direct control.

### Middle outcome

If W2 clearly improves strict Writer metrics but both tasks remain flat, move immediately to `V7-3` and `V7-4`.

### Bad outcome

If W2 destabilizes badly or direct 32-slot injection is obviously too noisy, do **not** declare bandwidth failure. Move to `V7-3` bridge-first wide Writer.

## 13.6 Acceptance criteria

`V7-2` is complete only if the repo can answer both questions below with evidence:

1. Did increasing direct Writer bandwidth materially improve strict Writer-memory metrics?
2. Did co-scaling the projector change usefulness relative to a shared low-rank control?

---

## 14. Phase V7-3 — Wide external Writer with bounded query compression

## 14.1 Purpose

This phase operationalizes the owner’s core idea:

> the Writer may write a much larger memory than what is ultimately injected, because Reader queries can later compress it.

This is where `PLANv7` deliberately reopens the query interface **only as a compression bridge**.

## 14.2 Fixed settings

- best depth from `V7-1`
- `S3 + C2 + L5`
- groupwise clipping
- accumulation `4`
- owner projector LR override
- seed `61109`
- train steps `300`

## 14.3 Main bridge matrix

| Arm ID | Writer | Bridge | Projector | Tasks |
|---|---|---|---|---|
| B_ctrl | best direct arm from V7-2 | B0 | matched | GSM8K, TriviaQA |
| B_W3_q8 | W3 | B1 | P2 | GSM8K, TriviaQA |
| B_W3_q16 | W3 | B2 | P2 | GSM8K, TriviaQA |
| B_W3_q16_s8 | W3 | B3 | P2 | GSM8K, TriviaQA |
| B_W4_q16 | W4 | B2 | P3 | GSM8K, TriviaQA |

### Optional advanced arm

Only open if W3 / W4 still seem slot-bandwidth limited rather than optimization-limited:

- `B_W4x_true_highdim`: optional true high-dimensional `M_long` branch

## 14.4 Exact W3+B2 config

```yaml
method:
  writer:
    memory_slots: 64
    hidden_dim: 3072
    num_heads: 8
    transformer_layers: 4
    conditioning_layers: 3
  reader:
    num_queries: 16
    use_query_gating: false
    condition_on_context: true
    conditioning_mode: add
    attention_mode: standard
    dropout: 0.0
    query_residual_scale: 0.0
  fuser:
    arch: resampler
    hidden_dim: 1536
    num_heads: 8
    short_slots: 16
    dropout: 0.0
runtime:
  pilot_memory_path_variant: two_level
  pilot_projector_token_source: short_slots
  pilot_reader_context_mode: prompt_summary
  pilot_deep_prefix_projector_mode: per_layer_low_rank
  pilot_deep_prefix_rank: 128
```

## 14.5 What counts as success in V7-3

A bridge arm is successful if it does at least one of the following relative to the best direct control:

1. improves actual score on GSM8K or TriviaQA,
2. preserves score while markedly improving strict Writer-memory metrics,
3. preserves score while enabling much larger Writer bandwidth without instability.

## 14.6 Promotion logic

### If B1 or B2 beats the direct control

Then the project’s next canonical architecture becomes:

- wide external Writer
- bounded query-compression bridge
- pure mid-layer injection

### If bridge arms stabilize the Writer but tasks remain flat

Go directly to `V7-4` forced consumption.

### If bridge arms are also flat

Open the reconstruction auxiliary before escalating to any more radical architecture change.

---

## 15. Phase V7-4 — Forced memory consumption and minimal receiver reopening

## 15.1 Purpose

This phase tests whether the bottleneck has moved from **writing** to **consuming**.

## 15.2 Fixed settings

Use the best branch from `V7-2` or `V7-3` as the control.

## 15.3 Main matrix

| Arm ID | Variant | Tasks |
|---|---|---|
| F0 | control | GSM8K, TriviaQA |
| F1_num_mask | writer sees full prompt, backbone sees GSM8K numbers masked | GSM8K |
| F2_rx_only | receiver-only adaptation then joint unfreeze | GSM8K, TriviaQA |
| F3_anneal | starvation annealing | GSM8K |
| F4_dyn_budget | dynamic short-slot budget by task | GSM8K, TriviaQA |

## 15.4 Exact F2 config

```yaml
method:
  receiver_lora:
    enabled: true
    target_layers: [12,13,14,15]   # or active winning depth
    target_modules: [k_proj, v_proj]
    rank: 4
    alpha: 8.0
    dropout: 0.0
runtime:
  stage_a_steps: 75
  stage_b_steps: 125
  pilot_receiver_lora_learning_rate: 5.0e-5
```

## 15.5 Acceptance criteria

A forced-consumption branch is worth keeping only if it changes **actual primary-task scores**, not just logprobs.

If it only raises logprob deltas while the answer metrics stay flat, it is diagnostic only and does not become the new mainline.

---

## 16. Phase V7-5 — Targeted auxiliary revisit, including reconstruction

## 16.1 Purpose

This phase is intentionally late.

It only opens **after** bandwidth, projector co-scaling, and bounded query compression have been given a fair attempt.

## 16.2 Fixed settings

Use the best branch from `V7-3` or `V7-4` as the base.

## 16.3 Main matrix

| Arm ID | Aux family | Tasks |
|---|---|---|
| A0 | L5 baseline | GSM8K, TriviaQA |
| A1 | L5 + reconstruction-lite | GSM8K, TriviaQA |
| A2 | L5 + VICReg-lite | GSM8K, TriviaQA |
| A3 | L5 + contrastive-lite | GSM8K, TriviaQA |
| A4 | L5 + reconstruction-lite + VICReg-lite | GSM8K, TriviaQA |

### Optional only if cleanly supported in code

| Arm ID | Aux family |
|---|---|
| A5 | L5 + Barlow-lite |

## 16.4 Exact default weights

### A0

```yaml
runtime:
  pilot_aux_loss_mode: orthogonality_coverage
  pilot_writer_slot_orthogonality_weight: 0.05
  pilot_writer_support_coverage_weight: 0.05
```

### A1

```yaml
runtime:
  pilot_aux_loss_mode: orthogonality_coverage
  pilot_writer_slot_orthogonality_weight: 0.05
  pilot_writer_support_coverage_weight: 0.05
  pilot_reconstruction_aux_mode: hashed_bow
  pilot_reconstruction_aux_weight: 0.02
  pilot_reconstruction_vocab_size: 1024
  pilot_reconstruction_hidden_dim: 1024
  pilot_reconstruction_weight_schedule: three_stage_decay
```

### A2

```yaml
runtime:
  pilot_aux_loss_mode: orthogonality_coverage
  pilot_writer_slot_orthogonality_weight: 0.05
  pilot_writer_support_coverage_weight: 0.05
  pilot_vicreg_loss_weight: 0.02
  pilot_vicreg_invariance_weight: 1.0
  pilot_vicreg_variance_weight: 1.0
  pilot_vicreg_covariance_weight: 1.0
  pilot_vicreg_variance_target: 1.0
```

### A3

```yaml
runtime:
  pilot_aux_loss_mode: orthogonality_coverage
  pilot_writer_slot_orthogonality_weight: 0.05
  pilot_writer_support_coverage_weight: 0.05
  pilot_contrastive_loss_weight: 0.02
  pilot_contrastive_temperature: 0.10
  pilot_contrastive_queue_size: 64
```

## 16.5 Acceptance criteria

Keep an auxiliary family only if it does one of the following **without hurting the other primary benchmark**:

- improves actual task score,
- or materially improves strict Writer-memory metrics while keeping primary-task score non-regressive.

The reconstruction-style auxiliary is considered especially important because it addresses the “what should the Writer encode?” problem more directly than decorrelation alone.

---

## 17. Phase V7-6 — Multi-seed confirmation, paper-facing comparator, and decision point

## 17.1 Purpose

Once a best branch exists, confirm it before reopening broader macro goals.

## 17.2 Confirmation protocol

Take the best one or two branches and run:

- seeds: `61109, 61110, 61111`
- tasks: `GSM8K`, `TriviaQA`, `FEVER`
- controls:
  - frozen no-memory control
  - additive continuity baseline
  - best earlier direct baseline if the winner is a bridge branch

## 17.3 Optional paper-facing comparator

If time and code budget permit, add **one** minimal comparator inspired by MemGen’s training logic:

- fixed insertion
- frozen backbone
- same latent-token budget as the best `PLANv7` branch
- no trigger learning

This is a positioning comparator, not the mainline program.

## 17.4 Decision logic after V7-6

### Path P — external Writer survives as the main thesis

Choose this path if:

- at least partial success is reproduced across three seeds,
- strict Writer-memory metrics improve,
- and the best branch is not obviously a projector-only illusion.

### Path Q — external Writer remains unresolved but not dead

Choose this path if:

- Writer metrics improve,
- but primary-task scores still fail to move consistently.

Then the next narrow branch should test a stronger integrated Writer or a true high-dimensional `M_long` extension before any drastic thesis reversal.

### Path R — architecture pivot required

Only choose this if all of the following are true:

- the oracle gate is weak,
- wide Writer + bridge + forced consumption + reconstruction still fail,
- and three-seed confirmation finds no real primary-task gain.

Only then is a backbone-native Writer / integrated weaver pivot justified.

---

## 18. Kill criteria, stop rules, and anti-loop rules

This section exists to prevent another long but causally confused phase.

## 18.1 Kill criterion A — direct injection ceiling warning

If **all** of the following are true after `V7-0`:

- `O_ctx_mid` flat on both primary tasks,
- `O_sup_mid` flat on both primary tasks,
- and mid4 baseline shows no better first-answer-token behavior than early4,

then mark direct mid-layer injection as **high risk** and do not spend large compute on direct `W4` arms.

## 18.2 Kill criterion B — Writer still near rank-1 after W2

If after `V7-2` the best W2 arm still has:

- `writer_memory_slot_effective_rank < 2.0`
- and `memory_long_common_mode_energy_ratio > 0.995`

then do **not** reopen another broad aux sweep yet. Move first to bounded query compression and/or sparse-cache diagnostics.

## 18.3 Kill criterion C — no primary-task movement after bridge + consumption + reconstruction

If after `V7-5` no branch improves either GSM8K or TriviaQA actual score across three seeds, then the external Writer line is not paper-ready and the next branch must escalate architecturally.

## 18.4 Anti-loop rule

Do **not** run another FEVER-only sweep during V7.

## 18.5 Anti-confusion rule

Do **not** allow FEVER, route-live, or positive logprob shifts to overrule actual primary-task score movement.

---

## 19. Resource policy for one RTX PRO 6000 96GB

`PLANv7` is intentionally staged for one GPU.

## 19.1 Main resource rules

1. Do not open all axes at once.
2. Keep direct prefix tokens at or below `32` for routine screens.
3. Use bridge compression for `64+` slot Writers.
4. Keep the backbone frozen.
5. Keep the receiver tiny except in the dedicated reopening phase.

## 19.2 Recommended safe envelope

| Path | Safe routine envelope |
|---|---|
| direct path | W0 / W1 / W2 up to `32` slots |
| bridge path | W3 / W4 with `short_slots <= 16` |
| receiver reopening | rank `4` only in dedicated phase |

## 19.3 Temp-directory policy

Because the repo already encountered temp-directory failures during `V6-5`, future V7 runners should keep the stabilized policy:

- prefer system temp for speed when free-space guardrails pass
- otherwise fall back to data-disk temp
- log the selected temp policy explicitly at run start

## 19.4 Suggested run ordering to minimize wasted compute

1. `V7-0`
2. `V7-1`
3. `V7-2`
4. `V7-3`
5. `V7-4`
6. `V7-5`
7. `V7-6`

Do not skip phases and then come back unless a kill criterion explicitly forces the jump.

---

## 20. Concrete implementation map by file and script

## 20.1 Files likely to change immediately

### Data / task bundle

- `src/memtotal/tasks/writer_jointpeft_data.py`
- `src/memtotal/tasks/registry.py`
- `src/memtotal/tasks/sources.py`

### Model / memory modules

- `src/memtotal/models/memory.py`
- `src/memtotal/models/backbone.py`
- `src/memtotal/training/m4_shared_injection.py`

### Summaries / analysis

- `scripts/update_writer_deep_prefix_jointpeft_summary.py`
- new `V7-*` summary scripts

## 20.2 New config files to add

### Method configs

- `configs/method/writer_direct_deep_prefix_no_lora_mid4.yaml`
- `configs/method/writer_direct_deep_prefix_receiver_lora_r2_mid4.yaml`
- `configs/method/writer_direct_deep_prefix_receiver_lora_r4_mid4.yaml`
- `configs/method/writer_direct_deep_prefix_receiver_lora_r2_additive7.yaml`

### Writer family fragments

- `configs/method/writer_bandwidth_w1.yaml`
- `configs/method/writer_bandwidth_w2.yaml`
- `configs/method/writer_bandwidth_w3.yaml`
- `configs/method/writer_bandwidth_w4.yaml`

### Bridge configs

- `configs/method/query_compressor_q8_s8.yaml`
- `configs/method/query_compressor_q16_s16.yaml`
- `configs/method/query_compressor_q16_s8.yaml`

### Experiment templates

- `configs/exp/writer_direct_gsm8k_triviaqa_mid4_template.yaml`
- `configs/exp/writer_direct_triviaqa_mid4_template.yaml`
- `configs/exp/writer_bridge_gsm8k_triviaqa_mid4_template.yaml`
- `configs/exp/writer_continuity_additive_template.yaml`

## 20.3 New scripts to add

- `scripts/run_planv7_v7_0_metrics_oracle_qwen25.sh`
- `scripts/update_planv7_v7_0_metrics_oracle_summary.py`
- `scripts/run_planv7_v7_1_width_depth_scout_qwen25.sh`
- `scripts/update_planv7_v7_1_width_depth_summary.py`
- `scripts/run_planv7_v7_2_direct_bandwidth_qwen25.sh`
- `scripts/update_planv7_v7_2_direct_bandwidth_summary.py`
- `scripts/run_planv7_v7_3_bridge_qwen25.sh`
- `scripts/update_planv7_v7_3_bridge_summary.py`
- `scripts/run_planv7_v7_4_consumption_qwen25.sh`
- `scripts/update_planv7_v7_4_consumption_summary.py`
- `scripts/run_planv7_v7_5_aux_qwen25.sh`
- `scripts/update_planv7_v7_5_aux_summary.py`
- `scripts/run_planv7_v7_6_confirmation_qwen25.sh`
- `scripts/update_planv7_v7_6_confirmation_summary.py`

## 20.4 Tests that should be added

- strict collapse-gate unit tests
- mid4 receiver-layer dispatch tests
- per-layer projector shape / initialization tests
- sparse-cache fallback tests
- oracle context-echo / support-echo tests
- TriviaQA bundle materialization tests
- writer-context split prompt masking tests
- reconstruction auxiliary summary-contract tests

---

## 21. Immediate execution slice: the first commands after accepting this plan

This section intentionally mirrors the explicitness style of `PLANv4`.

### Step 1 — freeze the governance

Create the active exec plan:

```text
docs/exec-plans/active/20260311-planv7-v7-0-metrics-oracle-qwen25.md
```

### Step 2 — implement only the V7-0 code delta

Do **not** start the W2/W3 bandwidth runs yet.

Required first wave only:

1. TriviaQA bundle support in `writer_jointpeft_data.py`
2. mid4 configs
3. strict collapse metrics in the summary pipeline
4. oracle path(s)
5. owner-LR discrepancy metadata logging

### Step 3 — run `V7-0`

Run:

- `C0`
- `C_add`
- `C_early`
- `C_mid`
- `O_ctx_early`
- `O_ctx_mid`
- `O_sup_early`
- `O_sup_mid`

### Step 4 — only then open the low-cost width × depth scout

Do **not** open `V7-2` or `V7-3` before `V7-0` publishes and is read.

### Step 5 — choose the winning depth before widening aggressively

This is the main sequencing protection from `PLANv7`.

---

## 22. Final blunt summary

This plan intentionally preserves the core scientific judgment of `PLANv6.1` while making five critical upgrades:

1. It locks the owner’s new hard constraints as actual defaults.
2. It explicitly separates architecture anchor from FEVER stabilization winner.
3. It introduces projector co-scaling earlier, because a wide Writer without a wider projector is not a fair test.
4. It adds a bounded query-compression bridge as the correct way to keep the external Writer while allowing much larger memory bandwidth.
5. It adds a reconstruction-style auxiliary because the current system still lacks a short supervision path telling the Writer what to encode.

If this plan succeeds, the next credible architecture claim becomes:

> **A widened external Writer, injected at the network middle and optionally compressed by a tiny query interface, can produce useful machine-native memory on GSM8K and TriviaQA without abandoning the project’s long-term write-long / read-short thesis.**

If it fails, the project will at least fail **honestly**, after finally giving the external Writer a fair test.

---

## 23. References and recommended literature

The references below are selected because they directly inform the design choices in this plan.

### PEFT / prefix foundations

1. Li, X. L., & Liang, P. (2021). **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** ACL 2021.  
   URL: https://arxiv.org/abs/2101.00190

2. Liu, X. et al. (2022). **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks.** ACL 2022.  
   URL: https://arxiv.org/abs/2110.07602

3. Hu, E. J. et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR 2022.  
   URL: https://arxiv.org/abs/2106.09685

4. Hugging Face. **PEFT documentation / repository.**  
   URLs: https://github.com/huggingface/peft ; https://huggingface.co/docs/peft/en/index

### Theory and modern prefix limitations

5. Petrov, A., Torr, P. H. S., & Bibi, A. (2024). **When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations.** ICLR 2024.  
   URL: https://arxiv.org/abs/2310.19698

6. Wang, H. et al. (2025). **Prefix-Tuning+: Modernizing Prefix-Tuning through Attention Independent Prefix Data.**  
   URL: https://arxiv.org/abs/2506.13674

### Compression / latent bottleneck work

7. Jaegle, A. et al. (2021). **Perceiver: General Perception with Iterative Attention.** ICML 2021.  
   URL: https://arxiv.org/abs/2103.03206

8. Mu, J., Li, X. L., & Goodman, N. (2023). **Learning to Compress Prompts with Gist Tokens.** NeurIPS 2023.  
   URL: https://arxiv.org/abs/2304.08467

9. Ge, T. et al. (2024). **In-context Autoencoder for Context Compression in a Large Language Model.** ICLR 2024.  
   URL: https://arxiv.org/abs/2307.06945

10. Chevalier, A. et al. (2023). **Adapting Language Models to Compress Contexts / AutoCompressors.** EMNLP 2023.  
    URL: https://arxiv.org/abs/2305.14788

### Anti-collapse / redundancy-reduction objectives

11. Bardes, A., Ponce, J., & LeCun, Y. (2022). **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.** ICLR 2022.  
    URL: https://arxiv.org/abs/2105.04906

12. Zbontar, J. et al. (2021). **Barlow Twins: Self-Supervised Learning via Redundancy Reduction.** ICML 2021.  
    URL: https://arxiv.org/abs/2103.03230

### Continuous latent reasoning inspiration

13. Xu, Y. et al. (2025). **SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs.**  
    URL: https://arxiv.org/abs/2502.12134

### Memory-specific comparison point

14. Zhang, G., Fu, M., & Yan, S. (2025). **MemGen: Weaving Generative Latent Memory for Self-Evolving Agents.** arXiv 2025; public OpenReview / repo list it as ICLR 2026 poster.  
    URLs: https://arxiv.org/abs/2509.24704 ; https://openreview.net/forum?id=vI56m4Iu4e ; https://github.com/KANABOON1/MemGen

### Backbone reference

15. Qwen Team (2024). **Qwen2.5-1.5B-Instruct model card.**  
    URL: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

---

## 24. One-line execution shorthand for future agents

If a future agent needs a single sentence version of this plan, it is:

> **Run `V7-0` first; keep `PLANv6.1` as the backbone; preserve `S3 + C2 + L5` as the architecture anchor; import `V6-5` stabilization defaults; choose depth by an early-vs-mid scout; widen the Writer aggressively; co-scale the projector; use bounded query compression for wide `M_long`; force consumption only after bandwidth is fair; and judge success only by GSM8K / TriviaQA actual score movement under strict Writer-memory gates.**

---

**End of PLANv7**
