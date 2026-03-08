# PLAN.md — MemTOTAL Execution Plan After M5.2

**Version:** 2026-03-08  
**Scope:** research + engineering execution plan for the next major MemTOTAL workstream  
**Audience:** project owner / core contributors / agents operating inside this repo  
**Language:** English  

---

## 0. What this document is, and what it is not

This file is the **next-step execution plan** for the MemTOTAL repository after the real `M5.2 writer objective rewrite` run (`2761fb2`) and the current docs bundle review.

It is written to be:

- **operational**: someone should be able to execute from it,
- **theory-aware**: the order of work should match the actual hypotheses being tested,
- **stable**: it should reduce random branching and prevent the project from spending another long cycle changing too many variables at once,
- **compatible with the current repo**: it should reuse what already exists in `src/`, `configs/`, `scripts/`, `docs/`, and `results/`.

This document does **not** replace:

- `docs/MAIN_IDEA.md` for the paper-level claim,
- `docs/TODO_LIST.md` for milestone bookkeeping,
- `docs/EXPERIMENTS_INFO.md` for artifact/reporting contract,
- existing run summaries for historical evidence.

Instead, this plan sits **between** those documents and the code: it explains **what to do next, in what order, why, and with which stop criteria**.

---

## 1. Inputs that this plan assumes have been reviewed

This plan assumes the following have already been read or inspected:

1. The current public repository state.
2. The uploaded local repo bundle and docs bundle.
3. Existing repo documentation, especially:
   - `docs/MAIN_IDEA.md`
   - `docs/TODO_LIST.md`
   - `docs/EXPERIMENTS_INFO.md`
   - `docs/briefs/20260307-m4-shared-injection-brief.md`
4. Active code paths, especially:
   - `src/memtotal/models/memory.py`
   - `src/memtotal/pipeline.py`
   - `src/memtotal/training/m3.py`
   - `src/memtotal/training/m4_shared_injection.py`
5. Recent result artifacts, especially:
   - `results/generated/review/m4-fever-shared-injection-alignment-qwen25/`
   - `results/generated/review/m5-fever-writer-reasoner-alignment-qwen25/`
   - `results/generated/review/m5-fever-writer-objective-rewrite-qwen25/`

This plan is therefore **not speculative in a vacuum**. It is built around the actual code and the actual failure modes already observed.

---

## 2. One crucial correction: distinguish the **method track** from the **active FEVER harness track**

Before setting any next-step plan, we must freeze one important interpretation.

### 2.1 The repository as a whole already contains the full two-level MemTOTAL method

At the repo level, the project **already implements** the core MemTOTAL method family:

- `MemoryWriter`
- `MemoryReader`
- `MemoryFuser`
- `MemoryInjector`
- the full `Writer -> Reader -> Fuser -> Injector` execution path in `pipeline.py`
- Stage A / B / C training and adaptation in `m3.py`

So, as a **codebase**, MemTOTAL is **not** missing:

- explicit `M_long -> M_short` structure,
- multi-query readout,
- query/fuser meta-training hooks,
- query-only adaptation machinery.

### 2.2 But the currently active FEVER M4/M5 line is a narrower branch

The line that has consumed the most recent effort (`M4.7 -> M5.1 -> M5.2`) is **not** the full two-level method. It is a narrower, more diagnostic branch:

```text
structured support-set encoder
-> writer
-> shared low-rank deep prefix projector
-> frozen Qwen
```

That branch is testing a narrower proposition:

> Can support-derived latent memory be injected through a shared deep-prefix path such that frozen Qwen consumes it on the main attention path?

This is important because it changes how we interpret the latest failures.

### 2.3 Consequence for planning

From now on, the project should explicitly separate two questions:

1. **Substrate question** (current FEVER branch):  
   Can the frozen backbone reliably consume a learned latent prefix in a content-sensitive way?

2. **Full MemTOTAL question** (main paper method):  
   Does explicit `M_long -> Reader -> M_short` improve transfer, stability, compression, and CDMI behavior?

The immediate plan must therefore do **both** of the following, in order:

- settle the single-level FEVER substrate question with **one decisive objective pass**, then
- move into the true two-level proof-of-concept using the **already existing Reader/Fuser modules**.

---

## 3. Current diagnosis: what is already settled vs. what is still open

This section is the diagnosis that drives the plan.

### 3.1 What is already settled enough to stop debating

The recent runs jointly establish the following.

#### Settled A — Frozen Qwen does read the injected path

The project is no longer at “maybe the prefix is ignored.”

- explicit support text helps,
- main-chain consumption has been observed,
- deep prompt attention consumption is non-zero,
- `I_real > I_shuffle / I_zero` has appeared locally in earlier runs.

So the system is **not prefix-blind**.

#### Settled B — The writer family is not empty noise

The latent coming out of the current writer family is not “completely useless.”

Earlier probes and the FEVER line together indicate that:

- the writer latent can contain information,
- structured support-set encoding is better than pooled support block,
- a trainable writer is better than freezing it under the current structured FEVER path.

So the failure is **not** “writer output contains absolutely nothing.”

#### Settled C — Same-schema warm-start and latent anchoring help preserve a local manifold

`M5.1` and `M5.2` show that:

- same-schema warm-start matters,
- latent anchoring can preserve the warm-start manifold,
- but preservation alone is not sufficient to pass `screen248-val` selection.

So “keep the manifold intact” is useful, but **not enough**.

#### Settled D — The current teacher-margin auxiliary path is too weak / too sparse

The `M5.2` canonical result says the current `teacher_margin` hook stayed effectively dormant.

That means the current aux path is failing **as a training signal design**, not just as a hyperparameter choice.

### 3.2 What the latest failure most likely means

The strongest current interpretation is:

> The active shared-injection FEVER path already has some real content sensitivity, but the latent written by the current writer/projector is still only weakly readable by frozen Qwen. Under tiny-data, frozen-receiver, narrow-channel training, task-first gradients quickly find a cheaper label-bias shortcut than stable support-conditioned discrimination.

In plainer language:

- Qwen can consume the prefix,
- the latent channel is not dead,
- but the current latent geometry is not yet aligned to the frozen reasoner’s decision geometry,
- so optimization collapses into a dominant-label attractor.

### 3.3 One specific warning signal: the hinge schedule is now suspicious

In `M5.2`, all three arms collapse into a dominant-label trajectory around **step 16**.

That is noteworthy because the strongest-competitor hinge schedule reaches full strength at roughly the same point under the current setup.

This does **not** prove the hinge caused collapse, but it is strong enough that the next phase must treat the hinge schedule as an explicit design variable, not as an unquestioned constant.

### 3.4 What is still open

The following are still unresolved:

1. Can a **dense, task-space teacher objective** stabilize the shared-injection path where `teacher_margin` failed?
2. Is the current bottleneck really **objective-side**, or is the active FEVER path now limited by being **single-level**?
3. If the single-level path remains fragile, will explicit `M_long -> M_short` improve stability by forcing a more structured readout?
4. When do we actually need receiver-side adaptation, and when would that only hide a memory-side failure?

---

## 4. Project-wide governing rules

These rules apply to all next phases.

### 4.1 Preserve repo milestone numbering

`docs/TODO_LIST.md` already uses `M0–M6` for the project.

This plan will therefore use **Workstreams** and **Phases** instead of inventing new official repo milestone numbers beyond what already exists.

Concretely:

- the next single-level decisive pass is still conceptually **`M5.3`**,
- the later two-level proof-of-concept will be called **`Two-Level PoC`** or **`TL-PoC`**, not a conflicting repo-wide `M6`.

### 4.2 One major question per phase

Do **not** change multiple hypothesis families at once.

Each phase should answer one dominant question:

- objective question,
- architecture question,
- transfer question,
- CDMI question,
- receiver fallback question.

### 4.3 `screen248-test` remains the primary capability gate

Keep the current rule:

- `screen248-val` selection decides whether a candidate checkpoint is worth promoting,
- `screen248-test` is the primary capability gate,
- `fixed64` is legacy-only reporting and **must not veto** the milestone.

### 4.4 No return to dead lines unless a new fact appears

Do **not** go back to the old `candidate-conditioned residual family` unless a genuinely new mechanism changes the question. The repo and docs have already narrowed away from that line.

### 4.5 No premature scale jump

Do **not** jump to:

- Qwen3-8B,
- broader benchmark sweeps,
- large Story Cloze/NarrativeQA campaigns,
- receiver adaptation,
- full token-level KL,

before the immediate single-level substrate question has had one decisive dense-teacher test.

### 4.6 Do not overclaim from FEVER

FEVER remains useful, but the project must remember its role:

- FEVER is currently the **best substrate testbed**,
- FEVER is **not** the final proof of reasoning-memory behavior,
- FEVER should not be allowed to silently become the whole paper.

### 4.7 Every phase must leave a clean artifact trail

Every new phase must update:

- configs,
- scripts,
- result summaries,
- review artifacts,
- docs bundle,
- a one-paragraph diagnosis note in docs.

No manual copy/paste summaries. Use scripts.

---

## 5. Program overview

The project should now proceed in **five workstreams**, with strict ordering.

### Workstream A — Decisive single-level `M5.3`

Goal: settle whether a dense, task-space teacher-aided objective can stabilize the current shared-injection FEVER substrate.

### Workstream B — True Two-Level PoC (`TL-PoC`)

Goal: activate the already-implemented `Reader/Fuser` path inside the active FEVER harness and test explicit `M_long -> M_short`.

### Workstream C — Transfer / meta-learning evidence refresh

Goal: return from FEVER-only diagnostics to the repo’s native Stage B/C claim: fixed Writer, adaptable Reader queries/fuser.

### Workstream D — CDMI and benchmark expansion

Goal: test whether the two-level path plus query-side adaptation actually reduces cross-domain interference under matched budgets.

### Workstream E — Conditional receiver-side fallback

Goal: only if needed, add minimal receiver adaptation in a way that preserves interpretability of failure attribution.

The rest of this document specifies these workstreams in detail.

---

# Workstream A — Decisive `M5.3`: dense teacher-aided objective under shared injection

## 6. Workstream A objective

### 6.1 Exact research question

The question for Workstream A is:

> If we keep the current single-level FEVER shared-injection path fixed, and replace the dormant sparse `teacher_margin` auxiliary signal with a dense, task-space teacher-alignment signal, can we stabilize content-sensitive behavior enough to pass the existing gate?

This is **not** a general distillation phase.

It is a narrow test of whether the current blocker is mainly:

- insufficient alignment signal, vs.
- insufficient architecture.

### 6.2 What must stay fixed

To protect attribution, keep the following constant relative to current M5 runs unless code compatibility forces a change:

- backbone: `Qwen2.5-1.5B-Instruct`
- task family: FEVER phase-2 validation/test protocol
- support representation: structured support-set encoder
- arm type: shared injection / candidate-independent path
- injection type: sparse deep prefix
- deep prefix layers: current canonical set
- deep prefix rank: current canonical rank
- support serialization: current canonical support serialization
- same warm-start family: same-schema warm-start
- primary gate protocol: unchanged
- `fixed64`: legacy only

### 6.3 What is allowed to change

Only these are allowed to change in the canonical M5.3 pass:

1. **teacher auxiliary objective design**,
2. **teacher auxiliary schedule**,
3. **hinge schedule / weight**,
4. **diagnostic logging**.

Do **not** simultaneously change:

- support protocol,
- support bank construction,
- projector architecture,
- backbone size,
- receiver adapters,
- target task.

---

## 7. Canonical M5.3 loss design

## 7.1 Total loss

For Workstream A, the canonical loss should be:

\[
L = L_{CE} + \lambda_h(t) L_{hinge} + \lambda_a(t) L_{anchor} + \lambda_t(t) L_{teach}
\]

Where:

- `L_CE`: standard FEVER 3-way choice cross-entropy on injected-path logits,
- `L_hinge`: strongest-competitor margin loss,
- `L_anchor`: latent-anchor continuation loss already implemented,
- `L_teach`: new dense teacher-alignment term.

### 7.2 Why this structure

- `CE` keeps task supervision anchored to the actual target.
- `hinge` can improve discrimination, but should now be treated as **potentially destabilizing** and therefore softer/later than before.
- `anchor` protects the warm-start manifold.
- `teach` should provide the missing directional signal toward the frozen reasoner’s decision geometry.

---

## 8. Replace sparse `teacher_margin` with dense **choice-space distillation**

### 8.1 Canonical teacher target

For each training example, define:

- `z_S`: student / injected-path 3-way logits,
- `z_T`: teacher-text 3-way logits,
- `z_A`: base / no-memory 3-way logits,
- `y`: gold label.

The teacher must remain the **explicit support text path**, because that is the current strongest available task-aligned target within the same backbone family.

### 8.2 Margin helper

Define the gold-vs-strongest-competitor margin:

\[
m(z, y) = z_y - \max_{c \neq y} z_c
\]

Then define teacher advantage over base:

\[
\Delta = m(z_T, y) - m(z_A, y)
\]

### 8.3 Continuous sample weighting

Do **not** use a hard gate like “activate only if `teacher_margin > base_margin`”.

Instead, define a smooth sample weight:

\[
w(\Delta) = \sigma\left(\frac{\Delta - \delta}{\gamma}\right)
\]

Recommended initial values:

- `delta = 0.0`
- `gamma = 0.25`

This gives:

- low weight when teacher is not clearly better than base,
- high weight when teacher provides a large useful margin advantage,
- no discontinuous on/off behavior.

### 8.4 Canonical distillation term

Use a 3-way choice-space KL term, **not** full token-distribution KL:

\[
p_T^{(\tau)} = \mathrm{softmax}(z_T / \tau), \quad p_S^{(\tau)} = \mathrm{softmax}(z_S / \tau)
\]

\[
L_{teach} = \tau^2 \cdot w(\Delta) \cdot \mathrm{KL}(p_T^{(\tau)} \Vert p_S^{(\tau)})
\]

Recommended initial temperature:

- `tau = 2.0`

### 8.5 Why choice-space KL, not full token KL

Because the active question is **not** full language-model imitation.

The current FEVER shared-injection path controls a very narrow decision channel. What we need is:

- dense,
- stable,
- task-relevant,
- cheap,
- easy-to-interpret

alignment.

Three-way choice-space KL fits that question better than full token-distribution distillation.

### 8.6 Optional fallback if KL is numerically unstable

If KL causes instability or excessive overconfidence, add a switchable alternative:

- `teacher_choice_js`

with Jensen–Shannon divergence over the same 3-way distributions.

But **do not** make JS the first canonical attempt. Use KL first because it is simpler and more standard for distillation-style alignment.

---

## 9. Hinge schedule revision

### 9.1 Canonical position

The hinge term remains useful in principle, but in the current FEVER shared-injection setting it is now suspicious enough that its schedule must be softened.

### 9.2 Recommended new default

Change from the current early/strong ramp to a **later and smaller** hinge.

Recommended initial values:

- `pilot_competitor_hinge_weight_max: 0.05`
- `pilot_competitor_hinge_start_step: 12`
- `pilot_competitor_hinge_ramp_steps: 12`

Rationale:

- keep some margin pressure,
- but give CE + anchor + teacher time to shape the latent first,
- reduce the chance that the model discovers the dominant-label shortcut too early.

### 9.3 What not to do

Do **not** remove hinge permanently based on intuition alone.

If needed, use a **single hinge-off audit arm** only when the main comparison is inconclusive.

---

## 10. Anchor schedule

Keep anchor continuation in M5.3.

Recommended default:

- keep the same basic anchor continuation family as in M5.2,
- do **not** overcomplicate it,
- do **not** replace it with new manifold penalties yet.

This preserves a known stabilizer while letting the new teacher term provide direction.

---

## 11. Required M5.3 experimental matrix

To keep this phase decisive but compact, the required matrix is:

| Arm | Purpose | Loss |
|---|---|---|
| `control-safe-hinge` | baseline for new schedule | `CE + smaller/delayed hinge + anchor` |
| `canonical-dense-teacher` | main hypothesis test | `CE + smaller/delayed hinge + anchor + dense teacher KL` |

Optional only if needed:

| Arm | When to run | Loss |
|---|---|---|
| `control-no-hinge-audit` | only if both required arms still collapse too early | `CE + anchor` |
| `freeze-writer-meaningful-init` | only after canonical shows movement or passes | same as canonical, but freeze writer and only adapt allowed downstream pieces |

### Why this matrix

This matrix answers the most important question first:

- does dense teacher alignment improve over the safest reasonable control?

It avoids wasting the immediate M5.3 budget on attribution questions that only matter **after** the canonical path shows progress.

---

## 12. Exact code changes for Workstream A

### 12.1 File: `src/memtotal/training/m4_shared_injection.py`

Add or modify the following.

#### A. Extend alignment aux mode registry

Current modes:

- `off`
- `teacher_margin`

Add:

- `teacher_choice_kl`
- `teacher_choice_js`

#### B. Add helper functions

Recommended new helpers:

- `_score_margin_tensor(...)` (already exists; reuse)
- `_teacher_advantage_weight(...)`
- `_choice_distribution(...)`
- `_alignment_aux_choice_loss(...)`
- `_effective_rank(...)`
- `_class_entropy(...)`

#### C. Refactor `_alignment_aux_loss(...)`

Refactor into a dispatcher that supports:

- `off`
- `teacher_margin`
- `teacher_choice_kl`
- `teacher_choice_js`

Do **not** bolt the new logic into a long if/else without factoring out helper routines.

#### D. Log additional diagnostics

Add per-step logging of at least:

- `teacher_choice_kl`
- `teacher_choice_js`
- `teacher_advantage_weight_mean`
- `teacher_advantage_weight_max`
- `teacher_margin_minus_base_margin`
- `teacher_margin_minus_active_margin`
- `active_class_entropy`
- `teacher_class_entropy`
- `base_class_entropy`
- `memory_slot_effective_rank`
- `support_state_effective_rank`
- `grad_norm_support_encoder`
- `grad_norm_writer`
- `grad_norm_prefix_projector`
- `writer_to_projector_grad_ratio`

#### E. Keep existing logs

Do not remove existing useful logs such as:

- `latent_anchor_loss`
- `anchor_support_cosine`
- `anchor_writer_slot_cosine`
- `teacher_margin_aux_active`
- `prefix_norm` summaries
- dynamics-recovery outputs

### 12.2 File: `configs/exp/`

Create new config files rather than overwriting M5.2 configs.

Recommended names:

- `m5_fever_qwen25_phase2_val_objective_dense_teacher_control_common.yaml`
- `m5_fever_qwen25_phase2_val_objective_dense_teacher_canonical_common.yaml`
- optional: `m5_fever_qwen25_phase2_val_objective_dense_teacher_hinge_off_audit_common.yaml`

### 12.3 File: `scripts/`

Create a new runner script rather than mutating the old M5.2 runner in place.

Recommended name:

- `run_m5_fever_dense_teacher_qwen25.sh`

This should mirror the M5.2 script structure as much as possible:

- warm-start manifest
- split/support bank prep
- per-arm selected suite
- dynamics recovery
- `screen248-test` if selected
- heldout A/B if primary gate passes
- `fixed64` legacy if promoted
- summary JSON + report generation
- rsync to `runs/review` and `results/generated/review`

### 12.4 File: summary scripts / analysis

Either:

- extend the existing summary utilities cleanly, or
- add a new dedicated summary script.

Recommended new files:

- `scripts/update_m5_dense_teacher_summary.py`
- optional helper in `src/memtotal/analysis/m4_shared_injection.py`

The summary should compare **canonical vs control**, not canonical vs the older M5.2 arm names.

### 12.5 Tests

Add tests for:

- mode parsing,
- continuous weight monotonicity,
- KL/JS auxiliary loss shape and finiteness,
- effective-rank calculation,
- summary-script behavior on missing optional gate files,
- checkpoint compatibility for the new config family.

Run:

```bash
python -m unittest discover -s tests -v
```

before promoting the phase.

---

## 13. Recommended initial M5.3 config values

Below is the recommended canonical starting point.

### 13.1 Control arm

```yaml
runtime:
  pilot_alignment_aux_mode: off
  pilot_alignment_aux_weight_max: 0.0
  pilot_alignment_aux_start_step: 0
  pilot_alignment_aux_ramp_steps: 0

  pilot_latent_anchor_weight_start: 0.10
  pilot_latent_anchor_weight_end: 0.02
  pilot_latent_anchor_decay_steps: 16

  pilot_choice_ce_weight: 1.0
  pilot_competitor_hinge_weight_max: 0.05
  pilot_competitor_hinge_start_step: 12
  pilot_competitor_hinge_ramp_steps: 12

  pilot_train_steps: 32
  pilot_snapshot_steps: [0, 2, 4, 8, 12, 16, 24, 32]
```

### 13.2 Canonical dense-teacher arm

```yaml
runtime:
  pilot_alignment_aux_mode: teacher_choice_kl
  pilot_alignment_aux_weight_max: 0.05
  pilot_alignment_aux_start_step: 0
  pilot_alignment_aux_ramp_steps: 8
  pilot_alignment_aux_temperature: 2.0
  pilot_alignment_aux_advantage_center: 0.0
  pilot_alignment_aux_advantage_scale: 0.25
  pilot_alignment_aux_apply_only_to_real_memory: true

  pilot_latent_anchor_weight_start: 0.10
  pilot_latent_anchor_weight_end: 0.02
  pilot_latent_anchor_decay_steps: 16

  pilot_choice_ce_weight: 1.0
  pilot_competitor_hinge_weight_max: 0.05
  pilot_competitor_hinge_start_step: 12
  pilot_competitor_hinge_ramp_steps: 12

  pilot_train_steps: 32
  pilot_snapshot_steps: [0, 2, 4, 8, 12, 16, 24, 32]
```

### 13.3 Notes on these values

- These are **starting values**, not a broad sweep invitation.
- The point is to run **one decisive canonical pair**.
- If these fail, the next question should quickly become architecture, not a long hyperparameter spiral.

---

## 14. M5.3 success criteria and stop criteria

### 14.1 Success criteria

Workstream A counts as a substantive success if:

1. `canonical-dense-teacher` passes `screen248-val` selection,
2. then passes the primary `screen248-test` gate,
3. and outperforms the control in the intended direction.

Minimal intended direction:

- selected canonical checkpoint shows stronger and more stable `I_real > I_shuffle`,
- no trivial support-bank brittleness,
- no immediate dominant-label collapse at or before the main selected region.

### 14.2 Useful partial success

Even if it does not fully pass, treat Workstream A as **informative success** if canonical shows one of the following relative to control:

- sustained real-vs-shuffle separation over more snapshots,
- delayed collapse,
- lower dominant-label fraction,
- better macro-F1 at matched snapshots,
- lower teacher-student KL / margin gap while preserving content sensitivity.

This matters because Workstream B can then reuse the better objective even if FEVER gating is not fully passed.

### 14.3 Hard stop criteria

Do **not** let single-level FEVER objective iteration continue indefinitely.

Stop Workstream A after:

- one canonical control pair,
- plus at most one optional hinge-off audit if the result is truly ambiguous.

Then move on.

### 14.4 Decision table after Workstream A

#### Case A — Canonical passes, control does not

Interpretation: dense teacher alignment is load-bearing.

Action:

- freeze this objective family as the default substrate,
- carry it into Workstream B.

#### Case B — Both fail, but canonical is clearly less collapsed / more content-sensitive

Interpretation: teacher alignment helps, but single-level architecture still bottlenecks.

Action:

- carry the canonical objective into Workstream B,
- shift attention to architecture.

#### Case C — Control and canonical both fail similarly, but hinge-off audit is better

Interpretation: hinge schedule is the main destabilizer.

Action:

- carry the hinge-off or very-light-hinge version into Workstream B,
- do not spend another cycle repairing teacher loss first.

#### Case D — Canonical underperforms control, and all arms collapse equally

Interpretation: dense teacher formulation is not helping enough in the single-level path.

Action:

- stop iterating on single-level objective design,
- move directly to Workstream B with the least-collapsed control objective.

---

# Workstream B — True Two-Level PoC (`TL-PoC`)

## 15. Why Workstream B is now essential

Workstream A is only about the current single-level substrate.

But the actual paper-level MemTOTAL claim is about:

- writing a larger latent `M_long`,
- reading it through multiple adaptable queries,
- compressing into a shorter `M_short`,
- and injecting that into the frozen reasoner.

The repository already contains the relevant components. The project now needs to **activate them in the active harness**, not keep circling inside the single-level FEVER branch.

---

## 16. Workstream B objective

The exact question is:

> Does explicit `M_long -> Reader -> Fuser -> M_short` improve the active shared-injection path relative to the single-level writer-to-projector path, under controlled injection budgets?

This is the **actual architectural proof-of-concept** phase.

---

## 17. Engineering principle for Workstream B

### 17.1 Reuse the existing memory modules

Do **not** write a second custom Reader/Fuser from scratch inside the FEVER branch.

Directly reuse:

- `MemoryReader`
- `MemoryFuser`

from:

- `src/memtotal/models/memory.py`

### 17.2 Minimal integration, not a giant refactor

`m4_shared_injection.py` is already large. Do not perform a giant cleanliness refactor before the first two-level pilot.

Use one of these two options:

#### Preferred short-term option

Add a controlled runtime switch inside the current FEVER shared-injection harness, e.g.:

- `pilot_memory_path_variant: single_level | two_level`

#### Preferred medium-term cleanup (only after the first pilot works)

Extract shared helpers into a common module and split the runtimes.

But that refactor should happen **after** the first positive or informative two-level pilot, not before.

---

## 18. Required architecture change in the FEVER shared-injection path

### 18.1 Current path

```text
support rows
-> structured support-set encoder
-> writer
-> prefix projector
-> frozen Qwen
```

### 18.2 New two-level path

```text
support rows
-> structured support-set encoder
-> writer = M_long
-> reader(context = prompt summary)
-> fuser = M_short
-> prefix projector
-> frozen Qwen
```

### 18.3 Shape contract

Recommended initial shape family:

- `writer.memory_slots = 8`  → `L = 8`
- `reader.num_queries = 4`   → `H = 4`
- `fuser.short_slots = 8` or `4` depending on the subphase
- projector consumes `K = short_slots`, not necessarily `L`

### 18.4 Key runtime implication

The prefix projector must now accept **the number of short slots**, not blindly assume the number of writer slots.

That means the active FEVER runtime must support:

- `memory_long` length `L`,
- `memory_short` length `K`,
- projector prefix token count bound to `K`.

This is mandatory if the two-level path is to test actual compression.

---

## 19. Reader context policy

### 19.1 Canonical choice

For the first two-level FEVER PoC, use **candidate-independent prompt context**, not candidate-conditioned reading.

Recommended context:

- a pooled summary of the FEVER prompt / claim text,
- computed once per example and reused.

### 19.2 Why

This preserves continuity with the current shared-injection line:

- one shared latent memory per example,
- no candidate-conditioned readout yet,
- no extra attribution confusion.

### 19.3 What not to do yet

Do **not** introduce:

- candidate-conditioned Reader queries,
- pair-conditioned readouts,
- candidate-specific prefixes,

in the first two-level PoC.

That is a later extension only if the shared two-level path already works.

---

## 20. Two-level subphases

Workstream B should itself be staged.

## 20.1 Subphase B0 — Bridge insertion without compression

### Objective

Answer:

> Can the active FEVER harness tolerate the Reader/Fuser path at all, before shortening the injected memory?

### Setup

Compare:

- `SL-8`: current best single-level substrate (8 injected slots)
- `TL-H4-K8`: two-level path with 4 queries and 8 short slots

This keeps the injected slot budget unchanged while inserting explicit read/fuse structure.

### Why B0 matters

If `TL-H4-K8` is worse than `SL-8` immediately, the project first needs to debug integration and context/readout behavior before making claims about bottleneck compression.

## 20.2 Subphase B1 — Bottleneck test

### Objective

Answer:

> Does explicit write-long/read-short help when we actually compress the injected memory?

### Setup

Compare:

- `TL-H4-K8`
- `TL-H4-K4`

This isolates the effect of reducing the injected budget from 8 to 4 while keeping the same Reader design.

## 20.3 Subphase B2 — Multi-query specialization test

### Objective

Answer:

> Do multiple queries matter beyond the existence of a two-level bottleneck?

### Setup

Compare:

- `TL-H1-K4`
- `TL-H4-K4`

This makes the multi-query factor explicit.

### Note

`H=1` is intentionally a weak ablation. It is not meant to be optimal; it is meant to answer whether multiple read heads are load-bearing.

---

## 21. Required implementation changes for Workstream B

### 21.1 File: `src/memtotal/training/m4_shared_injection.py`

Add controlled support for a two-level path.

Recommended additions:

- runtime flag: `pilot_memory_path_variant`
- runtime flag: `pilot_reader_context_mode`
- runtime flag: `pilot_reader_num_queries` (or use method config)
- runtime flag: `pilot_fuser_short_slots` (or use method config)
- runtime flag: `pilot_projector_token_source: writer_slots | short_slots`

### 21.2 Runtime construction

When `pilot_memory_path_variant == two_level`:

- instantiate `MemoryReader`
- instantiate `MemoryFuser`
- build `memory_long = writer.write(...)`
- obtain `reader_context` from prompt summary
- call `reader.read(memory_long, context=reader_context)`
- call `fuser.fuse(readouts)` to obtain `memory_short`
- feed `memory_short` to the existing projector

### 21.3 Prefix projector compatibility

Important: the current prefix projector weights are per-slot and do not inherently depend on the slot count in their parameter shapes. Therefore, warm-starting projector weights remains plausible **even when `K != L`**, as long as runtime shape validation is updated accordingly.

That said, the runtime must:

- validate slot counts against the actual tensor being fed,
- not conflate `writer_memory_slots` with `projector_prefix_tokens`.

### 21.4 Checkpoint loading policy

For the first two-level PoC:

- warm-start `support_encoder`, `writer`, and `prefix_projector` from the best available single-level checkpoint,
- initialize `Reader` and `Fuser` fresh unless a compatible pre-trained query/fuser init is already proven safe,
- keep the loading logic explicit and fail-fast on shape mismatches.

### 21.5 Do not overcomplicate initialization yet

For the first two-level FEVER PoC, do **not** begin with a complicated Stage-B meta-trained Reader/Fuser initialization unless compatibility has already been verified.

The first question is whether the architecture helps at all in the active harness.

---

## 22. Recommended config family for Workstream B

Create new method/config variants rather than mutating old ones.

### 22.1 Method configs

Recommended new files:

- `configs/method/memory_bootstrap_transformer_real_twolevel_h4_k8.yaml`
- `configs/method/memory_bootstrap_transformer_real_twolevel_h4_k4.yaml`
- `configs/method/memory_bootstrap_transformer_real_twolevel_h1_k4.yaml`

Each should keep:

- writer slots `L = 8`
- reader residual scale from the repo’s canonical memory method family
- fuser arch defaulting to `resampler`

### 22.2 Experiment configs

Recommended new files:

- `configs/exp/tl_poc_fever_qwen25_bridge_h4_k8.yaml`
- `configs/exp/tl_poc_fever_qwen25_bottleneck_h4_k4.yaml`
- `configs/exp/tl_poc_fever_qwen25_specialization_h1_k4.yaml`

If Workstream A produced a winning objective, these should inherit that objective family.

---

## 23. Diagnostics required for Workstream B

This workstream must not rely only on top-line accuracy.

Add the following diagnostics.

### 23.1 Memory-level diagnostics

- `memory_long_effective_rank`
- `memory_short_effective_rank`
- per-slot norm summaries for both `M_long` and `M_short`
- compression ratio bookkeeping (`L`, `H`, `K`)

### 23.2 Reader diagnostics

- query attention maps over `M_long`
- per-query attention entropy
- pairwise similarity between query attention distributions
- slot coverage (how many writer slots receive nontrivial attention)
- gate values if reader gating is ever enabled later

### 23.3 Fuser diagnostics

- short-slot norm summary
- pairwise cosine between short slots
- collapse indicators across short slots

### 23.4 Task-level diagnostics

Keep all existing FEVER comparison diagnostics:

- `I_real`
- `I_shuffle`
- `I_zero`
- macro-F1
- dominant-label fraction
- regressions vs base
- selection summary

---

## 24. Success criteria for Workstream B

### 24.1 Strong success

Any of the following counts as strong architectural evidence:

1. `TL-H4-K8` beats `SL-8`, or is clearly more stable under the same injection budget.
2. `TL-H4-K4` matches or beats `TL-H4-K8` while using fewer injected slots.
3. `TL-H4-K4` beats `TL-H1-K4`, and the query diagnostics show nontrivial specialization.

### 24.2 Medium success

Counts as medium success if:

- two-level variants do not yet pass the gate,
- but they delay collapse, improve content sensitivity, or reduce collapse while lowering prefix budget.

### 24.3 Failure interpretation

If all two-level variants fail, use diagnostics to separate:

#### Failure mode B-1 — memory-side capacity / geometry problem

Signals:

- low effective rank in `M_long`,
- short slots collapse immediately,
- H=4 offers no specialization over H=1,
- attention maps are uniform or degenerate.

Action:

- improve memory-side capacity / regularization before touching the receiver.

#### Failure mode B-2 — receiver bottleneck problem

Signals:

- `M_long` has healthy rank,
- H=4 shows differentiated attention,
- `M_short` remains nontrivial,
- but task behavior still gets flattened downstream.

Action:

- receiver-side fallback becomes admissible later.

---

# Workstream C — Transfer / meta-learning evidence refresh

## 25. Why Workstream C matters

Even if the FEVER shared-injection path works, that still does not prove the core MemTOTAL claim.

The paper-level claim is about:

- a reusable Writer,
- adaptable Reader queries,
- query-side transfer/few-shot adaptation,
- cross-domain robustness / CDMI mitigation.

The repository already has Stage A/B/C logic for this. The plan should therefore return to that path once the active shared-injection branch is no longer ambiguous.

---

## 26. Workstream C objective

The key question is:

> Once the two-level architecture is stabilized, can the project show that a mostly fixed Writer plus query-side (and optionally fuser-side) adaptation transfers better than alternatives under matched budgets?

---

## 27. Canonical transfer stance

### 27.1 Keep Writer fixed by default

The default claim should remain:

- **Writer fixed**,
- adapt **queries only** as the primary story,
- adapt **queries + fuser** as the secondary stronger variant,
- use **writer-only** as a diagnostic ablation, not the main claim.

This aligns with the repo’s Stage C target family:

- `q_only`
- `w_only`
- `w_plus_q`

### 27.2 Meta-learning stance

Stay close to the repo’s existing Stage B logic:

- query/fuser-side meta-training,
- ANIL-like feature reuse interpretation,
- keep the backbone frozen,
- do not jump to full MAML complexity unless absolutely necessary.

---

## 28. Workstream C execution order

## 28.1 C0 — Interface compatibility refresh

Before any big transfer claim:

- ensure the positive two-level FEVER architecture uses the same module sizes as the Stage B/C configs,
- ensure checkpoints can be exported/reloaded cleanly,
- ensure Stage C can load the same Writer/Reader/Fuser family without brittle manual surgery.

## 28.2 C1 — Core4 smoke refresh with the stabilized two-level family

Use the repo’s existing `core4` setup as the first transfer refresh.

Canonical current domain family in repo artifacts:

- `math`
- `code`
- `qa`
- `narrative`

Canonical source/target split already present in repo:

- source: `{math, code, qa}`
- target: `narrative`

Refresh Stage B and Stage C under the stabilized two-level family.

## 28.3 C2 — Near-domain control

Add at least one nearer transfer setup, for example by rotating target domain among the less distant subsets, while keeping budget matched.

Purpose:

- establish a near-vs-far transfer axis,
- support later CDMI analysis.

## 28.4 C3 — Full few-shot curves

Once the refreshed two-level family is stable:

- rerun Stage C shot curves,
- rerun step curves,
- report `q_only` as canonical,
- keep `w_only` and `w_plus_q` as attribution lines.

---

## 29. Required comparisons for Workstream C

At minimum, compare:

1. `q_only` — canonical claim
2. `w_only` — diagnostic
3. `w_plus_q` — stronger but less minimal adaptation
4. prompt/deep-prompt baseline family from the repo’s baseline suite
5. LoRA / IA3-style PEFT baselines where already available in the repo baseline program

### Important reporting rule

Whenever possible, report:

- parameter count updated,
- shot budget,
- step budget,
- target-domain performance,
- positive-gain rate across seeds,
- compute/runtime notes if available.

This keeps the comparison fair and paper-credible.

---

## 30. Success criteria for Workstream C

Counts as success if the stabilized two-level family shows:

1. positive query-only adaptation under the repo’s Stage C evaluation contract,
2. stronger or more sample-efficient few-shot transfer than the main baseline families,
3. a credible near/far transfer narrative,
4. evidence that query-side adaptation is doing real work rather than being a decorative extra.

Strongest preferred story:

- `q_only` is already competitive or clearly efficient,
- `w_plus_q` offers an upper bound,
- the gap between zero-shot and adapted target performance is meaningfully positive.

---

# Workstream D — CDMI and benchmark expansion

## 31. Why CDMI must stay central

CDMI is not a cosmetic add-on. It is part of the project’s locked claim.

The paper story is stronger if it can say:

- the Writer is broadly reusable,
- but domain-specific readout is still needed,
- and that need becomes especially visible when a shared memory system spans far-apart domains.

---

## 32. Operational CDMI definition for this plan

For target domain `d_t`, define an interference gap relative to a matched-budget near/target-proximal reference.

One usable operational form is:

\[
IG(d_t) = \mathrm{Perf}(d_t \mid \text{mixed-source init}) - \mathrm{Perf}(d_t \mid \text{target-proximal init})
\]

Where:

- `mixed-source init` uses the broad shared Writer / meta-readout setup,
- `target-proximal init` is a matched-budget reference using a closer or target-only training exposure.

CDMI mitigation means making this gap **less negative** after Reader-side adaptation.

### Recommended reporting

Report both:

- pre-adaptation interference gap,
- post-adaptation interference gap.

That shows whether adaptable reading actually repairs cross-domain interference.

---

## 33. Benchmark roles

Use benchmarks according to what they are good for.

### 33.1 FEVER

Role:

- substrate test,
- controlled shared-injection diagnostics,
- fast gating.

Not the final reasoning-memory benchmark.

### 33.2 Story Cloze / ROCStories

Role:

- cheap narrative bridge,
- useful target domain in the current `core4` smoke family.

Warning:

- do not overclaim from it alone,
- story-ending / author-style bias is known in the literature.

### 33.3 NarrativeQA

Role:

- stronger narrative comprehension benchmark,
- better for deeper narrative memory/comprehension claims than Story Cloze alone.

### 33.4 MemoryAgentBench

Role:

- later-stage memory-agent benchmark,
- useful only after the two-level memory path is already stable in simpler settings.

### 33.5 GSM8K / code / QA benchmarks already in repo narrative

Role:

- source domains for transfer and CDMI structure,
- eventually part of the full paper table.

---

## 34. Recommended CDMI experiment order

### D0 — Use current core4 as the first CDMI scaffold

Keep the repo’s current scaffold as the first near/far story:

- source = `{math, code, qa}`
- far target = `narrative`

### D1 — Add one nearer target rotation

Rotate one of the non-narrative domains into the target role to produce a closer transfer/control setting.

### D2 — Report gap before and after query adaptation

For each target domain:

- zero-shot / no adaptation,
- query-only adaptation,
- optionally query+fuser adaptation.

### D3 — Compare against at least one baseline family

At minimum compare against one matched-budget baseline without the same adaptable readout story.

This could be:

- soft/deep prompt baseline,
- LoRA/IA3 baseline,
- MemGen-like reference path if the repo contract already supports it.

---

# Workstream E — Conditional receiver-side fallback

## 35. When receiver adaptation is allowed

Receiver adaptation is **not** the next default step.

It becomes admissible only if:

1. Workstream A has been completed,
2. Workstream B has been completed,
3. diagnostics suggest the memory path is nontrivial,
4. and the remaining bottleneck looks genuinely receiver-side.

---

## 36. Decision rule for receiver fallback

### Receiver-side fallback is justified if all of the following hold

- the injected path shows some real/shuffle separation at some stage,
- `M_long` has nontrivial effective rank,
- multi-query Reader shows meaningful specialization,
- `M_short` is not obviously collapsed,
- teacher-text path remains strong,
- yet the frozen receiver still flattens the effect downstream.

### Receiver-side fallback is **not** justified if any of the following hold

- memory effective rank is very low,
- all queries behave the same,
- short slots collapse immediately,
- no content sensitivity survives even before the receiver sees the prefix.

In that case, the problem is still memory-side.

---

## 37. Canonical receiver fallback design

If receiver adaptation becomes necessary, start with **tiny LoRA** on carefully chosen receiver submodules.

### 37.1 Recommended first fallback

- apply LoRA only to selected `k_proj` / `v_proj` modules,
- only on the same deep-prefix layers or the most relevant attention layers,
- keep rank tiny (`r = 4` or `8`),
- keep the backbone otherwise frozen,
- do not simultaneously add large MLP adapters.

### 37.2 Why LoRA first

- implementation is straightforward,
- attribution is still reasonably interpretable,
- it is a standard PEFT fallback,
- it directly targets how the receiver consumes the injected information.

### 37.3 IA3 option

If IA3 is easier to implement in the current code path or preferred for parameter efficiency, it is an acceptable alternative. But do not introduce both LoRA and IA3 at the same time in the first fallback comparison.

---

# 38. Concrete file-by-file work plan

This section converts the strategy into engineering edits.

## 38.1 Workstream A file map

### Core code

- `src/memtotal/training/m4_shared_injection.py`
  - add dense teacher aux modes
  - add sample weighting
  - add KL/JS computation
  - add extra diagnostics

### Analysis / summary

- `src/memtotal/analysis/m4_shared_injection.py`
  - add dense-teacher comparison summary helpers
- `scripts/update_m5_dense_teacher_summary.py`

### Configs

- `configs/exp/m5_fever_qwen25_phase2_val_objective_dense_teacher_control_common.yaml`
- `configs/exp/m5_fever_qwen25_phase2_val_objective_dense_teacher_canonical_common.yaml`
- optional audit config

### Scripts

- `scripts/run_m5_fever_dense_teacher_qwen25.sh`

### Tests

- add/extend tests for new aux modes and summaries

## 38.2 Workstream B file map

### Core code

- `src/memtotal/training/m4_shared_injection.py`
  - add `two_level` runtime path
  - instantiate `MemoryReader` and `MemoryFuser`
  - add prompt-summary reader context
  - route projector input through `M_short`

### Configs

- new method configs for `H4/K8`, `H4/K4`, `H1/K4`
- new experiment configs for bridge/bottleneck/specialization pilots

### Scripts

- `scripts/run_tl_poc_fever_qwen25.sh`
- `scripts/update_tl_poc_summary.py`

### Analysis

- add query-attention and compression diagnostics

### Tests

- shape-contract tests for `L != K`
- checkpoint compatibility tests
- prompt-context cache tests if added

## 38.3 Workstream C file map

### Config / scripts

Reuse as much as possible from existing Stage B/C infrastructure:

- `scripts/run_m3_core4_stage_b_probe_suite.sh`
- `scripts/run_m3_core4_stage_c_curve_suite.sh`
- `scripts/run_m3_core4_stage_c_probe_suite.sh`
- related Stage C sweep/audit scripts

### Needed updates

- ensure the positive two-level family is represented in method configs used by Stage B/C
- ensure checkpoint loading remains compatible
- update docs/report bundle after refresh

## 38.4 Workstream D/E file map

- mostly config and analysis additions,
- minimal code changes unless receiver adapters are truly activated.

---

# 39. Suggested run order (strict)

This run order is important.

## Step 1 — Implement and run Workstream A

Required:

1. dense teacher aux modes
2. new control + canonical configs
3. new runner + summary script
4. tests
5. one decisive M5.3 run family

## Step 2 — Freeze the best substrate objective family

After Workstream A, choose exactly one of:

- dense-teacher canonical,
- safe-hinge control,
- hinge-off audit,

as the default substrate objective for Workstream B.

## Step 3 — Implement Workstream B bridge (`TL-H4-K8`)

Do this before real compression.

## Step 4 — Run Workstream B bottleneck (`TL-H4-K4`)

Do this only after the bridge path is functionally sane.

## Step 5 — Run Workstream B specialization (`TL-H1-K4` vs `TL-H4-K4`)

Only after the bottleneck path is alive.

## Step 6 — Refresh Stage B/C transfer evidence

Only after a two-level FEVER PoC exists.

## Step 7 — Move to CDMI experiments

Only after transfer evidence exists.

## Step 8 — Receiver fallback only if diagnostics justify it

Never earlier.

---

# 40. Practical runbook templates

These are templates, not exact commands to copy without adjusting paths.

## 40.1 Workstream A

```bash
# 1) tests
python -m unittest discover -s tests -v

# 2) dense teacher run
./scripts/run_m5_fever_dense_teacher_qwen25.sh \
  <seed> \
  <run_root> \
  <result_root> \
  <phase0_metrics> \
  <resume_stage_b_root> \
  <warm_start_checkpoint>
```

## 40.2 Workstream B

```bash
# 1) bridge path (TL-H4-K8)
./scripts/run_tl_poc_fever_qwen25.sh \
  --config configs/exp/tl_poc_fever_qwen25_bridge_h4_k8.yaml \
  --seed <seed> \
  --warm_start <best_substrate_ckpt>

# 2) bottleneck (TL-H4-K4)
./scripts/run_tl_poc_fever_qwen25.sh \
  --config configs/exp/tl_poc_fever_qwen25_bottleneck_h4_k4.yaml \
  --seed <seed> \
  --warm_start <best_substrate_ckpt>

# 3) specialization (TL-H1-K4)
./scripts/run_tl_poc_fever_qwen25.sh \
  --config configs/exp/tl_poc_fever_qwen25_specialization_h1_k4.yaml \
  --seed <seed> \
  --warm_start <best_substrate_ckpt>
```

## 40.3 Workstream C

```bash
# Refresh Stage B
./scripts/run_m3_core4_stage_b_probe_suite.sh

# Refresh Stage C curves/probes with stabilized two-level family
./scripts/run_m3_core4_stage_c_probe_suite.sh
./scripts/run_m3_core4_stage_c_curve_suite.sh
```

---

# 41. Risks and anti-patterns

## 41.1 Anti-pattern: another long single-level objective loop

Do not let the project spend multiple more cycles on:

- M5.3,
- M5.4,
- M5.5,
- more tiny objective rewrites,

without moving into the true two-level path.

One decisive single-level pass is enough.

## 41.2 Anti-pattern: calling FEVER success the whole paper

Even if FEVER passes, that is still only:

- substrate confirmation,
- architecture staging,
- not the full paper claim.

## 41.3 Anti-pattern: introducing candidate-conditioned readout too early

This would add a new confound before the shared two-level path has been settled.

## 41.4 Anti-pattern: jumping to bigger models before attribution is clean

Qwen3-8B is useful later, but not as a substitute for answering the current causal questions.

## 41.5 Anti-pattern: overclaiming from Story Cloze

Story Cloze is useful as a narrative bridge benchmark, but known bias concerns mean it should not be the sole narrative proof.

## 41.6 Anti-pattern: using receiver adapters as a band-aid

Receiver adaptation must remain a conditional fallback, or it will erase the project’s ability to say whether the memory mechanism itself works.

---

# 42. Documentation update policy after each workstream

After each completed workstream:

1. update the relevant run summary JSON/MD,
2. update the high-level comparison summary,
3. add a short diagnosis note to docs,
4. rebuild `docs_review_bundle.zip`,
5. keep `runs/review` and `results/generated/review` synchronized,
6. update `TODO_LIST.md` status lines,
7. only update `MAIN_IDEA.md` if the conceptual claim changes.

---

# 43. Recommended final project narrative if the plan works

If the plan succeeds, the strongest final story is:

1. **Substrate established:** frozen Qwen can consume learned shared latent prefixes, but the path is optimization-fragile in a single-level design.
2. **Objective clarified:** dense task-space teacher alignment and/or safer objective scheduling stabilize the substrate better than sparse margin hooks.
3. **Architecture validated:** explicit `M_long -> Reader -> M_short` improves stability and/or efficiency relative to the single-level path.
4. **Transfer validated:** fixed Writer + adaptable Reader queries/fuser transfer across domains under matched budgets.
5. **CDMI validated:** adaptable readout reduces cross-domain interference, especially on far-domain targets.
6. **Fallback contained:** receiver-side adaptation is only used when diagnostics specifically justify it.

That is a coherent paper story.

---

# 44. Recommended references for design justification

These are not the only relevant papers, but they are enough to anchor the current plan.

## Prompt / prefix / deep prompt tuning

- Li, X. L., & Liang, P. (2021). **Prefix-Tuning: Optimizing Continuous Prompts for Generation**. ACL.  
  https://aclanthology.org/2021.acl-long.353/

- Liu, X. et al. (2021/2022). **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**.  
  https://arxiv.org/abs/2110.07602

## PEFT / receiver-side fallback

- Hu, E. J. et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**.  
  https://openreview.net/forum?id=nZeVKeeFYf9

- Liu, H. et al. (2022). **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning**.  
  https://papers.nips.cc/paper_files/paper/2022/file/0cde695b83bd186c1fd456302888454c-Paper-Conference.pdf

## Adaptation stability / objective design

- Kumar, A. et al. (2022). **Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution**.  
  https://iclr.cc/virtual/2022/poster/5945

- Wang, Y. et al. (2024). **MiniLLM: Knowledge Distillation of Large Language Models**.  
  https://iclr.cc/virtual/2024/poster/19420

## Meta-learning / feature reuse

- Finn, C., Abbeel, P., & Levine, S. (2017). **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**.  
  https://proceedings.mlr.press/v70/finn17a.html

- Raghu, A. et al. (2020). **Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML**.  
  https://openreview.net/pdf?id=rkgMkCEtPB

## Benchmark references

- Thorne, J. et al. (2018). **FEVER: a Large-scale Dataset for Fact Extraction and Verification**.  
  https://aclanthology.org/N18-1074/

- Mostafazadeh, N. et al. (2016). **A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories** (Story Cloze / ROCStories).  
  https://aclanthology.org/N16-1098.pdf

- Kočiský, T. et al. (2018). **The NarrativeQA Reading Comprehension Challenge**.  
  https://aclanthology.org/Q18-1023/

- Hu, Y. et al. (2025). **MemoryAgentBench: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions**.  
  https://arxiv.org/abs/2507.05257

## Memory / reasoning context for later paper positioning

- Zhang, J. et al. (2025). **LightThinker: Thinking Step-by-Step Compression**.  
  https://arxiv.org/abs/2502.15589

- Zhou, Z. et al. (2025/2026). **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents**.  
  https://arxiv.org/abs/2506.15841

---

# 45. Final directive

If only one sentence survives from this plan, it should be this:

> Run **one decisive dense-teacher M5.3** to settle the single-level substrate, then stop circling in FEVER-only objective tweaks and **activate the true two-level Writer → Reader → Fuser path** that MemTOTAL already has in the repo.

