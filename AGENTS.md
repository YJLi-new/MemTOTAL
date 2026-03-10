# AGENTS.md - Long-Lived Guidance for Codex / Agents (Repo Root)

> This file is the **map for Codex/agents, not an encyclopedia**.
> It is responsible for only four things: **entry-point navigation, the execution loop, research guardrails, and delivery format**.
> Deeper knowledge should be written into and maintained in: `PLANv6.md`, `docs/MAIN_IDEA.md`, `docs/EXPERIMENTS_INFO.md`, `docs/exec-plans/`, and `docs/`.

---

## 0) One-Sentence TL;DR (always do this first)

**Before starting any task, open `PLANv6.md` first and execute against it strictly.**

- `PLANv6.md`: single entry point / current execution plan / DoD / artifact requirements / milestone order
- `docs/MAIN_IDEA.md`: method definition / training stages / core hypotheses / key differences from MemGen
- `docs/EXPERIMENTS_INFO.md`: experiment protocol / baselines / tables and figures / statistical rules / paper-facing artifact standards

> Rules:
> 1. Do **not** compress away or delete critical information from these three documents.
> 2. You may reorganize structure, add a table of contents, improve cross-links, and tighten wording; you may not remove key experiments, controls, DoD items, script entry points, or paper claims.
> 3. If documents conflict: **method authority belongs to `docs/MAIN_IDEA.md`; experiment authority belongs to `docs/EXPERIMENTS_INFO.md`; execution order and acceptance belong to `PLANv6.md`.**

---

## 1) Document Responsibilities (separate responsibilities before doing work)

### `PLANv6.md` - the single authoritative runbook / backlog
Every task starts here. It defines:
- milestone order
- P0/P1/P2 priorities
- the Definition of Done (DoD) for each task
- when you must read `docs/MAIN_IDEA.md` and/or `docs/EXPERIMENTS_INFO.md`
- harness-first constraints: ExecPlans, reproducibility, results governance, and delivery format

### `docs/MAIN_IDEA.md` - method and paper-story specification
Read this whenever the task touches any of the following:
- the definition or interface of Writer / Reader / Fuser / Injector
- the notation and shapes for `M_long / M_short / queries / segment`
- the training stages (Stage A/B/C) and freeze/adaptation strategy
- the core paper claims:
  1. general Writer
  2. Reader queries that meta-train and then adapt in few-shot form
  3. write-long / read-short
  4. CDMI mitigation across large domain gaps

### `docs/EXPERIMENTS_INFO.md` - experiment protocol and paper-facing standards
Read this whenever the task touches any of the following:
- main tables, cross-domain generalization, few-shot curves, CDMI, continual learning, efficiency, or mechanism analysis
- baseline lists and budget alignment
- task suites, metrics, and statistical standards
- the final output format for figures / tables / CSV / TeX

---

## 2) The Role of AGENTS: a map, not an encyclopedia

This file should stay intentionally short. Do not turn it into a giant manual. Long-lived knowledge belongs in:
- `PLANv6.md`: task breakdown, DoD, priorities, runbook
- `docs/MAIN_IDEA.md`: method definition, hypotheses, paper narrative
- `docs/EXPERIMENTS_INFO.md`: experiment matrix, baselines, statistics, figures
- `docs/exec-plans/`: execution plans and relay status for multi-hour work
- `docs/`: architecture, golden principles, design decisions, quality tracking, focused notes

> Any knowledge that is important and likely to be reused should be written back into the repo and versioned, not left in chat history or in a person's head.

---

## 3) Standard Execution Loop (must follow this)

### 3.1 Ask -> Plan -> Code -> Validate -> Record -> PR

1. **Ask / Understand**
   First locate the relevant milestone, item, and DoD in `PLANv6.md`.
   Do not work "by feel" outside the plan.

2. **Plan**
   - If the task is expected to take <=30-60 minutes and is local in scope: you may implement directly, but the final output still needs reproduction and validation.
   - If the task is expected to take >60 minutes, spans multiple modules, or requires exploration: write a mini-plan first.
   - If the task is multi-hour, cross-milestone, or likely to be interrupted: you must write an **ExecPlan**.

3. **Code**
   Keep changes reviewable and reversible.
   Prefer short PRs; do not dump one huge change set at once.

4. **Validate**
   You must run tests / eval / sanity plots / summary scripts.
   "The code runs" does not mean "the task is done."
   If a result is paper-facing, you must verify that the budget, seed, split, script entry point, and summary path are all correct.

5. **Record**
   Write new knowledge back into the repo: docs, comments, scripts, tests, configs, decision logs.
   Do not leave critical knowledge only in chat.
   If the same problem happens twice, prefer upgrading it into a script / test / lint / structural check instead of repeating verbal reminders.

6. **PR / Deliver**
   Every delivery must state clearly:
   - which `PLANv6.md` item was completed
   - which files changed
   - reproduction commands
   - validation commands
   - results and artifact paths
   - known issues and next steps

### 3.2 ExecPlan Conventions (required for multi-hour tasks)

You must write an ExecPlan if any of the following is true:
- expected runtime >2 hours
- spans multiple milestones or modules
- requires a handoff across agents or machines
- needs multiple rounds of trial and error or large config exploration

Path convention:
- `docs/exec-plans/active/<YYYYMMDD>-<short-name>.md`

Minimum content:
- Purpose
- Context
- Plan of Work
- Concrete Steps
- Validation & Acceptance
- Progress (with timestamps)
- Decision Log
- Surprises & Discoveries

> The plan must be **self-contained**: another agent should be able to continue using only the plan file and the current working tree.

---

## 4) Harness-First Principles (when something fails, fix the system before blindly retrying)

### 4.1 The repo is the system of record
For an agent, knowledge that is not discoverable at runtime effectively does not exist.
Therefore:
- important knowledge must be discoverable, versioned, and linkable inside the repo
- key splits, budgets, paths, decisions, exceptions, and traps must be written into markdown / YAML / scripts
- do not leave critical context only in chat, verbal discussion, or temporary notes

### 4.2 Agent legibility is the goal
The code, scripts, configs, logs, and tables you write are not only for humans; they are also for future agents.
Prefer:
- stable, explicit dependencies and abstractions that can be reasoned about from inside the repo
- single-command training / eval / analysis entry points
- structured logs, stable naming, clear directories, and strict output contracts

### 4.3 If capability is missing, patch the harness first
When a task fails, do not default to "try again."
First ask:
- which capability is missing?
- can that capability be turned into long-lived repo ability via a script, test, lint, doc, or structural constraint?

Typical repair directions:
- missing path/dependency -> add `setup_*.sh`
- missing unified eval -> add `run_eval.py` / `report.py`
- missing result governance -> add run-directory contracts / automatic aggregation / CI checks
- missing repeated standards -> add custom lint / structural tests / golden principles

### 4.4 Strict boundaries, flexible internals
Constrain strictly:
- doc authority
- task order and DoD
- budget alignment
- result governance
- run naming, artifact structure, and reproduction commands

Within those boundaries, local implementation freedom is fine.
The goal is not "match one human's aesthetic"; the goal is **correct, maintainable, reproducible, and legible to future agents**.

### 4.5 Under high throughput, prefer short PRs and fix-forward
- Keep PRs short-lived, easy to review, and easy to revert
- For infra / scripts / docs / small refactors, prefer fix-forward
- But for anything affecting **paper numbers, dataset splits, budget fairness, or main conclusions**, validate before merge

### 4.6 Do garbage collection
Agents copy existing repo patterns, including bad ones.
Therefore:
- continuously remove stale configs, outdated scripts, duplicate helpers, and dead docs
- upgrade recurring review comments into mechanical rules
- maintain the golden principles / quality tracker / tech-debt tracker in `docs/` when `PLANv6.md` requires it

### 4.7 GitHub review export is a separate publication surface
- The local working repo may stay full, but the GitHub-facing review branch is a curated publication layer for external review and reproduction.
- The default downloadable GitHub `.zip` for `YJLi-new/MemTOTAL` must stay under `31 MB`.
- Therefore the GitHub review branch should contain:
  - review-facing docs,
  - the minimal code / config / script / test surface needed for external reproduction,
  - governed latest experimental result bundles,
  - no bulky raw traces or run artifacts such as `train_events.json`, `task_case_dump.jsonl`, checkpoints, or full local run directories.
- Do not keep the review branch artificially tiny; use as much of the `31 MB` budget as is useful for review and reproduction while staying safely below the hard cap.
- Maintain the lightweight GitHub export with:
  - `scripts/build_github_review_snapshot.py`
  - `scripts/push_github_review_snapshot.sh`
  - `docs/GITHUB_REVIEW_EXPORT.md`
- Use `gh` to maintain the GitHub-side publication state. The default branch for external download should remain the lightweight `review` branch unless a newer documented policy replaces it.
- When docs or latest governed review artifacts change, refresh and republish the lightweight review branch in sync with the local research repo.

---

## 5) Research-Critical Hard Guardrails (must not be violated)

### 5.1 Paper-claim guardrails
The final paper's core selling points are fixed and must not be diluted or rewritten into a different story:

1. **General Writer**: `M_long` is generated by a writer trained on a general field and then reused across domains
2. **Fast-adapting Reader**: queries meta-train and then adapt to new domains in few-shot / few-step form
3. **Write long, read short**: high-capacity writing with low-bandwidth injection
4. **CDMI mitigation**: across large domain gaps such as math vs narrative, we are more stable than MemGen

### 5.2 Narrative guardrails
- Do not turn this method into "another MemGen"
- Do not relabel meta-learning as ordinary multi-task training
- Do not remove CDMI from the main experiments or demote it into a tiny appendix example
- Do not make unfair comparisons without explicit labeling (shots, steps, parameter count, training tokens, wall time)

### 5.3 Result-governance guardrails
- **No manual number copying**; every table and figure must be aggregated by script from `metrics.json` / `jsonl`
- **Do not keep only the prettiest run**; aggregate according to the seed / CI rules in `docs/EXPERIMENTS_INFO.md`
- **Do not silently change task splits, metrics, or budgets**; every change must be written back into the repo docs with a reason

---

## 6) Minimum Repo and Artifact Contract (details defer to `PLANv6.md`)

### 6.1 Directory and knowledge layout
Recommended layout:
- `src/`: core code
- `configs/`: training / eval YAMLs
- `scripts/`: single-command entry points
- `runs/`: raw experiment artifacts (do not commit)
- `results/`: aggregated tables, figures, TeX, CSV
- `docs/`: exec plans, architecture, design, and quality docs

### 6.2 Every training/eval entry point should support
- `--config <yaml>`
- `--seed`
- `--output_dir`
- `--resume` (optional but recommended)

### 6.3 Every run should save at least
- config snapshot
- seed
- git hash
- metrics (`json/jsonl/csv`)
- key logs
- key figures or generated intermediate CSVs

### 6.4 Definition of a single run
One **run** = one training or one complete eval for a fixed `{backbone, method variant, task/domain, seed, key hyperparameters}`.
Whenever possible, shots x steps grids should be looped within one run instead of fragmented into many tiny jobs.

---

## 7) Research North Star (minimal method summary for agents)

This project studies an internal memory mechanism for reasoning LLMs / agents:

- each reasoning segment writes into a high-capacity latent buffer: `M_long`
- multiple Reader queries read multiple facets from `M_long`
- the readouts are fused into a very short `M_short`
- only `M_short` is injected into the next reasoning context
- the Writer should be as general as possible; the Reader queries should meta-train and then adapt quickly to new domains

> The detailed and strict definitions live in `docs/MAIN_IDEA.md`.
> If an implementation touches interfaces, training stages, freezing/unfreezing, or the CDMI narrative, go back to `docs/MAIN_IDEA.md` instead of guessing from this summary.

---

## 8) Validation and Delivery Format (must be included every time)

Every milestone delivery must include:

1. **Completed work**: the corresponding `PLANv6.md` item and a short explanation
2. **Modified files**: file list
3. **Reproduction**: commands + config + seed + output directory
4. **Validation**: tests / eval / plots
5. **Results**: key metrics + artifact paths
6. **Known issues**: failure points, risks, next-step suggestions
7. **Doc updates**: which repo docs were updated, and why they had to be updated

---

## 9) What to do when things are unclear (do not guess)

When any of the following is unclear, do not invent defaults or commands:
- dataset download permissions or paths
- benchmark harness integration state
- large-scale training requirements when VRAM / queue / storage is uncertain
- document authority conflicts
- baseline budget alignment

Correct handling sequence:
1. Check `PLANv6.md`
2. Then check `docs/MAIN_IDEA.md` and `docs/EXPERIMENTS_INFO.md`
3. If still unclear: list 2-3 options with risks / cost / recommended option, and wait for human direction

---

## 10) Final reminders (important)

- **Do the paper-critical work first**: main tables, cross-domain few-shot, CDMI, continual learning, efficiency, mechanism analysis
- **AGENTS is a map, not an encyclopedia**: do not move large bodies of detail here
- **The repo is the system of record**: important knowledge must be versioned back into the repo
- **Failure does not mean "retry once more"**: it usually means missing scripts, tests, rules, or knowledge entry points; patch the harness first
- **If a rule matters enough to repeat, upgrade it into an executable mechanism** (script / test / lint / structural check)
- **The GitHub default downloadable snapshot is budgeted**: keep the public review branch under `31 MB` and treat that as a hard publication constraint, not a suggestion.
