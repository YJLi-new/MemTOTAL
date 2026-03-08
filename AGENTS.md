# AGENTS.md — 给 Codex / Agent 的长期指引（本仓库根目录）

> 本文件是 **Codex/Agent 的地图，不是百科全书**。  
> 它只负责四件事：**入口导航、执行循环、研究红线、交付格式**。  
> 更深的知识请写入并维护在：`PLANv2.md`、`MAIN_IDEA.md`、`EXPERIMENTS_INFO.md`、`docs/exec-plans/` 与 `docs/`。

---

## 0) 一句话 TL;DR（永远先做这件事）

**任何任务开始之前：先打开并严格按 `PLANv2.md` 执行。**

- `PLANv2.md`：唯一入口 / 当前执行计划 / DoD / 产物要求 / 里程碑顺序
- `MAIN_IDEA.md`：方法定义 / 训练阶段 / 核心假设 / 与 MemGen 的关键差异
- `EXPERIMENTS_INFO.md`：实验协议 / baseline / 主表与图 / 统计规范 / 论文产物口径

> 规则：  
> 1. **不要自行压缩或删除**这三份文档中的关键信息。  
> 2. 允许重排结构、加目录、补交叉链接、改措辞；不允许删掉关键实验、对照、DoD、脚本入口或论文主张。  
> 3. 若文档冲突：**方法口径以 `MAIN_IDEA.md` 为准；实验口径以 `EXPERIMENTS_INFO.md` 为准；执行顺序与验收以 `PLANv2.md` 为准。**

---

## 1) 文档分工（先分工，再做事）

### `PLANv2.md` —— 单一权威 runbook / backlog
所有任务都从这里开始。它负责：
- 里程碑顺序（M0–M6）
- P0/P1/P2 优先级
- 每项任务的 Definition of Done（DoD）
- 何时必须去读 `MAIN_IDEA.md` / `EXPERIMENTS_INFO.md`
- harness-first 约束：ExecPlan、可复现、结果治理、交付格式

### `MAIN_IDEA.md` —— 方法与论文叙事的规格说明
当任务涉及下面任一项时，必须读取：
- Writer / Reader / Fuser / Injector 的定义与接口
- `M_long / M_short / queries / segment` 的符号与形状
- 训练阶段（Stage A/B/C）与冻结/适配策略
- 核心论文主张：  
  1) 通用 Writer  
  2) 可 meta-train → few-shot 适配的 Reader queries  
  3) 写长读短  
  4) CDMI（跨大域记忆干扰）缓解

### `EXPERIMENTS_INFO.md` —— 实验协议与论文产物口径
当任务涉及下面任一项时，必须读取：
- 主表、跨域泛化、few-shot 曲线、CDMI、持续学习、效率、机制分析
- baseline 清单与预算对齐
- 任务套件、指标、统计规范
- 图/表/CSV/TEX 的最终产出格式

---

## 2) AGENTS 的定位：只做“地图”，不做“百科”

本文件刻意保持短小。不要把它写成超长手册。长期知识应进入：
- `PLANv2.md`：任务分解、DoD、优先级、runbook
- `MAIN_IDEA.md`：方法定义、假设、论文叙事
- `EXPERIMENTS_INFO.md`：实验矩阵、baseline、统计、图表
- `docs/exec-plans/`：多小时任务的执行计划与接力状态
- `docs/`：架构、golden principles、设计决策、质量追踪、专题说明

> 任何“重要但会被反复用到”的知识，都应被写回 repo 并版本化，而不是留在聊天记录或人脑里。

---

## 3) 标准执行循环（必须按这个走）

### 3.1 Ask → Plan → Code → Validate → Record → PR

1. **Ask / Understand**  
   先在 `PLANv2.md` 里定位对应里程碑、条目与 DoD。  
   不允许脱离 PLAN “凭感觉”做事。

2. **Plan**  
   - 若任务预计 ≤30–60 分钟且改动局部：可直接实现，但最终输出仍要包含复现与验证。  
   - 若任务预计 >60 分钟、跨多个模块、或需要探索：先写 mini-plan。  
   - 若任务是 multi-hour / 跨里程碑 / 可能中断：必须写 **ExecPlan**。

3. **Code**  
   小步提交、保持可 review、可回滚。  
   优先短 PR；不要一次性堆大改动。

4. **Validate**  
   必须跑测试 / eval / sanity plot / 汇总脚本。  
   “代码能跑”不等于“任务完成”。  
   如果结果要进论文，必须验证：预算、seed、split、脚本入口、汇总路径都正确。

5. **Record**  
   把新知识写回 repo：文档、注释、脚本、测试、配置、决策日志。  
   不要把关键信息留在聊天记录里。  
   如果一个问题重复出现两次，优先把它升级成脚本 / 测试 / lint / 结构检查，而不是继续口头提醒。

6. **PR / Deliver**  
   交付时必须写清：
   - 完成了哪个 PLANv2 条目
   - 改了哪些文件
   - 复现命令
   - 验证命令
   - 结果与产物路径
   - 已知问题与下一步

### 3.2 ExecPlan 约定（多小时任务必须写）

满足任一条件必须写 ExecPlan：
- 预计 >2 小时
- 跨多个里程碑 / 模块
- 需要 agent / 机器之间接力
- 需要多轮试错或大量配置探索

路径约定：
- `docs/exec-plans/active/<YYYYMMDD>-<short-name>.md`

最低内容：
- Purpose
- Context
- Plan of Work
- Concrete Steps
- Validation & Acceptance
- Progress（带时间戳）
- Decision Log
- Surprises & Discoveries

> 计划必须 **self-contained**：只靠计划文件 + 当前 working tree，另一个 agent 也能继续。

---

## 4) Harness-first 原则（遇到失败时先修系统，不要盲目重试）

### 4.1 Repo 是 system-of-record
对 agent 来说，运行时看不见的知识就等于不存在。  
因此：
- 重要知识必须在 repo 内可发现、可版本化、可链接
- 关键 split、预算、路径、决策、例外、坑点必须写进 markdown / yaml / 脚本
- 不要把关键上下文只留在聊天、口头讨论、临时笔记里

### 4.2 Agent legibility is the goal
你写的代码、脚本、配置、日志、表格，不只是给人看，也要给未来的 agent 看。  
优先选择：
- 稳定、明确、可在 repo 内推理的依赖与抽象
- 单命令可运行的训练/评测/分析入口
- 结构化日志、稳定命名、清晰目录、严格输出约定

### 4.3 缺能力时，先补 harness
当任务失败时，默认不要“再试一次”。先问：
- 缺的是哪种能力？
- 这能力能否通过脚本、测试、lint、文档、结构约束变成 repo 的长期能力？

典型修复方向：
- 缺路径/依赖 → 补 `setup_*.sh`
- 缺统一评测 → 补 `run_eval.py` / `report.py`
- 缺结果治理 → 补 run 目录契约 / 自动汇总 / CI 检查
- 缺重复规范 → 补 custom lint / 结构测试 / golden principles

### 4.4 严边界、宽内部
严格约束：
- 文档口径
- 任务顺序与 DoD
- 预算对齐
- 结果治理
- run 命名、产物结构、复现命令

在这些边界之内，允许局部实现自由。  
目标不是“符合某个人类审美”，而是**正确、可维护、可复现、对未来 agent 也可读**。

### 4.5 吞吐高时，优先短 PR 与 fix-forward
- 保持 PR 短小、寿命短、可快速 review 与回滚
- 对 infra / 脚本 / 文档 / 小型 refactor，倾向 fix-forward
- 但任何会影响**论文数字、数据切分、预算公平性、主要结论**的改动，必须先验证再合并

### 4.6 做“垃圾回收”（garbage collection）
agent 会复制 repo 里已有模式，包括坏模式。  
因此：
- 持续清理 stale 配置、过时脚本、重复 helper、失效文档
- 把 recurring review comment 升级为机械化规则
- 维护 `docs/` 中的 golden principles / quality tracker / tech-debt tracker（若 PLANv2 要求）

---

## 5) 研究导向的硬性红线（不能违反）

### 5.1 论文主张红线
本项目最终论文的核心卖点固定为四条，不得在执行中被稀释或改写为别的故事：

1. **Writer 通用**：`M_long` 的生成器在 general field 上训练，一次训练，多域复用  
2. **Reader 可快速适配**：queries 经 meta-train 后，对新域 few-shot / 少步数适配  
3. **写长读短**：高容量写入、低带宽注入  
4. **CDMI 缓解**：面对 math ↔ narrative 这类大域差时，我们比 MemGen 更稳

### 5.2 口径红线
- 不要把本方法实现成“另一个 MemGen”
- 不要把 meta-learning 偷换成普通多任务训练
- 不要把 CDMI 从主实验里删掉或降级成附录里的小例子
- 不要在没有明确标注的情况下做不公平对比（shots、steps、参数量、训练 token、wall time）

### 5.3 结果治理红线
- **不允许手工抄数**；所有表/图都必须由脚本从 `metrics.json` / `jsonl` 自动汇总
- **不允许只保留最好看的一次 run**；按 `EXPERIMENTS_INFO.md` 的 seed/CI 规范汇总
- **不允许 silently 改 task split / metric / budget**；任何变化必须写回 repo 文档并记录原因

---

## 6) 最小仓库与产物契约（详细以 PLANv2 为准）

### 6.1 目录与知识位置
推荐目录：
- `src/`：核心代码
- `configs/`：训练/评测 YAML
- `scripts/`：单命令入口
- `runs/`：原始实验产物（不要进 git）
- `results/`：汇总表、图、tex、csv
- `docs/`：exec plans、架构、设计与质量文档

### 6.2 所有训练/评测入口都必须支持
- `--config <yaml>`
- `--seed`
- `--output_dir`
- `--resume`（可选但推荐）

### 6.3 每个 run 至少要保存
- config 快照
- seed
- git hash
- metrics（json/jsonl/csv）
- 关键日志
- 关键图表或生成的中间 CSV

### 6.4 单个 run 的定义
一个 **run** = 固定 `{backbone, 方法变体, 任务/域, seed, 关键超参}` 的一次训练或一次完整评测。  
shots × steps 网格应尽量在一个 run 内循环完成，而不是拆成大量零碎 job。

---

## 7) 研究北极星（给 agent 的最小方法摘要）

本项目研究一种面向推理型 LLM/Agent 的内部记忆机制：

- 每个 reasoning segment 写入一个高容量 latent buffer：`M_long`
- 多个 Reader queries 从 `M_long` 读取多个 facet
- 读出结果融合为极短的 `M_short`
- 只有 `M_short` 被注入下一段推理上下文
- Writer 尽量通用；Reader queries 可 meta-train 并对新域快速适配

> 细节与严格定义在 `MAIN_IDEA.md`。  
> 任何实现若涉及接口、训练阶段、冻结/解冻、CDMI 叙事，必须回看 `MAIN_IDEA.md`，不要靠这里的简略描述硬猜。

---

## 8) 验证与交付格式（每次都必须给）

每次阶段性交付必须包含：

1. **完成内容**：对应 PLANv2 条目与简短说明  
2. **修改文件**：文件列表  
3. **复现方式**：命令 + config + seed + output_dir  
4. **验证方式**：tests / eval / plots  
5. **结果**：关键指标 + 产物路径  
6. **已知问题**：失败点、风险、下一步建议  
7. **若有文档更新**：指出更新了哪些 repo 文档，以及为什么必须更新它们

---

## 9) 遇到不确定时怎么做（不要硬猜）

遇到以下情况时，不要编造命令或假设默认设置：
- 数据下载权限 / 路径不明
- benchmark harness 未接好
- 需要大规模训练但显存 / 队列 / 存储不确定
- 文档口径冲突
- baseline 预算对齐不清楚

正确处理流程：
1. 先查 `PLANv2.md`
2. 再查 `MAIN_IDEA.md` / `EXPERIMENTS_INFO.md`
3. 仍不明确：列出 2–3 个方案，写清风险/成本/推荐方案，等待人类决策

---

## 10) 最后提醒（非常重要）

- **先做 paper-critical 的东西**：主表、跨域 few-shot、CDMI、持续学习、效率、机制分析
- **AGENTS 是地图，不是百科**：不要把大量细节回填到这里
- **Repo 是 system-of-record**：重要知识必须版本化写回仓库
- **失败不是“再试一次”**：失败意味着缺脚本、缺测试、缺规则、缺知识入口；优先补 harness
- **如果一个规则重要到会重复提醒，就把它升级成可执行机制**（脚本 / 测试 / lint / 结构检查）
