# TODO_LIST.md — Agent-first Runbook / Backlog（M0–M6）

> 这是本项目的**单一权威执行入口**。  
> 所有任务都从这里开始；任何实现、实验、分析、文档整理，都必须能映射到这里的某个里程碑条目。  
> 若文档之间有冲突：  
> - **方法定义 / 训练阶段 / 核心假设**：以 `MAIN_IDEA.md` 为准  
> - **实验协议 / baseline / 图表口径**：以 `EXPERIMENTS_INFO.md` 为准  
> - **任务顺序 / DoD / 产物要求**：以本文件为准  

---

## 0) 先锁定“论文必须交付什么”（不要先做次要工作）

在进入里程碑之前，先记住：以下六类产物是**论文主干**，不允许被删减或降级到“有空再做”。

### Paper-critical locked deliverables
1. **主结果大表**（Qwen2.5-1.5B-Instruct + Qwen3-8B；ours zero-shot + few-shot；对齐 MemGen 与参数高效适配 baseline）
2. **跨域 few-shot / 少步数适配曲线**
3. **CDMI 主表 + 主图**（Math+Code vs Math+Narrative）
4. **持续学习 / 域冲突结果**（shared queries vs per-domain bank，最好含 hard switch）
5. **性能–成本对照**（inference + adaptation）
6. **机制分析**（至少两张：query specialization、适配前后读法变化）

> 任何任务若无法服务于这六类产物，应降低优先级。

---

## 1) Agent-first 执行协议（workflow contract）

这部分高于具体实现细节。  
你的目标不是“把代码写出来”，而是把 **环境 + 反馈回路 + 可复现产物** 搭起来，让后续工作能被 agent 端到端完成。

### 1.1 先 Ask 再 Code：Plan-first
- **预计 ≤30–60 分钟、改动局部**：可以直接实现，但最终输出仍要包含“做了什么 / 怎么复现 / 怎么验证”
- **预计 >60 分钟，或跨多个模块/里程碑**：先产出 mini-plan
- **预计 multi-hour（数小时）或多天任务**：必须写 **ExecPlan**

### 1.2 ExecPlans（多小时任务的唯一可重启状态）
- 路径约定：`docs/exec-plans/active/<YYYYMMDD>-<short-name>.md`
- 必须 self-contained：只看该计划 + 当前 working tree 就能继续
- 最低字段：
  - Purpose
  - Context
  - Plan of Work
  - Concrete Steps
  - Validation & Acceptance
  - Progress（带时间戳）
  - Decision Log
  - Surprises & Discoveries
- 每次中断前必须更新 Progress

### 1.3 PR / 提交节奏
- 尽量保持短 PR、小步提交、可回滚
- 默认采用：**小改动快合、问题后续修、重复问题升格为规则** 的 fix-forward 风格
- 但任何影响**论文数字、数据切分、预算公平性、主要结论**的改动，必须先验证再合并
- 每个阶段性交付必须包含：
  - 对应 TODO 条目
  - 修改文件
  - 复现命令
  - 验证命令
  - 结果与产物路径
  - 已知问题与下一步

### 1.4 Repo 是唯一系统记录（system-of-record）
- 决策、口径、路径、split、预算、坑点：必须写进 repo
- 不要把关键知识留在聊天记录里
- 反复出错的点应升级成：
  - 脚本
  - 测试
  - lint / 结构检查
  - 文档中的固定规则

### 1.5 环境问题必须脚本化
- `setup_env.sh`
- `setup_data.sh`
- 单命令训练 / 评测 / 画图
- 遇到依赖或路径问题，不要反复“再试一次”，应补脚本与文档

### 1.6 agent legibility（对 agent 可读）必须被显式建设
本项目不是只要“能运行”，还要“对未来 agent 可读、可查、可验证”。因此：
- 任何重要流程都要有**单命令入口**与**结构化输出**
- 任何重要实验都要有**最小 smoke test** 与**固定输出目录契约**
- 任何关键数字都要能从 raw metrics 自动追溯到图表
- 任何重复出现的 review comment，都要考虑升级为机械化约束

### 1.7 失败时先补 harness，不要盲目重试
默认流程：
1. 判断失败是不是来自缺脚本 / 缺文档 / 缺测试 / 缺约束 / 缺可观测性
2. 若是，优先补 harness
3. 只有在 harness 足够完备后，再增加试验次数

---

## 2) 读前导航：什么时候必须去读另外两份文档

### 2.1 什么时候读 `MAIN_IDEA.md`
当任务涉及以下任一项时，必须先读 `MAIN_IDEA.md`：
- Writer / Reader / Fuser / Injector 的实现
- `M_long / M_short / queries / segment` 的定义与形状
- Stage A/B/C 训练流程
- Q-only / W-only / W+Q 的意义
- CDMI 的问题定义与核心主张
- 与 MemGen / LightThinker / MetaPrompting 的差异写法

优先看的小节：
- 锁定贡献
- 背景与问题定义
- CDMI
- 方法总览
- 训练与适配流程
- 与 MemGen 的关键差异

### 2.2 什么时候读 `EXPERIMENTS_INFO.md`
当任务涉及以下任一项时，必须先读 `EXPERIMENTS_INFO.md`：
- 接 benchmark
- 跑 baseline
- 设计/修改实验矩阵
- 画表画图
- 统计与复现
- 预算对齐（shots / steps / 参数量 / token / wall time）

优先看的小节：
- Paper-critical locked artifacts
- RQs
- 任务与数据
- Baselines
- 实验组 A–G
- 统计与复现规范
- 工程 runbook

---

## 3) 全局硬性要求（所有里程碑都适用）

### 3.1 最终论文必须覆盖两档 backbone
- **~Qwen2.5-1.5B-Instruct**：主力实验、消融、效率、曲线
- **~Qwen3-8B**：证明方法在更强模型上也成立

> 允许前期只用小 backbone 打通，但在进入主结果阶段之前，**Qwen2.5-1.5B-Instruct + Qwen3-8B  必须齐全**。

### 3.2 run 定义（固定）
一个 **run** = 固定 `{backbone, 方法变体, 任务/域, seed, 关键超参}` 的一次训练或完整评测。  
shots × steps 网格尽量在单个 run 内完成，并导出同一个 `adapt_curve.csv`。

### 3.3 所有 run 必须保存
- config 快照
- seed
- git hash
- metrics（json/jsonl/csv）
- 关键日志
- 关键图表或中间 CSV
- 若有 profiling：wall time / GPU hours / peak memory

### 3.4 绝不允许手工抄数
所有论文表/图都必须由统一汇总脚本自动生成。

### 3.5 seed 与统计
- 主结果：至少 3 seeds
- few-shot 曲线：建议 5 seeds
- 报告 mean ± 95% CI

### 3.6 CDMI 是 P0 主张，不可砍
如果算力紧张，可以推迟 bonus benchmark，但不能删掉 CDMI。

### 3.7 “重要到会重复提醒”的规则，必须变成机制
如果某个约束重复提醒两次以上，就把它升级成：
- lint
- 结构测试
- 汇总器检查
- 配置验证
- 文档固定规则

---

## M0. 工程地基：先把最小闭环与可复现框架搭起来

> 阅读指引：  
> - 先读 `MAIN_IDEA.md` 的“方法总览 / 实现契约 / 训练阶段”  
> - 再读 `EXPERIMENTS_INFO.md` 的“统一评测与结果汇总规范”

### P0 必须
- [ ] 建立仓库骨架
  - [ ] `src/`：models / data / training / eval / analysis / utils
  - [ ] `configs/`：tasks / exp / method
  - [ ] `scripts/`：环境、数据、训练、评测、画图、profiling
  - [ ] `docs/exec-plans/active/` 与 `docs/exec-plans/completed/`
  - [ ] `docs/ARCHITECTURE.md`（最小架构地图）
  - [ ] `docs/golden-principles.md`（可机械化的长期规则；可先空壳）
  - [ ] `docs/tech-debt-tracker.md`（记录 recurring cleanup / drift）
  - **DoD**：目录结构固定，最小命令入口可用，docs 骨架存在

- [ ] 建立统一命令行契约
  - [ ] 所有训练/评测入口都支持 `--config --seed --output_dir`
  - [ ] 支持保存 config snapshot
  - [ ] 支持 dry-run / smoke-run（至少一种）
  - **DoD**：`python -m train ...`、`python -m eval ...`、`python -m analysis ...` 全部能输出到统一 `runs/`

- [ ] 选择并锁定一个初始 backbone（用于前期打通）
  - [ ] 包装成 `BackboneWrapper`
  - [ ] 导出 hidden states / generate / KV 相关接口
  - **DoD**：固定 seed 下同 prompt 输出稳定

- [ ] 定义最小方法 API
  - [ ] `MemoryWriter.write(state) -> M_long`
  - [ ] `MemoryReader.read(M_long, context) -> {r_i}`
  - [ ] `MemoryFuser.fuse({r_i}) -> M_short`
  - [ ] `MemoryInjector.inject(M_short, next_inputs) -> next_inputs_with_memory`
  - **DoD**：toy task 上能跑通写→读→注入→继续生成，并能把梯度传回 Reader

- [ ] 建立统一日志与结构检查
  - [ ] 记录 loss / eval metric / wall time / token 数 / 显存峰值
  - [ ] 结构测试：run 是否带 config / seed / git hash
  - [ ] 文档链接检查：`AGENTS.md / TODO_LIST.md / MAIN_IDEA.md / EXPERIMENTS_INFO.md`
  - [ ] 结果治理检查：禁止手工生成主表文件
  - **DoD**：CI 一键跑通基础测试与 lint

- [ ] 建立“agent 可读”的最小 smoke harness
  - [ ] `scripts/setup_env.sh`
  - [ ] `scripts/setup_data.sh`
  - [ ] `scripts/dev_boot_smoke.sh`：从干净 checkout 启动最小训练/最小评测
  - [ ] `scripts/collect_artifacts.sh`：收集 metrics / logs / config / git hash
  - **DoD**：新 worktree/新机器上按文档可一次性跑通 smoke 流程

### P1 重要
- [ ] 实现 `Segmenter`
  - [ ] 句界 / delimiter / 自定义 token 边界至少支持一种
  - **DoD**：分段在不同机器 / batch 条件下完全一致

- [ ] 建立统一汇总器雏形
  - [ ] 能从 `metrics.json` 自动生成简单 CSV
  - [ ] 能做最小 sanity plot
  - **DoD**：不需要手工抄数

- [ ] 建立最小 profiling 接口
  - [ ] 记录 wall time / tokens / peak memory
  - [ ] 保存到统一 json/csv
  - **DoD**：任一 run 都能顺手产出 profiling 结果

### P2 加分
- [ ] doc gardening / 结构 lint
- [ ] 自动生成 run summary
- [ ] 自动检测 stale config / stale docs / orphan results

---

## M1. 复现 MemGen 作为强对照（先可运行，再可比）

> 阅读指引：  
> - `EXPERIMENTS_INFO.md`：Baselines、主结果口径、主套件任务  
> - `MAIN_IDEA.md`：与 MemGen 的关键差异（避免复现时把我们自己的方法写歪）

### P0 必须
- [x] 跑通 MemGen 官方实现至少 1 个 benchmark
- [x] 把 MemGen 纳入统一 baseline 接口
- [x] 对齐统一评测脚本与产物路径
- **DoD**：同一台机器、同一数据子集上，MemGen 能稳定产出可评测结果

### P1 重要
- [x] 对齐 MemGen 主实验任务清单
- [ ] 对齐它的插入/触发相关配置
- [x] 为 MemGen baseline 保存固定的 config/seed 模板
- **DoD**：你们的 `TaskEvaluator` 能直接读取并汇总 MemGen 输出

当前已对齐并真实 smoke 的 MemGen 任务：`gsm8k`、`gpqa`、`triviaqa`、`kodcode`、`rocstories`、`story_cloze`。统一模板与已验证种子见 `docs/baselines/memgen.md`；当前 M1 主要剩余项是 trigger / insertion 相关配置对齐。

### P2 加分
- [ ] 跑通 trigger on / off 两种版本
- [x] 记录 MemGen 训练/评测的常见坑，写回 repo
- [ ] 如某个坑稳定重现，升级为脚本或文档规则

---

## M2. 正式实现我们的方法：通用 Writer + 多 query Reader + 短注入

> 阅读指引：  
> `MAIN_IDEA.md` 的“方法总览 / 四个模块契约 / 与 MemGen 差异”是本里程碑唯一方法口径来源。

### P0 必须
- [ ] 实现 Writer
  - [ ] 至少两种可切换实现：MLP / small transformer（或等价）
  - [ ] 支持 freeze / unfreeze / save / load
  - **DoD**：在固定输入下输出稳定、可被 Reader 正常消费

- [ ] 实现 Reader
  - [ ] `H` 个 learned queries
  - [ ] cross-attention 读 `M_long`
  - [ ] 支持 batch / mask / variable segment
  - **DoD**：读出 shape 正确，梯度稳定

- [ ] 实现 Fuser
  - [ ] 支持 `H×d -> K×d`
  - [ ] 至少实现一个简单版（MLP / identity when K=H）
  - **DoD**：能输出合法 `M_short`

- [ ] 实现 Injector
  - [ ] 至少实现 prefix injection
  - [ ] 支持通过 config 切换注入开关
  - **DoD**：开关注入会引起可测的生成变化

### P1 重要
- [ ] 实现 Query-Gating
  - [ ] off / random / learned
  - **DoD**：能记录每段 gating 统计

- [ ] 实现多种注入位置
  - [ ] segment / delimiter / random / none
  - **DoD**：只改 config 即可切换

- [ ] 给方法模块补最小结构约束
  - [ ] 输入输出 shape 校验
  - [ ] domain / task conditioning 的命名和保存约定
  - **DoD**：常见 shape 错误与错误配置能在早期报错，而不是训练半天后崩

### P2 加分
- [ ] 更高级的注入方式（如 KV 初始化）
- [ ] query 多样性正则接口
- [ ] reader/fuser 的更细粒度 profiling

---

## M3. 训练流水线：Writer 预训 + queries meta-train + few-shot 适配

> 阅读指引：  
> - `MAIN_IDEA.md`：Stage A/B/C、H1–H4  
> - `EXPERIMENTS_INFO.md`：few-shot 协议、meta-split、适配对象消融

### P0 必须
- [ ] Stage A：Writer 通用预训
  - [ ] 搭建 general-field 数据管线
  - [ ] 记录数据版本 / hash / 采样规则
  - [ ] 输出冻结可复用的 writer checkpoint
  - **DoD**：能得到 `writer.ckpt`，并在后续实验中加载复用

- [ ] Stage B：queries meta-train
  - [ ] 实现 episode sampler
  - [ ] 优先实现 ANIL（或等价的一阶近似）
  - [ ] Writer 固定；queries 是主要更新对象
  - **DoD**：得到 `queries_meta_init.pt`，并能在 source domains 上观察到适配收益

- [ ] Stage C：few-shot / 少步数适配
  - [ ] 输入：新域 k-shot
  - [ ] 输出：适配后的 queries checkpoint + 适配曲线
  - **DoD**：固定一个 target domain，自动产出 shot-curve 与 step-curve

### P1 重要
- [ ] 适配对象消融
  - [ ] Q-only
  - [ ] W-only
  - [ ] W+Q
  - **DoD**：三条曲线齐全，且对齐相同预算

- [ ] Reader 学习方式消融
  - [ ] random queries
  - [ ] non-meta multi-task queries
  - [ ] meta-trained queries
  - **DoD**：能直接比较 meta 的价值

- [ ] 把训练失败常见模式写成显式检查
  - [ ] Reader 忽略 memory
  - [ ] Writer 输出退化为噪声
  - [ ] Fuser 折叠为单一 token
  - **DoD**：至少有 2–3 个 smoke ablation 可快速识别退化

### P2 加分
- [ ] query 多样性正则
- [ ] Writer 随机 query 正则预训变体
- [ ] active ExecPlan 模板自动生成脚本

---

## M4. benchmark 接入与统一评测 harness

> 阅读指引：  
> `EXPERIMENTS_INFO.md` 的“任务与数据 / 统一评测与结果汇总规范”是本里程碑的唯一口径来源。

### P0 必须
- [ ] 接入主套件 benchmark
  - [ ] GSM8K
  - [ ] MATH
  - [ ] GPQA
  - [ ] TriviaQA / PopQA
  - [ ] BigCodeBench / KodCode
  - [ ] ALFWorld
  - **DoD**：至少 4 个任务先跑通统一评测，再扩到 6–8 个

- [ ] 接入 OOD / hard transfer 套件
  - [ ] FEVER
  - [ ] ScienceWorld（如资源允许）
  - **DoD**：可进入跨域泛化实验

- [ ] 接入 Narrative 域（CDMI 必需）
  - [ ] 优先 Story Cloze / ROCStories
  - [ ] 尽量再接 NarrativeQA
  - **DoD**：可以构造 `Math+Code` vs `Math+Narrative`

- [ ] 接入 MemoryAgentBench（强烈建议）
  - **DoD**：能跑出四类能力分项结果

### P1 重要
- [ ] 固化 meta-split 到配置文件
- [ ] 固化 task prompt / CoT / 工具模板
- [ ] 写清数据许可与下载路径
- [ ] 每个 benchmark 至少提供一个 smoke subset

### P2 加分
- [ ] LoCoMo / LongMemEval
- [ ] 更多 agent memory benchmarks
- [ ] benchmark artifact 体积与缓存清理策略

---

## M5. baseline 全家桶：把“meta-read queries”的优势钉死

> 阅读指引：  
> `EXPERIMENTS_INFO.md` 的 Baselines、预算对齐与 CDMI 组是本里程碑的唯一口径来源。

### P0 必须
- [ ] Vanilla
- [ ] CoT
- [ ] MemGen
- [ ] Prompt Tuning
- [ ] LoRA
- [ ] MetaPrompting
- **DoD**：这些 baseline 都能在同一套 shot/step 网格下运行，并进入统一汇总器

### P1 重要
- [ ] LightThinker
- [ ] 外部记忆 / RAG 强 baseline（如果主表需要）
- **DoD**：至少有一条非 internal memory 路线可与我们对照

### P2 加分
- [ ] 更完整的 memory agent baseline 家族
- [ ] 更多 prompt-based / adapter-based meta-learning variants
- [ ] baseline 自动预算检查脚本

---

## M6. 论文产物收尾：主表、曲线、CDMI、效率、机制分析

> 阅读指引：  
> - `EXPERIMENTS_INFO.md`：图表与结果结构  
> - `MAIN_IDEA.md`：如何解释结果、如何把结果绑定到四条主张

### P0 必须（主文级）
- [ ] 主结果大表
  - [ ] Qwen2.5-1.5B-Instruct + Qwen3-8B 
  - [ ] ours zero-shot + ours few-shot
  - [ ] baseline 全家桶
  - **DoD**：自动生成 `table_main.csv/.tex`

- [ ] 跨域 few-shot / 少步数适配曲线
  - **DoD**：`shot_curve.pdf`、`step_curve.pdf`

- [ ] CDMI 主表 + 主图
  - [ ] `Math+Code (near)` vs `Math+Narrative (far)`
  - [ ] Interference Gap
  - **DoD**：`table_cdmi.csv/.tex` + `fig_cdmi_gap.pdf`

- [ ] 持续学习 / 域冲突
  - [ ] shared queries
  - [ ] per-domain query bank
  - [ ] 推荐：hard switch
  - **DoD**：性能矩阵 + 遗忘指标

- [ ] 效率与成本分析
  - [ ] inference cost
  - [ ] adaptation cost
  - **DoD**：`table_efficiency.csv/.tex` + 至少一张性能–成本图

- [ ] 机制分析
  - [ ] query specialization
  - [ ] adaptation 前后读法变化
  - **DoD**：至少 2 张机制图

### P1 重要（强 appendix）
- [ ] `L/H/K` 容量消融
- [ ] 注入位置与 gating 消融
- [ ] writer latent probe
- [ ] MemoryAgentBench 分项深入分析

### P2 加分
- [ ] 长期记忆 benchmark
- [ ] 更复杂的 query-gating 训练
- [ ] 自动化 slurm / submitit 实验编排
- [ ] 定期 cleanup / garbage collection PR（清理 stale runs / stale configs / stale docs）

---

## 4) 每个里程碑完成时的统一交付格式（必须）

每次阶段性交付必须输出：

1. **完成了哪个 TODO 条目**
2. **改了哪些文件**
3. **复现命令**
4. **验证命令**
5. **关键结果**
6. **产物路径**
7. **已知问题**
8. **下一步建议**
9. **是否新增/更新了 repo 内知识**（文档/规则/脚本/测试）

---

## 5) 如果资源不够，优先级怎么砍（只能这样砍）

### 绝对不能砍
- 主结果大表
- 跨域 few-shot 曲线
- CDMI
- 持续学习（至少 shared vs bank）
- 效率分析
- 机制分析（至少两张）

### 可以后移
- LightThinker
- 外部记忆大礼包
- LoCoMo / LongMemEval
- 更复杂的 gating 训练

### 可以只做小规模验证
- NarrativeQA（如果 Story Cloze 已跑通）
- MemoryAgentBench 深挖
- writer latent probe 的大网格版本

---

## 6) 给 Agent 的最后一句话

> **不要把这个项目做成“一个能跑的 memory module”；要把它做成“一篇主张清晰、证据完整、能直接写进论文的 memory paper”。**  
> 判断一项工作值不值得优先做，请永远回到四个锁定主张：  
> **通用 Writer、可迁移 Reader、写长读短、CDMI 缓解。**  
> 如果某条规则或坑会重复出现，就把它升级成 repo 内的长期能力，而不是继续靠人类提醒。
