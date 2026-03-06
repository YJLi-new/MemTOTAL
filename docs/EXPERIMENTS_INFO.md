# EXPERIMENTS_INFO.md

> 角色定位：这是给实现/实验 Agent 的 **实验协议、对照口径与论文产物规范**。  
> - **任务顺序 / DoD / 何时做什么**：看 `TODO_LIST.md`
> - **方法定义 / 论文主张 / 核心假设**：看 `MAIN_IDEA.md`
> - **本文件负责**：告诉你**必须跑什么、怎么对齐、怎么出表出图、怎样才算能写进论文**

---

## 0) 先给 Agent 的一句话版本

这篇论文最终必须用实验回答四个问题：

1. **Writer 是否真能学出跨域可写的 latent buffer？**
2. **Reader queries 是否真能通过 meta-train 学会“怎么读”，并在新域 few-shot / 少步数快速适配？**
3. **写长读短是否在性能–成本上有意义？**
4. **面对 CDMI（math ↔ narrative 这类大域差）时，我们是否比 MemGen 更稳？**

> 如果某个实验不能帮助回答这四个问题，它要么是次要实验，要么不该优先跑。

---

## 1) 实验设计总原则（锁定不变）

### P1. 与 MemGen 可比
主实验结构要尽量对齐 MemGen 的强项：
- 多领域主结果
- 跨域泛化
- 持续学习/遗忘
- 插入/触发策略
- 效率分析

### P2. 与我们的方法主张强绑定
额外必须回答：
- 通用 Writer 的可读性
- meta-trained Reader 的 few-shot 适配效率
- 与 Prompt / LoRA / MetaPrompting 的预算对齐对比
- CDMI 压力测试

### P3. 统一预算，禁止隐形作弊
所有关键对照必须显式对齐：
- shots
- adaptation steps
- 可训练参数量（或至少报告）
- 训练 token / wall time / GPU hours（至少报告一种或多种）
- seed 数

### P4. 不手工抄数
所有论文表/图必须由统一汇总脚本从原始 `metrics.json` / `jsonl` 自动生成。

---

## 2) 先锁定论文最核心的产物（paper-critical locked artifacts）

这些是**主文必须有**的，不可删减：

### A. 主结果大表
至少包含：
- 2 个 backbone：**~Qwen2.5-1.5B-Instruct 与 ~Qwen3-8B**
- 你们方法：zero-shot（meta-init） + few-shot adapted
- 对照：MemGen、Vanilla/CoT、Prompt/LoRA、MetaPrompting、（可选）LightThinker
- 任务：尽量对齐 MemGen 主套件（至少 6–8 个）

### B. 跨域 few-shot / 少步数适配曲线
- shot-curve
- step-curve
- 至少 1 个 OOD / hard transfer 设定

### C. CDMI 主表 + 主图
- `Math + Code (near)` vs `Math + Narrative (far)`
- MemGen 与 Ours 的 Interference Gap
- 至少一张机制解释图（attention / gating / head importance）

### D. 持续学习 / 域冲突结果
- shared queries
- per-domain query bank
- （推荐）hard switch：Math → Narrative → Math

### E. 性能–成本对照
- inference cost
- adaptation cost
- 至少一张“性能–成本”图或表

### F. 机制分析
至少 2 张图：
- query specialization / head ablation
- adaptation 前后“读法变化”

> 这六类产物完成，论文主干才算成立。

---

## 3) 研究问题（RQs）与对应证据

### RQ1：总体效果
在多领域任务上，我们是否优于：
- 无记忆方法（Vanilla / CoT）
- 外部记忆方法（MemoryBank / ExpeL / AWM 等）
- internal latent memory（MemGen）
- 推理压缩路线（LightThinker，可选）

**证据**：
- 主结果大表
- 分任务表现
- zero-shot 与 few-shot 两个版本

### RQ2：Reader 的 meta-learning 是否真的带来跨域快速适配？
固定 Writer 时，queries 是否能够：
- zero-shot 就“会读”
- few-shot / 少步数适配到新域
- 以更低参数/时间成本达到接近大模块微调的效果

**证据**：
- shot-curve
- step-curve
- Q-only / W-only / W+Q
- Prompt / LoRA / MetaPrompting 对照

### RQ3：通用 Writer 是否真的可跨域复用？
Writer 是否学到一种跨域可写、跨域可读的 latent memory language，而不是只在 source tasks 上有效？

**证据**：
- 固定 Writer + 不同 Reader 的跨域测试
- meta vs non-meta reader zero-shot 对比
- writer latent probe / domain separability（机制分析）

### RQ4：CDMI 是否存在？我们能否缓解？
跨大域共享 memory 时，MemGen 是否更易掉点？我们的 Reader 适配是否能缩小 Interference Gap？

**证据**：
- `Math+Code` vs `Math+Narrative`
- Interference Gap
- hard switch continual
- query attention / gating 分布变化

### RQ5：写长读短是否有成本–效果优势？
在只注入极短 `M_short` 的前提下，我们是否仍能获得强于无记忆/接近长链方法的效果？

**证据**：
- `K` sweep
- `L/H/K` 容量消融
- inference cost 分解（Writer/Reader/Reasoner）

### RQ6：多 queries 是否真的形成分工？
queries 是否读取不同 facet，且 few-shot 适配能改变“读法”而不是硬背答案？

**证据**：
- head ablation
- attention entropy / coverage
- adaptation 前后对比

---

## 4) 任务与数据（按优先级组织）

## 4.1 主套件（优先对齐 MemGen）
优先覆盖：
- **ALFWorld**（agent / embodied）
- **TriviaQA / PopQA / GPQA**（knowledge QA）
- **KodCode / BigCodeBench**（code）
- **GSM8K / MATH**（math reasoning）

要求：
- 主表尽量覆盖 6–8 个任务
- 各任务指标统一封装进 `TaskEvaluator`
- PopQA 在没有训练集时可以沿用 MemGen 的 TriviaQA→PopQA 设定，但必须明确记录

## 4.2 OOD / hard transfer 套件
优先：
- **ScienceWorld**
- **FEVER**

这些任务用于：
- Train on one → Test on several others
- zero-shot vs few-shot 适配曲线
- 证明“学到的是读 memory 的能力，不是背某个数据集”

## 4.3 CDMI 的 Narrative 域（必须）
至少接一个，最好两个：

### 优先级 1：Story Cloze / ROCStories
- 指标简单（accuracy）
- 适合大规模 grid / curriculum / CDMI 近远域对照
- 适合作为 `Math + Narrative` 的 narrative 端

### 优先级 2：NarrativeQA
- 更强的长叙事理解
- 更贴近“跨段 memory + 关系保持”
- 更重，但如果跑通会让 CDMI 更有说服力

> 注意：Narrative 域的作用不是“再加一个 benchmark 凑热闹”，而是服务于 **CDMI stress test**。

## 4.4 Memory-centric 套件（强烈建议至少接一个）
### MemoryAgentBench
用来展示：
- 准确检索
- test-time learning
- 长程理解
- 冲突消解 / 选择性遗忘

它与我们的“Reader few-shot 适配”天然匹配，尤其适合做 TTL 维度的强证据。

## 4.5 可选加分项
- LoCoMo
- LongMemEval

这类可作为“长期对话/长期记忆”的外部验证，但不是 paper-critical。

---

## 5) 模型与 backbone 设定（锁定）

### 5.1 最终论文必须覆盖两档 backbone
- **~Qwen2.5-1.5B-Instruct**：主力实验、消融、效率、曲线
- **~Qwen3-8B**：证明方法在更强 backbone 上仍成立

> 允许 M0/M1 用更小模型快速打通，但进入论文结果阶段后，**Qwen2.5-1.5B-Instruct + Qwen3-8B 都是必须项**。

### 5.2 关键方法超参
至少需要 sweep：
- `L_long ∈ {64, 128, 256, 512}`
- `H_queries ∈ {K, 2K, 4K}`
- `K_short ∈ {2, 4, 8, 16}`

优先级：
1. `K`
2. `H`
3. `L`

---

## 6) Baselines（必须覆盖，且预算对齐）

## 6.1 论文主表必须出现的 baseline
1. **Vanilla**
2. **CoT**
3. **MemGen**
4. **Prompt Tuning**
5. **LoRA**
6. **MetaPrompting**
7. （可选但加分）**LightThinker**
8. （可选）外部记忆 / RAG / MemoryBank / ExpeL / AWM

## 6.2 我们方法最关键的内部消融
这些不是可选项，必须有：

### Reader 学习方式
- random queries
- non-meta multi-task queries
- meta-trained queries（ours）

### 适配对象
- Q-only
- W-only
- W+Q

### 容量与带宽
- `L`
- `H`
- `K`

### 注入与读取策略
- segment / delimiter / random / none
- gating: off / random / learned

---

## 7) 训练与评测协议（统一口径）

## 7.1 Stage A / B / C（与 MAIN_IDEA 严格对齐）
### Stage A：Writer 预训
- general-field 数据
- 输出可冻结复用的 Writer
- 保存数据版本与 hash

### Stage B：queries meta-train
- episode 采样
- support/query split
- 优先 ANIL
- Writer 固定，queries 为主要更新对象

### Stage C：新域 few-shot 适配
- Writer 固定
- 从 meta-init 出发
- 只更新 queries（默认）
- 输出适配曲线与 adapted queries checkpoint

---

## 8) Few-shot 适配协议（锁定网格）

### 8.1 推荐标准网格
- `shots ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128}`
- `steps ∈ {0, 1, 3, 5, 10, 20, 50}`
- seeds：
  - 主结果至少 3
  - few-shot 曲线推荐 5

### 8.2 工程建议
- 不要把每个 `(shot, step)` 拆成独立 job
- 应在单个 run 内循环完成整个网格
- 输出：
  - `adapt_curve.csv`
  - `adapt_cost.json`
  - 每域最终 checkpoint（不是每步都存模型）

---

## 9) meta-train / meta-test split（锁定）

建议按**领域**而非单个数据集做 split：

- **Math**：GSM8K / MATH / AQuA
- **Code**：KodCode / BigCodeBench
- **Knowledge QA**：TriviaQA / PopQA / GPQA
- **Embodied / Interactive**：ALFWorld / ScienceWorld
- **Fact Verification**：FEVER
- **Narrative**：Story Cloze / NarrativeQA

推荐两种设定：

### S1. Leave-one-domain-out
每次留一个 domain 做 meta-test，其余做 meta-train。  
适合做稳健平均。

### S2. Hard shift
meta-train 只用 `Math + QA`，meta-test 用 `Code + Embodied + FEVER`。  
适合强调强域迁移。

---

## 10) Paper-critical 实验组（A–G）

## A. 主结果（必须）
### A1. 多任务主表
- 任务：至少 6–8 个
- backbone：Qwen2.5-1.5B-Instruct + Qwen3-8B
- 方法：ours zero-shot、ours few-shot、MemGen、Prompt/LoRA/MetaPrompting、Vanilla/CoT
- 输出：
  - `table_main.csv`
  - `table_main.tex`

### A2. 报告原则
- ours 至少同时报告：
  - zero-shot（meta-init）
  - few-shot（例如 16-shot / 10-step）
- 不能只报你们最好看的版本

---

## B. 跨域泛化 + few-shot 适配（必须）
### B1. Train-on-one → Test-on-several-others
对齐 MemGen 风格，但增强为：
- 固定 Writer
- queries 在 source domains meta-train
- target domains 上做 zero-shot / 4-shot / 16-shot / 64-shot

### B2. 适配对象消融
- Q-only
- W-only
- W+Q

### B3. 域外“读能力”probe
- meta-trained queries zero-shot
- non-meta multi-task queries zero-shot
- random init queries

> 目标：证明 meta 学到的是“怎么读 memory”，而不是 task memorization

---

## C. CDMI（必须，主文级）
### C1. 训练设定
在严格同预算下，只改变混合的第二域：

- `Train-MC (near)` = Math + Code
- `Train-MN (far)` = Math + Narrative

### C2. 比较对象
- `MemGen(MC)` vs `MemGen(MN)`
- `Ours(MC)` vs `Ours(MN)`

其中 ours 至少含三种读取方式：
- zero-shot（meta-init）
- few-shot 适配
- per-domain query bank  
（可选再加 learned gating）

### C3. 评测重点
- 主看 math 任务的下降
- narrative 任务作为 trade-off 参考

### C4. 核心指标
\[
\Delta_{\text{CDMI}}^{\text{MemGen}} = \text{Score}(\text{MemGen-MC}) - \text{Score}(\text{MemGen-MN})
\]

\[
\Delta_{\text{CDMI}}^{\text{Ours}} = \text{Score}(\text{Ours-MC}) - \text{Score}(\text{Ours-MN})
\]

### C5. 必须产出
- `table_cdmi.csv/.tex`
- `fig_cdmi_gap.pdf`
- （推荐）`fig_cdmi_curve.pdf`
- （推荐）`fig_cdmi_gating.pdf`

---

## D. 持续学习 / 域冲突（必须）
### D1. shared queries
- 同一套 queries 顺序适配多个域
- 报告遗忘

### D2. per-domain query bank
- 每域单独保存 queries
- 报告“参数开销 vs 遗忘缓解”

### D3. hard switch continual（强烈建议）
- `Math → Narrative → Math`
- 观察：
  - Narrative 加入后 math 立即掉多少
  - 回到 math 后能恢复多少

---

## E. 插入与读取策略（必须）
### E1. 注入位置
- segment
- delimiter
- random
- none

### E2. query gating
- off
- random
- learned

### E3. 必须产出
- 一张消融表
- 一张 gating / 触发频率图

---

## F. 容量与效率（必须）
### F1. `L/H/K` 消融
- `K_short ∈ {2, 4, 8, 16}`
- `H_queries ∈ {K, 2K, 4K}`
- `L_long ∈ {64, 128, 256, 512}`

### F2. inference cost
至少报告：
- 平均 wall time
- 生成 token 数
- peak GPU memory
- Writer time / Reader time / Reasoner time

### F3. adaptation cost
至少报告：
- 达到某阈值提升所需的 shots / steps
- 更新参数量
- wall time / GPU hours

### F4. 必须产出
- `table_efficiency.csv/.tex`
- `fig_perf_vs_cost.pdf`
- `fig_perf_vs_K.pdf`（推荐）

---

## G. 机制分析（必须至少做两项）
### G1. query specialization
- head ablation
- attention entropy / coverage
- 不同领域的重要性排序是否不同

### G2. 适配前后“读法变化”
- attention map before vs after adaptation
- readout embedding 可视化（t-SNE / UMAP 可选）

### G3. writer latent probe（推荐）
- 用 probe 看 writer 输出是否过度 domain-specific
- 目标不是让它“完全不可分”，而是说明它仍然是共享表征空间

---

## 11) Paper-secondary / appendix-only 实验（可做但不能挤占主线）
- LightThinker 细致对比
- 外部记忆系统全家桶
- MemoryAgentBench 深挖四能力
- LoCoMo / LongMemEval
- 更复杂的 gating 训练（如 RL）

> 这些都是加分项，但如果 paper-critical 六类产物没完成，不要优先做它们。

---

## 12) 统一评测与结果汇总规范

### 12.1 统一 eval 入口
所有方法都应走统一入口，例如：
- `python -m eval.run_eval --task gsm8k --method ours --ckpt ...`

输出至少包含：
- `predictions.jsonl`
- `metrics.json`

### 12.2 统一汇总器
必须有一个 `report.py`（或等价脚本）负责：
- 扫描 `results/**/metrics.json`
- 自动生成主表、曲线、消融表
- 导出 csv / tex / figure

### 12.3 命名规范
run 目录中必须显式写入关键超参，例如：
- backbone
- method
- L/H/K
- inject mode
- seed

---

## 13) 统计与复现规范（不能丢）

1. 主结果至少 3 seeds  
2. few-shot 曲线尽量 5 seeds  
3. 报告 mean ± 95% CI  
4. 记录硬件、驱动、主要依赖版本  
5. 明确训练 token / wall time / 参数量对齐方式  
6. 明确数据版本、split、meta-split  
7. 保存 config 快照与 git hash

---

## 14) 工程 runbook（给 Codex 的执行建议）

## 14.1 推荐目录
- `configs/tasks/`
- `configs/exp/`
- `configs/method/`
- `scripts/`
- `src/data/`
- `src/models/`
- `src/training/`
- `src/eval/`
- `src/analysis/`
- `results/`

## 14.2 推荐脚本清单
- `00_prepare_data.sh`
- `10_pretrain_writer.sh`
- `20_meta_train_queries.sh`
- `30_adapt_queries.sh`
- `40_eval_main_table.sh`
- `41_eval_cross_domain.sh`
- `42_eval_continual.sh`
- `43_eval_cdmi.sh`
- `50_run_ablations.sh`
- `60_profile_efficiency.sh`

## 14.3 优先执行顺序（从零到论文图）
1. 跑通单任务端到端（GSM8K）
2. 跑通 meta-train + few-shot 曲线
3. 接入 Prompt / LoRA / MetaPrompting
4. 跑通 MemGen baseline
5. 先做最小主表（4 任务）
6. 扩到完整主套件
7. 做 CDMI
8. 做持续学习
9. 做效率与机制分析

---

## 15) 最后给 Agent 的一句话

> **不要把实验做成“很多图”，要把实验做成“很多证据”，而且这些证据必须逐条回答 `MAIN_IDEA.md` 里的四个锁定主张。**  
> 如果你不确定某个实验是否值得跑，先问：它能否帮助证明“通用 Writer / 可迁移 Reader / 写长读短 / CDMI 缓解”中的至少一条？