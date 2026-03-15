# MAIN_IDEA.md

> 角色定位：这是给实现/实验 Agent 的 **方法规格说明书（why / what / how）**。  
> - **做事顺序 / 排期 / DoD**：以 `TODO_LIST.md` 为准  
> - **实验矩阵 / 对照口径 / 出表出图**：以 `EXPERIMENTS_INFO.md` 为准  
> - **方法定义 / 核心假设 / 论文主张 / 与 MemGen 的差异**：以本文为准  

---

## 0) 先给 Agent 的一句话版本（必须记住）

**我们不是“也做一个 latent memory”。**  
我们的核心主张是：

1. **通用 Writer**：在 general field 上训练一个可跨域复用的 `M_long` 写入器  
2. **可快速适配的 Reader queries**：先多领域 meta-train“怎么读 latent buffer”，再在新域 few-shot / 少步数适配  
3. **写长读短**：内部写入高容量 `M_long`，真正注入推理流的只有极短 `M_short`  
4. **CDMI 缓解**：面对 math ↔ narrative 这类大域差时，域差异主要被压缩到 Reader queries 上，因此比 MemGen 更不容易被跨域记忆污染

> 任何实现、实验或写作，如果没有直接服务于这四条主张，就说明跑偏了。

---

## 1) 论文级一句话 claim（建议在摘要/引言复用）

我们提出一种**两级 latent memory**：通用 Writer 将推理状态写入高容量 latent buffer `M_long`；多个可 meta-train 的 Reader queries 从 `M_long` 读出并压缩为极短的 `M_short` 注入后续推理。该设计把跨域共享放在写入空间，把领域差异集中到可快速适配的读取接口上，从而在维持低注入成本的同时，提高跨域泛化与 test-time adaptation 能力，并缓解跨大域记忆干扰（CDMI）。

---

## 2) 这项工作的“锁定贡献”（locked contributions）

### C1. 两级带宽控制：写长、读短、注入更短
- 内部缓存：`M_long ∈ R^{L×d}`
- 外部注入：`M_short ∈ R^{K×d}`, 且 `K ≪ L`

这意味着：
- Writer 可以承载更高容量的内部状态
- 真正进入上下文的 memory 很短，控制推理成本
- 方法不依赖把长中间思维链直接塞回上下文

### C2. 把“跨域适配”放在 Reader，而不是 Writer / Backbone
- Writer 主要负责学会“如何写一种跨域可读的 memory language”
- Reader queries 负责“当前域该读什么、忽略什么”
- meta-learning 的对象是 **memory readout interface**，不是 prompt 本身，也不是整个主模型

### C3. 提出并显式测量 CDMI（Cross-domain Memory Interference）
- 当同一记忆系统同时覆盖差异很大的域（如 math ↔ narrative）时，latent memory 容易携带无关模式并干扰当前域推理
- 我们把这个现象做成 **明确的 stress test 与量化指标（Interference Gap）**
- 这不是附带现象，而是论文要主动讲清楚的主要问题定义之一

> 对 reviewer 的核心信息：  
> **我们的新意不在“latent memory”本身，而在“通用写入 + 可迁移读取 + 跨大域干扰缓解”的整体范式。**

---

## 3) 背景与问题定义

### 3.1 为什么需要内部 memory，而不是只靠长上下文
推理型 LLM 往往需要跨多个 reasoning segment 保持计划、约束、局部结论和中间状态。直接保留长链会带来：
- 上下文变长
- KV cache 压力增大
- 推理时延上升
- 长链中无关细节可能反而干扰后续决策

### 3.2 现有相关路线与它们的局限

#### MemGen
MemGen 通过 **memory trigger + memory weaver** 在推理过程中动态生成并插入 machine-native latent memory，让记忆与推理细粒度交织。其强项是：
- internal latent memory
- agent 场景
- 跨域泛化与持续学习叙事

但在我们的视角下，MemGen 仍有两个潜在不足：
1. 写入与注入更接近同一层表达，缺少显式“高容量缓存 vs 低容量注入”的分离  
2. 当 memory 覆盖大域差时，缺少一个明确“把域差异集中到读取接口”的机制；这正是我们提出 CDMI stress test 的动机

#### LightThinker
LightThinker 通过压缩中间 thinking steps 为更短表示来降低推理成本。它说明“**中间状态压缩**”是有效方向，但它的重点不在“跨域记忆写入/读取接口”，也不涉及“meta-train 一个会读 memory 的小模块”。

#### MetaPrompting / Prompt Tuning / LoRA
这些方法说明“小模块快速适配”可行，但它们适配的对象主要是：
- 输入侧 prompt
- 模型层内 adapter / LoRA
而不是“memory readout interface”。

### 3.3 我们要解决的确切问题
我们关注的是下面这个更具体的问题：

> **能否学出一种跨域可写的内部 latent memory，并把领域差异主要交给一个很小、可快速适配的 Reader 接口，从而在不显著增加上下文成本的前提下提高跨域推理、few-shot adaptation 和跨大域鲁棒性？**

---

## 4) CDMI：跨大域记忆干扰（Cross-domain Memory Interference）

这是本文必须突出的 stress test，不是附属实验。

### 4.1 定义
当同一 memory system 被迫同时覆盖两个差异很大的域（例如 **math ↔ narrative**）时，写入到 latent memory 中的模式会变得高度多模态。  
如果读取机制不具备足够的域选择性，当前域推理时可能会：
- 读到 irrelevant facets
- 把另一域的结构偏好带入当前推理
- 导致性能下降或不稳定

我们把这种现象称为 **CDMI（Cross-domain Memory Interference）**。

### 4.2 为什么它对我们重要
CDMI 直接服务于论文主张：
- 它把“跨域泛化”从抽象优点变成一个可被证伪的硬问题
- 它提供一个 reviewer 很容易理解的新 stress test
- 它使“通用 Writer + 可适配 Reader”这一设计显得必要，而不是任意拼装

### 4.3 我们的预测
- **MemGen 类方法**：当统一 memory 覆盖大域差时，目标域表现会更容易劣于“单域/近域训练”的版本  
- **我们的方法**：通过 few-shot/少步数适配 Reader queries，能在新域更快学会“该读什么”，从而缩小 Interference Gap

---

## 5) 方法总览（Method Overview）

### 5.1 关键对象与符号
- `segment`：推理过程中的自然片段（如每段 thinking、句界、或自定义标记边界）
- `s_t`：第 `t` 个 segment 的状态表示
- `M_long ∈ R^{L×d}`：长 latent buffer（高容量、机器原生、不可读）
- `Q = {q_1, …, q_H}, q_i ∈ R^d`：H 个 Reader queries
- `r_i ∈ R^d`：第 `i` 个 query 的读出向量
- `M_short ∈ R^{K×d}`：短 memory tokens，真正注入下一段推理
- `K ≪ L`

### 5.2 数据流
#### Step 1: Write
从当前 segment 状态写入长 latent buffer：
\[
M^{(t)}_{\text{long}} = f_{\text{write}}(s_t)
\]

Writer 的目标不是生成可读文本，而是生成一种**跨域可读的内部记忆语言**。

#### Step 2: Read
每个 query 对 `M_long` 做 cross-attention 读出：
\[
r_i^{(t)} = \mathrm{Attn}(q_i, K=M^{(t)}_{\text{long}}, V=M^{(t)}_{\text{long}})
\]

直觉上：
- 不同 queries 应读取不同 facet
- 这些 facet 不一定显式对齐“计划/事实/约束”，但在训练后应呈现分工

#### Step 3: Fuse
将多个读出融合为短 memory tokens：
\[
M^{(t)}_{\text{short}} = f_{\text{fuse}}(r_1^{(t)}, \dots, r_H^{(t)})
\]

#### Step 4: Inject
把 `M_short` 注入下一段推理输入：
\[
x_{t+1} \leftarrow [M^{(t)}_{\text{short}}; x_{t+1}]
\]

> 关键点：`M_long` 是高容量的内部缓存；`M_short` 才是低带宽的外显 memory。  
> 这正是我们与 MemGen 最重要的结构差异之一。

---

## 6) 四个模块的最小契约（Implementation Contract）

### 6.1 Writer
- 输入：当前 segment 的状态 `s_t`
- 输出：`M_long`，shape `[B, L, d]`
- 责任：
  - 生成高容量 latent buffer
  - 尽量形成跨域可写、跨域可读的 memory language
- 默认实验设定：
  - 经 Stage A 训练后**多数实验冻结**

### 6.2 Reader
- 输入：`M_long`、可选当前上下文 / domain conditioning
- 参数：
  - `H` 个 queries
  - 可选 query-gating
- 输出：
  - `H` 个读出向量 `{r_i}`
- 责任：
  - 决定“读什么 / 不读什么”
  - 承担主要的域适配压力

### 6.3 Fuser
- 输入：`{r_i}`
- 输出：`M_short`，shape `[B, K, d]`
- 责任：
  - 把多 query 读出的信息压缩到极短 memory tokens

### 6.4 Injector
- 输入：`M_short` 和下一段推理输入
- 输出：带 memory 的下一段输入
- 责任：
  - 提供统一的注入位置与形式
  - 支持 segment 边界 / 句界 / 随机 等可切换策略（用于实验）

---

## 7) 训练与适配流程（Stage A / B / C）

### Stage A：通用 Writer 训练（General-field pretraining）
目标：得到一个可复用的 Writer，使得 `M_long` 具备“跨域可读性”。

要求：
- 训练数据尽量覆盖多个领域与体裁
- 训练后 Writer 默认冻结
- 训练记录必须保存数据版本 / hash / 关键超参

### Stage B：queries 的跨域 meta-training
目标：让 Reader queries 学会“如何读 latent buffer”，而不是只记住某个数据集。

推荐设定：
- 以 **ANIL** 为默认首选：inner-loop 只更新 queries（和可选 fuser），更贴合本文设定
- 每个 episode 采样一个 domain / task
- support set 用于快速适配
- query/val set 用于 outer-loop 更新 meta-init

### Stage C：新领域 few-shot / 少步数适配
目标：在固定 Writer 的前提下，只更新 Reader queries，就让新域很快“会读”。

输出必须包括：
- zero-shot（不适配）
- shot-curve
- step-curve
- 每域 adapted queries checkpoint
- 适配 wall time / 参数量 / 显存统计

---

## 8) 我们到底和 MemGen 哪里不同（必须讲得非常清楚）

### 对齐点
- 都使用 internal latent memory
- 都把 memory 与 reasoning flow 交织
- 都关心 agent / 多步推理 / 跨域

### 关键差异点（必须在方法段和实验段都重复）
#### D1. 两级记忆，而非直接织入同层 latent tokens
- MemGen 更接近“生成一段 latent tokens 并插入 reasoning”
- 我们显式分出：
  - `M_long`：高容量内部缓存
  - `M_short`：极短注入 memory

#### D2. 域差异主要由 Reader 承担，而不是 Writer/主模型
- MemGen 的主要学习对象是 trigger + weaver
- 我们把域适配主要放在 queries 上
- 这让 few-shot / 少步数适配更自然，也更参数高效

#### D3. 我们把 CDMI 做成显式问题定义与主要实验
- 不是只说“跨域泛化好”
- 而是专门问：**跨大域时会不会记忆互相污染？**
- 并用 Interference Gap 量化

---

## 9) 与其它相关工作的关系（只保留最需要的 anchors）

### MemGen
定位：最重要的直接对照  
我们需要对齐其：
- 主表任务结构
- 跨域泛化
- 持续学习
- 插入/触发消融
- 效率分析

### LightThinker
定位：压缩推理链的强相关路线  
我们借它说明“短注入 memory 有价值”，但我们的重点是：
- internal latent buffer
- 可适配的 readout interface
- CDMI stress test

### Perceiver IO
定位：query-to-latent readout 的结构先例  
我们借它说明：**用多个 query 从 latent array 读取不同输出是合理、成熟的架构思想。**

### MAML / ANIL
定位：few-shot adaptation 的方法论基础  
我们不是把它用在整个模型或 soft prompt，而是用在 **memory readout interface** 上。

### Prompt Tuning / LoRA / MetaPrompting
定位：参数高效适配与 meta initialization 的直接对照  
这些是必须对齐的 baseline，而不是 related work 里一笔带过的名字。

---

## 10) 我们的核心假设（实验必须逐条验证）

### H1. 通用 Writer 假设
一个用 general field 训练的 Writer 能形成跨域可写、跨域可读的 latent memory language。

### H2. 可迁移 Reader 假设
经过多领域 meta-training 的 queries 学到的是“怎么读 memory”的 transferable skill，而不是 task memorization。

### H3. 写长读短假设
虽然内部写入容量大，但由于只注入极短 `M_short`，整体推理成本仍接近短上下文方法。

### H4. CDMI 缓解假设
在 math ↔ narrative 这类大域差设定下，我们比 MemGen 有更小的 Interference Gap。

> 注意：  
> 任何实验设计都要能明确回答这些假设，而不是只是“又跑了一张表”。

---

## 11) 失败模式与实现注意事项（避免 agent 跑偏）

### 11.1 容易出现的退化解
- Reader 基本不看 `M_long`
- Writer 输出对 Reader 不可用的噪声
- Fuser 把所有 query 读出压成几乎同一 token
- queries 只记住 source tasks，不会迁移

### 11.2 为避免退化，至少要做的检查
- 关掉 memory 的 ablation：确认 `M_short` 是 load-bearing
- random queries vs meta-trained queries：确认 meta 真有用
- Q-only / W-only / W+Q：确认“适配放在 queries”确实合理
- head ablation：确认多 query 不是摆设

### 11.3 数据与 split 的红线
- meta-train domains 与 meta-test domains 不可混
- 若 Writer 的 general-field 训练集可能含目标 benchmark 内容，必须显式声明并尽量去重
- CDMI 的 near/far 对照必须严格同预算

---

## 12) 推理阶段契约（Inference Contract）

每个样本的推理必须被组织为多个 segments；在每个 segment 边界执行：

1. `M_long ← Writer(s_t)`
2. `{r_i} ← Reader(M_long, context)`
3. `M_short ← Fuser({r_i})`
4. `next_inputs ← Injector(M_short, next_inputs)`
5. 继续生成下一段

可选增强：
- query-gating / reader-trigger：按段选择启用哪些 queries
- 记录每段 attention / gating / 读出统计，供后续机制分析

---

## 13) 这份文档与另外两份文档的关系（最后再提醒一次）

### 本文负责
- 方法定义
- 核心假设
- 论文主张
- 与 MemGen/LightThinker/MetaPrompting 的差异
- CDMI 问题定义

### `EXPERIMENTS_INFO.md` 负责
- 用什么实验来证明这些主张
- 跑哪些 baseline / grid / ablation
- 输出什么表与图
- 如何统计与复现

### `TODO_LIST.md` 负责
- 先做什么、后做什么
- 每个里程碑的 DoD
- 哪些任务是 P0 / P1 / P2
- 什么时候必须回来看本文 / 实验文档

---

## 14) 给 Agent 的最后一句话

如果你只能记住一句话，请记住这句：

> **这篇工作的关键不是“有记忆”，而是“通用写入 + 可适配读取 + 写长读短 + CDMI 缓解”。**  
> 任何实现、实验或论文文字，都必须让这四点更清楚，而不是更模糊。