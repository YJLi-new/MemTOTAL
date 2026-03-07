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
- [x] 对齐它的插入/触发相关配置
- [x] 为 MemGen baseline 保存固定的 config/seed 模板
- **DoD**：你们的 `TaskEvaluator` 能直接读取并汇总 MemGen 输出

当前已对齐并真实 smoke 的 MemGen 任务：`gsm8k`、`gpqa`、`triviaqa`、`kodcode`、`rocstories`、`story_cloze`。统一模板与已验证种子见 `docs/baselines/memgen.md`；`trigger_active / insertion_profile / requires_trained_checkpoint / load_model_path` 已进入显式配置契约，且 `trigger.active=True` 的最小 gsm8k smoke 路径已验证可跑。后续剩余的是“正式 trigger baseline 权重来源”问题，不再是配置层缺口。

### P2 加分
- [x] 跑通 trigger on / off 两种版本
- [x] 记录 MemGen 训练/评测的常见坑，写回 repo
- [x] 如某个坑稳定重现，升级为脚本或文档规则

---

## M2. 正式实现我们的方法：通用 Writer + 多 query Reader + 短注入

> 阅读指引：  
> `MAIN_IDEA.md` 的“方法总览 / 四个模块契约 / 与 MemGen 差异”是本里程碑唯一方法口径来源。

### P0 必须
- [x] 实现 Writer
  - [x] 至少两种可切换实现：MLP / small transformer（或等价）
  - [x] 支持 freeze / unfreeze / save / load
  - **DoD**：在固定输入下输出稳定、可被 Reader 正常消费

- [x] 实现 Reader
  - [x] `H` 个 learned queries
  - [x] cross-attention 读 `M_long`
  - [x] 支持 batch / mask / variable segment
  - **DoD**：读出 shape 正确，梯度稳定

- [x] 实现 Fuser
  - [x] 支持 `H×d -> K×d`
  - [x] 至少实现一个简单版（MLP / identity when K=H）
  - **DoD**：能输出合法 `M_short`

- [x] 实现 Injector
  - [x] 至少实现 prefix injection
  - [x] 支持通过 config 切换注入开关
  - **DoD**：开关注入会引起可测的生成变化

### P1 重要
- [x] 实现 Query-Gating
  - [x] off / random / learned
  - **DoD**：能记录每段 gating 统计

- [x] 实现多种注入位置
  - [x] segment / delimiter / random / none
  - **DoD**：只改 config 即可切换

- [x] 给方法模块补最小结构约束
  - [x] 输入输出 shape 校验
  - [x] domain / task conditioning 的命名和保存约定
  - **DoD**：常见 shape 错误与错误配置能在早期报错，而不是训练半天后崩

当前 M2 已完成并验证的最小方法骨架：
- `MemoryWriter`: `mlp` / `transformer` 两档实现，已补 `freeze()/unfreeze()/save_to()/load_from()`
- `MemoryReader`: `H` 个 learned queries + cross-attention，支持 `memory_mask`
- `MemoryFuser`: `linear` / `resampler`
- `MemoryInjector`: `enabled` 开关与 `position in {segment, delimiter, random, none}` 已进入 config 契约，且各配置都已真实 smoke
- `Query-Gating`: `off / random / learned` 三档已进入 `gating_mode` 配置契约；`train/eval metrics.json` 会记录 `gating_mode / mean_gate / mean_active_queries / mean_segment_gate / mean_segment_active_queries`，`predictions.jsonl` 会记录每个样本的 `gates` 与 `segment_stats`
- `Conditioning schema`: 固定保存为 `domain_name` 与可选 `task_name`；若数据缺失 `method.reader.conditioning.domain_key`，会在 forward 早期直接报错
- 已验证命令：
  - `python -m unittest discover -s tests -v`
  - `python -m train --config configs/exp/smoke_qwen25_transformer_writer.yaml --seed 123 --output_dir runs/verify/m2-transformer-writer-v2/train`
  - `python -m eval --config configs/exp/smoke_qwen25_transformer_writer.yaml --seed 123 --output_dir runs/verify/m2-transformer-writer-v2/eval --checkpoint runs/verify/m2-transformer-writer-v2/train/checkpoint.pt`
  - `python -m analysis --config configs/exp/smoke_qwen25_transformer_writer.yaml --seed 123 --output_dir results/generated/m2-transformer-writer-summary-v2 --input_root runs/verify/m2-transformer-writer-v2`
  - `python -m train --config configs/exp/smoke_qwen25_transformer_writer_learned_gating.yaml --seed 231 --output_dir runs/verify/m2-learned-gating-v2/train`
  - `python -m eval --config configs/exp/smoke_qwen25_transformer_writer_learned_gating.yaml --seed 231 --output_dir runs/verify/m2-learned-gating-v2/eval --checkpoint runs/verify/m2-learned-gating-v2/train/checkpoint.pt`
  - `python -m train --config configs/exp/smoke_qwen25_transformer_writer_delimiter_injection.yaml --seed 211 --output_dir runs/verify/m2-delimiter-injection/train`
  - `python -m eval --config configs/exp/smoke_qwen25_transformer_writer_delimiter_injection.yaml --seed 211 --output_dir runs/verify/m2-delimiter-injection/eval --checkpoint runs/verify/m2-delimiter-injection/train/checkpoint.pt`
  - `python -m train --config configs/exp/smoke_qwen25_transformer_writer_random_injection.yaml --seed 223 --output_dir runs/verify/m2-random-injection/train`
  - `python -m eval --config configs/exp/smoke_qwen25_transformer_writer_random_injection.yaml --seed 223 --output_dir runs/verify/m2-random-injection/eval --checkpoint runs/verify/m2-random-injection/train/checkpoint.pt`
  - `python -m train --config configs/exp/smoke_qwen25_transformer_writer_no_injection.yaml --seed 227 --output_dir runs/verify/m2-no-injection/train`
  - `python -m eval --config configs/exp/smoke_qwen25_transformer_writer_no_injection.yaml --seed 227 --output_dir runs/verify/m2-no-injection/eval --checkpoint runs/verify/m2-no-injection/train/checkpoint.pt`
  - `python -m analysis --config configs/exp/smoke_qwen25_transformer_writer.yaml --seed 241 --output_dir results/generated/m2-p1-summary --input_root runs/verify`

说明：当前 M2 P1 已满足模块契约与配置切换层面的 DoD；真正服务于论文图表的系统性 gating 频率图、跨段机制图和 Stage B/C 适配比较，仍属于后续 M3/M4 的训练与分析任务。

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
- [x] Stage A：Writer 通用预训
  - [x] 搭建 general-field 数据管线
  - [x] 记录数据版本 / hash / 采样规则
  - [x] 输出冻结可复用的 writer checkpoint
  - **DoD**：能得到 `writer.ckpt`，并在后续实验中加载复用

- [x] Stage B：queries meta-train
  - [x] 实现 episode sampler
  - [x] 优先实现 ANIL（或等价的一阶近似）
  - [x] Writer 固定；queries 是主要更新对象
  - **DoD**：得到 `queries_meta_init.pt`，并能在 source domains 上观察到适配收益

- [x] Stage C：few-shot / 少步数适配
  - [x] 输入：新域 k-shot
  - [x] 输出：适配后的 queries checkpoint + 适配曲线
  - **DoD**：固定一个 target domain，自动产出 shot-curve 与 step-curve

当前 M3 smoke 已完成并验证的部分：
- 已新增 `toy_meta_smoke` 数据与 meta split；当前配置是 `source={math, code, qa}`、`target=narrative`、`support_size=2`、`query_size=2`
- 已新增 benchmark-native `core4_transfer_smoke` meta split 配置：`configs/tasks/benchmarks/meta/core4_transfer_smoke.yaml`
  - `dataset_sources={gsm8k->math, kodcode->code, gpqa->qa, story_cloze->narrative}`
  - 当前 canonical 配置已提升为 `smoke8/3x3`：每个 source 使用 `eval-real-smoke8.jsonl`，并固定 `support_size=3`、`query_size=3`
  - `sampling_policy=uniform_examples`
  - 该配置不再依赖 toy label 结构，可直接跑真实 benchmark smoke 的 `Stage A/B/C`
- Stage A 已真实产出 `writer.ckpt`，并把 `dataset_sha256`、domain 采样规则写入 `meta_data_manifest.json`
- Stage B 已真实产出 `queries_meta_init.pt`，当前实现是 first-order ANIL 近似，inner-loop 更新 `reader.queries + fuser`，Writer 固定
- Stage B 现支持 `query_learning_mode in {meta_trained, non_meta_multitask, random}`，并把模式写入 `queries_meta_init.pt` / `metrics.json`
- Stage B/C 现支持 `runtime.query_objective in {label_prototype, continuation_retrieval}`；其中 benchmark-native `core4` 路径固定走 `continuation_retrieval`
- Stage C 已真实产出 `queries_adapted.pt`、`adapt_curve.csv` 与 `adapt_cost.json`；当前默认 `adaptation_target=q_only`，并已按 `Q-only / W-only / W+Q` 跑通对齐预算的 smoke 曲线
- Stage C 现支持 `expected_query_learning_mode` 校验，避免把 `random / non-meta / meta-trained` 的 reader init resume 混到错误 run 里
- Stage C 的 `adapt_curve.csv` 现已同时写出 `objective_loss / task_score / task_metric_name`，并把 `best_adapt_task_score` 作为 benchmark-native 主曲线字段
- `analysis` 现支持 `analysis_mode=m3_failure_checks`，会显式跑 `zero_memory / writer_noise / collapsed_fuser` 三个 smoke ablation，并输出 `failure_checks.json`、`failure_ablation_summary.csv`、`failure_ablation_summary.svg`
- 已新增 benchmark-native runbook：`scripts/10_pretrain_writer.sh`、`scripts/20_meta_train_queries.sh`、`scripts/30_adapt_queries.sh`
- 已验证命令：
  - `python -m unittest discover -s tests -v`
  - `python -m train --config configs/exp/m3_stage_a_qwen25_smoke.yaml --seed 301 --output_dir runs/verify/m3-stage-a`
  - `python -m train --config configs/exp/m3_stage_b_qwen25_smoke.yaml --seed 303 --output_dir runs/verify/m3-stage-b --resume runs/verify/m3-stage-a`
  - `python -m train --config configs/exp/m3_stage_c_qwen25_smoke.yaml --seed 307 --output_dir runs/verify/m3-stage-c --resume runs/verify/m3-stage-b`
  - `python -m analysis --config configs/exp/m3_stage_c_qwen25_smoke.yaml --seed 307 --output_dir results/generated/m3-smoke-summary --input_root runs/verify`
  - `./scripts/10_pretrain_writer.sh 2501 runs/verify/m3-core4-qwen25/stage-a`
  - `./scripts/20_meta_train_queries.sh 2503 runs/verify/m3-core4-qwen25/stage-b runs/verify/m3-core4-qwen25/stage-a`
  - `./scripts/30_adapt_queries.sh 2507 runs/verify/m3-core4-qwen25/stage-c runs/verify/m3-core4-qwen25/stage-b`
  - `./scripts/10_pretrain_writer.sh 2511 runs/verify/m3-core4-qwen3/stage-a configs/exp/m3_stage_a_core4_qwen3_smoke.yaml`
  - `./scripts/20_meta_train_queries.sh 2513 runs/verify/m3-core4-qwen3/stage-b runs/verify/m3-core4-qwen3/stage-a configs/exp/m3_stage_b_core4_qwen3_smoke.yaml`
  - `./scripts/30_adapt_queries.sh 2517 runs/verify/m3-core4-qwen3/stage-c runs/verify/m3-core4-qwen3/stage-b configs/exp/m3_stage_c_core4_qwen3_smoke.yaml`

最新已验证结果：
- Stage B：`runs/verify/m3-stage-b/metrics.json` 当前记录 `mean_zero_shot_query_loss=0.6781679193178812`、`mean_adapted_query_loss=0.6538897852102915`、`mean_adaptation_gain=0.02427813410758972`
- Stage C：`runs/verify/m3-stage-c/adapt_curve.csv` 当前记录 target domain `narrative` 上从 `zero_shot_query_loss=0.7023470401763916` 下降到 `best_adapt_query_loss=0.6856379508972168`
- benchmark-native `core4` smoke：
  - canonical 配置现已从早期 `smoke4/2x2` 升级为 `smoke8/3x3`
  - `runs/verify/m3-core4-qwen25/stage-b/metrics.json` 当前记录 `query_objective=continuation_retrieval`、`query_candidate_pool_policy=exclude_support_for_query_eval`、`support_candidate_pool_policy=support_only_for_inner_loop`、`source_eval_task_score=0.125`、`mean_adaptation_gain=1.903374989827474e-05`
  - `runs/verify/m3-core4-qwen25/stage-c/metrics.json` 当前记录 `zero_shot_task_score=0.3333333333333333`、`best_adapt_task_score=0.3333333333333333`、`task_metric_name=accuracy`、`best_adapt_query_loss=1.5992218255996704`
  - `runs/verify/m3-core4-qwen3/stage-b/metrics.json` 当前记录 `query_objective=continuation_retrieval`、`query_candidate_pool_policy=exclude_support_for_query_eval`、`support_candidate_pool_policy=support_only_for_inner_loop`、`source_eval_task_score=0.0`、`mean_adaptation_gain=0.0007965167363484701`
  - `runs/verify/m3-core4-qwen3/stage-c/metrics.json` 当前记录 `zero_shot_task_score=1.0`、`best_adapt_task_score=1.0`、`best_adapt_query_loss=1.522208571434021`
- Stage C 适配对象消融：`runs/verify/m3-adaptation-targets-canonical/`
  - `Q-only`：`reader.queries`，`trainable_parameter_count=256`，`0.7023470401763916 -> 0.7023470401763916`
  - `W-only`：`writer`，`trainable_parameter_count=71744`，`0.7023470401763916 -> 0.694838285446167`
  - `W+Q`：`writer+reader.queries`，`trainable_parameter_count=72000`，`0.7023470401763916 -> 0.694838285446167`
- Reader 学习方式消融：`runs/verify/m3-reader-learning-modes-canonical/`
  - `meta-trained`：Stage B `mean_adaptation_gain=0.02427813410758972`；Stage C `zero_shot_query_loss=0.7023470401763916`
  - `non-meta multi-task`：Stage B `source_eval_query_loss=0.6913747191429138`；Stage C `zero_shot_query_loss=0.7048434019088745`
  - `random`：Stage B `source_eval_query_loss=0.6923675537109375`；Stage C `zero_shot_query_loss=0.7098537683486938`
- 训练失败模式显式检查：`results/generated/m3-failure-checks-canonical/`
  - `reader_uses_memory`：通过，`0.6890493258833885 -> 0.7001845389604568`
  - `writer_beats_noise`：通过，`0.6890493258833885 -> 0.6904618516564369`
  - `fuser_avoids_collapse`：未通过，当前 `base_short_slot_diversity≈0`，`collapsed_fuser` 与 base loss 持平
- `Fuser collapse` follow-up：`results/generated/m3-fuser-fix-failure-checks-v2/`
  - fresh Stage B canonical run：`runs/verify/m3-fuser-fix-canonical/stage-b-meta/metrics.json` 当前记录 `mean_zero_shot_query_loss=0.5933724492788315`、`mean_adapted_query_loss=0.5082866847515106`、`mean_adaptation_gain=0.08508576452732086`
  - failure checks 三项现已全部通过：`checks_pass_rate=1.0`
  - `reader_uses_memory`：`0.6569329723715782 -> 0.6758842468261719`
  - `writer_beats_noise`：`0.6569329723715782 -> 0.6875795591622591`，当前 harness 使用 `writer_noise_trials=8` 以降低单次噪声抽样方差
  - `fuser_avoids_collapse`：`base_short_slot_diversity=0.004472408443689346`，`collapsed_fuser_query_loss=0.6617699563503265`

说明：`MAIN_IDEA.md` 与 `EXPERIMENTS_INFO.md` 都把 Stage C 默认口径锁定为“只更新 queries”；因此这里已显式把 `runtime.adaptation_target` 引入配置层，并将默认实现对齐为 `q_only`。此前 code drift 中的 `queries + fuser` 更新方式不再作为 Stage C 默认口径。
说明：当前 canonical toy smoke 上，Reader 学习方式的 target zero-shot loss 呈现 `meta-trained < non-meta < random`，但三者的 `q_only` few-shot accuracy 仍都保持 `0.5`；因此这里完成的是“可直接比较 meta 价值的 harness”，不是论文级结论。
说明：退化模式检查条目现在不只是“显式检查 + smoke ablation harness”，还已经完成了一轮真实 follow-up 修复。当前 canonical follow-up run 中，三项检查均通过，说明这套 harness 既能抓出结构退化，也能验证修复是否真正生效。
说明：benchmark-native `core4` smoke 现在已经打通真实 benchmark 子集上的 `Stage A/B/C` artifact contract、多 source meta-split 与真实 `task_score` 曲线；最新 canonical follow-up 已进一步把 episode 结构提升到 `smoke8/3x3`，并在“query eval 排除 support、support inner-loop 只看 support pool”的 episode-aware retrieval 协议下，把两档 backbone 的 Stage B `mean_adaptation_gain` 都翻成了正值。当前这还只能写成“最小 smoke 证据已成立”，不能写成“source-domain meta gain 已稳定成立”，因为 margin 仍然很小。

说明：当前 M3 P0 的 smoke DoD 已完成，重点是先把 Stage A/B/C 的 artifact contract、resume 链路、meta split、以及“source-domain 有正向适配收益”的最小证据打通。更强的 few-shot 曲线、更多 seeds、以及 target-domain accuracy 提升仍属于后续 M4/M5 的正式实验工作。

### P1 重要
- [x] 适配对象消融
  - [x] Q-only
  - [x] W-only
  - [x] W+Q
  - **DoD**：三条曲线齐全，且对齐相同预算

- [x] Reader 学习方式消融
  - [x] random queries
  - [x] non-meta multi-task queries
  - [x] meta-trained queries
  - **DoD**：能直接比较 meta 的价值

- [x] 把训练失败常见模式写成显式检查
  - [x] Reader 忽略 memory
  - [x] Writer 输出退化为噪声
  - [x] Fuser 折叠为单一 token
  - **DoD**：至少有 2–3 个 smoke ablation 可快速识别退化

### P2 加分
- [ ] query 多样性正则
- [ ] Writer 随机 query 正则预训变体
- [ ] active ExecPlan 模板自动生成脚本

---

## M4. benchmark 接入与统一评测 harness

> 阅读指引：  
> `EXPERIMENTS_INFO.md` 的“任务与数据 / 统一评测与结果汇总规范”是本里程碑的唯一口径来源。

当前已完成的 M4 foundation：
- 已新增统一 benchmark registry / prompt template / `TaskEvaluator` 层，代码位于 `src/memtotal/tasks/`
- 已为 `gsm8k`、`math`、`gpqa`、`triviaqa`、`kodcode`、`story_cloze`、`rocstories`、`fever`、`alfworld` 建立本地 smoke subset 配置与样例数据
- `python -m eval` 现可对 benchmark smoke 统一输出 `benchmark_id / task_domain / smoke_subset / evaluator_type`
- `scripts/run_benchmark_smoke_suite.sh` 已真实跑通 6 个代表任务：`gsm8k / gpqa / kodcode / story_cloze / fever / alfworld`
- 最新汇总位于 `results/generated/m4-benchmark-smoke/20260306T132413Z/summary.csv`
- 已新增真实数据源 registry / materialize 脚本 / manifest：
  - `scripts/setup_benchmark_data.sh`
  - `docs/benchmark-data.md`
  - `data/benchmarks/source_summary.json`
- 已真实 materialize 并跑通 real-source smoke eval 的任务：
  - `gsm8k`
  - `math`
  - `gpqa`
  - `triviaqa`
  - `kodcode`
  - `story_cloze`
  - `narrativeqa`
  - `rocstories`
  - `fever`
  - `alfworld`
  - `memoryagentbench`
- `MemoryAgentBench` 当前已通过真实 HF 数据源打通 4 类能力 smoke：
  - `AR`: `ruler_qa1_197K`
  - `TTL`: `icl_trec_coarse_6600shot_balance`
  - `LRU`: `infbench_sum_eng_shots2`
  - `CR`: `factconsolidation_mh_6k`
- `NarrativeQA` 当前通过官方 `deepmind/narrativeqa` 的 `validation` split 接入 real-source smoke，并已进一步升级为 `runtime-pool question-aware full_text_segmented` 视图：materialize 时会保留完整 `story_chunk_pool`，并用结构化正文起点探测先裁掉明显导论；load/eval 时再按 `task.narrativeqa_runtime.segment_budget=6` 和 `question_aware` selector 选出实际注入的 story chunks；统一评测仍先使用 `qa_f1` 代理指标。
- `qa_f1` 与 `memoryagentbench` 这类生成式任务现在已经改为真正评估 `generated_text`，不再误用空字符串占位；因此 `MemoryAgentBench` smoke 的 capability 分项从 `20260306T153938Z` 这版开始才是有效的统一代理结果。
- `benchmark_narrativeqa_qwen3_real_smoke.yaml` 已新增并真实跑通，当前 NarrativeQA 这条 smoke 路径已经覆盖两档固定 backbone：`Qwen2.5-1.5B-Instruct` 与 `Qwen3-8B`
- `NarrativeQA` 的 runtime selector 现在已补上显式消融：`anchor_only / question_aware / oracle_like_proxy` 三档都能通过同一份 `story_chunk_pool` 运行，相关配置为 `configs/exp/benchmark_narrativeqa_qwen25_real_smoke{,_anchor_only,_oracle_like_proxy}.yaml`，统一汇总位于 `results/generated/m4-narrativeqa-selector-ablations/summary.csv`
- 当前 qwen25 selector smoke 的结果是：`anchor_only mean_similarity=0.0182232353836298`、`question_aware mean_similarity=0.015230493620038033`、`oracle_like_proxy mean_similarity=-0.019175926223397255`；这些数字只用于验证 selector contract 已生效，不代表正式方法优劣
- 同一组 selector 消融现已补到 `Qwen3-8B`：配置为 `configs/exp/benchmark_narrativeqa_qwen3_real_smoke{,_anchor_only,_oracle_like_proxy}.yaml`，统一汇总位于 `results/generated/m4-narrativeqa-selector-ablations-qwen3/summary.csv`；当前 qwen3 stub 结果为 `question_aware=0.03560512885451317`、`anchor_only=-0.011595143005251884`、`oracle_like_proxy=-0.033963803201913834`
- 最新 real-source smoke 汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T163014Z/summary.csv`
- 说明：这部分完成的是“统一任务契约 + smoke subset + 统一 eval harness”，不是正式 benchmark 主结果；`MemoryAgentBench` 当前为了本地 stub-harness 可运行，会把 context 截断到 `512` tokens，`NarrativeQA` 当前也只是“官方 full story -> runtime-selected 6 chunk excerpt”的 smoke 版本，因此两者都不是正式长上下文协议结果

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
  - [x] 尽量再接 NarrativeQA
  - **DoD**：可以构造 `Math+Code` vs `Math+Narrative`

- [x] 接入 MemoryAgentBench（强烈建议）
  - **DoD**：能跑出四类能力分项结果

### P1 重要
- [x] 固化 meta-split 到配置文件
- [x] 固化 task prompt / CoT / 工具模板
- [x] 写清数据许可与下载路径
- [x] 每个 benchmark 至少提供一个 smoke subset

### P2 加分
- [ ] LoCoMo / LongMemEval
- [ ] 更多 agent memory benchmarks
- [ ] benchmark artifact 体积与缓存清理策略

---

## M5. baseline 全家桶：把“meta-read queries”的优势钉死

> 阅读指引：  
> `EXPERIMENTS_INFO.md` 的 Baselines、预算对齐与 CDMI 组是本里程碑的唯一口径来源。

### P0 必须
- [x] Vanilla
- [x] CoT
- [x] MemGen
- [x] Prompt Tuning
- [x] LoRA
- [x] MetaPrompting
- **DoD**：这些 baseline 都能在同一套 shot/step 网格下运行，并进入统一汇总器

当前进展：
- 已新增最小 `prompting` baseline family，支持 `Vanilla / CoT`
- 统一入口仍是 `python -m eval --config ...`
- 当前已真实 smoke 验证 `GSM8K + Story Cloze`，汇总位于 `results/generated/m5-prompt-baseline-smoke/summary.csv`
- 同一套 `Vanilla / CoT` smoke 现已补到 `Qwen3-8B`，汇总位于 `results/generated/m5-prompt-baseline-smoke-qwen3/summary.csv`
- 同一套 `Vanilla / CoT` 现已进一步补到 real-source smoke：qwen25 汇总位于 `results/generated/m5-prompt-baseline-real-smoke/summary.csv`，qwen3 汇总位于 `results/generated/m5-prompt-baseline-real-smoke-qwen3/summary.csv`
- 已新增最小 `adapter` baseline family，支持 `Prompt Tuning / LoRA / IA3 / Prefix Tuning` 的 `train -> checkpoint -> eval -> summary` 闭环；当前先在 `story_cloze` qwen25 smoke 上真实验证，汇总位于 `results/generated/m5-adapter-baseline-smoke/summary.csv`
- 同一套 adapter smoke 现已补到 `Qwen3-8B`，汇总位于 `results/generated/m5-adapter-baseline-smoke-qwen3/summary.csv`
- 同一套 adapter 现已进一步补到 `story_cloze` real-source smoke；历史汇总位于 `results/generated/m5-adapter-baseline-real-smoke/summary.csv`
- 已新增最小 `MetaPrompting` scaffold，当前先在 `story_cloze` 的 qwen25/qwen3 smoke 上真实验证，汇总位于 `results/generated/m5-metaprompting-smoke/summary.csv`
- 同一套 `MetaPrompting` 现已进一步补到 `story_cloze` real-source smoke，统一汇总位于 `results/generated/m5-metaprompting-real-smoke/summary.csv`
- `prompting / meta_prompting` 当前已支持 `support_examples > 0` 的 in-context few-shot demos，并已在 `story_cloze` real-source 上真实跑通两档 backbone 的 `2-shot` smoke，汇总位于 `results/generated/m5-prompt-fewshot-real-smoke/summary.csv`
- 已新增 `baseline_budget_audit`，当前会自动检查 `prompting / meta_prompting / adapter` 的预算字段完整性与双 backbone 覆盖，已验证汇总位于 `results/generated/m5-baseline-budget-audit/summary.csv`
- `baseline_budget_audit` 当前已进一步扩到 `prompting / meta_prompting / adapter / rag / lightthinker / memory_bank` 六个 family；最新 `metrics.json` 为 `rows_collected=60`、`checks_pass_rate=1.0`
- 已新增最小 `story_cloze` baseline grid smoke suite：单个命令会在同一套 suite 内循环 `shots={0,2}`、`steps={0,4}`，并产出 `adapt_curve.csv / adapt_cost.json / summary.csv`，结果位于 `results/generated/m5-story-cloze-baseline-grid-smoke/`
- 同一套 grid 现已支持导入外部 baseline 点；当前已把 `MemGen` 的 `story_cloze / Qwen2.5-1.5B-Instruct / 0-shot / 0-step` 评测结果导入到 `results/generated/m5-story-cloze-baseline-grid-with-memgen-smoke/`
- 已新增更接近协议的 `story_cloze` protocol-smoke grid：在 `smoke8` real-source 子集上跑 `shots={0,1,2,4}`、`steps={0,1,3,5}`，并保留同一条曲线里的 `MemGen` 外部零样本点；结果位于 `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/`
- 已新增最小 `memory_bank` baseline family，支持 `family=memory_bank`、`mode=episodic_bank`，并已真实跑通 `story_cloze` real-source smoke 的双 backbone 配置：
  - qwen25: `runs/verify/baseline_memory_bank_story_cloze_qwen25_real_smoke/metrics.json`
  - qwen3: `runs/verify/baseline_memory_bank_story_cloze_qwen3_real_smoke/metrics.json`
- `memory_bank` 当前已接入 baseline grid 与 budget audit：
  - minimal grid: `results/generated/m5-story-cloze-baseline-grid-smoke/`
  - protocol-smoke grid: `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/`
  - budget audit: `results/generated/m5-baseline-budget-audit/summary.csv`
- adapter family 当前已新增 `IA3` 变体，支持 `mode=ia3`，并已真实跑通 `story_cloze` real-source smoke 的双 backbone 配置：
  - qwen25: `runs/verify/baseline_ia3_story_cloze_qwen25_real_smoke/{train,eval}/metrics.json`
  - qwen3: `runs/verify/baseline_ia3_story_cloze_qwen3_real_smoke/{train,eval}/metrics.json`
- adapter family 当前已进一步新增 `Prefix Tuning` 变体，支持 `mode=prefix_tuning`，并已真实跑通 `story_cloze` real-source smoke 的双 backbone 配置：
  - qwen25: `runs/verify/baseline_prefix_tuning_story_cloze_qwen25_real_smoke/{train,eval}/metrics.json`
  - qwen3: `runs/verify/baseline_prefix_tuning_story_cloze_qwen3_real_smoke/{train,eval}/metrics.json`
- `Prefix Tuning` 当前参数量已进入统一预算口径：当前 `trainable_parameter_count=4416`
- adapter grid 当前已更新到 `variant_count=20`：
  - minimal grid: `cell_count=48`、`train_run_count=24`、`eval_run_count=48`
  - protocol-smoke grid: `cell_count=152`、`imported_eval_count=1`
  - dual-import protocol grid: `cell_count=152`、`imported_eval_count=2`
- protocol 与 dual-import protocol 当前都已在同一目录重跑验证缓存复用：最新 `adapt_cost.json` 为 `train_run_count=0`、`eval_run_count=0`、`reused_train_run_count=104`、`reused_eval_run_count=152`
- `MemGen / story_cloze / Qwen3-8B` 真实 smoke 现已完成，`runs/verify/memgen-story-cloze-qwen3-smoke-v2/metrics.json` 当前记录 `compute_reward=1.0`
- dual-import protocol suite 现已把 `MemGen / Qwen2.5-1.5B-Instruct` 与 `MemGen / Qwen3-8B` 两个 `0-shot / 0-step` 外部点同时导入到 `results/generated/m5-story-cloze-baseline-grid-protocol-with-memgen-dual-smoke/`
- 当前 dual-import suite 的 `adapt_cost.json` 为 `cell_count=152`、`variant_count=20`、`imported_eval_count=2`、`skipped_import_count=0`、`reused_train_run_count=104`、`reused_eval_run_count=152`
- 说明：`M5 / P0` 当前已经在 protocol-smoke contract 层面满足 DoD；这表示 baseline family 已能进入同一套 `shot/step` 网格与统一汇总，不表示主表级 baseline 已完成

### P1 重要
- [x] LightThinker
- [x] 外部记忆 / RAG 强 baseline（如果主表需要）
- **DoD**：至少有一条非 internal memory 路线可与我们对照

当前进展：
- 已新增最小 `lightthinker` baseline family，支持 `family=lightthinker`、`mode=compress_then_answer`
- 当前已真实跑通 `story_cloze` real-source smoke：
  - qwen25: `runs/verify/baseline_lightthinker_story_cloze_qwen25_real_smoke/metrics.json`
  - qwen3: `runs/verify/baseline_lightthinker_story_cloze_qwen3_real_smoke/metrics.json`
- 当前会额外写出 `mean_thought_sketch_tokens / lightthinker_compression_prompt / lightthinker_thought_sketch`
- `lightthinker` 当前已进一步接入 `story_cloze` baseline grid：
  - minimal grid: `results/generated/m5-story-cloze-baseline-grid-smoke/`
  - protocol-smoke grid: `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/`
- 已新增最小 `rag` baseline family，支持 `family=rag`、`mode=retrieval_augmented`，当前提供 `lexical_overlap / dense_stub` 两档 retriever
- 当前已真实跑通 `story_cloze` real-source smoke：
  - qwen25: `runs/verify/baseline_rag_story_cloze_qwen25_real_smoke/metrics.json`
  - qwen3: `runs/verify/baseline_rag_story_cloze_qwen3_real_smoke/metrics.json`
- `rag` 现已接入统一 baseline grid 与 budget audit：
  - protocol-smoke grid: `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/`
  - budget audit: `results/generated/m5-baseline-budget-audit/summary.csv`
- 说明：这一步完成的是“至少一条外部记忆路线可进入统一评测与 grid 对照”，还不是更强的 `MemoryBank / ExpeL / AWM` 全家桶复现
- 说明：`LightThinker` 当前也只是“最小统一 scaffold”，不是正式论文级复现

### P2 加分
- [x] 更完整的 memory agent baseline 家族
- [x] 更多 prompt-based / adapter-based meta-learning variants
- [x] baseline 自动预算检查脚本

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
