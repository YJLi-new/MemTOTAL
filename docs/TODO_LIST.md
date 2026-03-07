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
- Stage C 的 `adapt_curve.csv` 现还会额外写出 `task_proxy_score / task_proxy_name / task_margin`；当前 multiple-choice 任务的默认 proxy 是 `gold_choice_probability`
- 已新增真实 `Qwen2.5-1.5B-Instruct` 的 `story_cloze` decision-interface pilot：
  - `src/memtotal/training/m3_real_pilot.py`
  - `src/memtotal/analysis/story_cloze_real_pilot.py`
  - `scripts/run_m3_story_cloze_real_pilot_qwen25.sh`
  - 固定流程是 `screen256 -> split -> A/B screening -> fixed100 -> A/B/C/D/E compare`
- 这轮 real qwen25 pilot 的当前结果已经明确：
  - `screen248` 上 `A=base_only` 为 `accuracy=0.6411290322580645`
  - 但 hard `fixed100` 上 `A/B/C/D/E` 全部是 `task_score=0.2`
  - `A -> C` 只有极小 `mean_margin_gain=0.0016285324096679688`、`mean_proxy_gain=0.00023484499303563528`
  - `C -> D` 与 `C -> E` 的 `mean_task_gain` 都是 `0.0`
- 因而，这轮已经能下一个明确结论：
  - 当前这版 `candidate_conditioned_late_fusion` 在真实 qwen25 上还没有带来任何 `story_cloze` choice flip
  - 当前也还看不出真实 memory 内容效应，`real memory` 与 `shuffled memory` 几乎重合
  - 顺着“推力不够”做的全局 `support_grid_search` 也已经试过：看 `pilot-support8` 会把 `alpha` 压到 `0`，换成额外的 `calibration-hard32` 也仍然回到 `alpha=1`
  - 下一步更值得做的是 competitor-aware inner-loop / conditional residual calibration，而不是继续扫全局 loss/sample
- 已新增 `story_cloze` 的离线 oracle 审计：
  - `configs/exp/stage_c_real_pilot_oracle_audit.yaml`
  - `scripts/run_m3_story_cloze_real_oracle_audit.sh`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/oracle/`
- oracle 当前给出的结论非常关键：
  - `best-of-two oracle` 仍然是 `0.2`
  - 但 `per-case alpha oracle` 达到 `0.96`
  - `80` 个 base-wrong cases 里，有 `76` 个能被离线 `alpha_i` 翻正
  - 但这 `76` 个全部需要 `|alpha| >= 32`，而且正负号几乎对半分：`+37 / -39`
  - 因而当前更像是“residual family 里有 signal，但缺 case-conditional routing / sign selection”，而不是“单一全局 alpha 不够大”
- 已新增 `FEVER` 的 real qwen25 control pilot：
  - `configs/exp/fever_real_pilot_split.yaml`
  - `configs/exp/fever_real_fixed_set_builder.yaml`
  - `configs/exp/stage_c_real_pilot_compare_fever.yaml`
  - `scripts/run_m3_fever_real_pilot_qwen25.sh`
  - review 结果位于 `runs/review/m3-fever-real-pilot-qwen25/` 与 `results/generated/review/m3-fever-real-pilot-qwen25/`
- FEVER control 当前给出的结论也已经明确：
  - `A=base_only=0.25`
  - `B=base+shared_summary residual=0.75`
  - `C=base+candidate_conditioned residual=0.25`
  - `D=base+candidate_conditioned residual+shuffled memory=0.25`
  - `A -> B` 当前有 `flip_count_delta=32`
  - 但 `A -> C` 与 `C -> D` 都是 `0`
- 已完成更保守的 `shared + candidate delta` 修补：
  - `F=base+shared residual+candidate delta`
  - `G=base+shared residual+candidate delta+shuffled memory`
  - `Story Cloze` 上 `B -> F` 只有极小 `mean_margin_gain=0.0005981`，但 `flip_count_delta=0`
  - `Story Cloze` 上 `F -> G=0`
  - `FEVER` 上 `F=0.671875`，虽然明显优于旧的 `C=0.25`，但仍低于 `B=0.75`
  - `FEVER` 上 `B -> F flip_count_delta=-5`
  - `FEVER` 上 `F -> G=0`
- 因而，这轮判别实验后的最稳妥结论已经变成：
  - `Story Cloze` 仍更像 artifact-heavy stress test，不适合作为 memory idea 的单一生死判官
  - 当前 low-bandwidth residual family 不是整体死亡，因为 `shared_summary residual` 在 FEVER 上是 load-bearing 的
  - 旧的 `candidate_conditioned late fusion` 已被判定为坏分支；新的 `shared + candidate delta` 只把它修到了“不会完全乱来”
  - 但当前 candidate 增量仍然没有显出 real-memory 内容效应，因为 `real` 与 `shuffled` 仍重合
  - 因而不能直接跳到 routing / sign selection，因为还没有先证明 `F-G` 是可用的 memory-only signal
  - 已补完 `F-G` content audit：`Story Cloze` 上 `B + (F-G)=0.2`、`oracle_per_case_alpha_content=0.2`；`FEVER` 上 `B + (F-G)=0.75`、`oracle_per_case_alpha_content=0.75`
  - `FEVER` 上虽然 `best_of_BF=0.78125`，但 `best_of_B_plus_content=0.75`，说明补出来的那一点来自 branch form，而不是 real-memory content
  - 当前 candidate residual family 的主体效应来自 branch form，不来自 memory content
  - 下一步不该继续修这条 `shared + candidate delta`，也不该直接上 pairwise / routing / qwen3
  - 若后续仍要继续做 candidate-specific `Stage C`，应从新的 residual family 重新开始，并且先在 `FEVER` 上过 capability gate，再回 `Story Cloze`
- 已完成 `FEVER-first` 的 `choice_repair_ce_margin` fresh pilot：
  - 历史 `B-old=shared_summary+continuation_retrieval=0.75`
  - fresh `B-newObj=shared_summary+choice_repair_ce_margin=0.75`
  - fresh `R-real=shared+candidate_conditioned+repair objective+real memory=0.25`
  - fresh `R-shuffle=0.25`
  - fresh `R-zero=0.25`
  - `B-old -> B-newObj flip_count_delta=0`
  - `B-newObj -> R-real flip_count_delta=-32`
  - `R-shuffle -> R-real flip_count_delta=0`
  - `R-zero -> R-real flip_count_delta=0`
  - `gate_passed=false`
- 因而，这条 candidate-conditioned family 的结论又进一步收紧了一层：
  - 问题已经不只是 `continuation_retrieval` objective 不对
  - 即便换成更对题的 repair objective，这条 current `candidate-conditioned residual family` 也仍然不 load-bearing
  - 下一步不该继续修 current family，也不该直接上 `Qwen3-8B` / routing / sign selection
  - 如果还要继续 candidate-specific `Stage C`，应直接换 residual family，并继续先拿 `FEVER` 做 capability gate
- 已完成 `M4` 的 `FEVER-first shared generative injection` 三段式 gate：
  - 已新增 `teacher-text upper bound`、`writer information audit`、`shared latent prefix injection` 的真实 qwen25 scaffold
  - `writer information audit` 不是单一线性 probe；已同时包含 `linear + shallow MLP` 两档 probe，并显式比较 `real / shuffle / zero`
  - `LatentPrefixProjector` 与 `score_continuations(prefix_embeddings=...)` 的梯度链路已经真实跑通
  - fresh `results/generated/review/m4-fever-shared-injection-qwen25/phase0-gate-sweep/metrics.json` 当前记录：
    - `phase0_gate_passed=true`
    - `selected_prompt_variant=answer_slot_labels`
    - `selected_support_serialization=example_blocks_raw8`
    - `selected_pair_accuracy_gain=0.4274193548387097`
    - `selected_pair_macro_f1_gain=0.5351999379825547`
  - fresh `results/generated/review/m4-fever-shared-injection-qwen25/phase1-writer-audit/metrics.json` 当前记录：
    - `label_probe_passed=true`
    - `semantic_probe_passed=true`
    - `phase1_probe_passed=true`
    - `phase1_gate_passed=true`
  - fresh `results/generated/review/m4-fever-shared-injection-qwen25/phase2-compare/metrics.json` 当前记录：
    - `gate_passed=false`
    - `A=0.25 / macro_f1=0.2`
    - `T=0.53125 / macro_f1=0.5294117647058824`
    - `I-real=0.390625 / macro_f1=0.40613432046536224`
    - `I-shuffle=0.546875 / macro_f1=0.503078982597055`
    - `I-zero=0.25 / macro_f1=0.2`
  - 也就是，这轮已经不再支持“prefix 主链路完全没动”这个解释；当前 immediate blocker 已继续收缩成：
    - frozen Qwen 已经开始消费 prefix，因为 `I-real > I-zero`
    - 但 current real support / writer family 给出的方向仍然错于 `shuffle`，因为 `I-shuffle > I-real`
  - 下一步不该直接跳到 `Story Cloze / candidate-conditioned injection / Qwen3 / KL`；应先继续做 main-chain injection 的内容方向诊断与升级
- Stage C canonical `core4` 配置现已加入 `runtime.target_eval_repeats=3`；`adapt_curve.csv` 会同步写出 `target_eval_repeats / evaluated_query_examples`，用于把单一 target query 子集上的偶然波动与真正的 official few-shot 提升区分开
- `analysis` 现支持 `analysis_mode=m3_failure_checks`，会显式跑 `zero_memory / writer_noise / collapsed_fuser` 三个 smoke ablation，并输出 `failure_checks.json`、`failure_ablation_summary.csv`、`failure_ablation_summary.svg`
- 已新增 benchmark-native runbook：`scripts/10_pretrain_writer.sh`、`scripts/20_meta_train_queries.sh`、`scripts/30_adapt_queries.sh`
- 已新增 benchmark-native `Stage B probe` harness：`scripts/run_m3_core4_stage_b_probe_suite.sh` + `configs/exp/m3_stage_b_probe_summary.yaml`
  - 原始 probe runs 默认落数据盘
  - `probe_summary.csv/.svg` 和 `best_by_backbone` 写回仓库
  - 复用逻辑现已校验 `config.snapshot + seed`，不会把错配置的旧 probe 静默复用
- 已新增 benchmark-native `Stage C probe` harness：`scripts/run_m3_core4_stage_c_probe_suite.sh` + `configs/exp/m3_stage_c_probe_summary.yaml`
  - 原始 probe runs 默认落数据盘
  - 同一 backbone 下 `q_only / w_only / w_plus_q` 现在强制共用同一个 seed，避免 target episode 漂移
  - `probe_summary.csv/.svg` 会把 Stage C 曲线和 q-only gradient audit 合到同一份 summary
- 已新增 benchmark-native `Stage C q-only budget probe` harness：`scripts/run_m3_core4_stage_c_qonly_budget_probe_suite.sh`
  - 当前固定扫描 `adapt_learning_rate in {0.2, 1.0, 5.0}` 与 `adapt_steps in {3, 10}`
  - 同一 backbone 下所有 budget variant 也强制共用同一个 seed
- 已新增 benchmark-native `Stage C q-only seed sweep` harness：`scripts/run_m3_core4_stage_c_qonly_seed_sweep.sh` + `configs/exp/m3_stage_c_seed_sweep_summary.yaml`
  - 当前固定跑 canonical `q_only` 配置，并在每个 backbone 上扫描 5 个 target seeds
  - 统一汇总会产出 `seed_sweep.csv/.svg` 与 `positive_gain_rate / mean_task_gain`
- canonical `Stage C` 现进一步加入 `runtime.target_episode_repeats=3`
  - 当前每个 `shot/step` 点会在 3 组 target support/query episodes 上聚合
  - `adapt_curve.csv` 现同步写出 `target_episode_repeats / evaluated_target_episodes`
- 已新增 benchmark-native `Stage C sensitivity audit`：`scripts/run_m3_core4_stage_c_sensitivity_audit.sh`
- 已新增 benchmark-native `Stage C q-only target-split sweep` harness：`scripts/run_m3_core4_stage_c_qonly_target_split_sweep.sh`
  - 固定 `aggregate_support + ep3 + uniform`，对比 `target_split_policy in {random, proxy_topk_support, proxy_bottomk_support}`
  - 注意：以上 v1 sweep 基于旧版 `shot`-耦合 eval/support protocol；该口径已在后续里程碑中判定存在评测泄漏
  - 公平 fixed-holdout 重扫 `results/generated/m3-core4-stage-c-qonly-target-split-sweep-v2-fixed-eval/metrics.json` 现显示：三档 split 在两档 backbone 上的 official `mean_task_gain` 全部为 `0.0`
  - qwen25 三档 `mean_proxy_gain` 仅在 `2.416e-6 ~ 2.419e-6` 间波动；qwen3 三档仅在 `2.554e-6 ~ 2.560e-6` 间波动
  - 因此 `target_split_policy` 现已从 canonical blocker 列表中拿掉，默认配置回收为最朴素的 `random`
- 已新增 benchmark-native `Stage C q-only support-bank sweep` harness：`scripts/run_m3_core4_stage_c_qonly_support_bank_sweep.sh`
  - 固定公平 fixed-holdout eval 与 `target_split_policy=random`，对比 `target_support_bank_size in {max_shot, auto}`
  - fresh `results/generated/m3-core4-stage-c-qonly-support-bank-sweep-v1/metrics.json` 当前显示：两档 backbone 的 official `mean_task_gain` 仍全部为 `0.0`
  - 但 qwen3 的 `mean_proxy_gain` 已从 `7.294e-7` 提升到 `9.086e-6`，并且 5 个 seeds 里有 4 个从 `best_step=0` 推到 `best_step=3`
  - qwen25 这条线上暂时只看到极弱 proxy 波动，`auto` 与 `max_shot` 仍基本打平
  - 因此 canonical `target_support_bank_size` 现保留为 `auto`，但新的 blocker 也更明确：bank 变大本身还不够，下一步要直接扩 support negative pool
- 已新增 benchmark-native `Stage C q-only support-negative-pool sweep` harness：`scripts/run_m3_core4_stage_c_qonly_support_negative_pool_sweep.sh`
  - 固定公平 fixed-holdout eval、`target_split_policy=random`、`target_support_bank_size=auto`，对比 `target_support_negative_pool in {support_bank, source_plus_support_bank}`
  - fresh `results/generated/m3-core4-stage-c-qonly-support-negative-pool-sweep-v1/metrics.json` 当前显示：`source_plus_support_bank` 在两档 backbone 上都优于 `support_bank`
  - qwen25：official `mean_task_gain` 仍为 `0.0`，但 `mean_proxy_gain` 从 `6.080e-6` 提升到 `1.466e-5`，并把 `best_step=3` 的种子数从 `1/5` 推到 `4/5`
  - qwen3：`mean_task_gain` 从 `0.0` 提升到 `0.02222222222222222`，`positive_gain_rate` 从 `0.0` 提升到 `0.2`，`mean_proxy_gain` 也从 `6.798e-6` 提升到 `1.775e-5`
  - 因此 canonical `target_support_negative_pool` 现已切到 `source_plus_support_bank`
- 已新增 benchmark-native `Stage C q-only support-negative-sampler sweep` harness：`scripts/run_m3_core4_stage_c_qonly_support_negative_sampler_sweep.sh`
  - 固定公平 fixed-holdout eval、`target_split_policy=random`、`target_support_bank_size=auto`、`target_support_negative_pool=source_plus_support_bank`，当前 fresh `results/generated/m3-core4-stage-c-qonly-support-negative-sampler-sweep-v2/metrics.json` 已对比 `target_support_negative_sampler in {deterministic_id, hard_by_continuation, hard_by_current_model}`
  - `hard_by_continuation` 仍不足以继续抬高 official `mean_task_gain`
 - `hard_by_current_model` 当前则在两档 backbone 上都给出最高 `mean_proxy_gain`
 - qwen25：`deterministic_id=1.106e-5 < hard_by_continuation=1.950e-5 < hard_by_current_model=2.561e-5`
 - qwen3：`hard_by_continuation=3.102e-5 < deterministic_id=3.389e-5 < hard_by_current_model=4.391e-5`
  - 但三档 sampler 在这轮 5-seed 上的 official `mean_task_gain` 仍全部为 `0.0`
  - 因此 canonical `target_support_negative_sampler` 现切到 `hard_by_current_model`，并把它作为下一步继续攻 official gain 的主 probe 口径
- 已用新 canonical `target_support_negative_sampler=hard_by_current_model` 重跑 `Stage C curve suite / step saturation audit`
  - fresh `results/generated/m3-core4-stage-c-step-saturation-audit-v2/metrics.json` 当前显示：两档 backbone 的 `mean_step0_to_final_task_gain` 仍然都为 `0.0`
  - 但 fresh `results/generated/m3-core4-stage-c-curve-suite-v2/step_curve.csv` 同时显示：两档 backbone 的 `mean_task_proxy_gain` 都会随着 `step=0 -> 5` 单调上升
  - qwen25：`0.5002879 -> 0.5003087`
  - qwen3：`0.5010271 -> 0.5010556`
  - 这说明当前 blocker 已进一步收缩成“为什么 proxy 改善始终无法跨过 official rank-flip 阈值”，下一步应直接看 margin / rank-flip audit，而不是再怀疑 inner-loop 是否完全无效
- 已新增 `Stage C margin / rank-flip audit`：`scripts/run_m3_core4_stage_c_margin_audit.sh`
  - fresh `results/generated/m3-core4-stage-c-margin-audit-v1/metrics.json` 当前显示：两档 backbone 的 `cross_zero_margin_rate` 都仍然是 `0.0`
  - 同时两档 backbone 的 `margin_improves_rate` 都是 `0.6`
  - qwen25：`mean_zero_shot_task_margin=0.0010697 -> mean_final_task_margin=0.0011538`
  - qwen3：`mean_zero_shot_task_margin=0.0041197 -> mean_final_task_margin=0.0042338`
  - 这说明后续 steps 现在更多是在放大已经接近正确或已经正确的 case 的 margin，而不是把原本错误的 seed 真正翻到 0 以上
- 已把 `margin audit` 进一步拆成 conditional `negative_only` 摘要：fresh `results/generated/m3-core4-stage-c-margin-audit-v2/metrics.json`
  - qwen25：只有 `2` 个负 margin seeds，`negative_only margin_improves_rate=0.5`，`mean_margin_gap_closed=1.8814e-5`
  - qwen3：同样只有 `2` 个负 margin seeds，`negative_only margin_improves_rate=0.5`，但 `mean_margin_gap_closed=-4.8262e-5`
  - 两档 backbone 的 `negative_only cross_zero_margin_rate` 仍都为 `0.0`
  - 这说明当前 canonical `hard_by_current_model` 虽然能持续抬 proxy，但增益还没有稳定集中到真正错误的 seeds 上；下一步应直接做负 margin seeds 的条件化 shot/step 分析
- 已新增 `Stage C negative-seed curve audit`：`scripts/run_m3_core4_stage_c_negative_seed_curve_audit.sh`
  - fresh `results/generated/m3-core4-stage-c-negative-seed-curve-audit-v1/metrics.json` 当前显示：负 margin seeds 上 `shot` 本身几乎没有纯作用，`shot=0/1/2/3` 的 `mean_margin_gap_to_flip` 完全不变
  - qwen25：`mean_zero_shot_gap_to_flip = mean_max_shot_step0_gap_to_flip = 0.07331074049903287`，但 `step=0 -> 5` 后会极弱下降到 `0.07329192648952207`
  - qwen3：`mean_zero_shot_gap_to_flip = mean_max_shot_step0_gap_to_flip = 0.0828861221153703`，而 `step=0 -> 5` 后反而升到 `0.08293438402728902`
  - 这说明当前的实际杠杆不是“多给 shot”，而是“step 在负 margin seeds 上朝哪个方向走”；下一步应直接检查 qwen3 的负 margin seeds 为什么朝错误方向更新
- 已新增 `Stage C q-only retrieval-negative-count sweep`：`scripts/run_m3_core4_stage_c_qonly_negative_count_sweep.sh`
  - fresh `results/generated/m3-core4-stage-c-qonly-negative-count-sweep-v1/metrics.json` 当前显示：`retrieval_negative_count={3,7,15}` 会改变 proxy，但不会改变 official gain；三档在两档 backbone 上的 `mean_task_gain` 都仍为 `0.0`
  - qwen25 当前是 `neg15 > neg7 > neg3`；qwen3 当前是 `neg15 > neg3 > neg7`
  - 因此 negative-count 已从 canonical blocker 列表中拿掉；下一步不应再在这条线上盲扫
- 已新增 `Stage C q-only retrieval-loss sweep`：`scripts/run_m3_core4_stage_c_qonly_retrieval_loss_sweep.sh`
  - fresh `results/generated/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1/metrics.json` 当前显示：`retrieval_loss_type=cross_entropy_plus_margin` 是当前最强的 fair loss 变体
  - qwen25：`mean_proxy_gain=4.921e-05 -> 1.015e-04`，`mean_margin_gain=1.974e-04 -> 4.072e-04`
  - qwen3：`mean_proxy_gain=3.531e-05 -> 7.791e-05`，`mean_margin_gain=1.422e-04 -> 3.139e-04`
  - 但三档 loss 在两档 backbone 上的 official `mean_task_gain` 仍全部为 `0.0`
  - 因此 canonical `Stage C` 现已切到 `retrieval_loss_type=cross_entropy_plus_margin`、`retrieval_margin_value=0.1`
- 已用新的 canonical `retrieval_loss_type=cross_entropy_plus_margin` 重跑 q-only 5-seed sweep：`results/generated/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical/metrics.json`
  - 当前两档 backbone 仍都是 `mean_task_gain=0.0`、`positive_gain_rate=0.0`
  - 也就是说，当前 blocker 已继续收缩成“如何把更强的 proxy / margin gain 推过 official rank-flip 阈值”，而不是“该用哪一类 retrieval loss”
- 已新增 benchmark-native `Stage C` case-level error attribution：`scripts/run_m3_core4_stage_c_error_attribution.sh`
  - `Stage C` 现会直接产出 `task_case_dump.jsonl`，把每个 target case 的 `gold / competitor / margin / support_ids` 落盘
  - fresh canonical replay `results/generated/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump/metrics.json` 当前与 v5 一样，仍是两档 backbone `mean_task_gain=0.0`
  - 但 `results/generated/m3-core4-stage-c-error-attribution-v1/metrics.json` 现给出更具体的 blocker：
    - 61 个配对 case 里真正 near-threshold 但没翻正的只有 2 个
    - qwen25 还有 9 个 `improving_but_unflipped` cases，qwen3 只有 1 个
    - 很多 remaining wrong cases 会被打上 `story_context_favors_competitor`
  - 因此下一步不应再扫全局 loss/sample 策略，而应优先做 near-threshold 非线性推力和 case-level targeted objective
- 已新增 benchmark-native `Stage C curve suite` harness：`scripts/run_m3_core4_stage_c_curve_suite.sh`
  - 单个 seed/run 直接产出更接近正式协议的 `adapt_shots={0,1,2,3}`、`adapt_steps=5` 曲线
  - 分析层会自动汇总 `curve_rows.csv`、`shot_curve.csv/.svg`、`step_curve.csv/.svg`
  - 修正协议后的 `results/generated/m3-core4-stage-c-curve-suite-v2-fixed-eval/shot_curve.csv` 当前显示：两档 backbone 的 official `shot_curve` 已全部打平
    - qwen25：`0/1/2/3-shot` 的 `mean_task_gain` 全部为 `0.0`
    - qwen3：`0/1/2/3-shot` 的 `mean_task_gain` 全部为 `0.0`
  - 同目录 `step_curve.csv` 也显示：在公平 fixed-holdout eval 下，两档 backbone 的 `step=0..5` official `mean_task_gain` 也全部为 `0.0`
- 已新增 benchmark-native `Stage C step saturation audit`：`scripts/run_m3_core4_stage_c_step_saturation_audit.sh`
  - 会把 canonical curve suite 拆成 `zero->step0` 与 `step0->final` 两段收益
  - 修正协议后的 `results/generated/m3-core4-stage-c-step-saturation-audit-v2-fixed-eval/metrics.json` 当前显示：
    - qwen25：`mean_zero_to_step0_task_gain=0.0`，`mean_step0_to_final_task_gain=0.0`
    - qwen3：`mean_zero_to_step0_task_gain=0.0`，`mean_step0_to_final_task_gain=0.0`
  - 也就是说，旧版 “收益全部在 `zero->step0`” 的说法本身也是评测泄漏副产物；当前真实 blocker 已收缩成“为什么公平 eval 下 official few-shot gain 完全为 0”
  - 当前直接比较 `query shift` 与 `memory shift` 对 `readouts / summary / candidate scores` 的影响量级
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
  - qwen25 canonical 当前已提升到 `meta_episodes=16`
  - `runs/verify/m3-core4-qwen25/stage-b/metrics.json` 当前记录 `query_objective=continuation_retrieval`、`query_candidate_pool_policy=exclude_support_for_query_eval`、`support_candidate_pool_policy=support_only_for_inner_loop`、`meta_episodes=16`、`source_eval_task_score=0.125`、`mean_adaptation_gain=6.527453660964966e-05`
  - qwen3 canonical 当前保持 `meta_episodes=6`
  - `runs/verify/m3-core4-qwen3/stage-b/metrics.json` 当前记录 `query_objective=continuation_retrieval`、`query_candidate_pool_policy=exclude_support_for_query_eval`、`support_candidate_pool_policy=support_only_for_inner_loop`、`meta_episodes=6`、`source_eval_task_score=0.0`、`mean_adaptation_gain=0.0007965167363484701`
  - `results/generated/m3-core4-stage-b-probe-suite-v2/metrics.json` 当前记录 `best_by_backbone`：qwen25 最优为 `qwen25-canonical(meta_episodes=16, meta_learning_rate=0.05)`，qwen3 最优为 `qwen3-canonical(meta_episodes=6, meta_learning_rate=0.05)`
  - `method.reader.query_residual_scale` 已作为默认 reader 参数化修补进入 `configs/method/memory_bootstrap.yaml`，当前默认值为 `1.0`
  - 旧的 `runs/verify/m3-core4-{qwen25,qwen3}/stage-c` 结果仍保留作“修补前基线”，但不再代表当前 q-only 参数化状态
  - `results/generated/m3-stage-c-sensitivity-audit/qwen25/metrics.json` 当前记录 `mean_query_delta_score_norm=0.017036181564132374`、`mean_memory_delta_score_norm=0.022845592349767685`、`query_to_memory_score_delta_ratio=0.7457097764552213`
  - `results/generated/m3-stage-c-sensitivity-audit/qwen3/metrics.json` 当前记录 `mean_query_delta_score_norm=0.029695838689804077`、`mean_memory_delta_score_norm=0.022970018287499745`、`query_to_memory_score_delta_ratio=1.2928086655448818`
  - `results/generated/m3-core4-stage-c-probe-suite-v2/metrics.json` 当前记录 `seed_consistent_by_backbone={Qwen2.5-1.5B-Instruct: true, Qwen3-8B: true}`；同一 target episode 下，两档 backbone 的 `q_only` 都已变成 `adaptation_effective=True`
  - 同一份 Stage C probe v2 当前记录：qwen25 的 `q_only / w_only / w_plus_q` 都停在 `best_adapt_task_score=0.6666666666666666`，但 q-only 的 `best_adapt_query_loss=1.549757957458496` 已与 writer-inclusive 变体几乎打平；qwen3 的三条线也都停在 `best_adapt_task_score=0.6666666666666666`，且 q-only 的 `best_adapt_query_loss=1.6978594064712524` 与 `w_only / w_plus_q` 仅有 `1e-4` 量级差距
  - qwen25 的 fresh q-only gradient audit 当前记录 `queries_grad_norm=0.002236595833389479`、`fuser_grad_norm=0.14431411868747007`、`writer_grad_norm=0.030927921998883723`、`query_to_fuser_grad_ratio=0.015498108249776321`、`query_to_writer_grad_ratio=0.07231639531004393`
  - qwen3 的 fresh q-only gradient audit 当前记录 `queries_grad_norm=0.0004174626431004455`、`fuser_grad_norm=0.047350607186002446`、`writer_grad_norm=0.007302603536826137`、`query_to_fuser_grad_ratio=0.00881641583730005`、`query_to_writer_grad_ratio=0.057166275150394304`
  - `results/generated/m3-core4-stage-c-qonly-budget-probe-suite-v2/metrics.json` 当前显示：q-only 已不再是“预算不够导致完全不动”；两档 backbone 的所有 budget variant 都已经 `adaptation_effective=True`
  - 同一份 q-only budget probe 当前记录：qwen25 最好的 q-only objective 出现在 `lr=5.0, steps=10`，`best_adapt_query_loss=1.63358736038208`；qwen3 则是 canonical `lr=0.2, steps=3` 已最佳，`best_adapt_query_loss=1.5680545568466187`
  - `results/generated/m3-core4-stage-c-probe-suite-v3/metrics.json` 现进一步显示：在某些 target seed 上 official `task_score` 会粗到三条线全打平，例如 qwen25 当前是 `0.0 -> 0.0`；但新的 `best_adapt_task_proxy_score` 已能把 `q_only / w_only / w_plus_q` 分到 `0.4828866223494212 / 0.48302534222602844 / 0.48302459716796875`
  - 同一份 Stage C probe v3 当前记录：qwen3 的三条线 official `task_score` 也仍然打平在 `0.6666666666666666`，但 probe summary 现在会在平手时用 `best_adapt_task_proxy_score` 作为二级比较键，因此 fresh best row 回到了 `q_only`
  - `results/generated/m3-core4-stage-c-qonly-budget-probe-suite-v3/metrics.json` 现进一步说明：两档 backbone 在 official `task_score=0.6666666666666666` 完全打平时，proxy 仍能分出预算差异；当前 qwen25 与 qwen3 的 fresh best budget 都落在 `lr=5.0, steps=10`
  - `results/generated/m3-core4-stage-c-probe-suite-v4/metrics.json` 现进一步显示：将 canonical `Stage C` target 评测改成 `target_eval_repeats=3` 后，official `task_score` 已不再冻结。qwen25 的三条线当前都从 `0.6666666666666666 -> 0.8888888888888888`；qwen3 的三条线当前都从 `0.5555555555555555 -> 0.7777777777777778`
  - `results/generated/m3-core4-stage-c-qonly-budget-probe-suite-v4/metrics.json` 同时说明：多 query-set 聚合确实能让 qwen25 的 q-only official `accuracy` 从 `0.3333333333333333 -> 0.5555555555555555`；但 qwen3 在这一组 target seed 上仍是 `0.4444444444444444 -> 0.2222222222222222`，即 target-side不稳定性还没真正消失
  - `results/generated/m3-core4-stage-c-qonly-seed-sweep/metrics.json`、`results/generated/m3-core4-stage-c-qonly-seed-sweep-v2/metrics.json`、`results/generated/m3-core4-stage-c-qonly-seed-sweep-v4-bottomk/metrics.json` 这些早期正向 official gain 现已判定受到旧版 Stage C eval/support protocol 泄漏影响，不能再作为 canonical few-shot 证据
  - 修正协议后的 `results/generated/m3-core4-stage-c-qonly-seed-sweep-v5-fixed-eval/metrics.json` 当前记录：qwen25 与 qwen3 的 canonical `q_only` 都是 `mean_task_gain=0.0`、`positive_gain_rate=0.0`，只有 `1e-6` 级 `mean_proxy_gain`
  - `results/generated/m3-core4-stage-c-probe-suite-v5/metrics.json` 当前也显示：多 episode 聚合后，两档 backbone 的 fresh best row 都回到了 `q_only`
  - `results/generated/m3-core4-stage-c-qonly-budget-probe-suite-v5/metrics.json` 当前显示：在同一 target episode 聚合口径下，qwen25 仍偏好 `lr=5.0, steps=10`，而 qwen3 的 canonical `lr=0.2, steps=3` 已足够；也就是 qwen3 当前的主要问题已不再是预算不足
  - `results/generated/m3-core4-stage-c-probe-suite-v6/metrics.json` 现进一步显示：把 canonical `Stage C` 明确成 `target_episode_policy=aggregate_support` 后，两档 backbone 的 fresh best row 仍都回到 `q_only`；当前 qwen25 是 `0.6296296296296295 -> 0.6666666666666666`，qwen3 是 `0.6296296296296295 -> 0.8888888888888888`
  - `results/generated/m3-core4-stage-c-qonly-budget-probe-suite-v6/metrics.json` 同时显示：在 `aggregate_support` 口径下，两档 backbone 的最优 budget 都回到了 canonical `lr=0.2, steps=3`，说明当前 main issue 已不再是 Stage C q-only 预算
  - `results/generated/m3-core4-stage-c-qonly-policy-sweep-v1/metrics.json` 现进一步把 policy 本身从 blocker 列表里拿掉：在同一组 5 seeds 上，`aggregate_support` 与 `independent` 给出的 `mean_task_gain` 完全一致，qwen25 都是 `-0.059259259259259255`，qwen3 都是 `0.0962962962962963`；但 `aggregate_support` 的 `mean_support_updates` 从 `9.0` 降到 `3.0`
  - `results/generated/m3-core4-stage-c-qonly-episode-budget-sweep-v1/metrics.json` 现进一步显示：在固定 `aggregate_support` 与同一组 5 seeds 的口径下，`target_episode_repeats=1` 的均值反而是两档 backbone 最优。当前 qwen25 是 `ep1=0.08888888888888889 > ep3=0.02222222222222222 > ep5=-0.013333333333333336`；qwen3 是 `ep1=0.022222222222222233 > ep5=0.013333333333333358 > ep3=0.007407407407407407`
  - `results/generated/m3-core4-stage-c-qonly-support-weight-sweep-v1/metrics.json` 现进一步显示：在固定 `aggregate_support + ep3` 与同一组 5 seeds 的口径下，`target_support_weighting in {uniform, proxy_softmax, proxy_top1}` 的 official `mean_task_gain` 基本完全一致。当前 qwen25 三档都为 `-0.11111111111111112` 左右，qwen3 三档都为 `-0.022222222222222233`
  - 这一步之后，`target split` 已在公平协议下正式排掉，`support_bank_size` 也已证明“有帮助但单独不够”；当前新的 blocker 已收缩成“如何把 `source_plus_support_bank` 已经带来的 proxy 与 qwen3 official 改善进一步放大，并传到 qwen25 official score”
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
说明：benchmark-native `core4` smoke 现在已经打通真实 benchmark 子集上的 `Stage A/B/C` artifact contract、多 source meta-split 与真实 `task_score` 曲线；最新 follow-up 进一步修掉了 Stage C 里 `shot`-耦合 target episode / query pool / support pool 的协议泄漏，并把 `target_split_policy={random, proxy_topk_support, proxy_bottomk_support}` 在公平 fixed-holdout 口径下重新 sweep 了一遍。现在还额外有十六套正式 probe/curve harness：Stage B probe 用于比较 backbone-specific 预算，Stage C probe 用于在同 seed 下比较 `q_only / w_only / w_plus_q`，Stage C q-only budget probe 用于验证不同 `lr/steps` 下的 q-only 行为，Stage C sensitivity audit 用于直接比较 query path 和 memory path 的函数影响量级，Stage C q-only target-split sweep 用于直接比较 target support/query 抽样结构，Stage C q-only support-bank sweep 用于直接比较 support bank 规模，Stage C q-only support-negative-pool sweep 用于直接比较 support negatives 来源，Stage C q-only support-negative-sampler sweep 用于直接比较 negatives 排序策略，Stage C q-only support-selection-policy sweep 用于直接比较 support 选择策略是否会污染 zero-shot holdout 或带来真实 few-shot 收益，Stage C q-only retrieval-negative-count sweep 用于直接比较 support/query retrieval 负样本压力，Stage C q-only retrieval-loss sweep 用于直接比较 inner-loop retrieval loss 形态，Stage C curve suite 用于直接产出更接近正式协议的 `shot_curve / step_curve`，Stage C step saturation audit 用于把 `zero->step0` 与 `step0->final` 两段收益拆开，Stage C margin audit 用于直接检查离 rank flip 还差多少，Stage C negative-seed curve audit 用于只看错误 seeds 上的 shot/step 作用，`episode_trace.json` 则用于直接复盘每个 target episode 的 support/query/eval 组成。同时，Stage C 现在已有并行的 `task_proxy_score` 观测层：当 official `task_score` 因为样本太少或指标太粗而打平时，仍能继续观测 `gold_choice_probability` 这种更平滑的 target-side 变化。当前最新证据已经说明：benchmark-native Stage C 的主 blocker 已不再是 `q_only` 参数化无效，也不再是 `target_split_policy` 选择；`target_support_bank_size=auto` 已证明“有帮助但单独不够”，`target_support_negative_pool=source_plus_support_bank` 已成为第一条真正更强的公平杠杆，`target_support_negative_sampler=hard_by_current_model` 则已经是当前最强的 negative sampler。再往前一步，fresh `retrieval_negative_count={3,7,15}` 对照又说明 negative-count 不是新的主杠杆；而 `retrieval_loss_type={cross_entropy, margin_pairwise, cross_entropy_plus_margin}` 对照进一步显示，`cross_entropy_plus_margin` 现在已经成为当前最强的 fair inner-loop loss，因此 canonical `Stage C` 已切到 `retrieval_loss_type=cross_entropy_plus_margin`、`retrieval_margin_value=0.1`。但更关键的是，fresh canonical `results/generated/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical/metrics.json` 又再次确认：即便在更强的 loss 下，两档 backbone 的 official `mean_task_gain` 仍然都是 `0.0`。也就是说，当前真正剩下的 blocker 已继续收缩成“为什么更强的 proxy / margin gain 仍然过不了 official rank-flip 阈值”，而不是“该选哪一种 negative count / retrieval loss”。

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

当前新增的 `M4` 方法侧 pivot：
- 已新增 `FEVER-first shared generative injection` scaffold：
  - `base_only`
  - `teacher-text upper bound`
  - `writer information audit`
  - `shared latent prefix injection` 的 `real / shuffle / zero`
- `M4.1` 的上游 gate 已全部通过：
  - `Phase 0` 已真实通过：`A_winner=answer_slot_labels`，`T_winner=answer_slot_labels + example_blocks_raw8`
  - `Phase 1 writer audit` 也已通过：`label_probe_passed=true`、`semantic_probe_passed=true`、`phase1_gate_passed=true`
  - 因而当前已不再适合把问题归因成“prompt/support surface 没修好”或“writer 完全没信息”
- `M4.1` 的 `Phase 2` 真实 shared injection 已经进入训练与 step 级诊断：
  - 单看最早一版 stable compare，当前 `A=0.25 / macro_f1=0.2`，`T=0.53125 / macro_f1=0.5294`，`I-real=0.390625 / macro_f1=0.4061`，`I-shuffle=0.546875 / macro_f1=0.5031`，`I-zero=0.25 / macro_f1=0.2`
  - 这说明 prefix 主链路已经不再是零效应，因为 `I-real > I-zero`
  - 但它也说明当前 real support latent 方向仍然会被错误 support 反超，因为 `I-shuffle > I-real`
- `M4.2` 又把 blocker 继续收紧到 `support variant + checkpoint selection`：
  - `raw8` support 下，`I-real` 在 `step32` 仍能到 `0.515625 / macro_f1=0.4572`，并且 `flip_gain_vs_shuffle=8`、`flip_gain_vs_zero=17`
  - `triad6` support 下，`I-real` 在 `step32` 已达到 `0.578125 / macro_f1=0.5238`，并且 `flip_gain_vs_shuffle=16`、`flip_gain_vs_zero=21`
  - 但两条线在 `step64` 都被 `dynamics-audit` 标成 `overshoot_detected=true`
  - 当前更准确的结论已经变成：shared injection 不是“完全没有 real-memory signal”，而是“已经能出正信号，但默认 support/step 口径会把它毁掉”
- 因而当前最值得继续的不是回去修 score-side residual family，也不是直接扩 `Story Cloze` / `Qwen3-8B`，而是：
  - 把 `support variant + checkpoint selection` 做成正式 capability gate
  - 在这条 gate 站稳后，再升级到更强的主链路注入（如 `deep prompt / per-layer prefix`）

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
