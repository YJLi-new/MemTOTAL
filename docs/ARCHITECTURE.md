# Architecture Map

## Scope

本仓库当前已经完成 `docs/TODO_LIST.md` 的 M0 bootstrap 基座，并补上了 M2 的最小方法骨架；仍然不启动大训练。当前目标是让后续 agent 能按统一入口、统一输出和统一目录契约推进 Stage A/B/C 与论文实验，而不是停留在只会跑 baseline adapter。

## Top-Level Layout

- `train.py` / `eval.py` / `analysis.py`: 根入口，满足 `python -m train|eval|analysis`
- `scripts/run_memgen.sh`: MemGen baseline 的统一 adapter 入口，先支持 dry-run launch 计划与后续真实执行桥接
- `src/memtotal/models/`: backbone wrapper、Writer/Reader/Fuser/Injector、Segmenter
- `src/memtotal/training/`: smoke 训练闭环
- `src/memtotal/training/m3.py`: Stage A/B/C 的 toy smoke runner，负责 `writer.ckpt`、`queries_meta_init.pt`、`adapt_curve.csv` 等产物
- `src/memtotal/eval/`: 统一评测入口与 `predictions.jsonl` / `metrics.json`
- `src/memtotal/analysis/`: 统一汇总器，扫描 `runs/**/metrics.json` 并生成 `summary.csv` / `summary.svg`
- `src/memtotal/baselines/`: 外部 baseline 适配层，当前已接入 MemGen launch adapter
- `configs/tasks|method|exp/`: 任务、方法、实验配置
- `scripts/`: setup、train/eval/analysis 包装、profiling、smoke、artifact 收集、CI 风格检查
- `runs/`: 原始 run 产物
- `results/`: 自动汇总结果

## Method Boundary

- `MemoryWriter.write(state) -> M_long`
- `MemoryReader.read(M_long, context) -> readouts`
- `MemoryFuser.fuse(readouts) -> M_short`
- `MemoryInjector.inject(M_short, next_inputs) -> injected_inputs`

当前方法层的已验证范围：

- `MemoryWriter`
  - `arch=mlp`：LayerNorm + MLP，直接写出 `[B, L, d]`
  - `arch=transformer`：learned slot embeddings + state conditioning + Transformer encoder
  - 支持 `freeze()` / `unfreeze()` / `save_to()` / `load_from()`
- `MemoryReader`
  - `H` 个 learned queries
  - 基于 cross-attention 读取 `M_long`
  - 支持 `context` conditioning、`memory_mask`、`gating_mode in {off, random, learned}`
- `MemoryFuser`
  - `arch=linear`：简单投影压缩
  - `arch=resampler`：learned short queries 读取 readouts，形成 `M_short`
- `MemoryInjector`
  - 当前支持 `prefix` 注入
  - `enabled` 开关已进入 config 契约；关闭时会同时关闭生成侧 memory token 注入
  - `position in {segment, delimiter, random, none}` 已进入 config 契约

当前 train/eval 产物里，方法层额外会写出：

- `gating_mode`
- `mean_gate`
- `mean_active_queries`
- `mean_segment_gate`
- `mean_segment_active_queries`
- `injection_position`
- `conditioning_schema`
- `predictions.jsonl` 中每个样本的 `gates`
- `predictions.jsonl` 中每个样本的 `segment_stats`
- `predictions.jsonl` 中每个样本的 `conditioning`

这些实现当前仍运行在 deterministic stub backbone 与 toy pipeline 上，用于先验证接口、梯度、注入路径与结果治理；真实 Qwen 权重加载保留在 `BackboneWrapper` 扩展点中。

## M3 Smoke Boundary

- `toy_meta_smoke` 目前承载 Stage A/B/C 的最小验证：`data/toy/meta_samples.jsonl`
- 当前 toy meta 数据采用“每域 2 个 label、每个 label 2 个样本”的结构，便于分层 support/query 采样与最小 few-shot 曲线验证
- Stage A：
  - 产出 `writer.ckpt`
  - 保存 `meta_data_manifest.json`，记录 `dataset_sha256` 与 domain split
- Stage B：
  - 产出 `queries_meta_init.pt`
  - 支持 `query_learning_mode in {meta_trained, non_meta_multitask, random}`
  - `meta_trained` 当前实现为 first-order ANIL 近似，inner-loop 更新 `reader.queries + fuser`
  - `non_meta_multitask` 使用固定 Writer + source-domain 全局 label bank 的普通多任务训练
  - `random` 不做 Stage B 更新，只落 reader-side 随机初始化快照
  - episode 采样按 label 分层，评估使用 domain 内 label prototype，而不是逐样本候选集
- Stage C：
  - 默认按 `adaptation_target=q_only` 对齐 `MAIN_IDEA.md` / `EXPERIMENTS_INFO.md` 的 Stage C 契约，只更新 `reader.queries`
  - 支持 `adaptation_target in {q_only, w_only, w_plus_q}`
  - 支持 `expected_query_learning_mode`，会在 resume 阶段校验 reader init 来源
  - 产出 `queries_adapted.pt`
  - 若 writer 参与适配，则额外产出 `writer_adapted.ckpt`
  - 产出 `adapt_curve.csv` / `adapt_curve.json` / `adapt_cost.json`
  - `adapt_curve.csv` 当前会显式写出 `query_learning_mode / adaptation_target / trainable_module / trainable_parameter_count`

当前 M3 smoke 已经把工件、resume 链路、Stage C 适配对象配置契约、以及最小 meta-train 收益证据搭起来；但它仍是 toy smoke，不应替代后续真实任务上的 few-shot 结果。

## Backbone Policy

当前代码层只允许两档 backbone：

- `Qwen2.5-1.5B-Instruct`
- `Qwen3-8B`

任何新配置若使用其他 backbone，应视为配置错误并直接失败。

## Run Contract

每个 run 最少写入：

- `config.snapshot.yaml`
- `run_info.json`
- `metrics.json`
- `profiling.json`
- `predictions.jsonl`（评测）
- `checkpoint.pt`（训练）

若当前目录没有 git 元数据，`run_info.json` 会显式写入 `git_hash: "nogit"`，而不是静默缺失。

## Baseline Boundary

- `MemGen-master/` 保持为官方参考实现目录。
- 本仓库通过 `src/memtotal/baselines/run_memgen.py` 生成统一 launch plan、run snapshot、真实执行桥接和输出翻译层。
- 当前已真实验证 `gsm8k`、`gpqa`、`kodcode`、`rocstories`、`story_cloze`、`triviaqa` smoke eval，并将官方静态 `answer.json` 或动态 `conversations.txt` 翻译为统一 `predictions.jsonl` / `metrics.json`。
- `analysis` 会把 MemGen 的 `compute_reward` 视为主分数字段之一，与自有 eval 的 `accuracy` 一起进入 `summary.csv` / `summary.svg`。
