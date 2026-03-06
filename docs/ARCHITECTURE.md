# Architecture Map

## Scope

本仓库当前已经完成 `docs/TODO_LIST.md` 的 M0 bootstrap 基座，并补上了 M2 的最小方法骨架；仍然不启动大训练。当前目标是让后续 agent 能按统一入口、统一输出和统一目录契约推进 Stage A/B/C 与论文实验，而不是停留在只会跑 baseline adapter。

## Top-Level Layout

- `train.py` / `eval.py` / `analysis.py`: 根入口，满足 `python -m train|eval|analysis`
- `scripts/run_memgen.sh`: MemGen baseline 的统一 adapter 入口，先支持 dry-run launch 计划与后续真实执行桥接
- `src/memtotal/models/`: backbone wrapper、Writer/Reader/Fuser/Injector、Segmenter
- `src/memtotal/training/`: smoke 训练闭环
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
  - 支持 `context` conditioning、`memory_mask`、可选 query gating
- `MemoryFuser`
  - `arch=linear`：简单投影压缩
  - `arch=resampler`：learned short queries 读取 readouts，形成 `M_short`
- `MemoryInjector`
  - 当前支持 `prefix` 注入
  - `enabled` 开关已进入 config 契约；关闭时会同时关闭生成侧 memory token 注入

这些实现当前仍运行在 deterministic stub backbone 与 toy pipeline 上，用于先验证接口、梯度、注入路径与结果治理；真实 Qwen 权重加载保留在 `BackboneWrapper` 扩展点中。

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
