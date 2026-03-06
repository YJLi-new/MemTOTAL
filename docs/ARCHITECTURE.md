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

当前 M3 smoke 已经把工件、resume 链路、Stage C 适配对象配置契约、Reader 学习方式消融、以及最小 meta-train 收益证据搭起来；但它仍是 toy smoke，不应替代后续真实任务上的 few-shot 结果。

## M4 Benchmark Scaffold

- 新增 `src/memtotal/tasks/registry.py`
  - 统一维护 benchmark `domain / evaluator_type / metric_name / prompt_template`
  - 当前已登记：`gsm8k`、`math`、`gpqa`、`triviaqa`、`kodcode`、`story_cloze`、`rocstories`、`fever`、`alfworld`
- 新增 `src/memtotal/tasks/evaluator.py`
  - 当前统一支持：
    - `exact_match`
    - `multiple_choice`
    - `dataset_label_classification`（向后兼容 toy smoke）
- `python -m eval` 现通过 `TaskEvaluator` 统一输出：
  - `benchmark_id`
  - `task_domain`
  - `smoke_subset`
  - `evaluator_type`
  - `normalized_prediction`
  - `normalized_reference`
- 当前本地 contract smoke 数据位于 `data/benchmarks/smoke/*.jsonl`
  - 这些是仓库内 smoke subset，用于验证 prompt/evaluator/run contract
  - 不是正式 benchmark 下载替身，也不代表论文主结果
- 一键回归入口：
  - `scripts/run_benchmark_smoke_suite.sh`
  - 当前会顺序跑 `gsm8k / gpqa / kodcode / story_cloze / fever / alfworld` 六个 smoke eval，并汇总到 `summary.csv/.svg`
- 新增 `src/memtotal/tasks/sources.py` 与 `src/memtotal/tasks/setup_data.py`
  - 统一维护真实数据源 registry、访问方式、materialize 路径与许可备注
  - 当前已能真实 materialize：
    - `gsm8k`
    - `math`
    - `gpqa`
    - `triviaqa`
    - `story_cloze`
    - `narrativeqa`
    - `kodcode`
    - `rocstories`
    - `fever`
    - `alfworld`
    - `memoryagentbench`
- 新增真实数据入口脚本：
  - `scripts/setup_benchmark_data.sh`
  - `scripts/run_real_benchmark_smoke_suite.sh`
- 真实来源 smoke 当前会落到：
  - `data/benchmarks/materialized/<benchmark_id>/eval-real-smoke4.jsonl`
  - `data/benchmarks/manifests/<benchmark_id>.json`
  - `data/benchmarks/source_summary.json`
  - `alfworld` 目前通过 `src/memtotal/tasks/alfworld_env.py` 走官方 TextWorld 资产与一次 expert transition materialize，不再停留在手写 contract 样例
  - `narrativeqa` 目前通过 `deepmind/narrativeqa` 的官方 HF 数据源走 `full_text_segmented` smoke 视图：materialize 时保留完整 `story_chunk_pool`，并用结构化起点探测先跳过明显 intro / editorial front matter；load/eval 时再按 `task.narrativeqa_runtime` 选择真正注入的 story chunks。当前 selector 已显式支持 `anchor_only / question_aware / oracle_like_proxy` 三档，默认仍是 `question_aware` + `6` 段 budget；统一评测当前仍使用 `qa_f1` 代理口径
  - `memoryagentbench` 目前通过 `src/memtotal/tasks/memoryagentbench.py` 走官方 Hugging Face 数据源，并在 manifest 中显式记录 `AR / TTL / LRU / CR` 四类能力的 smoke source 与当前的 context truncation 预算
  - `configs/exp/benchmark_narrativeqa_qwen3_real_smoke.yaml` 已提供 NarrativeQA 的 `Qwen3-8B` 同构 smoke 配置，并已真实验证跑通
  - 当前最新 real-source smoke 汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T163014Z/summary.csv`
  - `NarrativeQA` selector 消融的统一汇总位于 `results/generated/m4-narrativeqa-selector-ablations/summary.csv`，可直接对比 `anchor_only / question_aware / oracle_like_proxy`
  - `Qwen3-8B` 版本的 NarrativeQA selector 消融也已真实跑通，汇总位于 `results/generated/m4-narrativeqa-selector-ablations-qwen3/summary.csv`

当前 `M4` 已不只是本地 contract smoke。现在已有 11 个 benchmark 的真实来源 smoke 子集进入统一 eval 与统一汇总，但这仍然只是“真实数据入口已打通”，不是正式 benchmark 主结果。其中特别需要区分：`MemoryAgentBench` 当前是“真实 source + 截断 context 的 smoke scaffold”，`NarrativeQA` 当前是“真实 full story source + runtime-selected selector-ablation excerpt + qa_f1 代理评测”的 smoke scaffold，二者都不是正式长上下文主结果。

## M3 Failure Checks

- `analysis_mode=m3_failure_checks` 会加载 `writer.ckpt + queries_meta_init.pt`
- 当前显式运行三个退化 ablation：
  - `zero_memory`
  - `writer_noise`
  - `collapsed_fuser`
- 输出：
  - `failure_checks.json`
  - `failure_ablation_summary.csv`
  - `failure_ablation_summary.svg`
- 当前检查规则：
  - `reader_uses_memory`: `zero_memory` loss 必须显著劣于 base
  - `writer_beats_noise`: `writer_noise` loss 必须显著劣于 base
  - `fuser_avoids_collapse`: base slot diversity 与 `collapsed_fuser` loss gap 不能同时退化
- `writer_noise` 当前支持通过 `runtime.failure_checks.writer_noise_trials` 配置多次噪声抽样取期望，避免 tiny smoke 上单次抽样的高方差把检查打成偶然平局。

当前最新 canonical follow-up run `results/generated/m3-fuser-fix-failure-checks-v2/` 已通过三项检查：
- `reader_uses_memory`: `0.6569329723715782 -> 0.6758842468261719`
- `writer_beats_noise`: `0.6569329723715782 -> 0.6875795591622591`，`writer_noise_trials=8`
- `fuser_avoids_collapse`: `base_short_slot_diversity=0.004472408443689346`，`collapsed_fuser_query_loss=0.6617699563503265`

当前这组结果依赖两条工程修正：一是 `M_short` 的下游摘要不再用简单均值，而是通过 position-sensitive `summary_proj` 保留 slot identity；二是 `writer_noise` 检查改为多次噪声抽样平均，减少 tiny smoke 方差。

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
