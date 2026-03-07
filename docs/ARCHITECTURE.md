# Architecture Map

## Scope

本仓库当前已经完成 `docs/TODO_LIST.md` 的 M0 bootstrap 基座，并补上了 M2 的最小方法骨架；仍然不启动大训练。当前目标是让后续 agent 能按统一入口、统一输出和统一目录契约推进 Stage A/B/C 与论文实验，而不是停留在只会跑 baseline adapter。

## Top-Level Layout

- `train.py` / `eval.py` / `analysis.py`: 根入口，满足 `python -m train|eval|analysis`
- `scripts/run_memgen.sh`: MemGen baseline 的统一 adapter 入口，先支持 dry-run launch 计划与后续真实执行桥接
- `scripts/run_m3_core4_stage_b_probe_suite.sh`: benchmark-native `core4` 的 Stage B probe suite，会在数据盘跑 probe variants，并把 `probe_summary.csv/.svg` 写回仓库
- `scripts/run_m3_core4_stage_c_probe_suite.sh`: benchmark-native `core4` 的 Stage C probe suite，会在数据盘并排跑 `q_only / w_only / w_plus_q`，再把 `probe_summary.csv/.svg` 与 q-only gradient audit 关联结果写回仓库
- `scripts/run_m3_core4_stage_c_qonly_budget_probe_suite.sh`: benchmark-native `core4` 的 Stage C `q_only` budget probe，会在同一 target episode 上扫描 `adapt_learning_rate / adapt_steps`
- `scripts/run_m3_core4_stage_c_sensitivity_audit.sh`: benchmark-native `core4` 的 Stage C sensitivity audit，会对比 `query shift` 与 `memory shift` 对 `readouts / summary / candidate scores` 的影响量级
- `scripts/run_m3_core4_stage_c_qonly_seed_sweep.sh`: benchmark-native `core4` 的 Stage C `q_only` seed sweep，会固定 canonical `q_only` 配置并在多个 target seeds 上重复适配，再汇总 `task_gain` 的分布
- `scripts/run_m3_core4_stage_c_curve_suite.sh`: benchmark-native `core4` 的 Stage C curve suite，会用 canonical `q_only` 配置跑 `adapt_shots={0,1,2,3}` 与 `adapt_steps=5`，再自动汇总 `shot_curve.csv/.svg` 与 `step_curve.csv/.svg`
- `scripts/run_m3_core4_stage_c_step_saturation_audit.sh`: benchmark-native `core4` 的 Stage C step saturation audit，会把 canonical curve suite 拆成 `zero->step0` 与 `step0->final` 两段收益
- `scripts/run_m3_core4_stage_c_qonly_policy_sweep.sh`: benchmark-native `core4` 的 Stage C `q_only` policy sweep，会把 `target_episode_policy in {independent, aggregate_support}` 放到同一组 seeds 上直接对照
- `scripts/run_m3_core4_stage_c_qonly_episode_budget_sweep.sh`: benchmark-native `core4` 的 Stage C `q_only` episode-budget sweep，会把 `target_episode_repeats in {1,3,5}` 放到同一组 seeds 上直接对照
- `scripts/run_m3_core4_stage_c_qonly_support_weight_sweep.sh`: benchmark-native `core4` 的 Stage C `q_only` support-weight sweep，会把 `target_support_weighting in {uniform, proxy_softmax, proxy_top1}` 放到同一组 seeds 上直接对照
- `scripts/run_m3_core4_stage_c_qonly_target_split_sweep.sh`: benchmark-native `core4` 的 Stage C `q_only` target-split sweep，会把 `target_split_policy in {random, proxy_topk_support, proxy_bottomk_support}` 放到同一组 seeds 上直接对照
- `scripts/move_hf_cache_to_data_disk.sh`: 将 `~/.cache/huggingface` 迁到数据盘并回接符号链接，避免系统盘被模型/数据缓存打满
- `scripts/cleanup_hf_cache.sh`: 清理 Hugging Face datasets cache 与未完成的模型下载碎片，用于 real-source benchmark / MemGen 的磁盘治理
- `src/memtotal/models/`: backbone wrapper、Writer/Reader/Fuser/Injector、Segmenter
- `src/memtotal/training/`: smoke 训练闭环
- `src/memtotal/training/m3.py`: Stage A/B/C runner，当前同时承载 `toy_meta_smoke` 与 benchmark-native `core4_transfer_smoke`；负责 `writer.ckpt`、`queries_meta_init.pt`、`adapt_curve.csv` 等产物
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
  - 支持 `query_residual_scale`，会把 gated query residual 直接注入 `readouts`；当前 bootstrap 默认值为 `1.0`
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
- benchmark-native smoke 现已新增 `configs/tasks/benchmarks/meta/core4_transfer_smoke.yaml`
  - `dataset_sources={gsm8k, kodcode, gpqa, story_cloze}`
  - `source_domains={math, code, qa}`
  - `target_domain=narrative`
  - 当前 canonical 配置已提升为 `smoke8/3x3`：每个 benchmark source 使用 `eval-real-smoke8.jsonl`，并固定 `support_size=3`、`query_size=3`
  - `sampling_policy=uniform_examples`
- Stage A：
  - 产出 `writer.ckpt`
  - 保存 `meta_data_manifest.json`，记录 `dataset_sha256` 与 domain split
- Stage B：
  - 产出 `queries_meta_init.pt`
  - 支持 `query_learning_mode in {meta_trained, non_meta_multitask, random}`
  - 支持 `query_objective in {label_prototype, continuation_retrieval}`
  - `meta_trained` 当前实现为 first-order ANIL 近似，inner-loop 更新 `reader.queries + fuser`
  - `non_meta_multitask` 使用固定 Writer + source-domain 全局 label bank 的普通多任务训练
  - `random` 不做 Stage B 更新，只落 reader-side 随机初始化快照
  - toy 路径继续按 label 分层采样并使用 domain 内 label prototype
  - benchmark-native `core4` 路径改为统一 `continuation_retrieval` 目标：正样本是当前 continuation，负样本来自同域或全局 source pool
  - benchmark-native `core4` 当前进一步固定为 episode-aware pool 协议：query/val 侧候选池显式排除 support continuations，inner-loop support update 只在 support pool 内做 retrieval
  - `metrics.json` 现已同时记录 `source_eval_query_loss/source_eval_query_accuracy` 与 `source_eval_task_score/source_eval_metric_name`
  - `metrics.json` 现也显式记录 `retrieval_negative_count / meta_episodes / inner_steps / inner_learning_rate / meta_learning_rate`，便于直接对照 Stage B probe 而不必回看 `config.snapshot`
  - 当前 canonical Stage B 已变成 backbone-specific 口径：qwen25 使用 `meta_episodes=16`，qwen3 保持 `meta_episodes=6`
  - 新增 `src/memtotal/analysis/m3_probe.py` 与 `configs/exp/m3_stage_b_probe_summary.yaml`；probe suite 会产出 `best_by_backbone`，并用 `config.snapshot + seed` 校验复用，避免把旧 probe 误当成新结果
- Stage C：
  - 默认按 `adaptation_target=q_only` 对齐 `MAIN_IDEA.md` / `EXPERIMENTS_INFO.md` 的 Stage C 契约，只更新 `reader.queries`
  - 支持 `adaptation_target in {q_only, w_only, w_plus_q}`
  - 支持 `expected_query_learning_mode`，会在 resume 阶段校验 reader init 来源
  - 产出 `queries_adapted.pt`
  - 若 writer 参与适配，则额外产出 `writer_adapted.ckpt`
  - 产出 `adapt_curve.csv` / `adapt_curve.json` / `adapt_cost.json`
  - `adapt_curve.csv` 当前会显式写出 `query_learning_mode / query_objective / adaptation_target / trainable_module / trainable_parameter_count / objective_loss / task_score / task_metric_name`
  - `adapt_curve.csv` 现额外写出 `task_proxy_score / task_proxy_name / task_margin`；当前在 multiple-choice 任务上使用 `gold_choice_probability` 作为更平滑的 target-side proxy
  - `adapt_curve.csv` 现也写出 `target_eval_repeats / evaluated_query_examples`；当前 canonical `core4` Stage C 配置默认 `target_eval_repeats=3`，即每个 `shot/step` 点会在同一 support set 下采样 3 组 target query 子集再求平均
  - `adapt_curve.csv` 现进一步写出 `target_episode_repeats / evaluated_target_episodes`；当前 canonical `core4` Stage C 配置默认 `target_episode_repeats=3`，即每个 `shot/step` 点会在 3 组 target support/query episodes 上聚合后再选 best row
  - `runtime.target_episode_policy` 现支持 `independent / aggregate_support`；当前 canonical `core4` Stage C 配置使用 `aggregate_support`，即在相同 `target_episode_repeats=3` 下把多个 target support episodes 的 support loss 先聚合，再做一次共享 update
  - `runtime.target_split_policy` 现支持 `random / proxy_topk_support / proxy_bottomk_support`；公平 fixed-holdout 重扫后，三档在 official score 上完全打平，因此 canonical 配置现已回收为最朴素的 `random`
  - `runtime.target_support_bank_size` 现支持 `auto / max_shot / all_non_holdout / 正整数`；当前 canonical 配置使用 `auto`，即至少给 support-side retrieval 留出 `retrieval_negative_count + 1` 的候选空间，再受 target 域非 holdout 上限裁剪
  - `runtime.target_support_negative_pool` 现支持 `support_bank / source_plus_support_bank`；当前 canonical 配置已切到 `source_plus_support_bank`，即在 target support bank 之外，再把 source domains 的 continuations 作为 support inner-loop negatives 接入
  - `runtime.target_support_negative_sampler` 现支持 `deterministic_id / hard_by_continuation / hard_by_current_model`；fresh 5-seed 对照显示 `hard_by_current_model` 现在能在两档 backbone 上都给出最高 `mean_proxy_gain`，因此当前 canonical probe 已切到 `hard_by_current_model`
  - `runtime.target_support_weighting` 现支持 `uniform / proxy_softmax / proxy_top1`；当前 canonical 仍保留 `uniform`，因为 fresh support-weight sweep 尚未观察到对 official `task_score` 的稳定改善
  - `metrics.json` 现也显式记录 `retrieval_negative_count / adapt_learning_rate / adapt_steps / adapt_shots`，便于直接复核 Stage C few-shot 口径
  - `metrics.json` 与 `adapt_curve.csv` 现也显式记录 `support_grad_norm / support_update_max_abs / support_update_l2` 诊断；Stage C 会额外给出 `adaptation_effective_threshold / adaptation_effective`，用于区分“分数没涨”与“参数几乎没更新”
  - 新增 `configs/exp/m3_stage_c_gradient_audit_qwen25.yaml` 与 `src/memtotal/analysis/m3_gradient_audit.py`，可在不改训练逻辑的前提下审计同一 target support loss 对 `queries / reader_non_query / fuser / writer` 的 counterfactual 梯度大小
- 新增 `src/memtotal/analysis/m3_stage_c_probe.py` 与 `configs/exp/m3_stage_c_probe_summary.yaml`；probe suite 会把 `Stage C` 的 `q_only / w_only / w_plus_q` 放到同一份 summary 中，并显式检查同一 backbone 下三条曲线是否共用同一个 seed；在 `task_score` 打平时，会用 `task_proxy_score` 作为二级比较键
  - 新增 `src/memtotal/analysis/m3_stage_c_seed_sweep.py` 与 `configs/exp/m3_stage_c_seed_sweep_summary.yaml`；seed sweep 会把 canonical `q_only` 在多个 target seeds 上的 `task_gain / proxy_gain` 分布写成 `seed_sweep.csv/.svg`

当前 M3 smoke 已经把 toy 路径与 benchmark-native `core4` 路径都接进统一 artifact contract、resume 链路与 summary。当前 benchmark-native `core4` 还只是 smoke 级协议验证，不代表正式 few-shot 结果；但 canonical 配置现已从早期 `smoke4/2x2` 升级为 `smoke8/3x3`，并进一步通过 Stage B probe suite 收口到 backbone-specific episode budget：qwen25 当前 canonical `meta_episodes=16`，`runs/verify/m3-core4-qwen25/stage-b/metrics.json` 记录 `mean_adaptation_gain=6.527453660964966e-05`；qwen3 当前 canonical 仍为 `meta_episodes=6`，`runs/verify/m3-core4-qwen3/stage-b/metrics.json` 记录 `mean_adaptation_gain=0.0007965167363484701`。正式 probe suite `results/generated/m3-core4-stage-b-probe-suite-v2/metrics.json` 当前也显示两档 backbone 的最佳变体都回到各自 canonical。Stage C 这条线则已经继续收口到十九步：先通过 `query_residual_scale` 修复 q-only 参数化，让 query path 真正进入有效控制回路；再通过 `task_proxy_score` 把 coarse `task_score` 打平时的 target-side变化显式暴露出来；随后把 canonical target 评测升级为 `target_eval_repeats=3` 的多 query-set 官方聚合；再补一条 `q_only` target-seed sweep，把单 seed 的正向结果放回分布里看；再把 canonical `Stage C` 升级为 `target_episode_repeats=3` 的多 target support/query episode 聚合；再用 policy sweep 证明 `aggregate_support` 与 `independent` 在同 seed 上效果等价但前者更省；再用 episode-budget sweep 证明 `target_episode_repeats` 不是越大越稳；再用 support-weight sweep 证明 `target_support_weighting` 也不是主杠杆；再用 target-split sweep 给出过一轮启发式方向；再新增 curve suite，把 canonical `q_only` 直接汇总成 `shot_curve / step_curve`；再新增 step saturation audit，把 `zero->step0` 和 `step0->final` 两段收益拆开；再修掉 Stage C 里 `shot`-耦合 episode/query/support pool 的评测泄漏，使 query/eval 改为固定 holdout、support inner-loop 改为固定 support bank；再在公平 fixed-holdout 口径下重跑 `target_split_policy={random, proxy_topk_support, proxy_bottomk_support}` 的 5-seed sweep，并确认三档在两档 backbone 上的 official `mean_task_gain` 全部为 `0.0`；再新增 `support_bank_size={max_shot, auto}` 的 5-seed 对照，证明 `auto` 确实在恢复 inner-loop 信号但单独不够；随后新增 `support_negative_pool={support_bank, source_plus_support_bank}` 的 5-seed 对照，证明 source-augmented negatives 已经是真正更强的杠杆；再把 `target_support_negative_sampler` 扩到 `{deterministic_id, hard_by_continuation, hard_by_current_model}` 的 5-seed 对照，确认 `hard_by_current_model` 在两档 backbone 上都给出最高 `mean_proxy_gain`；随后新增 `margin audit`；最新又把它拆成 conditional `negative_only` 摘要。当前 `results/generated/m3-core4-stage-c-margin-audit-v2/metrics.json` 现显示：两档 backbone 各只有 `2` 个负 margin seeds；qwen25 只有 `1/2` 在缩小 gap，平均只关掉 `1.88e-5`，qwen3 也是 `1/2`，而且平均 gap 还略微变差。也就是说，在 canonical `source_plus_support_bank + hard_by_current_model` 下，step-level gain 的确存在，但还没有稳定集中到真正错误的 runs 上。因此，当前 blocker 已进一步收缩成“如何把 gain 更聚焦地送到负 margin runs”；下一步应直接做负 margin seeds 的条件化 shot/step 分析或相应采样策略，而不是回头只调 split、bank size 或 sampler。

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
  - `data/benchmarks/materialized/<benchmark_id>/eval-real-smoke<k>.jsonl`
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
- `src/memtotal/baselines/prompting.py` 现提供最小 prompt baseline family：
  - `family=prompting`
  - `mode in {vanilla, cot}`
  - 统一走 `python -m eval`，但绕过 `MemoryRuntime`
  - 同时支持 `family=meta_prompting`、`mode=planner_critic` 的最小 meta-prompt scaffold
- `src/memtotal/baselines/adapters.py` 现提供最小 adapter baseline family：
  - `family=adapter`
  - `mode in {prompt_tuning, lora, ia3, prefix_tuning}`
  - 统一走 `python -m train` 产出 checkpoint，再由 `python -m eval --checkpoint ...` 评测
  - 当前仅支持 candidate-selection 任务的 smoke 训练
- `src/memtotal/baselines/grid_runner.py` 现提供最小 baseline grid runner：
  - 当前针对 `story_cloze` real-source smoke
  - 会在单个 suite 内循环 `shots / steps`
  - 支持通过 `grid.imports` 把外部 baseline run 导入同一条 `adapt_curve.csv`
  - `grid.imports` 支持 `allow_missing: true`，会把未就绪外部点写入 `skipped_imports`
  - 支持通过 `grid.config_overrides` 在不复制模板配置的前提下覆写 `task/runtime` 字段
  - 支持通过 `grid.reuse_existing_runs` 复用已存在的 `train/eval` 产物，并通过 `config.snapshot + seed` 校验避免误复用
  - 输出 `adapt_curve.csv`、`adapt_cost.json`、`summary.csv`
  - 当前 dual-import protocol suite 已真实导入 `MemGen / story_cloze` 的 qwen25 与 qwen3 两个 `0-shot / 0-step` 外部点；对应 `adapt_cost.json` 为 `imported_eval_count=2`、`skipped_import_count=0`
- 本仓库通过 `src/memtotal/baselines/run_memgen.py` 生成统一 launch plan、run snapshot、真实执行桥接和输出翻译层。
- `run_memgen.py` 现在会在启动前做磁盘空间 preflight，并在空间不足时直接给出固定清理建议，而不是等官方进程中途失败。
- 当前已真实验证 `gsm8k`、`gpqa`、`kodcode`、`rocstories`、`story_cloze`、`triviaqa` smoke eval，并将官方静态 `answer.json` 或动态 `conversations.txt` 翻译为统一 `predictions.jsonl` / `metrics.json`。
- `analysis` 会把 MemGen 的 `compute_reward` 视为主分数字段之一，与自有 eval 的 `accuracy` 一起进入 `summary.csv` / `summary.svg`。
- prompt baseline 当前会额外写：
  - `metrics.json.baseline_family`
  - `metrics.json.baseline_mode`
  - `metrics.json.support_examples`
  - `metrics.json.train_steps`
  - `metrics.json.trainable_parameter_count`
  - `predictions.jsonl[].baseline_prompt`
  - `predictions.jsonl[].baseline_support_ids`
  - `predictions.jsonl[].candidate_scores`
- 当前已真实 smoke 验证 `configs/exp/baseline_{vanilla,cot}_{gsm8k,story_cloze}_qwen25_smoke.yaml`，汇总位于 `results/generated/m5-prompt-baseline-smoke/summary.csv`
- 同一套 qwen3 配置 `configs/exp/baseline_{vanilla,cot}_{gsm8k,story_cloze}_qwen3_smoke.yaml` 也已真实跑通，汇总位于 `results/generated/m5-prompt-baseline-smoke-qwen3/summary.csv`
- 同一套 `Vanilla / CoT` 现已推进到 `gsm8k / story_cloze` 的 real-source smoke：
  - qwen25: `results/generated/m5-prompt-baseline-real-smoke/summary.csv`
  - qwen3: `results/generated/m5-prompt-baseline-real-smoke-qwen3/summary.csv`
- 当前最小 `MetaPrompting` smoke 汇总位于 `results/generated/m5-metaprompting-smoke/summary.csv`
- 同一套 `MetaPrompting` real-source smoke 汇总位于 `results/generated/m5-metaprompting-real-smoke/summary.csv`
- `prompting / meta_prompting` 当前已支持 `support_examples > 0` 的 in-context few-shot demos；当前 `2-shot story_cloze` real-source 汇总位于 `results/generated/m5-prompt-fewshot-real-smoke/summary.csv`
- 当前最小 `RAG` real-source smoke 已接入：
  - qwen25: `runs/verify/baseline_rag_story_cloze_qwen25_real_smoke/metrics.json`
  - qwen3: `runs/verify/baseline_rag_story_cloze_qwen3_real_smoke/metrics.json`
- `rag` baseline 当前会额外写出 `baseline_retriever / mean_support_retrieval_score / baseline_support_scores`
- 当前最小 `memory_bank` real-source smoke 也已接入：
  - qwen25: `runs/verify/baseline_memory_bank_story_cloze_qwen25_real_smoke/metrics.json`
  - qwen3: `runs/verify/baseline_memory_bank_story_cloze_qwen3_real_smoke/metrics.json`
- `memory_bank` baseline 当前会额外写出 `mean_memory_bank_entry_count / mean_memory_bank_selection_score / baseline_memory_bank_entries`
- 当前最小 `LightThinker` real-source smoke 也已接入：
  - qwen25: `runs/verify/baseline_lightthinker_story_cloze_qwen25_real_smoke/metrics.json`
  - qwen3: `runs/verify/baseline_lightthinker_story_cloze_qwen3_real_smoke/metrics.json`
- `lightthinker` baseline 当前会额外写出 `mean_thought_sketch_tokens / lightthinker_compression_prompt / lightthinker_thought_sketch`
- `lightthinker` 现已进入 `story_cloze` 的 minimal/protocol baseline grid；当前会以 prompt-style family 的方式沿 `shot` 维展开，而不占用 `step>0` 训练预算
- 当前 `Prompt Tuning / LoRA / IA3 / Prefix Tuning` 的最小 adapter smoke 已接入：
  - `results/generated/m5-adapter-baseline-smoke/summary.csv`
  - `runs/verify/baseline_prefix_tuning_story_cloze_qwen25_real_smoke/eval/metrics.json`
  - `runs/verify/baseline_prefix_tuning_story_cloze_qwen3_real_smoke/eval/metrics.json`
- 同一套 qwen3 adapter smoke 汇总位于 `results/generated/m5-adapter-baseline-smoke-qwen3/summary.csv`
- 同一套 `Prompt Tuning / LoRA / IA3 / Prefix Tuning` 现已推进到 `story_cloze` real-source smoke；其中历史汇总位于 `results/generated/m5-adapter-baseline-real-smoke/summary.csv`
- `IA3` 与 `Prefix Tuning` 现也已接入同一套 adapter harness：
  - qwen25: `runs/verify/baseline_ia3_story_cloze_qwen25_real_smoke/eval/metrics.json`
  - qwen3: `runs/verify/baseline_ia3_story_cloze_qwen3_real_smoke/eval/metrics.json`
  - qwen25 prefix: `runs/verify/baseline_prefix_tuning_story_cloze_qwen25_real_smoke/eval/metrics.json`
  - qwen3 prefix: `runs/verify/baseline_prefix_tuning_story_cloze_qwen3_real_smoke/eval/metrics.json`
- baseline run 当前会统一写出 `support_examples / train_steps / trainable_parameter_count / budget_signature`
- `analysis_mode=baseline_budget_audit` 已接入统一 `python -m analysis`，当前会检查 `prompting / meta_prompting / adapter / rag / lightthinker / memory_bank` 六个 family 的预算字段与双 backbone 覆盖，汇总位于 `results/generated/m5-baseline-budget-audit/summary.csv`
- 当前最小 baseline grid smoke 汇总位于 `results/generated/m5-story-cloze-baseline-grid-smoke/`，并已真实产出 `adapt_curve.csv`
- 当前 `MemGen` 的 `story_cloze / Qwen2.5-1.5B-Instruct / 0-shot / 0-step` 外部评测点已可通过 `configs/exp/m5_story_cloze_baseline_grid_with_memgen_smoke.yaml` 导入到同一套 grid 汇总，产物位于 `results/generated/m5-story-cloze-baseline-grid-with-memgen-smoke/`
- 当前更接近协议的 grid smoke 汇总位于 `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/`：它使用 `story_cloze` real-source `smoke8` 子集与 `shots={0,1,2,4}`、`steps={0,1,3,5}`，并通过 `grid.config_overrides` 复用同一套 baseline 模板配置；当前 variant 数已扩到 `20`，其中包含 `rag + memory_bank + lightthinker + ia3 + prefix_tuning`
- 同一 protocol-smoke suite 已真实验证缓存复用：在相同输出目录上重跑时，`adapt_cost.json` 会记录 `reused_train_run_count=52`、`reused_eval_run_count=100`；本轮新增 `ia3` 后补跑了 `26` 个 train cell 和 `26` 个 eval cell
- 当前 dual-import protocol suite 位于 `results/generated/m5-story-cloze-baseline-grid-protocol-with-memgen-dual-smoke/`：它已真实导入 qwen25 与 qwen3 的 `MemGen` 点，并把 `memory_bank + ia3` 一起纳入同一套 `18` 个 variant 的 protocol-smoke 汇总
- `scripts/watch_memgen_story_cloze_qwen3_refresh_grid.sh` 现提供一个任务定制 watcher：等待 `runs/verify/memgen-story-cloze-qwen3-smoke-v2/metrics.json` 出现后，自动刷新 dual-import protocol suite
