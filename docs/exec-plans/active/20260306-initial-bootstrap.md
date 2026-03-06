## Purpose

建立本仓库的 M0 初始 bootstrap：先把目录骨架、统一入口、配置层、最小 smoke harness、结果治理与两档 backbone 占位配置搭起来，使后续 agent 可以按 `docs/TODO_LIST.md` 继续推进，而不是手工拼装环境。

## Context

- 执行顺序与 DoD 以 `docs/TODO_LIST.md` 为准。
- 方法定义与训练阶段以 `docs/MAIN_IDEA.md` 为准。
- 评测与汇总规范以 `docs/EXPERIMENTS_INFO.md` 为准。
- 当前仓库几乎只有治理文档和 `MemGen-master/` 参考实现，缺少项目级骨架与统一入口。
- 固定支持的 backbone 仅有：
  - `Qwen2.5-1.5B-Instruct`
  - `Qwen3-8B`

## Plan of Work

1. 建立 M0/P0 需要的目录骨架和文档骨架。
2. 实现统一的 Python CLI 入口，覆盖 train / eval / analysis，并统一接受 `--config --seed --output_dir`。
3. 建立最小方法 API 与可运行 toy/smoke 路径，保证写->读->融合->注入链路可跑通。
4. 建立 run/output 契约、config snapshot、环境信息与结果汇总脚手架。
5. 提供 `setup_env.sh`、`setup_data.sh`、`dev_boot_smoke.sh`、`collect_artifacts.sh` 等 agent-friendly 入口。
6. 用轻量 smoke 验证当前 bootstrap 成果，不启动真实大训练。

## Concrete Steps

1. 创建 `src/`、`configs/`、`scripts/`、`results/`、`runs/`、`tests/`、`docs/` 相关骨架。
2. 编写 `pyproject.toml` 与包入口，使 `python -m train` / `python -m eval` / `python -m analysis` 可运行。
3. 定义基础配置格式与两档 backbone 的 method/exp/task 示例配置。
4. 实现：
   - `BackboneWrapper`
   - `MemoryWriter`
   - `MemoryReader`
   - `MemoryFuser`
   - `MemoryInjector`
   - `Segmenter`
5. 实现结构化输出：
   - `metrics.json`
   - `predictions.jsonl`
   - `config.snapshot.yaml`
   - `run_info.json`
6. 实现最小报告器，从 `runs/**/metrics.json` 生成 CSV。
7. 补充 smoke test 与结构测试，验证 run 目录契约与文档存在性。

## Validation & Acceptance

- `python -m train --config ... --seed 123 --output_dir ... --dry-run` 可成功完成一次 smoke train。
- `python -m eval --config ... --seed 123 --output_dir ... --dry-run` 可生成 `predictions.jsonl` 与 `metrics.json`。
- `python -m analysis --config ... --seed 123 --output_dir ... --input_root ...` 可生成汇总 CSV。
- `scripts/dev_boot_smoke.sh` 能从当前 worktree 触发 train + eval + analysis 最小闭环。
- 测试覆盖：
  - 文档链接存在
  - run 目录包含 config/seed/git hash(or fallback)
  - 最小方法 API smoke

## Progress

- 2026-03-06 17:22 UTC: 已读取 `docs/AGENTS.md`、`docs/TODO_LIST.md`，并按 M0 指引补读 `docs/MAIN_IDEA.md` 的方法总览/实现契约/Stage A-B-C，以及 `docs/EXPERIMENTS_INFO.md` 的统一评测与结果汇总规范。
- 2026-03-06 17:22 UTC: 确认当前仓库缺少项目级代码骨架；`MemGen-master/` 仅作为后续 M1 基线参考，不作为本次 bootstrap 的主执行层。
- 2026-03-06 17:47 UTC: 已建立 `src/`、`configs/`、`scripts/`、`tests/`、`data/toy/`、`results/`、`runs/` 骨架，并补齐 `docs/ARCHITECTURE.md`、`docs/golden-principles.md`、`docs/tech-debt-tracker.md`。
- 2026-03-06 17:47 UTC: 已实现 `python -m train` / `python -m eval` / `python -m analysis` 入口、最小四模块 API、stub `BackboneWrapper`、统一 run snapshot、汇总器和 smoke tests。
- 2026-03-06 17:47 UTC: `./scripts/dev_boot_smoke.sh` 已跑通，产物位于 `runs/smoke/20260306T094253Z/` 与 `results/generated/20260306T094253Z/`。
- 2026-03-06 17:55 UTC: 已补齐 profiling 管线；train/eval/analysis 会统一写出 `profiling.json` 与 `profiling.csv`，并将 `wall_time_sec`、`token_count`、`peak_device_memory_bytes` 合并进 `metrics.json`。
- 2026-03-06 17:55 UTC: 已补齐 `scripts/run_train.sh`、`scripts/run_eval.sh`、`scripts/run_analysis.sh`、`scripts/profile_run.sh`；`dev_boot_smoke.sh` 现通过这些 wrapper 跑通。
- 2026-03-06 17:55 UTC: 已补齐 repo lint：文档交叉引用检查、结果治理检查、artifact 收集脚本测试；analysis 额外生成 `summary.svg` sanity plot。
- 2026-03-06 17:55 UTC: 最新 smoke run 位于 `runs/smoke/20260306T095453Z/`，对应汇总输出位于 `results/generated/20260306T095453Z/`；`python -m unittest discover -s tests -v` 当前为 9 项通过。
- 2026-03-06 18:04 UTC: 已建立 MemGen baseline adapter：`scripts/run_memgen.sh` + `src/memtotal/baselines/run_memgen.py`，支持统一 `--config --seed --output_dir --dry-run` 契约，并为 `Qwen2.5-1.5B-Instruct` / `Qwen3-8B` 生成可执行 launch 计划。
- 2026-03-06 18:04 UTC: `./scripts/run_memgen.sh --config configs/exp/memgen_gsm8k_qwen25_eval.yaml --seed 11 --output_dir runs/verify/memgen-dry-run --dry-run` 与对应 Qwen3 dry-run 均已通过；当前单测总数为 10。
- 2026-03-06 18:10 UTC: 已安装 MemGen 运行所需关键依赖（`omegaconf`、`accelerate`、`datasets`、`trl`、`safetensors`、`peft`、`transformers` 等），并验证 `python MemGen-master/main.py --help` 可正常启动。
- 2026-03-06 18:11 UTC: 已为 MemGen GSM8K builder 补充 `num_workers` 与 `max_{train,valid,test}_samples` 子集开关，并为 smoke 配置锁定本地 Qwen2.5 权重、小子集和 `sdpa` attention backend。
- 2026-03-06 18:12 UTC: 已修正 MemGen `build_working_dir()` 对本地模型路径的命名问题；真实 run 现落在 `MemGen-master/results/evaluate/gsm8k/Qwen2.5-1.5B-Instruct/...`，不再错误落到 `root/`。
- 2026-03-06 18:12 UTC: `./scripts/run_memgen.sh --config configs/exp/memgen_gsm8k_qwen25_smoke_eval.yaml --seed 22 --output_dir runs/verify/memgen-smoke-translated` 已真实成功，adapter 已翻译出统一 `predictions.jsonl` 与 `metrics.json`，当前 `compute_reward=0.25`（4 个 test 样本 smoke 子集）。
- 2026-03-06 18:13 UTC: adapter 指标桥已补充 `num_predictions` 与 `wall_time_sec`；`runs/verify/memgen-smoke-translated-v2/metrics.json` 当前记录 `compute_reward=0.25`、`num_predictions=4`、`wall_time_sec=31.361597`。
- 2026-03-06 18:17 UTC: 已验证 `python -m analysis ... --input_root runs/verify` 能把 MemGen adapter run 汇总进 `results/generated/verify-summary/summary.csv`；`gsm8k` 与 `rocstories` translated run 已进入统一结果层。
- 2026-03-06 18:17 UTC: 已补充 `docs/baselines/memgen.md`，把 MemGen 任务矩阵、固定模板/种子、统一输出桥与常见坑显式写回仓库。
- 2026-03-06 18:17 UTC: 统一 analysis 已将 `compute_reward` 纳入主分数字段选择逻辑，避免 MemGen run 在 `summary.svg` 中被错误显示为 0。
- 2026-03-06 18:17 UTC: 已新增 `configs/exp/memgen_story_cloze_qwen25_smoke_eval.yaml`，用于补齐 Narrative/CDMI 第二个轻量基线模板。
- 2026-03-06 18:21 UTC: `./scripts/run_memgen.sh --config configs/exp/memgen_story_cloze_qwen25_smoke_eval.yaml --seed 41 --output_dir runs/verify/memgen-story-cloze-smoke` 已真实成功，当前 `compute_reward=0.75`、`num_predictions=4`、`wall_time_sec=28.020674`。
- 2026-03-06 18:21 UTC: 重新汇总后，`results/generated/verify-summary-v2/summary.csv` 已包含 `story_cloze` translated run；sanity plot 现直接展示 MemGen `compute_reward`，并对训练损失使用正值映射，避免负宽度图元。
- 2026-03-06 18:27 UTC: 已新增 `configs/exp/memgen_gpqa_qwen25_smoke_eval.yaml` 并补齐 `MemGen-master/data/gpqa/builder.py` 的子集裁剪/并行控制；首次真实 smoke 失败原因已定位为 `Idavidrein/gpqa` gated dataset 认证缺失，而非 harness 或模型启动错误。
- 2026-03-06 18:34 UTC: 已新增 `configs/exp/memgen_triviaqa_qwen25_smoke_eval.yaml` 并补齐 `MemGen-master/data/triviaqa/builder.py` 的子集裁剪/并行控制；`./scripts/run_memgen.sh --config configs/exp/memgen_triviaqa_qwen25_smoke_eval.yaml --seed 62 --output_dir runs/verify/memgen-triviaqa-smoke-v2` 已真实成功。
- 2026-03-06 18:34 UTC: 已为 MemGen adapter 补齐动态环境翻译逻辑；`triviaqa` 这类只写 `evaluate/conversations.txt` 的任务，现在也会落统一 `predictions.jsonl` 与 `metrics.json`，当前 `compute_reward=0.0`、`num_predictions=4`、`wall_time_sec=80.757644`。
- 2026-03-06 18:36 UTC: 已为 `gpqa` 增加 Hugging Face gated dataset preflight；`./scripts/run_memgen.sh --config configs/exp/memgen_gpqa_qwen25_smoke_eval.yaml --seed 52 --output_dir runs/verify/memgen-gpqa-smoke-preflight-v2` 现会在 adapter 层直接返回 `returncode=2`，并把 `huggingface-cli login` / `HF_TOKEN` 提示写入 `metrics.json` 与 `memgen_process.json`。
- 2026-03-06 18:53 UTC: 在完成 Hugging Face 登录后，`./scripts/run_memgen.sh --config configs/exp/memgen_gpqa_qwen25_smoke_eval.yaml --seed 53 --output_dir runs/verify/memgen-gpqa-smoke-v2` 已真实成功，当前 `compute_reward=0.0`、`num_predictions=4`、`wall_time_sec=35.295111`；说明 `gpqa` 的剩余阻塞已被环境认证解除，adapter / builder / translation 契约成立。
- 2026-03-06 18:59 UTC: 已新增 `configs/exp/memgen_kodcode_qwen25_smoke_eval.yaml` 并补齐 `MemGen-master/data/kodcode/builder.py` 的子集裁剪/并行控制；`./scripts/run_memgen.sh --config configs/exp/memgen_kodcode_qwen25_smoke_eval.yaml --seed 71 --output_dir runs/verify/memgen-kodcode-smoke` 已真实成功，当前 `compute_reward=0.25`、`num_predictions=4`、`wall_time_sec=41.675389`。
- 2026-03-06 18:59 UTC: 截至当前，MemGen 已在 `gsm8k`、`gpqa`、`triviaqa`、`kodcode`、`rocstories`、`story_cloze` 六个任务上完成统一 smoke 接入；M1 现阶段的主要剩余工作已从“任务接入”转为“trigger / insertion 配置对齐与验证”。
- 2026-03-06 19:02 UTC: 已新增 `configs/exp/memgen_gsm8k_qwen25_smoke_eval_trigger_on.yaml`；`./scripts/run_memgen.sh --config configs/exp/memgen_gsm8k_qwen25_smoke_eval_trigger_on.yaml --seed 81 --output_dir runs/verify/memgen-gsm8k-trigger-on-smoke` 已真实成功，当前 `compute_reward=0.5`、`num_predictions=4`、`wall_time_sec=31.25347`。
- 2026-03-06 19:02 UTC: 该 trigger-on 结果仅用于验证 `trigger.active=True` 路径和统一输出桥，不视为正式可比的 MemGen trigger baseline；后者仍需对齐触发器权重来源与插入配置口径。
- 2026-03-06 19:06 UTC: 已将 `trigger_active / insertion_profile / requires_trained_checkpoint / load_model_path` 纳入 MemGen adapter 的显式配置契约，并新增 `configs/exp/memgen_gsm8k_qwen25_eval_trigger_trained_template.yaml` 作为正式 trigger baseline 模板。
- 2026-03-06 19:06 UTC: `./scripts/run_memgen.sh --config configs/exp/memgen_gsm8k_qwen25_eval_trigger_trained_template.yaml --seed 91 --output_dir runs/verify/memgen-trigger-trained-template-preflight` 已验证 preflight 直报 checkpoint 缺失；`./scripts/run_memgen.sh --config configs/exp/memgen_kodcode_qwen25_smoke_eval.yaml --seed 72 --output_dir runs/verify/memgen-kodcode-smoke-v2` 证明 `TOKENIZERS_PARALLELISM=false` 已消除先前稳定出现的 tokenizers fork 警告。
- 2026-03-06 11:17 UTC: 已将方法层从纯 M0 stub 升级为 M2 skeleton：`MemoryWriter` 现支持 `mlp` / `transformer` 两档实现，`MemoryReader` 补齐 learned queries + cross-attention + `memory_mask`，`MemoryFuser` 补齐 `linear` / `resampler` 两档，`MemoryInjector` 新增 `enabled` 配置开关并同步控制生成侧 memory 注入。
- 2026-03-06 11:17 UTC: 新增 `configs/method/memory_bootstrap_transformer.yaml` 与 `configs/exp/smoke_qwen25_transformer_writer.yaml`；已真实跑通 `train -> eval -> analysis` 轻量闭环，产物位于 `runs/verify/m2-transformer-writer-v2/` 与 `results/generated/m2-transformer-writer-summary-v2/`。
- 2026-03-06 11:17 UTC: 最新验证包括 `python -m unittest discover -s tests -v`（23 项通过）、transformer-writer toy train（`final_loss=0.04995205998420715`）、eval（`accuracy=0.5`）、analysis（`rows_collected=2`）。
- 2026-03-06 11:21 UTC: 已将 Query-Gating 从布尔值升级为显式 `gating_mode` 契约，支持 `off / random / learned`；并在 `train/eval` 的 `metrics.json` 记录 `gating_mode / mean_gate / mean_active_queries`，在 `predictions.jsonl` 记录每样本 `gates`。
- 2026-03-06 11:21 UTC: 新增 `configs/exp/smoke_qwen25_transformer_writer_learned_gating.yaml`；已真实跑通 learned-gating toy 闭环，产物位于 `runs/verify/m2-learned-gating/` 与 `results/generated/m2-learned-gating-summary/`，当前 eval `accuracy=0.25`、`mean_gate=0.5174825489521027`、`mean_active_queries=3.0`。
- 2026-03-06 11:33 UTC: 已将 runtime 升级为真正按 segment 执行 `write -> read -> fuse -> inject` 的 toy 形态；`predictions.jsonl` 现会写出每个 segment 的 `segment_stats`，包括 `mean_gate / active_queries / gates / injection_anchor`。
- 2026-03-06 11:33 UTC: 已补齐 `method.injector.position` 配置契约，支持 `segment / delimiter / random / none` 四档，并新增对应 smoke 配置；已真实验证 `delimiter`、`random`、`none` 以及默认 `segment` 的 train/eval 路径。
- 2026-03-06 11:33 UTC: 已固定 `method.reader.conditioning` 契约：统一保存 `domain_name` 与可选 `task_name`，若缺少 `domain_key` 会在 forward 早期报错；`train/eval metrics.json` 会额外写出 `conditioning_schema`。
- 2026-03-06 11:33 UTC: `position=none` 暴露出 toy train 在“无注入即无梯度”配置下的 harness 缺口，现已修复为显式记录 `loss_has_grad=false` 并安全结束 run，而不是在 `backward()` 阶段崩溃。
- 2026-03-06 11:48 UTC: 已新增 `toy_meta_smoke` 数据与 M3 的 Stage A/B/C runner：`run_stage_a()` 会产出 `writer.ckpt` 与 `meta_data_manifest.json`，`run_stage_b()` 会产出 `queries_meta_init.pt`，`run_stage_c()` 会产出 `queries_adapted.pt` 与 `adapt_curve.csv`。
- 2026-03-06 11:48 UTC: 新增 `configs/exp/m3_stage_{a,b,c}_qwen25_smoke.yaml`，并用 `python -m train ... --resume ...` 顺序真实跑通仓库内 smoke，产物位于 `runs/verify/m3-stage-a/`、`runs/verify/m3-stage-b/`、`runs/verify/m3-stage-c/`。
- 2026-03-06 11:48 UTC: 统一 analysis 已补齐 M3 指标主分数字段识别；`best_adapt_query_accuracy`、`zero_shot_query_accuracy`、`mean_adaptation_gain` 不再在 `summary.csv` 中退化成 `none/0.0`。
- 2026-03-06 11:48 UTC: 当前 M3 仍有一个明确阻塞未过 DoD：toy smoke 下 `Stage B mean_adaptation_gain` 依然为负，因此尚不能宣称 source-domain meta-train 收益已成立；该问题已登记进 `docs/tech-debt-tracker.md`。
- 2026-03-06 11:55 UTC: 已将 `toy_meta_smoke` 改成“每域 2 label × 每 label 2 样本”的结构，并把 episode sampler 改为按 label 分层采样；Stage B/C 的分类目标也已切换到 domain 内 label prototype，而不是逐样本候选。
- 2026-03-06 11:55 UTC: 重新跑通 M3 顺序 smoke 后，`runs/verify/m3-stage-b/metrics.json` 当前记录 `mean_adaptation_gain=0.02427813410758972`，说明 source-domain 上已经能观察到正向适配收益；原 blocker 已解除。
- 2026-03-06 11:55 UTC: 当前 `runs/verify/m3-stage-c/adapt_curve.csv` 记录 target domain `narrative` 上 `zero_shot_query_loss=0.7023470401763916 -> best_adapt_query_loss=0.6856379508972168`，但 accuracy 仍为 `0.5 -> 0.5`；这说明 M3 smoke contract 已成立，但更强的 target few-shot 提升仍留给后续正式实验。
- 2026-03-06 12:22 UTC: 已将 Stage C 的 code drift 显式收口到文档契约：`MAIN_IDEA.md` 与 `EXPERIMENTS_INFO.md` 都要求 Stage C 默认“只更新 queries”，因此仓库现引入 `runtime.adaptation_target in {q_only, w_only, w_plus_q}`，并把默认 `m3_stage_c_qwen25_smoke.yaml` 对齐为 `q_only`。
- 2026-03-06 12:22 UTC: 已真实跑通三组适配对象消融，当前 canonical 结果位于 `runs/verify/m3-adaptation-targets-canonical/`：`Q-only` 保持 `0.7023470401763916 -> 0.7023470401763916`，`W-only` 与 `W+Q` 均为 `0.7023470401763916 -> 0.694838285446167`；三组均输出 `adapt_curve.csv`、`adapt_cost.json`，且预算对齐为 `shots={0,2}`、`steps=3`、`lr=0.2`。
- 2026-03-06 12:17 UTC: 已为 Stage B 引入 `query_learning_mode in {meta_trained, non_meta_multitask, random}`，并新增 `m3_stage_b_qwen25_smoke_{non_meta,random}.yaml`；Stage C 同时新增 `expected_query_learning_mode` 校验和 `m3_stage_c_qwen25_smoke_{non_meta,random}.yaml`，避免 resume 错误混用 reader init。
- 2026-03-06 12:17 UTC: 已真实跑通 Reader 学习方式消融，canonical 结果位于 `runs/verify/m3-reader-learning-modes-canonical/`：`meta-trained` 在 target `narrative` 上得到 `zero_shot_query_loss=0.7023470401763916`，`non-meta` 为 `0.7048434019088745`，`random` 为 `0.7098537683486938`。三者当前在 `q_only` few-shot accuracy 上仍都保持 `0.5`，说明 harness 已能比较初始化质量，但更强的 few-shot 提升仍需后续任务。
- 2026-03-06 12:37 UTC: 已为 `analysis` 增加 `m3_failure_checks` 模式，并真实跑通 `zero_memory / writer_noise / collapsed_fuser` 三类 smoke ablation；结果位于 `results/generated/m3-failure-checks-canonical/`。
- 2026-03-06 12:37 UTC: 当前 canonical meta run 的 failure checks 中，`reader_uses_memory` 与 `writer_beats_noise` 已通过，但 `fuser_avoids_collapse` 未通过，当前观测是 `base_short_slot_diversity≈0` 且 `collapsed_fuser` 与 base loss 持平。这说明退化检查 harness 已经成立，而且它确实发现了当前 toy 路线里的一个结构问题。
- 2026-03-06 13:13 UTC: 已对 `Fuser collapse` blocker 做 follow-up 修复：`MemoryFuser(resampler)` 现在保留 short-query residual；M3 的分类路径与 failure checks 不再对 `M_short` 做简单均值，而改用 position-sensitive `summary_proj` 摘要。fresh canonical run 位于 `runs/verify/m3-fuser-fix-canonical/stage-b-meta/`，当前 `mean_adaptation_gain=0.08508576452732086`。
- 2026-03-06 13:13 UTC: 已把 `writer_noise` failure check 升级为可配置的多次噪声抽样均值，避免 tiny smoke 上单次抽样方差误判；当前 canonical 配置使用 `writer_noise_trials=8`，`results/generated/m3-fuser-fix-failure-checks-v2/metrics.json` 现记录 `checks_pass_rate=1.0`，三项检查全部通过。
- 2026-03-06 13:24 UTC: 已启动 `M4` foundation：新增 `src/memtotal/tasks/{registry,evaluator}.py`，统一登记 benchmark 的 `domain / evaluator_type / metric_name / prompt_template`，并把 `eval` 输出扩展为 `benchmark_id / task_domain / smoke_subset / evaluator_type`。
- 2026-03-06 13:24 UTC: 已建立本地 benchmark smoke subset 与配置：`gsm8k`、`math`、`gpqa`、`triviaqa`、`kodcode`、`story_cloze`、`rocstories`、`fever`、`alfworld`；`scripts/run_benchmark_smoke_suite.sh` 已真实跑通 6 个代表任务，汇总位于 `results/generated/m4-benchmark-smoke/20260306T132413Z/summary.csv`。
- 2026-03-06 13:37 UTC: 已新增 benchmark source registry、materialize CLI 与数据来源文档：`src/memtotal/tasks/sources.py`、`src/memtotal/tasks/setup_data.py`、`scripts/setup_benchmark_data.sh`、`docs/benchmark-data.md`。当前 `data/benchmarks/source_summary.json` 与 `data/benchmarks/manifests/*.json` 会显式记录 access / source_url / split / license_note / local path。
- 2026-03-06 13:37 UTC: 已真实 materialize `gsm8k`、`gpqa`、`triviaqa`、`story_cloze`、`kodcode`、`rocstories` 的真实来源 smoke 子集，落在 `data/benchmarks/materialized/*/eval-real-smoke4.jsonl`；`fever` 与 `alfworld` 当前只写 `manual_pending` manifest。
- 2026-03-06 13:37 UTC: `scripts/run_real_benchmark_smoke_suite.sh` 已真实跑通 6 个 real-source smoke eval，汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T133708Z/summary.csv`。当前分数仍只是 stub-backbone contract 结果，不代表正式 benchmark 表现。
- 2026-03-06 13:52 UTC: 已将 `math` 与 `fever` 升级为真实来源 smoke，而不再停留在本地 contract 样例。`math` 当前通过 `EleutherAI/hendrycks_math` 的四个 config 聚合 materialize；`fever` 当前使用公开的 `Dzeniks/fever_3way`，并在 canonicalization 时映射成 `SUPPORTS / REFUTES / NOT_ENOUGH_INFO`。
- 2026-03-06 13:52 UTC: `./scripts/setup_benchmark_data.sh` 与 `./scripts/run_real_benchmark_smoke_suite.sh` 已重新真实跑通，最新 real-source smoke 汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T135108Z/summary.csv`。当前 8 个任务已进入统一 real-source eval：`gsm8k / math / gpqa / triviaqa / kodcode / story_cloze / rocstories / fever`。
- 2026-03-06 14:18 UTC: 已将 `alfworld` 从 `manual_pending` 升级为真实 TextWorld smoke。`src/memtotal/tasks/alfworld_env.py` 现在会准备官方 TextWorld 资产、挑选 `valid_seen` 子集，并真实执行一次 hand-coded expert transition，再导出 `data/benchmarks/materialized/alfworld/eval-real-smoke4.jsonl`。
- 2026-03-06 14:18 UTC: `python -m eval --config configs/exp/benchmark_alfworld_qwen25_real_smoke.yaml --seed 707 --output_dir runs/verify/benchmark_alfworld_qwen25_real_smoke` 已真实成功；`./scripts/run_real_benchmark_smoke_suite.sh` 也已重新跑通，最新汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T141831Z/summary.csv`。当前 9 个任务已进入统一 real-source eval，其中 `alfworld` 的 `smoke_subset=textworld_real_smoke4`。
- 2026-03-06 14:46 UTC: 已将 `MemoryAgentBench` 接入统一真实来源 smoke。新增 `src/memtotal/tasks/memoryagentbench.py`、`configs/tasks/benchmarks/materialized/memoryagentbench_real_smoke.yaml` 与 `configs/exp/benchmark_memoryagentbench_qwen25_real_smoke.yaml`，当前从官方 HF 数据集里各选 1 个代表 source 覆盖 `AR / TTL / LRU / CR` 四类能力，并把 capability 分项结果写入 `metrics.json.capability_scores`。
- 2026-03-06 14:46 UTC: `./scripts/setup_benchmark_data.sh`、`python -m eval --config configs/exp/benchmark_memoryagentbench_qwen25_real_smoke.yaml --seed 707 --output_dir runs/verify/benchmark_memoryagentbench_qwen25_real_smoke` 与 `./scripts/run_real_benchmark_smoke_suite.sh` 已真实跑通。最新 real-source smoke 汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T144612Z/summary.csv`。当前统一 real-source eval 已覆盖 10 个任务；其中 `memoryagentbench` 当前是“真实 source + 512-token context truncation”的 smoke scaffold，不是正式长上下文主结果。
- 2026-03-06 15:13 UTC: 已将 `NarrativeQA` 接入统一真实来源 smoke。新增 `configs/tasks/benchmarks/materialized/narrativeqa_real_smoke.yaml` 与 `configs/exp/benchmark_narrativeqa_qwen25_real_smoke.yaml`，当前走官方 `deepmind/narrativeqa` 的 `validation` split，并使用数据集文档明确支持的 `summary_only` 视图；统一评测先接 `qa_f1` 代理口径。
- 2026-03-06 15:13 UTC: `python -m memtotal.tasks.setup_data --benchmarks narrativeqa --max_examples 4 --seed 701 --output_root data/benchmarks/materialized --manifest_root data/benchmarks/manifests --summary_path data/benchmarks/source_summary.json` 与 `python -m eval --config configs/exp/benchmark_narrativeqa_qwen25_real_smoke.yaml --seed 707 --output_dir runs/verify/benchmark_narrativeqa_qwen25_real_smoke` 已真实跑通；当前 materialized `NarrativeQA` 样本会把 `document.id` 扩成 `document.id-q{index}`，避免同一故事的多问题样本在预测产物里撞 `id`。
- 2026-03-06 15:13 UTC: 已为 `scripts/run_real_benchmark_smoke_suite.sh` 增加 `SKIP_SETUP_BENCHMARK_DATA=1`，用于避免数据已存在时重复跑 setup 被 Hugging Face 缓存锁拖住。`SKIP_SETUP_BENCHMARK_DATA=1 ./scripts/run_real_benchmark_smoke_suite.sh` 已真实跑通，最新汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T151341Z/summary.csv`。当前统一 real-source eval 已覆盖 11 个任务。
- 2026-03-06 15:39 UTC: `NarrativeQA` 已从早期 `summary_only` scaffold 升级到 `full_text_segmented` smoke。`src/memtotal/tasks/sources.py` 现在会优先读取官方 `document.text`，切成 `160`-word chunks，剔除明显 front matter 后再均匀抽取 `4` 段进入 materialized JSONL；`src/memtotal/tasks/registry.py` 则会把这些 chunks 组织成真正的 segment-aware prompt。
- 2026-03-06 15:39 UTC: 已修正统一评测里生成式任务的计分路径。`qa_f1` 与 `memoryagentbench` 现在都会评估 `generated_text`，不再误用空字符串占位；`src/memtotal/tasks/setup_data.py` 也已改为增量 merge `data/benchmarks/source_summary.json`，避免单任务 materialize 时把全量 source summary 覆盖掉。
- 2026-03-06 15:39 UTC: `./scripts/setup_benchmark_data.sh`、`python -m eval --config configs/exp/benchmark_narrativeqa_qwen25_real_smoke.yaml --seed 707 --output_dir runs/verify/benchmark_narrativeqa_qwen25_real_smoke`、`SKIP_SETUP_BENCHMARK_DATA=1 ./scripts/run_real_benchmark_smoke_suite.sh` 与 `python -m unittest discover -s tests -v` 已重新真实跑通。最新 real-source smoke 汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T153938Z/summary.csv`；当前全量单测为 `57` 项通过。
- 2026-03-07 00:09 UTC: 已把 `NarrativeQA` 再向论文主线推进一档：`src/memtotal/tasks/sources.py` 现在会将 full story 切成 `128`-word chunks，保留 `6` 段 budget，并通过 `chronological anchors + question-overlap` 选择 story chunks。materialized JSONL 会显式记录 `story_selected_indexes / story_selection_strategy / story_query_token_count`，便于后续追踪 segment 选择。
- 2026-03-07 00:09 UTC: `./scripts/setup_benchmark_data.sh`、`python -m eval --config configs/exp/benchmark_narrativeqa_qwen25_real_smoke.yaml --seed 707 --output_dir runs/verify/benchmark_narrativeqa_qwen25_real_smoke`、`SKIP_SETUP_BENCHMARK_DATA=1 ./scripts/run_real_benchmark_smoke_suite.sh` 与 `python -m unittest discover -s tests -v` 已重新真实跑通。最新 real-source smoke 汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T160937Z/summary.csv`；当前全量单测为 `58` 项通过。
- 2026-03-07 00:20 UTC: 已把 `NarrativeQA` 的选段逻辑从 materialize 前移到 runtime。`src/memtotal/tasks/sources.py` 现在会在 JSONL 中保留完整 `story_chunk_pool`；`src/memtotal/tasks/registry.py` 会在 `load_task_dataset` 时按 `task.narrativeqa_runtime` 选择真正进入 prompt 的 story chunks。这样后续改 budget/selector 时不需要重写数据。
- 2026-03-07 00:20 UTC: `src/memtotal/eval/run_eval.py` 现会把 `story_chunk_pool_size / story_selected_indexes / story_runtime_selector / story_runtime_segment_budget` 这类 NarrativeQA runtime 元数据写入 `predictions.jsonl`。`./scripts/setup_benchmark_data.sh`、`python -m eval --config configs/exp/benchmark_narrativeqa_qwen25_real_smoke.yaml --seed 707 --output_dir runs/verify/benchmark_narrativeqa_qwen25_real_smoke`、`SKIP_SETUP_BENCHMARK_DATA=1 ./scripts/run_real_benchmark_smoke_suite.sh` 与 `python -m unittest discover -s tests -v` 已重新真实跑通；最新 real-source smoke 汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T162018Z/summary.csv`。
- 2026-03-07 00:30 UTC: 已为 `NarrativeQA` 增加更强的结构化正文起点清洗：`src/memtotal/tasks/sources.py` 现在会在 chunk pool 上检测 `dramatis personae / act i / scene i / chapter i / book i / prologue / induction` 这类起点信号，并把 `story_start_index` 写入 materialized JSONL。当前首个样本已从旧版的前言区域推进到 `story_start_index=70`，chunk pool 起点也已落到剧本正文入口附近。
- 2026-03-07 00:30 UTC: 已新增 `configs/exp/benchmark_narrativeqa_qwen3_real_smoke.yaml`，并真实跑通 `python -m eval --config configs/exp/benchmark_narrativeqa_qwen3_real_smoke.yaml --seed 707 --output_dir runs/verify/benchmark_narrativeqa_qwen3_real_smoke`。当前 `Qwen2.5-1.5B-Instruct` 与 `Qwen3-8B` 都能跑同一条 NarrativeQA runtime-pool smoke 路径；最新 qwen25 suite 汇总位于 `results/generated/m4-real-benchmark-smoke/20260306T163014Z/summary.csv`。
- 2026-03-06 16:39 UTC: 已把 `NarrativeQA` 的 runtime selector 补成显式消融契约。`src/memtotal/tasks/sources.py` 现支持 `anchor_only / question_aware / oracle_like_proxy` 三档 selector；`src/memtotal/tasks/registry.py` 会把 selector 与 answer-aware proxy 选择结果写回样本元数据；`src/memtotal/eval/run_eval.py` 会把 `story_runtime_selector / story_runtime_segment_budget` 写入 `metrics.json`。
- 2026-03-06 16:39 UTC: 已新增 `configs/exp/benchmark_narrativeqa_qwen25_real_smoke_anchor_only.yaml` 与 `configs/exp/benchmark_narrativeqa_qwen25_real_smoke_oracle_like_proxy.yaml`，并真实跑通三组 qwen25 selector smoke。统一汇总位于 `results/generated/m4-narrativeqa-selector-ablations/summary.csv`；当前 stub 结果为 `anchor_only=0.0182232353836298`、`question_aware=0.015230493620038033`、`oracle_like_proxy=-0.019175926223397255`（指标为 `mean_similarity`），只用于确认 selector 路径生效。

## Decision Log

- 2026-03-06: 本轮只做 M0/P0 foundation work，不启动重训练或全 benchmark sweep。
- 2026-03-06: backbone 支持范围锁定为 `Qwen2.5-1.5B-Instruct` 与 `Qwen3-8B`；任何新建配置只允许这两档。
- 2026-03-06: 当前目录不是 git worktree；运行记录将保存 git hash，若不可用则显式写入 `nogit`，避免静默缺失。
- 2026-03-06: 先实现 deterministic toy backbone 与 smoke harness，再接真实模型加载，避免一开始把 bootstrap 绑定到高成本权重下载。
- 2026-03-06: `run_info.json` 额外记录 GPU/驱动/显存信息；当前本机为 `NVIDIA RTX PRO 6000 Blackwell`，驱动 `590.44.01`，CUDA `13.1` 环境。
- 2026-03-06: profiling 默认随每次 train/eval/analysis 自动落盘，而不是单独依赖人工执行，减少后续实验漏记 wall time / token / memory 的风险。
- 2026-03-06: `results/` 只允许 `generated/` 与 `reports/` 承载受治理产物，并通过测试机械检查，避免手工主表文件漂移。
- 2026-03-06: MemGen 接入先走 adapter-first 路线，先统一 launch/config/output bridge，再决定是否在当前环境安装其完整依赖并执行真实 benchmark。
- 2026-03-06: 不在 M1 早期直接跑重 benchmark；先确保官方入口、依赖、launch plan、输出桥接都可用，再选择最小可控子集进行首次真实运行。
- 2026-03-06: 对 MemGen 首次真实运行，优先通过代码层可配置项移除 `flash_attention_2` 和全量数据依赖，而不是额外安装 `flash_attn` / `deepspeed` 这类更重的系统依赖。
- 2026-03-06: MemGen 的任务清单对齐先按“真实 smoke + 固定模板 + 统一分析可读”推进，不把 `TODO_LIST.md` 的“主套件全覆盖”偷换成只做单一 benchmark。
- 2026-03-06: 对 gated 数据任务，优先把阻塞点显式写回仓库，并继续推进其他可公开访问的主套件任务，而不是在未认证环境里反复手工重试。
- 2026-03-06: M2 当前只收口“可验证的模块骨架与配置契约”，不提前宣称 Stage A/B/C 已完成；训练流水线仍留在 M3。
- 2026-03-06: Query-Gating 先按配置契约 + 样本级统计落地，不把当前单步 toy runtime 误写成“已经具备正式按-segment 统计”；真正多段统计留给后续 Stage runtime。
- 2026-03-06: 对 `injection_position=none` 这类合法 ablation，不把“无梯度”视为配置错误；训练 harness 应显式记录而不是直接失败。
- 2026-03-06: M3 当前优先保证 Stage A/B/C 的 artifact contract 与 resume 链路成立；若 smoke 结果尚未出现正向 meta gain，应把它记录为未达 DoD，而不是调文案掩盖。
- 2026-03-06: 若 toy smoke 因任务构造本身阻碍 few-shot 观测，应优先重构 toy 数据与 episode 结构，而不是只盲扫学习率。
- 2026-03-06: Stage C 的默认适配对象若与文档定义冲突，应以 `MAIN_IDEA.md` / `EXPERIMENTS_INFO.md` 的“queries-only by default” 为准；writer-inclusive 变体通过显式 `runtime.adaptation_target` 配置进入 ablation，而不是混入默认口径。
- 2026-03-06: Reader 学习方式消融优先通过显式 `query_learning_mode` 工件契约完成，而不是靠临时脚本保存不同 queries 快照；后续真实 benchmark 也应沿用同一 resume 契约。
- 2026-03-06: 训练失败模式检查的目标不是“让所有 smoke 都通过”，而是把退化显式暴露出来；若 canonical smoke 被新检查抓出问题，应优先记录并修结构，而不是降低阈值掩盖。
- 2026-03-06: 对 `writer_noise` 这类带随机性的退化检查，优先通过多次抽样估计期望而不是下调阈值；这样既保留“writer 必须优于噪声”的检查意图，也避免 tiny smoke 的偶然抽样把 harness 打成不稳定平局。
- 2026-03-06: 进入 `M4` 时，优先补“统一 benchmark 契约层”和本地 smoke subset，而不是直接宣称真实 benchmark 已接入；只有当真实数据路径、许可、缓存和统一评测都打通后，才能把条目算作正式完成。
- 2026-03-06: 对 benchmark 数据源，若上游 metadata 没有明确 license 字段，则在仓库内显式写“需核对上游卡片”，而不是靠记忆补许可证；数据合规说明优先准确，不优先好看。

## Surprises & Discoveries

- 当前顶层没有 `.git/`，不能假定能直接读取 commit hash。
- 顶层仅有治理文档和 `MemGen-master/` 子目录，说明 agent legibility 和统一实验入口仍需从零搭建。
- `pip install -e .` 与 smoke/测试链路均能在当前环境直接通过，说明最小 bootstrap 依赖闭环已经成立。
- 将 shell wrapper 纳入 `dev_boot_smoke.sh` 后，脚本层入口也被真实回归覆盖，不再只是“存在但未验证”的壳。
- MemGen 官方实现当前输出的是其自身 `results/.../answer.json` / `launcher.json` 结构；要完全纳入本仓库统一汇总器，还需要后续做真实输出翻译层。
- MemGen 依赖安装后，官方入口只暴露了一个来自其 `math_utils.py` 的 `SyntaxWarning`，但 `--help` 能正常返回，说明当前问题不在 import/环境层。
- MemGen adapter 现已能把 `answer.json` 翻译为统一 `predictions.jsonl` 和摘要指标，但尚未统一生成与我们主方法完全同构的详细 profiling 字段。
- Narrative 侧除了 `rocstories`，`story_cloze` 也能走同一套 builder / adapter / translation 契约，适合继续作为 CDMI smoke 支点。
- `gpqa` 的首要阻塞是 Hub 认证，不是数据预处理或 reward 逻辑；这类问题应该前置为环境规则。
- `triviaqa` 的官方动态评测会反复提示模型输出 `<search>` / `<answer>` 标签；即便 reward 很低，只要统一翻译层能读到 `conversations.txt`，就仍然满足“可评测可汇总”的 M1 目标。
- `torch.nn.TransformerEncoderLayer(norm_first=True)` 会在当前 PyTorch 版本下持续触发 nested tensor warning；把 skeleton writer 改为默认 post-norm 后，warning 已在测试与 smoke 中消失，减少了后续 agent 调试噪声。
- 在 toy runtime 里把 `next_prompt` 作为单块文本编码会掩盖注入位置差异；升级为按 `segment_inputs + delimiter + suffix` 组装后，`segment / delimiter / random / none` 的差异已经真实进入 `injected_inputs` 与生成侧 memory token 宽度。
