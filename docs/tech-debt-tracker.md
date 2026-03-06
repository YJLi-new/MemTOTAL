# Tech Debt Tracker

## Open

- 将 `BackboneWrapper` 从当前 stub 模式扩展为真实 Hugging Face/Qwen 加载路径，并补充权重缓存与设备策略。
- 将当前 toy smoke 数据替换为与 `docs/EXPERIMENTS_INFO.md` 主套件兼容的数据准备流水线。
- 增加更严格的结构 lint，覆盖 run 命名、结果目录和 generated-only 报表规则。
- 让 MemGen adapter 在真实运行后补齐 profiling / wall time / token / 显存字段，并与我们自己的 `metrics.json` 结构进一步对齐。
- 评估是否需要兼容官方 `load_model_path=<file>.safetensors` 的旧路径语义；当前 smoke 路径通过 `load_model_path=null` 绕过了这条版本漂移问题。
- 验证 MemGen `trigger.active=True` 路径，并决定哪些坑需要升级为强制脚本规则。
- 继续验证 `gpqa` 在无认证环境下的 preflight 与有认证环境下的真实 smoke 结果是否保持一致可复现。
- 为 `trigger.active=True` 补正式可比的 checkpoint / 权重来源约束；当前仓库只验证了未训练 trigger 的 smoke 路径。
- 当前 `toy_meta_smoke` 的 canonical Stage B run 已能体现正的 `mean_adaptation_gain`，但该信号对 seed 仍敏感；后续需要更稳的 toy 任务或更多 seeds，避免把单个正向 smoke 误当成稳定规律。
- Stage C 适配对象消融现已完成，但在 canonical toy smoke 上仍表现为 `Q-only` 基本不动、`W-only/W+Q` 只降低 loss 而不提升 accuracy；后续需要更丰富的 toy 任务或真实任务验证更强的 few-shot 提升。
- Reader 学习方式消融现已完成，但当前 toy smoke 的信号主要体现在 target zero-shot loss 的排序 `meta-trained < non-meta < random`，而不是 few-shot accuracy 的分离；后续需要更能体现 few-shot query update 的 toy 任务或真实 benchmark。
- `m3_failure_checks` 现已通过三项检查，但 `base_short_slot_diversity=0.004472408443689346` 仍然偏小；后续若迁移到更复杂 toy 任务或真实任务，仍应继续监控 `collapsed_fuser` 间隙是否稳定存在。
- `M4` 当前新增的是本地 benchmark smoke contract，而不是正式 benchmark 数据接入；后续仍需要把 `data/benchmarks/smoke/*.jsonl` 替换成与 `EXPERIMENTS_INFO.md` 对齐的真实下载 / 缓存 / 许可路径。
- `TaskEvaluator` 当前只覆盖 `exact_match / multiple_choice / dataset_label_classification`；如果后续接 `MemoryAgentBench` 分项、代码执行式评测或更复杂 agent reward，需要扩成任务专属 evaluator。
- `M4` 当前虽然已经 materialize 并验证了 `gsm8k / math / gpqa / triviaqa / kodcode / story_cloze / rocstories / fever / alfworld` 的真实来源 smoke 子集，但 `MemoryAgentBench` 的正式数据入口仍未打通。
- `ALFWorld` 当前打通的是 TextWorld transition-style smoke，而不是完整 THOR / visual stack；如果后续论文需要 embodied 视觉结果，需要补 `ai2thor/cv2` 环境、预算说明和更重的运行治理。
- 当前若上游 Hugging Face metadata 没有结构化 license 字段，仓库只会写“需核对上游卡片”而不会自行补写；后续若要对外发布数据副本，需要补更严格的 license 审核流程。
- `rocstories` 当前通过 `hf://` CSV 路径 materialize，而不是老式 dataset script；后续需要确认这种路径在 CI/离线缓存环境中的稳定性。

## Resolved In This Bootstrap

- 缺少统一 CLI、run contract 与结果汇总脚手架。
- 缺少 agent 可重启的 `ExecPlan` 与 M0 架构地图。
- 缺少 profiling、结果治理 lint、artifact 收集验证与脚本 wrapper 回归。
- 缺少 MemGen 统一 dry-run adapter、最小真实 eval 路径、以及官方 `answer.json` 到统一 `predictions.jsonl` 的翻译桥。
- 缺少 MemGen 可复用模板索引、任务矩阵与常见坑记录。
- 统一 analysis 之前没有把 MemGen `compute_reward` 当作主分数字段处理。
- 动态环境任务之前只会写官方 `conversations.txt`，不会进入统一 `predictions.jsonl`；现已补齐动态翻译分支。
- `gpqa` gated dataset 认证缺失此前会在官方进程里晚失败；现已补齐 adapter preflight 并写明 `huggingface-cli login` / `HF_TOKEN` 提示。
- `trigger_active / insertion_profile / requires_trained_checkpoint / load_model_path` 已进入 MemGen adapter 的显式配置契约。
- `kodcode` 的 tokenizers fork 警告已升级为脚本规则：adapter 默认设置 `TOKENIZERS_PARALLELISM=false`。
- `Fuser collapse` blocker 已完成一轮 follow-up 修复：`MemoryFuser` 的 `resampler` 现保留 short-query slot identity，下游 M3 分类/检查改为使用 position-sensitive `summary_proj`，canonical failure checks 已从 `2/3` 提升到 `3/3` 通过。
- `writer_noise` failure check 已改为可配置的多次噪声抽样均值，避免 tiny smoke 上单次噪声抽样导致的高方差误判；当前 canonical 配置使用 `writer_noise_trials=8`。
- `M4` benchmark foundation 已补齐统一 registry / prompt template / evaluator scaffold，本地 smoke 子集与 `scripts/run_benchmark_smoke_suite.sh` 已能真实跑通 6 个代表任务并进入统一汇总。
- `M4` 已进一步补齐 benchmark source registry、数据来源文档、materialize 脚本与 manifest；真实来源 smoke 子集现已覆盖 `gsm8k / gpqa / triviaqa / kodcode / story_cloze / rocstories` 并通过统一 eval 汇总。
