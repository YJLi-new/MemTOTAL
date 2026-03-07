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
- `TaskEvaluator` 现已扩到 `memoryagentbench / qa_f1`，并且生成式任务现已改为真正评估 `generated_text`；但当前仍是本地代理评测版本：`MemoryAgentBench` 只覆盖官方非 API 指标的本地代理，`NarrativeQA` 当前也只是 `qa_f1` smoke 指标。若后续要严格复现实验论文中的 `Long-Range Understanding` 或 NarrativeQA 正式评测，还需要补更接近官方协议的指标与外部评测依赖治理。
- `MemoryAgentBench` 的真实来源 smoke 已打通，但当前为了本地 stub-harness 可运行，materialize 时会把 context 截断到 `512` tokens；正式长上下文实验仍需要补无截断路径、预算说明和更强的 runtime。
- `NarrativeQA` 当前已经从 `summary_only` 升级到 `runtime-pool full_text_segmented` real-source smoke，并且具备 `anchor_only / question_aware / oracle_like_proxy` selector 消融；但这仍然只是“官方 full story -> runtime-selected 6 chunk excerpt”的轻量版本。虽然现在已补了基于 `dramatis personae / act i / chapter i` 的结构化正文起点探测，后续若要把它升级成更强的 CDMI 证据，仍需要补更完整的 full-story runtime、更稳的 front-matter / intro 清洗和正式指标口径。
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
- `MemoryAgentBench` 的真实来源 smoke、四类能力分项与统一汇总入口现已接入：`AR / TTL / LRU / CR` 当前都会进入 `metrics.json` 的 `capability_scores`，并在 `summary.csv` 里展开成独立列。
- `NarrativeQA` 的真实来源 smoke 已升级到 `runtime-pool question-aware full_text_segmented` 视图，并接入统一 registry / eval / summary；当前会进入 `results/generated/m4-real-benchmark-smoke/*/summary.csv`，并与 `Story Cloze / ROCStories` 一起构成 Narrative 域 smoke 入口。
- `NarrativeQA` 当前已具备两档固定 backbone 的 smoke 配置：`Qwen2.5-1.5B-Instruct` 与 `Qwen3-8B` 都能跑通同一条 runtime-pool NarrativeQA 路径。
- `NarrativeQA` selector 消融当前已接入统一汇总；但 qwen25 stub smoke 下的排序是 `anchor_only > question_aware > oracle_like_proxy`（按 `mean_similarity`），这说明 contract 生效了，不说明启发式已接近正式最优，后续仍需要用真实长上下文方法与正式指标重评。
- `Qwen3-8B` 上同一组 NarrativeQA selector smoke 的排序又变成了 `question_aware > anchor_only > oracle_like_proxy`，进一步说明当前 smoke 更像“backbone-sensitive contract check”而不是“稳定实验结论”；后续若要用于论文主表，仍需升级到真实权重与正式指标。
- `M5` 的 `prompting` baseline family 现已接入 `Vanilla / CoT`，并补到了两档固定 backbone以及 `GSM8K + Story Cloze` 的 real-source smoke；但当前仍是 zero-shot smoke 验证，后续还需要补更多真实 benchmark、few-shot/step 预算对齐，以及主表需要的 Prompt Tuning / LoRA / MetaPrompting 家族。
- `prompting / meta_prompting` 当前已支持最小 in-context few-shot demo 注入，并在 `story_cloze` real-source 上真实跑通了 `2-shot` smoke；但这还不是正式 shot-curve，后续仍需要把更多 shots、更多任务和 seeds 纳入统一网格。
- 当前 `story_cloze` baseline grid suite 只覆盖最小 smoke 网格 `shots={0,2}`、`steps={0,4}`；它证明了 suite contract 已经存在，但离 `EXPERIMENTS_INFO.md` 里锁定的正式网格还差很远。
- baseline grid 现已能把 `MemGen` 的 `story_cloze / Qwen2.5-1.5B-Instruct / 0-shot / 0-step` 与 `story_cloze / Qwen3-8B / 0-shot / 0-step` 两个外部点都接进同一条 `adapt_curve.csv`；但这仍只是零样本单点导入，不是正式的 `MemGen` shot/step 网格。
- baseline grid 虽然已经验证了 `allow_missing -> refreshed` 这条补点链路，但当前还没有把外部点成本建模成与内部 baseline 同构的 `shot/step` 预算口径；`MemGen` 仍处于“统一汇总有了，统一预算还没有”的状态。
- `story_cloze` grid 现已进一步扩到 protocol-smoke 版本：`smoke8` 子集上的 `shots={0,1,2,4}`、`steps={0,1,3,5}` 已真实跑通；但这仍只是锁定网格的一个小前缀，距离正式 `{0,1,2,4,8,16,32,64,128} x {0,1,3,5,10,20,50}` 还差得很远。
- `grid.reuse_existing_runs` 现已落地并在 protocol-smoke suite 上验证，可避免仅因导入点/汇总变化而重复执行整套 grid；当前已收紧到 `required artifacts + config.snapshot + seed` 匹配，但仍没有更稳定的显式 config hash / schema version。
- `memory_bank` 现已作为更完整的 memory-agent 风格 scaffold 接入 `story_cloze` real-source smoke、protocol-smoke grid 与 budget audit；但它当前仍是“结构化 memory entries + 有限容量 bank”的最小实现，不是官方 `MemoryBank` 论文级系统复现。
- `M5` 的 `adapter` baseline family 现已接入最小 `Prompt Tuning / LoRA` 闭环，并补到了两档固定 backbone和 `story_cloze` real-source smoke；但当前仍只支持 candidate-selection 任务，后续还需要补到更多任务和更正式的 few-shot/step 网格。
- `M5` 的 `MetaPrompting` 现已接入最小 `planner_critic` scaffold，但当前仍是单次 prompt protocol，而不是正式多轮/多-agent MetaPrompting 复现；后续若要进入主表，需要补更接近原方法的交互与预算口径。
- `MetaPrompting` 当前已补到 `story_cloze` real-source smoke，但还没有进 `gsm8k / narrativeqa / gpqa` 等更强任务，也没有与 Prompt Tuning / LoRA 对齐到正式 shot/step 网格。
- `M5 / P1` 的最小 `LightThinker` 路线现已接入 `story_cloze` real-source smoke、budget audit 与 baseline grid；但当前只是一条 `compress -> answer` prompt scaffold，不是正式 `LightThinker` 论文级复现。
- `M5 / P1` 的最小 `RAG` 路线现已接入 `story_cloze` real-source smoke 与 protocol-smoke grid；但当前只是一条检索式 prompt baseline，不是 `MemoryBank / ExpeL / AWM` 级别的完整系统对照。
- `baseline_budget_audit` 现已能自动检查 `prompting / meta_prompting / adapter / rag / lightthinker / memory_bank` 的预算字段与双 backbone 覆盖；但 `MemGen` 仍未纳入同一条自动预算审计，因为它的外部训练/权重成本还没有在本仓库统一建模。
- benchmark materialize 现在已支持按 `max_examples` 生成不同的 real-smoke 文件名，避免 `smoke4` 与 `smoke8` 互相覆盖；但目前 manifest 仍只记录每个 benchmark 最近一次 materialize 的那个子集，还不是“多子集并存”的完整台账。
- Hugging Face cache 现已迁到 `/root/autodl-tmp/.cache/huggingface`，并新增了固定脚本做 cache 迁移/清理；但这仍是“本机 runbook 级”约束，后续若换机器或换容器镜像，最好把数据盘/cache 根路径做成更显式的配置。
- `MemGen / Qwen3-8B / story_cloze` 的真实 smoke 已完成并接进 dual-import protocol grid；当前剩余缺口不再是“能不能跑通”，而是“如何把 MemGen 的外部训练/权重成本折算进与 prompt/adapter 同构的预算比较”。
