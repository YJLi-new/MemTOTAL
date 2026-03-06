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
- 当前 `toy_meta_smoke` 已能体现正的 `Stage B mean_adaptation_gain`，但 Stage C 在 target domain 上目前主要体现为 loss 下降，accuracy 仍持平；后续需要用更丰富的 toy 任务或真实任务验证更强的 few-shot 提升。

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
