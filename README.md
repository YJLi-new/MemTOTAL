# MemTOTAL

面向论文主线的实验仓库：做 `M_long -> M_short` 的 Writer/Reader 记忆压缩、跨域 few-shot 适配和 CDMI 路线验证。

## 当前状态

- 现在已经打通 benchmark-native `M3 core4` smoke：`gsm8k + kodcode + gpqa + story_cloze`。
- Stage C 当前 runner 仍用 `target_eval_repeats=3`、`target_episode_repeats=3`、`target_episode_policy=aggregate_support`、`target_split_policy=proxy_bottomk_support`，但 `target_split_policy` 现在只算暂定值，需在修正后的公平评测下重新 sweep。
- 最新进展：已经修掉 benchmark-native `Stage C` 的评测泄漏。现在 `shot` 不再改变 target episode seed，query/eval 改为固定 holdout pool，support inner-loop 也改为固定 support bank。
- 最新结论：之前那些正向 official gain 主要是协议泄漏，不是可信 few-shot 提升。修正后，canonical `q_only` 在两档 backbone 上的 5-seed official `mean_task_gain` 都回到 `0.0`；只剩 `1e-6` 级 proxy 变化。
- 当前新的下一步重点是恢复 Stage C 在公平 fixed-holdout eval 下的真实学习信号，然后再重跑 `target_split_policy` 和 few-shot curve。
- 当前只支持两个 backbone：`Qwen2.5-1.5B-Instruct`、`Qwen3-8B`。
