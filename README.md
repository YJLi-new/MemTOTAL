# MemTOTAL

面向论文主线的实验仓库：做 `M_long -> M_short` 的 Writer/Reader 记忆压缩、跨域 few-shot 适配和 CDMI 路线验证。

## 当前状态

- 现在已经打通 benchmark-native `M3 core4` smoke：`gsm8k + kodcode + gpqa + story_cloze`。
- Stage C 当前 canonical 是 `target_split_policy=random`、`target_support_bank_size=auto`；其余仍是 `target_eval_repeats=3`、`target_episode_repeats=3`、`target_episode_policy=aggregate_support`。
- 最新进展：已经修掉 benchmark-native `Stage C` 的评测泄漏。现在 `shot` 不再改变 target episode seed，query/eval 改为固定 holdout pool，support inner-loop 也改为固定 support bank。
- 最新结论：之前那些正向 official gain 主要是协议泄漏，不是可信 few-shot 提升。修正后，canonical `q_only` 在两档 backbone 上的 5-seed official `mean_task_gain` 都回到 `0.0`；公平 `target_split_policy={random,topk,bottomk}` 重扫后，三档在 official 上完全打平。进一步的 `support_bank_size={max_shot,auto}` 对照显示：`auto` 仍未拉出 official gain，但更容易把 `best_step` 从 `0` 推到 `3`，并在 qwen3 上带来更稳定的 proxy 提升。
- 当前新的下一步重点是继续增强 support-side retrieval 信号，优先扩 support negative pool，而不是再调 split policy。
- 当前只支持两个 backbone：`Qwen2.5-1.5B-Instruct`、`Qwen3-8B`。
