# MemTOTAL

面向论文主线的实验仓库：做 `M_long -> M_short` 的 Writer/Reader 记忆压缩、跨域 few-shot 适配和 CDMI 路线验证。

## 当前状态

- 现在已经打通 benchmark-native `M3 core4` smoke：`gsm8k + kodcode + gpqa + story_cloze`。
- Stage C 当前 canonical 是 `target_split_policy=random`、`target_support_bank_size=auto`、`target_support_negative_pool=source_plus_support_bank`；其余仍是 `target_eval_repeats=3`、`target_episode_repeats=3`、`target_episode_policy=aggregate_support`。
- 最新进展：已经修掉 benchmark-native `Stage C` 的评测泄漏。现在 `shot` 不再改变 target episode seed，query/eval 改为固定 holdout pool，support inner-loop 也改为固定 support bank。
- 最新结论：之前那些正向 official gain 主要是协议泄漏，不是可信 few-shot 提升。修正后，canonical `q_only` 在两档 backbone 上先回到 `0.0`；公平 `target_split_policy={random,topk,bottomk}` 重扫后，三档在 official 上完全打平。继续扩 support negative pool 之后，`source_plus_support_bank` 已经成为当前第一条真正更强的公平杠杆：qwen25 的 proxy 明显高于 `support_bank`，qwen3 则拿到了 `mean_task_gain=0.0222`、`positive_gain_rate=0.2`。
- 当前新的下一步重点是继续沿 support negatives 这条线做，而不是回头再调 split policy 或只改 bank 大小。
- 当前只支持两个 backbone：`Qwen2.5-1.5B-Instruct`、`Qwen3-8B`。
