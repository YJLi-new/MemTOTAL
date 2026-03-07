# MemTOTAL

面向论文主线的实验仓库：做 `M_long -> M_short` 的 Writer/Reader 记忆压缩、跨域 few-shot 适配和 CDMI 路线验证。

## 当前状态

- 现在已经打通 benchmark-native `M3 core4` smoke：`gsm8k + kodcode + gpqa + story_cloze`。
- Stage C 当前 canonical 用 `target_eval_repeats=3`、`target_episode_repeats=3`，并显式支持 `target_episode_policy`。
- 最新进展：已经补上 `target_episode_repeats={1,3,5}` 的 5-seed sweep。
- 最新结论：在当前 core4 smoke 上，`aggregate_support` 和 `independent` 的分数基本一样，但 `aggregate_support` 更省；同时 `target_episode_repeats=1` 的均值反而优于 `3/5`，说明“更多 target episodes”并不会自动更稳。
- 当前真正的 blocker 已经收口成 target-seed 方差本身，以及为什么更多 target episodes 会开始稀释适配信号。
- 当前只支持两个 backbone：`Qwen2.5-1.5B-Instruct`、`Qwen3-8B`。
