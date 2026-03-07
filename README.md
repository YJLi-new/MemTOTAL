# MemTOTAL

面向论文主线的实验仓库：做 `M_long -> M_short` 的 Writer/Reader 记忆压缩、跨域 few-shot 适配和 CDMI 路线验证。

## 当前状态

- 现在已经打通 benchmark-native `M3 core4` smoke：`gsm8k + kodcode + gpqa + story_cloze`。
- Stage C 当前 canonical 用 `target_eval_repeats=3`、`target_episode_repeats=3`，并显式支持 `target_episode_policy`。
- 最新结论：`aggregate_support` 和 `independent` 在同一组 5 seeds 上给出的 `q_only` task gain 基本一样，但 `aggregate_support` 的 target-side update 次数从 `9` 降到 `3`，所以目前保留它作为更省的等价策略。
- 当前真正的 blocker 不是 q-only 不生效，也不是 target episode policy 本身，而是 target-seed 方差仍然偏大。
- 当前只支持两个 backbone：`Qwen2.5-1.5B-Instruct`、`Qwen3-8B`。
