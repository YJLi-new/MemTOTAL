# MemTOTAL

面向论文主线的实验仓库：做 `M_long -> M_short` 的 Writer/Reader 记忆压缩、跨域 few-shot 适配和 CDMI 路线验证。

## 当前状态

- 现在已经打通 benchmark-native `M3 core4` smoke：`gsm8k + kodcode + gpqa + story_cloze`。
- Stage C 当前 canonical 用 `target_eval_repeats=3`、`target_episode_repeats=3`、`target_episode_policy=aggregate_support`、`target_split_policy=proxy_bottomk_support`。
- 最新进展：已经补上 `target_split_policy={random,proxy_topk_support,proxy_bottomk_support}` 的 5-seed real sweep，并用新的 canonical 配置重跑了 clean 5-seed q-only seed sweep；另外还新增了 benchmark-native `shot_curve / step_curve` 汇总 harness 和 `step_saturation_audit`。
- 最新结论：`proxy_bottomk_support` 在两档 backbone 上都明显优于 `random`，而 `proxy_topk_support` 稳定最差；clean canonical 5-seed 结果现为 qwen25 `mean_task_gain=0.2593`、qwen3 `mean_task_gain=0.1704`，两档 backbone 都达到 `positive_gain_rate=1.0`。进一步的 curve suite 和 step audit 说明：当前两档 backbone 的 `step0->final` 平均增益都是 `0.0`，也就是现有收益全部在 `zero->step0` 就拿到了。
- 当前新的下一步重点不再是继续调 target split，而是直接审计 Stage C inner-loop 为什么几乎不提供额外收益。
- 当前只支持两个 backbone：`Qwen2.5-1.5B-Instruct`、`Qwen3-8B`。
