# MemTOTAL

面向论文主线的实验仓库：做 `M_long -> M_short` 的 Writer/Reader 记忆压缩、跨域 few-shot 适配和 CDMI 路线验证。

## 当前状态

- 现在已经打通 benchmark-native `M3 core4` smoke：`gsm8k + kodcode + gpqa + story_cloze`。
- Stage C 当前 canonical 是 `target_split_policy=random`、`target_support_bank_size=auto`、`target_support_negative_pool=source_plus_support_bank`、`target_support_negative_sampler=hard_by_current_model`；其余仍是 `target_eval_repeats=3`、`target_episode_repeats=3`、`target_episode_policy=aggregate_support`。
- 最新进展：已经修掉 benchmark-native `Stage C` 的评测泄漏。现在 `shot` 不再改变 target episode seed，query/eval 改为固定 holdout pool，support inner-loop 也改为固定 support bank。
- 最新结论：之前那些正向 official gain 主要是协议泄漏，不是可信 few-shot 提升。修正后，canonical `q_only` 在两档 backbone 上先回到 `0.0`；公平 `target_split_policy={random,topk,bottomk}` 重扫后，三档在 official 上完全打平。继续扩 support negative pool 之后，`source_plus_support_bank` 已经成为当前第一条真正更强的公平杠杆：qwen25 的 proxy 明显高于 `support_bank`，qwen3 则拿到过 `mean_task_gain=0.0222`、`positive_gain_rate=0.2`。
- 最新补充结论：`hard_by_continuation` 不够强，但新加的 `hard_by_current_model` 已经在两档 backbone 上都给出当前最高的 proxy gain：qwen25 从 `1.106e-5` 提到 `2.561e-5`，qwen3 从 `3.389e-5` 提到 `4.391e-5`。因此 canonical probe 现切到 `source_plus_support_bank + hard_by_current_model`；但要明确，这一轮 5-seed official `mean_task_gain` 仍然是 `0.0`，所以它是“当前最强 proxy 杠杆”，还不是“已经打通 official gain”的解法。
- 最新新增结论：把 canonical 切到 `hard_by_current_model` 后重跑 `curve suite / step saturation audit`，official `step0->final` 仍然是 `0.0`；但 proxy 会随着 steps 单调上升。也就是说，现在的内循环不是没学到东西，而是 gain 还没跨过 multiple-choice 的 rank-flip 阈值。
- 最新补充结论：进一步做 `margin / rank-flip audit` 后，`cross_zero_margin_rate` 在两档 backbone 上都还是 `0.0`，而 `margin_improves_rate` 都是 `0.6`。这说明后续 steps 现在主要是在把已接近正确或已经正确的 case 再拉开一点，还没有真正救回原本错的 case。
- 最新补充结论：再把 `margin audit` 拆成 `negative_only` 之后，当前更清楚了。两档 backbone 各只有 `2` 个负 margin seeds；qwen25 只有 `1/2` 在缩小 gap，平均只关掉 `1.88e-5`，qwen3 也是 `1/2`，而且平均 gap 还略微变差。也就是说，当前 canonical gain 还没有稳定集中到真正错的 seeds 上。
- 最新补充结论：继续做 `negative-seed shot/step curve audit` 后，当前负 margin seeds 上 `shot` 本身几乎没有纯作用，`shot=0/1/2/3` 的 `gap-to-flip` 完全一样；真正有变化的是 inner-loop `step`。而且两档 backbone 方向相反：qwen25 会极弱地缩小 gap，qwen3 反而会继续放大 gap。
- 当前只支持两个 backbone：`Qwen2.5-1.5B-Instruct`、`Qwen3-8B`。
