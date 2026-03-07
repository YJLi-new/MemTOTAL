# M4 Shared Injection Brief

## 这轮在做什么

这轮不是继续修 `candidate-conditioned residual family`，而是正式把主战场从“旁路修分”切到“让 frozen Qwen 在主链路里看见并使用 memory”。

当前固定问题只有一个：

> 如果把 shared latent memory 直接作为 continuous prefix 注入 `Qwen2.5-1.5B-Instruct`，真实 memory 会不会第一次显著优于 `shuffle / zero`？

这轮最开始确实先做了更上游恢复：

1. `Phase 0`: `FEVER screen248` 上的 prompt / support gate
2. `Phase 1`: writer information audit
3. 只有前两步都通过，才会在 `fixed64` 上跑真实 `I-real / I-shuffle / I-zero`

## 关键路径

- 原始 runs：
  - `runs/review/m4-fever-shared-injection-qwen25/`
- 汇总结果：
  - `results/generated/review/m4-fever-shared-injection-qwen25/`
- 最关键的文件：
  - `results/generated/review/m4-fever-shared-injection-qwen25/phase0-gate-sweep/report.md`
  - `results/generated/review/m4-fever-shared-injection-qwen25/phase0-gate-sweep/metrics.json`
  - `results/generated/review/m4-fever-shared-injection-qwen25/phase1-writer-audit/report.md`
  - `results/generated/review/m4-fever-shared-injection-qwen25/phase1-writer-audit/summary.csv`
  - `results/generated/review/m4-fever-shared-injection-qwen25/phase2-compare/report.md`
  - `results/generated/review/m4-fever-shared-injection-qwen25/phase2-compare/arm_summary.csv`

## 这轮实际实现了什么

### 1. Phase 0 prompt/support gate

这一步不训练任何参数，只比较 frozen Qwen 在 `FEVER screen248` 上对不同 prompt 和 support surface 的反应。

当前已经补上：

- 3 个 `A` prompt 变体
- 3 个 `T` support serialization 变体
- `macro_f1 / accuracy / dominant_label_fraction / label_recall_by_class`

这一步的作用是：

- 如果 `T` 都不比 `A` 强，就说明 support bank 的显式文本还没让 frozen Qwen 受益
- 在这种情况下，继续训练 latent injection 没意义

### 2. Phase 1 writer information audit

这一步不训练新 writer，只审计当前 writer family 的 latent 到底有没有可读的 FEVER 任务信息。

当前已经实现：

- `linear probe`
- `shallow MLP probe`
- `real / shuffle / zero`

而且 probe target 现在都只预测任务语义本身，不再混入旧的 `teacher_gain_probe`。

### 3. Shared latent prefix injection scaffold

真正的 injected 路径也已经搭好了：

- support text
  -> `backbone.summarize_texts`
  -> `MemoryWriter`
  -> `LatentPrefixProjector`
  -> prefix embeddings
  -> `BackboneWrapper.score_continuations(prefix_embeddings=...)`

第一版固定只做 shared injection：

- 不做 candidate-conditioned
- 不做 pair-conditioned
- 不做 KL
- 不做 deep prompt

另外，这轮还修掉了一个关键 harness 问题：

- `BackboneWrapper.score_continuations(prefix_embeddings=...)` 现在会在 prefix 路径上保留梯度
- 所以 `Phase 2` 的 dry-run 链路已经能技术上完整跑通

## 这轮结果

### Phase 0：prompt/support gate 已通过

`screen248` 上的真实结果已不再是 collapse 到同一条错误 surface。

- `phase0_gate_passed = true`
- `A_winner = answer_slot_labels`
- `T_winner = answer_slot_labels + example_blocks_raw8`
- `selected_pair_accuracy_gain = 0.4274`
- `selected_pair_macro_f1_gain = 0.5352`

对应数字：

- `A_winner`
  - `accuracy = 0.29435483870967744`
  - `macro_f1 = 0.15160955347871236`
  - `dominant_label_fraction = 1.0`
- `T_winner`
  - `accuracy = 0.7217741935483871`
  - `macro_f1 = 0.6868094914612671`
  - `dominant_label_fraction = 0.5241935483870968`

也就是说：

- 当前 support bank 显式拼进 prompt 已经能明显帮助 frozen Qwen
- `teacher-text / support serialization / label verbalization` 不再是当前第一 blocker

### Phase 1：writer latent 的任务语义信号也已通过 gate

当前审计结果：

- `label_probe_passed = true`
- `semantic_probe_passed = true`
- `phase1_probe_passed = true`
- `phase1_gate_passed = true`

最佳 probe 数字：

- `label_probe_3way`
  - `real macro_f1 = 0.4434`
  - `best control = 0.3499`
- `verifiability_probe`
  - `real auroc = 0.7724`
  - `best control = 0.5568`
- `polarity_probe`
  - `real auroc = 0.5534`
  - `best control = 0.4863`

这轮的关键变化是：

- 现在已经不能再把 shared injection 失败归因成“writer 完全没信息”
- prompt/support surface 和 writer 信息这两道上游 gate 都已经通过

### Phase 2：真实 shared injection 已启动，而且 prefix 主链路已经真正动起来了

这轮已经真实跑完：

- `A = base_only`
- `T = teacher-text upper bound`
- `I-real`
- `I-shuffle`
- `I-zero`

最新真实结果是：

- `A = 0.25 / macro_f1 = 0.2`
- `T = 0.53125 / macro_f1 = 0.5294`
- `I-real = 0.390625 / macro_f1 = 0.4061`
- `I-shuffle = 0.546875 / macro_f1 = 0.5031`
- `I-zero = 0.25 / macro_f1 = 0.2`
- `gate_passed = false`

pairwise compare 里最关键的四条是：

- `A -> T: flip_delta = 18`
- `A -> I_real: flip_delta = 9`
- `I_zero -> I_real: flip_delta = 9`
- `I_shuffle -> I_real: flip_delta = -10`

## 现在能下的结论

这轮最重要的结论已经进一步推进了，不再是“shared injection 完全没动”，而是：

> 现在已经到了能公平评判 shared injection 的时候，而且当前这版 shared latent prefix injection 已经让 frozen Qwen 开始消费 prefix；但 current real support latent 的方向仍然是错的，因为 `I-real > I-zero` 却被 `I-shuffle` 反超。

更准确地说，当前卡住的是：

1. `teacher-text` 明确有用，说明 support bank 本身不是空的
2. writer latent 也已有可读任务信息
3. `writer -> latent prefix -> frozen Qwen` 这条主链路已经不再是零效应，因为 `I-real > I-zero`
4. 但 current real support latent 还没有形成正确方向的内容效应，因为 `I-shuffle > I-real`

所以：

- 这轮不能再把问题归因成上游 gate 没过
- 也不能继续把 shared injection 说成“尚未开始”或“完全没动”

## 这轮之后最合理的下一步

当前最该做的不是：

- 回去继续修旧 `candidate-conditioned residual`
- 也不是直接上 `Story Cloze`
- 更不是直接上 `Qwen3-8B` 或 KL

而是直接检查和升级主链路注入本身：

1. 检查 prefix 投影后的幅度、范数和层间可见性
2. 检查 frozen Qwen 是否真的对 prefix token 产生注意力消费
3. 直接解释为什么 `I-shuffle` 会反超 `I-real`
4. 在不回退到 score-side family 的前提下，优先尝试更强的 main-chain injection 方案

## 当前最稳妥的口径

- `candidate-conditioned residual family` 已经停止继续修补
- `M4` 的新主线仍然是对的：把 memory 放回 frozen Qwen 的主链路
- 但现在 immediate blocker 已经不在 injection 之前，而在 injection 本身的内容方向
- 因而下一轮的主要任务不是“继续修 gate”，而是“解释并修正 `real` 为何比 `shuffle` 更差”
