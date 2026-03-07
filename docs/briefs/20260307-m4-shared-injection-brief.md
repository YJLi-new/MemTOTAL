# M4 Shared Injection Brief

## 这轮在做什么

这轮不是继续修 `candidate-conditioned residual family`，而是正式把主战场从“旁路修分”切到“让 frozen Qwen 在主链路里看见并使用 memory”。

当前固定问题只有一个：

> 如果把 shared latent memory 直接作为 continuous prefix 注入 `Qwen2.5-1.5B-Instruct`，真实 memory 会不会第一次显著优于 `shuffle / zero`？

但这轮没有直接烧真实 injection 训练，而是先做更上游的恢复：

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
  - `results/generated/review/m4-fever-shared-injection-qwen25/phase2-dryrun-compare/metrics.json`

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

### Phase 0：当前 prompt/support surface 还没把 frozen Qwen带到正确决策面

`screen248` 上的真实结果是：

- `phase0_gate_passed = false`
- `A_winner = inline_short_labels`
- `T_winner = flat_raw8`
- 但两者完全打平，而且都塌缩成全预测 `SUPPORTS`

对应数字：

- `accuracy = 0.29435483870967744`
- `macro_f1 = 0.15160955347871236`
- `dominant_label_fraction = 1.0`
- `label_recall_by_class = {SUPPORTS: 1.0, REFUTES: 0.0, NOT_ENOUGH_INFO: 0.0}`

也就是说：

- 当前 support bank 显式拼进 prompt 还没有帮 frozen Qwen
- 当前 immediate blocker 首先是 `teacher-text / support serialization / label verbalization`

### Phase 1：writer latent 已经显出语义信息，但整体 gate 仍被 Phase 0 卡住

当前审计结果：

- `label_probe_passed = true`
- `semantic_probe_passed = true`
- `phase1_probe_passed = true`
- `phase1_gate_passed = false`

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

这里最关键的是：

- 这轮已经不再支持“writer 完全没信息”这个解释
- 真正没过的是上游 `Phase 0` prompt/support gate

### Phase 2：真实 injection 这轮没有启动，但 dry-run harness 已打通

由于 `Phase 0` 没过，这轮没有启动真实 `I-real / I-shuffle / I-zero`。

但技术上：

- prefix 梯度链路已经打通
- `I-real / I-shuffle / I-zero` 的 dry-run compare 已能端到端产出 `metrics.json`

所以当前不是“注入训练代码还没准备好”，而是“没有理由在一个还没过上游 gate 的 FEVER surface 上继续烧真实 injection 算力”。

## 现在能下的结论

这轮最重要的结论不是“shared injection 已经失败”，而是：

> 当前还没到能公平评判 shared injection 的那一步。

更准确地说，当前卡住的是：

1. `FEVER` 的 prompt / support serialization 还没让 frozen Qwen 从显式 support bank 中受益
2. 当前 writer family 的 latent 已经有可读语义信息，但这些信息还没被 frozen Qwen 的上游输入表面正确解锁

所以：

- 这轮没有启动真实 `I-real / I-shuffle / I-zero`
- 也不该据此得出“主链路注入不行”的结论

## 这轮之后最合理的下一步

当前最该做的不是：

- 回去继续修旧 `candidate-conditioned residual`
- 也不是直接上 `Story Cloze`
- 更不是直接上 `Qwen3-8B` 或 KL

而是先把上游 gate 修通：

1. 先修 `teacher-text` / support serialization / label verbalization，让 `T > A`
2. 在同一套 surface 下保持 writer information audit 的 `real > shuffle / zero`
3. 只有这两步先成立，shared injection 才值得真正启动

## 当前最稳妥的口径

- `candidate-conditioned residual family` 已经停止继续修补
- `M4` 的新主线仍然是对的：把 memory 放回 frozen Qwen 的主链路
- 但现在 immediate blocker 还在 injection 之前
- 因而下一轮的主要任务不是“训练更强 injection”，而是“先让 support bank 和 prompt surface 过 gate”
