# M4 Shared Injection Brief

## 这轮在做什么

这轮不是继续修 `candidate-conditioned residual family`，而是正式把主战场从“旁路修分”切到“让 frozen Qwen 在主链路里看见并使用 memory”。

固定问题只有一个：

> 如果把 shared latent memory 直接作为 continuous prefix 注入 `Qwen2.5-1.5B-Instruct`，真实 memory 会不会第一次显著优于 `shuffle / zero`？

但这轮没有直接烧 injection 训练，而是先加了两道更上游的 gate：

1. `teacher-text upper bound`
2. `writer information audit`

只有这两道 gate 先通过，才继续跑 `I-real / I-shuffle / I-zero`。

## 关键路径

- 原始 runs：
  - `runs/review/m4-fever-shared-injection-qwen25/`
- 汇总结果：
  - `results/generated/review/m4-fever-shared-injection-qwen25/`
- 最关键的文件：
  - `runs/review/m4-fever-shared-injection-qwen25/pilot-A-base-only/metrics.json`
  - `runs/review/m4-fever-shared-injection-qwen25/pilot-T-teacher-text/metrics.json`
  - `results/generated/review/m4-fever-shared-injection-qwen25/writer-audit/report.md`
  - `results/generated/review/m4-fever-shared-injection-qwen25/writer-audit/summary.csv`

## 这轮实际实现了什么

### 1. Teacher-text sanity check

先不让 writer 压缩任何东西，直接把 support bank 的显式文本拼进 frozen Qwen 的输入，验证 support 本身有没有帮助。

这一步的作用是：

- 如果 `T` 都不比 `A` 强，就说明当前 support serialization / prompt 本身还没给 Qwen 可利用的信号
- 在这种情况下，继续训练 latent injection 没意义

### 2. Writer Information Audit

这一步不训练新 writer，只审计当前 writer family 的 latent 到底有没有可读的任务信息。

这轮没有只做线性 probe，而是同时做了：

- `linear probe`
- `shallow MLP probe`

并且同时比较：

- `real`
- `shuffle`
- `zero`

避免把“线性不可读”误判成“完全没信息”。

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

另外也已经按更稳的工程口径实现：

- 先 warm up `LatentPrefixProjector`
- warmup 期间冻结 writer
- 再联合放开 writer

## 这轮结果

### Phase 0：teacher-text 没有带来收益

- `A = base_only = 0.25`
- `T = teacher-text = 0.25`
- `base_margin = -8.739530039747478e-05`
- `teacher_margin = -0.9794906545430422`

也就是说：

- 当前 support bank 显式拼进 prompt 还没有帮 frozen Qwen
- 而且 teacher 口径下的 margin 明显更差

### Phase 1：writer information audit 也没过 gate

当前审计结果：

- `phase0_support_has_value = false`
- `probe_gate_passed = false`
- `phase1_gate_passed = false`

三类 target 的最佳结果：

- `label_probe`
  - `best_real = 0.7031`
  - `best_control = 0.7500`
  - `gap = -0.0469`
- `base_margin_sign_probe`
  - `best_real = 0.6719`
  - `best_control = 0.7500`
  - `gap = -0.0781`
- `teacher_gain_probe`
  - `best_real = 0.5409`
  - `best_control = 0.4484`
  - `gap = 0.0925`

这里最关键的是：

- `teacher_gain_probe` 虽然 `real` 略高于 control
- 但仍然没过当前 `0.60` 的 gate 下限

因此，这轮并没有进入真正的 injection 训练。

## 现在能下的结论

这轮最重要的结论不是“shared injection 已经失败”，而是：

> 当前还没到能公平评判 shared injection 的那一步。

更准确地说，当前卡住的是更上游的两件事：

1. `support_text` 的显式序列化 / prompt 还没让 frozen Qwen 从 support bank 中受益
2. 当前 writer family 也还没在 `FEVER` 上暴露出足够可读、可消费的任务信息

所以：

- 这轮没有启动 `I-real / I-shuffle / I-zero`
- 也不该据此得出“主链路注入不行”的结论

## 这轮之后最合理的下一步

当前最该做的不是：

- 回去继续修旧 `candidate-conditioned residual`
- 也不是直接上 `Story Cloze`
- 更不是直接上 `Qwen3-8B` 或 KL

而是先把上游两道 gate 修通：

1. 先修 `teacher-text` / support serialization / prompt，让 `T > A`
2. 再提升当前 writer family 在 `FEVER` 上的可读任务信息，至少让 audit 稳定出现 `real > shuffle > zero`

只有这两步先成立，shared injection 才值得真正启动。

## 当前最稳妥的口径

- `candidate-conditioned residual family` 已经停止继续修补
- `M4` 的新主线是对的：把 memory 放回 frozen Qwen 的主链路
- 但现在 immediate blocker 还在 injection 之前
- 因而下一轮的主要任务不是“训练更强 injection”，而是“先让 support bank 和 writer representation 过 gate”
