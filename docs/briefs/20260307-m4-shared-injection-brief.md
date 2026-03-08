# M4 Shared Injection Brief

## 当前一句话结论

`shared injection` 已经不再是“完全没信号”。最新 stabilized run 说明：prefix norm blow-up 能被压住，但浅层 shared prefix 在预注册 `screen248-val` 口径下仍然不稳；当前 blocker 已从“norm 爆炸”进一步收缩成“support variant 敏感性 + shallow prefix 容量/优化权衡”。

## 这轮真正做了什么

这轮不是继续修旧的 `candidate-conditioned residual family`，而是把主战场切到：

> 让 frozen `Qwen2.5-1.5B-Instruct` 在主 attention 链路里直接消费 latent memory。

当前主线已经分成三段：

1. `Phase 0`
   - support 文本显式拼进 prompt 的 teacher-text sanity check
2. `Phase 1`
   - writer information audit
3. `Phase 2`
   - shared latent prefix injection

## 已经坐实的前提

### Phase 0 已通过

显式 support 文本对 frozen Qwen 有帮助。

对应结果：
- `A_winner` 很差，但不是因为 prompt surface 完全坏掉
- `T_winner` 明显强于 `A_winner`

结论：
- 当前 blocker 不再是 `FEVER` prompt / support 完全没价值

### Phase 1 已通过

当前 writer family 产出的 latent 不是“完全没信息”。

对应结果：
- `real` 在 `label / verifiability / polarity` probe 上均优于 `shuffle/zero`
- 这一步已经不支持“writer 完全没写出任务相关信息”的解释

结论：
- 当前 blocker 不再是 writer 完全无信息

## 当前最新真实结果：M4.3 + M4.4 dynamics recovery

review 路径：
- `runs/review/m4-fever-dynamics-recovery-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-qwen25/`
- `runs/review/m4-fever-dynamics-recovery-stabilized-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-stabilized-qwen25/`

最关键文件：
- `dynamics-recovery/selection.json`
- `dynamics-recovery/val_selection_report.md`
- `dynamics-recovery/prefix_norm_drift.csv`
- `dynamics-recovery/prefix_attention_consumption.csv`
- `dynamics-recovery/content_gap_curve.csv`

### 预注册 validation 结果

当前 `selection_passed=false`。

两条 support variant：
- `raw8`
- `triad6`

都没有在预注册规则下选出一个稳定 checkpoint。

两条线都只有到 `step64` 才同时出现：
- `I-real` 相对 `I-shuffle` 的 `+2 flips`
- `I-real` 相对 `I-zero` 的 `+2 flips`

但同时也都带来：
- `regressions_vs_base = 18`

因此：
- 本轮没有打开 `fixed64`
- 当前还不能说 capability gate 已通过

### M4.4 stabilized 新增结论

这轮显式加入了：
- `prefix slot/total norm cap`
- `grad clip`
- 更温和的 `lr / weight decay`

结果是：
- `raw8`
  - prefix norm 被稳定压住
  - 但 `step0..64` 全程都没有 `flip_gain`
- `triad6`
  - prefix norm 同样被稳定压住
  - 只在 `step64` 恢复出弱的 `I-real > I-shuffle / I-zero`
  - 仍然伴随 `regressions_vs_base=18`
- `selection_passed` 仍然是 `false`

因此最新解释是：

> 简单的 norm control 已经能消掉爆炸，但它还不能把 shared injection 变成预注册 validation 下稳定通过 gate 的方法。  
> 当前问题更像是 shallow prefix 的有效信号太脆弱，而不是 frozen Qwen 完全不读 prefix。

### 这轮新增的关键 observability

现在最值钱的新信息不在最终 accuracy，而在主链路 observability。

1. `prefix_attention_consumption.csv`
- 不再是空文件
- 说明 frozen Qwen 在 scoring choice 时，确实会把 prefix 当成可见上下文消费

2. `prefix_norm_drift.csv`
- 显示 prefix norm 快速爆炸
- `raw8 / I-real`
  - `prefix_l2`: `84.54 -> 7001.36`
- `triad6 / I-real`
  - `prefix_l2`: `86.66 -> 11397.91`

3. `content_gap_curve.csv`
- 说明 content gap 并不是完全没有出现
- 但它没有在 validation 规则下稳定到足以通过 gate

## 当前最稳妥的解释

当前已经可以排除：
- frozen Qwen 完全不看 prefix
- support bank 完全没价值
- writer latent 完全没信息

当前更合理的解释是：

> shared injection 的正信号已经出现过；在没有控制时会过冲，在加了强 norm control 后又会被压得太平。  
> 也就是说，当前问题不再只是 `dynamics stability`，而是 `shallow prefix capacity / optimization tradeoff`。

## 现在不该做什么

- 不再回去修旧的 score-side residual family
- 不直接扩到 `Story Cloze`
- 不直接上 `Qwen3-8B`
- 不直接上 KL
- 不用 `fixed64` 事后挑“最好 step”

## 现在最该做什么

继续留在 `shared injection` 主线，先做：
- 以 `triad6` 为 primary 的下一轮恢复
- 保持 `screen248-val` 的预注册 stopping rule
- 继续做 prefix norm / attention / content-gap observability
- 把 `deep prompt / per-layer prefix` 升级成明确 fallback，而不是继续修旧 residual family

只有当 `fixed64` 也稳定出现：
- `I-real > I-shuffle`
- `I-real > I-zero`

才进入：
- `Story Cloze` stress test
- candidate-conditioned / pair-conditioned injection
- `Qwen3-8B`
