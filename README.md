# MemTOTAL

面向论文主线的实验仓库：研究 `M_long -> M_short` 的通用 Writer / 可适配 Reader 记忆压缩、跨域 few-shot 适配，以及 CDMI 路线验证。

当前代码层只支持两个 backbone：
- `Qwen2.5-1.5B-Instruct`
- `Qwen3-8B`

根目录会持续刷新给外部 review 用的文档包：
- [docs_review_bundle.zip](/root/mydir/MemTOTAL/docs_review_bundle.zip)

## 最新状态

当前已经正式停掉旧的 `candidate-conditioned residual family`。最新主线是：
- `FEVER-first shared generative injection`
- 目标不是旁路修分，而是让 frozen Qwen 在主 attention 链路里直接消费 latent memory

这条主线当前已经有三条成立的前提：
- `Phase 0` 已通过：显式 support 文本对 frozen Qwen 有帮助
- `Phase 1` 已通过：当前 writer family 产出的 latent 不是“完全没信息”
- `Phase 2` 已经出现过 `I-real > I-shuffle > I-zero` 的正向内容信号

所以现在的 blocker 不再是：
- `Qwen` 会不会读 prefix
- support bank 是否完全没价值
- writer latent 是否完全没信息

而是：
- shared injection 的正信号目前还不稳定
- 训练动力学会在后期把已出现的正信号过冲掉
- 需要在不泄漏 `fixed64` 的前提下，用预注册 validation 把这个信号稳定下来

## 当前最可信的结论

`M4.3` 这轮已经把 shared injection 放到预注册 validation 口径下重新检查。

真实运行路径：
- `runs/review/m4-fever-dynamics-recovery-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-qwen25/`

关键结果：
- [selection.json](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/selection.json)
- [val_selection_report.md](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/val_selection_report.md)
- [prefix_norm_drift.csv](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/prefix_norm_drift.csv)
- [prefix_attention_consumption.csv](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/prefix_attention_consumption.csv)
- [content_gap_curve.csv](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/content_gap_curve.csv)

当前结论是：
- `selection_passed = false`
- 预注册 `screen248-val` 规则下，`raw8` 和 `triad6` 都没有出现可直接锁定的稳定 checkpoint
- 两条线只有到 `step64` 才出现：
  - `I-real` 相对 `I-shuffle` 的 `+2 flips`
  - `I-real` 相对 `I-zero` 的 `+2 flips`
- 但同时都带来：
  - `regressions_vs_base = 18`
- 因此这轮没有打开 `fixed64`

更关键的是，这轮 observability 已经补齐，所以现在能更细地解释失败原因：
- `prefix_attention_consumption` 不再是空的，说明 frozen Qwen 确实在看 prefix
- 但 `prefix_norm_drift` 显示 prefix 范数快速爆炸
  - `raw8 / I-real` 的 `prefix_l2` 从 `84.54` 升到 `7001.36`
  - `triad6 / I-real` 的 `prefix_l2` 从 `86.66` 升到 `11397.91`
- 也就是说，当前主矛盾已经收缩成：
  - 主链路消费存在
  - 但训练动力学明显失稳

因此，最新最稳妥的判断是：

> shared injection 这条路已经出现了正信号，但这个正信号目前是“可训练但不稳健”的。  
> 当前最该修的是 validation 下的 dynamics stability，而不是再回去修 score-side residual family。

## 现在不该做什么

- 不再继续修 current `candidate-conditioned residual family`
- 不再做 router / sign selection / 全局 alpha / 更多 seed sweep
- 不直接上 `Story Cloze`
- 不直接上 `Qwen3-8B`
- 不直接上 KL 蒸馏

## 现在最该做什么

继续留在 `shared injection` 主线，优先解决：
- `support variant`
- 训练动力学
- 预注册 early-stop / capability gate

更具体地说，下一步应优先做：
- 继续用 `screen248-train / screen248-val / fixed64-test` 的三段职责
- 保持 `fixed64` 只在 variant 和 stopping rule 预注册后打开一次
- 围绕 `support masking + prefix norm control + attention observability` 做 dynamics recovery
- 只有当 `fixed64` 也稳定出现 `I-real > I-shuffle > I-zero`，才进入：
  - `Story Cloze` stress test
  - candidate-conditioned / pair-conditioned injection
  - `Qwen3-8B`

## 关键结果目录

### 当前主线
- [m4-fever-dynamics-recovery-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25)
- [20260307-m4-shared-injection-brief.md](/root/mydir/MemTOTAL/docs/briefs/20260307-m4-shared-injection-brief.md)

### 已判死的旧分支
- [m3-story-cloze-real-pilot-qwen25](/root/mydir/MemTOTAL/results/generated/review/m3-story-cloze-real-pilot-qwen25)
- [m3-fever-real-pilot-qwen25](/root/mydir/MemTOTAL/results/generated/review/m3-fever-real-pilot-qwen25)

### 当前 active exec plan
- [20260306-initial-bootstrap.md](/root/mydir/MemTOTAL/docs/exec-plans/active/20260306-initial-bootstrap.md)
