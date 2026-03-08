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

这条主线当前已经坐实三件事：
- `Phase 0` 已通过：显式 support 文本对 frozen Qwen 有帮助
- `Phase 1` 已通过：当前 writer family 产出的 latent 不是“完全没信息”
- `Phase 2` 不论 shallow 还是 deep path，都已经出现过非零 main-chain consumption 与局部 `I_real > I_shuffle / I_zero` 信号

所以当前 blocker 不再是：
- `Qwen` 会不会读 prefix
- support bank 是否完全没价值
- writer latent 是否完全没信息

而是：
- shared injection 仍然没有在预注册 `screen248-val` 规则下稳定过 selection gate
- shallow prefix 会失稳，stabilized shallow prefix 会把有效学习压平
- sparse deep prompt 虽然提供了更强主链路带宽，但当前会很快顶到 prefix cap，并塌成强标签偏置
- 在真正打开 `screen248-test / fixed64` 双 gate 之前，还需要把“可训练信号”变成“稳定 capability”

## 当前最可信的结论

`M4.3` 已把 shared injection 放到预注册 validation 口径下重新检查，`M4.4` 又补了一轮显式稳定化，最新 `M4.5` 则把注入升级成 `5` 层稀疏 shared low-rank deep prompt。

真实运行路径：
- `runs/review/m4-fever-dynamics-recovery-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-qwen25/`
- `runs/review/m4-fever-dynamics-recovery-stabilized-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-stabilized-qwen25/`
- `runs/review/m4-fever-deep-prompt-recovery-qwen25/`
- `results/generated/review/m4-fever-deep-prompt-recovery-qwen25/`

关键结果：
- [selection.json](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/selection.json)
- [val_selection_report.md](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/val_selection_report.md)
- [prefix_norm_drift.csv](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/prefix_norm_drift.csv)
- [prefix_attention_consumption.csv](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/prefix_attention_consumption.csv)
- [content_gap_curve.csv](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25/dynamics-recovery/content_gap_curve.csv)

`M4.3` 的结论是：
- `selection_passed = false`
- 预注册 `screen248-val` 规则下，`raw8` 和 `triad6` 都没有出现可直接锁定的稳定 checkpoint
- 两条线只有到 `step64` 才出现：
  - `I-real` 相对 `I-shuffle` 的 `+2 flips`
  - `I-real` 相对 `I-zero` 的 `+2 flips`
- 但同时都带来：
  - `regressions_vs_base = 18`
- 因此这轮没有打开 `fixed64`

`M4.4 stabilized` 的结论是：
- norm 控制已经真的生效：
  - `raw8 / I-real` 的 `prefix_l2` 被稳定压在约 `135.76`
  - `triad6 / I-real` 的 `prefix_l2` 也被稳定压在约 `135.76`
- 但稳定化后的行为出现明显分化：
  - `raw8` 全程几乎完全不学，`step0..64` 都没有 `flip_gain`
  - `triad6` 只在 `step64` 恢复出弱的 `I-real > I-shuffle / I-zero`
  - `selection_passed` 仍然是 `false`
- 这说明当前 blocker 已从“norm 爆炸”进一步收缩成：
  - `triad6` 才像真实有效的 support variant
  - 浅层 shared prefix 的有效信号太脆弱
  - 单纯的 norm cap 会把过冲压住，也会把较早的有效学习一起压平

`M4.5 sparse deep prompt` 的最新结论是：
- canonical `triad6 + sparse_deep_prefix(0/7/14/21/27) + rank32 + warmup32 + uniform 4/6 masking` 已真实跑完
- `selection_passed = false`，所以这轮没有进入 `screen248-test / fixed64` 双 gate
- layer-wise `prefix_attention_consumption` 在 `0/7/14/21/27` 五层都非零，说明 deep prompt 确实被主链路消费
- 但 `I_real` 与 `I_shuffle` 的 aggregate `prefix_l2` 都在 `step16` 起迅速顶到 `~192` 的 total cap，并长期贴边
- `I_real` 只在 `step64` 出现弱的 `flip_gain_vs_shuffle=3`、`flip_gain_vs_zero=2`
- 同时 `regressions_vs_base=18`，且 `dominant_label_fraction` 从 `step16` 起基本塌到 `1.0`
- 因而当前新 blocker 已变成：
  - deep prompt 的主链路容量已够让模型“看见并放大某种信号”
  - 但它还没有把 `real-memory content effect` 稳定固化成泛化能力，反而更像很快学成了强偏置前缀

更关键的是，这轮 observability 已经补齐，所以现在能更细地解释失败原因：
- `prefix_attention_consumption` 不再是空的，说明 frozen Qwen 确实在看 prefix
- 但 `prefix_norm_drift` 显示 prefix 范数快速爆炸
  - `raw8 / I-real` 的 `prefix_l2` 从 `84.54` 升到 `7001.36`
  - `triad6 / I-real` 的 `prefix_l2` 从 `86.66` 升到 `11397.91`
- 也就是说，当前主矛盾已经收缩成：
  - 主链路消费存在
  - 但训练动力学明显失稳
- 而 `M4.4 + M4.5` 又进一步说明：
  - 就算把范数爆炸压住，validation gate 仍然不过
  - deep prompt 虽然把控制力推到了层内，但当前又会很快撞到 cap 并塌成偏置前缀
  - 所以下一轮不能只继续堆浅层 norm tweak，也不能把 deep prompt 当前失败简单归结成“checkpoint 没挑对”

因此，最新最稳妥的判断是：

> shared injection 这条路已经出现了正信号，但这个正信号目前仍是“可训练、可消费、但不稳健”。  
> 当前最该修的是 validation 下的 dynamics stability、cap-saturation 与 content-vs-bias 分离，而不是再回去修 score-side residual family。

## 现在不该做什么

- 不再继续修 current `candidate-conditioned residual family`
- 不再做 router / sign selection / 全局 alpha / 更多 seed sweep
- 不直接上 `Story Cloze`
- 不直接上 `Qwen3-8B`
- 不直接上 KL 蒸馏

## 现在最该做什么

继续留在 `shared injection` 主线，但下一步不再只是重复浅层 norm cap。当前优先级应改成：
- 以 `triad6` 为 primary，继续做 dynamics recovery
- 把 `raw8` 降为辅助对照，不再期望它承担主增益
- 保持 `deep prompt / per-layer prefix` 作为当前主线，而不是 fallback 口号
- 把 `screen248-test` 与 `fixed64` 明确降格为 selection 之后才允许打开的双 gate
- 下一轮优先尝试更强 bottleneck / 更强 regularization / 必要时重写 writer objective，而不是回旧 residual family

更具体地说，下一步应优先做：
- 继续用 `screen248-train / screen248-val / fixed64-test` 的三段职责
- 保持 `fixed64` 只在 variant 和 stopping rule 预注册后打开一次
- 围绕 `triad6 + support masking + prefix observability` 做下一轮恢复
- 只有当 `fixed64` 也稳定出现 `I-real > I-shuffle > I-zero`，才进入：
  - `Story Cloze` stress test
  - candidate-conditioned / pair-conditioned injection
  - `Qwen3-8B`

## 关键结果目录

### 当前主线
- [m4-fever-dynamics-recovery-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25)
- [m4-fever-dynamics-recovery-stabilized-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-stabilized-qwen25)
- [m4-fever-deep-prompt-recovery-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-deep-prompt-recovery-qwen25)
- [20260307-m4-shared-injection-brief.md](/root/mydir/MemTOTAL/docs/briefs/20260307-m4-shared-injection-brief.md)

### 已判死的旧分支
- [m3-story-cloze-real-pilot-qwen25](/root/mydir/MemTOTAL/results/generated/review/m3-story-cloze-real-pilot-qwen25)
- [m3-fever-real-pilot-qwen25](/root/mydir/MemTOTAL/results/generated/review/m3-fever-real-pilot-qwen25)

### 当前 active exec plan
- [20260306-initial-bootstrap.md](/root/mydir/MemTOTAL/docs/exec-plans/active/20260306-initial-bootstrap.md)
