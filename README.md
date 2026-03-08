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

这条主线当前已经坐实四件事：
- `Phase 0` 已通过：显式 support 文本对 frozen Qwen 有帮助
- `Phase 1` 已通过：当前 writer family 产出的 latent 不是“完全没信息”
- `Phase 2` 不论 shallow 还是 deep path，都已经出现过非零 main-chain consumption 与局部 `I_real > I_shuffle / I_zero` 信号
- `M4.7` 的 structured support-set alignment 已真实跑完：`canonical / freeze-writer / pooled-block` 三臂都没有通过 selection，但 canonical structured path 的最佳点明显强于两个 ablation

所以当前 blocker 不再是：
- `Qwen` 会不会读 prefix
- support bank 是否完全没价值
- writer latent 是否完全没信息
- 单一 `triad6` 静态重放是不是唯一主因

而是：
- shared injection 仍然没有在预注册 `screen248-val` 规则下稳定过 selection gate
- shallow prefix 会失稳，stabilized shallow prefix 会把有效学习压平
- sparse deep prompt 虽然提供了更强主链路带宽，但当前会很快塌成强标签偏置
- `M4.7` 的三臂都在 `step0` 就落在 `dominant_label_fraction=1.0` 的塌缩态，且没有任何一臂触发 selection
- 当前更像是 `support representation -> writer -> projector -> frozen reasoner` 的对齐问题，而不只是 static support memorization 或 projector 容量不足

## 当前最可信的结论

`M4.3` 已把 shared injection 放到预注册 validation 口径下重新检查，`M4.4` 又补了一轮显式稳定化，`M4.5` 把注入升级成 `5` 层稀疏 shared low-rank deep prompt，`M4.6` 专门测试了 anti-shortcut support protocol，最新 `M4.7` 则把 injected path 改成了 `structured support-set encoder -> writer -> sparse deep prompt -> frozen Qwen` 的三臂判因矩阵。

真实运行路径：
- `runs/review/m4-fever-dynamics-recovery-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-qwen25/`
- `runs/review/m4-fever-dynamics-recovery-stabilized-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-stabilized-qwen25/`
- `runs/review/m4-fever-deep-prompt-recovery-qwen25/`
- `results/generated/review/m4-fever-deep-prompt-recovery-qwen25/`
- `runs/review/m4-fever-anti-shortcut-recovery-qwen25/`
- `results/generated/review/m4-fever-anti-shortcut-recovery-qwen25/`
- `runs/review/m4-fever-shared-injection-alignment-qwen25/`
- `results/generated/review/m4-fever-shared-injection-alignment-qwen25/`

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

`M4.6 anti-shortcut recovery` 的最新结论是：
- canonical `Run A` 使用 `32` 个 `2/2/2` triad train episodes + uniform `5/6` masking
- control `Run B` 保持固定 `triad6` + 相同 deep prompt / optimizer / masking 预算
- 两条 run 的 `screen248-val` selection 都没有通过：
  - `run_a_selection_passed = false`
  - `run_b_selection_passed = false`
- 两条 run 的 anti-shortcut 汇总结论是：
  - `comparison_conclusion = run_a_equals_run_b`
  - `dominant_label_collapse_onset_step = 4`
  - `cap_saturation_onset_step = 80`
- 两条 run 都没有真正打开 `screen248-test` primary gate，因此也没有生成 `fixed64` legacy report
- `Run A` 在 `step64/80/96` 只恢复出弱的 `flip_gain_vs_zero = 2`，但始终 `flip_gain_vs_shuffle = 0`，同时 `regressions_vs_base = 18`
- `Run B` 与之几乎完全重合

这意味着：
- 把固定 `triad6` 换成 train-time episode bank，本轮并没有把 shared injection 从 shortcut attractor 里拉出来
- 当前 blocker 已不再像“static support memorization 是首因”
- 更接近的问题是：
  - projector-dominant shortcut 形成过早
  - writer latent 虽 probe-readable，但还没有稳定对齐到 frozen reasoner 真正会消费的方向
  - 下一轮应进入 `M5 writer–reasoner alignment under shared injection`

`M4.7 shared injection alignment` 的最新结论是：
- `canonical / freeze-writer / pooled-block` 三臂都没有通过 `screen248-val` earliest-pass selection，因此没有任何一臂打开 `screen248-test`
- 但 canonical structured path 的最佳候选明显强于两个 ablation：
  - canonical `step64`: `flip_gain_vs_shuffle=3`、`flip_gain_vs_zero=3`、`macro_f1=0.2259`、`task_score=0.3443`、`regressions_vs_base=16`
  - freeze-writer 最佳只到 `flip_gain_vs_zero=2`，且 `macro_f1=0.1646`、`regressions_vs_base=18`
  - pooled-block 最佳同样只到 `flip_gain_vs_zero=2`，且 `macro_f1=0.1646`、`regressions_vs_base=18`
- 三臂都从 `step0` 起表现为 `dominant_label_fraction=1.0`，说明当前失败不是“晚期过冲才塌”，而是 injected path 从一开始就更容易落到 label-biased attractor
- 因而，`structured support-set encoder + trainable writer` 这个方向有增量，但增量还不足以跨过预注册 selection gate
- 当前最准确的 blocker 已进一步收紧成：
  - pooled support block 确实比 structured support set 更差
  - freeze-writer 也确实比 trainable writer 更差
  - 但 canonical 仍然没有稳定到可过 gate，所以问题已经上移到 `writer–reasoner alignment`，而不再只是 support protocol 或 projector 结构

因此，最新最稳妥的判断是：

> shared injection 这条路已经出现了正信号，但这个正信号目前仍是“可训练、可消费、但不稳健”。  
> `M4.7` 进一步说明：structured support set 与 trainable writer 确实比两个 ablation 更接近有效方向，但还没有强到穿过预注册 selection gate。当前最该修的是 writer–reasoner alignment，而不是再回去修 score-side residual family。

## 现在不该做什么

- 不再继续修 current `candidate-conditioned residual family`
- 不再做 router / sign selection / 全局 alpha / 更多 seed sweep
- 不直接上 `Story Cloze`
- 不直接上 `Qwen3-8B`
- 不直接上 KL 蒸馏

## 现在最该做什么

继续留在 `shared injection` 主线，但下一步不再只是重复 `M4` 层面的 support protocol 微调。当前优先级应改成：
- 保持 `sparse deep prompt` 这条主链路，不回旧 residual family
- 把 `screen248-test` 固定为 primary capability gate，`fixed64` 只保留为 legacy report
- 把 `episode bank vs static triad6` 这类 support-side 对照降级为辅线，因为 `M4.6` 已经证明它不是当前第一主因
- 进入 `M5.1 writer–reasoner alignment under shared injection`

更具体地说，下一步应优先做：
- 不把“简单把总步数从 96 拉长”当主药；先收紧 trainable stack 的自由度、support representation 和初始化语义
- `freeze-writer` ablation 必须绑定有意义的初始化：至少与 canonical 同源，或直接使用已通过 `Phase 1` audit 的 writer checkpoint，而不是冻结随机 writer
- canonical 继续保持 `task-only`，teacher margin 只保留 dormant hook，不进入主实验矩阵
- 先尝试更对题的 `task loss + strongest-competitor margin` 对齐，再决定是否引入极轻量、晚启用的 teacher-aided margin distillation
- 继续保留现有 layer-wise attention / grad ratio / collapse observability，用来判断 shortcut 是谁先学出来的
- 只有当 `screen248-test` 真正稳定通过后，才打开：
  - `fixed64` legacy report
  - `Story Cloze` stress test
  - candidate-conditioned / pair-conditioned injection
  - `Qwen3-8B`

## 关键结果目录

### 当前主线
- [m4-fever-dynamics-recovery-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-qwen25)
- [m4-fever-dynamics-recovery-stabilized-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-dynamics-recovery-stabilized-qwen25)
- [m4-fever-deep-prompt-recovery-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-deep-prompt-recovery-qwen25)
- [m4-fever-anti-shortcut-recovery-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-anti-shortcut-recovery-qwen25)
- [m4-fever-shared-injection-alignment-qwen25](/root/mydir/MemTOTAL/results/generated/review/m4-fever-shared-injection-alignment-qwen25)
- [20260307-m4-shared-injection-brief.md](/root/mydir/MemTOTAL/docs/briefs/20260307-m4-shared-injection-brief.md)

### 已判死的旧分支
- [m3-story-cloze-real-pilot-qwen25](/root/mydir/MemTOTAL/results/generated/review/m3-story-cloze-real-pilot-qwen25)
- [m3-fever-real-pilot-qwen25](/root/mydir/MemTOTAL/results/generated/review/m3-fever-real-pilot-qwen25)

### 当前 active exec plan
- [20260306-initial-bootstrap.md](/root/mydir/MemTOTAL/docs/exec-plans/active/20260306-initial-bootstrap.md)
