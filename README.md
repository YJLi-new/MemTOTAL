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

这条主线当前已经坐实五件事：
- `Phase 0` 已通过：显式 support 文本对 frozen Qwen 有帮助
- `Phase 1` 已通过：当前 writer family 产出的 latent 不是“完全没信息”
- `Phase 2` 不论 shallow 还是 deep path，都已经出现过非零 main-chain consumption 与局部 `I_real > I_shuffle / I_zero` 信号
- `M4.7` 的 structured support-set alignment 已真实跑完：`canonical / freeze-writer / pooled-block` 三臂都没有通过 selection，但 canonical structured path 的最佳点明显强于两个 ablation
- `M5.1` 的 same-schema warm-start + task-first `CE + delayed hinge` 已真实跑完；随后 `M5.2` 的 `task-only / anchor-only / anchor+teacher_margin` objective rewrite 也已真实跑完：`latent anchor` 能保住 warm-start 流形，但 current `teacher_margin` hook 在 canonical 中全程 dormant
- `M5.3` 的 `control-safe-hinge / canonical-dense-teacher(choice-space KL)` 已真实跑完：dense teacher signal 这次不再 dormant，canonical 的 `alignment_aux_active_steps=18/32`，但仍没有通过 `screen248-val`，而且 `step8` 明显弱于 safe-hinge control
- `Workstream B / TL-PoC` 已真实跑完：`SL-8` 能在 `screen248-val` 选出 `step2`，但 `screen248-test` 仍未通过；`TL-H4-K8 / TL-H4-K4 / TL-H1-K4` 三条两层路径全部没有通过 selection，顶层 `tl-poc-summary.json` 当前记录 `comparison_conclusion=failure`、`failure_reason=bridge_not_alive`

所以当前 blocker 不再是：
- `Qwen` 会不会读 prefix
- support bank 是否完全没价值
- writer latent 是否完全没信息
- 单一 `triad6` 静态重放是不是唯一主因

而是：
- shared injection 仍然没有在预注册 FEVER gate 下稳定形成可泛化能力信号
- 单层路径的 objective-side 修补已经基本跑到头：`M5.1/M5.2/M5.3` 都没能把 warm-start 的局部正信号稳定固化成 `I_real > I_shuffle`
- 两层 `Writer -> Reader -> Fuser -> Injector` 路径已经正式接进 active FEVER harness，但当前 `TL-H4-K8 / TL-H4-K4 / TL-H1-K4` 全部没有活过 selection
- TL-PoC 的 diagnostics 更像 `Failure mode B-1 / memory-side capacity-geometry problem`：
  - `M_long` / `M_short` 的 effective rank 在末步仍约 `1.0-1.2`
  - reader attention entropy 约为 `2.0794 ≈ ln(8)`，基本是对 8 个 long slots 的均匀读法
  - `H=4` 没有比 `H=1` 形成更好的 query specialization
- 因而当前不该立刻动 receiver；更合理的下一步是先修两层路径自己的 memory-side readout geometry

## 当前最可信的结论

`M4.3` 已把 shared injection 放到预注册 validation 口径下重新检查，`M4.4` 又补了一轮显式稳定化，`M4.5` 把注入升级成 `5` 层稀疏 shared low-rank deep prompt，`M4.6` 专门测试了 anti-shortcut support protocol，`M4.7` 把 injected path 改成了 `structured support-set encoder -> writer -> sparse deep prompt -> frozen Qwen` 的三臂判因矩阵，`M5.1` 补了 `same-schema warm-start + task-first CE+delayed hinge` 的 writer–reasoner alignment 续跑，最新 `M5.2` 则进一步测试了 `task-only / anchor-only / anchor+teacher_margin` 的 objective rewrite。

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
- `runs/review/m5-fever-writer-reasoner-alignment-qwen25/`
- `results/generated/review/m5-fever-writer-reasoner-alignment-qwen25/`
- `runs/review/m5-fever-writer-objective-rewrite-qwen25/`
- `results/generated/review/m5-fever-writer-objective-rewrite-qwen25/`
- `runs/review/m5-fever-dense-teacher-qwen25/`
- `results/generated/review/m5-fever-dense-teacher-qwen25/`
- `runs/review/tl-poc-fever-qwen25/`
- `results/generated/review/tl-poc-fever-qwen25/`

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

`M5.1 writer–reasoner alignment` 的最新结论是：
- canonical / freeze-writer / pooled-block 三臂都用了有意义的 same-schema warm-start：
  - canonical 与 freeze-writer 都从 `M4.7 canonical step64` 续跑
  - pooled-block 从 `M4.7 pooled-block step64` 续跑
  - 顶层 manifest 见 [warm_start_manifest.json](/root/mydir/MemTOTAL/results/generated/review/m5-fever-writer-reasoner-alignment-qwen25/warm_start_manifest.json)
- canonical 采用 `task-first CE + delayed strongest-competitor hinge`，teacher hook 继续保持 dormant
- 真实结果依然是：
  - [alignment-summary.json](/root/mydir/MemTOTAL/results/generated/review/m5-fever-writer-reasoner-alignment-qwen25/alignment-summary.json)
  - `comparison_conclusion = failure`
  - `failure_reason = canonical_failed_selection`
  - 三臂都没有打开 `screen248-test`，因此 `fixed64` 也没有生成
- 但这轮给出的新判因比 `M4.7` 更强：
  - canonical 的最佳候选其实就是 warm-start 本身的 `step0`
  - `step0`: `macro_f1=0.2259`、`flip_gain_vs_zero=5`、`flip_gain_vs_shuffle=0`、`regressions_vs_base=16`
  - 继续训练到 `step8` 时，canonical 把 `regressions_vs_base` 压到 `1`，但 `flip_gain_vs_shuffle` 仍然没有恢复，`macro_f1` 只有 `0.1775`
  - 到 `step64`，canonical 又退化成 `flip_gain_vs_shuffle=-3`、`flip_gain_vs_zero=2`、`regressions_vs_base=3`
  - `freeze-writer` 最佳只到 `step8`: `flip_gain_vs_shuffle=1`、`flip_gain_vs_zero=2`、`macro_f1=0.1519`
  - `pooled-block` 最佳同样没有恢复 `vs_shuffle`，而且很早就回落到纯偏置态
- 这说明：
  - same-schema warm-start 本身并不够
  - `task-first CE+hinge` 能在早期压掉一部分回归，但仍不能把 `real > shuffle` 稳定起来
  - 下一步最合理的分流已不是 receptor adaptation，而是 `M5.2 writer objective rewrite`

`M5.2 writer objective rewrite` 的最新结论是：
- `task-only-control / anchor-only / canonical(anchor+teacher_margin)` 三臂都没有通过 `screen248-val` selection，因此没有任何一臂打开 `screen248-test`
- 顶层结果位于：
  - [objective-summary.json](/root/mydir/MemTOTAL/results/generated/review/m5-fever-writer-objective-rewrite-qwen25/objective-summary.json)
  - [objective-summary.md](/root/mydir/MemTOTAL/results/generated/review/m5-fever-writer-objective-rewrite-qwen25/objective-summary.md)
  - [warm_start_manifest.json](/root/mydir/MemTOTAL/results/generated/review/m5-fever-writer-objective-rewrite-qwen25/warm_start_manifest.json)
- 这轮最有信息量的新事实不是“三臂又失败了”，而是：
  - `anchor-only` 的 latent anchor 机制确实生效：
    - `step32 anchor_support_cosine≈0.9975`
    - `step32 anchor_writer_slot_cosine≈0.9999`
  - `anchor-only step8` 一度拿到：
    - `macro_f1=0.3175`
    - `flip_gain_vs_shuffle=2`
    - `flip_gain_vs_zero=7`
    - `regressions_vs_base=5`
  - 但仍然没有过 selection
  - canonical 的 `teacher_margin` hook 在 `32` 个训练 step 里 `teacher_margin_aux_active=0`
- 这说明：
  - latent anchor 能保住 warm-start 流形，但单靠保流形还不够
  - 当前 `teacher_margin` hook 太 dormant，这轮并没有真正把 teacher-facing alignment 信号写进训练
  - 下一步不应直接跳 receptor adaptation，而应进入“更强但仍 lightweight 的 engaged teacher-aided objective”

`M5.3 dense teacher under shared injection` 的最新结论是：
- 只跑了 `PLAN.md` 要求的 required pair：
  - `control-safe-hinge`
  - `canonical-dense-teacher(choice-space KL)`
- 顶层结果位于：
  - [dense-teacher-summary.json](/root/mydir/MemTOTAL/results/generated/review/m5-fever-dense-teacher-qwen25/dense-teacher-summary.json)
  - [dense-teacher-summary.md](/root/mydir/MemTOTAL/results/generated/review/m5-fever-dense-teacher-qwen25/dense-teacher-summary.md)
  - [warm_start_manifest.json](/root/mydir/MemTOTAL/results/generated/review/m5-fever-dense-teacher-qwen25/warm_start_manifest.json)
- 这轮把上一个 blocker 又缩了一步：
  - dense teacher signal 这次已经不再 dormant
  - canonical `alignment_aux_active_steps = 18 / 32`
  - `max_align_loss ≈ 0.0156`
  - 所以“teacher hook 根本没有被激活”这条解释现在已经被排除了
- 但 canonical 仍然没有通过 `screen248-val`，而且比 safe-hinge control 更弱：
  - control `step8`: `macro_f1=0.3216`、`flip_gain_vs_shuffle=4`、`flip_gain_vs_zero=7`、`regressions_vs_base=7`
  - canonical `step8`: `macro_f1=0.2879`、`flip_gain_vs_shuffle=0`、`flip_gain_vs_zero=5`、`regressions_vs_base=8`
  - 两臂最终都没有打开 `screen248-test`
  - collapse onset 也没有改善：canonical `step16`，control `step24`
- 这说明：
  - 这次失败不再是“teacher signal 太 sparse / 太 dormant”
  - 而是：即便 choice-space dense teacher KL 真正介入，single-level `writer -> projector -> frozen Qwen` 路径仍然不够稳
  - 按 [PLAN.md](/root/mydir/MemTOTAL/PLAN.md) 的 Workstream A 决策表，当前应停止继续修 single-level FEVER objective，改为进入 `Workstream B / TL-PoC`

因此，最新最稳妥的判断是：

> shared injection 这条路已经出现了正信号，但这个正信号目前仍是“可训练、可消费、但不稳健”。  
> `M5.3` 进一步说明：teacher-facing objective 这次已经真实介入，但 dense teacher 仍没把 single-level substrate 拉过预注册 gate。当前最该做的不是继续在 FEVER 单层 objective 上打转，也不是立刻动 receiver，而是激活 repo 已有的 `Writer -> Reader -> Fuser -> Injector` 两层路径。

`Workstream B / TL-PoC` 的最新结论是：
- 顶层结果位于：
  - [tl-poc-summary.json](/root/mydir/MemTOTAL/results/generated/review/tl-poc-fever-qwen25/tl-poc-summary.json)
  - [tl-poc-summary.md](/root/mydir/MemTOTAL/results/generated/review/tl-poc-fever-qwen25/tl-poc-summary.md)
- `SL-8` 作为 current least-collapsed single-level substrate，能在 `screen248-val` 选出 `step2`
- 但 `SL-8` 仍没有通过 `screen248-test`
- 三条两层变体都没有通过 selection：
  - `TL-H4-K8`
  - `TL-H4-K4`
  - `TL-H1-K4`
- top-level comparison 记录为：
  - `comparison_conclusion=failure`
  - `failure_reason=bridge_not_alive`
  - `bridge_supported=false`
  - `bottleneck_supported=false`
  - `specialization_supported=false`
- 这轮最关键的新事实不是“two-level 也失败了”，而是 failure mode 已经被进一步压缩成 `memory-side capacity / geometry`：
  - `TL-H4-K8` 末步 `memory_long_effective_rank≈1.00`、`memory_short_effective_rank≈1.20`
  - `TL-H4-K4` 末步 `memory_long_effective_rank≈1.00`、`memory_short_effective_rank≈1.11`
  - `TL-H1-K4` 末步 `memory_long_effective_rank≈1.00`、`memory_short_effective_rank≈1.11`
  - reader attention entropy 在三条 two-level run 上都约为 `2.0794 ≈ ln(8)`
  - `TL-H4-K8` 与 `TL-H4-K4` 的 `reader_attention_pairwise_cosine_mean=1.0`
  - `TL-H1-K4` 当然没有 query specialization，而 `H=4` 也没有比它表现出更健康的 specialization
- 这说明：
  - 当前问题已经不再是“如何把 Reader/Fuser 接进 active harness”
  - 也还不到“receiver 完全拒收，所以必须立刻动 `k_proj/v_proj`”的阶段
  - 更像是 `M_long -> Reader -> M_short` 这一层自己的表示几何没有立起来：readout 近似均匀、rank 接近 1、compression 过早塌缩
- 针对这个 `B-1` 解释，最新又补做了一轮 `TL bridge rescue`：
  - 顶层结果位于：
    - [bridge-rescue-summary.json](/root/mydir/MemTOTAL/results/generated/review/tl-bridge-rescue-fever-qwen25/bridge-rescue-summary.json)
    - [bridge-rescue-summary.md](/root/mydir/MemTOTAL/results/generated/review/tl-bridge-rescue-fever-qwen25/bridge-rescue-summary.md)
  - 实现上新增了两类显式 memory-side geometry 修正：
    - `MemoryWriter` 在 `support_set` 路径上保留 conditioned slot identity 的 residual (`support_query_residual_scale=1.0`)
    - 训练期加入 `memory_long / memory_short / reader_attention` diversity regularization
  - 真实结果仍为：
    - `comparison_conclusion=failure`
    - `failure_reason=no_bridge_geometry_gain`
    - `tl_h4_k8_rescue_selection_passed=false`
    - `tl_h4_k8_rescue_primary_gate_passed=false`
  - rescue 没有带来正向几何改善：
    - `tl_h4_k8_rescue_dominant_label_collapse_onset_step=2`
    - `tl_h4_k8_rescue_reader_query_entropy_mean=2.0794`
    - `tl_h4_k8_rescue_reader_query_argmax_unique_mean=0.5885`，低于原始 `TL-H4-K8` 的 `0.6458`
    - `pilot-I-real` 训练事件里，`memory_long_effective_rank` 从 step1 到 step32 基本始终钉在 `≈1.0`
    - `reader_attention_pairwise_cosine_mean` 与新增的 `reader_attention_diversity_loss` 也都持续贴着最坏边界 `1.0`

因此，最新最稳妥的判断应更新为：

> shared injection 主线没有破产，但 current single-level objective family 已基本跑到头；同时，两层路径虽然已经正式进入 active FEVER harness，甚至补做了 first bridge-rescue，也还没有把 bridge 自己做活。下一步最该修的不是 teacher loss，也不是 receiver，而是更具体的 `M_long` 写入 / readout geometry。

## 现在不该做什么

- 不再继续修 current `candidate-conditioned residual family`
- 不再做 router / sign selection / 全局 alpha / 更多 seed sweep
- 不直接上 `Story Cloze`
- 不直接上 `Qwen3-8B`
- 不继续在 single-level FEVER 路径上做 `M5.4 / M5.5` 式的 objective 小修补
- 不直接上 full token-level KL 或 receptor adaptation
- 不在 `bridge_not_alive` 的状态下直接跳去 Workstream C / transfer refresh

## 现在最该做什么

继续留在 `shared injection` 主线，但下一步已经不再是“进入 Workstream B”，因为 Workstream B 的首轮 TL-PoC 已经跑完。当前优先级应改成：
- 保持 `frozen Qwen + structured support-set encoder + sparse deep prefix + two_level path` 这一 substrate，不回旧 residual family
- 保持 `screen248-test` 为 primary capability gate，`fixed64` 只保留为 legacy report
- 继续把 `control-safe-hinge` 视为当前 least-collapsed single-level substrate objective
- 在 two-level path 内优先修 `Failure mode B-1`，而不是 receiver：
  - 优先解决 `M_long` 从 step1 起就接近 rank-1 的写入几何
  - 避免 `Reader` 对 `8` 个 long slots 的近均匀注意力
  - 让 `H=4` 真正出现 query specialization，而不是和 `H=1` 本质等价
  - 避免 `M_short` 在压缩前就塌成近 rank-1
  - 不再把“再加一点 diversity regularization”当成默认主药；下一轮应更直接地约束 long-slot basis / slot factorization
- 只有当 two-level FEVER bridge 真正活起来后，才打开：
  - `Stage B/C` transfer refresh
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
- [m5-fever-writer-reasoner-alignment-qwen25](/root/mydir/MemTOTAL/results/generated/review/m5-fever-writer-reasoner-alignment-qwen25)
- [m5-fever-writer-objective-rewrite-qwen25](/root/mydir/MemTOTAL/results/generated/review/m5-fever-writer-objective-rewrite-qwen25)
- [m5-fever-dense-teacher-qwen25](/root/mydir/MemTOTAL/results/generated/review/m5-fever-dense-teacher-qwen25)
- [tl-poc-fever-qwen25](/root/mydir/MemTOTAL/results/generated/review/tl-poc-fever-qwen25)
- [tl-bridge-rescue-fever-qwen25](/root/mydir/MemTOTAL/results/generated/review/tl-bridge-rescue-fever-qwen25)
- [20260307-m4-shared-injection-brief.md](/root/mydir/MemTOTAL/docs/briefs/20260307-m4-shared-injection-brief.md)

### 已判死的旧分支
- [m3-story-cloze-real-pilot-qwen25](/root/mydir/MemTOTAL/results/generated/review/m3-story-cloze-real-pilot-qwen25)
- [m3-fever-real-pilot-qwen25](/root/mydir/MemTOTAL/results/generated/review/m3-fever-real-pilot-qwen25)

### 当前 active exec plan
- [20260306-initial-bootstrap.md](/root/mydir/MemTOTAL/docs/exec-plans/active/20260306-initial-bootstrap.md)
