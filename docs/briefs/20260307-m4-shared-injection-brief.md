# M4 Shared Injection Brief

## 当前一句话结论

`shared injection` 已经不是“完全没信号”，但截至 `M5.3` 仍没有在预注册 `screen248-val` 规则下稳定选出 checkpoint。最新 dense-teacher 实验说明：same-schema warm-start、task-first continuation、latent anchor，以及真正会激活的 choice-space teacher KL 都已经被单独测过。当前最真实的 blocker 已经从 `shallow prefix norm blow-up` 继续演化成：

> main-chain consumption 已成立，deep prompt 也已成立；structured support set、trainable writer、same-schema warm-start、latent anchor 也都已显示出方向性增量。  
> 但 current single-level `writer -> projector -> frozen Qwen` 路径，即使在 engaged dense teacher KL 下，仍不能把这条增量稳定固化成 `I_real > I_shuffle` 的可泛化能力信号。所以下一步应从 Workstream A 转入 `Workstream B / TL-PoC`，而不是继续做 `M5.4/M5.5` 式 objective 小修补或立刻做 receptor adaptation。

## 已经坐实的前提

### Phase 0 已通过

显式 support 文本对 frozen Qwen 有帮助。

### Phase 1 已通过

当前 writer family 产出的 latent 不是“完全没信息”。

### 旧 residual family 已判死

`candidate-conditioned residual family` 在更对题的 repair/content audit 下仍然表现为 `real = shuffle = zero`，不再是当前主线。

## M4.3 / M4.4 / M4.5 / M4.6 / M4.7 / M5.1 / M5.2 的连续结论

review 路径：
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

最关键文件：
- `dynamics-recovery/selection.json`
- `dynamics-recovery/dual_gate_summary.json`
- `dynamics-recovery/val_selection_report.md`
- `dynamics-recovery/prefix_norm_drift.csv`
- `dynamics-recovery/prefix_attention_consumption.csv`
- `dynamics-recovery/content_gap_curve.csv`
- `anti-shortcut-comparison.json`
- `anti-shortcut-comparison.md`
- `alignment-summary.json`
- `alignment-summary.md`
- `warm_start_manifest.json`

### M4.3 shallow dynamics recovery

- `raw8` 和 `triad6` 都只在 `step64` 左右出现弱的 `I_real > I_shuffle / I_zero`
- 但同时都有 `regressions_vs_base=18`
- `selection_passed=false`

含义：
- prefix 已经不再是“完全没用”
- 但训练动力学明显失稳

### M4.4 stabilized shallow prefix

- `prefix norm cap + grad clip + 更温和 lr/wd` 成功压住了 norm blow-up
- `raw8` 几乎完全不学
- `triad6` 只在 `step64` 恢复出弱内容信号
- `selection_passed=false`

含义：
- blocker 不再只是数值爆炸
- shallow prefix 的容量/优化权衡本身有问题

### M4.5 triad6 sparse deep prompt

canonical 配置：
- `triad6`
- `sparse_deep_prefix` at layers `0/7/14/21/27`
- shared low-rank projector `rank=32`
- projector-only warmup `32` steps
- total steps `96`
- train-time `4/6` uniform support masking
- validation after selection only; `screen248-test` 与 `fixed64` 都降格为 post-selection gate

真实结果：
- `selection_passed=false`
- `screen248_test_gate_passed=false`
- `fixed64_gate_passed=false`
- 因为 selection 没过，这轮没有实际打开双 gate

关键现象：
- layer-wise `prefix_attention_consumption` 在 `0/7/14/21/27` 五层都非零
- `I_real` aggregate `prefix_l2` 从 `17.95` 在 `step16` 起迅速顶到 `~192`
- `I_shuffle` 也在 `step16` 起顶到同样的 `~192`
- `I_real` 只在 `step64` 出现：
  - `flip_gain_vs_shuffle=3`
  - `flip_gain_vs_zero=2`
- 但同一步同时出现：
  - `regressions_vs_base=18`
- `dominant_label_fraction` 从 `step16` 起基本塌到 `1.0`

含义：
- deep prompt 确实给了更强主链路控制力，模型在多层里会消费 prefix
- 但当前 real/shuffle 都会被快速推到同一个 cap 边界
- 所以当前学到的更像是“强偏置前缀”，不是稳定的 content-sensitive memory usage

### M4.6 anti-shortcut deep prompt recovery

canonical 设计：
- `Run A`: `32` 个 train-time `2/2/2` triad episodes，uniform `5/6` masking
- `Run B`: 固定 `triad6`，其余 optimizer / deep prompt / masking 预算保持相同
- primary gate 只看 `screen248-test`
- `fixed64` 只保留为 legacy report，不再 veto milestone

真实结果：
- `run_a_selection_passed=false`
- `run_b_selection_passed=false`
- `comparison_conclusion=run_a_equals_run_b`
- 两条 run 都没有打开 `screen248-test`，因此也没有生成 `fixed64` report
- 两条 run 的关键 onset 完全一致：
  - `dominant_label_collapse_onset_step=4`
  - `cap_saturation_onset_step=80`
- `Run A` 最接近的候选也只是：
  - `step64/80/96`
  - `flip_gain_vs_shuffle=0`
  - `flip_gain_vs_zero=2`
  - `regressions_vs_base=18`
- `Run B` 基本完全重合

含义：
- “静态 triad6 重放导致 shortcut” 不是当前唯一主因
- 把 support protocol 换成 episode bank，并没有阻止系统极早塌成 label bias
- 当前更接近的是 `writer/projector/frozen-reasoner` 对齐问题，而不是 support bank 组织问题

### M4.7 structured support-set alignment

canonical 设计：
- injected path 不再走 `support_text_block -> pooled summarize`
- 改成 `6` 条 support row 单独 `summarize_texts`
- 再过 `StructuredSupportSetEncoder`
- 再过 `MemoryWriter(input_schema=support_set)`
- 最后进入同一条 `sparse_deep_prefix(0/7/14/21/27, rank=16)` receiver
- 同时并排做三臂：
  - `canonical`
  - `freeze-writer`
  - `pooled-block`

真实结果：
- 三臂都没有通过 `screen248-val` earliest-pass selection
- 因此三臂都没有打开 `screen248-test`，也没有生成 `fixed64` legacy report
- 但 canonical structured path 的最佳点明显强于两个 ablation：
  - `step64`
  - `flip_gain_vs_shuffle=3`
  - `flip_gain_vs_zero=3`
  - `macro_f1=0.2259`
  - `task_score=0.3443`
  - `regressions_vs_base=16`
- `freeze-writer` 最佳只恢复出弱的 `vs_zero`：
  - 最佳在 `step80/96`
  - `flip_gain_vs_shuffle=0`
  - `flip_gain_vs_zero=2`
  - `macro_f1=0.1646`
  - `regressions_vs_base=18`
- `pooled-block` 也只恢复出弱的 `vs_zero`：
  - 最佳在 `step64/80/96`
  - `flip_gain_vs_shuffle=0`
  - `flip_gain_vs_zero=2`
  - `macro_f1=0.1646`
  - `regressions_vs_base=18`
- 三臂都从 `step0` 起表现为 `dominant_label_fraction=1.0`

含义：
- structured support set 确实比 pooled block 更对
- trainable writer 也确实比 freeze-writer 更对
- 但当前这两个改动带来的增量还不足以通过 selection gate
- 所以 blocker 已进一步收紧成：
  - `support representation -> writer -> frozen reasoner` 的对齐仍然不够
  - 问题已经不再只是 support protocol，也不再只是 projector 结构

### M5.1 same-schema warm-start alignment

canonical 设计：
- 继续保留 `structured support-set encoder -> writer -> sparse_deep_prefix -> frozen Qwen`
- canonical 与 freeze-writer 都从 `M4.7 canonical step64` 的 same-schema checkpoint 续跑
- pooled-block 从 `M4.7 pooled-block step64` 续跑
- objective 固定为 `task-first CE + delayed strongest-competitor hinge`
- `teacher_margin` 继续只保留 dormant hook

真实结果：
- 三臂都没有通过 `screen248-val` earliest-pass selection
- 因此三臂都没有打开 `screen248-test`，也没有生成 `fixed64`
- top-level summary 位于：
  - `results/generated/review/m5-fever-writer-reasoner-alignment-qwen25/alignment-summary.json`
  - `results/generated/review/m5-fever-writer-reasoner-alignment-qwen25/alignment-summary.md`
- warm-start manifest 位于：
  - `results/generated/review/m5-fever-writer-reasoner-alignment-qwen25/warm_start_manifest.json`
- canonical 的最佳候选其实是 warm-start 本身的 `step0`：
  - `macro_f1=0.2259`
  - `flip_gain_vs_shuffle=0`
  - `flip_gain_vs_zero=5`
  - `regressions_vs_base=16`
- canonical 在续跑后的关键变化是：
  - `step8`: `macro_f1=0.1775`、`flip_gain_vs_shuffle=0`、`flip_gain_vs_zero=2`、`regressions_vs_base=1`
  - `step64`: `macro_f1=0.2130`、`flip_gain_vs_shuffle=-3`、`flip_gain_vs_zero=2`、`regressions_vs_base=3`
- `freeze-writer` 最佳只到：
  - `step8`
  - `macro_f1=0.1519`
  - `flip_gain_vs_shuffle=1`
  - `flip_gain_vs_zero=2`
  - `regressions_vs_base=0`
- `pooled-block` 最佳是：
  - `step8`
  - `macro_f1=0.2448`
  - `flip_gain_vs_shuffle=0`
  - `flip_gain_vs_zero=1`
  - `regressions_vs_base=13`

含义：
- same-schema warm-start 本身并不够
- task-first `CE + delayed hinge` 的确能在早期把 canonical 的 regression 压下来，但并没有恢复 `real > shuffle`
- 因而这轮进一步排除了：
  - “只是因为 writer 初始化语义太差”
  - “只是因为 canonical 还没有 continuation training”
- 当前最合理的下一步已从 `M5.1` 收紧成：
  - `M5.2 writer objective rewrite`
  - 继续 shared injection
  - 继续 frozen Qwen
  - 不立刻做 receptor adaptation

### M5.2 writer objective rewrite

- 三臂分别为：
  - `task-only-control`
  - `anchor-only`
  - `canonical(anchor+teacher_margin)`
- 三臂都没有通过 `screen248-val`
- `anchor-only` 的 early-step cosine 证明 latent anchor 的保流形机制生效：
  - `step32 anchor_support_cosine≈0.9975`
  - `step32 anchor_writer_slot_cosine≈0.9999`
- `anchor-only step8` 一度拿到：
  - `macro_f1=0.3175`
  - `flip_gain_vs_shuffle=2`
  - `flip_gain_vs_zero=7`
  - `regressions_vs_base=5`
- canonical 的 `teacher_margin` hook 在 `32` 个训练 step 中 `teacher_margin_aux_active=0`

含义：
- `latent anchor` 能保住 warm-start 流形，但单靠保流形还不够
- current `teacher_margin` hook 太 dormant，这轮没有真正把 teacher-facing alignment 信号写进训练
- 当前最合理的下一步已从 `M5.2` 收紧成：
  - `M5.3 engaged teacher-aided objective`
  - 继续 shared injection
  - 继续 frozen Qwen
  - 仍不立刻做 receptor adaptation

### M5.3 dense teacher under shared injection

- 只跑了 `PLAN.md` 要求的 required pair：
  - `control-safe-hinge`
  - `canonical-dense-teacher(choice-space KL)`
- 顶层结果位于：
  - `results/generated/review/m5-fever-dense-teacher-qwen25/dense-teacher-summary.json`
  - `results/generated/review/m5-fever-dense-teacher-qwen25/dense-teacher-summary.md`
  - `results/generated/review/m5-fever-dense-teacher-qwen25/warm_start_manifest.json`
- canonical 的 dense teacher signal 这次已经不再 dormant：
  - `alignment_aux_active_steps = 18 / 32`
  - `max_align_loss ≈ 0.0156`
- 但 canonical 仍然没有通过 `screen248-val`，而且在关键 `step8` 上比 control 更弱：
  - control `step8`: `macro_f1=0.3216`、`flip_gain_vs_shuffle=4`、`flip_gain_vs_zero=7`、`regressions_vs_base=7`
  - canonical `step8`: `macro_f1=0.2879`、`flip_gain_vs_shuffle=0`、`flip_gain_vs_zero=5`、`regressions_vs_base=8`
- 两臂最终都没有打开 `screen248-test`
- collapse onset 也没有改善：
  - control `step24`
  - canonical `step16`

含义：
- 现在已经可以排除“teacher hook 根本没有真正介入训练”
- 但即便 choice-space dense teacher KL 真正介入，single-level `writer -> projector -> frozen Qwen` 路径仍然不够稳
- 按 `PLAN.md` 的 Workstream A 判定表，这轮应视为：
  - dense teacher 没有成为 load-bearing fix
  - stop iterating single-level FEVER objective design
  - move to `Workstream B / TL-PoC`

## 当前最稳妥的解释

当前已经可以排除：
- frozen Qwen 完全不读 prefix
- support bank 完全没价值
- writer latent 完全没信息
- static `triad6` 重放是当前唯一首因

当前更合理的解释是：

> shared injection 这条路已经证明了 main-chain access 和局部内容效应都存在。  
> 但无论 shallow、deep、episode bank、structured support-set alignment、same-schema warm-start continuation，还是 engaged dense teacher KL，当前单层 shared-injection 路径都还不能把这种内容效应稳定固化成通过预注册 validation 的能力信号。

## 现在不该做什么

- 不回旧 residual family
- 不把问题表述成“再挑一个更好的 checkpoint”
- 不把 `fixed64` 当隐式 selection set
- 不直接上 `Story Cloze` 或 `Qwen3-8B`
- 不继续在 FEVER single-level 路径上做更多 objective 小修补
- 不直接上 full token-level KL / receptor adaptation
- 不再把 `episode bank vs static triad6` 当作当前第一主线
- 不把“简单拉长训练步数”当成下一轮主药

## 现在最该做什么

- 继续留在 `shared injection` 主线
- 把 `screen248-test` 固定为 primary capability gate，`fixed64` 只保留为 legacy report
- 把 `control-safe-hinge` 固定为当前 least-collapsed single-level substrate objective
- 下一轮优先进入 `Workstream B / TL-PoC`：
  - `pilot_memory_path_variant: single_level | two_level`
  - 先跑 `TL-H4-K8` bridge，对比 `SL-8`
  - 再跑 `TL-H4-K4` bottleneck
  - 最后跑 `TL-H1-K4` vs `TL-H4-K4` specialization
- 继续增强 observability，重点盯：
  - `memory_long_effective_rank`
  - `memory_short_effective_rank`
  - per-query attention entropy / specialization
  - writer/projector grad ratio
  - label-bias collapse
