# M4 Shared Injection Brief

## 当前一句话结论

`shared injection` 已经不是“完全没信号”，但截至 `M4.6` 仍没有在预注册 `screen248-val` 规则下稳定选出 checkpoint。最新 anti-shortcut 实验说明：把固定 `triad6` 换成 episode bank 也没有救回 selection；当前最真实的 blocker 已经从 `shallow prefix norm blow-up` 继续演化成：

> main-chain consumption 已成立，deep prompt 也已成立；但当前系统仍会极早塌成 label-biased shortcut，而且 episode-bank support protocol 还不足以把它拉回稳定的 real-memory content effect。

## 已经坐实的前提

### Phase 0 已通过

显式 support 文本对 frozen Qwen 有帮助。

### Phase 1 已通过

当前 writer family 产出的 latent 不是“完全没信息”。

### 旧 residual family 已判死

`candidate-conditioned residual family` 在更对题的 repair/content audit 下仍然表现为 `real = shuffle = zero`，不再是当前主线。

## M4.3 / M4.4 / M4.5 / M4.6 的连续结论

review 路径：
- `runs/review/m4-fever-dynamics-recovery-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-qwen25/`
- `runs/review/m4-fever-dynamics-recovery-stabilized-qwen25/`
- `results/generated/review/m4-fever-dynamics-recovery-stabilized-qwen25/`
- `runs/review/m4-fever-deep-prompt-recovery-qwen25/`
- `results/generated/review/m4-fever-deep-prompt-recovery-qwen25/`
- `runs/review/m4-fever-anti-shortcut-recovery-qwen25/`
- `results/generated/review/m4-fever-anti-shortcut-recovery-qwen25/`

最关键文件：
- `dynamics-recovery/selection.json`
- `dynamics-recovery/dual_gate_summary.json`
- `dynamics-recovery/val_selection_report.md`
- `dynamics-recovery/prefix_norm_drift.csv`
- `dynamics-recovery/prefix_attention_consumption.csv`
- `dynamics-recovery/content_gap_curve.csv`
- `anti-shortcut-comparison.json`
- `anti-shortcut-comparison.md`

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

## 当前最稳妥的解释

当前已经可以排除：
- frozen Qwen 完全不读 prefix
- support bank 完全没价值
- writer latent 完全没信息
- static `triad6` 重放是当前唯一首因

当前更合理的解释是：

> shared injection 这条路已经证明了 main-chain access 和局部内容效应都存在。  
> 但无论 shallow、deep，还是 anti-shortcut episode bank，当前训练动力学都还不能把这种内容效应稳定固化成通过预注册 validation 的能力信号。

## 现在不该做什么

- 不回旧 residual family
- 不把问题表述成“再挑一个更好的 checkpoint”
- 不把 `fixed64` 当隐式 selection set
- 不直接上 `Story Cloze` 或 `Qwen3-8B`
- 不先上 KL / teacher matching
- 不再把 `episode bank vs static triad6` 当作当前第一主线

## 现在最该做什么

- 继续留在 `shared injection` 主线
- 把 `screen248-test` 固定为 primary capability gate，`fixed64` 只保留为 legacy report
- 继续增强 observability，重点盯：
  - cap saturation
  - writer/projector grad ratio
  - real-vs-shuffle 的层内分离
  - label-bias collapse
- 下一轮优先进入：
  - `M5 writer–reasoner alignment under shared injection`
  - 先做 task-first 的 objective rewrite
  - 如仍需要 teacher signal，只引入轻量、晚启用的 margin-style distillation
