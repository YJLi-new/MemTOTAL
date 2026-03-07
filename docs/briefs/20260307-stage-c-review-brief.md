# Stage C External Review Brief

## 项目背景

`MemTOTAL` 的论文主线是：先把长上下文写成高容量 `M_long`，再通过可适配的 Reader queries 从 `M_long` 里读出低带宽 `M_short`，做跨域 few-shot 适配，并证明它在 CDMI 场景下比 MemGen 更稳。

当前我们已经把方法跑到了 benchmark-native 的 `M3 core4` smoke 协议，而不是只停在 toy：
- `math`: `gsm8k`
- `code`: `kodcode`
- `qa`: `gpqa`
- `narrative`: `story_cloze`

固定 backbone 只有两个：
- `Qwen2.5-1.5B-Instruct`
- `Qwen3-8B`

## 当前已经打通的东西

- `Stage A/B/C` 的 artifact contract、resume 链路、统一 summary 都已经成立。
- `Stage B` 已经有正向 meta-gain 的 smoke 证据：
  - `runs/review/m3-core4-qwen25-stage-b/metrics.json`
  - `runs/review/m3-core4-qwen3-stage-b/metrics.json`
- `Stage C` 的协议泄漏已经修掉：
  - target eval holdout 与 support bank 已解耦
  - `shot` 不再偷偷改变 holdout/query pool
  - `episode_trace.json` 已能落每个 target episode 的组成
- `q_only` 参数化问题已经修过：
  - query residual path 已进入默认方法配置
  - query path 对 score 的影响不再是 `1e-5` 级噪声

## 当前真正的困境

现在的核心问题不是“模型完全没学到”，而是：

1. `margin` 在稳步缩小  
2. `proxy` 在稳步上升  
3. 但 official multiple-choice `task_score` 还是翻不过 rank-flip 阈值

也就是：模型在靠近正确答案，但推力还不够把错误 case 真正翻正。

## 已经排掉的方向

这些方向现在都已经有脚本和真实结果，不再是主 blocker：

- `target_split_policy`
  - fair fixed-holdout 下已排掉
- `target_support_selection_policy`
  - `plain` 与 `label_diverse_if_possible` 没有带来稳定 official 差异
- `retrieval_negative_count`
  - `results/generated/review/m3-core4-stage-c-qonly-negative-count-sweep-v1/metrics.json`
  - `neg_count={3,7,15}` 会改 proxy，但 official `mean_task_gain` 全部仍是 `0.0`
- 纯全局 loss 家族扫描
  - `results/generated/review/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1/metrics.json`
  - `cross_entropy_plus_margin` 是当前最强的 fair loss，但 official 仍然没翻

## 最新进展

### 1. retrieval-loss sweep

同一组 5 seeds 上：

- qwen25
  - `cross_entropy`: `mean_proxy_gain=4.9209594726551395e-05`
  - `cross_entropy_plus_margin`: `mean_proxy_gain=0.00010149743821884494`
  - `mean_margin_gain` 也从 `1.9743320428663031e-04` 提到 `4.072369800673589e-04`
- qwen3
  - `cross_entropy`: `mean_proxy_gain=3.530979156495251e-05`
  - `cross_entropy_plus_margin`: `mean_proxy_gain=7.79112180074204e-05`
  - `mean_margin_gain` 也从 `1.4221276053124032e-04` 提到 `3.138918222652519e-04`

结论：`cross_entropy_plus_margin` 是当前最强的 fair inner-loop loss。

### 2. canonical 已切到更强 loss

当前默认 `Stage C` 已切到：

- `target_support_negative_pool=source_plus_support_bank`
- `target_support_negative_sampler=hard_by_current_model`
- `retrieval_loss_type=cross_entropy_plus_margin`
- `retrieval_margin_value=0.1`

### 3. 但 fresh canonical replay 仍没翻 official

fresh 5-seed replay：
- `results/generated/review/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical/metrics.json`

当前记录：
- qwen25: `mean_task_gain=0.0`
- qwen3: `mean_task_gain=0.0`

所以当前最新结论是：

> 默认口径已经推进到“更强 proxy / 更强 margin”，但 official rank flip 仍未打通。

### 4. 现在已经有 case-level error attribution

最新 canonical case-dump replay：
- `runs/review/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump/`
- `results/generated/review/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump/`
- `results/generated/review/m3-core4-stage-c-error-attribution-v1/`

当前新增的硬证据是：
- `Stage C` 现在会直接写 `task_case_dump.jsonl`
- fresh error-attribution 当前收到了 `61` 个 zero-shot vs best-adapt 配对 case
- 其中真正接近 rank flip 但仍没翻正的只有 `2` 个
  - qwen25: `1`
  - qwen3: `1`
- qwen25 还有 `9` 个 “margin/proxy 都在变好，但还没翻正” 的 case；qwen3 只有 `1` 个
- 很多 remaining wrong cases 会被打上 `story_context_favors_competitor`
  - qwen25: `14`
  - qwen3: `16`

这说明当前 blocker 比之前想得更具体：
- 不是“大量 case 都离翻正只差一点”
- 而是“只有极少数 case 真正贴近边界，且一批错例的 story context 本身就更像 competitor”

所以现在更值得问高手的是：
- 对那极少数 near-threshold cases，应该怎样加非线性推力
- 对那些 `story_context_favors_competitor` 的错例，应该怎样做 case-level 归因与 targeted objective，而不是继续扫全局策略

## 当前最值得外部高手判断的问题

我们现在更需要的不是再扫一个全局 loss / sample 策略，而是以下两类方向的建议：

### 1. 非线性推力

既然线性 margin 改善已经稳定存在，但始终跨不过 rank-flip 阈值，是否应该在接近 flip 边界时引入非线性推力？

例如：
- near-threshold case reweighting
- margin-to-logit amplification
- hard-case-only secondary objective
- step-dependent or confidence-dependent update scaling

### 2. 错误归因

我们下一步准备直接抓那些：
- `margin` 已缩到接近 `0`，例如 `-0.01 ~ -0.001`
- 但最终仍未翻正的 bad cases

重点看：
- 是不是某类长尾词汇卡住了翻转
- 是不是某类逻辑跳跃卡住了翻转
- 是不是 continuation scorer 已经偏向正确答案，但 candidate ranking 仍被某个局部模式锁死

## 可直接打开核验的审阅包

为了解决 `.gitignore` 以前不随仓库携带实际结果的问题，这次已经把关键结果镜像到了可随 repo 一起 push 的路径：

- `runs/review/m3-core4-qwen25-stage-b/`
- `runs/review/m3-core4-qwen3-stage-b/`
- `runs/review/m3-core4-stage-c-qonly-negative-count-sweep-v1/`
- `runs/review/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1/`
- `runs/review/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical/`
- `runs/review/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump/`
- `results/generated/review/m3-core4-stage-c-qonly-negative-count-sweep-v1/`
- `results/generated/review/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1/`
- `results/generated/review/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical/`
- `results/generated/review/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump/`
- `results/generated/review/m3-core4-stage-c-error-attribution-v1/`
- `results/generated/review/m3-core4-stage-c-margin-audit-v3-fixed-holdout/`
- `results/generated/review/m3-core4-stage-c-negative-seed-curve-audit-v2-fixed-holdout/`
- `results/generated/review/m3-core4-stage-c-curve-suite-v3-fixed-holdout/`

刷新脚本：
- `scripts/publish_review_artifacts.sh`
