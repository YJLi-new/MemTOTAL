# Story Cloze Real Pilot Brief

## 这轮在做什么

这轮不是继续在 `stub` 上扫 `Stage C`，而是先把真实 `Qwen2.5-1.5B-Instruct` 接进来，再用真实 backbone 做一个能拆因的 `story_cloze` pilot。

固定设计：
- 数据：从 `screen256` 里构造 `fixed100`
- backbone：真实 `Qwen2.5-1.5B-Instruct`
- 单 seed
- 同一批 support / eval holdout
- 五条实验臂

五条臂：
- `A = base_only`
- `B = base + shared_summary residual`
- `C = base + candidate_conditioned residual`
- `D = base + candidate_conditioned residual + shuffled memory`
- `E = base + candidate_conditioned residual + choice_ce_plus_margin`

## 关键路径

- 原始 runs：
  - `runs/review/m3-story-cloze-real-pilot-qwen25/`
- 汇总结果：
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/`
- 最关键的对照文件：
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/arm_summary.csv`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/arm_pairwise_compare.csv`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/report.md`

## 结果

这轮 real pilot 的五条臂在同一个 hard `fixed100` 上全部没有翻正收益：

- `A`: `task_score=0.2`
- `B`: `task_score=0.2`
- `C`: `task_score=0.2`
- `D`: `task_score=0.2`
- `E`: `task_score=0.2`

pairwise compare：

- `A -> B`
  - `flip_count_delta=0`
  - `mean_margin_gain=-0.0002198`
- `A -> C`
  - `flip_count_delta=0`
  - `mean_margin_gain=0.0016285`
  - `mean_proxy_gain=0.0002348`
- `C -> D`
  - `flip_count_delta=0`
  - `mean_task_gain=0.0`
- `C -> E`
  - `flip_count_delta=0`
  - `mean_task_gain=0.0`

## 这轮能下的结论

这轮已经能排除一件事：

> 仅靠把 `Stage C` 从 `shared_summary` 换成当前这版 `candidate_conditioned late fusion`，还不足以在真实 qwen25 上带来 `story_cloze` 的 choice flip。

而且 `shuffled memory` 与真实 memory 几乎重合，说明当前这版 experimental branch 里，真实 memory 内容效应还没有显出来。

同样，`choice_ce_plus_margin` 这版 target objective 也没有在这轮 pilot 上带来额外收益。

## 当前更像的 blocker

现在更像的不是“shared-summary 是唯一问题”，而是：

- 当前 `candidate-conditioned residual` 仍然太弱
- residual 可能没有真正进入对的 choice-level decision interface
- `fixed100` 里的样本过难，而且几乎没有自然 near-threshold bucket
- 顺着“推力不够”做的全局 residual calibration 也还是负的：
  - 离线后处理里，`alpha` 只有放到几十以上才开始出现少量 flip
  - 但把 `support_grid_search` 正式接进 runtime 后，不管看 `pilot-support8` 还是额外构造的 `calibration-hard32`，都没选出比 `alpha=1.0` 更好的全局尺度

当前 `fixed100` 的真实 bucket 分布是：
- `base_correct_control = 20`
- `improving_but_unflipped = 40`
- `stubborn_wrong_story_context = 20`
- `stubborn_wrong_other = 20`

也就是说，这批 pilot 样本里并没有自然形成的 `near_threshold_bad`。

## 下一步建议

下一轮不要再回去扫全局 loss/sample。

更合理的顺序是：
- 先扩大 screening pool，重新构造真正带 near-threshold cases 的 fixed-set
- 再做 competitor-aware inner-loop / direct pairwise objective
- 再做 conditional / case-level residual calibration，而不是只看统一的全局 `alpha`
