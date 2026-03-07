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

## 第一轮结果

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

## 第一轮能下的结论

这轮已经能排除一件事：

> 仅靠把 `Stage C` 从 `shared_summary` 换成当前这版 `candidate_conditioned late fusion`，还不足以在真实 qwen25 上带来 `story_cloze` 的 choice flip。

而且 `shuffled memory` 与真实 memory 几乎重合，说明当前这版 experimental branch 里，真实 memory 内容效应还没有显出来。

同样，`choice_ce_plus_margin` 这版 target objective 也没有在这轮 pilot 上带来额外收益。

## 第一轮之后更像的 blocker

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

## 第二轮补充：oracle 与 FEVER control

这轮后来又补了两件判别实验。

### 1. Story Cloze 离线 oracle

路径：
- `results/generated/review/m3-story-cloze-real-pilot-qwen25/oracle/`

关键结果：
- `best-of-two oracle` 仍然只有 `0.2`
- `per-case alpha oracle` 达到 `0.96`
- `80` 个 base-wrong cases 里，有 `76` 个能被离线 `alpha_i` 翻正

但这个上界不能被误读成“问题已经解决”。因为这些翻正几乎都依赖极端而且不稳定的 per-case 缩放：
- `76/76` 个被翻正的 wrong cases 全都需要 `|alpha| >= 32`
- 其中正负号几乎对半分：`+37 / -39`

这说明：
- 当前 residual family 里确实有 decision signal
- 但它不是“用一个统一全局 `alpha` 放大一点就能用”的那种 signal
- 当前真正缺的是 case-conditional routing / sign selection

### 2. FEVER real qwen25 control pilot

路径：
- `runs/review/m3-fever-real-pilot-qwen25/`
- `results/generated/review/m3-fever-real-pilot-qwen25/compare/`

四条臂：
- `A = base_only`
- `B = base + shared_summary residual`
- `C = base + candidate_conditioned residual`
- `D = base + candidate_conditioned residual + shuffled memory`

结果：
- `A = 0.25`
- `B = 0.75`
- `C = 0.25`
- `D = 0.25`

pairwise：
- `A -> B`: `flip_count_delta=32`
- `A -> C`: `flip_count_delta=0`
- `C -> D`: `flip_count_delta=0`

这一步很关键。它说明：
- low-bandwidth memory residual 不是整体上都不 load-bearing
- 在更干净的 control task 上，`shared_summary residual` 是真的能帮 decision 的
- 当前坏掉的是这版 `candidate_conditioned late fusion`，而不是 memory idea 本身

## 现在最稳妥的结论

- `Story Cloze` 当前更像 artifact-heavy stress test，而不是总 idea 的生死判官。
- 当前 `candidate_conditioned` 分支的失败，不能直接推出“memory idea 不行”。
- 但它已经足够说明：这版 `candidate_conditioned late fusion` 不能直接扩大 sweep，也不该直接上 `Qwen3-8B`。
- 如果下一轮要继续，应该先在像 `FEVER` 这样的 control task 上把 candidate-conditioned decision path 修到能稳定优于 `shared_summary`，再回到 `Story Cloze`。

## 下一步建议

下一轮不要再回去扫全局 loss/sample。

更合理的顺序现在是：
- 先保留 `FEVER` 作为正 control，修 candidate-conditioned decision path
- 优先尝试 `shared residual + candidate delta` 这类更保守的晚融合，而不是直接用当前这版 candidate residual 替换决策
- 只有当新的 candidate 分支先在 `FEVER` 上跑出真实净增益，再回到 `Story Cloze`

## 第三轮补充：shared + candidate delta

这轮又补了一条更保守的 real qwen25 分支，不再让 candidate-conditioned residual 直接替代 shared 路径，而是改成：

- `F = base + shared residual + candidate delta`
- `G = base + shared residual + candidate delta + shuffled memory`

其中 candidate 增量是：
- 先算 `conditioned_residual - shared_residual`
- 再做跨候选零均值中心化
- 只在 `shared residual` 自己不够确定时，通过一个固定 gate 加到最终分数上

### Story Cloze 结果

路径：
- `runs/review/m3-story-cloze-real-pilot-qwen25/`
- `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/`

结果：
- `F = 0.2`
- `G = 0.2`
- `B -> F`: `flip_count_delta=0`
- `B -> F`: `mean_margin_gain=0.0005981`
- `F -> G`: `flip_count_delta=0`

这说明：
- 新分支没有把 `Story Cloze` 从 `0.2` 拉起来
- 它比 `B` 多了一点点 proxy / margin，但还没有变成任何真实 flip
- 更关键的是，`F` 与 `G` 完全重合，当前仍然看不到 real-memory 内容效应

### FEVER 结果

路径：
- `runs/review/m3-fever-real-pilot-qwen25/`
- `results/generated/review/m3-fever-real-pilot-qwen25/compare/`

结果：
- `A = 0.25`
- `B = 0.75`
- `C = 0.25`
- `D = 0.25`
- `F = 0.671875`
- `G = 0.671875`

pairwise：
- `B -> F`: `flip_count_delta=-5`
- `C -> F`: `flip_count_delta=27`
- `F -> G`: `flip_count_delta=0`

这一步给出的结论非常明确：
- `shared + candidate delta` 确实修掉了旧 `candidate_conditioned` 分支那种“完全不起作用甚至乱推”的最坏状态
- 但它仍然没有带来 real-memory 内容效应，因为 `F = G`
- 而且它还伤害了本来已经工作的 `B=shared_summary residual`

### 到这里最稳妥的结论

- 现在不能说 candidate 增量路径已经成功。
- 更准确的说法是：它把旧坏分支修到了“不会完全胡来”，但还没有成为 load-bearing 的 memory 通道。
- 这时还不能直接跳到 routing / sign selection，因为先要判断 `F-G` 里到底还有没有真实 memory 内容效应。

## 第四轮补充：content audit（先审判 `F-G`）

这轮没有再训练，也没有再重跑 qwen。只复用了现有 `A/B/F/G` review bundle，新增了两组 analysis-only 审计：

- `results/generated/review/m3-story-cloze-real-pilot-qwen25/content-audit/`
- `results/generated/review/m3-fever-real-pilot-qwen25/content-audit/`

核心不是再看 `F` 本身，而是拆三段：

- `delta_candidate_total = F - B`
- `delta_content = F - G`
- `delta_branch = G - B`

并额外构造一个离线 synthetic arm：

- `B_plus_content = B + (F - G)`

### Story Cloze content audit

结果：

- `B = 0.2`
- `F = 0.2`
- `G = 0.2`
- `B_plus_content = 0.2`
- `oracle_best_of_BF = 0.2`
- `oracle_per_case_alpha_content = 0.2`

关键指标：

- `best_of_BF_flip_gain = 0`
- `best_of_B_plus_content_flip_gain = 0`
- `oracle_per_case_alpha_content_flip_gain = 0`
- `content_alignment_rate_shared_wrong = 0.3375`

这说明：

- 当前 `Story Cloze` 上，candidate 分支里几乎没有可用的 `memory-only` 内容信号。
- 不是 “gate 还不够聪明”，而是 `F-G` 这条量本身就没有给出能翻转 official decision 的内容效应。

### FEVER content audit

结果：

- `B = 0.75`
- `F = 0.671875`
- `G = 0.671875`
- `B_plus_content = 0.75`
- `oracle_best_of_BF = 0.78125`
- `oracle_per_case_alpha_content = 0.75`

关键指标：

- `best_of_BF_flip_gain = 2`
- `best_of_B_plus_content_flip_gain = 0`
- `oracle_per_case_alpha_content_flip_gain = 0`
- `content_alignment_rate_shared_wrong = 0.625`

这说明：

- `F` 相对 `B` 的少量补充，不来自 `real-memory content`，而更像来自 branch form 本身。
- 一旦把 `G` 当 control variate 显式减掉，`B + (F-G)` 不会比 `B` 更好。
- 换句话说，当前 candidate 分支还没有通过最基本的“真实内容效应”判别。

## 现在最稳妥的总结论

- `memory idea` 还不能判死，因为 `shared_summary residual` 在 `FEVER` 上仍然是明确正 control。
- 但当前这条 `candidate` residual family 已经可以开始判：它的主体效应来自 branch form，不来自 real-memory content。
- 因此，下一步不该直接做 raw routing / sign selection，也不该继续放大这条 `shared + candidate delta`。
- 如果后续还要继续做 candidate-specific `Stage C`，应该从新的 residual family 重新开始，并且先过 `FEVER` 再回 `Story Cloze`。
