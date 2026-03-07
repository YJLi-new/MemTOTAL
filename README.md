# MemTOTAL

面向论文主线的实验仓库：研究 `M_long -> M_short` 的通用 Writer / 可适配 Reader 记忆压缩、跨域 few-shot 适配，以及 CDMI 路线验证。

当前代码层只支持两个 backbone：
- `Qwen2.5-1.5B-Instruct`
- `Qwen3-8B`

## 最新进展

- benchmark-native `M3 core4` 主链已经打通：`gsm8k + kodcode + gpqa + story_cloze` 的 `Stage A/B/C`、统一产物、统一分析都可运行。
- 真实 `Qwen2.5-1.5B-Instruct` 的最小闭环已经打通：`BackboneWrapper(load_mode=hf_causal_lm)` 现支持真实 `summarize_texts`、`score_continuations` 与本地 staged model 目录加载。
- 最新判别实验已经完成三步：
  - `story_cloze` 上的真实 qwen25 `fixed100` pilot，外加离线 oracle 审计
  - `FEVER` 上的真实 qwen25 `fixed64` control pilot
  - 基于上述结论重做了更保守的 `shared + candidate delta` real pilot

## 最新结论

- `story_cloze` 这轮 real pilot 的五条臂是：
  - `A = base_only`
  - `B = base + shared_summary residual`
  - `C = base + candidate_conditioned residual`
  - `D = base + candidate_conditioned residual + shuffled memory`
  - `E = base + candidate_conditioned residual + choice-aligned objective`
- 当前 `story_cloze` 结果仍是负的：`A/B/C/D/E` 在同一批 hard `fixed100` 上全部都是 `task_score=0.2`，`flip_count_delta` 全部为 `0`。
- `candidate-conditioned` 相比 `base_only` 只带来了很小的平均 `margin/proxy` 改善：
  - `A -> C`: `mean_margin_gain=0.0016285`
  - `A -> C`: `mean_proxy_gain=0.0002348`
- `C -> D` 和 `C -> E` 基本完全重合：
  - `C -> D`: `mean_task_gain=0.0`
  - `C -> E`: `mean_task_gain=0.0`
- 这轮新增的离线 oracle 审计把问题进一步缩小了：
  - `best-of-two oracle` 仍然只有 `0.2`
  - `per-case alpha oracle` 达到 `0.96`
  - `80` 个 base-wrong case 里，有 `76` 个能被离线 `alpha_i` 翻正
  - 但这 `76` 个全部都需要 `|alpha| >= 32`，而且正负号几乎对半分：`+37 / -39`
- 这说明 `story_cloze` 上不是“residual 完全没信息”，而是：
  - 当前 residual family 缺的是 case-conditional routing / sign selection
  - 单一全局 `alpha` 不足以把它变成可用决策
- `FEVER` control pilot 给出了另一条关键证据：
  - `A = 0.25`
  - `B = 0.75`
  - `C = 0.25`
  - `D = 0.25`
  - 也就是 `shared_summary residual` 在更干净的 multiple-choice task 上是 load-bearing 的，但当前 `candidate_conditioned residual` 分支没有工作，而且和 `shuffled memory` 完全重合
- 这轮新增的更保守分支是：
  - `F = base + shared residual + candidate delta`
  - `G = base + shared residual + candidate delta + shuffled memory`
- `story_cloze` 上，`F` 比 `B` 只多了极小的平均改善，但仍然没有任何 flip，而且 `F` 与 `G` 完全重合：
  - `B -> F`: `mean_task_gain=0.0`
  - `B -> F`: `mean_margin_gain=0.0005981`
  - `F -> G`: `mean_task_gain=0.0`
- `FEVER` 上，`F` 明显优于旧的坏分支 `C`，但仍然低于工作正常的 `B`，而且同样 `F = G`：
  - `B = 0.75`
  - `F = 0.671875`
  - `G = 0.671875`
  - `B -> F`: `flip_count_delta=-5`
  - `C -> F`: `flip_count_delta=27`
  - `F -> G`: `flip_count_delta=0`
- 这轮新增的 `F-G` content audit 已经把问题进一步收紧：
  - `Story Cloze` 上，`B + (F-G)` 仍然是 `0.2`
  - `Story Cloze` 上，`oracle_per_case_alpha_content` 仍然是 `0.2`
  - `Story Cloze` 上，`content_alignment_rate_shared_wrong=0.3375`
  - `FEVER` 上，`B + (F-G)` 仍然是 `0.75`
  - `FEVER` 上，`oracle_per_case_alpha_content` 仍然是 `0.75`
  - `FEVER` 上，`best_of_BF=0.78125` 只说明 branch form 还能补一点 `B` 的盲点，不说明 memory 内容在起作用
  - 两个任务上 `continue_candidate_branch=false`
- 当前最合理的结论不是“memory idea 不行”，而是：
  - `Story Cloze` 仍是 artifact-heavy stress test，不适合作为唯一生死判官
  - `shared_summary residual` 在 `FEVER` 上仍是明确的正 control
  - 当前这条 `candidate` residual family 的主体效应来自 branch form，而不是 real-memory content
  - 因而下一步不该直接做 raw routing / sign selection，也不该继续扩大 `candidate-conditioned` sweep 或上 `Qwen3-8B`
  - 如果后续还要回到 candidate-specific `Stage C`，应从新的 residual family 重新开始，而不是继续修这条 `shared + candidate delta`

## 关键结果路径

- 最新 brief：
  - `docs/briefs/20260307-story-cloze-real-pilot-brief.md`
- Story Cloze real pilot 原始运行：
  - `runs/review/m3-story-cloze-real-pilot-qwen25/`
- Story Cloze real pilot 汇总：
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/`
- Story Cloze 关键 compare / oracle 文件：
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/arm_pairwise_compare.csv`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/arm_summary.csv`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/report.md`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/oracle/oracle_summary.csv`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/oracle/oracle_case_deltas.csv`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/content-audit/report.md`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/content-audit/content_oracle_summary.csv`
- FEVER control pilot 原始运行：
  - `runs/review/m3-fever-real-pilot-qwen25/`
- FEVER control pilot 汇总：
  - `results/generated/review/m3-fever-real-pilot-qwen25/`
  - `results/generated/review/m3-fever-real-pilot-qwen25/compare/arm_pairwise_compare.csv`
  - `results/generated/review/m3-fever-real-pilot-qwen25/content-audit/report.md`
  - `results/generated/review/m3-fever-real-pilot-qwen25/content-audit/content_oracle_summary.csv`

## 现在最重要的下一步

- 不再继续扫全局 loss / sample 网格。
- 不直接进入更大的 `candidate-conditioned` sweep，也不直接上 `Qwen3-8B`。
- 下一轮更值得做的是：
  - 先保留 `FEVER` 作为正 control，不再继续扩 story sweep
  - 不再把精力放在 `candidate delta` 的 routing / sign selection 上，因为这轮 content audit 已经显示 `F-G` 本身几乎不是可用的 memory-only signal
  - 下一轮如果继续探索 candidate-specific `Stage C`，应该优先设计新的 residual family，再先拿 `FEVER` 做 capability probe
  - `Story Cloze` 只保留为后续 stress test，不再作为当前 candidate 分支的主开发面
