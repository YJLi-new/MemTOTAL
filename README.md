# MemTOTAL

面向论文主线的实验仓库：研究 `M_long -> M_short` 的通用 Writer / 可适配 Reader 记忆压缩、跨域 few-shot 适配，以及 CDMI 路线验证。

当前代码层只支持两个 backbone：
- `Qwen2.5-1.5B-Instruct`
- `Qwen3-8B`

## 最新进展

- benchmark-native `M3 core4` 主链已经打通：`gsm8k + kodcode + gpqa + story_cloze` 的 `Stage A/B/C`、统一产物、统一分析都可运行。
- 真实 `Qwen2.5-1.5B-Instruct` 的最小闭环已经打通：`BackboneWrapper(load_mode=hf_causal_lm)` 现支持真实 `summarize_texts`、`score_continuations` 与本地 staged model 目录加载。
- 最新判别实验已经完成两步：
  - `story_cloze` 上的真实 qwen25 `fixed100` pilot，外加离线 oracle 审计
  - `FEVER` 上的真实 qwen25 `fixed64` control pilot

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
- 当前最合理的结论不是“memory idea 不行”，而是：
  - `Story Cloze` 更像 artifact-heavy stress test
  - 当前坏掉的是这版 `candidate_conditioned late fusion` 实例化
  - 现阶段不该继续做更大的 `candidate-conditioned` sweep，也不该直接上 `Qwen3-8B`

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
- FEVER control pilot 原始运行：
  - `runs/review/m3-fever-real-pilot-qwen25/`
- FEVER control pilot 汇总：
  - `results/generated/review/m3-fever-real-pilot-qwen25/`
  - `results/generated/review/m3-fever-real-pilot-qwen25/compare/arm_pairwise_compare.csv`

## 现在最重要的下一步

- 不再继续扫全局 loss / sample 网格。
- 不直接进入更大的 `candidate-conditioned` sweep，也不直接上 `Qwen3-8B`。
- 下一轮更值得做的是：
  - 先保留 `FEVER` 作为正 control，拆清楚为什么 `shared_summary` 有用而 `candidate_conditioned` 失效
  - 把 `candidate-conditioned` 改成更保守的 decision path，例如 `shared residual + candidate delta`，而不是直接替换成当前这版 choice residual
  - 只有在新的 candidate 分支先通过 `FEVER` 这种 control task 之后，再回到 `Story Cloze`
