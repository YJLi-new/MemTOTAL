# MemTOTAL

面向论文主线的实验仓库：研究 `M_long -> M_short` 的通用 Writer / 可适配 Reader 记忆压缩、跨域 few-shot 适配，以及 CDMI 路线验证。

当前代码层只支持两个 backbone：
- `Qwen2.5-1.5B-Instruct`
- `Qwen3-8B`

## 最新进展

- benchmark-native `M3 core4` 主链已经打通：`gsm8k + kodcode + gpqa + story_cloze` 的 `Stage A/B/C`、统一产物、统一分析都可运行。
- 真实 `Qwen2.5-1.5B-Instruct` 的最小闭环已经打通：`BackboneWrapper(load_mode=hf_causal_lm)` 现支持真实 `summarize_texts`、`score_continuations` 与本地 staged model 目录加载。
- 最新判别实验已经完成：`story_cloze` 上做了真实 qwen25 的 `100` 条 fixed-set、单 seed、五臂对照 pilot。

## 最新结论

- 这轮 real pilot 的五条臂是：
  - `A = base_only`
  - `B = base + shared_summary residual`
  - `C = base + candidate_conditioned residual`
  - `D = base + candidate_conditioned residual + shuffled memory`
  - `E = base + candidate_conditioned residual + choice-aligned objective`
- 当前结果是负的：`A/B/C/D/E` 在同一批 hard `fixed100` 上全部都是 `task_score=0.2`，`flip_count_delta` 全部为 `0`。
- `candidate-conditioned` 相比 `base_only` 只带来了很小的平均 `margin/proxy` 改善：
  - `A -> C`: `mean_margin_gain=0.0016285`
  - `A -> C`: `mean_proxy_gain=0.0002348`
- 但 `C -> D` 和 `C -> E` 基本完全重合：
  - `C -> D`: `mean_task_gain=0.0`
  - `C -> E`: `mean_task_gain=0.0`
- 这说明：在当前 real qwen25 + 当前实例化下，问题不只是旧的 `shared_summary` 接口；至少这版 `candidate-conditioned late fusion` 还没有带来真实 flip，也还没有显出真实 memory 内容效应。
- 我又顺着“推力不够”做了一步残差标定试探：
  - 离线后处理显示，只有把 `alpha` 放到几十以上才开始出现少量 flip；明显改善甚至要到几千量级
  - 但把 `support_grid_search` 正式接进 runtime 后，无论看 `pilot-support8` 还是额外构造的 `calibration-hard32`，全局单标量 `alpha` 仍然选不出比 `1.0` 更好的值
- 当前更像是：不是完全没有 residual signal，而是单一全局缩放既不够强，也不够有选择性。

## 关键结果路径

- 最新 brief：
  - `docs/briefs/20260307-story-cloze-real-pilot-brief.md`
- 最新 real pilot 原始运行：
  - `runs/review/m3-story-cloze-real-pilot-qwen25/`
- 最新 real pilot 汇总：
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/`
- 关键 compare 文件：
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/arm_pairwise_compare.csv`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/arm_summary.csv`
  - `results/generated/review/m3-story-cloze-real-pilot-qwen25/compare/report.md`
- 额外的 calibration 探针原始运行：
  - `runs/review/m3-story-cloze-real-pilot-qwen25/pilot-F-candidate-calibrated-v2/`

## 现在最重要的下一步

- 不再继续扫全局 loss / sample 网格。
- 优先检查为什么 `candidate-conditioned` 分支在真实 Qwen 上只带来极小 margin 改善，却完全没有 choice flip。
- 下一轮更值得做的是：
  - competitor-aware inner-loop / direct pairwise target objective
  - case-level / conditional residual calibration，而不是单一全局 `alpha`
  - 扩大 screening pool，重新找真正 near-threshold 的 fixed-set
