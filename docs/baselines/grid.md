# Baseline Grid Smoke

本文件记录 `M5` 当前最小 baseline `shot/step` grid smoke。

## Scope

- 任务：`story_cloze` real-source smoke
- backbones:
  - `Qwen2.5-1.5B-Instruct`
  - `Qwen3-8B`
- families:
  - `prompting`
  - `meta_prompting`
  - `adapter`
- 当前 smoke 网格：
  - `shots = {0, 2}`
  - `steps = {0, 4}`

说明：

- `prompting / meta_prompting` 当前只有 `step=0`
- `adapter` 当前支持 `shot=0, step=0` 的 zero-adaptation init 点
- `adapter` 的 `shot=0, step>0` 会被自动剪掉，因为没有 support 数据可更新

## Entry Points

- config: `configs/exp/m5_story_cloze_baseline_grid_smoke.yaml`
- script: `scripts/run_story_cloze_baseline_grid.sh`
- module: `python -m memtotal.baselines.grid_runner`

## Verified Commands

```bash
./scripts/run_story_cloze_baseline_grid.sh 991 results/generated/m5-story-cloze-baseline-grid-smoke
./scripts/run_story_cloze_baseline_grid.sh 992 results/generated/m5-story-cloze-baseline-grid-smoke-dryrun --dry-run
```

## Verified Outputs

- `results/generated/m5-story-cloze-baseline-grid-smoke/adapt_curve.csv`
- `results/generated/m5-story-cloze-baseline-grid-smoke/adapt_cost.json`
- `results/generated/m5-story-cloze-baseline-grid-smoke/summary.csv`
- `results/generated/m5-story-cloze-baseline-grid-smoke/summary.svg`

当前已验证：

- `cell_count = 24`
- `variant_count = 10`
- `train_run_count = 12`
- `eval_run_count = 24`

## Current Smoke Signals

- qwen25:
  - `meta_prompting`: `0-shot=0.5`，`2-shot=0.75`
  - `prompt_tuning`: `2-shot 0-step=0.75`，`2-shot 4-step=1.0`
  - `lora`: `2-shot 0-step=0.75`，`2-shot 4-step=1.0`
- qwen3:
  - `prompting / meta_prompting`: 当前 `0-shot` 与 `2-shot` 都是 `1.0 / 0.75` 这一量级，不形成明显 few-shot 差异
  - `adapter`: 当前 `0-step` 就已达到 `1.0`，`4-step` 不再提升

这些数字仍然只是 stub-backbone contract smoke，不是论文结果。它们的意义是：

- baseline 家族现在不再只是孤立单点 smoke
- 仓库已经具备“在单个 suite 内循环 shot/step 网格并产出 `adapt_curve.csv`”的最小能力
