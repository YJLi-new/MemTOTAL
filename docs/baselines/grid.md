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
- grid runner 现已支持 `grid.imports`，用于把外部 baseline 的既有评测点导入同一条 `adapt_curve.csv`
- grid runner 现已支持 `grid.reuse_existing_runs`，可在同一输出目录上复用已有 `train/eval` 产物，避免只改汇总配置时把整套 grid 重跑一遍
- 当前已验证导入 `MemGen` 的 `story_cloze` `Qwen2.5-1.5B-Instruct` `0-shot / 0-step` 外部评测点

## Entry Points

- config: `configs/exp/m5_story_cloze_baseline_grid_smoke.yaml`
- script: `scripts/run_story_cloze_baseline_grid.sh`
- module: `python -m memtotal.baselines.grid_runner`
- import variant config: `configs/exp/m5_story_cloze_baseline_grid_with_memgen_smoke.yaml`
- import variant script: `scripts/run_story_cloze_baseline_grid_with_memgen.sh`
- protocol-smoke config: `configs/exp/m5_story_cloze_baseline_grid_protocol_smoke.yaml`
- protocol-smoke script: `scripts/run_story_cloze_baseline_grid_protocol_smoke.sh`

## Verified Commands

```bash
./scripts/run_story_cloze_baseline_grid.sh 991 results/generated/m5-story-cloze-baseline-grid-smoke
./scripts/run_story_cloze_baseline_grid.sh 992 results/generated/m5-story-cloze-baseline-grid-smoke-dryrun --dry-run
./scripts/run_story_cloze_baseline_grid_with_memgen.sh 993 results/generated/m5-story-cloze-baseline-grid-with-memgen-smoke
python -m memtotal.tasks.setup_data --benchmarks story_cloze --max_examples 8 --seed 701 --output_root data/benchmarks/materialized --manifest_root data/benchmarks/manifests --summary_path data/benchmarks/source_summary.json
./scripts/run_story_cloze_baseline_grid_protocol_smoke.sh 997 results/generated/m5-story-cloze-baseline-grid-protocol-smoke
```

## Verified Outputs

- `results/generated/m5-story-cloze-baseline-grid-smoke/adapt_curve.csv`
- `results/generated/m5-story-cloze-baseline-grid-smoke/adapt_cost.json`
- `results/generated/m5-story-cloze-baseline-grid-smoke/summary.csv`
- `results/generated/m5-story-cloze-baseline-grid-smoke/summary.svg`
- `results/generated/m5-story-cloze-baseline-grid-with-memgen-smoke/adapt_curve.csv`
- `results/generated/m5-story-cloze-baseline-grid-with-memgen-smoke/adapt_cost.json`
- `results/generated/m5-story-cloze-baseline-grid-with-memgen-smoke/summary.csv`
- `results/generated/m5-story-cloze-baseline-grid-with-memgen-smoke/summary.svg`
- `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/adapt_curve.csv`
- `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/adapt_cost.json`
- `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/summary.csv`
- `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/summary.svg`
- `data/benchmarks/materialized/story_cloze/eval-real-smoke8.jsonl`

当前已验证：

- `cell_count = 24`
- `variant_count = 10`
- `train_run_count = 12`
- `eval_run_count = 24`
- `imported_eval_count = 1`
- protocol-smoke:
  - `shots = {0, 1, 2, 4}`
  - `steps = {0, 1, 3, 5}`
  - `cell_count = 76`
  - `train_run_count = 52`
  - `eval_run_count = 76`
  - `imported_eval_count = 1`
  - 重跑复用验证：`train_run_count = 0`、`eval_run_count = 0`、`reused_train_run_count = 52`、`reused_eval_run_count = 76`

## Current Smoke Signals

- qwen25:
  - `meta_prompting`: `0-shot=0.5`，`2-shot=0.75`
  - `prompt_tuning`: `2-shot 0-step=0.75`，`2-shot 4-step=1.0`
  - `lora`: `2-shot 0-step=0.75`，`2-shot 4-step=1.0`
- qwen3:
  - `prompting / meta_prompting`: 当前 `0-shot` 与 `2-shot` 都是 `1.0 / 0.75` 这一量级，不形成明显 few-shot 差异
  - `adapter`: 当前 `0-step` 就已达到 `1.0`，`4-step` 不再提升
- imported external point:
  - `MemGen / Qwen2.5-1.5B-Instruct / 0-shot / 0-step`: `compute_reward = 0.75`
- protocol-smoke:
  - `qwen25 / vanilla`: `0-shot=0.625`，`1-shot=0.75`
  - `qwen25 / meta_prompting`: `0-shot=0.5`，`4-shot=0.625`
  - `qwen3 / prompt_tuning`: `0-shot=0.5`，`4-shot 5-step=0.75`
  - `qwen3 / lora`: `0-shot=0.5`，`4-shot 5-step=0.75`

这些数字仍然只是 stub-backbone contract smoke，不是论文结果。它们的意义是：

- baseline 家族现在不再只是孤立单点 smoke
- 仓库已经具备“在单个 suite 内循环 shot/step 网格并产出 `adapt_curve.csv`”的最小能力
- 统一 grid 现在也能把外部 baseline 点导入同一张曲线，而不需要手工抄数
- materialize 层现在不会再因为不同 `max_examples` 覆盖同一个 real-smoke 文件，`smoke4` 和 `smoke8` 可以并存
- 只改导入点或汇总配置时，grid suite 现在可以直接复用已有 run，避免重复计算
