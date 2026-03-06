# Prompt Baselines

本文件记录 `M5` 当前已接入的最小 prompt baseline family。

## Scope

- family: `prompting`
- modes:
  - `vanilla`
  - `cot`
- 统一入口: `python -m eval --config ...`
- 当前目标: 先把 `Vanilla / CoT` 接入统一 `TaskEvaluator / metrics.json / predictions.jsonl / summary.csv`

## Config Contract

```yaml
baseline:
  family: prompting
  mode: vanilla | cot
  cot_suffix: Think step by step, then give the final answer.
```

说明：
- `vanilla` 直接使用 task prompt
- `cot` 会在 prompt 末尾追加 `cot_suffix`
- 当前 prompt baseline 不走 `MemoryRuntime`，因此不会写入 memory-specific gating / injection 信息
- 但它会写入：
  - `metrics.json.baseline_family`
  - `metrics.json.baseline_mode`
  - `predictions.jsonl[].baseline_prompt`
  - `predictions.jsonl[].candidate_scores`（仅多选/分类任务）

## Verified Smoke Configs

- `configs/exp/baseline_vanilla_gsm8k_qwen25_smoke.yaml`
- `configs/exp/baseline_cot_gsm8k_qwen25_smoke.yaml`
- `configs/exp/baseline_vanilla_story_cloze_qwen25_smoke.yaml`
- `configs/exp/baseline_cot_story_cloze_qwen25_smoke.yaml`
- `configs/exp/baseline_vanilla_gsm8k_qwen3_smoke.yaml`
- `configs/exp/baseline_cot_gsm8k_qwen3_smoke.yaml`
- `configs/exp/baseline_vanilla_story_cloze_qwen3_smoke.yaml`
- `configs/exp/baseline_cot_story_cloze_qwen3_smoke.yaml`

## Verified Commands

```bash
python -m eval --config configs/exp/baseline_vanilla_gsm8k_qwen25_smoke.yaml --seed 811 --output_dir runs/verify/m5-prompt-baseline-smoke/vanilla-gsm8k
python -m eval --config configs/exp/baseline_cot_gsm8k_qwen25_smoke.yaml --seed 811 --output_dir runs/verify/m5-prompt-baseline-smoke/cot-gsm8k
python -m eval --config configs/exp/baseline_vanilla_story_cloze_qwen25_smoke.yaml --seed 811 --output_dir runs/verify/m5-prompt-baseline-smoke/vanilla-story-cloze
python -m eval --config configs/exp/baseline_cot_story_cloze_qwen25_smoke.yaml --seed 811 --output_dir runs/verify/m5-prompt-baseline-smoke/cot-story-cloze
python -m analysis --config configs/exp/baseline_vanilla_gsm8k_qwen25_smoke.yaml --seed 811 --output_dir results/generated/m5-prompt-baseline-smoke --input_root runs/verify/m5-prompt-baseline-smoke
python -m eval --config configs/exp/baseline_vanilla_gsm8k_qwen3_smoke.yaml --seed 821 --output_dir runs/verify/m5-prompt-baseline-smoke-qwen3/vanilla-gsm8k
python -m eval --config configs/exp/baseline_cot_gsm8k_qwen3_smoke.yaml --seed 821 --output_dir runs/verify/m5-prompt-baseline-smoke-qwen3/cot-gsm8k
python -m eval --config configs/exp/baseline_vanilla_story_cloze_qwen3_smoke.yaml --seed 821 --output_dir runs/verify/m5-prompt-baseline-smoke-qwen3/vanilla-story-cloze
python -m eval --config configs/exp/baseline_cot_story_cloze_qwen3_smoke.yaml --seed 821 --output_dir runs/verify/m5-prompt-baseline-smoke-qwen3/cot-story-cloze
python -m analysis --config configs/exp/baseline_vanilla_gsm8k_qwen3_smoke.yaml --seed 821 --output_dir results/generated/m5-prompt-baseline-smoke-qwen3 --input_root runs/verify/m5-prompt-baseline-smoke-qwen3
```

## Current Smoke Results

汇总路径：
- `results/generated/m5-prompt-baseline-smoke/summary.csv`
- `results/generated/m5-prompt-baseline-smoke-qwen3/summary.csv`

当前 qwen25 stub smoke：
- `vanilla-gsm8k`: `accuracy=0.0`
- `cot-gsm8k`: `accuracy=0.0`
- `vanilla-story-cloze`: `accuracy=1.0`
- `cot-story-cloze`: `accuracy=1.0`

当前 qwen3 stub smoke：
- `vanilla-gsm8k`: `accuracy=0.0`
- `cot-gsm8k`: `accuracy=0.0`
- `vanilla-story-cloze`: `accuracy=1.0`
- `cot-story-cloze`: `accuracy=1.0`

这些数字只说明 baseline harness 已接入统一评测链，不是论文结果。
