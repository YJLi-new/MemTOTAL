# LightThinker Baseline

本文件记录 `M5 / P1` 当前接入的最小 `LightThinker` scaffold。

## Scope

- family: `lightthinker`
- mode: `compress_then_answer`
- 当前只做最小两阶段 prompt：
  - `compress`
  - `answer`
- 当前 verified task:
  - `story_cloze` real-source smoke
- 当前 verified backbones:
  - `Qwen2.5-1.5B-Instruct`
  - `Qwen3-8B`

## Contract

- 入口仍是 `python -m eval --config ...`
- 当前不做训练：
  - `support_examples = 0`
  - `train_steps = 0`
  - `trainable_parameter_count = 0`
- 预算 scope:
  - `compressed_reasoning_prompt`

当前会额外写出：

- `metrics.json.mean_thought_sketch_tokens`
- `predictions.jsonl[].lightthinker_compression_prompt`
- `predictions.jsonl[].lightthinker_thought_sketch`

## Verified Commands

```bash
python -m eval --config configs/exp/baseline_lightthinker_story_cloze_qwen25_real_smoke.yaml --seed 1201 --output_dir runs/verify/baseline_lightthinker_story_cloze_qwen25_real_smoke
python -m eval --config configs/exp/baseline_lightthinker_story_cloze_qwen3_real_smoke.yaml --seed 1201 --output_dir runs/verify/baseline_lightthinker_story_cloze_qwen3_real_smoke
python -m analysis --config configs/exp/m5_baseline_budget_audit.yaml --seed 961 --output_dir results/generated/m5-baseline-budget-audit --input_root runs/verify
```

## Current Verified Output

- `runs/verify/baseline_lightthinker_story_cloze_qwen25_real_smoke/metrics.json`
- `runs/verify/baseline_lightthinker_story_cloze_qwen3_real_smoke/metrics.json`
- `results/generated/m5-baseline-budget-audit/summary.csv`

当前已验证结果：

- qwen25 real-source smoke:
  - `accuracy = 0.75`
  - `mean_thought_sketch_tokens = 16.0`
- qwen3 real-source smoke:
  - `accuracy = 0.5`
  - `mean_thought_sketch_tokens = 16.0`
- `baseline_budget_audit` 当前已把 `lightthinker` 纳入自动检查：
  - `rows_collected = 50`
  - `checks_pass_rate = 1.0`

说明：

- 这一步完成的是“把推理压缩路线接进统一 harness”
- 不是正式 `LightThinker` 论文级复现
- 当前结果仍然只是 stub-backbone contract smoke，不是论文主表结论
