# RAG Baseline

本文件记录 `M5 / P1` 当前接入的最小外部记忆 baseline。

## Scope

- family: `rag`
- mode: `retrieval_augmented`
- 当前 retriever:
  - `lexical_overlap`
  - `dense_stub`
- 当前 verified task:
  - `story_cloze` real-source smoke
- 当前 verified backbones:
  - `Qwen2.5-1.5B-Instruct`
  - `Qwen3-8B`

## Contract

- 入口仍是 `python -m eval --config ...`
- `baseline.support_examples` 表示检索后注入 prompt 的 memory 条数
- `train_steps = 0`
- `trainable_parameter_count = 0`
- `budget_scope = external_memory_prompt`

当前会额外写出：

- `metrics.json.baseline_retriever`
- `metrics.json.mean_support_retrieval_score`
- `predictions.jsonl[].baseline_support_ids`
- `predictions.jsonl[].baseline_support_scores`
- `predictions.jsonl[].baseline_retriever`

## Verified Commands

```bash
python -m eval --config configs/exp/baseline_rag_story_cloze_qwen25_real_smoke.yaml --seed 1101 --output_dir runs/verify/baseline_rag_story_cloze_qwen25_real_smoke
python -m eval --config configs/exp/baseline_rag_story_cloze_qwen3_real_smoke.yaml --seed 1101 --output_dir runs/verify/baseline_rag_story_cloze_qwen3_real_smoke
python -m memtotal.baselines.grid_runner --config configs/exp/m5_story_cloze_baseline_grid_protocol_smoke.yaml --seed 997 --output_dir results/generated/m5-story-cloze-baseline-grid-protocol-smoke
```

## Current Verified Output

- `runs/verify/baseline_rag_story_cloze_qwen25_real_smoke/metrics.json`
- `runs/verify/baseline_rag_story_cloze_qwen3_real_smoke/metrics.json`
- `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/adapt_curve.csv`
- `results/generated/m5-story-cloze-baseline-grid-protocol-smoke/adapt_cost.json`

当前已验证结果：

- qwen25 real-source smoke:
  - `accuracy = 1.0`
  - `mean_support_retrieval_score = 0.24647887323943662`
- qwen3 real-source smoke:
  - `accuracy = 0.5`
  - `mean_support_retrieval_score = 0.24647887323943662`
- protocol-smoke grid:
  - `variant_count = 12`
  - `cell_count = 84`
  - `eval_run_count = 8`
  - `reused_eval_run_count = 76`
  - `rag / qwen3` 当前 best cell 是 `1-shot / 0-step / 0.75`

这些数字仍然只是 stub-backbone contract smoke，不是论文主结果。
