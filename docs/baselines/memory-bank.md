# Memory Bank Baseline

本文件记录 `M5 / P2` 当前最小 `memory_bank` baseline scaffold。

## Scope

- family: `memory_bank`
- mode: `episodic_bank`
- backbones:
  - `Qwen2.5-1.5B-Instruct`
  - `Qwen3-8B`
- 当前任务：
  - `story_cloze` real-source smoke

说明：

- 这条线完成的是“MemoryBank 风格外部记忆 scaffold 接入统一评测链”，不是 `MemoryBank / ExpeL / AWM` 论文级完整复现
- 当前实现是零训练路线：先选 support，再压成有限容量的结构化 memory entries，再用同一 backbone 做候选打分

## Runtime Contract

- 统一入口仍是 `python -m eval`
- 当前配置字段：
  - `baseline.family=memory_bank`
  - `baseline.mode=episodic_bank`
  - `baseline.support_examples=<shots>`
  - `baseline.memory_bank.selector in {overlap_then_recency, dense_stub}`
  - `baseline.memory_bank.eviction_policy in {topk, recency}`
  - `baseline.memory_bank.bank_capacity=<int>`
- 当前会额外写出：
  - `metrics.json.mean_memory_bank_entry_count`
  - `metrics.json.mean_memory_bank_selection_score`
  - `metrics.json.memory_bank_selector`
  - `metrics.json.memory_bank_eviction_policy`
  - `predictions.jsonl[].baseline_memory_bank_entries`

## Verified Commands

```bash
python -m eval --config configs/exp/baseline_memory_bank_story_cloze_qwen25_real_smoke.yaml --seed 1107 --output_dir runs/verify/baseline_memory_bank_story_cloze_qwen25_real_smoke
python -m eval --config configs/exp/baseline_memory_bank_story_cloze_qwen3_real_smoke.yaml --seed 1109 --output_dir runs/verify/baseline_memory_bank_story_cloze_qwen3_real_smoke
```

## Current Smoke Signals

- qwen25:
  - `accuracy = 0.75`
  - `mean_memory_bank_entry_count = 2.0`
  - `mean_memory_bank_selection_score = 0.27268316751629673`
- qwen3:
  - `accuracy = 0.25`
  - `mean_memory_bank_entry_count = 2.0`
  - `mean_memory_bank_selection_score = 0.27268316751629673`

这些分数仍然只是 stub-backbone contract smoke，不是论文结果。
