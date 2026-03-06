# Baseline Budget Audit

本文件记录 `M5` 当前的 baseline 预算治理约定。

## Scope

- 入口：`python -m analysis`
- analysis mode: `baseline_budget_audit`
- 当前纳入 audit 的 family：
  - `prompting`
  - `meta_prompting`
  - `adapter`
  - `rag`
- 当前先不把 `MemGen` 纳入 `M5` audit 通过条件，因为它仍走外部 repo / 外部 checkpoint 成本，不适合和当前仓库内的 few-shot adapter 预算直接混算。

## Budget Fields

所有纳入 `M5` 的 baseline run 现在都会尽量写出：

- `metrics.json.baseline_family`
- `metrics.json.baseline_mode`
- `metrics.json.support_examples`
- `metrics.json.train_steps`
- `metrics.json.trainable_parameter_count`
- `metrics.json.budget_scope`
- `metrics.json.budget_signature`

当前约定：

- `prompting / meta_prompting`
  - `support_examples = 0`
  - `train_steps = 0`
  - `trainable_parameter_count = 0`
- `rag`
  - `support_examples` 取配置中的真实检索 memory 条数
  - `train_steps = 0`
  - `trainable_parameter_count = 0`
- `adapter`
  - `support_examples` 取配置中的真实 support 数
  - `train_steps` 取当前 run 的真实训练步数
  - `trainable_parameter_count` 取当前 adapter 的可训练参数量

## Verified Command

```bash
./scripts/run_baseline_budget_audit.sh 961 results/generated/m5-baseline-budget-audit runs/verify
```

## Current Verified Output

- `results/generated/m5-baseline-budget-audit/summary.csv`
- `results/generated/m5-baseline-budget-audit/summary.svg`
- `results/generated/m5-baseline-budget-audit/baseline_budget_report.json`

最新已验证结果：

- `rows_collected = 48`
- `checks_pass_rate = 1.0`
- `issues_found = 0`

这说明当前仓库内已接入 `M5` audit 的 `prompting / meta_prompting / adapter / rag` baseline runs，预算字段和双 backbone 覆盖都能通过自动检查。
