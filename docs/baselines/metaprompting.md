# MetaPrompting Baseline

本文件记录 `M5` 当前已接入的最小 `MetaPrompting` scaffold。

## Scope

- family: `meta_prompting`
- mode: `planner_critic`
- 当前目标: 先把 `MetaPrompting` 接进统一 `python -m eval` / `metrics.json` / `predictions.jsonl` / `summary.csv`
- 当前 smoke 范围: `story_cloze`

## Config Contract

```yaml
baseline:
  family: meta_prompting
  mode: planner_critic
  support_examples: 0  # optional; >0 means in-context few-shot demos
  planner_role: Planner: decompose the task into concise reasoning steps.
  solver_role: Solver: solve using the plan and inspect each option.
  critic_role: Critic: verify the draft and correct any mistakes.
  finalizer_role: Finalizer: provide only the final answer or option label.
```

说明：
- 当前实现仍复用 `src/memtotal/baselines/prompting.py`
- 只是把 prompt protocol 升级为 `Planner / Solver / Critic / Finalizer`
- 当前也支持 `support_examples > 0` 的 in-context few-shot demos
- 当前仍是单次前向 smoke scaffold，不是正式多-agent / 多轮 MetaPrompting 复现

## Verified Smoke Configs

- `configs/exp/baseline_metaprompting_story_cloze_qwen25_smoke.yaml`
- `configs/exp/baseline_metaprompting_story_cloze_qwen3_smoke.yaml`
- `configs/exp/baseline_metaprompting_story_cloze_qwen25_real_smoke.yaml`
- `configs/exp/baseline_metaprompting_story_cloze_qwen3_real_smoke.yaml`
- `configs/exp/baseline_metaprompting_story_cloze_qwen25_real_2shot.yaml`
- `configs/exp/baseline_metaprompting_story_cloze_qwen3_real_2shot.yaml`

## Verified Commands

```bash
python -m eval --config configs/exp/baseline_metaprompting_story_cloze_qwen25_smoke.yaml --seed 941 --output_dir runs/verify/m5-metaprompting-smoke/qwen25
python -m eval --config configs/exp/baseline_metaprompting_story_cloze_qwen3_smoke.yaml --seed 941 --output_dir runs/verify/m5-metaprompting-smoke/qwen3
python -m analysis --config configs/exp/baseline_metaprompting_story_cloze_qwen25_smoke.yaml --seed 941 --output_dir results/generated/m5-metaprompting-smoke --input_root runs/verify/m5-metaprompting-smoke
python -m eval --config configs/exp/baseline_metaprompting_story_cloze_qwen25_real_smoke.yaml --seed 951 --output_dir runs/verify/m5-metaprompting-real-smoke/qwen25
python -m eval --config configs/exp/baseline_metaprompting_story_cloze_qwen3_real_smoke.yaml --seed 951 --output_dir runs/verify/m5-metaprompting-real-smoke/qwen3
python -m analysis --config configs/exp/baseline_metaprompting_story_cloze_qwen25_real_smoke.yaml --seed 951 --output_dir results/generated/m5-metaprompting-real-smoke --input_root runs/verify/m5-metaprompting-real-smoke
python -m eval --config configs/exp/baseline_metaprompting_story_cloze_qwen25_real_2shot.yaml --seed 981 --output_dir runs/verify/m5-prompt-fewshot-real-smoke/qwen25-metaprompting-2shot
python -m eval --config configs/exp/baseline_metaprompting_story_cloze_qwen3_real_2shot.yaml --seed 981 --output_dir runs/verify/m5-prompt-fewshot-real-smoke/qwen3-metaprompting-2shot
python -m analysis --config configs/exp/baseline_metaprompting_story_cloze_qwen25_real_2shot.yaml --seed 981 --output_dir results/generated/m5-prompt-fewshot-real-smoke --input_root runs/verify/m5-prompt-fewshot-real-smoke
```

## Current Smoke Results

汇总路径：
- `results/generated/m5-metaprompting-smoke/summary.csv`
- `results/generated/m5-metaprompting-real-smoke/summary.csv`
- `results/generated/m5-prompt-fewshot-real-smoke/summary.csv`

当前 stub smoke：
- `qwen25`: `accuracy=0.0`
- `qwen3`: `accuracy=1.0`

当前 real-source `story_cloze` smoke：
- `qwen25`: `accuracy=0.75`
- `qwen3`: `accuracy=1.0`

当前 real-source `2-shot story_cloze` smoke：
- `qwen25`: `accuracy=0.75`
- `qwen3`: `accuracy=0.75`

这些数字只说明 `MetaPrompting` scaffold 已接入统一 baseline 评测链，不是论文结果。
