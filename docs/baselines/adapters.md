# Adapter Baselines

本文件记录 `M5` 当前已接入的最小 adapter baseline family。

## Scope

- family: `adapter`
- modes:
  - `prompt_tuning`
  - `lora`
- 当前目标: 先把 `Prompt Tuning / LoRA` 接成统一的 `train -> checkpoint -> eval -> summary`
- 当前 smoke 范围: `story_cloze` 多选任务

## Config Contract

```yaml
baseline:
  family: adapter
  mode: prompt_tuning | lora
  support_examples: 1
  prompt_tuning:
    prompt_tokens: 4
  lora:
    rank: 4
    alpha: 8.0
```

说明：
- `prompt_tuning` 当前通过可训练 `soft_prompts` 的均值偏移 prompt state
- `lora` 当前通过低秩残差 `prompt_state + scale * B(A(prompt_state))`
- 当前 adapter baseline 只支持 `multiple_choice / dataset_label_classification` smoke；不支持 exact-match 生成式任务

## Verified Smoke Configs

- `configs/exp/baseline_prompt_tuning_story_cloze_qwen25_smoke.yaml`
- `configs/exp/baseline_lora_story_cloze_qwen25_smoke.yaml`

## Verified Commands

```bash
python -m train --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir runs/verify/m5-adapter-baseline-smoke/prompt-tuning/train
python -m eval --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir runs/verify/m5-adapter-baseline-smoke/prompt-tuning/eval --checkpoint runs/verify/m5-adapter-baseline-smoke/prompt-tuning/train/checkpoint.pt
python -m train --config configs/exp/baseline_lora_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir runs/verify/m5-adapter-baseline-smoke/lora/train
python -m eval --config configs/exp/baseline_lora_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir runs/verify/m5-adapter-baseline-smoke/lora/eval --checkpoint runs/verify/m5-adapter-baseline-smoke/lora/train/checkpoint.pt
python -m analysis --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir results/generated/m5-adapter-baseline-smoke --input_root runs/verify/m5-adapter-baseline-smoke
```

## Current Smoke Results

汇总路径：
- `results/generated/m5-adapter-baseline-smoke/summary.csv`

当前 qwen25 stub smoke：
- `prompt_tuning/train`: `final_loss=0.31129494309425354`, `trainable_parameter_count=256`
- `prompt_tuning/eval`: `accuracy=1.0`
- `lora/train`: `final_loss=0.2841453552246094`, `trainable_parameter_count=512`
- `lora/eval`: `accuracy=1.0`

这些数字只说明 adapter baseline harness 已接入统一训练/评测链，不是论文结果。
