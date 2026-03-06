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
- `configs/exp/baseline_prompt_tuning_story_cloze_qwen3_smoke.yaml`
- `configs/exp/baseline_lora_story_cloze_qwen3_smoke.yaml`
- `configs/exp/baseline_prompt_tuning_story_cloze_qwen25_real_smoke.yaml`
- `configs/exp/baseline_lora_story_cloze_qwen25_real_smoke.yaml`
- `configs/exp/baseline_prompt_tuning_story_cloze_qwen3_real_smoke.yaml`
- `configs/exp/baseline_lora_story_cloze_qwen3_real_smoke.yaml`

## Verified Commands

```bash
python -m train --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir runs/verify/m5-adapter-baseline-smoke/prompt-tuning/train
python -m eval --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir runs/verify/m5-adapter-baseline-smoke/prompt-tuning/eval --checkpoint runs/verify/m5-adapter-baseline-smoke/prompt-tuning/train/checkpoint.pt
python -m train --config configs/exp/baseline_lora_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir runs/verify/m5-adapter-baseline-smoke/lora/train
python -m eval --config configs/exp/baseline_lora_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir runs/verify/m5-adapter-baseline-smoke/lora/eval --checkpoint runs/verify/m5-adapter-baseline-smoke/lora/train/checkpoint.pt
python -m analysis --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_smoke.yaml --seed 911 --output_dir results/generated/m5-adapter-baseline-smoke --input_root runs/verify/m5-adapter-baseline-smoke
python -m train --config configs/exp/baseline_prompt_tuning_story_cloze_qwen3_smoke.yaml --seed 921 --output_dir runs/verify/m5-adapter-baseline-smoke-qwen3/prompt-tuning/train
python -m eval --config configs/exp/baseline_prompt_tuning_story_cloze_qwen3_smoke.yaml --seed 921 --output_dir runs/verify/m5-adapter-baseline-smoke-qwen3/prompt-tuning/eval --checkpoint runs/verify/m5-adapter-baseline-smoke-qwen3/prompt-tuning/train/checkpoint.pt
python -m train --config configs/exp/baseline_lora_story_cloze_qwen3_smoke.yaml --seed 921 --output_dir runs/verify/m5-adapter-baseline-smoke-qwen3/lora/train
python -m eval --config configs/exp/baseline_lora_story_cloze_qwen3_smoke.yaml --seed 921 --output_dir runs/verify/m5-adapter-baseline-smoke-qwen3/lora/eval --checkpoint runs/verify/m5-adapter-baseline-smoke-qwen3/lora/train/checkpoint.pt
python -m analysis --config configs/exp/baseline_prompt_tuning_story_cloze_qwen3_smoke.yaml --seed 921 --output_dir results/generated/m5-adapter-baseline-smoke-qwen3 --input_root runs/verify/m5-adapter-baseline-smoke-qwen3
python -m train --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_real_smoke.yaml --seed 931 --output_dir runs/verify/m5-adapter-baseline-real-smoke/qwen25-prompt-tuning/train
python -m eval --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_real_smoke.yaml --seed 931 --output_dir runs/verify/m5-adapter-baseline-real-smoke/qwen25-prompt-tuning/eval --checkpoint runs/verify/m5-adapter-baseline-real-smoke/qwen25-prompt-tuning/train/checkpoint.pt
python -m train --config configs/exp/baseline_lora_story_cloze_qwen25_real_smoke.yaml --seed 931 --output_dir runs/verify/m5-adapter-baseline-real-smoke/qwen25-lora/train
python -m eval --config configs/exp/baseline_lora_story_cloze_qwen25_real_smoke.yaml --seed 931 --output_dir runs/verify/m5-adapter-baseline-real-smoke/qwen25-lora/eval --checkpoint runs/verify/m5-adapter-baseline-real-smoke/qwen25-lora/train/checkpoint.pt
python -m train --config configs/exp/baseline_prompt_tuning_story_cloze_qwen3_real_smoke.yaml --seed 931 --output_dir runs/verify/m5-adapter-baseline-real-smoke/qwen3-prompt-tuning/train
python -m eval --config configs/exp/baseline_prompt_tuning_story_cloze_qwen3_real_smoke.yaml --seed 931 --output_dir runs/verify/m5-adapter-baseline-real-smoke/qwen3-prompt-tuning/eval --checkpoint runs/verify/m5-adapter-baseline-real-smoke/qwen3-prompt-tuning/train/checkpoint.pt
python -m train --config configs/exp/baseline_lora_story_cloze_qwen3_real_smoke.yaml --seed 931 --output_dir runs/verify/m5-adapter-baseline-real-smoke/qwen3-lora/train
python -m eval --config configs/exp/baseline_lora_story_cloze_qwen3_real_smoke.yaml --seed 931 --output_dir runs/verify/m5-adapter-baseline-real-smoke/qwen3-lora/eval --checkpoint runs/verify/m5-adapter-baseline-real-smoke/qwen3-lora/train/checkpoint.pt
python -m analysis --config configs/exp/baseline_prompt_tuning_story_cloze_qwen25_real_smoke.yaml --seed 931 --output_dir results/generated/m5-adapter-baseline-real-smoke --input_root runs/verify/m5-adapter-baseline-real-smoke
```

## Current Smoke Results

汇总路径：
- `results/generated/m5-adapter-baseline-smoke/summary.csv`
- `results/generated/m5-adapter-baseline-smoke-qwen3/summary.csv`
- `results/generated/m5-adapter-baseline-real-smoke/summary.csv`

当前 qwen25 stub smoke：
- `prompt_tuning/train`: `final_loss=0.31129494309425354`, `trainable_parameter_count=256`
- `prompt_tuning/eval`: `accuracy=1.0`
- `lora/train`: `final_loss=0.2841453552246094`, `trainable_parameter_count=512`
- `lora/eval`: `accuracy=1.0`

当前 qwen3 stub smoke：
- `prompt_tuning/train`: `final_loss=0.40971383452415466`, `trainable_parameter_count=256`
- `prompt_tuning/eval`: `accuracy=0.5`
- `lora/train`: `final_loss=0.36231037974357605`, `trainable_parameter_count=512`
- `lora/eval`: `accuracy=0.5`

当前 real-source `story_cloze` smoke：
- qwen25 `prompt_tuning/train`: `final_loss=0.4601891338825226`
- qwen25 `prompt_tuning/eval`: `accuracy=0.75`
- qwen25 `lora/train`: `final_loss=0.6477532982826233`
- qwen25 `lora/eval`: `accuracy=0.75`
- qwen3 `prompt_tuning/train`: `final_loss=0.5017770528793335`
- qwen3 `prompt_tuning/eval`: `accuracy=1.0`
- qwen3 `lora/train`: `final_loss=0.544849157333374`
- qwen3 `lora/eval`: `accuracy=1.0`

这些数字只说明 adapter baseline harness 已接入统一训练/评测链，不是论文结果。
