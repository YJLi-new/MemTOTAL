# MemGen Baseline Notes

## Scope

本仓库当前只支持以下两档 backbone 的 MemGen 接入：

- `Qwen2.5-1.5B-Instruct`
- `Qwen3-8B`

统一入口固定为：

```bash
./scripts/run_memgen.sh --config <configs/exp/*.yaml> --seed <int> --output_dir <runs/...>
```

## Task Matrix

| Task | Experiment role | Official config | Repo template | Status |
| --- | --- | --- | --- | --- |
| `gsm8k` | Math / main suite | `MemGen-master/configs/latent_memory/gsm8k.yaml` | `configs/exp/memgen_gsm8k_qwen25_smoke_eval.yaml`, `configs/exp/memgen_gsm8k_qwen25_eval.yaml`, `configs/exp/memgen_gsm8k_qwen3_eval.yaml` | real smoke passed, unified translation passed |
| `rocstories` | Narrative / CDMI | `MemGen-master/configs/latent_memory/rocstories.yaml` | `configs/exp/memgen_rocstories_qwen25_smoke_eval.yaml` | real smoke passed, unified translation passed |
| `story_cloze` | Narrative / CDMI | `MemGen-master/configs/latent_memory/story_cloze.yaml` | `configs/exp/memgen_story_cloze_qwen25_smoke_eval.yaml` | real smoke passed, unified translation passed |
| `gpqa` | Knowledge QA / main suite | `MemGen-master/configs/latent_memory/gpqa.yaml` | `configs/exp/memgen_gpqa_qwen25_smoke_eval.yaml` | real smoke passed, unified translation passed |
| `triviaqa` | Knowledge QA / main suite | `MemGen-master/configs/latent_memory/triviaqa.yaml` | `configs/exp/memgen_triviaqa_qwen25_smoke_eval.yaml` | real smoke passed, dynamic translation passed |
| `kodcode` | Code / main suite | `MemGen-master/configs/latent_memory/kodcode.yaml` | `configs/exp/memgen_kodcode_qwen25_smoke_eval.yaml` | real smoke passed, unified translation passed |
| `cosmosqa` | Extra narrative QA | `MemGen-master/configs/latent_memory/cosmosqa.yaml` | pending | optional, not on paper-critical path yet |

## Fixed Templates And Seeds

当前已经验证过的模板与种子：

- `configs/exp/memgen_gsm8k_qwen25_smoke_eval.yaml` with `--seed 23`
- `configs/exp/memgen_rocstories_qwen25_smoke_eval.yaml` with `--seed 31`
- `configs/exp/memgen_story_cloze_qwen25_smoke_eval.yaml` with `--seed 41`
- `configs/exp/memgen_triviaqa_qwen25_smoke_eval.yaml` with `--seed 62`
- `configs/exp/memgen_gpqa_qwen25_smoke_eval.yaml` with `--seed 53`
- `configs/exp/memgen_kodcode_qwen25_smoke_eval.yaml` with `--seed 71`
- `configs/exp/memgen_gsm8k_qwen25_smoke_eval_trigger_on.yaml` with `--seed 81` for trigger-path smoke validation
- `configs/exp/memgen_gsm8k_qwen25_eval.yaml` with `--seed 11` for dry-run launch-plan validation
- `configs/exp/memgen_gsm8k_qwen3_eval.yaml` with `--seed 17` for dry-run launch-plan validation
- `configs/exp/memgen_gpqa_qwen25_smoke_eval.yaml` with `--seed 52` verifies gated-dataset preflight when HF auth is absent

统一分析入口已能直接读取这些 translated run 的 `metrics.json` / `predictions.jsonl`，并在 `summary.csv` 里汇总 `compute_reward`。其中动态环境任务会从 `conversations.txt` 翻译到统一 `predictions.jsonl`。

当前里程碑的可视化快照已保存为：

- `docs/assets/milestones/20260306-m1-memgen-summary.svg`

## Harness Rules

- smoke 配置默认锁定 `model.attn_implementation=sdpa`，避免当前环境缺少 `flash_attention_2` 时直接失败
- smoke 配置默认锁定 `max_prompt_aug_num=1` 与 `max_inference_aug_num=1`，先验证可运行性，不提前做大 sweep
- `gpqa` 在真正启动官方进程前会先做 Hugging Face 认证 preflight；未登录时直接在 adapter 层失败并写明原因
- baseline 配置层现在显式暴露以下字段：
  - `trigger_active`
  - `insertion_profile`
  - `requires_trained_checkpoint`
  - `load_model_path`
- 真实 run 输出会同时保留两层产物：
  - 统一层：`metrics.json`、`predictions.jsonl`
  - 官方原始层：`memgen_raw/answer.json`、`memgen_raw/launcher.json`、`memgen_raw/log.txt`
- `run_memgen.py` 会把官方静态任务 `answer.json` 或动态任务 `conversations.txt` 翻译成统一 `predictions.jsonl`，并把 `compute_reward`、`num_predictions`、`wall_time_sec` 写回统一 `metrics.json`
- 当前已验证 `model.trigger.active=True` 的最小 smoke 路径，配置见 `configs/exp/memgen_gsm8k_qwen25_smoke_eval_trigger_on.yaml`
- 正式可比的 trigger baseline 模板见 `configs/exp/memgen_gsm8k_qwen25_eval_trigger_trained_template.yaml`；若 checkpoint 缺失，adapter 会在 preflight 阶段直接报错
- adapter 运行 MemGen 时默认注入 `TOKENIZERS_PARALLELISM=false`，用脚本规则消除 `kodcode` 一类 code-eval task 的稳定 fork 警告

## Known Pitfalls

- 本地模型路径会让官方 `build_working_dir()` 误生成 `root/...` 目录；已在 `MemGen-master/main.py` 修正为使用 `Path(model_name).name`
- 直接使用 `load_model_path=<file>.safetensors` 与当前 transformers 版本存在兼容漂移；当前 smoke 路径统一用 `load_model_path: null`
- Narrative 任务的数据切分和子集裁剪必须通过 builder 配置项控制，不能手工改官方数据文件；对应规则已固化进 `BaseBuilder.limit_split()`
- `trigger on` 版本尚未稳定验证；当前仓库内已验证的是 `trigger.active=False` 路径
- `gpqa` 使用的 `Idavidrein/gpqa` 当前是 gated dataset；没有 Hugging Face 认证时只能验证到 launch / config 层，不能完成真实 smoke
- 现在这类 `gpqa` 认证缺失会由 adapter preflight 直接报错，不再先启动官方进程再失败
- `triviaqa` 属于动态环境任务，官方输出不是 `answer.json` 而是 `conversations.txt`；统一 adapter 已补动态翻译分支
- `kodcode` 评测会在 reward 计算里 fork 子进程执行测试代码；该坑已升级为脚本规则，adapter 默认设置 `TOKENIZERS_PARALLELISM=false`
- `trigger.active=True` 在 `load_model_path=null` 时也能跑通，但这只是 trigger-path smoke，不代表训练后的正式 MemGen trigger baseline；正式可比版本仍需要对齐触发器权重来源
