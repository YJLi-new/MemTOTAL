# Benchmark Data Sources

本文件记录 `M4` 当前已经固定下来的 benchmark 数据来源、访问方式、materialize 路径与许可备注。

说明：
- 这里的 `real smoke` 指“从真实上游数据源 materialize 出来的小子集”，不是手写 toy 样例。
- 若上游 Hugging Face metadata 没有公开结构化 license 字段，这里不会擅自补写，而是明确标注为“需核对上游卡片”。
- 当前统一 materialize 入口是：

```bash
./scripts/setup_benchmark_data.sh
```

默认输出：
- `data/benchmarks/materialized/<benchmark_id>/eval-real-smoke4.jsonl`
- `data/benchmarks/manifests/<benchmark_id>.json`
- `data/benchmarks/source_summary.json`

## Current Registry

| Benchmark | Upstream source | Access | Current materialize status | Local path | License note |
| --- | --- | --- | --- | --- | --- |
| `gsm8k` | `gsm8k` / `main` / `test` | public | auto | `data/benchmarks/materialized/gsm8k/eval-real-smoke4.jsonl` | HF metadata license field is blank; verify upstream card before redistribution |
| `gpqa` | `Idavidrein/gpqa` / `gpqa_diamond` / `train` | gated | auto | `data/benchmarks/materialized/gpqa/eval-real-smoke4.jsonl` | gated dataset; metadata license field is blank |
| `triviaqa` | `mandarjoshi/trivia_qa` / `rc.wikipedia.nocontext` / `validation` | public | auto | `data/benchmarks/materialized/triviaqa/eval-real-smoke4.jsonl` | HF metadata license field is blank; verify upstream card |
| `story_cloze` | `gimmaru/story_cloze-2016` / `test` | public | auto | `data/benchmarks/materialized/story_cloze/eval-real-smoke4.jsonl` | HF metadata license field is blank; verify upstream card |
| `kodcode` | `KodCode/KodCode-Light-RL-10K` / `train` | public | auto | `data/benchmarks/materialized/kodcode/eval-real-smoke4.jsonl` | HF metadata license field is blank; verify upstream card |
| `rocstories` | `hf://datasets/wza/roc_stories/ROCStories__spring2016.csv` | public | auto | `data/benchmarks/materialized/rocstories/eval-real-smoke4.jsonl` | CSV-backed dataset; verify upstream card manually |
| `fever` | pending | manual | pending | pending | source registration still missing |
| `alfworld` | environment assets + game files | manual | pending | pending | requires environment/game files, not just flat JSONL |

## Notes

- `gpqa` 当前依赖你已经完成的 `huggingface-cli login`。未登录时 materialize 会失败。
- `triviaqa` 当前 real smoke 选择 `validation`，不是因为协议锁死，而是因为它适合快速验证统一评测链；正式实验时仍需按协议确认 split。
- `rocstories` 由于当前 `datasets` 版本不再支持旧脚本式加载，仓库现在跟随 MemGen 直接走 `hf://` CSV 路径。
- `fever` 和 `alfworld` 目前只完成了 task contract smoke，真实源接入仍是后续 TODO。
