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
| `math` | `EleutherAI/hendrycks_math` / `{algebra, geometry, number_theory, precalculus}` / `test` | public | auto | `data/benchmarks/materialized/math/eval-real-smoke4.jsonl` | MIT (from dataset card README metadata) |
| `gpqa` | `Idavidrein/gpqa` / `gpqa_diamond` / `train` | gated | auto | `data/benchmarks/materialized/gpqa/eval-real-smoke4.jsonl` | gated dataset; metadata license field is blank |
| `triviaqa` | `mandarjoshi/trivia_qa` / `rc.wikipedia.nocontext` / `validation` | public | auto | `data/benchmarks/materialized/triviaqa/eval-real-smoke4.jsonl` | HF metadata license field is blank; verify upstream card |
| `story_cloze` | `gimmaru/story_cloze-2016` / `test` | public | auto | `data/benchmarks/materialized/story_cloze/eval-real-smoke4.jsonl` | HF metadata license field is blank; verify upstream card |
| `narrativeqa` | `deepmind/narrativeqa` / `validation` / `summary_only` | public | auto | `data/benchmarks/materialized/narrativeqa/eval-real-smoke4.jsonl` | Apache-2.0 (from official dataset card metadata) |
| `kodcode` | `KodCode/KodCode-Light-RL-10K` / `train` | public | auto | `data/benchmarks/materialized/kodcode/eval-real-smoke4.jsonl` | HF metadata license field is blank; verify upstream card |
| `rocstories` | `hf://datasets/wza/roc_stories/ROCStories__spring2016.csv` | public | auto | `data/benchmarks/materialized/rocstories/eval-real-smoke4.jsonl` | CSV-backed dataset; verify upstream card manually |
| `fever` | `Dzeniks/fever_3way` / `validation` | public | auto | `data/benchmarks/materialized/fever/eval-real-smoke4.jsonl` | MIT (from dataset card README metadata) |
| `alfworld` | official ALFWorld TextWorld release assets / `valid_seen` | public | auto | `data/benchmarks/materialized/alfworld/eval-real-smoke4.jsonl` | MIT (from the official ALFWorld GitHub repository) |
| `memoryagentbench` | `ai-hyz/MemoryAgentBench` / representative smoke rows from `Accurate_Retrieval + Test_Time_Learning + Long_Range_Understanding + Conflict_Resolution` | public | auto | `data/benchmarks/materialized/memoryagentbench/eval-real-smoke4.jsonl` | MIT (from the official dataset card metadata) |

## Notes

- `gpqa` 当前依赖你已经完成的 `huggingface-cli login`。未登录时 materialize 会失败。
- `triviaqa` 当前 real smoke 选择 `validation`，不是因为协议锁死，而是因为它适合快速验证统一评测链；正式实验时仍需按协议确认 split。
- `narrativeqa` 当前 real smoke 选择官方 `validation` split 的 `summary_only` 视图，而不是 full-story 版本；这样能先以较低成本把 Narrative 域的真实来源 QA 路线接进统一 harness。
- `rocstories` 由于当前 `datasets` 版本不再支持旧脚本式加载，仓库现在跟随 MemGen 直接走 `hf://` CSV 路径。
- `math` 当前 real smoke 是跨四个 config 的聚合子集，每个 config 取 1 个样本，用于验证多 config materialize 与统一 exact-match 评测链。
- `fever` 当前使用公开的 3-way 变体，并在 materialize 时把标签映射为 `SUPPORTS / REFUTES / NOT_ENOUGH_INFO`。
- `alfworld` 当前走官方 TextWorld 资产路径，外部资产会放在 `data/benchmarks/external/alfworld/`，并在 materialize 时真实执行一次 hand-coded expert transition，再导出 `eval-real-smoke4.jsonl`。
- `alfworld` 这轮打通的是 text-only TextWorld smoke，不是完整 THOR/视觉栈；后者若要跑，需要额外的 `ai2thor/cv2` 依赖和更重的执行环境。
- `memoryagentbench` 当前使用官方 Hugging Face 数据集里的 4 个代表 source：`ruler_qa1_197K`、`icl_trec_coarse_6600shot_balance`、`infbench_sum_eng_shots2`、`factconsolidation_mh_6k`，各取 1 个 query 组成 `eval-real-smoke4.jsonl`。
- `memoryagentbench` 的 current smoke 是“真实 source + 截断 context”的 scaffold：为了让本仓库现有 stub runtime 可运行，materialize 时会把 context 截断到 `512` tokens，并在 `data/benchmarks/manifests/memoryagentbench.json` 里显式记录这一点。它不是正式长上下文协议结果。
- `NarrativeQA` 当前的统一评测使用 `qa_f1` 代理口径，目的是先打通 real-source Narrative 域 smoke，而不是替代官方 full-story 正式结果。
- `scripts/run_real_benchmark_smoke_suite.sh` 现支持 `SKIP_SETUP_BENCHMARK_DATA=1`，用于在数据已 materialize 的情况下避免重复 setup 被 HF 缓存锁拖住。
