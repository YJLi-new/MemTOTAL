# Golden Principles

- `docs/TODO_LIST.md` 是执行入口；不得绕开里程碑顺序自行扩题。
- `docs/MAIN_IDEA.md` 定义方法口径；不得把方法实现成另一个 MemGen。
- `docs/EXPERIMENTS_INFO.md` 定义实验与汇总口径；不得手工抄数生成主表。
- 当前代码层只支持 `Qwen2.5-1.5B-Instruct` 与 `Qwen3-8B`。
- 所有训练/评测/分析入口必须支持 `--config --seed --output_dir`。
- 任何重要结果都必须能从 `runs/**/metrics.json` 自动回溯到汇总 CSV。

