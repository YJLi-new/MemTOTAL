# PLANv8 qwen34 Watch Progress Repair

## Purpose

The live qwen34 `PLANv8` watch surfaces were only counting final `metrics.json` files under the run root.

That made active long-running arms look stalled even when they were still healthy and producing:

- `snapshot_evals/step_*/metrics.json`
- `snapshot_metrics.live.jsonl`
- `train_trace.live.jsonl`

## Repair

Add a small watcher helper:

- `scripts/planv8_watch_progress.py`

And switch the qwen34 `V8-2` and `V8-3` queue watchers to log:

- final metrics count
- snapshot metrics count
- active arm count
- latest active arm id
- latest observed train step

This keeps unattended qwen34 monitoring aligned with the real state of the active pilot instead of the delayed final-artifact surface.
