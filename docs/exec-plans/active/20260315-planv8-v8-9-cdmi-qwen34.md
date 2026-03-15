# PLANv8 V8-9 CDMI Closeout for Qwen3-4B

## Purpose

Add the governed qwen34 `V8-9` closeout phase required by [`PLANv8.md`](/root/mydir/MemTOTAL/PLANv8.md):

- promote the best confirmed qwen34 branch from `V8-8`
- train a joint `GSM8K + TriviaQA` memory route
- measure cross-domain interference and support leakage
- publish a paper-closeout summary instead of reopening new unguided sweeps

## Deliverables

- runtime support for pre-canonicalized mixed-task datasets in [registry.py](/root/mydir/MemTOTAL/src/memtotal/tasks/registry.py)
- config builder [planv8_v8_9_config.py](/root/mydir/MemTOTAL/scripts/planv8_v8_9_config.py)
- runner [run_planv8_v8_9_cdmi.sh](/root/mydir/MemTOTAL/scripts/run_planv8_v8_9_cdmi.sh)
- qwen34 wrapper [run_planv8_v8_9_cdmi_qwen34.sh](/root/mydir/MemTOTAL/scripts/run_planv8_v8_9_cdmi_qwen34.sh)
- qwen34 queue [queue_planv8_qwen34_v8_9_after_v8_8.sh](/root/mydir/MemTOTAL/scripts/queue_planv8_qwen34_v8_9_after_v8_8.sh)
- summary builder [update_planv8_v8_9_summary.py](/root/mydir/MemTOTAL/scripts/update_planv8_v8_9_summary.py)
- focused tests in [test_task_registry.py](/root/mydir/MemTOTAL/tests/test_task_registry.py), [test_planv8_v8_9_config.py](/root/mydir/MemTOTAL/tests/test_planv8_v8_9_config.py), and [test_planv8_v8_9_summary.py](/root/mydir/MemTOTAL/tests/test_planv8_v8_9_summary.py)

## Experiment Shape

`V8-9` is gated strictly on `V8-8` success and uses the `best_confirmed_variant_id` from `v8-8-summary.json`.

The qwen34 runner materializes prompt-rendered canonical datasets for:

- `gsm8k/support|train|eval`
- `triviaqa/support|train|eval`
- `joint/support|train`

It then runs six governed conditions:

- `c0_math_self`
- `c1_trivia_self`
- `c2_joint_math`
- `c3_joint_trivia`
- `c4_math_support_on_trivia`
- `c5_trivia_support_on_math`

The summary reports:

- joint-vs-self score deltas
- cross-support penalties
- negative transfer rate
- gate and attention shifts between math and trivia evaluation
- a compression leakage risk flag tied to the promoted bridge family

## Validation

Required local validation for this milestone:

```bash
python -m py_compile \
  src/memtotal/tasks/registry.py \
  scripts/planv8_v8_9_config.py \
  scripts/update_planv8_v8_9_summary.py

bash -n \
  scripts/run_planv8_v8_9_cdmi.sh \
  scripts/run_planv8_v8_9_cdmi_qwen34.sh \
  scripts/queue_planv8_qwen34_v8_9_after_v8_8.sh \
  scripts/publish_review_artifacts.sh \
  scripts/arm_planv8_qwen34_chain.sh

python -m unittest \
  tests.test_task_registry \
  tests.test_planv8_v8_9_config \
  tests.test_planv8_v8_9_summary -v

python -m unittest tests.test_repo_lints tests.test_repo_contract -v
```

## Queueing

Once pushed, the qwen34 unattended chain should gain:

- `planv8_q34_queue_v89`
- `planv8_v89_q34`
- `planv8_v89_q34_watch`
- `planv8_v89_q34_post`
- a fresh dedicated watcher `planv8_q34_v89_superwatch`

`V8-9` stays dormant until `/root/autodl-tmp/results/generated/planv8-v8-8-multiseed-confirmation-qwen34/v8-8-summary.json` recommends `open_v8_9_cdmi`.
