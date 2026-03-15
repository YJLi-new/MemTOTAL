# PLANv8 V8-7 Comparator Harness For Qwen3-4B

## Scope

This milestone adds the governed `V8-7` comparator harness for the qwen34 branch.

It wires three comparator surfaces:

- `m0_nomemory_qwen34`: imported `V8-0` qwen34 floor reference
- `m1_text_rag_qwen34`: live lexical text-RAG comparator on the same PLANv8 split
- `m2_memgen_qwen34`: benchmark-native MemGen mini-eval context on `gsm8k` and `triviaqa`

The historical `m3_legacy_planv7_qwen25` comparator is imported from the existing multiseed confirmation summary, and `m4_best_v8_qwen34` is imported from the winning `V8-6` arm.

## Important Notes

- The `m1_text_rag_qwen34` comparator is the apples-to-apples external-memory baseline and is the gating comparator for `V8-8`.
- The `m2_memgen_qwen34` comparator is explicitly contextual only. It runs on the upstream MemGen benchmark-native mini-eval path, so it is reported but not used as the `V8-8` gate.
- The qwen34 RAG path needed a small runtime fix so retrieval baselines propagate `use_chat_template` and `chat_template_enable_thinking` into `BackboneWrapper`.

## Governed Outputs

- `scripts/planv8_v8_7_config.py`
- `scripts/planv8_v8_7_retrieval_eval.py`
- `scripts/run_planv8_v8_7_comparators.sh`
- `scripts/run_planv8_v8_7_comparators_qwen34.sh`
- `scripts/queue_planv8_qwen34_v8_7_after_v8_6.sh`
- `scripts/update_planv8_v8_7_summary.py`

## Promotion Rule

`V8-7` opens `V8-8` only if the selected qwen34 `V8-6` winner:

- beats the qwen34 no-memory floor on at least one primary task,
- is non-regressive on the other primary task,
- beats the qwen34 text-RAG comparator on at least one primary task,
- is non-regressive on the other primary task,
- and still shows a live route on at least one primary task.
