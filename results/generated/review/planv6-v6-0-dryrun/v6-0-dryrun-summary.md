# PLANv6 V6-0 Dry-Run Summary

- phase: V6-0
- arm_alias: V6_probe_dryrun
- gradient_probe_enabled: True
- groupwise_grad_clip: True
- post_unfreeze_start_step: 2
- writer_task_probe_post_unfreeze: 121.01485443115234
- writer_task_total_cosine_post_unfreeze: 1.0
- writer_support_attention_entropy_step2: 0.9994951486587524
- support_item_effective_rank_step2: 2.5160889625549316
- writer_clip_fraction_tail_50: 0.5
- projector_clip_fraction_tail_50: 1.0
- receiver_lora_clip_fraction_tail_50: 1.0

## Proved Fields Present
- grad_probe_writer_task_only_norm: True
- grad_probe_support_encoder_task_only_norm: True
- grad_probe_projector_task_only_norm: True
- grad_probe_receiver_lora_task_only_norm: True
- was_grad_clipped_writer: True
- was_grad_clipped_projector: True
- was_grad_clipped_receiver_lora: True
- writer_support_attention_entropy_mean: True
- support_item_effective_rank: True
- support_item_pairwise_cosine_mean: True
