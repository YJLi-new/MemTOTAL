# Writer Deep-Prefix JointPEFT Live Monitor

- updated_at_utc: 2026-03-09T10:22:35+00:00
- session_name: writer_jointpeft_train
- session_alive: False
- base_seed: 60917
- run_root: /root/autodl-tmp/runs/verify/writer-deep-prefix-jointpeft-qwen25
- monitor_output: /root/mydir/MemTOTAL/docs/exec-plans/active/20260309-writer-deep-prefix-jointpeft-monitor.md
- log_path: /root/autodl-tmp/runs/verify/writer-deep-prefix-jointpeft-qwen25/tmux-session.log
- completed_suites: 7/7

## source_stub_health_gsm8k
- state: completed
- arm_dir: /root/autodl-tmp/runs/verify/writer-deep-prefix-jointpeft-qwen25/gsm8k-source-stub-health/pilot-I-source-stub-health
- steps_recorded: 32
- latest_step: 32
- latest_loss: 8.554368
- latest_delta_answer_logprob: 0.000000
- latest_source_grad: 0.385202
- latest_writer_grad: 0.000000
- latest_projector_grad: 104.858205
- latest_receiver_grad: 70.205959
- latest_total_grad_norm_pre_clip: 126.165298
- latest_was_grad_clipped: True
- recent_loss_median: 6.674272
- recent_delta_answer_logprob_median: 0.000000
- recent_source_grad_median: 0.581035
- recent_writer_grad_median: 0.000000
- recent_projector_grad_median: 164.569908
- recent_receiver_grad_median: 67.527556
- recent_clipped_steps: 8
- recent_loss_trace: [6.300146, 4.027628, 7.048397, 11.003279, 4.502638, 5.900035, 15.112001, 8.554368]
- recent_source_grad_trace: [0.317822, 0.744227, 0.515741, 1.005443, 0.857973, 0.64633, 0.34054, 0.385202]
- recent_writer_grad_trace: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- recent_projector_grad_trace: [112.576816, 212.98797, 151.609264, 264.81627, 249.19739, 177.530551, 109.890566, 104.858205]
- recent_receiver_grad_trace: [64.849152, 72.211443, 63.15737, 125.503499, 75.292462, 59.867174, 50.587454, 70.205959]
- latest_optimizer_lrs: {'prefix_projector': 0.00015, 'receiver_lora': 5e-05, 'source_stub': 0.0001, 'writer_base': 0.0002}
- snapshot_steps_seen: [0, 8, 16, 32]
- latest_snapshot_step: 32
- latest_snapshot_accuracy: 0.000000
- latest_snapshot_macro_f1: 0.000000
- latest_snapshot_margin: -6.574059
- latest_snapshot_prefix_l2: 31.999990

## writer_gsm8k
- state: completed
- arm_dir: /root/autodl-tmp/runs/verify/writer-deep-prefix-jointpeft-qwen25/gsm8k-writer/pilot-I-writer-direct
- steps_recorded: 500
- latest_step: 500
- latest_loss: 5.577970
- latest_delta_answer_logprob: 0.000000
- latest_source_grad: 0.000000
- latest_writer_grad: 0.771282
- latest_projector_grad: 36.938889
- latest_receiver_grad: 12.956341
- latest_total_grad_norm_pre_clip: 39.159405
- latest_was_grad_clipped: True
- recent_loss_median: 3.852555
- recent_delta_answer_logprob_median: 0.000000
- recent_source_grad_median: 0.000000
- recent_writer_grad_median: 1.268602
- recent_projector_grad_median: 51.218046
- recent_receiver_grad_median: 17.084538
- recent_clipped_steps: 8
- recent_loss_trace: [1.593995, 3.142078, 6.499503, 4.427462, 2.68621, 7.308759, 3.277649, 5.57797]
- recent_source_grad_trace: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- recent_writer_grad_trace: [0.394084, 1.138882, 1.580997, 1.398323, 5.102468, 3.113664, 0.681796, 0.771282]
- recent_projector_grad_trace: [23.535184, 48.601571, 62.586368, 53.834521, 131.365038, 156.389443, 42.437994, 36.938889]
- recent_receiver_grad_trace: [9.321353, 17.60849, 19.697542, 16.560585, 54.824343, 51.328506, 12.436345, 12.956341]
- latest_optimizer_lrs: {'prefix_projector': 0.00015, 'receiver_lora': 5e-05, 'writer_base': 0.0001}
- snapshot_steps_seen: [0, 10, 25, 50, 100, 200, 350, 500]
- latest_snapshot_step: 500
- latest_snapshot_accuracy: 0.000000
- latest_snapshot_macro_f1: 0.000000
- latest_snapshot_margin: -4.294594
- latest_snapshot_prefix_l2: 203.466442

## writer_narrativeqa
- state: completed
- arm_dir: /root/autodl-tmp/runs/verify/writer-deep-prefix-jointpeft-qwen25/narrativeqa-writer/pilot-I-writer-direct
- steps_recorded: 500
- latest_step: 500
- latest_loss: 21.425985
- latest_delta_answer_logprob: 0.000000
- latest_source_grad: 0.000000
- latest_writer_grad: 1.091011
- latest_projector_grad: 51.831469
- latest_receiver_grad: 31.152167
- latest_total_grad_norm_pre_clip: 60.476948
- latest_was_grad_clipped: True
- recent_loss_median: 11.952153
- recent_delta_answer_logprob_median: 0.000000
- recent_source_grad_median: 0.000000
- recent_writer_grad_median: 0.778111
- recent_projector_grad_median: 35.929405
- recent_receiver_grad_median: 22.720313
- recent_clipped_steps: 7
- recent_loss_trace: [39.753727, 2.47832, 0.167821, 21.765287, 1.956231, 56.345333, 2.102262, 21.425985]
- recent_source_grad_trace: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- recent_writer_grad_trace: [1.295056, 0.217955, 0.002489, 1.126765, 0.221785, 1.793701, 0.465212, 1.091011]
- recent_projector_grad_trace: [111.201877, 20.027341, 0.219035, 68.237812, 12.110283, 92.468241, 18.679558, 51.831469]
- recent_receiver_grad_trace: [59.343354, 13.452752, 0.16792, 39.97683, 9.079422, 65.703363, 14.288458, 31.152167]
- latest_optimizer_lrs: {'prefix_projector': 0.00015, 'receiver_lora': 5e-05, 'writer_base': 0.0001}
- snapshot_steps_seen: [0, 10, 25, 50, 100, 200, 350, 500]
- latest_snapshot_step: 500
- latest_snapshot_accuracy: 0.057603
- latest_snapshot_macro_f1: 0.057603
- latest_snapshot_margin: -9.838728
- latest_snapshot_prefix_l2: 231.833047

## writer_fever
- state: completed
- arm_dir: /root/autodl-tmp/runs/verify/writer-deep-prefix-jointpeft-qwen25/fever-writer/pilot-I-writer-direct
- steps_recorded: 500
- latest_step: 500
- latest_loss: 7.158056
- latest_delta_answer_logprob: 0.000000
- latest_source_grad: 0.000000
- latest_writer_grad: 5.395142
- latest_projector_grad: 228.210251
- latest_receiver_grad: 80.126435
- latest_total_grad_norm_pre_clip: 241.892380
- latest_was_grad_clipped: True
- recent_loss_median: 0.150392
- recent_delta_answer_logprob_median: 0.000000
- recent_source_grad_median: 0.000000
- recent_writer_grad_median: 0.001460
- recent_projector_grad_median: 0.078085
- recent_receiver_grad_median: 0.022688
- recent_clipped_steps: 3
- recent_loss_trace: [0.236665, 0.150345, 0.150199, 5.608229, 0.150129, 0.150243, 0.150439, 7.158056]
- recent_source_grad_trace: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- recent_writer_grad_trace: [0.40231, 0.001366, 0.000469, 5.242057, 0.000435, 0.000771, 0.001554, 5.395142]
- recent_projector_grad_trace: [14.935112, 0.083276, 0.024516, 244.948774, 0.01805, 0.031214, 0.072895, 228.210251]
- recent_receiver_grad_trace: [5.50058, 0.017487, 0.010338, 74.854677, 0.006028, 0.011204, 0.027889, 80.126435]
- latest_optimizer_lrs: {'prefix_projector': 0.00015, 'receiver_lora': 5e-05, 'writer_base': 0.0001}
- snapshot_steps_seen: [0, 10, 25, 50, 100, 200, 350, 500]
- latest_snapshot_step: 500
- latest_snapshot_accuracy: 0.671875
- latest_snapshot_macro_f1: 0.534591
- latest_snapshot_margin: 3.233097
- latest_snapshot_prefix_l2: 203.063765

## Log Tail
```text
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
/root/miniconda3/lib/python3.12/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
/root/miniconda3/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:296: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/linear.py:125: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  return F.linear(input, self.weight, self.bias)
/root/miniconda3/lib/python3.12/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
/root/miniconda3/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:296: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/linear.py:125: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  return F.linear(input, self.weight, self.bias)
/root/mydir/MemTOTAL/src/memtotal/training/m4_shared_injection.py:3695: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  similarity = torch.matmul(normalized, normalized.transpose(1, 2))
/root/miniconda3/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:108: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
/root/miniconda3/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:115: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  attn_output = torch.matmul(attn_weights, value_states)
/root/mydir/MemTOTAL/src/memtotal/training/m4_shared_injection.py:3605: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  similarity = torch.matmul(normalized, normalized.transpose(1, 2))
/root/miniconda3/lib/python3.12/site-packages/torch/autograd/graph.py:829: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:233.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/root/miniconda3/lib/python3.12/site-packages/torch/autograd/graph.py:829: UserWarning: Memory Efficient attention defaults to a non-deterministic algorithm. To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False). (Triggered internally at /pytorch/aten/src/ATen/native/transformers/cuda/attention_backward.cu:775.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
review artifacts refreshed
[2026-03-09T10:06:49Z] training script completed; rerunning evaluation summary
review artifacts refreshed
[2026-03-09T10:06:49Z] evaluation summary completed
```
