"""Baseline adapters for external methods."""

from memtotal.baselines.adapters import AdapterBaselineOutput, AdapterBaselineRuntime, run_adapter_baseline_train
from memtotal.baselines.prompting import PromptBaselineOutput, PromptBaselineRuntime

__all__ = [
    "AdapterBaselineOutput",
    "AdapterBaselineRuntime",
    "PromptBaselineOutput",
    "PromptBaselineRuntime",
    "run_adapter_baseline_train",
]
