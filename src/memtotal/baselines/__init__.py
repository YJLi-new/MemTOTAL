"""Baseline adapters for external methods."""

from memtotal.baselines.adapters import AdapterBaselineOutput, AdapterBaselineRuntime, run_adapter_baseline_train
from memtotal.baselines.lightthinker import LightThinkerBaselineOutput, LightThinkerBaselineRuntime
from memtotal.baselines.memory_bank import MemoryBankBaselineOutput, MemoryBankBaselineRuntime
from memtotal.baselines.prompting import PromptBaselineOutput, PromptBaselineRuntime
from memtotal.baselines.retrieval import RetrievalBaselineRuntime

__all__ = [
    "AdapterBaselineOutput",
    "AdapterBaselineRuntime",
    "LightThinkerBaselineOutput",
    "LightThinkerBaselineRuntime",
    "MemoryBankBaselineOutput",
    "MemoryBankBaselineRuntime",
    "PromptBaselineOutput",
    "PromptBaselineRuntime",
    "RetrievalBaselineRuntime",
    "run_adapter_baseline_train",
]
