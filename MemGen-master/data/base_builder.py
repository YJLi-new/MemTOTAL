from abc import ABC, abstractmethod
from typing import Type

from datasets import DatasetDict

from data.base_env import BaseEnv

class BaseBuilder(ABC):

    def __init__(self, cfg: dict = None):
        super().__init__()
        
        self.mode = cfg.get("mode", "sft")
        self.config = cfg.get(self.mode)
    
    def get_dataset_dict(self) -> DatasetDict:
        method_builder_map = {
            "sft": self._build_sft_datasets,
            "grpo": self._build_rl_datasets,
        }

        if self.mode not in method_builder_map:
            raise ValueError("Unsupported datasets mode")
        
        return method_builder_map[self.mode]()

    def get_num_workers(self, default: int = 32) -> int:
        return int(self.config.get("num_workers", default))

    def limit_split(self, dataset, split_name: str):
        max_samples = self.config.get(f"max_{split_name}_samples")
        if max_samples is None:
            return dataset
        max_samples = min(int(max_samples), len(dataset))
        return dataset.select(range(max_samples))
    
    @abstractmethod
    def get_env_cls(self) -> Type[BaseEnv]:
        ...

    @abstractmethod
    def _build_sft_datasets(self) -> DatasetDict:
        ...
    
    @abstractmethod
    def _build_rl_datasets(self) -> DatasetDict:
        ...
    
