from data.base_builder import BaseBuilder
from data.base_env import (
    BaseEnv,
    StaticEnv,
    DynamicEnv,
)
from data.cosmosqa.builder import CosmosQABuilder
from data.gpqa.builder import GPQABuilder
from data.gsm8k.builder import GSM8KBuilder
from data.kodcode.builder import KodCodeBuilder
from data.rocstories.builder import RocStoriesBuilder
from data.story_cloze.builder import StoryClozeBuilder
from data.triviaqa.builder import TriviaQABuilder

_DATA_BUILDER_MAP = {
    "cosmosqa": CosmosQABuilder,
    "gpqa": GPQABuilder,
    "gsm8k": GSM8KBuilder,
    "kodcode": KodCodeBuilder,
    "rocstories": RocStoriesBuilder,
    "story_cloze": StoryClozeBuilder,
    "triviaqa": TriviaQABuilder,
}

def get_data_builder(dataset_cfg) -> BaseBuilder:
    # Normalize user-specified dataset names to avoid case/whitespace mismatches.
    name = dataset_cfg.get("name") if dataset_cfg else None
    if not name:
        raise ValueError("Unsupported dataset.")

    norm_name = str(name).strip().lower().replace("-", "_")
    if norm_name not in _DATA_BUILDER_MAP:
        raise ValueError("Unsupported dataset.")

    # Keep the normalized name for downstream logging/paths.
    dataset_cfg["name"] = norm_name

    builder_cls = _DATA_BUILDER_MAP[norm_name]
    builder = builder_cls(dataset_cfg)

    return builder
