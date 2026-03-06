from typing import Dict, List

from datasets import DatasetDict, load_dataset

from data.base_builder import BaseBuilder
from data.cosmosqa.env import CosmosQAEnv


CHOICE_LETTERS = ["A", "B", "C", "D"]


class CosmosQABuilder(BaseBuilder):
    """Builder for the CosmosQA multiple-choice dataset."""

    def get_env_cls(self):
        return CosmosQAEnv

    def _build_datasets(self) -> DatasetDict:
        # Use the parquet version to avoid deprecated loading scripts.
        raw_dataset = load_dataset("Samsoup/cosmos_qa")

        raw_train = raw_dataset["train"]
        raw_valid = raw_dataset["validation"]
        raw_test = raw_dataset["test"]

        num_proc = self.config.get("num_proc", 8)
        train_dataset = raw_train.map(self._preprocess, num_proc=num_proc).select_columns(self._keep_keys())
        valid_dataset = raw_valid.map(self._preprocess, num_proc=num_proc).select_columns(self._keep_keys())
        test_dataset = raw_test.map(self._preprocess, num_proc=num_proc).select_columns(self._keep_keys())

        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["valid"] = valid_dataset
        dataset_dict["test"] = test_dataset

        return dataset_dict

    def _build_sft_datasets(self) -> DatasetDict:
        return self._build_datasets()

    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()

    @classmethod
    def _preprocess(cls, example: Dict) -> Dict:
        options = [example[f"answer{i}"].strip() for i in range(len(CHOICE_LETTERS))]
        label_idx = int(example.get("label", 0))
        label_letter = cls._idx_to_letter(label_idx)

        prompt = (
            "Read the passage and answer the multiple-choice question.\n\n"
            f"Passage: {example['context'].strip()}\n"
            f"Question: {example['question'].strip()}\n\n"
            "Choices:\n"
            + "\n".join([f"{ltr}. {opt}" for ltr, opt in zip(CHOICE_LETTERS, options)])
            + "\nRespond with the best option letter inside <answer></answer>."
        )

        completion = f"Option {label_letter} is the best choice. <answer>{label_letter}</answer>"
        solution = {"label": label_letter, "text": options[label_idx]}

        return {
            "prompt": prompt,
            "completion": completion,
            "solution": solution,
            "answer_choices": options,
            "label": label_letter,
        }

    @classmethod
    def _idx_to_letter(cls, idx: int) -> str:
        if 0 <= idx < len(CHOICE_LETTERS):
            return CHOICE_LETTERS[idx]
        return str(idx)

    @classmethod
    def _keep_keys(cls) -> List[str]:
        return ["prompt", "completion", "solution", "answer_choices", "label"]
