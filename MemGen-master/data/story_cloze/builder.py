from typing import Dict, List

from datasets import DatasetDict, load_dataset

from data.base_builder import BaseBuilder
from data.story_cloze.env import StoryClozeEnv


class StoryClozeBuilder(BaseBuilder):
    """Builder for the Story Cloze Test (2016 split)."""

    def get_env_cls(self):
        return StoryClozeEnv

    def _build_datasets(self) -> DatasetDict:
        base_dataset = load_dataset("gimmaru/story_cloze-2016")["test"]

        valid_ratio = float(self.config.get("valid_ratio", 0.1))
        test_ratio = float(self.config.get("test_ratio", 0.1))
        if valid_ratio + test_ratio >= 1.0:
            raise ValueError("valid_ratio + test_ratio must be less than 1.")

        n = len(base_dataset)
        test_size = int(n * test_ratio)
        if test_size <= 0:
            test_size = 1
        if test_size >= n:
            test_size = n - 1

        split = base_dataset.train_test_split(test_size=test_size, shuffle=True)
        train_valid_dataset, raw_test_dataset = split["train"], split["test"]

        if len(train_valid_dataset) <= 1:
            raw_train_dataset = train_valid_dataset
            raw_valid_dataset = train_valid_dataset
        else:
            valid_size = int(len(train_valid_dataset) * valid_ratio)
            if valid_size <= 0:
                valid_size = 1
            if valid_size >= len(train_valid_dataset):
                valid_size = len(train_valid_dataset) - 1

            split = train_valid_dataset.train_test_split(test_size=valid_size, shuffle=True)
            raw_train_dataset, raw_valid_dataset = split["train"], split["test"]

        raw_train_dataset = self.limit_split(raw_train_dataset, "train")
        raw_valid_dataset = self.limit_split(raw_valid_dataset, "valid")
        raw_test_dataset = self.limit_split(raw_test_dataset, "test")

        num_proc = int(self.config.get("num_proc", self.config.get("num_workers", 8)))
        train_dataset = raw_train_dataset.map(self._preprocess, num_proc=num_proc).select_columns(self._keep_keys())
        valid_dataset = raw_valid_dataset.map(self._preprocess, num_proc=num_proc).select_columns(self._keep_keys())
        test_dataset = raw_test_dataset.map(self._preprocess, num_proc=num_proc).select_columns(self._keep_keys())

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
        sentences = [example.get(f"input_sentence_{i}", "").strip() for i in range(1, 5)]
        options = [example.get("sentence_quiz1", "").strip(), example.get("sentence_quiz2", "").strip()]
        label_idx = int(example.get("answer_right_ending", 1)) - 1
        label_idx = max(0, min(label_idx, 1))
        label_letter = "A" if label_idx == 0 else "B"

        story_prefix = "\n".join([f"{idx}. {sent}" for idx, sent in enumerate(sentences, start=1)])
        prompt = (
            "Choose the better ending for the story.\n\n"
            f"{story_prefix}\n\n"
            f"A) {options[0]}\nB) {options[1]}\n"
            "Return the option letter inside <answer></answer> and optionally restate the chosen ending."
        )

        completion = f"The better ending is option {label_letter}: {options[label_idx]}.\n<answer>{label_letter}</answer>"
        solution = {"label": label_letter, "text": options[label_idx]}

        return {
            "prompt": prompt,
            "completion": completion,
            "solution": solution,
            "options": options,
            "story_id": example.get("story_id"),
        }

    @classmethod
    def _keep_keys(cls) -> List[str]:
        return ["prompt", "completion", "solution", "options", "story_id"]
