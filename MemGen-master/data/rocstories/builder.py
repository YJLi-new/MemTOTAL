import os
from pathlib import Path
from typing import Dict, List

from datasets import DatasetDict, load_dataset

from data.base_builder import BaseBuilder
from data.rocstories.env import RocStoriesEnv


class RocStoriesBuilder(BaseBuilder):
    """Builder for ROCStories (predict the fifth sentence)."""

    def get_env_cls(self):
        return RocStoriesEnv

    def _get_local_data_files(self):
        """Try to locate local ROCStories CSVs before hitting the hub."""
        candidates = []
        cfg_dir = self.config.get("local_data_dir")
        env_dir = os.environ.get("ROCSTORIES_LOCAL_DIR")
        if cfg_dir:
            candidates.append(cfg_dir)
        if env_dir:
            candidates.append(env_dir)

        # Common fallback locations.
        repo_data_dir = os.path.abspath(os.path.join(os.getcwd(), "data"))
        package_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data"))
        candidates.extend([repo_data_dir, package_data_dir, "/data"])

        def _find_file(root: str, pattern: str):
            root_path = Path(root).expanduser().resolve()
            search_roots = [
                root_path,
                root_path / "rocstories",
                root_path / "RocStories",
                root_path / "ROCStories",
            ]
            for sroot in search_roots:
                try:
                    if not sroot.exists():
                        continue
                    match = next(sroot.rglob(pattern), None)
                    if match:
                        return str(match)
                except OSError:
                    # Ignore broken mounts (e.g., stale NFS handles) and keep searching.
                    continue
            return None

        for cand in candidates:
            base_dir = os.path.abspath(os.path.expanduser(cand))
            train_path = _find_file(base_dir, "ROCStories*winter2017*.csv")
            test_path = _find_file(base_dir, "ROCStories*spring2016*.csv")
            if train_path and test_path:
                return {"train": train_path, "test": test_path}

        return None

    def _build_datasets(self) -> DatasetDict:
        data_files = self._get_local_data_files()
        if data_files is None:
            data_files = {
                "train": "hf://datasets/wza/roc_stories/ROCStories_winter2017.csv",
                "test": "hf://datasets/wza/roc_stories/ROCStories__spring2016.csv",
            }
        else:
            print(f"Using local ROCStories data from: {os.path.dirname(data_files['train'])}")

        raw_dataset = load_dataset("csv", data_files=data_files)

        raw_train = raw_dataset["train"]
        raw_test = raw_dataset["test"]

        valid_ratio = float(self.config.get("valid_ratio", 0.1))
        test_size = int(len(raw_train) * valid_ratio)
        if test_size <= 0:
            test_size = 1
        if test_size >= len(raw_train):
            test_size = len(raw_train) - 1

        split = raw_train.train_test_split(test_size=test_size, shuffle=True)
        raw_train, raw_valid = split["train"], split["test"]
        raw_train = self.limit_split(raw_train, "train")
        raw_valid = self.limit_split(raw_valid, "valid")
        raw_test = self.limit_split(raw_test, "test")

        num_proc = int(self.config.get("num_proc", self.config.get("num_workers", 8)))
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
        sentences = [example.get(f"sentence{i}", "").strip() for i in range(1, 6)]
        context = "\n".join([f"{idx}. {sent}" for idx, sent in enumerate(sentences[:4], start=1)])

        prompt = (
            "You are given the first four sentences of a short story. "
            "Write a coherent fifth sentence that best continues the narrative.\n\n"
            f"{context}\n\n"
            "Provide only the final sentence wrapped inside <answer></answer>."
        )

        ending = sentences[4]
        completion = f"<answer>{ending}</answer>"

        return {
            "prompt": prompt,
            "completion": completion,
            "solution": ending,
            "story_id": example.get("storyid"),
            "story_title": example.get("storytitle"),
            "sentences": sentences,
        }

    @classmethod
    def _keep_keys(cls) -> List[str]:
        return ["prompt", "completion", "solution", "story_id", "story_title", "sentences"]
