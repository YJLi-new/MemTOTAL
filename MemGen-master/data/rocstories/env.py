import re
from difflib import SequenceMatcher
from typing import List

from data.base_env import StaticEnv


class RocStoriesEnv(StaticEnv):
    """Reward for ROCStories endings."""

    @classmethod
    def compute_reward(cls, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        scores = []
        for completion, target in zip(completions, solution):
            predicted = cls._extract_answer(completion)
            if not target:
                scores.append(0.0)
                continue

            # Exact or substring match is perfect.
            if target.lower() in completion.lower():
                scores.append(1.0)
                continue

            ratio = SequenceMatcher(None, predicted.lower(), target.lower()).ratio()
            scores.append(ratio)

        return scores

    @classmethod
    def _extract_answer(cls, completion: str) -> str:
        """Pull the model's proposed ending."""
        match = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match[-1].strip()

        # Fallback: grab the last sentence.
        sentences = re.split(r"(?<=[.!?])\s+", completion.strip())
        return sentences[-1] if sentences else completion.strip()
