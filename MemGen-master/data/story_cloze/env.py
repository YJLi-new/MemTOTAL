import re
from typing import List

from data.base_env import StaticEnv


class StoryClozeEnv(StaticEnv):
    """Reward function for Story Cloze multiple-choice endings."""

    LETTERS = ["A", "B"]

    @classmethod
    def compute_reward(
        cls,
        completions: List[str],
        solution: List,
        options: List[List[str]],
        **kwargs,
    ) -> List[float]:
        scores = []
        for completion, sol, opts in zip(completions, solution, options):
            correct_letter = sol.get("label") if isinstance(sol, dict) else str(sol)
            opts = opts or (sol.get("options") if isinstance(sol, dict) else [])

            pred = cls._extract_choice(completion, opts)
            if pred is None or correct_letter is None:
                scores.append(0.0)
                continue

            if pred.upper() == str(correct_letter).upper():
                scores.append(1.0)
                continue

            correct_idx = cls._letter_to_idx(correct_letter)
            if opts and 0 <= correct_idx < len(opts) and opts[correct_idx].lower() in completion.lower():
                scores.append(0.5)
                continue

            scores.append(0.0)

        return scores

    @classmethod
    def _extract_choice(cls, completion: str, options: List[str]):
        match = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match[-1].strip()
            normalized = cls._normalize_candidate(candidate, options)
            if normalized:
                return normalized

        boxed = re.findall(r"\\boxed{([^}]+)}", completion)
        if boxed:
            candidate = boxed[-1].strip()
            normalized = cls._normalize_candidate(candidate, options)
            if normalized:
                return normalized

        letter_match = re.search(r"\b([A-B])\b", completion, flags=re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

        digit_match = re.search(r"\b([0-1])\b", completion)
        if digit_match:
            return cls._idx_to_letter(int(digit_match.group(1)))

        for idx, opt in enumerate(options or []):
            if opt and opt.lower() in completion.lower():
                return cls._idx_to_letter(idx)

        return None

    @classmethod
    def _normalize_candidate(cls, candidate: str, options: List[str]):
        normalized = candidate.strip()
        upper = normalized.upper()
        if upper in cls.LETTERS:
            return upper

        if normalized.isdigit():
            idx = int(normalized)
            return cls._idx_to_letter(idx)

        for idx, opt in enumerate(options or []):
            if opt and opt.lower() in normalized.lower():
                return cls._idx_to_letter(idx)

        return None

    @classmethod
    def _idx_to_letter(cls, idx: int) -> str:
        return cls.LETTERS[idx] if 0 <= idx < len(cls.LETTERS) else str(idx)

    @classmethod
    def _letter_to_idx(cls, letter: str) -> int:
        try:
            return cls.LETTERS.index(str(letter).upper())
        except ValueError:
            return -1
