import re
from typing import List

from data.base_env import StaticEnv


class CosmosQAEnv(StaticEnv):
    """Reward function for CosmosQA multiple-choice answers."""

    LETTERS = ["A", "B", "C", "D"]

    @classmethod
    def compute_reward(
        cls,
        completions: List[str],
        solution: List,
        answer_choices: List[List[str]],
        **kwargs,
    ) -> List[float]:
        scores = []
        for completion, sol, options in zip(completions, solution, answer_choices):
            correct_letter = sol.get("label") if isinstance(sol, dict) else str(sol)
            options = options or (sol.get("answer_choices") if isinstance(sol, dict) else [])

            pred = cls._extract_choice(completion, options)
            if pred is None or correct_letter is None:
                scores.append(0.0)
                continue

            if pred.upper() == str(correct_letter).upper():
                scores.append(1.0)
                continue

            # Partial credit if the model echoes the correct option text.
            correct_idx = cls._letter_to_idx(correct_letter)
            if options and 0 <= correct_idx < len(options):
                if options[correct_idx].lower() in completion.lower():
                    scores.append(0.5)
                    continue

            scores.append(0.0)

        return scores

    @classmethod
    def _extract_choice(cls, completion: str, options: List[str]):
        """Extract the model's choice as a letter."""
        # Prefer explicit answer tags.
        match = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match[-1].strip()
            normalized = cls._normalize_candidate(candidate, options)
            if normalized:
                return normalized

        # Look for boxed answers.
        boxed = re.findall(r"\\boxed{([^}]+)}", completion)
        if boxed:
            candidate = boxed[-1].strip()
            normalized = cls._normalize_candidate(candidate, options)
            if normalized:
                return normalized

        # Direct letter mention.
        letter_match = re.search(r"\b([A-D])\b", completion, flags=re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

        # Numeric index.
        digit_match = re.search(r"\b([0-3])\b", completion)
        if digit_match:
            return cls._idx_to_letter(int(digit_match.group(1)))

        # Option text inclusion.
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
