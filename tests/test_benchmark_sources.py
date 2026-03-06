from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.tasks.sources import (
    _canonicalize_gpqa,
    _canonicalize_gsm8k,
    _canonicalize_story_cloze,
    _canonicalize_triviaqa,
    get_benchmark_source,
    materialize_benchmark_source,
)


class BenchmarkSourcesTest(unittest.TestCase):
    def test_gsm8k_canonicalizer_extracts_final_answer(self) -> None:
        row = {"question": "What is 2+2?", "answer": "We add.\n#### 4"}
        canonical = _canonicalize_gsm8k(row, index=0, seed=11)
        self.assertEqual(canonical["answer"], "4")

    def test_gpqa_canonicalizer_builds_deterministic_choices(self) -> None:
        row = {
            "Question": "Which option is correct?",
            "Correct Answer": "Alpha",
            "Incorrect Answer 1": "Beta",
            "Incorrect Answer 2": "Gamma",
            "Incorrect Answer 3": "Delta",
        }
        first = _canonicalize_gpqa(row, index=0, seed=13)
        second = _canonicalize_gpqa(row, index=0, seed=13)
        self.assertEqual(first, second)
        self.assertEqual(len(first["choices"]), 4)
        self.assertIn(first["label"], {"A", "B", "C", "D"})

    def test_triviaqa_canonicalizer_keeps_aliases(self) -> None:
        row = {
            "question_id": "q1",
            "question": "Who wrote Hamlet?",
            "answer": {"normalized_aliases": ["william shakespeare", "shakespeare"]},
        }
        canonical = _canonicalize_triviaqa(row, index=0, seed=17)
        self.assertEqual(canonical["answer"], "william shakespeare")
        self.assertEqual(len(canonical["aliases"]), 2)

    def test_story_cloze_canonicalizer_maps_label(self) -> None:
        row = {
            "input_sentence_1": "A",
            "input_sentence_2": "B",
            "input_sentence_3": "C",
            "input_sentence_4": "D",
            "sentence_quiz1": "Ending 1",
            "sentence_quiz2": "Ending 2",
            "answer_right_ending": 2,
        }
        canonical = _canonicalize_story_cloze(row, index=0, seed=19)
        self.assertEqual(canonical["label"], "B")
        self.assertEqual(canonical["answer"], "Ending 2")

    def test_manual_source_materialize_writes_pending_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = materialize_benchmark_source(
                benchmark_id="alfworld",
                output_root=tmp / "materialized",
                manifest_root=tmp / "manifests",
                max_examples=4,
                seed=23,
            )
            self.assertEqual(manifest["status"], "manual_pending")
            saved = json.loads((tmp / "manifests" / "alfworld.json").read_text())
            self.assertEqual(saved["status"], "manual_pending")

    def test_source_registry_marks_gpqa_gated(self) -> None:
        source = get_benchmark_source("gpqa")
        self.assertEqual(source.access, "gated")
        self.assertEqual(source.source_kind, "huggingface")


if __name__ == "__main__":
    unittest.main()
