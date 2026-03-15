from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.tasks.setup_data import main as setup_data_main


class SetupDataTest(unittest.TestCase):
    def test_subset_materialize_merges_existing_source_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_path = tmp / "source_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {"benchmark_id": "gsm8k", "status": "materialized"},
                            {"benchmark_id": "story_cloze", "status": "materialized"},
                        ]
                    }
                )
            )
            with patch(
                "memtotal.tasks.setup_data.materialize_benchmark_source",
                return_value={"benchmark_id": "narrativeqa", "status": "materialized"},
            ):
                self.assertEqual(
                    setup_data_main(
                        [
                            "--benchmarks",
                            "narrativeqa",
                            "--output_root",
                            str(tmp / "materialized"),
                            "--manifest_root",
                            str(tmp / "manifests"),
                            "--summary_path",
                            str(summary_path),
                        ]
                    ),
                    0,
                )
            merged = json.loads(summary_path.read_text())
            benchmark_ids = [item["benchmark_id"] for item in merged["benchmarks"]]
            self.assertEqual(set(benchmark_ids), {"gsm8k", "story_cloze", "narrativeqa"})
            self.assertIn("narrativeqa", benchmark_ids)


if __name__ == "__main__":
    unittest.main()
