from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from memtotal.tasks.writer_jointpeft_data import (
    deterministic_split_rows,
    materialize_writer_jointpeft_bundle,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class WriterJointPEFTDataTest(unittest.TestCase):
    def test_deterministic_split_rows_is_reproducible(self) -> None:
        rows = [{"id": f"row-{index}"} for index in range(10)]
        split_sizes = {"support": 2, "train": 5, "eval": 3}
        first = deterministic_split_rows(rows, split_sizes=split_sizes, seed=17)
        second = deterministic_split_rows(rows, split_sizes=split_sizes, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first["support"]), 2)
        self.assertEqual(len(first["train"]), 5)
        self.assertEqual(len(first["eval"]), 3)
        all_ids = {
            row["id"]
            for split_name in ("support", "train", "eval")
            for row in first[split_name]
        }
        self.assertEqual(len(all_ids), 10)

    @mock.patch("memtotal.tasks.writer_jointpeft_data.materialize_benchmark_source")
    def test_materialize_writer_jointpeft_bundle_writes_expected_medium_splits(
        self,
        mock_materialize_benchmark_source,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gsm8k_materialized = tmp_path / "gsm8k-materialized.jsonl"
            narrativeqa_materialized = tmp_path / "narrativeqa-materialized.jsonl"
            fever_support = tmp_path / "fever-support.jsonl"
            fever_eval = tmp_path / "fever-eval.jsonl"
            _write_jsonl(
                gsm8k_materialized,
                [{"id": f"gsm8k-{index}", "question": f"q{index}"} for index in range(128)],
            )
            _write_jsonl(
                narrativeqa_materialized,
                [{"id": f"narrativeqa-{index}", "question": f"q{index}"} for index in range(64)],
            )
            _write_jsonl(
                fever_support,
                [{"id": f"fever-support-{index}", "claim": f"claim-{index}"} for index in range(8)],
            )
            _write_jsonl(
                fever_eval,
                [{"id": f"fever-eval-{index}", "claim": f"claim-{index}"} for index in range(128)],
            )

            def fake_materialize(*, benchmark_id: str, **_: object) -> dict[str, str]:
                if benchmark_id == "gsm8k":
                    return {"materialized_path": str(gsm8k_materialized)}
                if benchmark_id == "narrativeqa":
                    return {"materialized_path": str(narrativeqa_materialized)}
                raise AssertionError(f"unexpected benchmark_id={benchmark_id}")

            mock_materialize_benchmark_source.side_effect = fake_materialize

            output_root = tmp_path / "bundle"
            source_output_root = tmp_path / "sources"
            manifest_root = tmp_path / "manifests"
            manifest = materialize_writer_jointpeft_bundle(
                output_root=output_root,
                source_output_root=source_output_root,
                manifest_root=manifest_root,
                seed=23,
                fever_support_path=fever_support,
                fever_eval_path=fever_eval,
            )

            self.assertEqual(
                mock_materialize_benchmark_source.call_args_list[0].kwargs["max_examples"],
                128,
            )
            self.assertEqual(
                mock_materialize_benchmark_source.call_args_list[1].kwargs["max_examples"],
                64,
            )
            self.assertEqual(manifest["tasks"]["gsm8k"]["splits"]["support"]["rows"], 8)
            self.assertEqual(manifest["tasks"]["gsm8k"]["splits"]["train"]["rows"], 80)
            self.assertEqual(manifest["tasks"]["gsm8k"]["splits"]["eval"]["rows"], 40)
            self.assertEqual(manifest["tasks"]["narrativeqa"]["splits"]["support"]["rows"], 8)
            self.assertEqual(manifest["tasks"]["narrativeqa"]["splits"]["train"]["rows"], 32)
            self.assertEqual(manifest["tasks"]["narrativeqa"]["splits"]["eval"]["rows"], 24)
            self.assertEqual(manifest["tasks"]["fever"]["splits"]["support"]["rows"], 8)
            self.assertEqual(manifest["tasks"]["fever"]["splits"]["train"]["rows"], 64)
            self.assertEqual(manifest["tasks"]["fever"]["splits"]["eval"]["rows"], 64)
            manifest_path = output_root / "split-manifest.json"
            self.assertTrue(manifest_path.exists())
            persisted = json.loads(manifest_path.read_text())
            self.assertEqual(persisted["seed"], 23)
            self.assertTrue((output_root / "gsm8k" / "support.jsonl").exists())
            self.assertTrue((output_root / "narrativeqa" / "train.jsonl").exists())
            self.assertTrue((output_root / "fever" / "eval.jsonl").exists())
