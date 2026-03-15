from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.sanitize_review_artifacts import REDACTION, sanitize_paths


class SanitizeReviewArtifactsTest(unittest.TestCase):
    def test_sanitize_paths_redacts_github_token_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact.jsonl"
            token = "gho_" + ("A" * 36)
            artifact.write_text(
                f'{{"text":"access_token={token}"}}\n',
                encoding="utf-8",
            )

            counts = sanitize_paths([root])

            self.assertEqual(counts[artifact], 1)
            payload = artifact.read_bytes()
            self.assertIn(REDACTION, payload)
            self.assertNotIn(token.encode("utf-8"), payload)


if __name__ == "__main__":
    unittest.main()
