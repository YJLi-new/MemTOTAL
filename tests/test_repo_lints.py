from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class RepoLintTest(unittest.TestCase):
    def test_governing_doc_cross_links_exist(self) -> None:
        checks = {
            ROOT / "docs/AGENTS.md": ["TODO_LIST.md", "MAIN_IDEA.md", "EXPERIMENTS_INFO.md"],
            ROOT / "docs/TODO_LIST.md": ["MAIN_IDEA.md", "EXPERIMENTS_INFO.md"],
        }
        for path, required_tokens in checks.items():
            text = path.read_text()
            for token in required_tokens:
                self.assertIn(token, text, msg=f"{path} missing reference to {token}")

    def test_results_governance_allows_only_managed_artifacts(self) -> None:
        controlled_suffixes = {".csv", ".tex", ".png", ".svg", ".pdf", ".json", ".jsonl"}
        allowed_top_dirs = {"generated", "reports"}
        results_root = ROOT / "results"
        for path in results_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in controlled_suffixes:
                continue
            relative = path.relative_to(results_root)
            if len(relative.parts) == 1:
                self.fail(f"Managed result artifact must not live at results root: {path}")
            self.assertIn(
                relative.parts[0],
                allowed_top_dirs,
                msg=f"Unmanaged results artifact detected: {path}",
            )

    def test_required_script_entrypoints_exist(self) -> None:
        required = [
            ROOT / "scripts/setup_env.sh",
            ROOT / "scripts/setup_data.sh",
            ROOT / "scripts/setup_benchmark_data.sh",
            ROOT / "scripts/dev_boot_smoke.sh",
            ROOT / "scripts/collect_artifacts.sh",
            ROOT / "scripts/run_train.sh",
            ROOT / "scripts/run_eval.sh",
            ROOT / "scripts/run_analysis.sh",
            ROOT / "scripts/profile_run.sh",
            ROOT / "scripts/run_memgen.sh",
            ROOT / "scripts/run_benchmark_smoke_suite.sh",
            ROOT / "scripts/run_real_benchmark_smoke_suite.sh",
        ]
        for path in required:
            self.assertTrue(path.is_file(), msg=f"Missing script entrypoint: {path}")


if __name__ == "__main__":
    unittest.main()
