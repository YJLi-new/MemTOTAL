# GitHub Review Export

This repo’s local working tree is the full research workspace. The GitHub-facing review export is a separate, lightweight snapshot intended for external review and artifact browsing.

## Rules

- The default downloadable GitHub `.zip` from the review branch must stay under `31 MB`.
- The default downloadable GitHub `.zip` for `YJLi-new/MemTOTAL` is expected to come from the lightweight `review` branch, not from the full local research branch.
- The review snapshot contains:
  - root review-facing docs and root text entrypoints,
  - the full `docs/` tree,
  - `src/`, `scripts/`, `configs/`, and `tests/` for lightweight reproduction,
  - governed review bundles under `results/generated/review/` and `runs/review/`,
  - no bulky raw traces such as `train_events.json` or `task_case_dump.jsonl`.
- The review snapshot is not the full research repo. It is a curated publication layer for documentation and latest useful experimental reports.
- The review snapshot should use as much of the `31 MB` budget as is useful for review and reproduction; do not keep it artificially tiny if more governed content can fit safely.

## Entry Points

These commands are run from the full local working repository, not from the lightweight GitHub review branch itself.

- Build a snapshot locally:
  - `python scripts/build_github_review_snapshot.py --output-root /tmp/memtotal-review-snapshot --print-summary`
- Push the lightweight snapshot to GitHub:
  - `bash scripts/push_github_review_snapshot.sh review`

## What Gets Included

- The full `docs/` tree.
- Root text entrypoints such as:
  - `README.md`
  - `AGENTS.md`
  - `PLANv7.md`
  - root `PLAN*.md` files
  - `pyproject.toml`
  - lightweight root entry scripts
- Reproduction surface:
  - `src/`
  - `scripts/`
  - `configs/`
  - `tests/`
- Governed review artifacts:
  - filtered `results/generated/review/`
  - filtered `runs/review/`

## Important Limitation

Publishing a lightweight `review` branch does not erase the full historical object store already present on the GitHub remote. If the total remote repository storage must also be reduced below the same budget, that requires either:

- rewriting the remote default branch/history to the review snapshot, or
- using a separate dedicated GitHub repository for the review export.

For the current repo policy, the practical enforcement target is the downloadable default-branch `.zip`. The export script therefore keeps the `review` branch lightweight and updates the GitHub default branch to `review` with `gh`.
