#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BRANCH_NAME="${1:-review}"
REMOTE_NAME="${2:-origin}"
SNAPSHOT_ROOT="${3:-$(mktemp -d /tmp/memtotal-github-review-XXXXXX)}"
SET_DEFAULT_BRANCH="${SET_DEFAULT_BRANCH:-1}"

SOURCE_COMMIT="$(git rev-parse HEAD)"
REMOTE_URL="$(git remote get-url "${REMOTE_NAME}")"
AUTHOR_NAME="$(git config --get user.name || true)"
AUTHOR_EMAIL="$(git config --get user.email || true)"

python scripts/build_github_review_snapshot.py \
  --output-root "${SNAPSHOT_ROOT}" \
  --source-commit "${SOURCE_COMMIT}"

git -C "${SNAPSHOT_ROOT}" init -b "${BRANCH_NAME}" >/dev/null
git -C "${SNAPSHOT_ROOT}" remote add origin "${REMOTE_URL}"
if [[ -n "${AUTHOR_NAME}" ]]; then
  git -C "${SNAPSHOT_ROOT}" config user.name "${AUTHOR_NAME}"
fi
if [[ -n "${AUTHOR_EMAIL}" ]]; then
  git -C "${SNAPSHOT_ROOT}" config user.email "${AUTHOR_EMAIL}"
fi

touch "${SNAPSHOT_ROOT}/.nojekyll"
git -C "${SNAPSHOT_ROOT}" add .
git -C "${SNAPSHOT_ROOT}" commit -m "review snapshot from ${SOURCE_COMMIT}" >/dev/null
git -C "${SNAPSHOT_ROOT}" push --force origin "HEAD:${BRANCH_NAME}"

if [[ "${SET_DEFAULT_BRANCH}" == "1" ]] && command -v gh >/dev/null 2>&1; then
  REPO_SLUG="$(gh repo view --json nameWithOwner --jq '.nameWithOwner')"
  gh api "repos/${REPO_SLUG}" -X PATCH -F "default_branch=${BRANCH_NAME}" >/dev/null
fi

echo "review snapshot pushed"
echo "branch=${BRANCH_NAME}"
echo "source_commit=${SOURCE_COMMIT}"
echo "snapshot_root=${SNAPSHOT_ROOT}"
