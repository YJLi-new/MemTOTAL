#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?root dir required}"
RESULT_ROOT="${2:?result root required}"
RUN_ROOT="${3:?run root required}"

cd "${ROOT_DIR}"

while [[ ! -f "${RESULT_ROOT}/v9-1-summary.json" ]]; do
  sleep 60
done

bash scripts/publish_review_artifacts.sh

git add \
  docs/exec-plans/active/20260315-planv9-v9-1-longhorizon-baselines-qwen34.md \
  results/generated/review/planv9-v9-1-longhorizon-baselines-qwen34 \
  runs/review/planv9-v9-1-longhorizon-baselines-qwen34

if ! git diff --cached --quiet --exit-code; then
  git commit -m "feat: publish planv9 v9-1 longhorizon baselines"
fi

env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy git push origin main
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  scripts/push_github_review_snapshot.sh review origin
