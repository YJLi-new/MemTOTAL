#!/usr/bin/env bash
set -euo pipefail

HF_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
DROP_DATASETS=false
DROP_INCOMPLETE=false

for arg in "$@"; do
  case "${arg}" in
    --drop-datasets)
      DROP_DATASETS=true
      ;;
    --drop-incomplete-model-downloads)
      DROP_INCOMPLETE=true
      ;;
    *)
      echo "unknown option: ${arg}" >&2
      exit 2
      ;;
  esac
done

echo "hf-root ${HF_ROOT}"
df -h "${HF_ROOT}" || true
du -sh "${HF_ROOT}"/* 2>/dev/null | sort -hr || true

if [[ "${DROP_DATASETS}" == "true" ]]; then
  rm -rf "${HF_ROOT}/datasets"
  echo "removed ${HF_ROOT}/datasets"
fi

if [[ "${DROP_INCOMPLETE}" == "true" ]]; then
  find "${HF_ROOT}/hub" -type f -name '*.incomplete' -print -delete 2>/dev/null || true
fi

echo "after-cleanup"
df -h "${HF_ROOT}" || true
du -sh "${HF_ROOT}"/* 2>/dev/null | sort -hr || true
