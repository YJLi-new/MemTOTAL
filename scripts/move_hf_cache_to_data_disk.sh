#!/usr/bin/env bash
set -euo pipefail

SOURCE="${HF_HOME:-$HOME/.cache/huggingface}"
TARGET="${1:-/root/autodl-tmp/.cache/huggingface}"

mkdir -p "$(dirname "${TARGET}")"

if [[ -L "${SOURCE}" ]]; then
  echo "already-linked ${SOURCE} -> $(readlink -f "${SOURCE}")"
  exit 0
fi

if [[ -d "${SOURCE}" ]]; then
  mv "${SOURCE}" "${TARGET}"
else
  mkdir -p "${TARGET}"
fi

ln -s "${TARGET}" "${SOURCE}"
echo "linked ${SOURCE} -> ${TARGET}"
