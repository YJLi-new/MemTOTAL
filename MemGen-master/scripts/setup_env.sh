#!/usr/bin/env bash
set -euo pipefail

# One-click environment setup for MemGen on CUDA 12.4.

ENV_NAME="${ENV_NAME:-memgen}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_DIR="${CONDA_DIR:-$HOME/.miniconda}"
INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

log() { echo "[${ENV_NAME}] $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQS_FILE="${REPO_ROOT}/requirements.txt"

# Ensure conda is present.
if [ ! -x "${CONDA_DIR}/bin/conda" ]; then
  log "Miniconda not found; installing to ${CONDA_DIR}"
  tmp_installer="$(mktemp "/tmp/miniconda_installer_XXXXXX.sh")"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${INSTALLER_URL}" -o "${tmp_installer}"
  else
    wget -q "${INSTALLER_URL}" -O "${tmp_installer}"
  fi
  bash "${tmp_installer}" -b -p "${CONDA_DIR}"
  rm -f "${tmp_installer}"
  "${CONDA_DIR}/bin/conda" init bash >/dev/null 2>&1 || true
else
  log "Using existing conda at ${CONDA_DIR}"
fi

# Load conda in the current shell.
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
elif [ -x "${CONDA_DIR}/bin/conda" ]; then
  eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
else
  log "Conda not found after installation; aborting"
  exit 1
fi

# Accept TOS for default channels to avoid non-interactive failures.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

# Create env if missing.
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Environment ${ENV_NAME} already exists"
else
  log "Creating environment ${ENV_NAME} with Python ${PYTHON_VERSION}"
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"
pip install --upgrade pip

# Install CUDA 12.4 PyTorch build.
log "Installing PyTorch (CUDA 12.4) from ${TORCH_INDEX}"
pip install --no-cache-dir torch torchvision torchaudio --index-url "${TORCH_INDEX}"

# Install project dependencies (skip torch pin and relax numpy for py3.10).
if [ -f "${REQS_FILE}" ]; then
  log "Installing project dependencies from ${REQS_FILE}"
  tmp_reqs="$(mktemp)"
  grep -vi '^torch' "${REQS_FILE}" > "${tmp_reqs}"
  # Relax numpy pin if it is incompatible with Python version.
  sed -i 's/^numpy==2.3.5/numpy>=2.1,<2.4/' "${tmp_reqs}"
  pip install --no-cache-dir -r "${tmp_reqs}"
  rm -f "${tmp_reqs}"
else
  log "No requirements.txt found; skipping project dependency install"
fi

log "Setup complete. Activate anytime with: conda activate ${ENV_NAME}"
