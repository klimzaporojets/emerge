#!/usr/bin/env bash
set -euo pipefail

# ---- conda setup ----
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rakg-py311

# ---- resolve directory of this script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- RAKG external repo ----
# RAKG is a third-party dependency: https://github.com/RUC-NLPIR/RAKG
# Clone it and set RAKG_REPO_PATH before running:
#   git clone https://github.com/RUC-NLPIR/RAKG.git
#   export RAKG_REPO_PATH=/path/to/RAKG
if [ -z "${RAKG_REPO_PATH:-}" ]; then
    echo "Error: RAKG_REPO_PATH is not set."
    echo "Clone the RAKG repo and set the env var:"
    echo "  git clone https://github.com/RUC-NLPIR/RAKG.git"
    echo "  export RAKG_REPO_PATH=/path/to/RAKG"
    exit 1
fi

# ---- PYTHONPATH: wrappers dir (general_io, prediction) + RAKG external repo ----
export PYTHONPATH="${SCRIPT_DIR}/..:${RAKG_REPO_PATH}:${PYTHONPATH:-}"

python -u "${SCRIPT_DIR}/wrapper.py" "$@"
