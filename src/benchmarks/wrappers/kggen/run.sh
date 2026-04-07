#!/usr/bin/env bash
set -euo pipefail

# ---- conda setup ----
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate kggen-py312

# ---- resolve directory of this script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- PYTHONPATH: wrappers dir (general_io, prediction) ----
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

python -u "${SCRIPT_DIR}/wrapper.py" "$@"
