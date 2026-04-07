#!/usr/bin/env bash
set -euo pipefail

# ---- conda setup ----
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate emerge

# ---- resolve directory of this script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- PYTHONPATH: this wrapper dir (benchmark_model, prediction, etc) + parent (general_io) ----
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

python -u "${SCRIPT_DIR}/wrapper.py" "$@"
