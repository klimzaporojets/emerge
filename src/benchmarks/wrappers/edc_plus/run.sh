#!/usr/bin/env bash
set -euo pipefail

# ---- conda setup ----
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate edc

# ---- resolve directory of this script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- EDC repo path ----
# EDC+ code is bundled in the edc_tt2kg/ subdirectory.
# Override with EDC_REPO_PATH env var if using an external checkout.
export EDC_REPO_PATH="${EDC_REPO_PATH:-${SCRIPT_DIR}/edc_tt2kg}"

# ---- PYTHONPATH: edc-tt2kg repo + wrappers dir (general_io, prediction) ----
export PYTHONPATH="${EDC_REPO_PATH}:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

python -u "${SCRIPT_DIR}/wrapper.py" "$@"
