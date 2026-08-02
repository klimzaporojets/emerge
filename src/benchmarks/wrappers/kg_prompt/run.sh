#!/usr/bin/env bash
set -euo pipefail

# ---- conda setup ----
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
# KG-aware baseline runs in the same `edc` env as the EDC+ wrapper (has openai /
# aiohttp / transformers required by edc.utils.unified_llm_client.UnifiedLLMClient).
conda activate edc

# ---- resolve directory of this script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- PYTHONPATH ----
#   ${SCRIPT_DIR}/..                      -> general_io, prediction (wrappers/)
#   ${SCRIPT_DIR}/../edc_plus/edc_tt2kg   -> edc.utils.unified_llm_client
export PYTHONPATH="${SCRIPT_DIR}/../edc_plus/edc_tt2kg:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

python -u "${SCRIPT_DIR}/wrapper.py" "$@"
