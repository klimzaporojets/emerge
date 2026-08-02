#!/usr/bin/env bash
set -euo pipefail

# ---- conda setup ----
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
# KG-aware baseline runs in the same env as the EDC+ wrapper (needs openai / aiohttp /
# transformers for edc.utils.unified_llm_client.UnifiedLLMClient). Defaults to `edc`;
# override with e.g. `export KG_PROMPT_ENV=my-edc-env` if your EDC+ env is named differently.
conda activate "${KG_PROMPT_ENV:-edc}"

# ---- resolve directory of this script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- PYTHONPATH ----
#   ${SCRIPT_DIR}/..                      -> general_io, prediction (wrappers/)
#   ${SCRIPT_DIR}/../edc_plus/edc_tt2kg   -> edc.utils.unified_llm_client
export PYTHONPATH="${SCRIPT_DIR}/../edc_plus/edc_tt2kg:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

python -u "${SCRIPT_DIR}/wrapper.py" "$@"
