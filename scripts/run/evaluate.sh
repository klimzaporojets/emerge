#!/bin/bash
# Run the EMERGE evaluation pipeline.
#
# Scores benchmark model predictions against ground-truth annotations.
# Produces a wiki_eval_result.pkl that the statistics notebooks consume.
#
# Requirements:
#   - conda env: emerge (see requirements/core.txt)
#   - GPU: 1x with 16GB+ VRAM (for BERTScore, sentence-transformers)
#   - RAM: 32GB minimum without KG snapshots
#          180GB+ if using KG snapshots (for relik-cie Exists evaluation)
#   - Time: ~2h with KG snapshots, ~30min without
#
# Usage:
#   ./scripts/run/evaluate.sh [config_file]
#
# Examples:
#   # Recommended — all 13 models, correct scoring, with KG snapshots:
#   ./scripts/run/evaluate.sh config/evaluation/s0x_evaluate_predictions/20260324_all_models_with_zs_fixed_with_kg/config.json
#
#   # Legacy scoring (reproduces original ICML submission numbers):
#   ./scripts/run/evaluate.sh config/evaluation/s0x_evaluate_predictions/20260324_all_models_with_zs_legacy_with_kg/config.json
#
#   # Default (no argument) — uses the recommended config:
#   ./scripts/run/evaluate.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG_FILE="${1:-config/evaluation/s0x_evaluate_predictions/20260324_all_models_with_zs_fixed_with_kg/config.json}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "data/evaluation_set" ]; then
    echo "Error: dataset not found. Run ./scripts/download_data.sh first."
    exit 1
fi

echo "Running EMERGE evaluation"
echo "  Config: $CONFIG_FILE"
echo ""

export PYTHONPATH="$PWD/src"
python -u -m evaluation.s0x_evaluate_predictions \
    --config_file "$CONFIG_FILE" \
    2>&1 | tee logs/evaluate_$(date +%Y%m%d_%H%M%S).log
