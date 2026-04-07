#!/bin/bash
# Compute inter-annotator agreement statistics.
#
# Calculates Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha
# between two human annotators and the LLM assessor.
# Outputs a LaTeX table matching the annotation agreement table in the paper.
#
# Requirements:
#   - conda env: emerge (see requirements/core.txt)
#   - CPU only, ~1GB RAM
#   - Time: <1 minute
#
# Usage:
#   ./scripts/run/annotation_agreement.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ ! -f "data/annotation/solved_disagreements.jsonl" ]; then
    echo "Error: annotation data not found. Run ./scripts/download_data.sh first."
    exit 1
fi

echo "Computing annotation agreement statistics"
echo ""

export PYTHONPATH="$PWD/src"
python -u -m dataset.emerge.s09e_annotation_stats_for_paper \
    --config_file config/dataset/emerge/s09e_annotation_stats_for_paper/20250917_revised/config.json
