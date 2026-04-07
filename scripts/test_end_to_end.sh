#!/bin/bash
# End-to-end test script for EMERGE benchmark.
# Run this on a GPU node (e.g., Snellius H100) to verify the repo works from scratch.
#
# Prerequisites:
#   - Conda available
#   - GPU with 16GB+ VRAM
#   - 180GB+ RAM (for KG snapshots in evaluation)
#
# Usage (on Snellius):
#   salloc -p gpu_h100 -n 1 --gpus 1 --cpus-per-task 10 -t 2:00:00 --mem=180G
#   bash scripts/test_end_to_end.sh
#
# The script assumes it is run from the repository root.

set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="emerge-test"
# If scratch storage is available, symlink data/ there to avoid filling home quota.
# This is important on Snellius where home is limited (~200GB).
SCRATCH_DATA="/scratch-shared/$USER/emerge-data"
if [ -d "/scratch-shared/$USER" ] && [ ! -L "$REPO_DIR/data" ]; then
    echo "Scratch storage detected. Symlinking data/ to $SCRATCH_DATA"
    mkdir -p "$SCRATCH_DATA"
    # Preserve data/README.md (tracked in git)
    if [ -f "$REPO_DIR/data/README.md" ]; then
        cp "$REPO_DIR/data/README.md" "$SCRATCH_DATA/README.md"
    fi
    rm -rf "$REPO_DIR/data"
    ln -sfn "$SCRATCH_DATA" "$REPO_DIR/data"
fi
DATA_DIR="$REPO_DIR/data"

echo "=== EMERGE end-to-end test ==="
echo "Repo:     $REPO_DIR"
echo "Data dir: $(readlink -f $DATA_DIR)"
echo ""

# ---- Step 1: Create conda env ----
echo "=== Step 1: Creating conda environment '$ENV_NAME' ==="
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Python: $(which python) ($(python --version))"

# ---- Step 2: Install PyTorch + core requirements ----
echo ""
echo "=== Step 2: Installing PyTorch + core requirements ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r "$REPO_DIR/requirements/core.txt"
python -c "import nltk; nltk.download('words'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Verify torch sees GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# ---- Step 3: Download dataset (test set + annotations) ----
echo ""
echo "=== Step 3: Downloading dataset ==="
cd "$REPO_DIR"
bash ./scripts/download_data.sh

# Verify
echo "Verifying evaluation_set..."
ls "$DATA_DIR/evaluation_set/snapshot_2024-01-01/" | head -5
echo "Verifying annotation..."
ls "$DATA_DIR/annotation/solved_disagreements.jsonl"

# ---- Step 4: Download KG snapshots ----
echo ""
echo "=== Step 4: Downloading KG snapshots ==="
bash ./scripts/download_data.sh --kg

# Verify
echo "Verifying KG snapshots..."
ls "$DATA_DIR/kg_snapshots/"*.tsv | head -3
ls "$DATA_DIR/kg_snapshots/"*.tsv.gz | head -3

# ---- Step 5: Download indices ----
echo ""
echo "=== Step 5: Downloading relation indices ==="
bash ./scripts/download_data.sh --indices

# Verify
echo "Verifying indices..."
ls "$DATA_DIR/indices/relik_edc_relation_indexes/"

# ---- Step 6: Run evaluation ----
echo ""
echo "=== Step 6: Running evaluation ==="
bash ./scripts/run/evaluate.sh

# Verify
echo "Verifying evaluation output..."
find "$REPO_DIR/output/s0x_evaluate_predictions/" -name "wiki_eval_result.pkl" | head -1

# ---- Step 7: Run annotation agreement ----
echo ""
echo "=== Step 7: Running annotation agreement ==="
bash ./scripts/run/annotation_agreement.sh

# ---- Step 8: Check notebooks have saved outputs ----
echo ""
echo "=== Step 8: Checking notebooks have saved outputs ==="
python -c "
import json, sys
notebooks = [
    'src/stats/evaluation/tables_results.ipynb',
    'src/stats/dataset/appendix_table_annotation_agreement.ipynb',
]
for nb_path in notebooks:
    with open(nb_path) as f:
        nb = json.load(f)
    cells_with_output = sum(1 for c in nb['cells'] if c['cell_type'] == 'code' and c.get('outputs'))
    total_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    status = 'OK' if cells_with_output > 0 else 'EMPTY'
    print(f'  {status}: {nb_path} ({cells_with_output}/{total_code} code cells have outputs)')
"

echo ""
echo "=== All tests passed ==="
echo ""
echo "To clean up:"
echo "  conda deactivate"
echo "  conda env remove -n $ENV_NAME"
