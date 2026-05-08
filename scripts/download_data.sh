#!/bin/bash
# Download the EMERGE dataset from HuggingFace Hub.
#
# Usage:
#   ./scripts/download_data.sh                    # downloads evaluation set + annotations
#   ./scripts/download_data.sh --kg               # also downloads KG snapshots (~3.7GB compressed)
#   ./scripts/download_data.sh --indices          # also downloads ReLiK/EDC+ relation indices (~400MB)
#   ./scripts/download_data.sh --corpus           # also downloads the full 233K-record corpus (~2.3GB)
#   ./scripts/download_data.sh --all              # downloads everything
#
# Requires: pip install huggingface_hub

set -euo pipefail

# ---- Configuration ----
HF_REPO="klimzaporojets/emerge-benchmark"
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"

# ---- Parse arguments ----
DOWNLOAD_KG=false
DOWNLOAD_INDICES=false
DOWNLOAD_CORPUS=false
for arg in "$@"; do
    case $arg in
        --kg) DOWNLOAD_KG=true ;;
        --indices) DOWNLOAD_INDICES=true ;;
        --corpus) DOWNLOAD_CORPUS=true ;;
        --all) DOWNLOAD_KG=true; DOWNLOAD_INDICES=true; DOWNLOAD_CORPUS=true ;;
        --help|-h)
            echo "Usage: $0 [--kg] [--indices] [--corpus] [--all]"
            echo ""
            echo "Downloads the EMERGE dataset from HuggingFace Hub into data/."
            echo ""
            echo "Options:"
            echo "  --kg       Download KG snapshots (~3.7GB compressed, ~22GB decompressed)"
            echo "             Needed for relik-cie Exists operation evaluation"
            echo "  --indices  Download ReLiK/EDC+ relation indices (~400MB)"
            echo "             Needed to re-run ReLiK and EDC+ (canonicalized) benchmarks"
            echo "  --corpus   Download the full 233K-record corpus (~2.3GB)"
            echo "             Input to scripts/stats/* — needed to reproduce paper §4.3 / Table 8 numbers"
            echo "  --all      Download everything (~6.5GB compressed download, ~25GB on disk after KG decompression)"
            echo "             Evaluation set + KG + indices + corpus"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---- Check dependencies ----
if ! python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
    echo "Error: huggingface_hub is not installed."
    echo "Install it with: pip install huggingface_hub"
    exit 1
fi

# ---- Download ----
echo "Downloading EMERGE dataset from HuggingFace Hub..."
echo "  Repository: $HF_REPO"
echo "  Target directory: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"

# Download evaluation set and annotation data
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=['evaluation_set/**', 'human_annotation/**'],
)
"

echo ""
echo "Downloaded:"
echo "  - data/evaluation_set/  (3,500 instances + 13 model predictions)"
echo "  - data/human_annotation/ (human annotation data)"

# Optionally download KG snapshots
if [ "$DOWNLOAD_KG" = true ]; then
    echo ""
    echo "Downloading KG snapshots (~3.7GB compressed)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=['kg_snapshots/**'],
)
"
    echo "  - data/kg_snapshots/    (7 yearly Wikidata KG snapshots, gzip compressed)"

    # Decompress KG snapshots
    echo ""
    echo "Decompressing KG snapshots..."
    for gz_file in "$DATA_DIR"/kg_snapshots/*.tsv.gz; do
        if [ -f "$gz_file" ]; then
            tsv_file="${gz_file%.gz}"
            if [ -f "$tsv_file" ]; then
                echo "  Skipping $(basename "$tsv_file") (already decompressed)"
            else
                echo "  Decompressing $(basename "$gz_file")..."
                gunzip -k "$gz_file"
            fi
        fi
    done
    echo "  KG snapshots decompressed."
fi

# Optionally download ReLiK/EDC+ indices
if [ "$DOWNLOAD_INDICES" = true ]; then
    echo ""
    echo "Downloading ReLiK/EDC+ relation indices (~400MB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=['indices/**'],
)
"
    echo "  - data/indices/         (per-snapshot relation indices for ReLiK and EDC+)"
fi

# Optionally download the full 233K-record corpus (input to scripts/stats/*)
if [ "$DOWNLOAD_CORPUS" = true ]; then
    echo ""
    echo "Downloading full 233K-record corpus (~2.3GB) — input for scripts/stats/* ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=['corpus/**'],
)
"
    echo "  - data/corpus/          (233K records across 7 yearly snapshots, post-QA, post-reorder)"
    echo "                          Used by scripts/stats/compute_405bv1_dataset_stats.py and figures_*.py"
fi

echo ""
echo "Done. You can now run the evaluation pipeline:"
echo "  ./scripts/run/evaluate.sh"
