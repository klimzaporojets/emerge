#!/bin/bash
# Download the EMERGE dataset from HuggingFace Hub.
#
# Usage:
#   ./scripts/download_data.sh                    # downloads evaluation set + annotations
#   ./scripts/download_data.sh --kg               # also downloads KG snapshots (~3.8GB compressed)
#   ./scripts/download_data.sh --indices          # also downloads ReLiK/EDC+ relation indices (~400MB)
#   ./scripts/download_data.sh --corpus           # also downloads the full 233K-instance corpus (~2.3GB)
#   ./scripts/download_data.sh --entity-dicts     # also downloads ALL gzipped KG-aware entity dictionaries (~17GB)
#   ./scripts/download_data.sh --entity-dicts=2019-01-01,2024-01-01   # only those snapshots' dictionaries
#   ./scripts/download_data.sh --all              # downloads everything EXCEPT the heavy entity dictionaries
#
#   EMERGE_DATA_DIR=/path ./scripts/download_data.sh --kg   # download to a custom dir (e.g. /projects on HPC)
#
# Requires: pip install huggingface_hub

set -euo pipefail

# ---- Configuration ----
# Download target defaults to <repo>/data, but can be redirected with the
# EMERGE_DATA_DIR env var (e.g. to project/scratch storage on an HPC cluster,
# when the home quota is too small for the ~3.8 GB gzipped KG snapshots):
#   EMERGE_DATA_DIR=/projects/.../data ./scripts/download_data.sh --kg
HF_REPO="klimzaporojets/emerge-benchmark"
DATA_DIR="${EMERGE_DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"

# ---- Parse arguments ----
DOWNLOAD_KG=false
DOWNLOAD_INDICES=false
DOWNLOAD_CORPUS=false
DOWNLOAD_ENTITY_DICTS=false
ENTITY_DICT_SNAPSHOTS=""   # empty = all hosted snapshots; else comma-separated (e.g. 2019-01-01,2024-01-01)
for arg in "$@"; do
    case $arg in
        --kg) DOWNLOAD_KG=true ;;
        --indices) DOWNLOAD_INDICES=true ;;
        --corpus) DOWNLOAD_CORPUS=true ;;
        --entity-dicts) DOWNLOAD_ENTITY_DICTS=true ;;
        --entity-dicts=*) DOWNLOAD_ENTITY_DICTS=true; ENTITY_DICT_SNAPSHOTS="${arg#*=}" ;;
        # --all deliberately does NOT include the heavy (~17GB) entity dictionaries;
        # request those explicitly with --entity-dicts.
        --all) DOWNLOAD_KG=true; DOWNLOAD_INDICES=true; DOWNLOAD_CORPUS=true ;;
        --help|-h)
            echo "Usage: $0 [--kg] [--indices] [--corpus] [--entity-dicts[=SNAPSHOTS]] [--all]"
            echo ""
            echo "Downloads the EMERGE dataset from HuggingFace Hub into data/."
            echo ""
            echo "Options:"
            echo "  --kg            Download KG snapshots (~3.8GB, kept gzipped; readers stream the .tsv.gz)"
            echo "                  Needed for relik-cie Exists operation evaluation"
            echo "  --indices       Download ReLiK/EDC+ RELATION indices (~400MB)"
            echo "                  Needed to re-run ReLiK and EDC+ (canonicalized) benchmarks"
            echo "  --corpus        Download the full 233K-instance corpus (~2.3GB)"
            echo "                  Input to scripts/stats/* — needed to reproduce paper §4.3 / Table 8 numbers"
            echo "  --entity-dicts[=SNAPSHOTS]"
            echo "                  Download the gzipped KG-aware entity DICTIONARIES (~2.7GB each, ~17GB for all 7)."
            echo "                  Optional: SNAPSHOTS = comma-separated list (e.g. 2019-01-01,2024-01-01) to fetch"
            echo "                  only some. Then rebuild each snapshot's embeddings on a GPU with"
            echo "                  scripts/kg_context/s08b_build_relik_index.py (see src/kg_context/README.md)."
            echo "  --all           Everything EXCEPT the heavy entity dictionaries (eval set + KG + indices + corpus)"
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
    echo "Downloading KG snapshots (~3.8GB compressed)..."
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
    echo "    Kept gzipped: all readers STREAM the .tsv.gz directly — do NOT decompress (~22GB)."
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
    allow_patterns=['indices/relik_edc_relation_indexes/**'],
)
"
    echo "  - data/indices/relik_edc_relation_indexes/   (per-snapshot relation indices for ReLiK and EDC+)"
fi

# Optionally download the KG-aware entity DICTIONARIES (gzipped documents.jsonl, opt-in, heavy).
# These feed scripts/kg_context/s08b_build_relik_index.py, which rebuilds each snapshot's
# embeddings.pt locally on a GPU (the dense embeddings are NOT hosted — they barely compress).
if [ "$DOWNLOAD_ENTITY_DICTS" = true ]; then
    echo ""
    echo "Downloading KG-aware entity dictionaries (gzipped; ~2.7GB per snapshot)..."
    ENTITY_DICT_SNAPSHOTS="${ENTITY_DICT_SNAPSHOTS}" python3 -c "
import os
from huggingface_hub import snapshot_download
snaps = os.environ['ENTITY_DICT_SNAPSHOTS'].strip()
if snaps:
    patterns = ['indices/relik_entity_index/%s/documents.jsonl.gz' % s.strip()
                for s in snaps.split(',') if s.strip()]
else:
    patterns = ['indices/relik_entity_index/**/documents.jsonl.gz']
print('  patterns:', patterns)
snapshot_download(
    repo_id='${HF_REPO}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=patterns,
)
"
    echo "  - data/indices/relik_entity_index/<snapshot>/documents.jsonl.gz (gzipped entity dictionaries)"
    echo ""
    echo "  NEXT: rebuild each snapshot's embeddings on a GPU (a few hours for the full ~6M-entity"
    echo "  dictionary; needs the 'relik' conda env):"
    echo "    python scripts/kg_context/s08b_build_relik_index.py \\"
    echo "        --documents data/indices/relik_entity_index/<snapshot>/documents.jsonl.gz \\"
    echo "        --out-dir   data/indices/relik_entity_index/<snapshot> --device cuda"
    echo "  See src/kg_context/README.md for the full KG-aware reproduction flow."
fi

# Optionally download the full 233K-instance corpus (input to scripts/stats/*)
if [ "$DOWNLOAD_CORPUS" = true ]; then
    echo ""
    echo "Downloading full 233K-instance corpus (~2.3GB) — input for scripts/stats/* ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=['corpus/**'],
)
"
    echo "  - data/corpus/          (233K instances (1.19M TKGU operations) across 7 yearly snapshots, post-QA, post-reorder)"
    echo "                          Used by scripts/stats/compute_405bv1_dataset_stats.py and figures_*.py"
fi

echo ""
echo "Done. You can now run the evaluation pipeline:"
echo "  ./scripts/run/evaluate.sh"
