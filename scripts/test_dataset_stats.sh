#!/bin/bash
# Lightweight smoke test for the dataset / stats migration scripts.
#
# What it covers (no GPU, no SLURM, no LLM API):
#   - bash syntax of download_data.sh
#   - --help works on every script in scripts/dataset/ and scripts/stats/
#   - pytest tests/ runs green (catches s05 regressions + chain-script bugs)
#
# Usage:
#   bash scripts/test_dataset_stats.sh                # ~5 s, no data needed
#   bash scripts/test_dataset_stats.sh --with-data    # ~2 min on first run
#                                                     # (pulls 2.3 GB corpus,
#                                                     # then verifies paper §4.3
#                                                     # headline numbers reproduce)
#
# Distinct from `scripts/test_end_to_end.sh` — that one runs the full benchmark
# evaluation pipeline and needs 16 GB+ GPU + 180 GB RAM. This script needs nothing.
#
# Exits 0 on success, non-zero on the first failure.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"
ok() { printf "${GREEN}OK${NC}   %s\n" "$1"; }
fail() { printf "${RED}FAIL${NC} %s\n" "$1"; exit 1; }

# ---- 1. download_data.sh syntax + flags wired up ----
echo "[1/4] bash syntax + --help on download_data.sh"
bash -n scripts/download_data.sh || fail "download_data.sh has bash syntax errors"
help_out=$(./scripts/download_data.sh --help)
echo "$help_out" | grep -q -- "--corpus" || fail "download_data.sh --help is missing --corpus"
echo "$help_out" | grep -q -- "--indices" || fail "download_data.sh --help is missing --indices"
echo "$help_out" | grep -q -- "--kg" || fail "download_data.sh --help is missing --kg"
ok "download_data.sh"

# ---- 2. --help on every ported script ----
echo "[2/4] --help on each ported script in scripts/dataset/ + scripts/stats/"
for script in scripts/dataset/{find_garbage_clusters,inspect_cluster_neighbors,build_reinput_for_garbage,merge_reinput_into_dataset}.py \
              scripts/stats/{compute_405bv1_dataset_stats,figures_dataset_stats_v2_405bv1,figure_tkgu_distribution_v2_405bv1}.py; do
    python3 "$script" --help >/dev/null 2>&1 || fail "$script --help failed (import error or argparse breakage)"
done
ok "all 7 scripts parse argparse cleanly"

# ---- 3. pytest tests/ green ----
echo "[3/4] pytest tests/"
if ! command -v pytest >/dev/null 2>&1; then
    fail "pytest not installed; run: pip install pytest"
fi
pytest tests/ -q || fail "pytest tests/ failed — see output above"
ok "pytest tests/ green"

# ---- 4. (optional) --with-data: pull the corpus and verify paper numbers ----
if [ "${1:-}" = "--with-data" ]; then
    echo "[4/4] --with-data: pulling corpus and reproducing paper headline numbers"
    if [ ! -d data/corpus ] || [ -z "$(ls data/corpus 2>/dev/null)" ]; then
        ./scripts/download_data.sh --corpus
    else
        echo "  data/corpus/ already populated — skipping download"
    fi
    stats_out=$(python3 scripts/stats/compute_405bv1_dataset_stats.py \
        --root data/corpus \
        --output-tex /tmp/_emerge_smoke_macros.tex \
        --output-json /tmp/_emerge_smoke_stats.json)
    echo "$stats_out" | tail -10

    # Paper numbers per Datasheet K.2 / §4.3 / Table 8:
    #   608,632 x / 207,271 e / 149,382 ee / 219,571 ee-kg / 9,522 d / 1,194,378 total
    for n in 608,632 207,271 149,382 219,571 9,522 1,194,378; do
        echo "$stats_out" | grep -q "$n" \
            || fail "paper headline number $n not reproduced — corpus may be wrong, or script regressed"
    done
    ok "paper headline numbers reproduced (608K x / 207K e / 149K ee / 220K ee-kg / 9.5K d / 1.19M total)"
else
    echo "[4/4] skipped (rerun with --with-data to pull corpus + verify paper numbers)"
fi

echo
printf "${GREEN}All checks passed.${NC}\n"
