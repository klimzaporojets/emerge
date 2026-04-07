#!/bin/bash
#SBATCH --job-name=wd02b_cat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --partition=rome
#SBATCH --output=logs/s02_wdata_cat_sort_%A.out

OUTPUT_BASE=$1

if [[ -z "$OUTPUT_BASE" ]]; then
  echo "Usage: sbatch $0 <output_base_dir>"
  echo "Example: sbatch $0 /path/to/storage/wikidata-processing/output/experiments/s02_normalize_history_graph/20250320"
  exit 1
fi

set -euo pipefail

mkdir -p $OUTPUT_BASE/s02b_cat/

echo "$(date) - concatenating files..."
cat $OUTPUT_BASE/s02a_generated_history_triples_filtered/* > $OUTPUT_BASE/s02b_cat/merged_file.csv
echo "$(date) - concatenation done, starting sort..."

sort -S 120G --parallel=5 $OUTPUT_BASE/s02b_cat/merged_file.csv \
  -o $OUTPUT_BASE/s02b_cat/merged_file_sorted.csv

echo "$(date) - sort done, removing unsorted file..."
rm $OUTPUT_BASE/s02b_cat/merged_file.csv

echo "$(date) - done."
