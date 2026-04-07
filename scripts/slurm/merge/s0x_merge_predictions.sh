#!/bin/bash
############################
# SLURM settings — merge predictions
############################
#SBATCH --job-name=merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --partition=rome
#SBATCH --output=logs/s0x_merge_predictions_%A.out

############################
# Arguments
############################
CONFIG_FILE="$1"
OUTPUT_LOG="$2"

if [[ -z "$CONFIG_FILE" || -z "$OUTPUT_LOG" ]]; then
  echo "Usage: sbatch $0 <config_file> <output_log>"
  echo ""
  echo "Example:"
  echo "  sbatch $0 config/merge/s0x_merge_predictions/20260324_all_models_with_zs/config.json logs/merge_20260324.log"
  exit 1
fi

############################
# Environment setup
############################
source /home/user/.bashrc
cd /path/to/emerge/
git pull

conda activate emerge

############################
# Run
############################
export PYTHONPATH="$PWD/src"

mkdir -p logs

python -u -m merge.s0x_merge_predictions \
    --config_file "${CONFIG_FILE}" \
    2>&1 | tee "${OUTPUT_LOG}"
