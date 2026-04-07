#!/bin/bash
############################
# SLURM settings
############################
#SBATCH --job-name=wp01a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --partition=rome
#SBATCH --output=logs_01_extract_title_changes_%A_%a.out

############################
# Arguments
############################
CONFIG_FILE="$1"
OUTPUT_LOG="$2"

if [[ -z "$CONFIG_FILE" || -z "$OUTPUT_LOG" ]]; then
  echo "Usage: sbatch $0 <config_file> <output_log>"
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

# 🔑 Make src discoverable, without making it a package
export PYTHONPATH="$PWD/src"

############################
# Run
############################
srun python -u -m dataset.wikipedia.s01_extract_title_changes \
    --config_file "${CONFIG_FILE}" \
    2>&1 | tee "${OUTPUT_LOG}"