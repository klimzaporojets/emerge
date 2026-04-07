#!/bin/bash
############################
# SLURM settings
############################
#SBATCH --job-name=wp01b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=51:00:00
#SBATCH --mem=16G
#SBATCH --partition=rome
#SBATCH --output=logs_01_history_links_extraction_%A_%a.out

############################
# Arguments
############################
NR_THREADS_PROCESSOR="$1"
CONFIG_FILE="$2"
OUTPUT_LOG="$3"

if [[ -z "$NR_THREADS_PROCESSOR" || -z "$CONFIG_FILE" || -z "$OUTPUT_LOG" ]]; then
  echo "Usage: sbatch $0 <nr_threads_processor> <config_file> <output_log>"
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
srun python -u -m dataset.wikipedia.s01_history_links_extraction \
    --nr_threads_processor "${NR_THREADS_PROCESSOR}" \
    --config_file "${CONFIG_FILE}" \
    --flush_individually \
    2>&1 | tee "${OUTPUT_LOG}"