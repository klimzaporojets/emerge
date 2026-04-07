#!/bin/bash
############################
# SLURM settings — for REBEL (local GPU model)
############################
#SBATCH --job-name=s02_rebel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu_h100
#SBATCH --output=logs/s02_run_rebel_%A.out

############################
# Arguments
############################
CONFIG_FILE="$1"
OUTPUT_LOG="$2"

if [[ -z "$CONFIG_FILE" || -z "$OUTPUT_LOG" ]]; then
  echo "Usage: sbatch $0 <config_file> <output_log>"
  echo ""
  echo "Example:"
  echo "  sbatch $0 config/benchmarks/s02_run_benchmarks/20260114_rebel/config.json logs/rebel.log"
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

python -u -m benchmarks.run_benchmarks \
    --config_file "${CONFIG_FILE}" \
    2>&1 | tee "${OUTPUT_LOG}"
