#!/bin/bash
############################
# SLURM settings — for ReLiK (local GPU model)
# ReLiK loads per-snapshot entity/relation indices into memory.
# relik-cie (BOTH) needs ~32GB for indices + model.
# relik-oie (TRIPLET) needs less (~16GB).
############################
#SBATCH --job-name=s02_relik
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=23:59:00
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --output=logs/s02_run_relik_%A.out

############################
# Arguments
############################
CONFIG_FILE="$1"
OUTPUT_LOG="$2"

if [[ -z "$CONFIG_FILE" || -z "$OUTPUT_LOG" ]]; then
  echo "Usage: sbatch $0 <config_file> <output_log>"
  echo ""
  echo "Examples:"
  echo "  # ReLiK CIE (entity linking + relation extraction):"
  echo "  sbatch $0 config/benchmarks/s02_run_benchmarks/relik_cie/config.json logs/relik_cie.log"
  echo ""
  echo "  # ReLiK Open IE (relation extraction only):"
  echo "  sbatch $0 config/benchmarks/s02_run_benchmarks/relik_oie/config.json logs/relik_oie.log"
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
