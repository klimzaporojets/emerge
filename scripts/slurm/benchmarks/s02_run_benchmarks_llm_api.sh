#!/bin/bash
############################
# SLURM settings — for LLM API models (KG-GEN, RAKG)
# These call external APIs, no GPU needed.
############################
#SBATCH --job-name=s02_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:59:00
#SBATCH --mem=8G
#SBATCH --partition=rome
#SBATCH --output=logs/s02_run_benchmarks_%A.out

############################
# Arguments
############################
CONFIG_FILE="$1"
OUTPUT_LOG="$2"

if [[ -z "$CONFIG_FILE" || -z "$OUTPUT_LOG" ]]; then
  echo "Usage: sbatch $0 <config_file> <output_log>"
  echo ""
  echo "Examples:"
  echo "  # KG-GEN GPT 5.1:"
  echo "  sbatch $0 config/benchmarks/s02_run_benchmarks/20260116_kggen_gpt_5_1/config.json logs/kggen_gpt_5_1.log"
  echo ""
  echo "  # RAKG Mistral Large:"
  echo "  sbatch $0 config/benchmarks/s02_run_benchmarks/20260114_rakg_mistral_large/config.json logs/rakg_mistral_large.log"
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
