#!/bin/bash
############################
# SLURM settings — for EDC/EDC+ (LLM API + embedder)
# Uses LLM API for OIE/SD/SC stages, local embedder for schema retrieval.
# No GPU needed if using API-based LLMs. Embedder (e5-mistral) runs on CPU.
############################
#SBATCH --job-name=s02_edc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00
#SBATCH --mem=32G
#SBATCH --partition=rome
#SBATCH --output=logs/s02_run_edc_%A.out

############################
# Arguments
############################
CONFIG_FILE="$1"
OUTPUT_LOG="$2"

if [[ -z "$CONFIG_FILE" || -z "$OUTPUT_LOG" ]]; then
  echo "Usage: sbatch $0 <config_file> <output_log>"
  echo ""
  echo "Examples:"
  echo "  # EDC+ ICL GPT 5.1:"
  echo "  sbatch $0 config/benchmarks/s02_run_benchmarks/edc_plus_icl_gpt_5_1/config.json logs/edc_plus_gpt_5_1.log"
  echo ""
  echo "  # EDC+ ZS Mistral Large:"
  echo "  sbatch $0 config/benchmarks/s02_run_benchmarks/edc_plus_zs_mistral_large/config.json logs/edc_plus_zs_mistral_large.log"
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
