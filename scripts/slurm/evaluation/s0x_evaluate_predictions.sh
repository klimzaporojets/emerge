#!/bin/bash
############################
# SLURM settings
############################
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --mem=180G
#SBATCH --partition=gpu_h100
#SBATCH --output=logs/s0x_evaluate_predictions_%A.out

############################
# Arguments
############################
CONFIG_FILE="$1"
EXPERIMENT_ID="$2"

if [[ -z "$CONFIG_FILE" || -z "$EXPERIMENT_ID" ]]; then
  echo "Usage: sbatch $0 <config_file> <experiment_id>"
  echo ""
  echo "Examples:"
  echo "  # Fixed scores, no KG snapshots:"
  echo "  sbatch $0 config/evaluation/s0x_evaluate_predictions/20260217_submitted_icml/config.json 20260217_fixed"
  echo ""
  echo "  # Fixed scores, WITH KG snapshots (needed for relik-cie x-triples):"
  echo "  sbatch $0 config/evaluation/s0x_evaluate_predictions/20260324_submitted_icml_fixed_with_kg/config.json 20260324_fixed_kg"
  echo ""
  echo "  # Legacy scores, WITH KG snapshots:"
  echo "  sbatch $0 config/evaluation/s0x_evaluate_predictions/20260324_submitted_icml_legacy_with_kg/config.json 20260324_legacy_kg"
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

srun python -u -m evaluation.s0x_evaluate_predictions \
    --config_file "${CONFIG_FILE}" \
    2>&1 | tee "logs/s0x_evaluate_predictions_${EXPERIMENT_ID}.log"
