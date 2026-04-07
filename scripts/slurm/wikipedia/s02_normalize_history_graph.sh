#!/bin/bash
#SBATCH --job-name=wp02b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=rome
#SBATCH --output=logs/s02_wpedia_normalize_history_graph_%A.out

config_path=$1
output_log_path=$2

############################
# Environment setup
############################
source /home/user/.bashrc
cd /path/to/emerge/
git pull

conda activate emerge

export PYTHONPATH="$PWD/src"
srun python -u -m dataset.wikipedia.s02b_normalize_history_graph \
    --config_file $config_path \
    2>&1 | tee $output_log_path
