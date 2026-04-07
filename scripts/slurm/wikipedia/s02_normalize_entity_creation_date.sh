#!/bin/bash
#SBATCH --job-name=wp02a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --partition=rome
#SBATCH --output=logs/s02_wpedia_normalize_entity_creation_date_%A.out

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
srun python -u -m dataset.wikipedia.s02_normalize_entity_creation_date \
    --config_file $config_path \
    2>&1 | tee $output_log_path
