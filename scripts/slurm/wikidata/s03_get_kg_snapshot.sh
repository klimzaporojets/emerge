#!/bin/bash
#SBATCH --job-name=wd03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --partition=rome
#SBATCH --output=logs/s03_get_kg_snapshot_%A.out

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
srun python -u -m dataset.wikidata.python.s03_get_kg_snapshot \
    --config_file $config_path \
    --debug_nr_triples -1 2>&1 | tee $output_log_path
