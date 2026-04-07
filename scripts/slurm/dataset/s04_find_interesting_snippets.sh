#!/bin/bash
#SBATCH --job-name=s04_interesting
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --output=log_s04_find_interesting_snippets_%A.out


config_file=$1
output_log=$2

source /home/user/.bashrc
cd /path/to/emerge
git pull

conda activate emerge
#
#python -u src/s02b_normalize_history_graph.py \
#--config_file $config_file \
#--second_run_cat_sorted 2>&1 | tee $output_log

export PYTHONPATH="$PWD/src"
python -u -m dataset.emerge.s04_find_interesting_snippets_v3 \
  --config_file $config_file \
  --device cuda:0 \
  --device2 cuda:0 \
  --batching_type method1 \
  --batch_size 20 \
  2>&1 | tee $output_log

#python -u src/s02b_normalize_history_graph.py --config_file experiments/s02b_normalize_concatenated/20240808/s02b_config_cat_sort_graph_normalization.json 2>&1 | tee logs/s02b_normalize_concatenated_20240808.log

