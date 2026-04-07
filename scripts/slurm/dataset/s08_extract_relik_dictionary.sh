#!/bin/bash
#SBATCH --job-name=s08_extract_relik_dictionary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --partition=rome
#SBATCH --output=log_s08_extract_relik_dictionary_%A.out

# Read the first two passed parameters
config_path=$1  # First parameter
output_log_path=$2  # Second parameter

source /home/user/.bashrc
cd /path/to/emerge
git pull

conda activate emerge


# Check if the correct number of arguments is provided
if [ "$#" -le 3 ]; then
    echo "Usage: $0 <config_file_path> <experiment_id> <start_port_s08_API_wiki_mapping> <num_instances_s08_API_wiki_mapping> <start_port_s08_API_only_deltas> <num_instances_s08_API_only_deltas> <cuda_ids>"
    exit 1
fi

# Assign input parameters to variables
config_file_path=$1
experiment_id=$2

start_port_wiki_mapping=$3
num_instances_wiki_mapping=$4

#start_port_only_deltas=$5
#num_instances_only_deltas=$6
#python_script=$3
#shift 5  # Shift the first 6 parameters out, leaving only the list

## Store the remaining arguments in an array
#list=("$@")
#
## Get the length of the array
#length=${#list[@]}

# Loop to start N instances of the Python script
mkdir logs/s08_API_v1_wiki_${experiment_id}
for ((i=0; i<num_instances_wiki_mapping; i++)); do
    port=$((start_port_wiki_mapping + i))
    echo "Starting instance wiki_mapping $((i + 1)) on port $port"
    # python -u src/s03_API_v6_wiki_mapping.py \
          #  --config_file experiments/s03_obtain_textual_delta_snippets_v6/20250124_all_beta/s03_config_obtain_textual_delta.json \
          #  --debug_size -1 \
          #  --api_port 8100 \
          #  2>&1 | tee logs/s03_API_v6_wiki_mapping_20250124_all_beta.log
    padded_id=$(printf "%02d" $i)
    export PYTHONPATH="$PWD/src"
    python -u -m dataset.emerge.s08_API_v1_wiki_mapping \
          --config_file $config_file_path \
          --debug_size -1 \
          --api_port $port > logs/s08_API_v1_wiki_${experiment_id}/s08_API_v1_wiki_mapping_${padded_id}_${port}.log 2>&1 &
done
sleep 300

python -u -m dataset.emerge.s08_extract_relik_dictionary \
  --config_file $config_file_path \
  --nr_threads_processor 79 \
  --api_ports_wiki_mapping 8000 \
  2>&1 | tee logs/s08_API_v1_wiki_${experiment_id}/s08_extract_relik_index.log
#  2>&1 | tee logs/s08_20250326_extract_relik_index.log

#python -u src/s07_reduce_duplicates_v3.py --config_file $config_path 2>&1 | tee $output_log_path

# python -u src/s07_reduce_duplicates_v3.py \
  #  --config_file experiments/s07_reduce_duplicates/20250324/s07_reduce_duplicates.json \
  #  2>&1 | tee logs/s07_20250324_reduce_duplicates_v3.log
