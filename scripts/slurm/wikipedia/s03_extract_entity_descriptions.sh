#!/bin/bash
#SBATCH --job-name=wp03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --partition=rome
#SBATCH --output=log_s03_extract_entity_descriptions_%A.out

# Usage:
#   sbatch scripts/slurm/wikipedia/s03_extract_entity_descriptions.sh \
#     config/dataset/wikipedia/s03_extract_entity_descriptions/20251101_slurm_english/config.json \
#     20251101 8000 1
#
# Arguments:
#   $1 - config file path
#   $2 - experiment id (used for log directory naming)
#   $3 - start port for wiki mapping API
#   $4 - number of API instances

# Check if the correct number of arguments is provided
if [ "$#" -le 3 ]; then
    echo "Usage: $0 <config_file_path> <experiment_id> <start_port_wiki_mapping> <num_instances_wiki_mapping>"
    exit 1
fi

# Assign input parameters to variables
config_file_path=$1
experiment_id=$2
start_port_wiki_mapping=$3
num_instances_wiki_mapping=$4

source /home/user/.bashrc
cd /path/to/emerge
git pull

conda activate emerge

export PYTHONPATH="$PWD/src"

# Loop to start N instances of the API server
mkdir -p logs/s03_API_wiki_${experiment_id}
for ((i=0; i<num_instances_wiki_mapping; i++)); do
    port=$((start_port_wiki_mapping + i))
    echo "Starting instance wiki_mapping $((i + 1)) on port $port"
    padded_id=$(printf "%02d" $i)
    python -u -m dataset.wikipedia.s03_API_wiki_mapping \
          --config_file $config_file_path \
          --debug_size -1 \
          --api_port $port > logs/s03_API_wiki_${experiment_id}/s03_API_wiki_mapping_${padded_id}_${port}.log 2>&1 &
done
# Wait for API servers to load data into memory
sleep 300

python -u -m dataset.wikipedia.s03_extract_entity_descriptions \
  --config_file $config_file_path \
  --nr_threads_processor 79 \
  --api_ports_wiki_mapping $start_port_wiki_mapping \
  2>&1 | tee logs/s03_API_wiki_${experiment_id}/s03_extract_entity_descriptions.log
