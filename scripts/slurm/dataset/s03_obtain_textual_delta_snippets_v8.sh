#!/bin/bash
#SBATCH --job-name=s03_passages
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=96:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu_a100
#SBATCH --gpus=2
#SBATCH --output=log_s03_obtain_textual_delta_snippets_v8_%A.out


source /home/user/.bashrc
cd /path/to/emerge
git pull

conda activate emerge

# Assign input parameters to variables
config_file_path=$1
experiment_id=$2

start_port_wiki_mapping=$3
num_instances_wiki_mapping=$4

start_port_only_deltas=$5
num_instances_only_deltas=$6
output_log=$7
nr_threads_processor=$8

#python_script=$3
shift 8  # Shift the first 6 parameters out, leaving only the list

# Store the remaining arguments in an array
list=("$@")

# Get the length of the array
length=${#list[@]}

# Loop to start N instances of the Python script
# in v8 we use v6 mapping python script
mkdir logs/s03_API_v8_wiki_${experiment_id}
for ((i=0; i<num_instances_wiki_mapping; i++)); do
    port=$((start_port_wiki_mapping + i))
    echo "Starting instance wiki_mapping $((i + 1)) on port $port"
    padded_id=$(printf "%02d" $i)
    export PYTHONPATH="$PWD/src"
    python -u -m dataset.emerge.s03_API_v6_wiki_mapping \
          --config_file $config_file_path \
          --debug_size -1 \
          --api_port $port > logs/s03_API_v8_wiki_${experiment_id}/s03_API_v8_wiki_mapping_${padded_id}_${port}.log 2>&1 &
done

# Loop to start N instances of the Python script
for ((i=0; i<num_instances_only_deltas; i++)); do
    port=$((start_port_only_deltas + i))
    index=$(( i % length ))
    echo "Starting instance only_deltas $((i + 1)) on port $port and on cuda ${list[index]}"
    padded_id=$(printf "%02d" $i)
    python -u -m dataset.emerge.s03_API_v8_only_deltas \
          --config_file $config_file_path \
          --device ${list[index]} \
          --debug_size -1 \
          --api_port $port > logs/s03_API_v8_wiki_${experiment_id}/s03_API_v8_wiki_only_deltas_${padded_id}_${port}.log 2>&1 &
done
sleep 1m
#wait  # Wait for all background processes to finish

# Generate the port sequence
ports_deltas_param=""
for ((i=0; i<num_instances_only_deltas; i++)); do
    ports_deltas_param+="$(($start_port_only_deltas + i)) "
done

# Generate the port sequence
ports_mapping_param=""
for ((i=0; i<num_instances_wiki_mapping; i++)); do
    ports_mapping_param+="$(($start_port_wiki_mapping + i)) "
done

python -u -m dataset.emerge.s03_obtain_textual_delta_snippets_v8 \
  --nr_threads_processor $nr_threads_processor \
  --api_ports_only_deltas $ports_deltas_param \
  --api_ports_wiki_mapping $ports_mapping_param \
  --config_file $config_file_path \
  2>&1 | tee $output_log

#python -u -m dataset.emerge.s03_obtain_textual_delta_snippets_v8 \
#  --nr_threads_processor 45 \
#  --api_ports_only_deltas 8200 8201 8202 8203 8204 8205 8206 8207 8208 8209 8210 8211 8212 8213 8214 8215 8216 8217 8218 8219 \
#  --api_ports_wiki_mapping 8000 \
#  --config_file experiments/s03_obtain_textual_delta_snippets_v8/20250311/s03_config_obtain_textual_delta.json \
#  2>&1 | tee logs/s03_config_obtain_textual_delta_v8_20250311.log
