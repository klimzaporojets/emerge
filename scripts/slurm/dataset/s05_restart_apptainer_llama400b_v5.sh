#!/bin/bash

# Count the number of instances of this script running (excluding the grep process itself)
# Get full path of the script
SCRIPT_PATH=$(realpath "$0")

# Count running instances excluding the current process
RUNNING_INSTANCES=$(pgrep -f "$SCRIPT_PATH" | grep -vw $$ | wc -l)

echo "Running instances of ${SCRIPT_PATH}: $RUNNING_INSTANCES"
if [ "$RUNNING_INSTANCES" -gt 0 ]; then
    echo "script_already_running ${SCRIPT_PATH}: $RUNNING_INSTANCES. Exiting."
    exit 1
fi

echo "Script_inside_to_be_run_continuing."

# Generate a random number between 1000 and 10000 milliseconds
RANDOM_MS=$((1000 + RANDOM % 9001))

# Convert milliseconds to seconds (as a floating-point number)
RANDOM_SEC=$(echo "scale=3; $RANDOM_MS / 1000" | bc)

# Wait for the random duration
sleep "$RANDOM_SEC"

echo "Waited for $RANDOM_MS milliseconds."

source /home/user/.bashrc
cd /path/to/data/wikipedia-processing

#conda activate emerge
mkdir logs/

echo "instance_llama_405b apptainer "
apptainer instance stop instance_llama_405b
echo "sleeping_after_stopping_instance_405b_1 apptainer"

sleep 120

echo "rerunning_apptainer for instance instance_405b_1"
#timestamp=$(date +"%Y-%m-%d %H:%M:%S")
timestamp=$(date +%s)

# --env "CUDA_LAUNCH_BLOCKING=1" \
#   --env "HF_HOME=/hf_cache" \
  #  --env "TRANSFORMERS_CACHE=/hf_cache" \
#   --env "HF_HOME=$HF_HOME" \
  #  --env "TRANSFORMERS_CACHE=$HF_HOME" \
apptainer instance run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
  --env "HF_HOME=/hf_cache" \
  --env "TRANSFORMERS_CACHE=/hf_cache" \
  --bind $HF_SCRATCH_CACHE:/hf_cache \
  --nv /path/to/storage/wiki-temp/wikipedia-temp/tgi-docker/text-generation-inference_3.3.4.sif \
  instance_llama_405b \
  --model-id hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4 \
  --huggingface-hub-cache /hf_cache \
  --port 8083 \
  --hostname 127.0.0.2 \
  --max-concurrent-requests 128 \
  --max-batch-size 64 \
  --num-shard 4 \
  --waiting-served-ratio 1.0 2>&1 | tee logs/tgi_llama_405b_${SLURM_JOB_ID}_output_8080_restarting_${timestamp}.log

#   --max-batch-total-tokens 771072 \
#  --max-total-tokens 12048 \
#  --max-input-tokens 10000 \

echo "sleeping again, waiting for instance_405b_1 to load"
sleep 900
