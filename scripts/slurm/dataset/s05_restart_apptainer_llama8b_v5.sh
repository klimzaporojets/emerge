#!/bin/bash
#############
# Count the number of instances of this script running (excluding the grep process itself)
#SCRIPT_NAME=$(basename "$0")
#RUNNING_INSTANCES=$(pgrep -fc "$SCRIPT_NAME")
#echo "Running instances of ${SCRIPT_NAME} is ${RUNNING_INSTANCES}"
## If more than 1 instance is found, exit
#if [ "$RUNNING_INSTANCES" -gt 1 ]; then
#    echo "Script_already_running_exiting $SCRIPT_NAME is already running. Exiting. $RUNNING_INSTANCES"
#    exit 1
#fi
##############
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

#is_ports=0
#for arg in "$@"; do
#    if [ "$arg" == "--" ]; then
#        if [ $is_ports -eq 0 ]; then
#            is_ports=1
#        fi
#    elif [ $is_ports -eq 0 ]; then
#        devices+=($arg)
#    elif [ $is_ports -eq 1 ]; then
#        ports+=($arg)
#    fi
#done
#echo "The passed ports are: $ports , the passed devices are: $devices , the passed other_params are $other_params"


## Ensure the number of devices matches the number of ports
#if [ "${#devices[@]}" -ne "${#ports[@]}" ]; then
#    echo "Error: The number of devices and ports must match."
#    exit 1
#fi


mkdir logs/$SLURM_JOB_ID

# Iterate over devices and ports
#for ((i = 0; i < ${#devices[@]}; i++)); do
#    device="${devices[i]}"
#    port="${ports[i]}"
#echo "stopping_instance_${device} apptainer "

apptainer instance stop instance_llama_8b
echo "sleeping_after_stopping_instance_llama_8b apptainer"

sleep 30

echo "rerunning_apptainer on CUDA device 0 with port 8080 "

#    CUDA_VISIBLE_DEVICES=$device apptainer instance run --env "HF_TOKEN=$HF_TOKEN" --bind $HF_SCRATCH_CACHE --nv text-generation-inference_2.4.0.sif instance_$device --model-id meta-llama/Meta-Llama-3.1-8B-Instruct --huggingface-hub-cache $HF_SCRATCH_CACHE --hostname 127.0.0.1 --port $port > logs/$SLURM_JOB_ID/tgi_output_$SLURM_JOB_ID_$port_$device.log 2>&1 &


#    CUDA_VISIBLE_DEVICES=$device apptainer instance run \
#      --env "HF_TOKEN=$HF_TOKEN" \
#      --env "CUDA_VISIBLE_DEVICES=$device" \
#      --env "HF_HOME=/hf_cache" \
#      --env "TRANSFORMERS_CACHE=/hf_cache" \
#      --bind $HF_SCRATCH_CACHE:/hf_cache \
#      --nv /path/to/storage/wiki-temp/wikipedia-temp/tgi-docker/text-generation-inference_3.3.4.sif \
#      instance_llama_8b \
#      --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
#      --huggingface-hub-cache /hf_cache \
#      --port 8080 \
#      --hostname 127.0.0.1 \
#      --max-concurrent-requests 64 \
#      --max-batch-size 64 \
#      --cuda-memory-fraction 1.0 \
#      > logs/tgi_${SLURM_JOB_ID}_llama8b_output_8080.log 2>&1 &
timestamp=$(date +%s)

#   --env "HF_HOME=/hf_cache" \
  #  --env "TRANSFORMERS_CACHE=/hf_cache" \
#   --env "HF_HOME=$HF_HOME" \
  #  --env "TRANSFORMERS_CACHE=$HF_HOME" \
apptainer instance run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "CUDA_VISIBLE_DEVICES=0" \
  --env "HF_HOME=/hf_cache" \
  --env "TRANSFORMERS_CACHE=/hf_cache" \
  --bind $HF_SCRATCH_CACHE:/hf_cache \
  --nv /path/to/storage/wiki-temp/wikipedia-temp/tgi-docker/text-generation-inference_3.3.4.sif \
  instance_llama_8b \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --huggingface-hub-cache /hf_cache \
  --port 8080 \
  --hostname 127.0.0.1 \
  --max-concurrent-requests 128 \
  --max-batch-size 64 \
  --waiting-served-ratio 1.0 2>&1 | tee logs/tgi_llama8b_${SLURM_JOB_ID}_output_8080_restarting_${timestamp}.log

echo "sleeping again on CUDA device 0 with port 8080 "
sleep 240

#done
