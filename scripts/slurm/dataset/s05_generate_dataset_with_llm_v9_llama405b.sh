#!/bin/bash
#SBATCH --job-name=s05_v9_405b_llama
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:59:00
#SBATCH --partition=gpu_h100
#SBATCH --mem=100G
#SBATCH --gpus=4
#SBATCH --output=log_s05_generate_dataset_with_llm_v9_405b_%A.out


config_path=$1
llm_assessor_name=$2
#wait_for_restart_time=$3

source /home/user/.bashrc
cd /path/to/emerge
git pull

conda activate emerge

#   --max-total-tokens 12048 \
  #  --max-input-tokens 10000 \

#apptainer instance run --env "HF_TOKEN=$HF_TOKEN" --env "CUDA_VISIBLE_DEVICES=0,1,2,3" --bind $HF_SCRATCH_CACHE --nv text-generation-inference_2.4.0.sif instance_405b_1 --model-id hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4 --huggingface-hub-cache $HF_SCRATCH_CACHE --hostname 127.0.0.1 --port 8080 > logs/tgi_${SLURM_JOB_ID}_output_8080.log 2>&1 &

# --env "CUDA_LAUNCH_BLOCKING=1" \
# --env HF_HOME=$HF_HOME --env TRANSFORMERS_CACHE=$HF_HOME

#  --env "HF_HOME=/hf_cache" \
#  --env "TRANSFORMERS_CACHE=/hf_cache" \
#  --env "HF_HOME=$HF_HOME" \
#  --env "TRANSFORMERS_CACHE=$HF_HOME" \
#   --max-concurrent-requests 128 \
  #  --max-batch-size 50 \
apptainer instance run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
  --env "HF_HOME=/hf_cache" \
  --env "TRANSFORMERS_CACHE=/hf_cache" \
  --bind $HF_SCRATCH_CACHE:/hf_cache \
  --nv /path/to/storage/wikipedia-processing/tgi-docker/text-generation-inference_3.3.4.sif \
  instance_llama_405b \
  --model-id hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4 \
  --huggingface-hub-cache /hf_cache \
  --port 8083 \
  --hostname 127.0.0.2 \
  --max-concurrent-requests 128 \
  --max-batch-size 64 \
  --num-shard 4 \
  --waiting-served-ratio 1.0

#  --max-batch-total-tokens 771072 \

sleep 900

export PYTHONPATH="$PWD/src"
python -u -m dataset.emerge.s05_generate_dataset_with_llm_v8 \
        --config_file $config_path \
        --api_llm_port 8083 \
        --shuffle_input_file \
        --llm_assessor_name $llm_assessor_name \
        --max_workers 128 \
        --wait_for_restart_time 1050
#        \
#        --restart_apptainer_script_name s05_restart_apptainer_llama400b_v5.sh
