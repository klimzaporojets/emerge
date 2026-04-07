#!/bin/bash
#SBATCH --job-name=s05_v9_8b_llama
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:59:00
#SBATCH --partition=gpu_h100
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --output=log_s05_generate_dataset_with_llm_v9_%A.out


config_path=$1
llm_assessor_name=$2
#wait_for_restart_time=$3


source /home/user/.bashrc
cd /path/to/emerge
git pull

conda activate emerge
#   --env "HF_HOME=/hf_cache" \
  #  --env "TRANSFORMERS_CACHE=/hf_cache" \
apptainer instance run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "CUDA_VISIBLE_DEVICES=0" \
  --env "HF_HOME=/hf_cache" \
  --env "TRANSFORMERS_CACHE=/hf_cache" \
  --bind $HF_SCRATCH_CACHE:/hf_cache \
  --nv /path/to/storage/wikipedia-processing/tgi-docker/text-generation-inference_3.3.4.sif \
  instance_llama_8b \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --huggingface-hub-cache /hf_cache \
  --port 8080 \
  --hostname 127.0.0.1 \
  --max-concurrent-requests 128 \
  --max-batch-size 64 \
  --waiting-served-ratio 0.3

sleep 120
# Meta-Llama-3.1-8B
export PYTHONPATH="$PWD/src"
python -u -m dataset.emerge.s05_generate_dataset_with_llm_v8 \
        --config_file $config_path \
        --api_llm_port 8080 \
        --shuffle_input_file \
        --llm_assessor_name $llm_assessor_name \
        --wait_for_restart_time 60 \
        --max_workers 128
