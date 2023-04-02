#!/bin/bash
#SBATCH --job-name=alpca
#SBATCH --ntasks=1
#SBATCH --mem=100Gb
#SBATCH --gres=gpu:a100l 
#SBATCH --cpus-per-task=8 
#SBATCH --time=12:00:00

source ~/py38/bin/activate
module load cuda/11.2/cudnn/8.1
export TRANSFORMERS_CACHE="/network/scratch/l/luyuchen/pretrained_llm/huggingface"

# Official weights
python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=128

