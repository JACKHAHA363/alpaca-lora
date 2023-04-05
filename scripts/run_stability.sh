#!/usr/bin/bash
#SBATCH --account=laion
#SBATCH --partition=g40
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
source /admin/home-luyuchen/fsx/py38/bin/activate
module load cuda

# Official weights
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=4 WORLD_SIZE=4 torchrun --nproc_per_node=4 --master_port=1234 finetune.py \
    --base_model '/admin/home-luyuchen/fsx/llama_hfconverted/7B' \
    --data_path 'alpaca_data_cleaned.json' \
    --output_dir './lora-alpaca' \
    --run_name 'clean_data' \
    --batch_size 256 \
    --micro_batch_size 64 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs \
    --group_by_length
