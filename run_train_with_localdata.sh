#!/bin/bash

# --- 환경 변수 설정 ---
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949

# --- 학습 실행 ---
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d train.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --epochs 1 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_frac 0.2 \
    --use_amp \
    --huggingface_train_repo "" \
    --hf_split_train "train" \
    --num_frames 10 \
    --additional_train_data "train_data" \
    --additional_data_label_format "real:0,fake:1" \
    --verbose 2
