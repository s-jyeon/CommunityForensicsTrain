#!/bin/bash
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128
HF_MODEL_REPO="OwensLab/commfor-model-384"
CKPT_PATH="trained_model/20260130-032827/commfor_train_best.pt"

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949


torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d train.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --epochs 50 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --hf_model_repo $HF_MODEL_REPO \
    --huggingface_train_repo "" \
    --additional_train_data "../train_data" \
    --additional_data_label_format "real:0,fake:1" \
    --ckpt_path "$CKPT_PATH" \
    --verbose 2
