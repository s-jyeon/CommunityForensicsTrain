#!/bin/bash
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128
HF_MODEL_REPO="OwensLab/commfor-model-384"
CKPT_PATH="./trained_model/20260201-064944/commfor_train_best.pt"
export PATH=/workspace/.venv/bin:$PATH
export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949
export WANDB_API_KEY="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"
export WANDB_PROJECT="dacon_hecto_deepfake"
export WANDB_ENTITY="yjneon339-kyonggi-university"          # 본인 wandb 계정/팀 slug
export WANDB_RUN_NAME="commfor_train_with_localdata_and_hf"

# TorchDynamo 비활성화
export TORCHDYNAMO_DISABLE=1

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d train.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --epochs 100 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --hf_model_repo $HF_MODEL_REPO \
    --huggingface_train_repo "OwensLab/CommunityForensics-Small" \
    --additional_train_data "../train_data" \
    --additional_data_label_format "real:0,fake:1" \
    --ckpt_path "$CKPT_PATH" \
    --verbose 2
