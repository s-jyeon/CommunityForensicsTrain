import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import os, sys
import logging
import random
import utils as ut
import models
import wandb
import gc

from torch.nn.parallel import DistributedDataParallel as DDP

logger: logging.Logger = ut.logger

def train(
        args=None,
        logger=None,
):
    # initialize distributed training
    rank, local_rank, world_size = ut.dist_setup()

    # Set random seed
    torch.manual_seed(args.seed+rank)
    np.random.seed(args.seed+rank)
    random.seed(args.seed+rank)

    # Set device
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size

    use_wandb = (rank == 0) and ("WANDB_API_KEY" in os.environ)


    # Load data
    trainLoader, valLoader = ut.get_dataloader(args, mode='train')
    args = ut.get_epochs_for_itrs(args, len(trainLoader))
    trainLoaderLen = len(trainLoader)

    # Load model
    try:
        model = models.ViTClassifier(
            model_size=args.model_size,
            input_size=args.input_size,
            patch_size=args.patch_size,
            freeze_backbone=args.freeze_backbone,
            device=local_rank, dtype=torch.float32).to(args.local_rank)
    except Exception as e:
        logger.error(f"Error loading model. rank={rank}: {e}")
        sys.exit(1)

    # Set optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.GradScaler(device=local_rank, enabled=args.use_amp)

    # Set scheduler
    if args.warmup_frac > 0:
        warmup_steps=round(args.warmup_frac*ut.get_total_itrs(args, trainLoaderLen))
    else:
        warmup_steps = round(args.warmup_epochs * trainLoaderLen)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*trainLoaderLen-warmup_steps, eta_min=ut.get_min_lr(args)) # set min_lr = lr if args.no_lr_schedule.

    # Set loss function
    criterion = nn.BCEWithLogitsLoss()

    # Load checkpoint if set
    if args.ckpt_path != '': # note that if this is set, it overrides the `--hf_model_repo` argument.
        if args.only_load_model_weights:
            model = ut.load_only_weights(model, args.ckpt_path, rank)
            epoch_start = 0
            total_itr = 0
        else:
            model, optimizer, scheduler, epoch_start, total_itr = ut.load_checkpoint(model, optimizer, scheduler, scaler, args.ckpt_path, rank)
            epoch_start = epoch_start+1 # Since it saves current epoch for ckpt, not next.
    elif args.hf_model_repo != '':
        model = ut.load_ckpt_from_huggingface(model, args.hf_model_repo, rank)
        epoch_start = 0
        total_itr = 0
    else:
        epoch_start = 0
        total_itr = 0

    # try compiling the model
    try:
        model = torch.compile(model, dynamic=True)
    except Exception as e:
        logger.error(f"Error compiling model. rank={rank}: {e}")
        #sys.exit(1)

    # Set DistributedDataParallel
    model = DDP(model, device_ids=[local_rank]) #DDP
    torch.cuda.empty_cache()
    dist.barrier()
    logger.info(f"Model loaded and DDP set. rank={rank}")
    best_val_ap = -1.0

    # ===== WandB init (rank 0 only) =====
    if use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", getattr(args, "wandb_project", None)),
            entity=os.environ.get("WANDB_ENTITY", getattr(args, "wandb_entity", None)),
            name=os.environ.get("WANDB_RUN_NAME", getattr(args, "run_name", None)),
            config=vars(args),
        )
        logger.info("WandB initialized")


    # Train
    local_window_loss=ut.LocalWindow(100)
# Train 루프 시작
    for epoch in range(epoch_start, args.epochs):
        gc.collect() 
        avgTrainLoss, total_itr = ut.train_one_epoch(
            args=args, epoch=epoch, model=model, train_loader=trainLoader,
            optimizer=optimizer, scheduler=scheduler, criterion=criterion,
            scaler=scaler, local_window_loss=local_window_loss,
            warmup_steps=warmup_steps, rank=rank, itr=total_itr,
        )

        if valLoader is not None:
            valLoss, valAcc, valAP = ut.evaluate_one_epoch(
                args=args, epoch=epoch, model=model, dataloader=valLoader,
                criterion=criterion, rank=rank, evalName="Val",
                separate_eval=False, add_sigmoid=(not args.dont_add_sigmoid),
            )
            wandb_log_dict = {
                "epoch": epoch+1, 
                "Loss/Train": avgTrainLoss, 
                "Loss/Val": valLoss, 
                "Acc/Val": valAcc, 
                "AP/Val": valAP
            }

            # ⭐ BEST MODEL SAVE: DDP wrapper 제거 후 저장
            if rank <= 0 and valAP > best_val_ap:
                best_val_ap = valAP
                # DDP라면 .module을 통해 원본 모델 접근
                model_to_save = model.module if hasattr(model, 'module') else model
                # torch.compile 사용 시 ._orig_mod 접근이 필요할 수 있음
                if hasattr(model_to_save, '_orig_mod'):
                    model_to_save = model_to_save._orig_mod
                
                torch.save(model_to_save.state_dict(), args.save_path.replace(".pt", "_best.pt"))
                logger.info(f"New Best Model Saved! Val AP: {best_val_ap:.4f}")

        else:
            valLoss, valAcc, valAP = -1, -1, -1
            wandb_log_dict = {"epoch": epoch+1, "Loss/Train": avgTrainLoss}

        # 1. wandb.log는 에포크마다 수행 (commit=True가 기본값)
        if use_wandb:
            wandb.log(wandb_log_dict)


        # 2. wandb.finish()는 여기서 삭제 (루프 밖으로 이동)
        gc.collect() 

    # --- 에포크 루프 종료 ---

    # 최종 모델 및 체크포인트 저장 (Rank 0만 수행)
    if rank <= 0:
        # DDP/Compile 언래핑
        final_model = model.module if hasattr(model, 'module') else model
        if hasattr(final_model, '_orig_mod'):
            final_model = final_model._orig_mod
            
        # 1. 최종 가중치 저장 (Inference용)
        torch.save(final_model.state_dict(), args.save_path)
        
        # 2. 체크포인트 저장 (학습 재개용 - optimizer 등 포함)
        # utils.py의 save_checkpoint 내부에서 이미 module 체크를 하므로 model 전달
        ckpt_path = args.save_path.replace('.pt', '_final_ckpt.pt')
        ut.save_checkpoint(model, optimizer, scheduler, scaler, epoch, total_itr, ckpt_path)
        
        if args.ckpt_keep_count > 0:
            ut.keep_only_topn_checkpoints(ckpt_path, args.ckpt_keep_count)
        
        logger.info(f"Final training results saved to {args.save_path}")

        # 3. WandB 종료
        if use_wandb:
            wandb.finish()


def main():
    args = ut.parse_args()
    args.random_port_offset = np.random.randint(-1000,1000) # randomize to avoid port conflict in same device
    
    if args.debug_port > 0:
        import debugpy
        debugpy.listen(('localhost', args.debug_port))
        logger.info(f"Waiting for debugger to attach on port {args.debug_port}...")
        debugpy.wait_for_client()
        debugpy.breakpoint()

    if args.gpus_list != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_list
        logger.info(f"Setting CUDA_VISIBLE_DEVICES to {args.gpus_list}.")
        args.gpus = len(args.gpus_list.split(','))

    assert args.gpus <= torch.cuda.device_count(), f'Not enough GPUs! {torch.cuda.device_count()} available, {args.gpus} required.'
    assert args.gpus > 0, f'Number of GPUs must be greater than 0!'
    assert args.cpus_per_gpu > 0, f'Number of CPUs per GPU must be greater than 0!'

    if args.ckpt_save_path == '':
        args.ckpt_save_path = args.save_path

    logger.info(f"Spawning processes on {args.gpus} GPUs.")
    logger.info(f"Verbosity: {args.verbose} (0: None, 1: Every epoch, 2: Every iteration)")

    logger.info(f"Model save name: {os.path.basename(args.save_path)}")

    train(
        args=args,
        logger=logger,
    )

if __name__ == "__main__":
    main()