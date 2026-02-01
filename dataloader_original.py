# dataloader.py (통합 수정본)
import torch
import pandas as pd
import numpy as np
import os
import random
import logging
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from custom_sampler import DistributedEvalSampler
import custom_transforms as ctrans
from functools import partial
import utils as ut

# ================================
# 지원 파일 확장자
# ================================
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# ================================
# 유틸: 프레임 균등 선택
# ================================
def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)

# ================================
# Dataset: Folder 기반 (이미지 + 비디오)
# ================================
class dataset_folder_based(Dataset):
    def __init__(self, args, dir, labels="real:0,fake:1", logger: logging.Logger=None, dtype=torch.float32):
        super().__init__()
        self.args = args
        self.dir = dir
        self.labels = self.parse_labels(labels)
        self.logger = logger if logger is not None else ut.logger
        self.dtype = dtype
        self.df = self.get_index(dir)

    def parse_labels(self, labels):
        return {k: int(v) for k, v in (x.split(':') for x in labels.split(','))}

    def get_label_int(self, label):
        if label not in self.labels:
            raise ValueError(f"Label {label} not found in {self.labels}")
        return self.labels[label]

    def get_index(self, dir):
        index_path = os.path.join(dir, 'index.csv')
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
            if self.args.rank == 0:
                self.logger.info(f"Loaded index file from {index_path}")
        else:
            df = self.index_directory(dir)
        return df

    # def index_directory(self, dir, report_every=1000):
    #     cols = ['ImagePath', 'Label', 'GeneratorName', 'FrameIdx']
    #     temp_dfs = []
    #     for root, dirs, files in os.walk(dir):
    #         for file in files:
    #             f_lower = file.lower()
    #             if not (any(f_lower.endswith(ext) for ext in IMAGE_EXTS) or
    #                     any(f_lower.endswith(ext) for ext in VIDEO_EXTS)):
    #                 continue
    #             generator_name = os.path.basename(os.path.dirname(root))
    #             label_name = os.path.basename(root)
    #             label_int = self.get_label_int(label_name)
    #             path = os.path.join(root, file)
    #             temp_dfs.append(pd.DataFrame([[path, label_int, generator_name, -1]], columns=cols))
    #             if len(temp_dfs) % report_every == 0 and self.args.rank==0:
    #                 print(f"\rIndexed {len(temp_dfs)} samples...", end='', flush=True)
    #     if not temp_dfs:
    #         raise RuntimeError(f"No valid images/videos found in {dir}")
    #     df = pd.concat(temp_dfs, ignore_index=True)
    #     df = df.sort_values(by=['GeneratorName','Label','ImagePath','FrameIdx']).reset_index(drop=True)
    #     df.to_csv(os.path.join(dir,'index.csv'), index=False)
    #     self.logger.info(f"Indexed directory {dir}")
    #     return df
    def index_directory(self, dir, report_every=1000):
        cols = ['ImagePath', 'Label', 'GeneratorName', 'FrameIdx']
        temp_dfs = []
    
        num_frames = self.args.num_frames  # 예: 8, 16 등
    
        for root, dirs, files in os.walk(dir):
            for file in files:
                f_lower = file.lower()
                if not (any(f_lower.endswith(ext) for ext in IMAGE_EXTS) or
                        any(f_lower.endswith(ext) for ext in VIDEO_EXTS)):
                    continue
    
                generator_name = os.path.basename(os.path.dirname(root))
                label_name = os.path.basename(root)
                label_int = self.get_label_int(label_name)
                path = os.path.join(root, file)
    
                if any(f_lower.endswith(ext) for ext in IMAGE_EXTS):
                    # 이미지: FrameIdx = -1 (프레임 개념 없음)
                    temp_dfs.append(pd.DataFrame([[path, label_int, generator_name, -1]], columns=cols))
                else:
                    # 비디오: num_frames개 프레임을 각각 샘플로 취급
                    for frame_idx in range(num_frames):
                        temp_dfs.append(pd.DataFrame([[path, label_int, generator_name, frame_idx]], columns=cols))
    
                if len(temp_dfs) % report_every == 0 and self.args.rank==0:
                    print(f"\rIndexed {len(temp_dfs)} samples...", end='', flush=True)
    
        if not temp_dfs:
            raise RuntimeError(f"No valid images/videos found in {dir}")
    
        df = pd.concat(temp_dfs, ignore_index=True)
        df = df.sort_values(by=['GeneratorName','Label','ImagePath','FrameIdx']).reset_index(drop=True)
        df.to_csv(os.path.join(dir,'index.csv'), index=False)
        self.logger.info(f"Indexed directory {dir}")
        return df

    def load_video(self, video_path, num_frames=None):
        num_frames = num_frames or self.args.num_frames
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return [Image.new('RGB', (224, 224)) for _ in range(num_frames)]
            indices = uniform_frame_indices(total_frames, num_frames)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret: continue
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
            if not frames:
                return [Image.new('RGB', (224, 224)) for _ in range(num_frames)]
            return frames
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return [Image.new('RGB', (224, 224)) for _ in range(num_frames)]

    # 추가함
    def load_video_frame(self, video_path, frame_idx):
        """
        비디오에서 frame_idx 위치의 프레임 한 장만 읽어서 PIL.Image 반환.
        frame_idx는 0 ~ num_frames-1 범위의 '가상 인덱스'이고,
        실제 비디오 길이에 맞춰 균등하게 매핑합니다.
        """
        num_frames = self.args.num_frames
    
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
            if total_frames <= 0:
                cap.release()
                return Image.new('RGB', (224, 224))
    
            # num_frames등분한 위치 중 frame_idx번째 프레임
            indices = uniform_frame_indices(total_frames, num_frames)
            # 안전하게 인덱스 범위 체크
            frame_pos = int(indices[min(frame_idx, len(indices) - 1)])
    
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            cap.release()
    
            if not ret:
                return Image.new('RGB', (224, 224))
    
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return img
    
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return Image.new('RGB', (224, 224))

    # def __getitem__(self, index):
    #     path = self.df.iloc[index]['ImagePath']
    #     label = int(self.df.iloc[index]['Label'])
    #     generator_name = self.df.iloc[index]['GeneratorName']

    #     if path.lower().endswith(tuple(VIDEO_EXTS)):
    #         frames = self.load_video(path)
    #         data = random.choice(frames)  # 랜덤 1프레임 선택
    #     else:
    #         data = Image.open(path).convert("RGB")
    #     return data, label, generator_name
    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['ImagePath']
        label = int(row['Label'])
        generator_name = row['GeneratorName']
        frame_idx = int(row['FrameIdx'])
    
        if path.lower().endswith(tuple(VIDEO_EXTS)):
            if frame_idx >= 0:
                # 인덱스에서 지정한 프레임 하나만 사용
                data = self.load_video_frame(path, frame_idx)
            else:
                # 혹시 FrameIdx가 -1로 들어온 비디오가 있다면, 기존 방식 fallback
                frames = self.load_video(path)
                data = frames[0]  # 또는 가운데 프레임 등으로 고정
        else:
            data = Image.open(path).convert("RGB")
    
        return data, label, generator_name

    def __len__(self):
        return len(self.df)

# ================================
# Transform wrapper: 항상 4D 반환
# ================================
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        data, lab, gen = self.subset[index]
        if self.transform:
            if isinstance(data, torch.Tensor):
                data = F.to_pil_image(data)
            data = self.transform(data)
            data = data.unsqueeze(0)  # (1, C, H, W)
        return data, lab, gen

# ================================
# Transform getter
# ================================
def determine_resize_crop_sizes(args):
    if args.input_size==224:
        return 256, 224
    elif args.input_size==384:
        return 440, 384

def get_transform(args, mode='train', dtype=torch.float32):
    resize, crop = determine_resize_crop_sizes(args)
    norm_mean, norm_std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    aug_list = [transforms.Resize(resize)]
    if mode=='train':
        if args.rsa_ops != '':
            aug_list.append(ctrans.RandomStateAugmentation(resize_size=resize, crop_size=crop,
                                                           auglist=args.rsa_ops,
                                                           min_augs=args.rsa_min_num_ops,
                                                           max_augs=args.rsa_max_num_ops))
        aug_list.append(transforms.RandomCrop(crop))
    else:
        aug_list.append(transforms.CenterCrop(crop))
    aug_list.extend([ctrans.ToTensor_range(0,1),
                     transforms.Normalize(norm_mean,norm_std),
                     transforms.ConvertImageDtype(dtype)])
    return transforms.Compose(aug_list)

# ================================
# Seed utilities
# ================================
def set_seeds_for_data(seed=11997733):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_seeds_for_worker(seed=11997733, id=0):
    seed = seed % (2**31)
    random.seed(seed+id)
    np.random.seed(seed+id)

def set_seeds_and_report(report=True, id=0):
    workerseed = torch.utils.data.get_worker_info().seed
    set_seeds_for_worker(workerseed, id)

def get_seedftn_and_generator(args, seed=None):
    rank = args.rank
    if seed is not None:
        seedftn = partial(set_seeds_and_report, False)
        seed_generator = torch.Generator(device='cpu')
        seed_generator.manual_seed(seed+rank)
    else:
        seedftn = None
        seed_generator = None
        seed = random.randint(0, 1000000000)
    return seedftn, seed_generator, seed

# ================================
# Train/Validation Dataloader
# ================================
def get_train_dataloaders(args, additional_data_path='', batch_size=32, num_workers=4, val_frac=0.01, logger=None, seed=None, **kwargs):
    logger = logger or ut.logger
    dataset = dataset_folder_based(args, additional_data_path)
    
    rank, world_size = args.rank, args.world_size
    seedftn, seed_generator, seed = get_seedftn_and_generator(args, seed)
    
    set_seeds_for_data(seed)
    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_ds = SubsetWithTransform(train_ds, transform=get_transform(args, 'train'))
    val_ds = SubsetWithTransform(val_ds, transform=get_transform(args, 'val'))
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedEvalSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, sampler=train_sampler, worker_init_fn=seedftn, generator=seed_generator)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True, sampler=val_sampler)
    
    if rank == 0:
        logger.info(f"Train/Val split: {len(train_ds)}/{len(val_ds)}")
    
    return trainloader, valloader

def get_test_dataloader(args, additional_data_path='', batch_size=32, num_workers=4, logger=None):
    logger = logger or ut.logger
    dataset = dataset_folder_based(args, additional_data_path)
    dataset = SubsetWithTransform(dataset, transform=get_transform(args, 'val'))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger.info(f"Test set size: {len(dataset)}")
    return loader
