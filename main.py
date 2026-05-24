"""python ablation_master_controlled_nomixup_consistency.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.3\
    --out ./out_ultimate_nocrmixup \
    --exp_name PALS_softMatch/ablation_master_controlled_nomixup_consistency/CIFAR100/cu1exp_daes_delta0.25_p0.05n0.3_bs256_gs0.8_ema0.999 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 0.8  --no_consistency --no_reliable_mixup  --no_unreliable_mixup"""

"""python ablation_master_controlled_nomixup_consistency.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.03 --nr 0.3\
    --out ./out_ultimate_nocrmixup \
    --exp_name PALS_softMatch/ablation_master_controlled_nomixup_consistency/CIFAR100/cu1exp_daes_delta0.25_p0.03n0.3_bs256_gs0.8_ema0.999 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 0.8  --no_consistency --no_reliable_mixup  --no_unreliable_mixup"""    

"""python ablation_master_controlled_nomixup_consistency.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.03 --nr 0.1\
    --out ./out_ultimate_nocrmixup \
    --exp_name PALS_softMatch/ablation_master_controlled_nomixup_consistency/CIFAR100/cu0exp_daes_delta0.25_p0.03n0.1_bs256_gs0.8_ema0.999 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 0 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 0.8  --no_consistency --no_reliable_mixup  --no_unreliable_mixup"""








# =N============================================================================
#           HYBRID PALS-SSL FRAMEWORK - THREE-PHASE TRAINING
#
# Desc: This version implements a sequential, three-tier training strategy per epoch:
# 1. SimSiam Phase: Train on the lowest-quality samples (remaining set)
#    using a self-supervised, feature-learning objective (SimSiam).
# 2. Hybrid Phase: Train on a mix of reliable (supervised) and high-quality
#    unsupervised (consensus set) samples using Mixup, CE loss, and SoftMatch.
# ==============================================================================
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import logging
from PIL import Image
from torch.amp import autocast, GradScaler
import torch.optim as optim
import wandb
import sys
import pandas as pd # CUB200依赖
import itertools
from torchvision.models import resnet18, resnet50
# 假设您的工具函数在以下路径
from data.dataset import CUB200Partial, CIFAR10Partial, CIFAR100Partial
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy ,ImageNetPolicy
from data.crowdsource import *
# ==============================================================================
#                      Section 0: 环境设置 (Environment Setup)
# ==============================================================================
def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# In Section 0: 环境设置 (Environment Setup)

def setup_logger(log_dir, filename="run.log", is_master=False, to_console=False):
    """
    Modified to allow disabling console output explicitly.
    """
    logger_name = f"logger_{log_dir.replace('/', '_')}_{filename}"
    logger = logging.getLogger(logger_name)
    
    if logger.hasHandlers():
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
            
    logger.setLevel(logging.INFO)
    logger.propagate = False 
    
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)
    
    # File Handler (Always active)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler (Only active if to_console is True)
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Ultimate Hybrid PALS-SSL Framework with Three-Phase Training')
    # 基本设置
    parser.add_argument('--exp_name', type=str, default='HybridPALS_ThreePhase_Run', help='Experiment name.')
    
    # 在 parse_args() 函数中修改：
    parser.add_argument('--dataset', type=str, default='CIFAR100', 
                        choices=['CIFAR10', 'CIFAR100', 'CIFAR100H', 'CUB200', 
                                'Treeversity', 'Benthic', 'Plankton',])
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--out', type=str, default='./out_ultimate', help='Directory for output')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1], help='List of random seeds.')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    # 部分标签 (PLL) 设置
    parser.add_argument('--pr', type=float, default=0.1, help='partial ratio (q)')
    parser.add_argument('--nr', type=float, default=0.0, help='noise ratio (eta)')
    parser.add_argument('--lpi', type=int, default=10, help='Labels Per Image (LPI) for crowdsource NPLL conversion')
    # 核心算法开关
    parser.add_argument('--reliable_selection_mode', type=str, default='pals', choices=['mine', 'pals'], help="Strategy for reliable set selection.")
    
    # 训练超参数
    parser.add_argument('--network', type=str, default='R18', help='Network architecture (R18, R50)')
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step'],
                        help='Type of learning rate scheduler (cosine or step).')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[60, 120, 160, 200],
                        help='Epoch milestones for the step learning rate scheduler.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='Decay rate (gamma) for the step learning rate scheduler.')    
    # 损失函数超参数
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Alpha for Mixup.')
    parser.add_argument('--lsr', type=float, default=0.5, help='Label smoothing rate.')
    parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for consistency loss.')

    # --- 🚀 消融实验开关 (Ablation Study Flags) ---
    parser.add_argument('--no_reliable_mixup', action='store_true', help='[Ablation] Disable Mixup on reliable set.')
    parser.add_argument('--no_rebalance', action='store_true', help='[Ablation] Disable class rebalancing on pseudo-labels.')
    parser.add_argument('--no_softmatch', action='store_true', help='[Ablation] Disable SoftMatch weighting (force weight=1.0).')
    parser.add_argument('--no_unreliable_mixup', action='store_true', help='[Ablation] Disable Mixup on unreliable set (use standard consistency).')
    
    # [新增] 完全禁用不可靠集训练
    parser.add_argument('--no_unreliable_training', action='store_true', help='[Ablation] COMPLETELY ignore unreliable set (Supervised Only).')
    
    # [新增] 消融：禁用 Middleware Rectification
    parser.add_argument('--no_rectify', action='store_true', help='[Ablation] Disable Middleware Gating/Rectification.')
    
    # [新增] 消融：EMA 因子
    parser.add_argument('--ema_alpha', type=float, default=0.999, help='EMA momentum factor (default: 0.999).')
    # ---------------------------------------------

    # KNN & 平衡参数
    parser.add_argument('--k_val', type=int, default=15, help='k for knn')
    parser.add_argument('--knn_iterations', type=int, default=2, help='Number of KNN purification iterations.')
    parser.add_argument('--delta', type=float, default=0.25, help='example selection quantile')
    
    # --- 🚀 [Added for Ablation Master Control] ---
    parser.add_argument('--consensus_power', type=float, default=2.0, help='Power for consensus proportion in dynamic weight (default: 2.0)')
    parser.add_argument('--fix_dynamic_weight', action='store_true', help='[Ablation] Fix dynamic consistency weight to 1.0 (Disable curriculum)')
    # ----------------------------------------------

    # --- 🚀 为方法 A (weighted) 添加这些参数 ---
    parser.add_argument('--start_correct', type=int, default=0, help='(Used by the selection function)')  
    parser.add_argument('--epoch', type=int, default=100, help='(Dummy epoch for selection function)')  
    # --- 添加结束 ---
    parser.add_argument('--sim_mode_1', type=str, default='exp',  # <--- 默认设为 daes
                        choices=['D', 'exp','daes'], # <--- 加入 daes
                        help='Similarity measure for Stage 1')

    parser.add_argument('--sim_mode_2', type=str, default='daes', 
                        choices=['None', 'exp', 'daes'], 
                        help='Similarity measure for Stage 2')    
    
    parser.add_argument('--warmup_epochs', type=int, default=250, help='Epochs for linear LR warmup.')
    # 日志
    parser.add_argument('--detailed_log', action='store_true', help='Enable detailed diagnostic logging.')
    # 1. Teacher Gating 控制 (干预机制)
    parser.add_argument('--gating_start_ratio', type=float, default=0.2, 
                        help='[Gating] Ratio of epochs before teacher gating starts (default: 0.2, means start at 20% epoch).')
    parser.add_argument('--gating_max_alpha', type=float, default=1.0, 
                        help='[Gating] Max influence of teacher (0.0 to 1.0). Set <1.0 to always keep some geometry signal.')

    # 2. DAES 算法控制 (拓扑构建)
    parser.add_argument('--daes_clamp', type=float, default=0.25, 
                        help='[DAES] Max temperature clamp (Anti-oversmoothing lock). Lower is sharper.')
    parser.add_argument('--daes_entropy_weight', type=float, default=0.2, 
                        help='[DAES] Sensitivity to local entropy (tau = base + weight * H).')
    parser.add_argument('--daes_sharpening_power', type=float, default=2.0, 
                        help='[DAES] Sharpening power for local mean calculation (CUB=2.0, Standard=1.0).')

    # 3. KNN 图构建
    parser.add_argument('--knn_heads', type=int, default=4, 
                        help='[KNN] Number of heads for metric learning (Robustness).')# 在 parse_args 函数中添加
    parser.add_argument('--no_consistency', action='store_true', 
                        help='[Ablation] Disable Consistency Reg. Train on WEAK images only.')
    return parser.parse_args()
# (在 Section 2: 数据处理与模型)

def get_pals_transforms(dataset_name):
    # --- (新增) 众包数据集的 Mean/Std ---
    if dataset_name == 'Treeversity':
        mean = [0.4439581940620345, 0.4509297096690951, 0.3691211738638277]
        std = [0.23407518616927706, 0.22764417468550843, 0.2600833107790479]
    elif dataset_name == 'Benthic':
        mean = [0.34728872821176615, 0.40013687864974884, 0.4110478166769647]
        std = [0.1286915489786319, 0.13644626747739305, 0.14258506692263767]
    elif dataset_name == 'Plankton':
        mean = [0.9663359216202008, 0.9663359216202008, 0.9663359216202008]
        std = [0.10069729102981237, 0.10069729102981237, 0.10069729102981237]
    elif dataset_name == 'CUB200':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else: # 默认为 CIFAR
        mean, std = ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]) if '100' in dataset_name else ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    # --- (修改) 扩展 Transform 逻辑 ---
    
    # (新增) CUB200 (使用 PALS 原始的强 Aug)
    if dataset_name == 'CUB200':
        weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            # CIFAR10Policy(), 
            ImageNetPolicy(),
            transforms.ToTensor(), 
            Cutout(n_holes=1, length=56), # <-- 关键！
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        
    elif dataset_name == 'Treeversity':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(int(224/0.875)), # (256)
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset_name == 'Benthic':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((112,112)),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset_name == 'Plankton':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else: # CIFAR
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    # 包含您截图中的所有新数据集
    if dataset_name in ['Turkey', 'Pig', 'MiceBone', 'QualityMRI', 'Synthetic', 
                        'verse_blended-vps', 'verse_mask1-vps', 'CIFAR10H']:
        
        # 使用 ImageNet 统计数据作为通用初始化
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # 如果是 CIFAR10H，可能图片很小 (32x32)，需要特殊处理
        if 'CIFAR' in dataset_name or 'Synthetic' in dataset_name:
             resize_size = 32
             crop_size = 32
        else:
             # 其他真实世界数据集 (Turkey, Pig等) 使用标准 224
             resize_size = 256
             crop_size = 224

        weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(), # 强增强
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return weak_transform, strong_transform, test_transform
def get_base_encoder(network_name, dataset_name):
    
    
    # --- (修改) 扩展使用预训练权重的条件 ---
    # use_pretrained = dataset_name in ['CUB200', 'Treeversity', 'Benthic', 'Plankton','Synthetic',]
    use_pretrained = dataset_name in ['CUB200', 'Treeversity', 'Benthic', 'Plankton',]
    
    if network_name == 'R50':
        base_model = resnet50(weights='IMAGENET1K_V1' if use_pretrained else None)
    else: # Default to R18
        base_model = resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)

    feature_dim = base_model.fc.in_features
    
    # if 'CIFAR' in dataset_name:
    if 'CIFAR' in dataset_name or 'Synthetic' in dataset_name:
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        
    encoder = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten())
    return encoder, feature_dim


## --- 🚀 MODIFICATION START: Removed SimSiam class --- ##
# class SimSiam(nn.Module): ... (Class Removed)
# def simsiam_loss_fn(p, z): ... (Function Removed)
## --- MODIFICATION END --- ##


class FeatureExtractionDataset(Dataset):
    def __init__(self, base_dataset, transform): 
        self.base_dataset, self.transform = base_dataset, transform
        
        # --- (修改) ---
        # 我们需要明确区分 CUB200 和 Crowdsource
        self.is_cub = isinstance(self.base_dataset, CUB200Partial)
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
        # --- (修改结束) ---

    def __len__(self): 
        return len(self.base_dataset)
        
    def __getitem__(self, index):
        # 1. 获取原始图像
        
        # --- (修改) ---
        if self.is_cub:
            # CUB200: .data 是 DataFrame. 必须用 .data_paths
            # (这是在破坏“封装”，但这是在你设定的约束下唯一可行的方法)
            img_path = os.path.join(self.base_dataset.root, 
                                    self.base_dataset.base_folder, 
                                    'images', 
                                    self.base_dataset.data_paths[index])
            img = Image.open(img_path).convert('RGB')
        
        elif self.is_crowd:
            # Crowdsource: .data 是 'list' of paths, 可以直接用 [index]
            img_path = self.base_dataset.data[index]
            img = Image.open(img_path).convert('RGB')

        else: 
            # CIFAR: .data 是 numpy 数组
            img = Image.fromarray(self.base_dataset.data[index])
        # --- (修改结束) ---
            
        # 2. 应用 *这个类* 自己的 transform (即 test_t)
        return self.transform(img), index
class SSLReadyDataset(Dataset):
    def __init__(self, base_dataset, indices, labels, weak_t, strong_t):
        self.base_dataset, self.indices, self.labels = base_dataset, indices, labels
        self.weak_t, self.strong_t = weak_t, strong_t
        
        # --- (修改) ---
        # 我们需要明确区分 CUB200 和 Crowdsource
        self.is_cub = isinstance(self.base_dataset, CUB200Partial)
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
        # --- (修改结束) ---

    def __len__(self): 
        return len(self.indices)
        
    def __getitem__(self, item_idx):
        original_idx, label = self.indices[item_idx], self.labels[item_idx]
        
        # 1. 获取原始图像
        
        # --- (修改) ---
        if self.is_cub:
            # CUB200: .data 是 DataFrame. 必须用 .data_paths
            img_path = os.path.join(self.base_dataset.root, 
                                    self.base_dataset.base_folder, 
                                    'images', 
                                    self.base_dataset.data_paths[original_idx])
            img = Image.open(img_path).convert('RGB')

        elif self.is_crowd:
            # Crowdsource: .data 是 'list' of paths
            img_path = self.base_dataset.data[original_idx]
            img = Image.open(img_path).convert('RGB')

        else: 
            # CIFAR: .data 是 numpy 数组
            img = Image.fromarray(self.base_dataset.data[original_idx])
        # --- (修改结束) ---
            
        # 2. 应用 *这个类* 自己的 transform (即 weak_t 和 strong_t)
        return self.weak_t(img), self.strong_t(img), label, original_idx    
class HybridDataset(Dataset):
    def __init__(self, base_dataset, data_list, weak_t, strong_t): self.base_dataset, self.data_list, self.weak_t, self.strong_t = base_dataset, data_list, weak_t, strong_t
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        original_idx, label, is_reliable = self.data_list[idx]
        if isinstance(self.base_dataset, CUB200Partial): img, _, _ = self.base_dataset[original_idx]
        else: img = Image.fromarray(self.base_dataset.data[original_idx])
        return self.weak_t(img), self.strong_t(img), label, is_reliable, original_idx

# ==============================================================================
#                      Section 3: 核心算法辅助
# ==============================================================================
@torch.no_grad()
def get_features(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval(); all_features, all_predictions, all_indices = [], [], []
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        with autocast('cuda'):
            features = encoder(images)
            predictions = F.softmax(classifier(features), dim=1)
        all_features.append(F.normalize(features.float())); all_predictions.append(predictions.float()); all_indices.append(indices.cpu())
    all_features, all_predictions, all_indices = torch.cat(all_features), torch.cat(all_predictions), torch.cat(all_indices)
    return all_features[torch.argsort(all_indices)], all_predictions[torch.argsort(all_indices)]


@torch.no_grad()
def select_reliable_set(pseudo_labels, scores, partial_labels, delta, mode, logger, device):
    logger.info(f"--- Reliable Set Selection (Mode: {mode.upper()}) ---"); num_classes = scores.shape[1]
    agreement_mask = (pseudo_labels.unsqueeze(1) == torch.arange(num_classes, device=device)) & (partial_labels.to(device) == 1)
    num_agree_positive = agreement_mask.sum(dim=0)[agreement_mask.sum(dim=0) > 0]
    if len(num_agree_positive) == 0: logger.warning("  -> No samples for 'm' calculation."); return []
    m = max(1, int(torch.quantile(num_agree_positive.float(), delta).item())); final_pairs, assigned = [], set()
    temp = []
    for c in range(num_classes):
        indices = torch.where(partial_labels[:, c] == 1)[0]
        if len(indices) > 0:
            posteriors = scores[indices, c]; num = min(m, len(posteriors))
            if num > 0: top_p, top_i = torch.topk(posteriors, k=num); temp.extend([(p.item(), indices[i].item(), c) for p, i in zip(top_p, top_i)])
    temp.sort(key=lambda x: x[0], reverse=True)
    for _, idx, label in temp:
        if idx not in assigned: final_pairs.append((idx, label)); assigned.add(idx)
    logger.info(f"  -> Selected {len(final_pairs)} reliable samples (target m per class={m}).")
    return final_pairs

class SoftMatchWeightManager:
    def __init__(self, num_samples, num_classes, n_sigma=2.0, momentum=0.99, device='cuda'): self.n_sigma, self.momentum, self.device = n_sigma, momentum, device; self.prob_model = torch.ones(num_samples, num_classes, device=device) / num_classes
    def __call__(self, preds, index, return_stats=False):
        self.prob_model[index] = self.momentum * self.prob_model[index] + (1 - self.momentum) * preds.detach(); max_probs_model = self.prob_model[index].max(dim=1)[0]; mu = max_probs_model.mean(); std = max_probs_model.std() if max_probs_model.size(0) > 1 else torch.tensor(1e-8, device=self.device); weights = torch.exp(-torch.pow(F.relu(mu - preds.max(dim=1)[0]), 2) / (2 * self.n_sigma * std**2 + 1e-8))
        return (weights.detach(), mu.item(), std.item()) if return_stats else weights.detach()




# (先添加 SoftMatch 的日志函数)
def log_softmatch_diagnostics(logger, diagnostics, num_hq_samples):
    if not diagnostics['weights']:
        logger.info("  -> [SoftMatch Diagnostics] No HQ-unsupervised samples trained.")
        return

    weights = torch.cat(diagnostics['weights'])
    confidences = torch.cat(diagnostics['confidences'])
    pseudo_labels = torch.cat(diagnostics['pseudo_labels'])
    true_labels = torch.cat(diagnostics['true_labels'])
    
    avg_mu = np.mean(diagnostics['mus'])
    avg_std = np.mean(diagnostics['stds'])
    
    logger.info(f"  -> [SoftMatch Diagnostics] --- Trained on {len(weights)}/{num_hq_samples} HQ samples ---")
    logger.info(f"    - Avg Weight: {weights.mean().item():.4f} (Min: {weights.min().item():.4f}, Max: {weights.max().item():.4f})")
    logger.info(f"    - Gaussian Center (μ): Avg {avg_mu:.4f} | Gaussian Width (σ): Avg {avg_std:.4f}")

    tiers = {"High (w>0.8)": (0.8, 1.1), "Mid (0.5-0.8)": (0.5, 0.8), "Low (w<0.5)": (0.0, 0.5)}
    for name, (lower, upper) in tiers.items():
        mask = (weights >= lower) & (weights < upper)
        num_in_tier = mask.sum().item()
        if num_in_tier == 0:
            continue
        
        pl_acc = (pseudo_labels[mask] == true_labels[mask]).float().mean().item() * 100
        avg_conf = confidences[mask].mean().item()
        avg_w = weights[mask].mean().item()
        logger.info(f"      - {name}: {num_in_tier:<5} samples | PL Acc: {pl_acc:.2f}% | Avg Conf: {avg_conf:.3f} | Avg Weight: {avg_w:.3f}")


def train_unified_loop(args, encoder, classifier, device, 
                       loader_sup, loader_unsup, optimizer, softmatch_manager, 
                       logger, num_classes, knn_pl, model_pl,
                       knn_scores, 
                       dynamic_consistency_weight,
                       rebalance_factor):
    
    # --- 0. 初始化与设置 (Init & Setup) ---
    encoder.train()
    classifier.train()
    scaler = GradScaler()  # 用于混合精度训练
    
    # 将辅助数据（KNN 伪标签、模型伪标签、KNN 分数）移动到 GPU
    knn_pl, model_pl = knn_pl.to(device), model_pl.to(device)
    knn_scores = knn_scores.to(device)

    # --- 1. 数据加载器准备 (Loader Preparation) ---
    # 消融实验：如果开启 --no_unreliable_training，则彻底禁用不可靠集训练
    if args.no_unreliable_training: 
        loader_unsup = None
    
    # 检查是否有可用的加载器
    loaders = [l for l in [loader_sup, loader_unsup] if l is not None]
    if not loaders: return

    # 创建无限迭代器 (Infinite Iterators)，方便在一个 epoch 内遍历完较长的数据集
    iter_sup = iter(itertools.cycle(loader_sup)) if loader_sup else None
    iter_unsup = iter(itertools.cycle(loader_unsup)) if loader_unsup else None
    
    # 计算当前 Epoch 需要迭代的步数（取两个加载器长度的最大值）
    num_batches = max(len(loader_sup) if loader_sup else 0, len(loader_unsup) if loader_unsup else 0)

    # 初始化指标记录器 (Metrics)
    meter = {'sup': 0, 'unsup': 0, 'cnt_sup': 0, 'cnt_unsup': 0}

    # --- ⚡ 辅助函数：通用 Mixup 前向传播 (Generic Mixup Forwarder) ⚡ ---
    # 这个闭包函数处理核心的前向传播逻辑，支持 Mixup 和标准训练
    def forward_with_strategy(images, targets, use_mixup, mixup_alpha, weights=None, force_lambda=None):
        """
        参数说明:
            images: 输入图像张量 (B, C, H, W) -> 可以是弱增强或强增强
            targets: 目标标签张量 (B, C) -> 可以是 One-hot 或 Soft Label
            use_mixup: 布尔值，是否启用 Mixup
            weights: 可选的样本权重 (用于 SoftMatch 加权 Loss)
            force_lambda: 强制指定的 Mixup 系数 (用于非可靠集的特殊逻辑)
        """
        if use_mixup:
            # === 路径 A: Mixup 训练 ===
            
            # 1. 确定 Mixup 系数 (Lambda)
            if force_lambda is not None:
                # 【关键修改】非可靠集逻辑：直接使用 SoftMatch 权重作为 Lambda
                # 注意：SoftMatch 权重形状通常是 (B,)，需要调整维度以匹配输入
                lam = force_lambda
                # 确保维度匹配：(B,) -> (B, 1, 1, 1) 用于图像，(B, 1) 用于标签
                lam_img = lam.view(-1, 1, 1, 1)
                lam_tgt = lam.view(-1, 1)
            else:
                # 基线/可靠集逻辑：从 Beta 分布中随机采样
                lam_val = np.random.beta(mixup_alpha, mixup_alpha)
                lam = torch.tensor(lam_val, device=device).float()
                lam_img = lam
                lam_tgt = lam
            
            # 生成随机排列索引
            perm = torch.randperm(images.size(0), device=device)
            
            # 2. 执行 Mixup 插值
            img_mix = lam_img * images + (1 - lam_img) * images[perm]
            target_mix = lam_tgt * targets + (1 - lam_tgt) * targets[perm]
            
            # 3. 前向传播
            preds = classifier(encoder(img_mix))
            # 计算交叉熵 Loss (Soft Target)
            loss_raw = -torch.sum(target_mix * F.log_softmax(preds, dim=1), dim=1)
            
            # 4. 应用 Loss 权重 (SoftMatch)
            # 如果提供了 weights，则对 Loss 进行加权平均
            if weights is not None:
                loss = (loss_raw * weights).mean()
            else:
                loss = loss_raw.mean()
            return loss

        else:
            # === 路径 B: 标准训练 (无 Mixup) ===
            preds = classifier(encoder(images))
            
            # 检查目标是软标签 (One-hot/Prob) 还是硬标签 (Index)
            if targets.dim() > 1: # 软标签/One-hot
                loss_raw = -torch.sum(targets * F.log_softmax(preds, dim=1), dim=1)
            else: # 硬标签索引
                loss_raw = F.cross_entropy(preds, targets, reduction='none')
                
            if weights is not None:
                loss = (loss_raw * weights).mean()
            else:
                loss = loss_raw.mean()
            return loss

    # ========================== 训练循环 (TRAINING LOOP) ==========================
    for _ in range(num_batches):
        optimizer.zero_grad()
        
        # ---------------------------------------------------------------
        # PART 1: 可靠集 (Reliable Set - Supervised)
        # ---------------------------------------------------------------
        loss_s = torch.tensor(0.0, device=device)
        
        if iter_sup:
            # 获取数据：弱增强(w), 强增强(s), 标签(l)
            sup_w, sup_s, sup_l, _ = next(iter_sup)
            sup_w, sup_s, sup_l = sup_w.to(device), sup_s.to(device), sup_l.to(device)
            B_s = sup_w.size(0)

            # 1.1 视图选择 (View Decision)
            # 消融实验：如果禁用一致性，仅使用弱增强
            if getattr(args, 'no_consistency', False):
                # 仅使用弱增强
                train_imgs_list = [sup_s] 
            else:
                # 【基线逻辑】可靠集同时使用弱增强和强增强
                train_imgs_list = [sup_w, sup_s]

            # 1.2 目标准备 (Target Preparation)
            # 监督学习使用 One-Hot 标签 (应用 Label Smoothing)
            oh_labels = F.one_hot(sup_l, num_classes).float()
            if args.lsr > 0: oh_labels = oh_labels * (1 - args.lsr) + args.lsr / num_classes

            # 1.3 Mixup 决策
            do_mixup = not args.no_reliable_mixup

            with autocast('cuda'):
                # 对列表中的每个视图（Weak/Strong）分别计算 Loss 并取平均
                loss_accum = 0.0
                for img in train_imgs_list:
                    loss_accum += forward_with_strategy(
                        images=img, 
                        targets=oh_labels, 
                        use_mixup=do_mixup, 
                        mixup_alpha=args.mixup_alpha,
                        force_lambda=None # 可靠集使用随机 Mixup
                    )
                loss_s = loss_accum / len(train_imgs_list)

            meter['sup'] += loss_s.item() * B_s
            meter['cnt_sup'] += B_s

        # ---------------------------------------------------------------
        # PART 2: 非可靠集 (Unreliable Set - Semi-Supervised)
        # ---------------------------------------------------------------
        loss_c = torch.tensor(0.0, device=device)

        if iter_unsup:
            unsup_w, unsup_s, _, unsup_idx = next(iter_unsup)
            unsup_w, unsup_s, unsup_idx = unsup_w.to(device), unsup_s.to(device), unsup_idx.to(device)
            B_u = unsup_w.size(0)

            # --- 目标生成 (Target Generation - Common) ---
            with torch.no_grad(), autocast('cuda'):
                # 始终使用弱增强图像生成伪标签
                logits_w = classifier(encoder(unsup_w))
                probs_w = F.softmax(logits_w, dim=1)

                # 分布再平衡 (Rebalance)
                if not args.no_rebalance:
                    probs_calibrated = probs_w * rebalance_factor
                    probs_calibrated /= (probs_calibrated.sum(dim=1, keepdim=True) + 1e-8)
                else:
                    probs_calibrated = probs_w

                # 生成最终目标
                pl_hard = torch.argmax(probs_calibrated, dim=1)
                target_oh = F.one_hot(pl_hard, num_classes).float()
                if args.lsr > 0:
                    target_final = target_oh * (1 - args.lsr) + args.lsr / num_classes
                else:
                    target_final = target_oh

                # 权重计算 (SoftMatch)
                if args.no_softmatch:
                    # 消融实验：如果不使用 SoftMatch，权重设为 1.0
                    inst_weights = torch.ones(B_u, device=device)
                else:
                    inst_weights = softmatch_manager(probs_w, unsup_idx)
            # --- 前向策略 (Forward Strategy) ---
            
            # 2.1 视图选择
            if getattr(args, 'no_consistency', False):
                # 消融：无一致性 -> 使用强增强 (Self-Training)
                train_imgs = unsup_s
            else:
                # 【基线逻辑】非可靠集仅使用强增强 (Strong Aug)
                train_imgs = unsup_s

            # 2.2 Mixup 决策
            do_mixup = not args.no_unreliable_mixup
            
            # 2.3 Mixup 系数逻辑 (关键！)
            # 如果启用了 SoftMatch (且没被消融掉)，使用 SoftMatch 权重作为 Lambda
            # 否则使用 None (随机)
            mixup_lambda_source = inst_weights if (not args.no_softmatch and do_mixup) else None

            # 2.4 执行计算
            with autocast('cuda'):
                loss_c = forward_with_strategy(
                    images=train_imgs, 
                    targets=target_final, 
                    use_mixup=do_mixup, 
                    mixup_alpha=args.mixup_alpha,
                    weights=inst_weights,       # SoftMatch 权重用于 Loss 加权
                    force_lambda=mixup_lambda_source # 【基线逻辑】SoftMatch 权重也用于 Mixup 插值系数
                )

            meter['unsup'] += loss_c.item() * B_u
            meter['cnt_unsup'] += B_u

        # ---------------------------------------------------------------
        # PART 3: 优化 (Optimize)
        # ---------------------------------------------------------------
        # 总 Loss = 可靠集 Loss + (动态权重 * 非可靠集 Loss)
        # 注意：这里的 loss_c 已经被 inst_weights (SoftMatch) 加权过了
        total_loss = loss_s + dynamic_consistency_weight * loss_c

        if total_loss > 0 and not torch.isnan(total_loss):
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # --- 日志记录 (Logging) ---
    def safe_mean(key, cnt_key): return meter[key] / meter[cnt_key] if meter[cnt_key] > 0 else 0.0
    
    if num_batches > 0:
        logger.info(f"  -> [Train Loss] "
                    f"Sup={safe_mean('sup', 'cnt_sup'):.4f}, "
                    f"SSL={safe_mean('unsup', 'cnt_unsup'):.4f}, "
                    f"DynWeight={dynamic_consistency_weight:.4f}")



@torch.no_grad()
def evaluate(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval(); correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast('cuda'): outputs = classifier(encoder(images))
        _, predicted = torch.max(outputs, 1); total += labels.size(0); correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0

def multi_head_knn_search_pytorch_chunked(feats, k, num_heads=4, chunk_size=4096):
    """
    使用 PyTorch 原生矩阵乘法实现的 Multi-Head KNN Search。
    
    Args:
        feats (Tensor): 输入特征 (N, D), 必须在 GPU 上。
        k (int): 检索的邻居数量 (不包含自身通常设为 k，但为了兼容性这里返回 k+1)。
        num_heads (int): 多头数量。
        chunk_size (int): 每次计算的 query 数量，防止显存爆炸 (OOM)。
                          RTX 5080 显存很大，可以设为 4096 或 8192。
    
    Returns:
        sims (Tensor): 相似度矩阵 (N, k+1)
        indices (Tensor): 索引矩阵 (N, k+1)
    """
    N, D_dim = feats.shape
    assert D_dim % num_heads == 0, f"Feature dim {D_dim} not divisible by heads {num_heads}"
    head_dim = D_dim // num_heads
    
    # 1. Multi-Head 预处理 (全 GPU)
    # 逻辑：将特征拆分为 head，分别归一化，再拼接回去
    # 这样计算出的点积就是各头 Cosine 相似度的总和
    feats_view = feats.view(N, num_heads, head_dim)
    feats_norm = F.normalize(feats_view, p=2, dim=2)
    feats_ready = feats_norm.reshape(N, D_dim) # (N, D) - 准备好用于矩阵乘法
    
    # 2. 分块计算 KNN (避免生成 N*N 的大矩阵)
    # 我们需要计算 S = Query @ Database^T
    # Database 就是 feats_ready 自身
    
    # 结果容器
    final_sims = []
    final_indices = []
    
    # 加上 no_grad 节省显存并加速
    with torch.no_grad():
        # 转置数据库以便乘法: (D, N)
        database_t = feats_ready.t()
        
        # 分块循环
        for i in range(0, N, chunk_size):
            # 取出一个 batch 的 query: (B, D)
            end_idx = min(i + chunk_size, N)
            query_chunk = feats_ready[i:end_idx]
            
            # 矩阵乘法: (B, D) @ (D, N) -> (B, N)
            # RTX 5080在这里会极快
            sim_matrix = torch.mm(query_chunk, database_t)
            
            # FAISS 的逻辑是返回 sum(heads) / num_heads
            # 我们在这里除，或者最后除都可以。为了数值稳定性，这里不除，最后统一除。
            
            # 直接获取 TopK，不需要保留巨大的 sim_matrix
            # k+1 是因为通常包含自己
            batch_k = min(k + 1, N) 
            batch_sims, batch_indices = torch.topk(sim_matrix, k=batch_k, dim=1, largest=True, sorted=True)
            
            final_sims.append(batch_sims)
            final_indices.append(batch_indices)
            
            # 手动清理临时变量（虽然 Python 会自动回收，但在大循环中显式释放更安全）
            del sim_matrix
    
    # 3. 拼接结果
    final_sims = torch.cat(final_sims, dim=0)       # (N, k+1)
    final_indices = torch.cat(final_indices, dim=0) # (N, k+1)
    
    # 4. 缩放处理 (对齐原 FAISS 逻辑)
    # 原逻辑：D_faiss = D_faiss / float(num_heads)
    final_sims = final_sims / float(num_heads)
    
    return final_sims, final_indices


import torch
import torch.nn.functional as F
import math
import numpy as np
import torch
import torch.nn.functional as F
import math
import numpy as np

import torch
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
import math


import torch
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
import math

# ==============================================================================
# 🛠️ Helper 1: 显存安全的高精度 KNN
# ==============================================================================
def knn_search_pytorch_chunked(feats, k, num_heads=1, chunk_size=4096):
    original_matmul_precision = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False 
        N, D_dim = feats.shape
        if not feats.is_contiguous(): feats = feats.contiguous()
        
        if num_heads > 1:
            head_dim = D_dim // num_heads
            feats_ready = F.normalize(feats.view(N, num_heads, head_dim), p=2, dim=2).reshape(N, D_dim) 
        else:
            feats_ready = F.normalize(feats, p=2, dim=1)

        final_sims, final_indices = [], []
        with torch.no_grad():
            database_t = feats_ready.t()
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                sim_matrix = torch.mm(feats_ready[i:end_idx], database_t)
                batch_sims, batch_indices = torch.topk(sim_matrix, k=min(k+1, N), dim=1, largest=True, sorted=True)
                final_sims.append(batch_sims); final_indices.append(batch_indices)
        return torch.cat(final_sims, dim=0) / float(num_heads), torch.cat(final_indices, dim=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_precision

# ==============================================================================
# 🛠️ Helper 2: 评估筛选质量
# ==============================================================================
def eval_selection_quality(current_soft, current_pl, candidate_mask, clean_targets, args, device):
    N, num_classes = current_soft.shape
    if clean_targets.device != current_soft.device: clean_targets = clean_targets.to(current_soft.device)
    
    prob_temp = torch.clamp(current_soft, min=1e-2, max=1-1e-2)
    discrepancy = -torch.log(prob_temp)
    
    is_in_candidate = candidate_mask.gather(1, current_pl.view(-1, 1)).squeeze(1) 
    agreement_measure = torch.zeros(N, num_classes, device=device)
    agreement_measure.scatter_(1, current_pl.view(-1, 1), is_in_candidate.view(-1, 1))
    num_clean_per_class = agreement_measure.sum(dim=0)
    
    if args.delta == 0.5: limit_per_class = torch.median(num_clean_per_class)
    elif args.delta == 1.0: limit_per_class = torch.max(num_clean_per_class)
    elif args.delta == 0.0: limit_per_class = torch.min(num_clean_per_class)
    else: limit_per_class = torch.quantile(num_clean_per_class.float(), args.delta)
    
    final_selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    for i in range(num_classes):
        idx_class_mask = (candidate_mask[:, i] == 1.0)
        samples_per_class = idx_class_mask.sum()
        if samples_per_class == 0: continue
        
        discrepancy_class = discrepancy[idx_class_mask, i]
        k_corrected = min(limit_per_class.item(), samples_per_class.item())
        if k_corrected < 1: continue
        
        _, top_idx_rel = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)
        idx_class_indices = idx_class_mask.nonzero().squeeze(1)
        final_selected_indices = idx_class_indices[top_idx_rel]
        final_selected_mask[final_selected_indices] = True
        
    n_selected = final_selected_mask.sum().item()
    if n_selected > 0:
        correct = (current_pl[final_selected_mask] == clean_targets[final_selected_mask]).sum().item()
        acc_selected = correct / n_selected
    else: acc_selected = 0.0
    return n_selected, acc_selected

# ==============================================================================
# 🛠️ Helper 3: DAES 亲和度矩阵计算 (参数化版)
# ==============================================================================
def get_adaptive_affinity_matrix(raw_D, neighbors_indices, current_soft_labels, args):
    """
    计算 DAES 动态边权重。
    输入 current_soft_labels 可以是原始 Soft 也可以是 Gated Soft，用于计算熵。
    """
    if raw_D.shape[1] > neighbors_indices.shape[1]: raw_D_neighbors = raw_D[:, 1:]
    else: raw_D_neighbors = raw_D

    att_temp = 0.5 
    spatial_weights = F.softmax(raw_D_neighbors / att_temp, dim=1).unsqueeze(-1)
    
    # 归一化用于计算分布形状 (熵)
    current_soft_norm = current_soft_labels / (current_soft_labels.sum(1, keepdim=True) + 1e-8)
    neighbor_labels = F.embedding(neighbors_indices, current_soft_norm) 
    local_mean = (neighbor_labels * spatial_weights).sum(dim=1)
    
    # [参数化] 锐化力度
    power = getattr(args, 'daes_sharpening_power', 2.0)
    if power != 1.0:
        local_mean = local_mean ** power
        local_mean = local_mean / (local_mean.sum(1, keepdim=True) + 1e-8)
    
    local_entropy = -torch.sum(local_mean * torch.log(local_mean + 1e-8), dim=1)
    norm_entropy = local_entropy / math.log(args.num_classes)
    
    # # [参数化] 熵敏感度 & 温度截断
    # entropy_weight = getattr(args, 'daes_entropy_weight', 0.2)
    # max_clamp = getattr(args, 'daes_clamp', 0.25)
    
    # tau_dynamic = 0.05 + (norm_entropy * entropy_weight)
    # tau_dynamic = torch.clamp(tau_dynamic, max=max_clamp).unsqueeze(1)
    # scaled_sim = torch.pow(raw_D, 2) / tau_dynamic
    # max_val, _ = scaled_sim.max(dim=1, keepdim=True)
    # weights = torch.exp(scaled_sim - max_val.detach()) 
    # Dynamic Tau
    # base_tau = 0.07
    # tau_dynamic = base_tau + (torch.pow(norm_entropy, 2) * 0.5) 
    # tau_dynamic = tau_dynamic.unsqueeze(1) 
    
    # scaled_sim = raw_D / tau_dynamic
    # max_val, _ = scaled_sim.max(dim=1, keepdim=True)
    # weights = torch.exp(scaled_sim - max_val.detach()) # Subtract max for stability    
    # return weights
    # Dynamic Tau
    base_tau = 0.07
    tau_dynamic = base_tau + (torch.pow(norm_entropy, 2) * 0.5) 
    # tau_dynamic = base_tau + (torch.pow(norm_entropy, 1) * 0.5) 
    tau_dynamic = tau_dynamic.unsqueeze(1) 
    
    # scaled_sim = raw_D / tau_dynamic
    # 增加一个可学习或可调的 power 参数
    scaled_sim = torch.pow(raw_D, 2) / tau_dynamic
    max_val, _ = scaled_sim.max(dim=1, keepdim=True)
    weights = torch.exp(scaled_sim - max_val.detach()) # Subtract max for stability
    return weights


# ==============================================================================
# 🛠️ Helper 4: 通用权重获取入口
# ==============================================================================
def get_weight_matrix(mode, raw_D, neighbors_indices, ref_soft_labels, args):
    if mode == 'daes':
        return get_adaptive_affinity_matrix(raw_D, neighbors_indices, ref_soft_labels, args)
    elif mode == 'exp':
        return torch.exp(raw_D / 0.1)
    else:
        return raw_D

# ==============================================================================
# 🛠️ Helper 5: 单步传播逻辑
# ==============================================================================
def run_propagation_step(input_soft, weight_tensor, target_neighbors, args, logger, 
                         clean_labels, candidate_mask, tag_name, log_result=False, is_active_path=False):
    N = input_soft.shape[0]
    device = input_soft.device
    
    knn_idx = target_neighbors.view(N, args.k_val+1, 1).expand(N, args.k_val+1, args.num_classes)
    knn_input = input_soft.expand(N, -1, -1)
    neighbors_soft = torch.gather(knn_input, 1, knn_idx)
    
    weighted_soft = torch.mul(neighbors_soft, weight_tensor.view(N, -1, 1))
    score = torch.sum(weighted_soft, 1)
    
    # 始终返回 Soft Output，硬化操作在外部 Middleware 进行
    output_soft = score / (score.sum(1).unsqueeze(-1) + 1e-8)
    _pl = torch.max(score, -1)[1]
    
    if log_result and logger:
        sel_size, sel_acc = eval_selection_quality(output_soft, _pl, candidate_mask, clean_labels, args, device)
        acc_knn = (_pl.eq(clean_labels)).float().mean().item()
        efficiency = ((sel_acc * sel_size) / N) * 100
        marker = "(*) " if is_active_path else "    "
        logger.info(f"{marker}[{tag_name:<20}] Sel: {sel_size:<5} | Prec: {sel_acc*100:05.2f}% | Eff: {efficiency:05.2f}% | KNN: {acc_knn*100:05.2f}%")
        
    return output_soft, _pl


#Middle soft singlelabel input
def reliable_pseudolabel_selection_weighted(logger, args, device, trainloader, features, epoch, model_preds=None, teacher_preds=None):
    """
    Algorithm: GCK-Selection (Hybrid Input Magnitude Variant)
    
    核心逻辑修改 (Stage 2 Input):
    - 无 Teacher: 输入为 One-Hot Hard Label (Value=1.0)。
    - 有 Teacher: 输入为 Gated Hard Label (Value=Gate)。
      即: 只保留 Argmax 那个类的概率，但其值不是 1.0，而是被 Teacher 降权后的 gate 值 (e.g., 0.3)。
      其余类别依然为 0。
    """
    # 1. 数据准备
    features = features.to(device)
    N = features.shape[0]
    labels = torch.tensor(trainloader.dataset.soft_labels, device=device, dtype=torch.float32)
    clean_labels = torch.tensor(trainloader.dataset.clean_labels, device=device, dtype=torch.long)
    
    # --- [New] Add Prior Weighting (Aligned with Trept Script) ---
    if hasattr(trainloader.dataset, 'weights'):
        prior = torch.tensor(trainloader.dataset.weights, device=device, dtype=torch.float32)
    else:
        prior = torch.ones_like(labels)
    initial_input = torch.mul(labels, prior) 
    # 2. 建图
    knn_heads = getattr(args, 'knn_heads', 4) 
    D_mh, neighbors_mh = knn_search_pytorch_chunked(features, args.k_val, num_heads=knn_heads, chunk_size=4096)

    # ----------------------------------------------------------------------
    # [Step 1] Stage 1: Geometric Blind Guessing (Exp)
    # ----------------------------------------------------------------------
    w_s1_raw = get_weight_matrix(args.sim_mode_1, D_mh, neighbors_mh[:, 1:], initial_input, args)
    w_s1 = w_s1_raw / (w_s1_raw.sum(dim=1, keepdim=True) + 1e-8)
    
    soft_iter1, pl_iter1 = run_propagation_step(
        initial_input, w_s1, neighbors_mh, args, logger, 
        clean_labels, labels, f"Main-Iter1 ({args.sim_mode_1})", 
        log_result=True, is_active_path=True
    )

    # ----------------------------------------------------------------------
    # [Middleware] 中间件层: Signal Hardening & Topology Gating
    # ----------------------------------------------------------------------
    
    # 1. 基础硬化 (默认 1.0)
    # pl_hard_idx = soft_iter1.max(dim=1)[1]
    pl_max_val, pl_hard_idx = soft_iter1.max(dim=1)
    # 创建基础 One-Hot-hard (N, C)
    # input_signal_base = torch.zeros_like(soft_iter1).scatter_(1, pl_hard_idx.view(-1, 1), 1.0)
    ## 创建基础 one-hot-soft-label Signal (N, C), 只有 Argmax 处有值，且值为 Confidence
    input_signal_base = torch.zeros_like(soft_iter1).scatter_(1, pl_hard_idx.view(-1, 1), pl_max_val.view(-1, 1))
    
    # 初始化变量
    input_signal_for_s2 = input_signal_base # 默认是 1.0 的硬标签
    # topology_ref_soft = soft_iter1          # 默认是 Stage 1 的软输出
    topology_ref_soft = input_signal_base          # 默认是 Stage 1 的1.0的硬标签，但是如果教师干预了就会变成软的计算daes，否则保持硬
    
    # 2. 教师干预 (如果未禁用 Rectification)
    if teacher_preds is not None and not args.no_rectify:
        p_teacher = teacher_preds.to(device)
        start_ratio = getattr(args, 'gating_start_ratio', 0.2)
        is_mature = epoch > (args.epochs * start_ratio)
        
        if is_mature:
            pl_geo = pl_iter1 
            teacher_conf = torch.gather(p_teacher, 1, pl_geo.view(-1, 1)).squeeze(1)
            
            max_alpha = getattr(args, 'gating_max_alpha', 1.0)
            current_alpha_raw = epoch / (args.epochs * 0.5 + 1e-6)
            alpha = min(max_alpha, current_alpha_raw)
            
            # 计算 Gate (N, 1) -> 范围 [0, 1]
            gate = (1.0 - alpha) + alpha * teacher_conf
            gate = gate.unsqueeze(1)
            
            # --- [关键修改点 Start] ---
            
            # A. 拓扑参考 (Topology Ref):
            # 使用 Gated Soft 信号。Teacher 不信的节点，熵变高，DAES 切断连接。
            topology_ref_soft = soft_iter1 * gate
            # topology_ref_soft = input_signal_base * gate          # 默认是 Stage 1 的1.0的硬标签，但是如果教师干预了就会变成软的计算daes，否则保持，用于计算熵的部分也可以
            
            # B. 信号输入 (Input Signal):
            # 需求: 只要那个 Hard 的类，但值要是 Gate 的值。
            # 实现: 直接用 Gate 乘以 One-Hot 向量。
            # 结果: [0, 0, 0.3, 0] (假设 Gate=0.3, Argmax=2)
            input_signal_for_s2 = input_signal_base * gate
            
            # --- [关键修改点 End] ---
            
            if logger: 
                logger.info(f"   >>> [Middleware] Gating Active (Alpha: {alpha:.2f}). Modulating Signal Magnitude.")
        else:
             if logger: logger.info(f"   >>> [Middleware] Model immature. Using Hard Input (1.0).")
    
    # ----------------------------------------------------------------------
    # [Step 2] Stage 2: Topology-Adaptive Propagation
    # ----------------------------------------------------------------------
    if args.sim_mode_2 is not None and args.sim_mode_2 != 'None':
        logger.info(f"   >>> [Main] Stage 2 Strategy: {args.sim_mode_2.upper()}")
        
        # 1. 计算权重: 使用 topology_ref_soft 
        # (有 Teacher 时是 Gated Soft，无 Teacher 时是 Raw Soft)
        w_s2 = get_weight_matrix(args.sim_mode_2, D_mh, neighbors_mh[:, 1:], topology_ref_soft, args)
        w_s2_norm = w_s2 / (w_s2.sum(dim=1, keepdim=True) + 1e-8)
        
        # 2. 执行传播: 使用 input_signal_for_s2
        # (有 Teacher 时是 Gated Hard [0, 0.3, 0]，无 Teacher 时是 Pure Hard [0, 1.0, 0])
        soft_labels_iter2, pseudo_labels_iter2 = run_propagation_step(
            input_signal_for_s2, 
            w_s2_norm, 
            neighbors_mh, 
            args, logger, 
            clean_labels, labels, 
            f"Main-Iter2 ({args.sim_mode_2})", 
            log_result=True, is_active_path=True
        )
        
        final_soft_labels = soft_labels_iter2
        final_pseudo_labels = pseudo_labels_iter2
    else:
        logger.info(f"   >>> [Main] Skipping Stage 2. Using Stage 1 results.")
        final_soft_labels = soft_iter1
        final_pseudo_labels = pl_iter1

    # ==============================================================================
    # 6. 最终筛选
    # ==============================================================================
# ==============================================================================
    # 6. 最终筛选
    # ==============================================================================
    prob_temp = torch.clamp(final_soft_labels, min=1e-2, max=1-1e-2)
    discrepancy_measure_final = -torch.log(prob_temp)

    max_idx = final_soft_labels.max(1)[1]
    agreement_measure = (labels.gather(1, max_idx.view(-1,1)).squeeze(1) == 1.0).float()
    
    num_clean_per_class = torch.zeros(args.num_classes, device=device)
    num_clean_per_class.scatter_add_(0, max_idx, agreement_measure)

    if args.delta == 0.5: num_samples2select_class = torch.median(num_clean_per_class)
    elif args.delta == 1.0: num_samples2select_class = torch.max(num_clean_per_class)
    elif args.delta == 0.0: num_samples2select_class = torch.min(num_clean_per_class)
    else: num_samples2select_class = torch.quantile(num_clean_per_class.float(), args.delta)

    final_selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    final_selected_values = torch.full((N, args.num_classes), float('inf'), device=device)
    
    # [修复] 循环变量为 c
    for c in range(args.num_classes):
        # [修复] 使用 c 索引 labels
        idx_class_mask = (labels[:, c] == 1.0)
        samples_per_class = idx_class_mask.sum()
        if samples_per_class == 0: continue
        
        # [修复] 使用 c 索引 discrepancy
        discrepancy_class = discrepancy_measure_final[idx_class_mask, c]
        
        k_corrected = min(num_samples2select_class.item(), samples_per_class.item())
        if k_corrected < 1: continue
            
        val, top_idx_rel = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)
        idx_class_indices = idx_class_mask.nonzero().squeeze(1)
        final_selected_indices = idx_class_indices[top_idx_rel]
        final_selected_mask[final_selected_indices] = True
        
        # [修复] 使用 c 保存 values
        final_selected_values[final_selected_indices, c] = val

    selected_examples = final_selected_mask.float() 
    _, selected_labels = torch.min(final_selected_values, dim=1)
    
    knn_scores = final_soft_labels.cpu()
    knn_pl = final_pseudo_labels.cpu()
    selected_examples = selected_examples.cpu()
    selected_labels = selected_labels.cpu()
    
    if model_preds is not None:
        model_pl = torch.max(model_preds, 1)[1].cpu()
    else:
        model_pl = None
    
    return selected_examples, selected_labels, knn_pl, model_pl, knn_scores



import datetime


import datetime

def run_single_experiment(args):
    start_time = time.time(); set_seed(args.seed)
    log_dir = os.path.join(args.out, args.exp_name, f"seed_{args.seed}")
    logger = setup_logger(log_dir)
    # logger = setup_logger(log_dir, to_console=True)
    logger.info(f"--- Starting Dynamic Strategy Run with Seed: {args.seed} ---"); logger.info(f"Settings: {vars(args)}")
    

    
    # 1. 自动判断当前的实验类型 (Experiment Type)
    #    根据 args 的开关，决定实验叫什么名字，方便在列表中一眼看清
    exp_type = "Baseline_Full" # 默认名
    
    if args.no_unreliable_training:
        exp_type = "Sup_Only"
    elif args.no_unreliable_mixup and args.no_softmatch and args.no_rebalance:
        # 这种组合很少见，但也写上防止万一
        exp_type = "Ablation_All_Removed"
    elif args.no_unreliable_mixup and args.no_softmatch:
        exp_type = "No_Mixup_No_SoftMatch"
    elif args.no_rebalance:
        exp_type = "No_Rebalance"
    elif args.no_softmatch:
        exp_type = "No_SoftMatch"
    elif args.no_unreliable_mixup:
        exp_type = "No_Unrel_Mixup" # 你的重点实验
    elif args.no_reliable_mixup:
        exp_type = "No_Rel_Mixup"
    
    # 2. 生成动态标签 (Dynamic Tags)
    #    除了基础信息，把关键参数和开关全放进 Tag
    current_tags = [
        args.dataset,
        args.network,
        f"PR{args.pr}",
        f"NR{args.nr}",
        f"BS{args.batch_size}",
    ]
    
    # 将 bool 开关转换为标签
    if args.no_unreliable_mixup: current_tags.append("No_Mixup")
    if args.no_softmatch:        current_tags.append("No_SoftMatch")
    if args.no_rebalance:        current_tags.append("No_Rebalance")
    if args.no_unreliable_training: current_tags.append("Sup_Only")

    # 3. 构造智能分组名 (Group Name)
    #    【关键】Group 必须是：除了 Seed 不同，其他都相同的标识符。
    #    这样 WandB 才能自动把 Seed 1,2,3 聚合在一起算均值。
    group_name = f"{args.dataset}_PR{args.pr}_NR{args.nr}_{exp_type}"

    # 4. 构造唯一的运行名 (Unique Run Name)
    #    格式：类型_种子_时间戳 (防止重复)
    #    时间戳只取后几位，保持简洁
    short_id = datetime.datetime.now().strftime('%H%M%S') 
    run_name = f"{exp_type}_S{args.seed}_{short_id}"

    # 5. 初始化 WandB
    wandb.init(
        project="CIFAR100_Ablation_Study", # 建议：为这次消融实验单独开一个 Project，或者保持原样
        # project="PALS_Crowdsource_Experiments", 
        
        name=run_name,       # 例如: No_Unrel_Mixup_S1_102433
        group=group_name,    # 例如: CIFAR100_PR0.05_NR0.3_No_Unrel_Mixup (三个种子这个名字一样)
        tags=current_tags,   # 包含所有关键信息
        config=args,
        reinit=True,
        mode="disabled"
    )
    device = torch.device(f"cuda:{args.cuda_dev}" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 数据加载 ---
    
    # --- (修改) 更新 num_classes_map ---
    # num_classes_map = {
    #     'CIFAR10': 10, 'CIFAR100': 100, 'CIFAR100H': 100, 'CUB200': 200,
    #     'Treeversity': 6,  # (来自 PALS_v1 命令)
    #     'Benthic': 8,      # (来自 PALS_v1 命令)
    #     'Plankton': 10     # (来自 PALS_v1 命令)
    # }
    # --- (修改) 更新 num_classes_map ---
    num_classes_map = {
        # 模拟数据集
        'CIFAR10': 10, 
        'CIFAR100': 100, 
        'CIFAR100H': 100,
        'CUB200': 200,
        
        # --- 根据 scan_dataset.py 结果更新 ---
        'Turkey': 3,          # 扫描结果显示 3 类
        'Pig': 4,             # 扫描结果显示 4 类
        'MiceBone': 3,        # 扫描结果显示 3 类
        'QualityMRI': 2,      # 扫描结果显示 2 类
        'Treeversity': 6,     # 通常是 6
        'Treeversity#6': 6,   # 你的扫描结果
        'Benthic': 8,         # 扫描结果显示 8 类
        'Plankton': 10,       # 扫描结果显示 10 类
        
        # --- 关键修改 ---
        'Synthetic': 6,       # 扫描结果明确显示是 6 类！
        # ----------------
        
        'CIFAR10H': 10,
        'verse_blended-vps': 4, # 扫描结果显示 4 类
        'verse_mask1-vps': 4    # 扫描结果显示 4 类
    }
    # (确保在 args.dataset 不在 map 中时给出错误)
    if args.dataset not in num_classes_map:
        raise ValueError(f"Dataset {args.dataset} not recognized in num_classes_map.")
    num_classes = num_classes_map[args.dataset]
    
    args.num_classes = num_classes
    args.seed_dataset = args.seed
    
    weak_t, strong_t, test_t = get_pals_transforms(args.dataset)
    crowdsource_datasets = [
        'Treeversity', 'Benthic', 'Plankton', 
        'Turkey', 'Pig', 'MiceBone', 'QualityMRI', 
        # 'Synthetic', 'verse_blended-vps', 'verse_mask1-vps', 'CIFAR10H'
        'Synthetic', 'verse_blended-vps', 'verse_mask1-vps', 'CIFAR10H'
    ]  
    # --- (重大修改) 替换数据加载逻辑 ---
    
    # 1. 模拟数据集 (CIFAR / CUB)
    if args.dataset in ['CIFAR10', 'CIFAR100', 'CIFAR100H']:
        is_h = 'H' in args.dataset
        BaseClass = CIFAR100Partial if '100' in args.dataset else CIFAR10Partial
        base_train_ds = BaseClass(args, train=True, download=True, transform=None)
        
        if hasattr(base_train_ds, 'partial_noise'):
            logger.info(f"Generating simulated NPLL noise for {args.dataset} (pr={args.pr}, nr={args.nr})")
            if '100' in args.dataset:
                base_train_ds.partial_noise(args.pr, args.nr, heirarchical=is_h)
            else:
                base_train_ds.partial_noise(args.pr, args.nr)
                
        TestClass = datasets.CIFAR100 if '100' in args.dataset else datasets.CIFAR10
        test_ds = TestClass(root=args.train_root, train=False, download=True, transform=test_t)

    elif args.dataset == 'CUB200':
        BaseClass = CUB200Partial
        base_train_ds = BaseClass(args, train=True, transform=None)
        if hasattr(base_train_ds, 'partial_noise'):
             logger.info(f"Generating simulated NPLL noise for CUB200 (pr={args.pr}, nr={args.nr})")
             base_train_ds.partial_noise(args.pr, args.nr)
        test_ds = BaseClass(args, train=False, transform=test_t)
        
# ... (CUB200 的加载逻辑不变) ...
  
    # 2. 真实众包数据集 (使用 Crowdsource 类)
    # elif args.dataset in ['Treeversity', 'Benthic', 'Plankton','Pig']:
    elif args.dataset in crowdsource_datasets:
        logger.info(f"Loading crowdsource dataset: {args.dataset} (lpi={args.lpi})")
        
        # --- (已修复) ---
        # 根据截图，所有三个数据集都有 5 个 fold。
        # 我们统一使用 PALS_v1 中 slice=1 的默认切分方式。
        train_split, test_split = ['fold1','fold4','fold5'], ['fold3']
        # train_split, test_split = ['fold1','fold2','fold4','fold5'], ['fold3']
        logger.info(f"Using splits: Train={train_split}, Test={test_split}")
        # --- (修复结束) ---
        
        # (重要) 确保 train_root 指向特定数据集的文件夹, e.g., './data/Treeversity#6'
        dataset_root = os.path.join(args.train_root, args.dataset)
        if args.dataset == 'Treeversity':
             dataset_root = os.path.join(args.train_root, 'Treeversity#6') # (PALS_v1 的特殊路径)
        
        logger.info(f"Loading from root: {dataset_root}")
        
        # 复制一份 args, 仅用于 Crowdsource 类
        crowd_args = copy.deepcopy(args)
        crowd_args.train_root = dataset_root
        
        base_train_ds = Crowdsource(
            crowd_args,
            splits=train_split,
            transform=None # Transform 在 SSLReadyDataset 中应用
        )
        test_ds = Crowdsource(
            crowd_args,
            splits=test_split,
            transform=test_t
        )
    else:
        raise ValueError(f"Dataset {args.dataset} loader not implemented.")
    # --- (修改结束) ---

    # (从这里开始，你的代码逻辑保持不变，它会自动处理)
    original_pl = torch.from_numpy(base_train_ds.soft_labels)
    true_labels = torch.from_numpy(base_train_ds.clean_labels)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # (在 Section 5: Main Experiment Logic)
# ...
# --- 2. 模型、优化器和调度器设置 ---
    
    ## --- 🚀 MODIFICATION START: Removed SimSiam, use encoder directly --- ##
    encoder, feature_dim = get_base_encoder(args.network, args.dataset)
    encoder = encoder.to(device) # No more simsiam_model
    classifier = nn.Linear(feature_dim, num_classes).to(device)

    # --- (已更正) 仅对众包数据集使用分层学习率 ---
    # if args.dataset in ['Treeversity', 'Benthic', 'Plankton']:
    if args.dataset in crowdsource_datasets:
        logger.info(f"Using differential LR for crowdsource dataset {args.dataset} (Encoder LR / 100)")
        # (来自 PALS_v1 的设置，encoder LR x 0.01)
        optimizer = optim.SGD([
            {'params': encoder.parameters(), 'lr': args.lr / 100.0}, 
            # {'params': encoder.parameters(), 'lr': args.lr}, 
            {'params': classifier.parameters(), 'lr': args.lr},
        ], lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        # (CIFAR 和 CUB-200 均使用标准优化器)
        logger.info(f"Using standard optimizer settings for {args.dataset}")
        params_to_optimize = list(encoder.parameters()) + list(classifier.parameters())
        optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    ## --- 🚀 MODIFICATION END --- ##
    
    # --- (调度器逻辑不变) ---
    if args.lr_scheduler == 'cosine':
        # ✅ 直接使用总 epoch 数
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        logger.info(f"Using CosineAnnealingLR scheduler with T_max = {args.epochs} epochs.")
    
    elif args.lr_scheduler == 'step':
        # Step 调度器通常不受 warmup 影响，保持原样即可
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
        logger.info(f"Using MultiStepLR scheduler with milestones at {args.lr_decay_epochs} and gamma={args.lr_decay_rate}.")    
        
    softmatch_manager = SoftMatchWeightManager(len(base_train_ds), num_classes, device=device)

    best_test_acc = 0.0
    final_epoch_test_acc = 0.0

    ema_model_probs = torch.ones(len(base_train_ds), num_classes, device=device) / num_classes
    ema_alpha = args.ema_alpha # [Param Control] Using argument instead of hardcoded 0.999
    # ema_alpha = 0.9 
# --- 🚀 混合方案：根据数据集类型估计目标分布 ---
    if hasattr(base_train_ds, 'weights'):
        # 适用于 Crowdsource: 使用数据驱动的估计
        logger.info(" -> [Re-balance] 正在从 'weights' (众包先验) 估计目标分布...")
        estimated_dist_np = base_train_ds.weights.mean(axis=0)
        estimated_dist_np = estimated_dist_np / estimated_dist_np.sum()
        p_target_estimated = torch.tensor(estimated_dist_np, device=device, dtype=torch.float).unsqueeze(0)
        logger.info(f" -> [Re-balance] 使用“估计的”目标分布: \n{estimated_dist_np}")
    else:
        # 适用于 CUB/CIFAR: 退回到“均匀分布”假设
        logger.info(" -> [Re-balance] 'weights' 属性未找到。退回到“均匀分布”假设。")
        p_target_estimated = (torch.ones(1, num_classes, device=device) / num_classes)
        logger.info(f" -> [Re-balance] 使用“均匀的”目标分布。")
    # --- 3. 主训练循环 ---
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f"======== Epoch {epoch+1}/{args.epochs} ========")
        



        
        ## --- 🚀 MODIFICATION START: Reverted to original selection logic --- ##
        # --- 3.1 数据划分 ---
        
        ## --- 🚀 MODIFICATION START: Call Selection Function --- ##
        logger.info(" -> [Selection] Getting features for all train data...")
        feature_loader = DataLoader(FeatureExtractionDataset(base_train_ds, test_t), batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
        features, predictions = get_features(encoder, classifier, feature_loader, device)

        # 创建一个模拟的 loader，仅用于传递 dataset 对象
        class MockTrainloader:
            def __init__(self, dataset):
                self.dataset = dataset
        mock_trainloader = MockTrainloader(base_train_ds)
        
        logger.info(f" -> [Selection] Running reliable_pseudolabel_selection_weighted (k={args.k_val}, delta={args.delta})...")
        
        # # 调用 *修改后* 的函数，获取所有必需的组件
        selected_examples_mask, selected_labels_all, knn_pl, model_pl, knn_scores = reliable_pseudolabel_selection_weighted(logger,
            args, 
            device, 
            mock_trainloader, 
            features, 
            epoch + 1,        # 传递当前 epoch (用于日志)
            model_preds=predictions,
            teacher_preds=ema_model_probs
            # teacher_preds=None
        )
        # 调用 *修改后* 的函数，获取所有必需的组件
        # selected_examples_mask, selected_labels_all, knn_pl, model_pl, knn_scores = reliable_pseudolabel_selection_weighted(logger,
        #     args, 
        #     device, 
        #     mock_trainloader, 
        #     features, 
        #     epoch + 1,        # 传递当前 epoch (用于日志)
        #     model_preds=predictions
        # )
        # --- [Diagnostic Info] ---
        num_selected = selected_examples_mask.sum().item()
        logger.info(f" -> [Selection] Function call complete. Total samples selected: {num_selected}")

        # 将掩码 (mask) 转换为脚本所需的格式 (list of pairs)
        rel_indices_tensor = torch.where(selected_examples_mask.bool())[0].cpu()
        
        # 'selected_labels_all' 包含 *所有* 样本基于差异度量的标签
        rel_labels_tensor = selected_labels_all[rel_indices_tensor].cpu()
        
        rel_indices = rel_indices_tensor.tolist()
        rel_labels = rel_labels_tensor.tolist()
        # reliable_pairs = list(zip(rel_indices, rel_labels)) # (这个变量现在已定义，但未使用)
        
        if num_selected > 0:
            rel_true_labels = true_labels[rel_indices_tensor] # 使用张量索引
            selection_acc = (rel_labels_tensor == rel_true_labels).float().mean().item()
            logger.info(f" -> [Selection] Reliable Set Accuracy (vs True Labels): {selection_acc * 100:.2f}%")
            wandb.log({'Reliable Set Accuracy': selection_acc * 100,
                       'Reliable Set Size': num_selected}, step=epoch+1) # (使用 epoch+1 避免 wandb 警告)
        else:
             logger.info(" -> [Selection] No reliable samples selected.")
             wandb.log({'Reliable Set Accuracy': 0,
                       'Reliable Set Size': 0}, step=epoch+1) # (使用 epoch+1)
        ## --- 🚀 MODIFICATION END --- ##


        # 这部分代码现在可以正常工作了
        rel_indices_set = set(rel_indices)

        num_unreliable = len(base_train_ds) - len(rel_indices_set)
        unreliable_ratio = num_unreliable / len(base_train_ds)
        unreliable_indices = list(set(range(len(base_train_ds))) - rel_indices_set)
        logger.info(f" -> Data Split: Reliable({len(rel_indices_set)}), Unreliable({len(unreliable_indices)})")
        
        # 这个日志函数现在也可以工作了
        if args.detailed_log:
            log_unified_partition_diagnostics(logger, true_labels, device, 
                                              rel_indices_set, unreliable_indices, 
                                              model_pl, knn_pl, predictions)
        
        # --- 3.2 动态一致性权重 ---
        reliable_set_len = len(rel_indices_set)
        consensus_proportion = 0.0 
        
        if reliable_set_len > 0:
            rel_indices_tensor = torch.tensor(list(rel_indices_set), dtype=torch.long, device=device)
            # (将 PL 移动到 device 进行比较)
            model_pred_rel = model_pl.to(device)[rel_indices_tensor]
            knn_pred_rel = knn_pl.to(device)[rel_indices_tensor]
            consensus_count_rel = (model_pred_rel == knn_pred_rel).sum().item()
            consensus_proportion = consensus_count_rel / reliable_set_len
        
        consensus_power = args.consensus_power
        # consensus_power = 1.0 #/consensus_power1
        
        # --- 🚀 [Ablation: Fixed Dynamic Weight] ---
        if args.fix_dynamic_weight:
            # 强制设为固定值 (默认 args.consistency_weight=1.0)
            # 这意味着不再考虑“不可靠比例”和“共识度”的课程学习效应
            dynamic_consistency_weight = args.consistency_weight 
            logger.info(f" -> [Ablation] Dynamic Weight FIXED to {dynamic_consistency_weight} (Curriculum Disabled)")
        else:
            # 原始动态逻辑
            dynamic_consistency_weight = unreliable_ratio * (consensus_proportion ** consensus_power) * args.consistency_weight
            logger.info(f" -> Dynamic Weight: UnreliableRatio={unreliable_ratio:.2f} * ReliableConsensus={consensus_proportion:.2f} (power {consensus_power}) -> Consistency Weight={dynamic_consistency_weight:.3f}")
        # -------------------------------------------

        # --- 3.3 全局再平衡因子 ---
        # (保持不变)
        logger.info(" -> [Re-balance] Calculating distribution rebalance factor...")
        p_model_biased = ema_model_probs.mean(dim=0, keepdim=True)
        # p_target = (torch.ones(1, num_classes, device=device) / num_classes)
        p_target = p_target_estimated
        rebalance_factor = (p_target / (p_model_biased + 1e-8)).detach()
        
        if (epoch + 1) % 10 == 0:
             max_bias, max_class = p_model_biased.max(dim=1)
             min_bias, min_class = p_model_biased.min(dim=1)
             logger.info(f"     [Re-balance Stats] Model Bias (EMA): Max={max_bias.item():.4f} (Class {max_class.item()}), Min={min_bias.item():.4f} (Class {min_class.item()})")
             logger.info(f"     [Re-balance Stats] Rebalance Factor: Max={rebalance_factor.max().item():.2f}, Min={rebalance_factor.min().item():.2f}")
        
        
        # --- 3.4 创建 DataLoaders ---
        loader_sup, loader_unsup = None, None
            
        if rel_indices:
            ds_sup = SSLReadyDataset(base_train_ds, list(rel_indices), list(rel_labels), weak_t, strong_t)
            loader_sup = DataLoader(ds_sup, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
        if unreliable_indices:
            ds_unsup = SSLReadyDataset(base_train_ds, unreliable_indices, [-1] * len(unreliable_indices), weak_t, strong_t)
            # loader_unsup = DataLoader(ds_unsup, batch_size=args.unreliable_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            loader_unsup = DataLoader(ds_unsup, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        # --- 3.5 调用训练循环 ---
        # (现在我们传递 knn_scores)
        train_unified_loop(args, encoder, classifier, device, 
                           loader_sup, loader_unsup, optimizer, softmatch_manager, 
                           logger, num_classes, 
                           knn_pl, model_pl,
                           knn_scores, # <--- 🚀 传递新参数
                           dynamic_consistency_weight,
                           rebalance_factor
                           )
        scheduler.step()
        
        ## --- 🚀 MODIFICATION: Use encoder --- ##
        test_acc = evaluate(encoder, classifier, test_loader, device)
        
        final_epoch_test_acc = test_acc 
        if test_acc > best_test_acc: 
            best_test_acc = test_acc
            
        ema_model_probs = ema_alpha * ema_model_probs + (1 - ema_alpha) * predictions
        
        logger.info(f"Epoch {epoch+1} Summary: Test Acc={test_acc:.2f}% | Best Acc={best_test_acc:.2f}% | LR={optimizer.param_groups[0]['lr']:.6f} | Time: {time.time() - epoch_start_time:.2f}s\n")
        
        # --- (新增) W&B 日志 (来自 PALS_v1) ---
        wandb.log({
            # 'Test Loss': 0, # evaluate 函数不返回 loss
            'Test Accuracy': test_acc,
            'Best Test Accuracy': best_test_acc,
            'Learning Rate': optimizer.param_groups[0]['lr']
        }, step=epoch + 1)

    # ... (循环结束) ...
    
    duration = time.time() - start_time
    logger.info(f"--- Run Finished (Duration: {duration/60:.2f} min). Best Acc: {best_test_acc:.2f}%, Final Epoch Acc: {final_epoch_test_acc:.2f}% ---")
    
    # --- (新增) W&B 结束 (来自 PALS_v1) ---
    wandb.finish()
    
    return best_test_acc, final_epoch_test_acc, duration


@torch.no_grad()
def log_unified_partition_diagnostics(logger, true_labels, device, 
                                      rel_indices_set, unreliable_indices, 
                                      model_pl, knn_pl, predictions):
    """
    为统一化的两阶段数据划分（可靠集 vs 不可靠集）设计的诊断函数。
    不可靠集内部会进一步分析“共识”与“非共识”样本的情况。
    """
    logger.info("--- Epoch Partition Diagnostics (Unified) ---")
    true_labels = true_labels.to(device)
    
    # --- 🚀 修复：将所有传入的伪标签张量移动到 GPU (device) ---
    model_pl, knn_pl, predictions = model_pl.to(device), knn_pl.to(device), predictions.to(device)
    # --- 修复结束 ---

    # 1. 可靠集诊断 (与之前相同)
    if rel_indices_set:
        rel_indices = torch.tensor(list(rel_indices_set), device=device, dtype=torch.long) # 确保 long
        # 可靠集的标签来自KNN提纯后的结果，我们用它和真实标签比较
        rel_acc = (knn_pl[rel_indices] == true_labels[rel_indices]).float().mean().item() * 100
        logger.info(f"  -> [Reliable Set]        Size: {len(rel_indices_set):<5} | Accuracy (vs KNN PL): {rel_acc:.2f}%")

    # 2. 不可靠集诊断 (新的核心逻辑)
    if unreliable_indices:
        unrel_indices_tensor = torch.tensor(unreliable_indices, device=device, dtype=torch.long) # 确保 long
        
        # 在不可靠集内部，动态划分出共识集与非共识集
        consensus_mask = (model_pl[unrel_indices_tensor] == knn_pl[unrel_indices_tensor])
        consensus_indices = unrel_indices_tensor[consensus_mask]
        non_consensus_indices = unrel_indices_tensor[~consensus_mask]

        # 2a. 共识集 (Consensus Set) 诊断
        if len(consensus_indices) > 0:
            con_pl_acc = (knn_pl[consensus_indices] == true_labels[consensus_indices]).float().mean().item() * 100
            con_model_conf = predictions[consensus_indices].max(dim=1)[0].mean().item()
            logger.info(f"  -> [Unreliable-Consensus]  Size: {len(consensus_indices):<5} | PL Acc (KNN): {con_pl_acc:.2f}% | Avg Model Conf: {con_model_conf:.3f}")

        # 2b. 非共识集 (Non-Consensus Set) 诊断
        if len(non_consensus_indices) > 0:
            non_con_pl_acc_knn = (knn_pl[non_consensus_indices] == true_labels[non_consensus_indices]).float().mean().item() * 100
            non_con_pl_acc_model = (model_pl[non_consensus_indices] == true_labels[non_consensus_indices]).float().mean().item() * 100
            logger.info(f"  -> [Unreliable-NonConsens] Size: {len(non_consensus_indices):<5} | PL Acc (KNN): {non_con_pl_acc_knn:.2f}% | PL Acc (Model): {non_con_pl_acc_model:.2f}%")

    # 3. 全局意见不合诊断 (与之前相同，依然很有用)
    disagreement_mask = (model_pl != knn_pl)
    num_disagree = disagreement_mask.sum().item()
    if num_disagree > 0:
        indices = torch.where(disagreement_mask)[0]
        model_acc_on_disagree = (model_pl[indices] == true_labels[indices]).float().mean().item() * 100
        knn_acc_on_disagree = (knn_pl[indices] == true_labels[indices]).float().mean().item() * 100
        logger.info(f"  -> [Global Disagreement]   Samples: {num_disagree:<5} ({num_disagree/len(model_pl):.2%}) | Model Acc: {model_acc_on_disagree:.2f}% | KNN Acc: {knn_acc_on_disagree:.2f}%")

# ==============================================================================
#                      MAIN (MODIFIED FOR DUAL STATS)
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    
    # (确保 wandb 已登录)
    # try:
    #     wandb.login()
    # except:
    #     print("Wandb login failed. Set wandb mode to 'disabled'.")
    #     wandb.init(mode="disabled")
    wandb.init(mode="disabled")
    all_best_accuracies, all_final_epoch_accuracies, all_durations = [], [], []
    
    master_log_dir = os.path.join(args.out, args.exp_name)
    master_logger = setup_logger(master_log_dir, "master_log.txt", is_master=True)
    master_logger.info("========================= Starting Experiment Series =========================")
    master_logger.info(f"Base Settings: {vars(args)}\n" + "="*80)

    for i, seed in enumerate(args.seeds):
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        
        master_logger.info(f"--- Starting Run {i+1}/{len(args.seeds)} with Seed: {seed} ---")
        
        # <<< --- MODIFICATION START --- >>>
        best_acc, final_acc, duration = run_single_experiment(run_args)
        
        all_best_accuracies.append(best_acc)
        all_final_epoch_accuracies.append(final_acc) # 收集 Final Acc
        all_durations.append(duration)
        
        master_logger.info(f"--- Run {i+1} Finished. Duration: {duration/60.0:.2f} min | Best Acc: {best_acc:.2f}% | Final Acc: {final_acc:.2f}% ---\n")
        # <<< --- MODIFICATION END --- >>>
        
    # <<< --- MODIFICATION START --- >>>
    mean_best_acc = np.mean(all_best_accuracies)
    std_best_acc = np.std(all_best_accuracies)
    mean_final_acc = np.mean(all_final_epoch_accuracies) # 计算 Final Acc 均值
    std_final_acc = np.std(all_final_epoch_accuracies)   # 计算 Final Acc 标准差
    avg_duration_minutes = np.mean(all_durations) / 60
    # <<< --- MODIFICATION END --- >>>

    master_logger.info("========================= FINAL SUMMARY =========================")
    master_logger.info(f"Experiment Name: {args.exp_name}\n")
    master_logger.info(f"Average Run Duration: {avg_duration_minutes:.2f} min\n")
    
    # <<< --- MODIFICATION START --- >>>
    master_logger.info(f"Individual Best Accuracies: {[f'{acc:.2f}%' for acc in all_best_accuracies]}")
    master_logger.info(f"Individual Final Epoch Accuracies: {[f'{acc:.2f}%' for acc in all_final_epoch_accuracies]}")
    
    master_logger.info(f"--> Final Reported (Best Acc): {mean_best_acc:.2f}% ± {std_best_acc:.2f}%")
    master_logger.info(f"--> Final Reported (Final Epoch Acc): {mean_final_acc:.2f}% ± {std_final_acc:.2f}%")
    # <<< --- MODIFICATION END --- >>>
    
    master_logger.info("="*80)

"""  
python trept_unsup_mixup_crowd_pals_MH_knn_daes_totalGPU_all_lsr_backup_ablation.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_backup_ablation \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_MH_knn_daes_totalGPU_all_lsr_backup_ablation/CIFAR100cu1/Ablation_NoMixup/consensus_power2/exp_daes/cuda1/delta0.25_model_predict_2_p0.05n0.5_bs256_rand_lam_softtarget \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 2    \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 exp --sim_mode_2 daes  --num_workers 16     --no_unreliable_mixup
python ablation_exp_daes_GCK.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_backup_ablation \
    --exp_name PALS_softMatch/ablation_exp_daes_GCK/CIFAR100cu1/Ablation_NoMixup/consensus_power2/daes_daes/cuda1/delta0.25_model_predict_3_p0.0n0.5_bs256_rand_lam_softtarget1 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 daes --sim_mode_2 daes  --num_workers 16   

python ablation_exp_daes.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.5\
    --out ./out_backup_ablation \
    --exp_name PALS_softMatch/ablation_exp_daes/CIFAR10cu1/Ablation/daes_None/cuda1/delta0.25_model_predict_3_p0.5n0.5_bs256_rand_lam_softtarget \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 daes --sim_mode_2 None  --num_workers 16     


python ablation_exp_daes_GCK_rectify_daes_delete_parse_confidence.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_exp_daes_GCK_rectify_daes_delete_parse_confidence/CIFAR10cu0/Ablation/exp_daes/delta0.25_model_predict_3_p0.5n0.3_gs1.1 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 0 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 1.1       

python ablation_exp_daes_GCK_rectify_daes_delete_parse_confidence.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_exp_daes_GCK_rectify_daes_delete_parse_confidence/CIFAR10cu1/Ablation/exp_daes/delta0.25_model_predict_3_p0.5n0.3_gs0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 0.5 


python ablation_exp_daes_GCK_rectify_daes_delete_parse_confidence.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_exp_daes_GCK_rectify_daes_delete_parse_confidence/CIFAR100cu0/Ablation/exp_daes/delta0.25_model_predict_3_p0.1n0.0_bs256_rand_lam_softtarget_gs1.1_ema0.999_copy \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 0 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 1.1

python ablation_exp_daes_GCK_rectify_daes_delete_parse_confidence.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_exp_daes_GCK_rectify_daes_delete_parse_confidence/CIFAR100cu0/Ablation/exp_daes/delta0.25_model_predict_3_p0.05n0.5_bs256_rand_lam_softtarget_gs1.1_ema0.999_copy \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 0 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 1.1

python ablation_master_controlled_nomixup_consistency.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled_nomixup_consistency/CIFAR100/cu0exp_daes_delta0.25_p0.05n0.5_bs256_gs0.8_ema0.999 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 0 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 0.8  --no_consistency --no_reliable_mixup  --no_unreliable_mixup


python ablation_master_controlled_nomixup_consistency.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled_nomixup_consistency/CIFAR100/cu0exp_daes_delta0.25_p0.05n0.5_bs256_gs0.8_ema0.999baseline \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --consistency_weight 1.0 \
    --seeds 1 2 3   \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 exp  --sim_mode_2 daes  --num_workers 16   --gating_start_ratio 0.8  



python ablation_master_controlled.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/CUB200/cu0exp_Daes/delta0.25_model_predict_123_p0.05n0.2_bs64_rand_lam_no_cross_mixup_noncensus1.0_all_lsr_warm250_new_reliable \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs  250  \
    --cuda_dev 0

python ablation_master_controlled.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/CUB200/cu0exp_Daes/delta0.25_model_predict_123_p0.05n0.0_bs64_rand_lam_no_cross_mixup_noncensus1.0_all_lsr_warm250_new_reliable \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs  250  \
    --cuda_dev 0
python ablation_master_controlled.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.01 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/CUB200/cu1exp_Daes/delta0.25_model_predict_123_p0.01n0.0_bs64_rand_lam_no_cross_mixup_noncensus1.0_all_lsr_warm250_new_reliable \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs  250  \
    --cuda_dev 1    

python ablation_master_controlled.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 10 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/Treeversity/cu1exp_Daes/delta1.0_lsr0.10k5_e50lpi10 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.1 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
    --epochs  50  \
    --cuda_dev 1 --k_val 5

python ablation_master_controlled.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 10 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/Treeversity/cu1exp_Daes/delta1.0_lsr0.00k5_e50lpi10 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.00 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
    --epochs  50  \
    --cuda_dev 1 --k_val 5


python ablation_master_controlled.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 3 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/Treeversity/cu0exp_Daes/delta1.0_lsr0.10k5_e50lpi3 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.1 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
    --epochs  50  \
    --cuda_dev 0 --k_val 5

python ablation_master_controlled.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 3 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/Treeversity/cu0exp_Daes/delta1.0_lsr0.00k5_e50lpi3 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.00 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
    --epochs  50  \
    --cuda_dev 0 --k_val 5












python ablation_master_controlled.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 3 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/Benthic/cu1exp_Daeslpi3/delta1.0_lsr0.00gs0.8 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.0 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
    --epochs  100  \
    --cuda_dev 1  --gating_start_ratio 0.8
python ablation_master_controlled.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 10 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/Benthic/cu0exp_Daeslpi3/delta1.0_lsr0.00gs0.8k10 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.0 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
    --epochs  100  \
    --cuda_dev 0  --gating_start_ratio 0.8 --k_val 10

python ablation_master_controlled.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 10 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/Benthic/cu0exp_Daeslpi3/delta1.0_lsr0.00gs0.8k5 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.0 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
    --epochs  100  \
    --cuda_dev 0  --gating_start_ratio 0.8 --k_val 5    
python ablation_master_controlled.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 10 \
    --pr 0.1 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/ablation_master_controlled/Benthic/cu0exp_Daeslpi10/delta1.0_lsr0.10e30 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.1 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
    --epochs  50  \
    --cuda_dev 0     --lr_decay_epochs 30  --gating_start_ratio 0.8
  
    """

