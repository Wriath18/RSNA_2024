import os
import gc
import sys
from PIL import Image
import cv2
import math, random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import timm
from transformers import get_cosine_schedule_with_warmup

import albumentations as A


rd = './data/rsnadata'


NOT_DEBUG = True # True -> run naormally, False -> debug mode, with lesser computing cost

OUTPUT_DIR = f'rsna24-results'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
# N_WORKERS = os.cpu_count() 
N_WORKERS = 4
USE_AMP = True # can change True if using T4 or newer than Ampere
SEED = 8620

IMG_SIZE = [192, 192]
IN_CHANS = 10
N_LABELS = 25
N_CLASSES = 3 * N_LABELS

AUG_PROB = 0.75

N_FOLDS = 5 if NOT_DEBUG else 2
EPOCHS = 20 if NOT_DEBUG else 2
MODEL_NAME = "edgenext_base.in21k_ft_in1k"

GRAD_ACC = 2
TGT_BATCH_SIZE = 32
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 3

LR = 1e-3 * TGT_BATCH_SIZE / 32
WD = 1e-2
AUG = True


os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_random_seed(seed: int = 8620, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore

set_random_seed(SEED)


df = pd.read_csv(f'{rd}/train.csv')
df.head()


df = df.fillna(-100)


label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}
df = df.replace(label2id)
df.head()


CONDITIONS = [
    'Spinal Canal Stenosis', 
    'Left Neural Foraminal Narrowing', 
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

LEVELS = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1',
]


class RSNA25DDataset(Dataset):
    def __init__(self, df, phase='train', transform=None):
        self.df = df
        self.phase = phase
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        views = {
            'sagittal_t1': np.zeros((192, 192, 10), dtype=np.float32),
            'sagittal_t2_stir': np.zeros((192, 192, 10), dtype=np.float32),
            'axial_t2': np.zeros((192, 192, 10), dtype=np.float32)
        }
        
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.float32)
        
        # Load Sagittal T1
        for i in range(10):
            try:
                p = f'./cvt_png/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                views['sagittal_t1'][..., i] = np.array(img) / 255.0
            except:
                pass
            
        # Load Sagittal T2/STIR
        for i in range(10):
            try:
                p = f'./cvt_png/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                views['sagittal_t2_stir'][..., i] = np.array(img) / 255.0
            except:
                pass
            
        # Load Axial T2
        axt2 = sorted(glob(f'./cvt_png/{st_id}/Axial T2/*.png'))
        step = len(axt2) / 10.0
        st = len(axt2)/2.0 - 4.0*step
        end = len(axt2) + 0.0001
                
        for i, j in enumerate(np.arange(st, end, step)):
            try:
                p = axt2[max(0, int((j-0.5001).round()))]
                img = Image.open(p).convert('L')
                views['axial_t2'][..., i] = np.array(img) / 255.0
            except:
                pass
            
        assert any(np.sum(view) > 0 for view in views.values())
            
        if self.transform is not None:
            # Apply the same transform to all slices of each view
            for key in views:
                transformed = []
                for i in range(views[key].shape[2]):
                    augmented = self.transform(image=views[key][..., i])
                    transformed.append(augmented['image'])
                views[key] = np.stack(transformed, axis=-1)

        # Convert to torch tensors and change to (C, H, W) format
        for key in views:
            views[key] = torch.from_numpy(views[key].transpose(2, 0, 1)).float()
                
        return views, torch.from_numpy(label).float()

# Define augmentations
def get_transforms(phase, img_size):
    list_transforms = []
    if phase == "train":
        list_transforms.extend([
            A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.9, 1), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.75),
        ])
    
    list_transforms.extend([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    return A.Compose(list_transforms)

# Usage
img_size = (192, 192)
transforms_train = get_transforms('train', img_size)
transforms_val = get_transforms('val', img_size)


class RSNA25DModel(nn.Module):
    def __init__(self, model_name, in_c=10, n_classes=75, pretrained=True):
        super().__init__()
        
        # Create separate branches for each view
        self.sagittal_t1_branch = self._create_branch(model_name, in_c, pretrained)
        self.sagittal_t2_stir_branch = self._create_branch(model_name, in_c, pretrained)
        self.axial_t2_branch = self._create_branch(model_name, in_c, pretrained)
        

        with torch.no_grad():
            dummy_input = torch.randn(1, in_c, 224, 224) 
            feature_dim = self.sagittal_t1_branch(dummy_input).shape[1]
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Final classification layer
        self.classifier = nn.Linear(feature_dim, n_classes)

    def _create_branch(self, model_name, in_c, pretrained):
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_c,
            num_classes=0,  # Remove the classification head
            global_pool='avg'
        )

    def forward(self, x):
        f_sagittal_t1 = self.sagittal_t1_branch(x['sagittal_t1'])
        f_sagittal_t2_stir = self.sagittal_t2_stir_branch(x['sagittal_t2_stir'])
        f_axial_t2 = self.axial_t2_branch(x['axial_t2'])
        

        combined_features = torch.cat([f_sagittal_t1, f_sagittal_t2_stir, f_axial_t2], dim=1)
        

        fused_features = self.fusion(combined_features)
        

        output = self.classifier(fused_features).float()
        
        return output

df = pd.read_csv(f'{rd}/train.csv')
df = df.fillna(-100)
df = df.replace(label2id)

# Split the data into three parts: 40%, 40%, and 20%
# train_df1, temp_df = train_test_split(df, test_size=0.6, random_state=SEED)
# train_df2, train_df3 = train_test_split(temp_df, test_size=0.33333, random_state=SEED)

# # For the third model, add 20% randomly chosen from the used 80%
# additional_data = pd.concat([train_df1, train_df2]).sample(frac=0.25, random_state=SEED)
# train_df3 = pd.concat([train_df3, additional_data]).drop_duplicates()

# train_dfs = [train_df1, train_df2, train_df3]



autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

train_dfs = []
val_dfs = []
for i in range(3):
    if i < 2:
        train_df, val_df = train_test_split(df.sample(frac=0.4, random_state=SEED+i), 
                                            test_size=0.2, random_state=SEED)
        train_dfs.append(train_df)
        val_dfs.append(val_df)
    else:
        train_df, val_df = train_test_split(df.sample(frac=0.2, random_state=SEED+i), 
                                            test_size=0.2, random_state=SEED)
        additional_data = pd.concat([train_dfs[0], train_dfs[1]]).sample(frac=0.25, random_state=SEED)
        train_df = pd.concat([train_df, additional_data]).drop_duplicates()
        train_dfs.append(train_df)
        val_dfs.append(val_df)

autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

for model_num, (train_df, val_df) in enumerate(zip(train_dfs, val_dfs), 1):
    print('#' * 30)
    print(f'Training Model {model_num}')
    print('#' * 30)
    print(f'Training data size: {len(train_df)}, Validation data size: {len(val_df)}')
    
    train_ds = RSNA25DDataset(train_df, phase='train', transform=transforms_train)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=N_WORKERS
    )
    
    val_ds = RSNA25DDataset(val_df, phase='val', transform=transforms_val)
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=N_WORKERS
    )

    model = RSNA25DModel(MODEL_NAME, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=True)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

    warmup_steps = EPOCHS / 10 * len(train_dl) // GRAD_ACC
    num_total_steps = EPOCHS * len(train_dl) // GRAD_ACC
    num_cycles = 0.475

    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f'start epoch {epoch}')
        model.train()
        total_loss = 0
        with tqdm(train_dl, leave=True) as pbar:
            optimizer.zero_grad()
            for idx, (x, t) in enumerate(pbar):  
                x = {k: v.to(device) for k, v in x.items()}
                t = t.long().to(device)
                
                with autocast:
                    loss = 0
                    y = model(x)
                    for col in range(N_LABELS):
                        pred = y[:, col * 3:col * 3 + 3]
                        gt = t[:, col]
                        loss = loss + criterion(pred, gt) / N_LABELS
                        
                    total_loss += loss.item()
                    if GRAD_ACC > 1:
                        loss = loss / GRAD_ACC
    
                if not math.isfinite(loss):
                    print(f"Loss is {loss}, stopping training")
                    sys.exit(1)
    
                pbar.set_postfix(
                    OrderedDict(
                        loss=f'{loss.item() * GRAD_ACC:.6f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                    )
                )
                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)
                
                if (idx + 1) % GRAD_ACC == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()                 
    
        train_loss = total_loss / len(train_dl)
        print(f'train_loss:{train_loss:.6f}')

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, t in tqdm(val_dl):
                x = {k: v.to(device) for k, v in x.items()}
                t = t.long().to(device)
                
                with autocast:
                    y = model(x)
                    loss = 0
                    for col in range(N_LABELS):
                        pred = y[:, col * 3:col * 3 + 3]
                        gt = t[:, col]
                        loss = loss + criterion(pred, gt) / N_LABELS
                    
                    val_loss += loss.item()

        val_loss /= len(val_dl)
        print(f'val_loss:{val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            print(f'epoch:{epoch}, best val loss updated to {best_val_loss:.6f}')
            fname = f'{OUTPUT_DIR}/best_model_{model_num}.pt'
            torch.save(model.state_dict(), fname)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_EPOCH:
                print(f'Early stopping triggered after {epoch} epochs')
                break

    print(f'Training completed for Model {model_num}')
    del model, optimizer, train_ds, train_dl, val_ds, val_dl
    gc.collect()
    torch.cuda.empty_cache()

print("Training completed for all three models.")