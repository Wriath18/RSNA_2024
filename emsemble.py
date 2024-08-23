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
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import timm
import albumentations as A

# Configuration
rd = './data/rsnadata'
OUTPUT_DIR = 'rsna24-results'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 12
USE_AMP = True
SEED = 8620

IMG_SIZE = [192, 192]
IN_CHANS = 10
N_LABELS = 25
N_CLASSES = 3 * N_LABELS

EPOCHS = 20
MODEL_NAME = "edgenext_base.in21k_ft_in1k"

BATCH_SIZE = 16
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 3

LR = 1e-5
WD = 1e-2

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_random_seed(seed: int = 8620, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic

set_random_seed(SEED)

# Dataset class
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
            for key in views:
                transformed = []
                for i in range(views[key].shape[2]):
                    augmented = self.transform(image=views[key][..., i])
                    transformed.append(augmented['image'])
                views[key] = np.stack(transformed, axis=-1)

        for key in views:
            views[key] = torch.from_numpy(views[key].transpose(2, 0, 1)).float()
                
        return views, torch.from_numpy(label).float()

# Model classes
class RSNA25DModel(nn.Module):
    def __init__(self, model_name, in_c=10, n_classes=75, pretrained=True):
        super().__init__()
        
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
        
        self.classifier = nn.Linear(feature_dim, n_classes)

    def _create_branch(self, model_name, in_c, pretrained):
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_c,
            num_classes=0,
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

class EnsembleModel(nn.Module):
    def __init__(self, model_paths):
        super().__init__()
        self.models = nn.ModuleList([RSNA25DModel(MODEL_NAME, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=False) for _ in range(3)])
        for model, path in zip(self.models, model_paths):
            model.load_state_dict(torch.load(path))
            model.eval()

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

# Augmentations
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

# Training function for the ensemble model
from torch.cuda import amp  # Importing amp

# Inside your training function, update the autocast usage:

def train_ensemble_model(train_df, val_df, model_paths):
    print("Training Ensemble Model")
    
    transforms_train = get_transforms('train', IMG_SIZE)
    transforms_val = get_transforms('val', IMG_SIZE)
    
    train_ds = RSNA25DDataset(train_df, phase='train', transform=transforms_train)
    val_ds = RSNA25DDataset(val_df, phase='val', transform=transforms_val)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=N_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False, num_workers=N_WORKERS)

    ensemble_model = EnsembleModel(model_paths)
    ensemble_model.to(device)

    final_model = RSNA25DModel(MODEL_NAME, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=True)
    final_model.to(device)

    optimizer = AdamW(final_model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(device))

    best_val_loss = float('inf')
    early_stopping_counter = 0

    scaler = torch.cuda.amp.GradScaler()  # Update scaler initialization

    for epoch in range(1, EPOCHS + 1):
        # Training
        final_model.train()
        total_train_loss = 0

        for x, t in tqdm(train_dl, desc=f'Epoch {epoch} Training'):
            x = {k: v.to(device) for k, v in x.items()}
            t = t.long().to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Update autocast usage
                ensemble_preds = ensemble_model(x)
                final_preds = final_model(x)

                loss = 0
                for col in range(N_LABELS):
                    ensemble_pred = ensemble_preds[:, col * 3:col * 3 + 3]
                    final_pred = final_preds[:, col * 3:col * 3 + 3]
                    gt = t[:, col]

                    if torch.isnan(final_pred).any() or torch.isinf(final_pred).any():
                        print("NaNs or Infs found in final_pred")
                        continue

                    loss_gt = criterion(final_pred, gt)
                    loss_kd = nn.KLDivLoss(reduction='batchmean')(
                        F.log_softmax(final_pred / 2, dim=1),
                        F.softmax(ensemble_pred / 2, dim=1)
                    )
                    loss += (0.7 * loss_gt + 0.3 * loss_kd) / N_LABELS

            if not torch.isfinite(loss):
                print(f"Warning: non-finite loss, ending training {loss}")
                sys.exit(1)

            total_train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss = total_train_loss / len(train_dl)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}')

        # Validation
        final_model.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for x, t in tqdm(val_dl, desc=f'Epoch {epoch} Validation'):
                x = {k: v.to(device) for k, v in x.items()}
                t = t.long().to(device)

                with torch.cuda.amp.autocast():  # Update autocast usage
                    final_preds = final_model(x)

                    batch_loss = 0
                    for col in range(N_LABELS):
                        final_pred = final_preds[:, col * 3:col * 3 + 3]
                        gt = t[:, col]

                        if torch.isnan(final_pred).any() or torch.isinf(final_pred).any():
                            print("NaNs or Infs found in final_pred")
                            continue

                        col_loss = criterion(final_pred, gt)
                        
                        if torch.isnan(col_loss) or torch.isinf(col_loss):
                            print(f"Warning: NaN or Inf loss for column {col}")
                            print(f"Predictions: {final_pred}")
                            print(f"Ground truth: {gt}")
                        else:
                            batch_loss += col_loss / N_LABELS

                    if torch.isfinite(batch_loss):
                        total_val_loss += batch_loss.item()
                        val_batches += 1
                    else:
                        print(f"Warning: non-finite batch loss in validation: {batch_loss}")

        val_loss = total_val_loss / val_batches if val_batches > 0 else float('nan')

        print(f'Epoch {epoch}, Validation Loss: {val_loss:.6f}')

        if torch.isfinite(torch.tensor(val_loss)) and val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            print(f'Best validation loss updated to {best_val_loss:.6f}')
            torch.save(final_model.state_dict(), f'{OUTPUT_DIR}/best_ensemble_model.pt')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_EPOCH:
                print(f'Early stopping triggered after {epoch} epochs')
                break

    print("Training completed for the ensemble model.")



# Main execution
if __name__ == "__main__":
    df = pd.read_csv(f'{rd}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    model_paths = [f'{OUTPUT_DIR}/best_model_{i+1}.pt' for i in range(3)]
    train_ensemble_model(train_df, val_df, model_paths)

print("Ensemble model training completed.")