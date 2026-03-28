# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:49:42 2026

@author: a0936
"""

# ============================================
# PetFinder Pawpularity - Strong Baseline
# CNN + Tabular + KFold
# ============================================

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from tqdm.auto import tqdm
import random


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "/kaggle/input/competitions/petfinder-pawpularity-score"
IMG_SIZE = 224
BATCH_SIZE = 128 #32
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load data
# -----------------------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

FEATURES = train.columns[1:-1]  # metadata

"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),          # 0~1 或 mean/std
    ToTensorV2()
])
"""


# -----------------------------
# Dataset
# -----------------------------
class PetDataset(Dataset):
    def __init__(self, df, is_test=False, transform=None):
        self.df = df
        self.is_test = is_test
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(
            DATA_DIR,
            "train" if not self.is_test else "test",
            f"{row['Id']}.jpg"
        )

        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Image not found: {img_path}")
        
        # ✅ 先轉 RGB（這時還是 uint8）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 🔥 augmentation（簡單但有效）
        if not self.is_test:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
        
        # ✅ 再 resize
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # ✅ 再轉 float32（不是 float64！）
        image = image.astype(np.float32) / 255.0
        
        # ✅ 最後轉 tensor format
        image = np.transpose(image, (2, 0, 1))
        """
        #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        #image = image / 255.0
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        if self.transform:
            image = self.transform(image=image)['image']

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        """
        
        features = row[FEATURES].values.astype(np.float32)
        
        if self.is_test:
            #return torch.tensor(image), torch.tensor(features)
            return image, torch.tensor(features, dtype=torch.float32)

        label = row["Pawpularity"] / 100.0
        #return torch.tensor(image), torch.tensor(features), torch.tensor(label)
        return image, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        

# -----------------------------
# Model
# -----------------------------
class PetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b2",
            pretrained=False,
            num_classes=0
        )        

        # 🔥 手動載入權重
        state_dict = torch.load(
            "/kaggle/input/datasets/neoccy/efficientnet-b2-pth/efficientnet_b2.pth",
            map_location="cpu"
        )

        self.backbone.load_state_dict(state_dict, strict=False)


        
        self.fc_img = nn.Linear(1408, 128)
        self.fc_meta = nn.Linear(len(FEATURES), 32)
        
        """
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        """

        self.head = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),     # 🔥 新增
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, meta):
        img_feat = self.backbone(x)
        img_feat = self.fc_img(img_feat)

        meta_feat = self.fc_meta(meta)

        x = torch.cat([img_feat, meta_feat], dim=1)
        x = self.head(x)

        return x.squeeze(1)


# -----------------------------
# Mixup（在這裡🔥）
# -----------------------------
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam



# -----------------------------
# Train function
# -----------------------------

"""
def train_fn(model, loader, optimizer, criterion):
    model.train()
    losses = []

    for imgs, meta, labels in loader:
        imgs, meta, labels = imgs.to(DEVICE).float(), meta.to(DEVICE).float(), labels.to(DEVICE).float()

        optimizer.zero_grad()
        preds = model(imgs, meta)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)
"""
def train_fn(model, loader, optimizer, criterion):
    model.train()
    losses = []

    pbar = tqdm(loader, desc="Train", leave=False)

    for imgs, meta, labels in pbar:
        imgs = imgs.to(DEVICE).float()
        meta = meta.to(DEVICE).float()
        labels = labels.to(DEVICE).float()

        # 🔥 在這裡（mixup）
        imgs, y_a, y_b, lam = mixup(imgs, labels)
        
        optimizer.zero_grad()
        preds = model(imgs, meta)

        # 🔥 loss 改這裡
        loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)
        #loss = criterion(preds, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 🔥
        optimizer.step()

        losses.append(loss.item())

        # 🔥 顯示 loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return np.mean(losses)



def valid_fn(model, loader, criterion):
    model.eval()
    losses = []

    pbar = tqdm(loader, desc="Valid", leave=False)

    with torch.no_grad():
        for imgs, meta, labels in pbar:
            imgs = imgs.to(DEVICE).float()
            meta = meta.to(DEVICE).float()
            labels = labels.to(DEVICE).float()

            preds = model(imgs, meta)
            loss = criterion(preds, labels)

            losses.append(loss.item())

            # 🔥 顯示 val loss
            pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    return np.mean(losses)


# -----------------------------
# KFold training
# -----------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train["fold"] = pd.qcut(train["Pawpularity"], 10, labels=False)

models = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(train, train["fold"])):
    print(f"\n===== FOLD {fold} =====")
    print(f"Fold {fold}")

    trn_df = train.iloc[trn_idx]
    val_df = train.iloc[val_idx]

    trn_loader = DataLoader(PetDataset(trn_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(PetDataset(val_df), batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    model = PetModel().to(DEVICE)
    # ✅ 冻结 backbone（加在這裡）
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        loss = train_fn(model, trn_loader, optimizer, criterion)
        val_loss = valid_fn(model, val_loader, criterion)
        scheduler.step()   # ✅ 每個 epoch 呼叫    
        print(f"Epoch {epoch} Train {train_loss:.4f} | Val {val_loss:.4f}")
        #print(f"Epoch {epoch} Loss {loss:.4f}")

    models.append(model)

# -----------------------------
# Inference
# -----------------------------
test_loader = DataLoader(PetDataset(test, is_test=True), batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

preds = []

for imgs, meta in test_loader:
    imgs, meta = imgs.to(DEVICE).float(), meta.to(DEVICE).float()

    fold_preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            p = model(imgs, meta)
            fold_preds.append(p.cpu().numpy())

    preds.append(np.mean(fold_preds, axis=0))

preds = np.concatenate(preds)
preds = preds * 100

# -----------------------------
# Submission
# -----------------------------
submission = pd.DataFrame({
    "Id": test["Id"],
    "Pawpularity": preds
})

submission.to_csv("submission.csv", index=False)

print("✅ submission.csv ready!")
