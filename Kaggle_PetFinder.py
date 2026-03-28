# -*- coding: utf-8 -*-
"""
Stable PetFinder Pawpularity Training Notebook
Includes:
- Logging
- Mixup
- Grad clip
- Freeze backbone
- Auto fc_img dimension
- EfficientNet-B2/B3 offline weights
"""

import os
import random
import logging
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from tqdm.auto import tqdm

# ==========================
# Logging 設定
# ==========================
logging.basicConfig(level=logging.INFO)

# ==========================
# Config
# ==========================
DATA_DIR = "/kaggle/input/competitions/petfinder-pawpularity-score"
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Load data
# ==========================
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
FEATURES = train.columns[1:-1]  # metadata columns

# ==========================
# Dataset
# ==========================
class PetDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.df = df
        self.is_test = is_test

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # simple augmentation
        if not self.is_test and random.random() < 0.5:
            image = cv2.flip(image, 1)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # C,H,W

        features = row[FEATURES].values.astype(np.float32)

        if self.is_test:
            return torch.tensor(image), torch.tensor(features, dtype=torch.float32)

        label = row["Pawpularity"] / 100.0
        return torch.tensor(image), torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ==========================
# Mixup
# ==========================
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ==========================
# Model
# ==========================
class PetModel(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b2", pretrained_path=None):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0
        )
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)

        # 自動抓 output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            out_dim = self.backbone(dummy).shape[1]
        logging.info(f"Backbone output dimension: {out_dim}")

        self.fc_img = nn.Linear(out_dim, 128)
        self.fc_meta = nn.Linear(len(FEATURES), 32)

        self.head = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
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

# ==========================
# Training / Validation
# ==========================
def train_fn(model, loader, optimizer, criterion):
    model.train()
    losses = []
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, meta, labels in pbar:
        imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
        imgs, y_a, y_b, lam = mixup(imgs, labels)

        optimizer.zero_grad()
        preds = model(imgs, meta)
        loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ← Grad clip
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return np.mean(losses)

def valid_fn(model, loader, criterion):
    model.eval()
    losses = []
    pbar = tqdm(loader, desc="Valid", leave=False)
    with torch.no_grad():
        for imgs, meta, labels in pbar:
            imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs, meta)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            pbar.set_postfix(val_loss=f"{loss.item():.4f}")
    return np.mean(losses)

"""
# ==========================
# KFold Training
# ==========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train["fold"] = pd.qcut(train["Pawpularity"], 10, labels=False)
models = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(train, train["fold"])):
    logging.info(f"===== FOLD {fold} =====")
    trn_df = train.iloc[trn_idx]
    val_df = train.iloc[val_idx]

    trn_loader = DataLoader(PetDataset(trn_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(PetDataset(val_df), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = PetModel(pretrained_path="/kaggle/input/datasets/neoccy/efficientnet-b2-pth/efficientnet_b2.pth").to(DEVICE)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        train_loss = train_fn(model, trn_loader, optimizer, criterion)
        val_loss = valid_fn(model, val_loader, criterion)
        scheduler.step()
        logging.info(f"Epoch {epoch} Train {train_loss:.4f} | Val {val_loss:.4f}")

    models.append(model)
"""

# ==========================
# KFold Training with fold progress
# ==========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train["fold"] = pd.qcut(train["Pawpularity"], 10, labels=False)
models = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(train, train["fold"])):
    logging.info(f"===== FOLD {fold} =====")
    trn_df = train.iloc[trn_idx]
    val_df = train.iloc[val_idx]

    trn_loader = DataLoader(PetDataset(trn_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(PetDataset(val_df), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = PetModel(pretrained_path="/kaggle/input/datasets/neoccy/efficientnet-b2-pth/efficientnet_b2.pth").to(DEVICE)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    fold_pbar = tqdm(range(EPOCHS), desc=f"Fold {fold} Epochs")  # 每個 fold 的 epoch 進度條
    for epoch in fold_pbar:
        train_loss = train_fn(model, trn_loader, optimizer, criterion)
        val_loss = valid_fn(model, val_loader, criterion)
        scheduler.step()
        fold_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

    models.append(model)


# ==========================
# Inference
# ==========================
test_loader = DataLoader(PetDataset(test, is_test=True), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
preds = []

for imgs, meta in test_loader:
    imgs, meta = imgs.to(DEVICE), meta.to(DEVICE)
    fold_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            p = model(imgs, meta)
            fold_preds.append(p.cpu().numpy())
    preds.append(np.mean(fold_preds, axis=0))

preds = np.concatenate(preds) * 100

# ==========================
# Submission
# ==========================
submission = pd.DataFrame({"Id": test["Id"], "Pawpularity": preds})
submission.to_csv("submission.csv", index=False)
logging.info("✅ submission.csv ready!")