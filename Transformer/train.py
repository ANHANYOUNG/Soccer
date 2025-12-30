"""
Transformer-based Pass Prediction Model - Training Script
Based on LSTM_2.ipynb architecture, converted to Transformer
"""

# file/path utilities
import os
import glob
from pathlib import Path
import pickle

# for data manipulation/math
import pandas as pd
import numpy as np
import random

# encoding (type_name to number) / split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

# progress bar
from tqdm import tqdm

# deep learning framework
import torch
from torch import nn
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import math

# ============================================================
# 0. Command-line Arguments (병렬 학습용)
# ============================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=None, help='Train specific fold only (0-4), None for all folds')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'overfit'], 
                    help='train: full training, overfit: single batch overfitting test')
parser.add_argument('--overfit_epochs', type=int, default=500, help='Epochs for overfit mode')
args, _ = parser.parse_known_args()

# ============================================================
# 1. Hyperparameters
# ============================================================

SEED = 42
SEEDS = [42, 123, 456]  # 3개 시드로 축소 (계산량 감소)

# cross-validation
N_SPLITS = 5
FOLD = args.fold if args.fold is not None else None  # CLI 인자 우선, 없으면 None (전체 fold)

# sequence length
K = 50
MIN_EVENTS = 2

# training parameters
EPOCHS = 150
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5

# model parameters
D_MODEL = 256          # Transformer hidden dimension
N_HEADS = 8            # Number of attention heads
N_LAYERS = 4           # Number of transformer encoder layers
DIM_FEEDFORWARD = 512  # Feed-forward network dimension
DROPOUT = 0.2          # Dropout rate
EMB_DIM = 16           # Embedding dimension for categorical features

# model options
USE_CLS_TOKEN = False  # CLS 토큰 미사용
USE_GAUSSIAN_NLL = True  # Gaussian NLL
PREDICT_DELTA = True  # Delta 예측 (상대 좌표)
USE_LAST_TOKEN = True  # Last-token pooling (압도적 성능 향상)

# augmentation parameters
USE_AUGMENT = False    # 증강 OFF
NOISE_STD = 0.5

# sequence truncation (마지막 K개 이벤트만 사용)
K_TRUNCATE = 50        # 최적값

# data loader parameters
NUM_WORKERS = 0

# paths
TRAIN_PATH = "../data/train.csv"
MODEL_SAVE_DIR = "./models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
print("Using device:", DEVICE)


# ============================================================
# 2. Data Loading and Preprocessing
# ============================================================

def load_and_preprocess_data(train_path):
    """Load and preprocess training data"""
    df = pd.read_csv(train_path)
    
    # sort events inside each episode by time, then action_id
    df = df.sort_values(["game_episode", "time_seconds", "action_id"]).reset_index(drop=True)
    
    # fill missing category text
    df["type_name"] = df["type_name"].fillna("__NA_TYPE__")
    df["result_name"] = df["result_name"].fillna("__NA_RES__")
    
    # change category text to idx(number)
    le_type = LabelEncoder()
    le_res = LabelEncoder()
    le_team = LabelEncoder()
    le_player = LabelEncoder()
    
    df["type_id"] = le_type.fit_transform(df["type_name"]) + 1
    df["res_id"] = le_res.fit_transform(df["result_name"]) + 1
    
    return df, le_type, le_res


def build_episodes(df, le_type, le_res):
    """Build episode sequences from dataframe"""
    
    # stadium constants
    STADIUM_X, STADIUM_Y = 105.0, 68.0
    CENTER_Y = STADIUM_Y / 2.0
    HALF_X = STADIUM_X / 2.0
    GOAL_X, GOAL_Y = STADIUM_X, CENTER_Y
    GOAL_POST_HALF = 3.66
    GOAL_Y_L = CENTER_Y - GOAL_POST_HALF
    GOAL_Y_R = CENTER_Y + GOAL_POST_HALF
    P_BOX_X_MIN = STADIUM_X - 16.5
    P_BOX_Y_MIN = CENTER_Y - 20.16
    P_BOX_Y_MAX = CENTER_Y + 20.16
    
    episodes = []
    targets = []
    episode_keys = []
    episode_game_ids = []
    
    for key, g in tqdm(df.groupby("game_episode"), desc="Building episodes"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        
        # target data is the last Pass event's end point
        if g.iloc[-1]["type_name"] != "Pass":
            pass_idxs = g.index[g["type_name"] == "Pass"]
            if len(pass_idxs) == 0:
                continue
            g = g.loc[:pass_idxs[-1]].reset_index(drop=True)
            if len(g) < 2:
                continue
        
        # target is the last event's end point
        tx, ty = float(g.loc[len(g)-1, "end_x"]), float(g.loc[len(g)-1, "end_y"])
        if np.isnan(tx) or np.isnan(ty):
            continue
        
        # compute dt inside episode
        t = g["time_seconds"].astype("float32").values
        dt = np.zeros_like(t, dtype="float32")
        dt[1:] = t[1:] - t[:-1]
        dt[dt < 0] = 0.0
        
        # extract start/end positions
        sx = g["start_x"].astype("float32").values
        sy = g["start_y"].astype("float32").values
        ex = g["end_x"].astype("float32").values
        ey = g["end_y"].astype("float32").values
        
        # leak-safe masking for last event's end
        ex_mask = ex.copy()
        ey_mask = ey.copy()
        ex_mask[-1] = 0.0
        ey_mask[-1] = 0.0
        
        # goal geometry features
        dxg = GOAL_X - sx
        dy_goal = np.maximum(0.0, np.maximum(GOAL_Y_L - sy, sy - GOAL_Y_R))
        dist_to_goal = np.sqrt(dxg**2 + dy_goal**2).astype("float32")
        
        # angles to goal posts
        alpha_L = np.arctan2(GOAL_Y_L - sy, dxg).astype("float32")
        alpha_R = np.arctan2(GOAL_Y_R - sy, dxg).astype("float32")
        theta_view = np.abs(alpha_R - alpha_L).astype("float32")
        
        # half line
        in_own_half = (sx < HALF_X).astype("float32")
        
        # penalty box
        dx_box = np.maximum(0.0, P_BOX_X_MIN - sx)
        dy_box = np.maximum(0.0, np.maximum(P_BOX_Y_MIN - sy, sy - P_BOX_Y_MAX))
        dist_p_box = np.sqrt(dx_box**2 + dy_box**2).astype("float32")
        
        # previous event features
        T = len(g)
        prev_dx = np.zeros(T, dtype="float32")
        prev_dy = np.zeros(T, dtype="float32")
        prev_valid = np.zeros(T, dtype="float32")
        
        if T > 1:
            dx_prev_raw = ex[:-1] - sx[:-1]
            dy_prev_raw = ey[:-1] - sy[:-1]
            prev_dx[1:] = dx_prev_raw
            prev_dy[1:] = dy_prev_raw
            prev_valid[1:] = 1.0
        
        # categorical idx per event
        type_id = g["type_id"].astype("int64").values
        res_id = g["res_id"].astype("int64").values
        
        # continuous features (12개)
        cont = np.stack([
            sx, sy, ex_mask, ey_mask, dt,
            dist_to_goal, theta_view, in_own_half,
            dist_p_box, prev_dx, prev_dy, prev_valid
        ], axis=1).astype("float32")
        
        # K truncation: 마지막 K개 이벤트만 사용 (최근 컨텍스트에 집중)
        if K_TRUNCATE is not None and len(cont) > K_TRUNCATE:
            cont = cont[-K_TRUNCATE:]
            type_id = type_id[-K_TRUNCATE:]
            res_id = res_id[-K_TRUNCATE:]
        
        episodes.append({
            "cont": cont,
            "type_id": type_id,
            "res_id": res_id,
            "last_start_x": sx[-1],  # 마지막 이벤트의 시작점 (delta 복원용)
            "last_start_y": sy[-1],
            "last_result_id": int(res_id[-1])  # 마지막 이벤트의 result (성공/실패)
        })
        
        # target: PREDICT_DELTA에 따라 절대좌표 또는 상대좌표
        if PREDICT_DELTA:
            # (dx, dy) = (end - start) of last event
            dx = tx - sx[-1]
            dy = ty - sy[-1]
            targets.append(np.array([dx, dy], dtype="float32"))
        else:
            targets.append(np.array([tx, ty], dtype="float32"))
        episode_keys.append(key)
        # game_id를 원본 컬럼에서 직접 가져옴 (문자열 파싱 오류 방지)
        episode_game_ids.append(str(g.iloc[0]["game_id"]))
    
    return episodes, targets, episode_keys, episode_game_ids


# ============================================================
# 3. Dataset and DataLoader
# ============================================================

class EpisodeDataset(Dataset):
    def __init__(self, episodes, targets, keys, augment=False, noise_std=0.0):
        self.episodes = episodes
        self.targets = targets
        self.keys = keys
        self.augment = augment
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        cont = ep["cont"].copy()
        
        if self.augment and self.noise_std > 0:
            noise = np.random.randn(cont.shape[0], 4).astype("float32") * self.noise_std
            cont[:, 0:4] += noise
            cont[:, 0] = np.clip(cont[:, 0], 0, 105)
            cont[:, 1] = np.clip(cont[:, 1], 0, 68)
            cont[:, 2] = np.clip(cont[:, 2], 0, 105)
            cont[:, 3] = np.clip(cont[:, 3], 0, 68)
        
        cont = torch.from_numpy(cont)
        type_id = torch.from_numpy(ep["type_id"])
        res_id = torch.from_numpy(ep["res_id"])
        y = torch.from_numpy(self.targets[idx])
        key = self.keys[idx]
        
        # 마지막 이벤트 정보: start_x, start_y, result_id
        last_start = torch.tensor([ep["last_start_x"], ep["last_start_y"]], dtype=torch.float32)
        last_result_id = torch.tensor(ep["last_result_id"], dtype=torch.long)
        
        return cont, type_id, res_id, y, key, last_start, last_result_id


def collate_fn(batch):
    conts, type_ids, res_ids, ys, keys, last_starts, last_result_ids = zip(*batch)
    lengths = torch.tensor([c.shape[0] for c in conts], dtype=torch.long)
    
    cont_pad = pad_sequence(conts, batch_first=True, padding_value=0.0)
    type_pad = pad_sequence(type_ids, batch_first=True, padding_value=0)
    res_pad = pad_sequence(res_ids, batch_first=True, padding_value=0)
    y = torch.stack(ys, dim=0).float()
    last_start = torch.stack(last_starts, dim=0).float()
    last_result_id = torch.stack(last_result_ids, dim=0).long()
    
    return cont_pad.float(), type_pad.long(), res_pad.long(), lengths, y, keys, last_start, last_result_id


# ============================================================
# 4. Transformer Model
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PassTransformer(nn.Module):
    """Transformer-based model for pass destination prediction"""
    
    def __init__(self, cont_dim, n_type, n_res, emb_dim=16, d_model=128, 
                 n_heads=8, n_layers=4, dim_feedforward=512, dropout=0.2,
                 use_cls_token=True, use_gaussian_nll=True, use_last_token=False):
        super().__init__()
        
        self.use_cls_token = use_cls_token
        self.use_gaussian_nll = use_gaussian_nll
        self.use_last_token = use_last_token
        
        # Embeddings for categorical features (type, result only)
        self.type_emb = nn.Embedding(n_type, emb_dim, padding_idx=0)
        self.res_emb = nn.Embedding(n_res, emb_dim, padding_idx=0)
        
        # Input projection (cont + type + res embeddings)
        in_dim = cont_dim + emb_dim + emb_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        
        # [CLS] token (learnable)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Attention pooling (used when not using CLS token and not using last token)
        if not use_cls_token and not use_last_token:
            self.attn_pool = nn.Linear(d_model, 1)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output dimension: 2 for (dx, dy), or 4 for Gaussian (mu_x, mu_y, log_var_x, log_var_y)
        out_dim = 4 if use_gaussian_nll else 2
        
        # Simple head (skip connection 제거)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim)
        )
        
        self.d_model = d_model
    
    def forward(self, cont_pad, type_pad, res_pad, lengths):
        B, T, _ = cont_pad.shape
        device = cont_pad.device
        
        # Embed categorical features
        te = self.type_emb(type_pad)  # (B, T, emb_dim)
        re = self.res_emb(res_pad)    # (B, T, emb_dim)
        
        # Concatenate all features
        x = torch.cat([cont_pad, te, re], dim=-1)  # (B, T, in_dim)
        
        # Project to d_model dimension
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Apply LayerNorm (패딩 영향 없음, 샘플별 독립 정규화)
        x = self.input_ln(x)  # (B, T, d_model)
        
        # Prepend [CLS] token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)
            T_new = T + 1
            # Update lengths for CLS token
            lengths_with_cls = lengths + 1
        else:
            T_new = T
            lengths_with_cls = lengths
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask (True for padding positions)
        idx = torch.arange(T_new, device=device).unsqueeze(0)  # (1, T_new)
        padding_mask = idx >= lengths_with_cls.unsqueeze(1)  # (B, T_new)
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # (B, T_new, d_model)
        
        # Pooling
        if self.use_cls_token:
            # Use [CLS] token output (first position)
            pooled = x[:, 0, :]  # (B, d_model)
        elif self.use_last_token:
            # Last-token pooling: 마지막 유효 토큰만 사용
            # lengths는 유효 토큰 수, 인덱스는 lengths-1
            last_indices = (lengths - 1).long()  # (B,)
            batch_idx = torch.arange(B, device=device)  # (B,)
            pooled = x[batch_idx, last_indices, :]  # (B, d_model)
        else:
            # Attention pooling (exclude CLS token position)
            attn_scores = self.attn_pool(x).squeeze(-1)  # (B, T)
            attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, d_model)
        
        # Layer norm
        pooled = self.layer_norm(pooled)
        
        # Output head
        out = self.head(pooled)  # (B, 2 or 4)
        
        return out


# ============================================================
# 5. Loss Functions
# ============================================================

class GaussianNLLLoss2D(nn.Module):
    """2D Gaussian Negative Log-Likelihood Loss"""
    def __init__(self, min_var=1e-4):
        super().__init__()
        self.min_var = min_var
    
    def forward(self, pred, target):
        # pred: (B, 4) -> (mu_x, mu_y, log_var_x, log_var_y)
        # target: (B, 2) -> (x, y)
        
        mu_x = pred[:, 0]
        mu_y = pred[:, 1]
        log_var_x = pred[:, 2]
        log_var_y = pred[:, 3]
        
        # Clamp log_var for numerical stability
        log_var_x = torch.clamp(log_var_x, min=-10, max=10)
        log_var_y = torch.clamp(log_var_y, min=-10, max=10)
        
        var_x = torch.exp(log_var_x) + self.min_var
        var_y = torch.exp(log_var_y) + self.min_var
        
        target_x = target[:, 0]
        target_y = target[:, 1]
        
        # NLL = 0.5 * (log(var) + (x - mu)^2 / var)
        nll_x = 0.5 * (log_var_x + (target_x - mu_x) ** 2 / var_x)
        nll_y = 0.5 * (log_var_y + (target_y - mu_y) ** 2 / var_y)
        
        return (nll_x + nll_y).mean()

# ============================================================
# 6. Training Functions
# ============================================================

def euclidean_sum_and_count(pred, true, last_start=None, predict_delta=False, use_gaussian=False):
    """
    Calculate Euclidean distance.
    If predict_delta=True, convert delta to absolute coordinates first.
    If use_gaussian=True, pred contains (mu_x, mu_y, log_var_x, log_var_y)
    """
    if use_gaussian:
        pred_xy = pred[:, :2]  # mu_x, mu_y only
    else:
        pred_xy = pred
    
    if predict_delta and last_start is not None:
        # Convert delta to absolute: end = start + delta
        pred_abs = pred_xy + last_start
        true_abs = true + last_start
    else:
        pred_abs = pred_xy
        true_abs = true
    
    # 클리핑: 경기장 범위 내로 제한 (0~105, 0~68)
    pred_clipped = pred_abs.clone()
    pred_clipped[:, 0] = torch.clamp(pred_clipped[:, 0], 0.0, 105.0)
    pred_clipped[:, 1] = torch.clamp(pred_clipped[:, 1], 0.0, 68.0)
    
    # 클리핑 전후 차이 계산 (튀는 예측 진단용)
    clip_diff = (pred_abs - pred_clipped).abs().sum().item()
    
    d = torch.sqrt(((pred_clipped - true_abs) ** 2).sum(dim=1))
    return d.sum().item(), d.numel(), clip_diff


def train_one_epoch(model, train_loader, optimizer, criterion, device, 
                    predict_delta=False, use_gaussian=False):
    model.train()
    
    tr_loss_sum = 0.0
    tr_loss_cnt = 0
    tr_euc_sum = 0.0
    tr_euc_cnt = 0
    tr_clip_diff = 0.0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for cont_pad, type_pad, res_pad, lengths, y, keys, last_start, last_result_id in pbar:
        cont_pad = cont_pad.to(device)
        type_pad = type_pad.to(device)
        res_pad = res_pad.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        last_start = last_start.to(device)
        last_result_id = last_result_id.to(device)
        
        optimizer.zero_grad()
        pred = model(cont_pad, type_pad, res_pad, lengths)
        
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        bsz = y.size(0)
        tr_loss_sum += loss.item() * bsz
        tr_loss_cnt += bsz
        
        e_sum, e_cnt, clip_d = euclidean_sum_and_count(
            pred.detach(), y, last_start, predict_delta, use_gaussian
        )
        tr_euc_sum += e_sum
        tr_euc_cnt += e_cnt
        tr_clip_diff += clip_d
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return tr_loss_sum / max(tr_loss_cnt, 1), tr_euc_sum / max(tr_euc_cnt, 1)


def validate(model, valid_loader, device, predict_delta=False, use_gaussian=False):
    model.eval()
    
    val_euc_sum = 0.0
    val_euc_cnt = 0
    val_clip_diff = 0.0
    
    pbar = tqdm(valid_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for cont_pad, type_pad, res_pad, lengths, y, keys, last_start, last_result_id in pbar:
            cont_pad = cont_pad.to(device)
            type_pad = type_pad.to(device)
            res_pad = res_pad.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            last_start = last_start.to(device)
            last_result_id = last_result_id.to(device)
            
            pred = model(cont_pad, type_pad, res_pad, lengths)
            e_sum, e_cnt, clip_d = euclidean_sum_and_count(
                pred, y, last_start, predict_delta, use_gaussian
            )
            val_euc_sum += e_sum
            val_euc_cnt += e_cnt
            val_clip_diff += clip_d
            
            pbar.set_postfix(dist=f"{(e_sum / max(e_cnt, 1)):.4f}")
    
    avg_dist = val_euc_sum / max(val_euc_cnt, 1)
    avg_clip = val_clip_diff / max(val_euc_cnt, 1)
    
    if avg_clip > 0.1:
        print(f"[WARNING] 튀는 예측 발견! 평균 클리핑 차이: {avg_clip:.2f}")
    
    return avg_dist


def overfit_single_batch(model, sample_batch, optimizer, criterion, device, 
                        epochs=500, predict_delta=False, use_gaussian=False):
    """
    Single batch overfitting test
    - 모델이 학습 가능한지 확인
    - Loss가 0에 가까워져야 정상
    """
    print("\n" + "=" * 60)
    print("SINGLE BATCH OVERFITTING TEST")
    print("=" * 60)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {sample_batch[0].shape[0]}")
    
    cont_pad, type_pad, res_pad, lengths, y, keys, last_start, last_result_id = sample_batch
    cont_pad = cont_pad.to(device)
    type_pad = type_pad.to(device)
    res_pad = res_pad.to(device)
    lengths = lengths.to(device)
    y = y.to(device)
    last_start = last_start.to(device)
    last_result_id = last_result_id.to(device)
    
    model.train()
    
    losses = []
    dists = []
    
    print("\n[Training Progress]")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(cont_pad, type_pad, res_pad, lengths)
        
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate distance
        with torch.no_grad():
            e_sum, e_cnt, _ = euclidean_sum_and_count(
                pred, y, last_start, predict_delta, use_gaussian
            )
            dist = e_sum / e_cnt
        
        losses.append(loss.item())
        dists.append(dist)
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [{epoch:03d}] loss={loss.item():.6f}, dist={dist:.4f}")
    
    # Diagnosis
    print("\n[Diagnosis]")
    first_loss = losses[0]
    last_loss = losses[-1]
    min_loss = min(losses)
    
    print(f"  First loss: {first_loss:.6f}")
    print(f"  Last loss:  {last_loss:.6f}")
    print(f"  Min loss:   {min_loss:.6f}")
    print(f"  First dist: {dists[0]:.4f}")
    print(f"  Last dist:  {dists[-1]:.4f}")
    
    if last_loss > first_loss * 1.5:
        print("\n  ⚠️ ERROR GOES UP! Check:")
        print("     - Loss function sign")
        print("     - Learning rate too high")
    elif any(np.isnan(losses)) or any(np.isinf(losses)):
        print("\n  ⚠️ ERROR EXPLODES! Check:")
        print("     - Lower learning rate")
        print("     - Check for NaN/Inf in data")
    elif last_loss > first_loss * 0.5:
        print("\n  ⚠️ ERROR PLATEAUS! Try:")
        print("     - Higher learning rate")
        print("     - More epochs")
    elif last_loss < 0.01:
        print("\n  ✓ Successfully overfitting! Model can learn.")
    else:
        print("\n  △ Partial overfit. May need more epochs.")
    
    return losses, dists


# ============================================================
# 7. Main Training Loop
# ============================================================

def train_single_seed(seed, episodes, targets, episode_keys, episode_game_ids, 
                      le_type, le_res, seed_idx=0, total_seeds=1, fold=None, mode='train'):
    """Train a model with a specific seed and fold"""
    
    print(f"\n{'='*60}")
    if fold is not None:
        print(f"Training seed {seed} ({seed_idx+1}/{total_seeds}) - Fold {fold} - Mode: {mode.upper()}")
    else:
        print(f"Training seed {seed} ({seed_idx+1}/{total_seeds}) - All folds - Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    # Set seed
    seed_everything(seed)
    
    # Create model save directory for this seed and fold
    if fold is not None:
        seed_save_dir = os.path.join(MODEL_SAVE_DIR, f"seed_{seed}_fold_{fold}")
    else:
        seed_save_dir = os.path.join(MODEL_SAVE_DIR, f"seed_{seed}")
    os.makedirs(seed_save_dir, exist_ok=True)
    
    # Split train/valid
    episode_game_ids_arr = np.array(episode_game_ids, dtype=str)  # 문자열로 유지
    unique_games = np.unique(episode_game_ids_arr)
    print(f"[GroupKFold] 총 {len(unique_games)}개 경기(game_id), {len(episodes)}개 에피소드")
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    # fold가 지정된 경우 해당 fold만, None이면 첫 번째 fold (0)
    target_fold = fold if fold is not None else 0
    
    tr_idx, va_idx = None, None
    for fold_i, (tr, va) in enumerate(
        gkf.split(np.zeros(len(episodes)), np.zeros(len(episodes)), groups=episode_game_ids_arr)
    ):
        if fold_i == target_fold:
            tr_idx, va_idx = tr, va
            break
    
    assert tr_idx is not None and va_idx is not None
    
    # 그룹 분포 확인
    train_games = set(episode_game_ids_arr[tr_idx])
    valid_games = set(episode_game_ids_arr[va_idx])
    overlap = train_games & valid_games
    if len(overlap) > 0:
        print(f"[WARNING] Train/Valid game_id 겹침 발생! {len(overlap)}개")
    else:
        print(f"[OK] Train {len(train_games)}개 경기, Valid {len(valid_games)}개 경기 (겹침 없음)")
    
    train_eps = [episodes[i] for i in tr_idx]
    train_tg = [targets[i] for i in tr_idx]
    train_keys = [episode_keys[i] for i in tr_idx]
    
    valid_eps = [episodes[i] for i in va_idx]
    valid_tg = [targets[i] for i in va_idx]
    valid_keys = [episode_keys[i] for i in va_idx]
    
    print(f"Train episodes: {len(train_eps)} | Valid episodes: {len(valid_eps)}")
    
    # Create datasets and dataloaders
    # USE_AUGMENT 플래그로 증강 제어
    train_ds = EpisodeDataset(train_eps, train_tg, train_keys, augment=USE_AUGMENT, noise_std=NOISE_STD)
    valid_ds = EpisodeDataset(valid_eps, valid_tg, valid_keys, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    n_type = len(le_type.classes_) + 1
    n_res = len(le_res.classes_) + 1
    cont_dim = episodes[0]["cont"].shape[1]
    
    model = PassTransformer(
        cont_dim=cont_dim,
        n_type=n_type,
        n_res=n_res,
        emb_dim=EMB_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        use_cls_token=USE_CLS_TOKEN,
        use_gaussian_nll=USE_GAUSSIAN_NLL,
        use_last_token=USE_LAST_TOKEN
    ).to(DEVICE)
    
    # Print model info (only first seed)
    if seed_idx == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"n_type: {n_type}, n_res: {n_res}, cont_dim: {cont_dim}")
        print(f"USE_CLS_TOKEN: {USE_CLS_TOKEN}, USE_GAUSSIAN_NLL: {USE_GAUSSIAN_NLL}, PREDICT_DELTA: {PREDICT_DELTA}, USE_LAST_TOKEN: {USE_LAST_TOKEN}")
    
    # Save model config
    config = {
        "cont_dim": cont_dim,
        "n_type": n_type,
        "n_res": n_res,
        "emb_dim": EMB_DIM,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "use_cls_token": USE_CLS_TOKEN,
        "use_gaussian_nll": USE_GAUSSIAN_NLL,
        "predict_delta": PREDICT_DELTA,
        "use_last_token": USE_LAST_TOKEN
    }
    with open(os.path.join(seed_save_dir, "model_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    # Criterion
    if USE_GAUSSIAN_NLL:
        criterion = GaussianNLLLoss2D()
    else:
        criterion = nn.SmoothL1Loss()
    
    # Optimizer, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # ============================================================
    # MODE: overfit - Single batch overfitting test
    # ============================================================
    if mode == 'overfit':
        print("\n[Overfit Mode] Testing single batch overfitting...")
        # Get one batch
        sample_batch = next(iter(train_loader))
        
        losses, dists = overfit_single_batch(
            model, sample_batch, optimizer, criterion, DEVICE,
            epochs=args.overfit_epochs,
            predict_delta=PREDICT_DELTA,
            use_gaussian=USE_GAUSSIAN_NLL
        )
        
        print("\n" + "=" * 60)
        print("OVERFIT TEST COMPLETE!")
        print("=" * 60)
        
        # Return dummy values for consistency
        return dists[-1], seed_save_dir
    
    # ============================================================
    # MODE: train - Full training
    # ============================================================
    # Training loop
    best_val = float('inf')
    best_state = None
    
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_euc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            predict_delta=PREDICT_DELTA, use_gaussian=USE_GAUSSIAN_NLL
        )
        val_dist = validate(
            model, valid_loader, DEVICE,
            predict_delta=PREDICT_DELTA, use_gaussian=USE_GAUSSIAN_NLL
        )
        
        scheduler.step(val_dist)
        current_lr = optimizer.param_groups[0]['lr']
        
        gap = val_dist - tr_euc
        print(f"[epoch {epoch:03d}] lr={current_lr:.2e} | "
              f"train_loss={tr_loss:.4f} | train_dist={tr_euc:.4f} | "
              f"valid_dist={val_dist:.4f} | gap={gap:.4f}")
        
        if val_dist < best_val:
            best_val = val_dist
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  >> New best! Saving model...")
            torch.save(best_state, os.path.join(seed_save_dir, "best_model.pt"))
    
    print(f"Seed {seed} - Best validation distance: {best_val:.4f}")
    
    return best_val, seed_save_dir


def main():
    print("=" * 60)
    print("Transformer Pass Prediction - Training")
    print("=" * 60)
    
    # Create model save directory
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data...")
    df, le_type, le_res = load_and_preprocess_data(TRAIN_PATH)
    
    # Build episodes
    print("\n[2/4] Building episodes...")
    episodes, targets, episode_keys, episode_game_ids = build_episodes(df, le_type, le_res)
    print(f"Total episodes: {len(episodes)}")
    
    # Save label encoders for inference (one copy at root)
    with open(os.path.join(MODEL_SAVE_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump({
            "le_type": le_type, 
            "le_res": le_res
        }, f)
    
    # Train with multiple seeds and folds for ensemble
    print(f"\n[3/4] Training with {len(SEEDS)} seeds - Mode: {args.mode.upper()}")
    if FOLD is None:
        if args.mode == 'overfit':
            print(f"  Overfit mode: Testing ONLY fold 0")
            print(f"  Total tests: {len(SEEDS)} seeds × 1 fold = {len(SEEDS)} tests")
        else:
            print(f"  Training ALL {N_SPLITS} folds (5-fold cross-validation)")
            print(f"  Total models: {len(SEEDS)} seeds × {N_SPLITS} folds = {len(SEEDS) * N_SPLITS} models")
    else:
        print(f"  Training ONLY fold {FOLD}")
        print(f"  Total models: {len(SEEDS)} seeds × 1 fold = {len(SEEDS)} models")
    
    results = []
    
    if args.mode == 'overfit':
        # Overfit mode: 첫 번째 시드, 첫 번째 fold만 테스트
        seed = SEEDS[0]
        fold = FOLD if FOLD is not None else 0
        best_val, seed_save_dir = train_single_seed(
            seed, episodes, targets, episode_keys, episode_game_ids,
            le_type, le_res, 0, 1, fold=fold, mode='overfit'
        )
        results.append({
            "seed": seed,
            "fold": fold,
            "best_val": best_val,
            "path": seed_save_dir
        })
    elif FOLD is None:
        # 5-fold 전체 학습
        for seed_idx, seed in enumerate(SEEDS):
            for fold in range(N_SPLITS):
                best_val, seed_save_dir = train_single_seed(
                    seed, episodes, targets, episode_keys, episode_game_ids,
                    le_type, le_res, seed_idx, len(SEEDS), fold=fold, mode=args.mode
                )
                results.append({
                    "seed": seed,
                    "fold": fold,
                    "best_val": best_val,
                    "path": seed_save_dir
                })
    else:
        # 특정 fold만 학습
        for seed_idx, seed in enumerate(SEEDS):
            best_val, seed_save_dir = train_single_seed(
                seed, episodes, targets, episode_keys, episode_game_ids,
                le_type, le_res, seed_idx, len(SEEDS), fold=FOLD, mode=args.mode
            )
            results.append({
                "seed": seed,
                "fold": FOLD,
                "best_val": best_val,
                "path": seed_save_dir
            })
    
    # Summary
    print(f"\n{'='*60}")
    if args.mode == 'overfit':
        print("Overfit Test Summary")
    else:
        print("Training Summary")
    print(f"{'='*60}")
    
    if args.mode == 'overfit':
        # Overfit mode: 결과만 출력
        for r in results:
            print(f"  Seed {r['seed']}, Fold {r['fold']}: final_dist = {r['best_val']:.4f}")
    elif FOLD is None:
        # 5-fold 전체: fold별로 그룹화
        for fold in range(N_SPLITS):
            fold_results = [r for r in results if r['fold'] == fold]
            if fold_results:
                avg_val = sum(r['best_val'] for r in fold_results) / len(fold_results)
                print(f"  Fold {fold}: {len(fold_results)} seeds, avg_val={avg_val:.4f}")
                for r in fold_results:
                    print(f"    - Seed {r['seed']}: {r['best_val']:.4f}")
        
        # 전체 평균
        overall_avg = sum(r['best_val'] for r in results) / len(results)
        print(f"\n  Overall average: {overall_avg:.4f} ({len(results)} models)")
    else:
        # 특정 fold만
        for r in results:
            print(f"  Seed {r['seed']}: valid_dist = {r['best_val']:.4f}")
        avg_val = sum(r['best_val'] for r in results) / len(results)
        print(f"\n  Average validation distance: {avg_val:.4f}")
    
    print(f"\n  Models saved to: {MODEL_SAVE_DIR}/seed_*_fold_*/best_model.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
