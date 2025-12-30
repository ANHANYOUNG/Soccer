"""
OOF (Out-of-Fold) Analysis Script
K 값에 따른 5-fold 평균/표준편차 비교로 최적 K 결정
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import math

# ============================================================
# Configuration
# ============================================================

# K values to compare
K_VALUES = [12, 20, 32, 50]

# 단일 K만 실행하려면 커맨드라인에서 지정: python oof_analysis.py --k 12 --gpu 0
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=None, help='Run only this K value')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
ARGS, _ = parser.parse_known_args()

# Fixed settings (submit_1 best settings)
SEED = 42
N_SPLITS = 5
EPOCHS = 100  # 분석용으로 줄임 (빠른 실험)
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Model settings
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.2
EMB_DIM = 16

USE_CLS_TOKEN = False
USE_GAUSSIAN_NLL = True
PREDICT_DELTA = True
USE_LAST_TOKEN = True
USE_SKIP_START = False  # submit_1 기준 OFF

TRAIN_PATH = "../data/train.csv"
DEVICE = f"cuda:{ARGS.gpu}" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Model & Data (simplified from train.py)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PassTransformer(nn.Module):
    def __init__(self, cont_dim, n_type, n_res, emb_dim=16, d_model=128, 
                 n_heads=8, n_layers=4, dim_feedforward=512, dropout=0.2,
                 use_cls_token=False, use_gaussian_nll=True, use_last_token=True):
        super().__init__()
        
        self.use_cls_token = use_cls_token
        self.use_gaussian_nll = use_gaussian_nll
        self.use_last_token = use_last_token
        
        self.type_emb = nn.Embedding(n_type, emb_dim, padding_idx=0)
        self.res_emb = nn.Embedding(n_res, emb_dim, padding_idx=0)
        
        in_dim = cont_dim + emb_dim + emb_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        if not use_cls_token and not use_last_token:
            self.attn_pool = nn.Linear(d_model, 1)
        
        self.layer_norm = nn.LayerNorm(d_model)
        out_dim = 4 if use_gaussian_nll else 2
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim)
        )
        self.d_model = d_model
    
    def forward(self, cont_pad, type_pad, res_pad, lengths):
        B, T, _ = cont_pad.shape
        device = cont_pad.device
        
        te = self.type_emb(type_pad)
        re = self.res_emb(res_pad)
        x = torch.cat([cont_pad, te, re], dim=-1)
        x = self.input_proj(x)
        x = self.input_ln(x)
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            T_new = T + 1
            lengths_with_cls = lengths + 1
        else:
            T_new = T
            lengths_with_cls = lengths
        
        x = self.pos_encoder(x)
        idx = torch.arange(T_new, device=device).unsqueeze(0)
        padding_mask = idx >= lengths_with_cls.unsqueeze(1)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        if self.use_cls_token:
            pooled = x[:, 0, :]
        elif self.use_last_token:
            last_indices = (lengths - 1).long()
            batch_idx = torch.arange(B, device=device)
            pooled = x[batch_idx, last_indices, :]
        else:
            attn_scores = self.attn_pool(x).squeeze(-1)
            attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        
        pooled = self.layer_norm(pooled)
        out = self.head(pooled)
        return out


class GaussianNLLLoss2D(nn.Module):
    def __init__(self, min_var=1e-4):
        super().__init__()
        self.min_var = min_var
    
    def forward(self, pred, target):
        mu_x, mu_y = pred[:, 0], pred[:, 1]
        log_var_x = torch.clamp(pred[:, 2], -10, 10)
        log_var_y = torch.clamp(pred[:, 3], -10, 10)
        var_x = torch.exp(log_var_x) + self.min_var
        var_y = torch.exp(log_var_y) + self.min_var
        nll_x = 0.5 * (log_var_x + (target[:, 0] - mu_x) ** 2 / var_x)
        nll_y = 0.5 * (log_var_y + (target[:, 1] - mu_y) ** 2 / var_y)
        return (nll_x + nll_y).mean()


class EpisodeDataset(Dataset):
    def __init__(self, episodes, targets, keys):
        self.episodes = episodes
        self.targets = targets
        self.keys = keys
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        cont = torch.from_numpy(ep["cont"])
        type_id = torch.from_numpy(ep["type_id"])
        res_id = torch.from_numpy(ep["res_id"])
        y = torch.from_numpy(self.targets[idx])
        last_start = torch.tensor([ep["last_start_x"], ep["last_start_y"]], dtype=torch.float32)
        return cont, type_id, res_id, y, self.keys[idx], last_start


def collate_fn(batch):
    conts, type_ids, res_ids, ys, keys, last_starts = zip(*batch)
    lengths = torch.tensor([c.shape[0] for c in conts], dtype=torch.long)
    cont_pad = pad_sequence(conts, batch_first=True, padding_value=0.0)
    type_pad = pad_sequence(type_ids, batch_first=True, padding_value=0)
    res_pad = pad_sequence(res_ids, batch_first=True, padding_value=0)
    y = torch.stack(ys, dim=0).float()
    last_start = torch.stack(last_starts, dim=0).float()
    return cont_pad.float(), type_pad.long(), res_pad.long(), lengths, y, keys, last_start


# ============================================================
# Data Processing
# ============================================================

def load_data():
    df = pd.read_csv(TRAIN_PATH)
    df = df.sort_values(["game_episode", "time_seconds", "action_id"]).reset_index(drop=True)
    df["type_name"] = df["type_name"].fillna("__NA_TYPE__")
    df["result_name"] = df["result_name"].fillna("__NA_RES__")
    
    le_type = LabelEncoder()
    le_res = LabelEncoder()
    df["type_id"] = le_type.fit_transform(df["type_name"]) + 1
    df["res_id"] = le_res.fit_transform(df["result_name"]) + 1
    
    return df, le_type, le_res


def build_episodes_with_k(df, k_truncate):
    """Build episodes with specific K truncation"""
    STADIUM_X, STADIUM_Y = 105.0, 68.0
    CENTER_Y = STADIUM_Y / 2.0
    HALF_X = STADIUM_X / 2.0
    GOAL_X = STADIUM_X
    GOAL_POST_HALF = 3.66
    GOAL_Y_L = CENTER_Y - GOAL_POST_HALF
    GOAL_Y_R = CENTER_Y + GOAL_POST_HALF
    P_BOX_X_MIN = STADIUM_X - 16.5
    P_BOX_Y_MIN = CENTER_Y - 20.16
    P_BOX_Y_MAX = CENTER_Y + 20.16
    
    episodes, targets, episode_keys, episode_game_ids = [], [], [], []
    
    for key, g in df.groupby("game_episode"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        
        if g.iloc[-1]["type_name"] != "Pass":
            pass_idxs = g.index[g["type_name"] == "Pass"]
            if len(pass_idxs) == 0:
                continue
            g = g.loc[:pass_idxs[-1]].reset_index(drop=True)
            if len(g) < 2:
                continue
        
        tx, ty = float(g.loc[len(g)-1, "end_x"]), float(g.loc[len(g)-1, "end_y"])
        if np.isnan(tx) or np.isnan(ty):
            continue
        
        t = g["time_seconds"].astype("float32").values
        dt = np.zeros_like(t, dtype="float32")
        dt[1:] = t[1:] - t[:-1]
        dt[dt < 0] = 0.0
        
        sx = g["start_x"].astype("float32").values
        sy = g["start_y"].astype("float32").values
        ex = g["end_x"].astype("float32").values
        ey = g["end_y"].astype("float32").values
        
        ex_mask, ey_mask = ex.copy(), ey.copy()
        ex_mask[-1], ey_mask[-1] = 0.0, 0.0
        
        dxg = GOAL_X - sx
        dy_goal = np.maximum(0.0, np.maximum(GOAL_Y_L - sy, sy - GOAL_Y_R))
        dist_to_goal = np.sqrt(dxg**2 + dy_goal**2).astype("float32")
        
        alpha_L = np.arctan2(GOAL_Y_L - sy, dxg).astype("float32")
        alpha_R = np.arctan2(GOAL_Y_R - sy, dxg).astype("float32")
        theta_view = np.abs(alpha_R - alpha_L).astype("float32")
        
        in_own_half = (sx < HALF_X).astype("float32")
        
        dx_box = np.maximum(0.0, P_BOX_X_MIN - sx)
        dy_box = np.maximum(0.0, np.maximum(P_BOX_Y_MIN - sy, sy - P_BOX_Y_MAX))
        dist_p_box = np.sqrt(dx_box**2 + dy_box**2).astype("float32")
        
        T = len(g)
        prev_dx, prev_dy, prev_valid = np.zeros(T, "float32"), np.zeros(T, "float32"), np.zeros(T, "float32")
        if T > 1:
            prev_dx[1:] = ex[:-1] - sx[:-1]
            prev_dy[1:] = ey[:-1] - sy[:-1]
            prev_valid[1:] = 1.0
        
        type_id = g["type_id"].astype("int64").values
        res_id = g["res_id"].astype("int64").values
        
        cont = np.stack([sx, sy, ex_mask, ey_mask, dt, dist_to_goal, theta_view, 
                         in_own_half, dist_p_box, prev_dx, prev_dy, prev_valid], axis=1).astype("float32")
        
        # K truncation
        if k_truncate is not None and len(cont) > k_truncate:
            cont = cont[-k_truncate:]
            type_id = type_id[-k_truncate:]
            res_id = res_id[-k_truncate:]
        
        episodes.append({
            "cont": cont, "type_id": type_id, "res_id": res_id,
            "last_start_x": sx[-1], "last_start_y": sy[-1]
        })
        
        dx, dy = tx - sx[-1], ty - sy[-1]
        targets.append(np.array([dx, dy], dtype="float32"))
        episode_keys.append(key)
        episode_game_ids.append(str(g.iloc[0]["game_id"]))
    
    return episodes, targets, episode_keys, episode_game_ids


# ============================================================
# Training & Validation
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, pbar=None):
    model.train()
    for cont_pad, type_pad, res_pad, lengths, y, keys, last_start in loader:
        cont_pad = cont_pad.to(device)
        type_pad = type_pad.to(device)
        res_pad = res_pad.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        pred = model(cont_pad, type_pad, res_pad, lengths)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def validate(model, loader, device):
    model.eval()
    total_dist, count = 0.0, 0
    
    with torch.no_grad():
        for cont_pad, type_pad, res_pad, lengths, y, keys, last_start in loader:
            cont_pad = cont_pad.to(device)
            type_pad = type_pad.to(device)
            res_pad = res_pad.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            last_start = last_start.to(device)
            
            pred = model(cont_pad, type_pad, res_pad, lengths)
            pred_xy = pred[:, :2]
            
            # Delta to absolute
            pred_abs = pred_xy + last_start
            true_abs = y + last_start
            
            # Clip
            pred_abs[:, 0] = torch.clamp(pred_abs[:, 0], 0.0, 105.0)
            pred_abs[:, 1] = torch.clamp(pred_abs[:, 1], 0.0, 68.0)
            
            d = torch.sqrt(((pred_abs - true_abs) ** 2).sum(dim=1))
            total_dist += d.sum().item()
            count += d.numel()
    
    return total_dist / max(count, 1)


def train_single_fold(episodes, targets, keys, game_ids, fold, le_type, le_res):
    """Train on single fold and return best validation distance"""
    seed_everything(SEED)
    
    game_ids_arr = np.array(game_ids, dtype=str)
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    tr_idx, va_idx = None, None
    for fold_i, (tr, va) in enumerate(gkf.split(np.zeros(len(episodes)), np.zeros(len(episodes)), groups=game_ids_arr)):
        if fold_i == fold:
            tr_idx, va_idx = tr, va
            break
    
    train_eps = [episodes[i] for i in tr_idx]
    train_tg = [targets[i] for i in tr_idx]
    train_keys = [keys[i] for i in tr_idx]
    
    valid_eps = [episodes[i] for i in va_idx]
    valid_tg = [targets[i] for i in va_idx]
    valid_keys = [keys[i] for i in va_idx]
    
    train_ds = EpisodeDataset(train_eps, train_tg, train_keys)
    valid_ds = EpisodeDataset(valid_eps, valid_tg, valid_keys)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    n_type = len(le_type.classes_) + 1
    n_res = len(le_res.classes_) + 1
    cont_dim = episodes[0]["cont"].shape[1]
    
    model = PassTransformer(
        cont_dim=cont_dim, n_type=n_type, n_res=n_res, emb_dim=EMB_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
        use_cls_token=USE_CLS_TOKEN, use_gaussian_nll=USE_GAUSSIAN_NLL,
        use_last_token=USE_LAST_TOKEN
    ).to(DEVICE)
    
    criterion = GaussianNLLLoss2D()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    best_val = float('inf')
    
    pbar = tqdm(range(1, EPOCHS + 1), desc=f"K={episodes[0]['cont'].shape[0] if len(episodes[0]['cont']) < 100 else '?'} Fold {fold}", leave=False)
    for epoch in pbar:
        train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, pbar)
        val_dist = validate(model, valid_loader, DEVICE)
        scheduler.step(val_dist)
        
        if val_dist < best_val:
            best_val = val_dist
        pbar.set_postfix({"val_dist": f"{val_dist:.4f}", "best": f"{best_val:.4f}"})
    
    return best_val


# ============================================================
# Main OOF Analysis
# ============================================================

def main():
    print("=" * 70)
    print("OOF (Out-of-Fold) Analysis for K Selection")
    print("=" * 70)
    print(f"K values to compare: {K_VALUES}")
    print(f"Folds: {N_SPLITS}")
    print(f"Epochs per fold: {EPOCHS}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # Load data
    print("\n[1/2] Loading data...")
    df, le_type, le_res = load_data()
    print(f"Loaded {len(df)} events")
    
    # Results storage
    results = {k: [] for k in K_VALUES}
    
    # 특정 K만 실행하는 경우
    k_values_to_run = [ARGS.k] if ARGS.k is not None else K_VALUES
    
    # For each K value
    for k in tqdm(k_values_to_run, desc="K values"):
        print(f"\n{'='*70}")
        print(f"Testing K = {k}")
        print(f"{'='*70}")
        
        # Build episodes with this K
        print(f"Building episodes with K={k}...")
        episodes, targets, keys, game_ids = build_episodes_with_k(df, k)
        print(f"Total episodes: {len(episodes)}")
        
        # Train all 5 folds
        for fold in range(N_SPLITS):
            print(f"\n  [Fold {fold}] Training...")
            best_val = train_single_fold(episodes, targets, keys, game_ids, fold, le_type, le_res)
            results[k].append(best_val)
            print(f"  [Fold {fold}] Best valid dist: {best_val:.4f}")
    
    # Summary (단일 K 모드일 때는 간략히)
    if ARGS.k is not None:
        k = ARGS.k
        vals = results[k]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"\n[K={k}] Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"Folds: {vals}")
        
        # 개별 결과 저장
        pd.DataFrame({k: vals}).to_csv(f"oof_results_k{k}.csv")
        print(f"Saved to: oof_results_k{k}.csv")
        return
    
    # 전체 K 모드일 때 Summary
    print("\n" + "=" * 70)
    print("OOF ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'K':<6} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} | Fold Results")
    print("-" * 70)
    
    for k in K_VALUES:
        vals = results[k]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        fold_str = ", ".join([f"{v:.2f}" for v in vals])
        print(f"{k:<6} | {mean_val:>10.4f} | {std_val:>10.4f} | {min_val:>10.4f} | {max_val:>10.4f} | [{fold_str}]")
    
    print("-" * 70)
    
    # Recommendation
    best_k = min(K_VALUES, key=lambda k: np.mean(results[k]) + 0.5 * np.std(results[k]))  # Mean + 0.5*Std 최소화
    print(f"\n[RECOMMENDATION] K = {best_k}")
    print(f"  (Based on Mean + 0.5*Std minimization for LB stability)")
    print("=" * 70)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.index = [f"fold_{i}" for i in range(N_SPLITS)]
    results_df.to_csv("oof_analysis_results.csv")
    print(f"\nResults saved to: oof_analysis_results.csv")


if __name__ == "__main__":
    main()
