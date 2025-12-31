# file/path utilities
import os
import glob
from pathlib import Path
import pickle
# for data manipulation/math
import pandas as pd
import numpy as np
import random
import math
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
# arg setting
import argparse



# -----------------------------------
# arg setting for 5 GPU
# -----------------------------------
parser = argparse.ArgumentParser()
# parallel training args
# ex) python3 train.py --fold 0
parser.add_argument('--fold', type=int, default=None, help='Train specific fold only (0-4), None for all folds')

# modes for single batch test
# ex) python3 train.py --mode overfit --overfit_epochs 1000
parser.add_argument('--mode', type=str, default='train', choices=['train', 'overfit'], 
                    help='train: full training, overfit: single batch overfitting test')

# overfit: single batch test -> to check training converges
parser.add_argument('--overfit_epochs', type=int, default=500, help='Epochs for overfit mode')
args, _ = parser.parse_known_args()

# -----------------------------------
# Hyperparameters
# -----------------------------------
SEED = 42
SEEDS = [42, 123, 456] # for ensemble

# cross-validation
N_SPLITS = 5
FOLD = args.fold if args.fold is not None else None # no args -> None (all folds), yes args -> specific fold

# sequence length
MIN_EVENTS = 2 # no need less than 2
K_TRUNCATE = 50 # sequence truncation

# training parameters
EPOCHS = 150
BATCH_SIZE = 256
LR = 1e-3

# regularization
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2

# model parameters
D_MODEL = 256          # Transformer hidden dimension: inner vector expression size, large -> more capacity, computation
N_HEADS = 8            # Number of attention heads, D_MODEL must be divisible by N_HEADS
N_LAYERS = 4           # Number of transformer encoder layers (self attention + MLP)
DIM_FEEDFORWARD = 512  # Feed-forward network dimension: D_MODEL -> DIM_FEEDFORWARD -> D_MODEL add unlinearity
EMB_DIM = 16           # Embedding dimension for categorical features(type_id, res_id)

# model options
USE_GAUSSIAN_NLL = True  # Gaussian NLL loss
PREDICT_DELTA = True     # predict end_x, end_y as delta from start_x, start_y

# augmentation parameters
USE_AUGMENT = False
NOISE_STD = 0.5

# paths
TRAIN_PATH = "../data/train.csv"
MODEL_SAVE_DIR = "./models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# for reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)
print("Using device:", DEVICE)


# -----------------------------------
# Data Loading and Preprocessing
# -----------------------------------

def load_and_preprocess_data(train_path):
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
    
    # shift by +1 to reserve 0 for padding
    df["type_id"] = le_type.fit_transform(df["type_name"]) + 1
    df["res_id"] = le_res.fit_transform(df["result_name"]) + 1
    
    return df, le_type, le_res


def build_episodes(df, le_type, le_res):    

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
    
    # model inputs/targets
    episodes = []
    targets = []
    episode_keys = []
    episode_game_ids = []

    # key = game_episode: {game_id}_{episode_id}
    for key, g in tqdm(df.groupby("game_episode"), desc="Loading episodes"):
        g = g.reset_index(drop=True) # index reset 0,1,2...
        if len(g) < 2:
            continue
        
        # fix last event as Pass
        if g.iloc[-1]["type_name"] != "Pass":
            pass_idxs = g.index[g["type_name"] == "Pass"]
            if len(pass_idxs) == 0:
                continue
            g = g.loc[:pass_idxs[-1]].reset_index(drop=True)
            if len(g) < 2:
                continue
        
        # target: last event's end point
        tx, ty = float(g.loc[len(g)-1, "end_x"]), float(g.loc[len(g)-1, "end_y"])
        if np.isnan(tx) or np.isnan(ty):
            continue
        
        # dt
        t = g["time_seconds"].astype("float32").values
        dt = np.zeros_like(t, dtype="float32")
        dt[1:] = t[1:] - t[:-1]
        dt[dt < 0] = 0.0
        
        # extract start/end positions
        sx = g["start_x"].astype("float32").values
        sy = g["start_y"].astype("float32").values
        ex = g["end_x"].astype("float32").values
        ey = g["end_y"].astype("float32").values
        
        # leak-safe masking
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
        
        # continuous features
        cont = np.stack([
            sx,            # 1
            sy,            # 2
            ex_mask,       # 3
            ey_mask,       # 4
            dt,            # 5
            dist_to_goal,  # 6
            theta_view,    # 7
            in_own_half,   # 8
            dist_p_box,    # 9
            prev_dx,       # 10
            prev_dy,       # 11
            prev_valid     # 12
        ], axis=1).astype("float32")
        
        # use last K events only (focus on recent context)
        if K_TRUNCATE is not None and len(cont) > K_TRUNCATE:
            cont = cont[-K_TRUNCATE:]
            type_id = type_id[-K_TRUNCATE:]
            res_id = res_id[-K_TRUNCATE:]
        
        episodes.append({
            "cont": cont,
            "type_id": type_id,
            "res_id": res_id,
            "last_start_x": sx[-1],
            "last_start_y": sy[-1],
            "last_result_id": int(res_id[-1])
        })
        
        # target
        if PREDICT_DELTA:
            # (dx, dy) = (end - start) of last event
            dx = tx - sx[-1]
            dy = ty - sy[-1]
            targets.append(np.array([dx, dy], dtype="float32"))
        else:
            targets.append(np.array([tx, ty], dtype="float32"))
        episode_keys.append(key)

        episode_game_ids.append(str(g.iloc[0]["game_id"]))
    
    return episodes, targets, episode_keys, episode_game_ids


# ------------------------------------
# Dataset and DataLoader
# ------------------------------------

class EpisodeDataset(Dataset):
    def __init__(self, episodes, targets, keys, augment=False, noise_std=0.0): # default
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
        
        # data augmentation: Gaussian noise on positions
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
        
        # last event info: start_x, start_y, result_id
        last_start = torch.tensor([ep["last_start_x"], ep["last_start_y"]], dtype=torch.float32)
        last_result_id = torch.tensor(ep["last_result_id"], dtype=torch.long)
        
        return cont, type_id, res_id, y, key, last_start, last_result_id


def collate_fn(batch):
    # batch: list of samples
    conts, type_ids, res_ids, ys, keys, last_starts, last_result_ids = zip(*batch)
    
    # original length before padding
    lengths = torch.tensor([c.shape[0] for c in conts], dtype=torch.long)
    
    # pad length to max in batch (pad back with 0, (ex) 1,2,...,Tmax,0,0,...)
    cont_pad = pad_sequence(conts, batch_first=True, padding_value=0.0)  # [B, Tmax, cont_dim]
    type_pad = pad_sequence(type_ids, batch_first=True, padding_value=0) # [B, Tmax]
    res_pad = pad_sequence(res_ids, batch_first=True, padding_value=0)   # [B, Tmax]

    # stack targets and last event info
    y = torch.stack(ys, dim=0).float()                            # [B, 2]
    last_start = torch.stack(last_starts, dim=0).float()          # [B, 2]
    last_result_id = torch.stack(last_result_ids, dim=0).long()   # [B]
    
    return cont_pad.float(), type_pad.long(), res_pad.long(), lengths, y, keys, last_start, last_result_id


# ------------------------------------
# Transformer Model
# ------------------------------------
class PositionalEncoding(nn.Module):
    # let transformer know about positions by adding sin/cos functions -> model can distinguish positions
    def __init__(self, d_model, dropout, max_len=500):
        super().__init__()

        # dropout declare (used in forward)
        self.dropout = nn.Dropout(p=dropout)
        
        # pe: (0,1,2,...,max_len-1) stores position vectors
        # pe[t,:] = position vector at position t
        pe = torch.zeros(max_len, d_model)

        # position: ([0],[1],[2],...,[max_len-1]) (column vector), shape (max_len, 1) by unsqueeze
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # controls frequencies: diff dim -> diff wavelength, shape (d_model/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # even idx: sin, odd idx: cos
        # why sin/cos? -> unique pattern for each position
        # position * div_term = position(row) x freq(col)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension to match x
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # store as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, T, d_model) = (# of episodes, length after padding, token dimension)
        x = x + self.pe[:, :x.size(1), :] # add positional encoding (adding unique pattern to let model know about positions)
        return self.dropout(x)

class PassTransformer(nn.Module):    
    # input: cont_dim + type_id embedding + result_id embedding
    # output: (dx, dy) or (mu_x, mu_y, log_var_x, log_var_y)
    def __init__(self, cont_dim, n_type, n_res, emb_dim, d_model, 
                 n_heads, n_layers, dim_feedforward, dropout,
                 use_gaussian_nll):
        super().__init__()
        
        # for easy change in hyperparam
        self.use_gaussian_nll = use_gaussian_nll

        # embedding: turn number idx to vector
        # embeddings for categorical features (type_id, result_id only), consider 0 as padding
        self.type_emb = nn.Embedding(n_type, emb_dim, padding_idx=0)
        self.res_emb = nn.Embedding(n_res, emb_dim, padding_idx=0)

        # input projection token: (cont + type embedding + res embedding) -> d_model
        in_dim = cont_dim + emb_dim + emb_dim
        self.input_proj = nn.Linear(in_dim, d_model) # project to d_model dimension
        self.input_ln = nn.LayerNorm(d_model) # layer norm
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=500)
        
        # one encoder layer = (self-attention + feed-forward) with dropout, layer norms
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )

        # Transformer encoder: stack of encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # normalize pooled output
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output dimension: 2 for (dx, dy), or 4 for Gaussian Loss (mu_x, mu_y, log_var_x, log_var_y)
        out_dim = 4 if use_gaussian_nll else 2

        # head(MLP(FC)), applied after pooling
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),         # linear projection (reducing dim slowly)
            nn.LayerNorm(d_model),                     
            nn.GELU(),                    
            nn.Dropout(dropout),                    
            nn.Linear(d_model, d_model // 2),    # reduce dimension
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim)
        )
        
        # for debug, print(d_model)
        self.d_model = d_model
    
    def forward(self, cont_pad, type_pad, res_pad, lengths):
        B, T, _ = cont_pad.shape
        device = cont_pad.device
        
        # embed(num idx to vector) categorical features
        te = self.type_emb(type_pad)  # (B, T, emb_dim)
        re = self.res_emb(res_pad)    # (B, T, emb_dim)
        
        # concatenate all features
        x = torch.cat([cont_pad, te, re], dim=-1)  # (B, T, cont_dim(12) + emb_dim(16) + emb_dim(16)) = (B, T, in_dim(44))

        # project to in_dim to d_model dimension
        x = self.input_proj(x)  # (B, T, d_model)
        
        # apply LayerNorm
        x = self.input_ln(x)  # (B, T, d_model)
        
        # add positional encoding
        x = self.pos_encoder(x)
        
        # create padding mask (True for padding positions)
        # ex) B=2, T=5, lengths=[3,5], idx=[0,1,2,3,4] -> padding_mask = [False,False,False,True,True],
        idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        padding_mask = idx >= lengths.unsqueeze(1)  # (B, T)
        
        # Transformer encoder ignoring padding positions
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)
        
        # last token pooling, last = length - 1
        last_indices = (lengths - 1).long()  # (B,)
        batch_idx = torch.arange(B, device=device)  # (B,), batch index to pick one token per sample
        pooled = x[batch_idx, last_indices, :]  # (B, T, d_model) -> (B, d_model)
        
        # Layer norm
        pooled = self.layer_norm(pooled)

        # Output head(MLP)
        out = self.head(pooled)  # (B, 2 or 4)
        
        return out


# ------------------------------------
# Loss Functions
# ------------------------------------

class GaussianNLLLoss2D(nn.Module):
    # not predicting (x, y) directly, but predicting mean and variance of Gaussian for (x, y)
    # input: from model -> (mu_x, mu_y, log_var_x, log_var_y) -> (target): (x, y)
    # mu = mean
    # log_var: for positive value
    def __init__(self, min_var=1e-4):
        super().__init__()

        # ============ setup for loss calculation ============
        # to avoid zero variance (denominator issue)
        self.min_var = min_var
    
    def forward(self, pred, target):
        # pred: (B, 4) -> (mu_x, mu_y, log_var_x, log_var_y)
        # target: (B, 2) -> (x, y)
        
        # as model trains to minimize NLL, it will learn to predict mean close to target and small variance
        mu_x = pred[:, 0]         # predicted mean dx
        mu_y = pred[:, 1]         # predicted mean dy
        log_var_x = pred[:, 2]    # predicted log variance dx
        log_var_y = pred[:, 3]    # predicted log variance dy
        
        # Clamp log_var for numerical stability
        log_var_x = torch.clamp(log_var_x, min=-10, max=10)
        log_var_y = torch.clamp(log_var_y, min=-10, max=10)
        
        # log_var -> var (ensure positive variance)
        var_x = torch.exp(log_var_x) + self.min_var
        var_y = torch.exp(log_var_y) + self.min_var
        
        target_x = target[:, 0]
        target_y = target[:, 1]
        # ======================================================

        # loss calculation
        # NLL = 0.5 * (log(var) + (x - mu)^2 / var)
        nll_x = 0.5 * (log_var_x + (target_x - mu_x) ** 2 / var_x)
        nll_y = 0.5 * (log_var_y + (target_y - mu_y) ** 2 / var_y)
        
        return (nll_x + nll_y).mean()

# ------------------------------------
# Training Functions
# ------------------------------------

def euclidean_sum_and_count(pred, true, last_start=None, predict_delta=False, use_gaussian=False):
    if use_gaussian:
        pred_xy = pred[:, :2]  # mu_x, mu_y only to cal euclidean distance, var not used
    else:
        pred_xy = pred
    
    # end = start + delta
    # pred_abs 절댓값이 아니라 절대적인 위치라는 뜻
    if predict_delta and last_start is not None:
        # Convert delta to absolute: end = start + delta
        pred_abs = pred_xy + last_start
        true_abs = true + last_start
    else:
        pred_abs = pred_xy
        true_abs = true

    # clipping, stadium size (0~105, 0~68)
    pred_clipped = pred_abs.clone()
    pred_clipped[:, 0] = torch.clamp(pred_clipped[:, 0], 0.0, 105.0)
    pred_clipped[:, 1] = torch.clamp(pred_clipped[:, 1], 0.0, 68.0)
    
    # clipping difference for debug
    clip_diff = (pred_abs - pred_clipped).abs().sum().item()
    
    # cal euclidean distance per sample
    d = torch.sqrt(((pred_clipped - true_abs) ** 2).sum(dim=1))

    # sum and count used for average cal
    # (total distance, number of samples, total clipping difference)
    return d.sum().item(), d.numel(), clip_diff

# declare one epoch
def train_one_epoch(model, train_loader, optimizer, criterion, device, 
                    predict_delta=False, use_gaussian=False):
    model.train()

    # used for epoch level averages
    tr_loss_sum = 0.0
    tr_loss_cnt = 0
    tr_euc_sum = 0.0
    tr_euc_cnt = 0
    tr_clip_diff = 0.0
    
    # progress bar wrappin train loader
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    # in pbar = in train loader
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
        optimizer.step()
        
        # for epoch level averages
        bsz = y.size(0)
        tr_loss_sum += loss.item() * bsz
        tr_loss_cnt += bsz
        
        # no backprop
        # for train evaluation
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
    
    # if avg_clip > 0.1:
    #     print(f"[WARNING] 튀는 예측 발견! 평균 클리핑 차이: {avg_clip:.2f}")
    
    return avg_dist

# debug: overfit single batch
# to check if model can learn by checking training loss goes down
def overfit_single_batch(model, sample_batch, optimizer, criterion, device, 
                        epochs=500, predict_delta=False, use_gaussian=False):
    print("\n" + "=" * 60)
    print("SINGLE BATCH TEST")
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
    elif last_loss < 0.5:
        print("\n  Successfully overfitting! Model can learn.")
    else:
        print("\n  Partial overfit. May need more epochs.")
    
    return losses, dists


# ------------------------------------
# Main Training Loop
# ------------------------------------
def train_single_seed(seed, episodes, targets, episode_keys, episode_game_ids, 
                      le_type, le_res, seed_idx=0, total_seeds=1, fold=None, mode='train'):
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
    episode_game_ids_arr = np.array(episode_game_ids, dtype=str)
    unique_games = np.unique(episode_game_ids_arr)
    print(f"[GroupKFold] 총 {len(unique_games)}개 경기(game_id), {len(episodes)}개 에피소드")
    
    gkf = GroupKFold(n_splits=N_SPLITS)

    # fold selection
    target_fold = fold if fold is not None else 0
    
    tr_idx, va_idx = None, None
    for fold_i, (tr, va) in enumerate(
        gkf.split(np.zeros(len(episodes)), np.zeros(len(episodes)), groups=episode_game_ids_arr)
    ):
        if fold_i == target_fold:
            tr_idx, va_idx = tr, va
            break
    
    assert tr_idx is not None and va_idx is not None
    
    # overlap check
    train_games = set(episode_game_ids_arr[tr_idx])
    valid_games = set(episode_game_ids_arr[va_idx])
    overlap = train_games & valid_games
    if len(overlap) > 0:
        print(f"[WARNING] Train/Valid game_id overlap! {len(overlap)}개")
    else:
        print(f"[OK] Train {len(train_games)}개 경기, Valid {len(valid_games)}개 경기 (no overlap)")
    
    train_eps = [episodes[i] for i in tr_idx]
    train_tg = [targets[i] for i in tr_idx]
    train_keys = [episode_keys[i] for i in tr_idx]
    
    valid_eps = [episodes[i] for i in va_idx]
    valid_tg = [targets[i] for i in va_idx]
    valid_keys = [episode_keys[i] for i in va_idx]
    
    print(f"Train episodes: {len(train_eps)} | Valid episodes: {len(valid_eps)}")
    
    # Create datasets and dataloaders
    # data augmentation if USE_AUGMENT is True
    train_ds = EpisodeDataset(train_eps, train_tg, train_keys, augment=USE_AUGMENT, noise_std=NOISE_STD)
    valid_ds = EpisodeDataset(valid_eps, valid_tg, valid_keys, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    # shift to avoid padding idx 0
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
        use_gaussian_nll=USE_GAUSSIAN_NLL
    ).to(DEVICE)
    
    # Print model info (only first seed)
    if seed_idx == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[Model Architecture]")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  n_type: {n_type}, n_res: {n_res}, cont_dim: {cont_dim}")
        print(f"\n[Hyperparameters]")
        print(f"  Training: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")
        print(f"  Regularization: WEIGHT_DECAY={WEIGHT_DECAY}, DROPOUT={DROPOUT}")
        print(f"  Model: D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_LAYERS={N_LAYERS}")
        print(f"  Model: DIM_FEEDFORWARD={DIM_FEEDFORWARD}, EMB_DIM={EMB_DIM}")
        print(f"  Options: USE_GAUSSIAN_NLL={USE_GAUSSIAN_NLL}, PREDICT_DELTA={PREDICT_DELTA}")
        print(f"  Sequence: K_TRUNCATE={K_TRUNCATE}, MIN_EVENTS={MIN_EVENTS}")
        print(f"  Augmentation: USE_AUGMENT={USE_AUGMENT}, NOISE_STD={NOISE_STD}")
    
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
        "use_gaussian_nll": USE_GAUSSIAN_NLL,
        "predict_delta": PREDICT_DELTA
    }
    with open(os.path.join(seed_save_dir, "model_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    # Criterion
    if USE_GAUSSIAN_NLL:
        criterion = GaussianNLLLoss2D()
    else:
        criterion = nn.SmoothL1Loss()
    
    # TODO optim 적합한지 공부
    # Optimizer, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # ------------------------------------
    # MODE: overfit - Single batch overfitting test
    # ------------------------------------
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
    
    # ------------------------------------
    # MODE: train - Full training
    # ------------------------------------
    # Training loop
    best_val = float('inf')
    best_state = None
    
    # epoch loop
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
        # Overfit mode
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
        # 5-fold
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
        # particular fold
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
        
        # overall average
        overall_avg = sum(r['best_val'] for r in results) / len(results)
        print(f"\n  Overall average: {overall_avg:.4f} ({len(results)} models)")
    else:
        # particular fold
        for r in results:
            print(f"  Seed {r['seed']}: valid_dist = {r['best_val']:.4f}")
        avg_val = sum(r['best_val'] for r in results) / len(results)
        print(f"\n  Average validation distance: {avg_val:.4f}")
    
    print(f"\n  Models saved to: {MODEL_SAVE_DIR}/seed_*_fold_*/best_model.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
