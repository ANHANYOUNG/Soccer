"""
Transformer Baseline - Start Simple (Debug Friendly)
======================================================
DL_DEBUG.md 가이드를 따르는 디버깅 친화적 구조

단계:
1. Get Your Model to Run - Shape/Casting/OOM 확인
2. Overfit A Single Batch - 학습 가능 여부 확인
3. Evaluate - Bias/Variance 분석
4. Improve - 점진적 개선

규칙:
- NO dropout/weight_decay
- Model normalization: LayerNorm 사용
- NO data normalization (raw 값 사용, 출력만 범위 제한)
- NO augmentation
- 범주형 ID는 1부터 시작 (0=PAD와 충돌 방지)
"""

import os
import random
import pickle
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math

# ============================================================
# 0. Arguments & Mode Selection
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='overfit', 
                    choices=['check', 'overfit', 'train'],
                    help='check: shape/forward 확인, overfit: single batch, train: full training')
parser.add_argument('--fold', type=int, default=0, help='Fold to train (0-4)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
args, _ = parser.parse_known_args()

# ============================================================
# 1. Hyperparameters (NO REGULARIZATION for debugging)
# ============================================================
SEED = args.seed
FOLD = args.fold

# Training
EPOCHS = args.epochs if args.epochs else {
    'check': 1,
    'overfit': 200,  # overfit은 오래 돌려서 확인
    'train': 100
}[args.mode]

BATCH_SIZE = 32 if args.mode in ['check', 'overfit'] else 256
LR = 5e-3  # Adam Otim
WEIGHT_DECAY = 0.0  # NO regularization

# Model (작게 시작)
D_MODEL = 128
N_HEADS = 4 # D_MODEL / N_HEADS = 정수
N_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.0  # NO dropout 
EMB_DIM = 16

# Data
TRAIN_PATH = "../data/train.csv"
MODEL_DIR = "./models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CV
N_SPLITS = 5

# Field bounds (축구장 크기)
FIELD_X = 105.0
FIELD_Y = 68.0


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Data Loading (NO data normalization - raw values)
# ============================================================
def load_data(path):
    """데이터 로드"""
    df = pd.read_csv(path)
    df = df.sort_values(["game_episode", "time_seconds", "action_id"]).reset_index(drop=True)
    
    # 결측 처리 (특수 문자열로)
    df["type_name"] = df["type_name"].fillna("__NA__")
    df["result_name"] = df["result_name"].fillna("__NA__")
    
    return df


def build_vocab(df, col):
    """
    vocab 생성 (1-indexed, 0=PAD)
    
    중요: 실제 값은 1부터 시작
    - 0: <PAD> (패딩용)
    - 1~: 실제 값들
    
    <UNK>는 사용하지 않음 (train에 없는 값이 test에 나오면 그냥 0으로)
    """
    unique_vals = df[col].unique().tolist()
    vocab = {"<PAD>": 0}
    for val in sorted(unique_vals):  # 정렬해서 재현성 확보
        vocab[val] = len(vocab)
    return vocab


def build_episodes(df, type_vocab, res_vocab, debug_print=False):
    """
    에피소드 단위로 데이터 구성
    - 마지막 이벤트가 Pass가 아니면 제외
    - Raw 값 사용 (정규화 없음)
    """
    episodes = []
    targets = []
    keys = []
    game_ids = []
    
    for ep_key, g in tqdm(df.groupby("game_episode"), desc="Building episodes", disable=not debug_print):
        g = g.reset_index(drop=True)
        
        if len(g) < 2:
            continue
        
        if g.iloc[-1]["type_name"] != "Pass":
            continue
        
        tx = float(g.iloc[-1]["end_x"])
        ty = float(g.iloc[-1]["end_y"])
        if np.isnan(tx) or np.isnan(ty):
            continue
        
        # dt 계산
        t = g["time_seconds"].astype("float32").values
        dt = np.zeros_like(t, dtype="float32")
        dt[1:] = t[1:] - t[:-1]
        dt = np.clip(dt, 0, None)
        
        # 좌표 (raw)
        sx = g["start_x"].astype("float32").values
        sy = g["start_y"].astype("float32").values
        ex = g["end_x"].astype("float32").values.copy()
        ey = g["end_y"].astype("float32").values.copy()
        
        # NaN → 0
        sx = np.nan_to_num(sx, nan=0.0)
        sy = np.nan_to_num(sy, nan=0.0)
        ex = np.nan_to_num(ex, nan=0.0)
        ey = np.nan_to_num(ey, nan=0.0)
        
        # 마지막 이벤트 end 마스킹 (leak 방지)
        ex[-1] = 0.0
        ey[-1] = 0.0
        
        # 연속형 피처: (T, 5) - Raw 값 그대로
        cont = np.stack([sx, sy, ex, ey, dt], axis=1).astype("float32")
        
        # 범주형 인덱스 (vocab lookup, 없으면 0=PAD로)
        type_ids = np.array([type_vocab.get(v, 0) for v in g["type_name"]], dtype="int64")
        res_ids = np.array([res_vocab.get(v, 0) for v in g["result_name"]], dtype="int64")
        
        episodes.append({
            "cont": cont,
            "type_id": type_ids,
            "res_id": res_ids,
        })
        # 타깃도 raw 값
        targets.append(np.array([tx, ty], dtype="float32"))
        keys.append(ep_key)
        game_ids.append(str(g.iloc[0]["game_id"]))
    
    return episodes, targets, keys, game_ids


# ============================================================
# 3. Dataset
# ============================================================
class EpisodeDataset(Dataset):
    def __init__(self, episodes, targets, keys):
        self.episodes = episodes
        self.targets = targets
        self.keys = keys
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return (
            torch.from_numpy(ep["cont"]),
            torch.from_numpy(ep["type_id"]),
            torch.from_numpy(ep["res_id"]),
            torch.from_numpy(self.targets[idx]),
            self.keys[idx]
        )


def collate_fn(batch):
    conts, type_ids, res_ids, ys, keys = zip(*batch)
    lengths = torch.tensor([c.shape[0] for c in conts], dtype=torch.long)
    
    # 0으로 패딩 (범주형 0 = <PAD>)
    cont_pad = pad_sequence(conts, batch_first=True, padding_value=0.0)
    type_pad = pad_sequence(type_ids, batch_first=True, padding_value=0)
    res_pad = pad_sequence(res_ids, batch_first=True, padding_value=0)
    y = torch.stack(ys)
    
    return cont_pad.float(), type_pad.long(), res_pad.long(), lengths, y, keys


# ============================================================
# 4. Model (Simple Transformer, NO dropout)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SimpleTransformer(nn.Module):
    """
    최소 구조 Transformer (디버깅용)
    
    Embedding padding_idx=0 사용
    → vocab은 1부터 시작해야 0과 충돌 안 함
    
    LayerNorm 사용 (BatchNorm 제거)
    """
    def __init__(self, cont_dim, n_type, n_res, emb_dim=16, d_model=64,
                 n_heads=4, n_layers=2, dim_ff=256, dropout=0.0):
        super().__init__()
        
        self.cont_dim = cont_dim
        self.emb_dim = emb_dim
        self.d_model = d_model
        
        # Embeddings (padding_idx=0)
        self.type_emb = nn.Embedding(n_type, emb_dim, padding_idx=0)
        self.res_emb = nn.Embedding(n_res, emb_dim, padding_idx=0)
        
        # Input projection with LayerNorm
        in_dim = cont_dim + emb_dim * 2
        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)  # LayerNorm 사용
        
        # Positional encoding (no dropout)
        self.pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoder (minimal, no dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="relu",  # 간단하게 ReLU
            batch_first=True,
            norm_first=False  # Post-LN (기본)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        
        # Output head with LayerNorm
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),  # LayerNorm 사용
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )
    
    def forward(self, cont, type_ids, res_ids, lengths, debug=False):
        B, T, _ = cont.shape
        device = cont.device
        
        # === Debug: Check shapes ===
        if debug:
            print(f"[Forward Debug]")
            print(f"  cont shape: {cont.shape}")  # (B, T, 5)
            print(f"  type_ids shape: {type_ids.shape}")  # (B, T)
            print(f"  res_ids shape: {res_ids.shape}")  # (B, T)
            print(f"  lengths: {lengths}")
            print(f"  type_ids range: [{type_ids.min().item()}, {type_ids.max().item()}]")
            print(f"  res_ids range: [{res_ids.min().item()}, {res_ids.max().item()}]")
        
        # Embeddings
        te = self.type_emb(type_ids)  # (B, T, emb_dim)
        re = self.res_emb(res_ids)    # (B, T, emb_dim)
        
        if debug:
            print(f"  type_emb shape: {te.shape}")
            print(f"  res_emb shape: {re.shape}")
        
        # Concat & project
        x = torch.cat([cont, te, re], dim=-1)  # (B, T, 5 + 2*emb_dim)
        
        if debug:
            print(f"  concat shape: {x.shape}")
        
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.input_ln(x)  # LayerNorm (B, T, d_model)
        
        if debug:
            print(f"  after proj+LN shape: {x.shape}")
        
        # Positional encoding
        x = self.pos_enc(x)
        
        # Padding mask: True = 무시할 위치
        idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        pad_mask = idx >= lengths.unsqueeze(1)  # (B, T)
        
        if debug:
            print(f"  pad_mask shape: {pad_mask.shape}")
            print(f"  pad_mask[0]: {pad_mask[0]}")
        
        # Transformer
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        
        if debug:
            print(f"  after encoder shape: {x.shape}")
        
        # Last-token pooling
        last_idx = (lengths - 1).long()
        batch_idx = torch.arange(B, device=device)
        pooled = x[batch_idx, last_idx]  # (B, d_model)
        
        if debug:
            print(f"  pooled shape: {pooled.shape}")
        
        # Output (BatchNorm은 Sequential에서 자동 처리)
        out = self.head(pooled)  # (B, 2)
        
        if debug:
            print(f"  output shape: {out.shape}")
        
        return out


# ============================================================
# 5. Debug Functions
# ============================================================
def check_model_runs(model, sample_batch, device):
    """
    Step 1: Get Your Model to Run
    - Shape mismatch 확인
    - Casting issue 확인
    - Forward pass 성공 여부
    """
    print("\n" + "=" * 60)
    print("STEP 1: Get Your Model to Run")
    print("=" * 60)
    
    cont, type_ids, res_ids, lengths, y, keys = sample_batch
    
    print("\n[Input Shapes]")
    print(f"  cont: {cont.shape}, dtype={cont.dtype}")
    print(f"  type_ids: {type_ids.shape}, dtype={type_ids.dtype}")
    print(f"  res_ids: {res_ids.shape}, dtype={res_ids.dtype}")
    print(f"  lengths: {lengths.shape}, dtype={lengths.dtype}")
    print(f"  y: {y.shape}, dtype={y.dtype}")
    
    # Move to device
    cont = cont.to(device)
    type_ids = type_ids.to(device)
    res_ids = res_ids.to(device)
    lengths = lengths.to(device)
    y = y.to(device)
    
    print("\n[Forward Pass with Debug]")
    try:
        model.eval()
        with torch.no_grad():
            out = model(cont, type_ids, res_ids, lengths, debug=True)
        print(f"\n✓ Forward pass SUCCESS!")
        print(f"  Output shape: {out.shape}")
        print(f"  Output sample: {out[0].cpu().numpy()}")
        print(f"  Target sample: {y[0].cpu().numpy()}")
    except Exception as e:
        print(f"\n✗ Forward pass FAILED: {e}")
        raise e
    
    print("\n[Backward Pass]")
    try:
        model.train()
        out = model(cont, type_ids, res_ids, lengths)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        print(f"✓ Backward pass SUCCESS!")
        print(f"  Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Backward pass FAILED: {e}")
        raise e
    
    print("\n[Check for NaN/Inf]")
    has_issue = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  ✗ NaN in gradient: {name}")
                has_issue = True
            if torch.isinf(param.grad).any():
                print(f"  ✗ Inf in gradient: {name}")
                has_issue = True
    if not has_issue:
        print("  ✓ No NaN/Inf in gradients")
    
    return True


def overfit_single_batch(model, sample_batch, device, epochs=200, lr=2e-3):
    """
    Step 2: Overfit A Single Batch
    
    정상 동작:
    - Loss가 0에 가깝게 감소해야 함
    - 감소하지 않으면 문제 있음
    
    문제 진단:
    - Error goes UP: 부호 반대?
    - Error EXPLODES: NaN/Inf, LR 너무 높음
    - Error OSCILLATES: LR 낮추기, 데이터 확인
    - Error PLATEAUS: LR 높이기
    """
    print("\n" + "=" * 60)
    print("STEP 2: Overfit A Single Batch")
    print("=" * 60)
    print(f"  Epochs: {epochs}")
    print(f"  LR: {lr}")
    print(f"  Batch size: {sample_batch[0].shape[0]}")
    
    cont, type_ids, res_ids, lengths, y, keys = sample_batch
    cont = cont.to(device)
    type_ids = type_ids.to(device)
    res_ids = res_ids.to(device)
    lengths = lengths.to(device)
    y = y.to(device)
    
    # 새 모델로 시작 (기존 가중치 영향 제거)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    dists = []
    
    print("\n[Training Progress]")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(cont, type_ids, res_ids, lengths)
        
        # MSE Loss
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        # Distance metric
        with torch.no_grad():
            dist = torch.sqrt(((pred - y) ** 2).sum(dim=1)).mean().item()
        
        losses.append(loss.item())
        dists.append(dist)
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{epoch:03d}] loss={loss.item():.6f}, dist={dist:.4f}")
    
    # 진단
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
        print("     - Target labels")
    elif any(np.isnan(losses)) or any(np.isinf(losses)):
        print("\n  ⚠️ ERROR EXPLODES! Check:")
        print("     - Lower learning rate")
        print("     - Gradient clipping")
        print("     - Input data for NaN/Inf")
    elif last_loss > first_loss * 0.5:
        print("\n  ⚠️ ERROR PLATEAUS! Try:")
        print("     - Higher learning rate")
        print("     - Different optimizer")
    elif last_loss < 0.01:
        print("\n  ✓ Successfully overfitting! Model can learn.")
    else:
        print("\n  △ Partial overfit. May need more epochs or higher LR.")
    
    return losses, dists


# ============================================================
# 6. Training Functions
# ============================================================
def train_epoch(model, loader, optimizer, device, epoch_num=0):
    """학습 - raw 좌표 공간에서 직접 계산"""
    model.train()
    total_loss = 0.0
    total_dist = 0.0
    count = 0
    
    pbar = tqdm(loader, desc=f"Train[{epoch_num:03d}]", leave=False)
    for cont, type_ids, res_ids, lengths, y, keys in pbar:
        cont = cont.to(device)
        type_ids = type_ids.to(device)
        res_ids = res_ids.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        pred = model(cont, type_ids, res_ids, lengths)
        
        # MSE Loss
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        
        # NO gradient clipping in debug mode
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            total_loss += loss.item() * y.size(0)
            dist = torch.sqrt(((pred - y) ** 2).sum(dim=1)).sum().item()
            total_dist += dist
            count += y.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / count, total_dist / count


@torch.no_grad()
def validate(model, loader, device):
    """Validation - raw 좌표 공간에서 직접 계산"""
    model.eval()
    total_dist = 0.0
    count = 0
    
    for cont, type_ids, res_ids, lengths, y, keys in loader:
        cont = cont.to(device)
        type_ids = type_ids.to(device)
        res_ids = res_ids.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        
        pred = model(cont, type_ids, res_ids, lengths)
        
        # Clipping to field bounds
        pred_clip = pred.clone()
        pred_clip[:, 0] = torch.clamp(pred_clip[:, 0], 0, 105)
        pred_clip[:, 1] = torch.clamp(pred_clip[:, 1], 0, 68)
        
        dist = torch.sqrt(((pred_clip - y) ** 2).sum(dim=1)).sum().item()
        total_dist += dist
        count += y.size(0)
    
    return total_dist / count


def train_full(model, train_loader, valid_loader, device, epochs=100, lr=LR):
    """
    Step 3: Full Training with Evaluation
    """
    print("\n" + "=" * 60)
    print("STEP 3: Full Training")
    print("=" * 60)
    print(f" LR: {lr}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    best_val = float("inf")
    best_state = None
    
    train_losses = []
    train_dists = []
    valid_dists = []
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_dist = train_epoch(model, train_loader, optimizer, device, epoch)
        val_dist = validate(model, valid_loader, device)
        
        train_losses.append(tr_loss)
        train_dists.append(tr_dist)
        valid_dists.append(val_dist)
        
        # Bias-Variance 분석
        bias = tr_dist  # train error (irreducible error는 0으로 가정)
        variance = val_dist - tr_dist  # val - train
        
        status = ""
        if val_dist < best_val:
            best_val = val_dist
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status = " *BEST*"
        
        print(f"[{epoch:03d}] train_loss={tr_loss:.4f} | "
              f"train_dist={tr_dist:.4f} | valid_dist={val_dist:.4f} | "
              f"gap={variance:.4f}{status}")
    
    # 최종 분석
    print("\n[Evaluation Summary]")
    print(f"  Best valid_dist: {best_val:.4f}")
    print(f"  Final train_dist: {train_dists[-1]:.4f}")
    print(f"  Final valid_dist: {valid_dists[-1]:.4f}")
    print(f"  Final gap (variance): {valid_dists[-1] - train_dists[-1]:.4f}")
    
    if train_dists[-1] > 15:
        print("\n  ⚠️ HIGH BIAS (Underfitting):")
        print("     - Increase model capacity (d_model, n_layers)")
        print("     - Train longer")
        print("     - Add features")
    
    if valid_dists[-1] - train_dists[-1] > 2:
        print("\n  ⚠️ HIGH VARIANCE (Overfitting):")
        print("     - Add regularization (dropout, weight_decay)")
        print("     - More data / augmentation")
        print("     - Reduce model capacity")
    
    return best_state, best_val, (train_losses, train_dists, valid_dists)


# ============================================================
# 7. Main
# ============================================================
def main():
    print("=" * 60)
    print(f"Transformer Baseline - Mode: {args.mode.upper()}")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Dropout: {DROPOUT} (NO dropout)")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Model normalization: LayerNorm")
    
    seed_everything(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_data(TRAIN_PATH)
    print(f"  Total rows: {len(df)}")
    
    # Build vocab
    print("\n[2] Building vocab...")
    type_vocab = build_vocab(df, "type_name")
    res_vocab = build_vocab(df, "result_name")
    print(f"  type_vocab size: {len(type_vocab)} (0=PAD, 1~={len(type_vocab)-1})")
    print(f"  res_vocab size: {len(res_vocab)} (0=PAD, 1~={len(res_vocab)-1})")
    
    # Debug: vocab 내용 확인
    print("\n  [Vocab Sample]")
    print(f"    type_vocab: {list(type_vocab.items())[:5]} ...")
    print(f"    res_vocab: {list(res_vocab.items())[:5]} ...")
    
    # Build episodes
    print("\n[3] Building episodes...")
    episodes, targets, keys, game_ids = build_episodes(df, type_vocab, res_vocab, debug_print=True)
    print(f"  Total episodes: {len(episodes)}")
    
    # Debug: 데이터 샘플 확인 (raw 값)
    print("\n  [Data Sample - RAW VALUES]")
    sample_ep = episodes[0]
    print(f"    Episode 0 length: {len(sample_ep['cont'])}")
    print(f"    cont[0] (sx,sy,ex,ey,dt): {sample_ep['cont'][0]}")
    print(f"    type_id range: [{sample_ep['type_id'].min()}, {sample_ep['type_id'].max()}]")
    print(f"    res_id range: [{sample_ep['res_id'].min()}, {sample_ep['res_id'].max()}]")
    print(f"    target: {targets[0]}")
    
    # Save vocab
    with open(os.path.join(MODEL_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump({"type_vocab": type_vocab, "res_vocab": res_vocab}, f)
    
    # Split data
    gkf = GroupKFold(n_splits=N_SPLITS)
    game_arr = np.array(game_ids)
    
    tr_idx, va_idx = None, None
    for i, (tr, va) in enumerate(gkf.split(episodes, groups=game_arr)):
        if i == FOLD:
            tr_idx, va_idx = tr, va
            break
    
    train_eps = [episodes[i] for i in tr_idx]
    train_tgt = [targets[i] for i in tr_idx]
    train_keys = [keys[i] for i in tr_idx]
    
    valid_eps = [episodes[i] for i in va_idx]
    valid_tgt = [targets[i] for i in va_idx]
    valid_keys = [keys[i] for i in va_idx]
    
    print(f"\n[4] Fold {FOLD}: Train={len(train_eps)}, Valid={len(valid_eps)}")
    
    # DataLoader
    train_ds = EpisodeDataset(train_eps, train_tgt, train_keys)
    valid_ds = EpisodeDataset(valid_eps, valid_tgt, valid_keys)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model
    n_type = len(type_vocab)
    n_res = len(res_vocab)
    cont_dim = 5
    
    model = SimpleTransformer(
        cont_dim=cont_dim,
        n_type=n_type,
        n_res=n_res,
        emb_dim=EMB_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_ff=DIM_FF,
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"\n[5] Model created")
    print(f"  n_type: {n_type}")
    print(f"  n_res: {n_res}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get sample batch
    sample_batch = next(iter(train_loader))
    
    # ============================================================
    # MODE: check - Shape/Forward 확인만
    # ============================================================
    if args.mode == 'check':
        check_model_runs(model, sample_batch, DEVICE)
        print("\n" + "=" * 60)
        print("CHECK COMPLETE!")
        print("=" * 60)
        return
    
    # ============================================================
    # MODE: overfit - Single batch 오버피팅
    # ============================================================
    if args.mode == 'overfit':
        check_model_runs(model, sample_batch, DEVICE)
        
        # 모델 초기화 (check에서 backward로 인한 영향 제거)
        model = SimpleTransformer(
            cont_dim=cont_dim,
            n_type=n_type,
            n_res=n_res,
            emb_dim=EMB_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dim_ff=DIM_FF,
            dropout=DROPOUT
        ).to(DEVICE)
        
        losses, dists = overfit_single_batch(model, sample_batch, DEVICE, epochs=EPOCHS, lr=LR)
        
        print("\n" + "=" * 60)
        print("OVERFIT TEST COMPLETE!")
        print("=" * 60)
        return
    
    # ============================================================
    # MODE: train - Full training
    # ============================================================
    if args.mode == 'train':
        check_model_runs(model, sample_batch, DEVICE)
        
        # 모델 초기화
        model = SimpleTransformer(
            cont_dim=cont_dim,
            n_type=n_type,
            n_res=n_res,
            emb_dim=EMB_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dim_ff=DIM_FF,
            dropout=DROPOUT
        ).to(DEVICE)
        
        best_state, best_val, history = train_full(
            model, train_loader, valid_loader, DEVICE, 
            epochs=EPOCHS, lr=LR
        )
        
        # Save
        save_dir = os.path.join(MODEL_DIR, f"seed_{SEED}_fold_{FOLD}")
        os.makedirs(save_dir, exist_ok=True)
        
        if best_state:
            torch.save(best_state, os.path.join(save_dir, "best_model.pt"))
            print(f"\n  Model saved to {save_dir}/best_model.pt")
        
        config = {
            "cont_dim": cont_dim,
            "n_type": n_type,
            "n_res": n_res,
            "emb_dim": EMB_DIM,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dim_ff": DIM_FF,
            "dropout": DROPOUT,
        }
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print(f"  Best valid_dist: {best_val:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
