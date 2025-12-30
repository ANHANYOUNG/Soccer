"""
Transformer Baseline - Inference Script
"""

import os
import pickle
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import math


# ============================================================
# 1. Model (Same as train.py)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    def __init__(self, cont_dim, n_type, n_res, emb_dim=16, d_model=64,
                 n_heads=4, n_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        
        self.type_emb = nn.Embedding(n_type, emb_dim, padding_idx=0)
        self.res_emb = nn.Embedding(n_res, emb_dim, padding_idx=0)
        
        in_dim = cont_dim + emb_dim * 2
        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
    
    def forward(self, cont, type_ids, res_ids, lengths):
        B, T, _ = cont.shape
        device = cont.device
        
        te = self.type_emb(type_ids)
        re = self.res_emb(res_ids)
        
        x = torch.cat([cont, te, re], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_enc(x)
        
        idx = torch.arange(T, device=device).unsqueeze(0)
        pad_mask = idx >= lengths.unsqueeze(1)
        
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        
        last_idx = (lengths - 1).long()
        batch_idx = torch.arange(B, device=device)
        pooled = x[batch_idx, last_idx]
        
        out = self.head(pooled)
        return out


# ============================================================
# 2. Feature Building
# ============================================================
def build_features(g, type_vocab, res_vocab):
    """단일 에피소드 피처 생성"""
    g = g.sort_values(["time_seconds", "action_id"]).reset_index(drop=True)
    
    # 결측 처리
    g["type_name"] = g["type_name"].fillna("__NA__")
    g["result_name"] = g["result_name"].fillna("__NA__")
    
    # 시간 차분
    t = g["time_seconds"].astype("float32").values
    dt = np.zeros_like(t, dtype="float32")
    dt[1:] = t[1:] - t[:-1]
    dt = np.clip(dt, 0, None)
    
    # 좌표
    sx = np.nan_to_num(g["start_x"].astype("float32").values, 0.0)
    sy = np.nan_to_num(g["start_y"].astype("float32").values, 0.0)
    ex = np.nan_to_num(g["end_x"].astype("float32").values, 0.0)
    ey = np.nan_to_num(g["end_y"].astype("float32").values, 0.0)
    
    # 마스킹: 마지막 end 좌표
    ex[-1] = 0.0
    ey[-1] = 0.0
    
    # 연속형 피처
    cont = np.stack([sx, sy, ex, ey, dt], axis=1).astype("float32")
    
    # 범주형 인덱스
    type_ids = np.array([type_vocab.get(v, 1) for v in g["type_name"]], dtype="int64")
    res_ids = np.array([res_vocab.get(v, 1) for v in g["result_name"]], dtype="int64")
    
    return cont, type_ids, res_ids


# ============================================================
# 3. Main Inference
# ============================================================
def main():
    MODEL_DIR = "./models"
    TEST_PATH = "../data/test.csv"
    SUBMISSION_PATH = "../data/sample_submission.csv"
    DATA_ROOT = "../data"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    print("=" * 60)
    print("Transformer Baseline - Inference")
    print("=" * 60)
    
    # Load vocab
    with open(os.path.join(MODEL_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    type_vocab = vocab["type_vocab"]
    res_vocab = vocab["res_vocab"]
    
    # Find models
    model_dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "seed_*_fold_*")))
    print(f"Found {len(model_dirs)} models")
    
    # Load models
    models = []
    for md in model_dirs:
        with open(os.path.join(md, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        model = SimpleTransformer(**config).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(md, "best_model.pt"), map_location=DEVICE))
        model.eval()
        models.append(model)
        print(f"  Loaded: {md}")
    
    # Load test data
    test_meta = pd.read_csv(TEST_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)
    
    pred_map = {}
    
    with torch.no_grad():
        for _, row in tqdm(test_meta.iterrows(), total=len(test_meta), desc="Inference"):
            ep_key = row["game_episode"]
            
            rel_path = str(row["path"])
            rel_path = rel_path[2:] if rel_path.startswith("./") else rel_path
            full_path = os.path.join(DATA_ROOT, rel_path)
            
            g = pd.read_csv(full_path)
            cont, type_ids, res_ids = build_features(g, type_vocab, res_vocab)
            
            # To tensor
            cont_t = torch.from_numpy(cont).unsqueeze(0).float().to(DEVICE)
            type_t = torch.from_numpy(type_ids).unsqueeze(0).long().to(DEVICE)
            res_t = torch.from_numpy(res_ids).unsqueeze(0).long().to(DEVICE)
            lengths = torch.tensor([cont.shape[0]], dtype=torch.long).to(DEVICE)
            
            # Ensemble
            preds = []
            for model in models:
                pred = model(cont_t, type_t, res_t, lengths)
                preds.append(pred.cpu().numpy())
            
            avg_pred = np.mean(preds, axis=0).squeeze()
            
            # Clipping
            avg_pred[0] = np.clip(avg_pred[0], 0, 105)
            avg_pred[1] = np.clip(avg_pred[1], 0, 68)
            
            pred_map[ep_key] = avg_pred
    
    # Build submission
    preds_x, preds_y = [], []
    missing = []
    
    for ep in submission["game_episode"]:
        if ep not in pred_map:
            missing.append(ep)
            preds_x.append(52.5)  # 중앙값
            preds_y.append(34.0)
        else:
            preds_x.append(float(pred_map[ep][0]))
            preds_y.append(float(pred_map[ep][1]))
    
    if missing:
        print(f"Warning: {len(missing)} missing episodes")
    
    submission["end_x"] = preds_x
    submission["end_y"] = preds_y
    
    # Save
    i = 0
    while True:
        out_name = f"submit_{i}.csv"
        if not os.path.exists(out_name):
            break
        i += 1
    
    submission[["game_episode", "end_x", "end_y"]].to_csv(out_name, index=False)
    print(f"\nSaved: {out_name}")
    print(f"Ensemble: {len(models)} models")


if __name__ == "__main__":
    main()
