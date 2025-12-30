"""
Transformer-based Pass Prediction Model - Inference Script
Based on LSTM_2.ipynb architecture, converted to Transformer
Supports multi-seed ensemble
"""

import os
import pickle
import glob

import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch import nn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math


# ============================================================
# 1. Model Definition (same as train.py)
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
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
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
        self.use_last_token = use_last_token  # Last-token pooling 옵션
        
        self.type_emb = nn.Embedding(n_type, emb_dim, padding_idx=0)
        self.res_emb = nn.Embedding(n_res, emb_dim, padding_idx=0)
        
        in_dim = cont_dim + emb_dim + emb_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)  # LayerNorm: 패딩 영향 없음
        
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Attention pooling (used when not using CLS token and not using last token)
        if not use_cls_token and not use_last_token:
            self.attn_pool = nn.Linear(d_model, 1)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        out_dim = 4 if use_gaussian_nll else 2
        
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
        
        te = self.type_emb(type_pad)
        re = self.res_emb(res_pad)
        
        x = torch.cat([cont_pad, te, re], dim=-1)
        x = self.input_proj(x)
        
        # Apply LayerNorm (패딩 영향 없음)
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
        
        # Pooling
        if self.use_cls_token:
            pooled = x[:, 0, :]
        elif self.use_last_token:
            # Last-token pooling: 마지막 유효 토큰만 사용
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


# ============================================================
# 2. Feature Building
# ============================================================

# Stadium constants
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


def build_episode_from_df(g, le_type, le_res):
    """Build features for a single episode
    Returns: cont, type_id, res_id, last_start_x, last_start_y
    """
    g = g.sort_values(["time_seconds", "action_id"]).reset_index(drop=True)
    
    g["type_name"] = g["type_name"].fillna("__NA_TYPE__")
    g["result_name"] = g["result_name"].fillna("__NA_RES__")
    
    # Handle unseen labels safely
    g.loc[~g["type_name"].isin(le_type.classes_), "type_name"] = "__NA_TYPE__"
    g.loc[~g["result_name"].isin(le_res.classes_), "result_name"] = "__NA_RES__"
    
    type_id = le_type.transform(g["type_name"]).astype("int64") + 1
    res_id = le_res.transform(g["result_name"]).astype("int64") + 1
    
    # dt
    t = g["time_seconds"].astype("float32").values
    dt = np.zeros_like(t, dtype="float32")
    dt[1:] = t[1:] - t[:-1]
    dt[dt < 0] = 0.0
    
    # coordinates
    sx = g["start_x"].astype("float32").values
    sy = g["start_y"].astype("float32").values
    ex = g["end_x"].astype("float32").values
    ey = g["end_y"].astype("float32").values
    
    # replace nan to 0.0
    sx = np.nan_to_num(sx, nan=0.0)
    sy = np.nan_to_num(sy, nan=0.0)
    ex = np.nan_to_num(ex, nan=0.0)
    ey = np.nan_to_num(ey, nan=0.0)
    
    # Get last start position for delta recovery
    last_start_x = float(sx[-1])
    last_start_y = float(sy[-1])
    
    # mask last end for leak-safe
    ex_mask = ex.copy()
    ey_mask = ey.copy()
    ex_mask[-1] = 0.0
    ey_mask[-1] = 0.0
    
    # goal segment distance
    dxg = GOAL_X - sx
    dy_goal = np.maximum(0.0, np.maximum(GOAL_Y_L - sy, sy - GOAL_Y_R)).astype("float32")
    dist_to_goal = np.sqrt(dxg**2 + dy_goal**2).astype("float32")
    
    # goal view angle
    alpha_L = np.arctan2(GOAL_Y_L - sy, GOAL_X - sx).astype("float32")
    alpha_R = np.arctan2(GOAL_Y_R - sy, GOAL_X - sx).astype("float32")
    theta_view = np.abs(alpha_R - alpha_L).astype("float32")
    
    # half line features
    in_own_half = (sx < HALF_X).astype("float32")
    
    # penalty box features
    dx_box = np.maximum(0.0, P_BOX_X_MIN - sx).astype("float32")
    dy_box = np.maximum(0.0, np.maximum(P_BOX_Y_MIN - sy, sy - P_BOX_Y_MAX)).astype("float32")
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
    
    # continuous features
    cont = np.stack([
        sx, sy, ex_mask, ey_mask, dt,
        dist_to_goal, theta_view, in_own_half,
        dist_p_box, prev_dx, prev_dy, prev_valid
    ], axis=1).astype("float32")
    
    return cont, type_id, res_id, last_start_x, last_start_y


# ============================================================
# 3. Main Inference
# ============================================================

def main():
    # Paths
    MODEL_DIR = "./models"
    TEST_META_PATH = "../data/test.csv"
    SUBMISSION_PATH = "../data/sample_submission.csv"
    DATA_ROOT = "../data"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    print("=" * 60)
    print("Transformer Pass Prediction - Ensemble Inference")
    print("=" * 60)
    
    # Find all seed_*_fold_* directories (new format: 3 seeds × 5 folds = 15 models)
    seed_dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "seed_*_fold_*")))
    
    if len(seed_dirs) == 0:
        # Fallback: try seed_* only (old format)
        seed_dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "seed_*")))
        if len(seed_dirs) == 0:
            # Final fallback: single model
            print("\n[INFO] No seed directories found, using single model mode")
            seed_dirs = [MODEL_DIR]
            single_mode = True
        else:
            print(f"\n[INFO] Found {len(seed_dirs)} seed-only models (old format)")
            single_mode = False
    else:
        print(f"\n[INFO] Found {len(seed_dirs)} seed+fold models for ensemble")
        single_mode = False
    
    # Load all models
    models = []
    configs = []
    predict_delta = False
    use_gaussian_nll = False
    
    for seed_dir in seed_dirs:
        print(f"\nLoading model from: {seed_dir}")
        
        # Load label encoders (same for all seeds)
        if len(models) == 0:
            if single_mode:
                enc_path = os.path.join(seed_dir, "label_encoders.pkl")
            else:
                enc_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
            
            with open(enc_path, "rb") as f:
                encoders = pickle.load(f)
            le_type = encoders["le_type"]
            le_res = encoders["le_res"]
        
        # Load model config
        config_path = os.path.join(seed_dir, "model_config.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        
        # Check for predict_delta and use_gaussian_nll flags
        if "predict_delta" in config:
            predict_delta = config.pop("predict_delta")
        if "use_gaussian_nll" not in config:
            config["use_gaussian_nll"] = False
        if "use_last_token" not in config:
            config["use_last_token"] = False  # 이전 모델과 호환성 유지
        
        # Remove deprecated parameters for compatibility
        if "use_skip_start" in config:
            config.pop("use_skip_start")
        
        use_gaussian_nll = config.get("use_gaussian_nll", False)
        use_last_token = config.get("use_last_token", False)
        
        print(f"  Config: d_model={config.get('d_model', 128)}, "
              f"n_layers={config.get('n_layers', 4)}, "
              f"use_cls_token={config.get('use_cls_token', False)}, "
              f"use_gaussian_nll={use_gaussian_nll}, "
              f"use_last_token={use_last_token}")
        
        # Create and load model
        model = PassTransformer(**config).to(DEVICE)
        model_path = os.path.join(seed_dir, "best_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        models.append(model)
        configs.append(config)
    
    print(f"\n[INFO] Loaded {len(models)} models")
    print(f"[INFO] predict_delta={predict_delta}, use_gaussian_nll={use_gaussian_nll}")
    
    # Load test data
    print("\n[Running Inference]...")
    test_meta = pd.read_csv(TEST_META_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)
    
    pred_map = {}
    
    with torch.no_grad():
        for _, row in tqdm(test_meta.iterrows(), total=len(test_meta), desc="Inference"):
            game_episode = row["game_episode"]
            
            rel_path = str(row["path"])
            rel_path = rel_path[2:] if rel_path.startswith("./") else rel_path
            full_path = os.path.join(DATA_ROOT, rel_path)
            
            g = pd.read_csv(full_path)
            cont, type_id, res_id, last_start_x, last_start_y = build_episode_from_df(g, le_type, le_res)
            
            # Convert to tensors
            cont_t = torch.from_numpy(cont).unsqueeze(0).to(DEVICE)
            type_t = torch.from_numpy(type_id).unsqueeze(0).to(DEVICE)
            res_t = torch.from_numpy(res_id).unsqueeze(0).to(DEVICE)
            lengths = torch.tensor([cont.shape[0]], dtype=torch.long).to(DEVICE)
            
            # Ensemble prediction (average)
            ensemble_preds = []
            for model in models:
                pred = model(cont_t.float(), type_t.long(), res_t.long(), lengths)
                pred_np = pred.squeeze(0).detach().cpu().numpy()
                
                if use_gaussian_nll:
                    # Output is [mu_x, mu_y, log_var_x, log_var_y]
                    # Take only mean (mu_x, mu_y)
                    pred_xy = pred_np[:2]
                else:
                    pred_xy = pred_np[:2]
                
                ensemble_preds.append(pred_xy)
            
            # Average predictions
            avg_pred = np.mean(ensemble_preds, axis=0).astype("float32")
            
            # Convert delta to absolute if needed
            if predict_delta:
                end_x = last_start_x + avg_pred[0]
                end_y = last_start_y + avg_pred[1]
                avg_pred = np.array([end_x, end_y], dtype="float32")
            
            pred_map[game_episode] = avg_pred
    
    # Align predictions to submission format
    preds_x = []
    preds_y = []
    missing = []
    
    for ge in submission["game_episode"].tolist():
        if ge not in pred_map:
            missing.append(ge)
            preds_x.append(0.0)
            preds_y.append(0.0)
            continue
        px, py = pred_map[ge]
        preds_x.append(float(px))
        preds_y.append(float(py))
    
    if len(missing) > 0:
        print(f"Warning: missing episodes in pred_map: {len(missing)}")
    
    submission["end_x"] = preds_x
    submission["end_y"] = preds_y
    
    print(f"Inference done: {len(submission)} episodes")
    print(f"Ensemble size: {len(models)} models")
    
    # Save submission to submit directory
    submit_dir = "./submit"
    os.makedirs(submit_dir, exist_ok=True)
    
    base = "Transformer_submit"
    ext = ".csv"
    
    i = 0
    while True:
        out_name = os.path.join(submit_dir, f"{base}_{i}{ext}")
        if not os.path.exists(out_name):
            break
        i += 1
    
    submission[["game_episode", "end_x", "end_y"]].to_csv(out_name, index=False)
    print(f"\n{'='*60}")
    print(f"Saved: {out_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
