"""
LightGBM-based Pass Prediction Model - Inference Script
Episode-level prediction for test data
"""

import os
import pickle
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# 1. Constants
# ============================================================

# Paths
MODEL_DIR = "./models"
TEST_META_PATH = "../data/test.csv"
SUBMISSION_PATH = "../data/sample_submission.csv"
DATA_ROOT = "../data"

# Stadium constants
STADIUM_X, STADIUM_Y = 105.0, 68.0
CENTER_X, CENTER_Y = STADIUM_X / 2.0, STADIUM_Y / 2.0
GOAL_X = STADIUM_X
GOAL_POST_HALF = 3.66
GOAL_Y_L = CENTER_Y - GOAL_POST_HALF
GOAL_Y_R = CENTER_Y + GOAL_POST_HALF


# ============================================================
# 2. Feature Engineering (same as train.py)
# ============================================================

def add_geometry_features(df):
    """Add geometric features from start positions"""
    df['dist_from_center'] = np.sqrt(
        (df['start_x'] - CENTER_X)**2 + (df['start_y'] - CENTER_Y)**2
    )
    df['dist_to_goal'] = np.sqrt(
        (df['start_x'] - GOAL_X)**2 + (df['start_y'] - CENTER_Y)**2
    )
    df['angle_from_origin'] = np.arctan2(
        df['start_y'] - CENTER_Y, df['start_x'] - CENTER_X
    )
    dx = GOAL_X - df['start_x']
    alpha_L = np.arctan2(GOAL_Y_L - df['start_y'], dx)
    alpha_R = np.arctan2(GOAL_Y_R - df['start_y'], dx)
    df['goal_view_angle'] = np.abs(alpha_R - alpha_L)
    return df


def add_time_features(df):
    """Add time-related features"""
    df['normalized_time'] = df['time_seconds'] % 2700
    df['time_in_half'] = df['normalized_time'] / 2700.0
    return df


def build_episode_features_single(g, le_team, le_player, le_type):
    """
    Build features for a single episode (test time).
    Same logic as training, but for one episode at a time.
    """
    g = g.sort_values(['time_seconds', 'action_id']).reset_index(drop=True)
    
    if len(g) < 1:
        return None, None, None
    
    # Add features
    g = add_geometry_features(g)
    g = add_time_features(g)
    
    # Calculate event-level displacement
    g['event_dx'] = g['end_x'] - g['start_x']
    g['event_dy'] = g['end_y'] - g['start_y']
    g['event_dist'] = np.sqrt(g['event_dx']**2 + g['event_dy']**2)
    
    # Last event
    last_row = g.iloc[-1]
    last_start_x = last_row['start_x']
    last_start_y = last_row['start_y']
    
    # Leak-safe aggregation
    g_safe = g.iloc[:-1] if len(g) > 1 else g.iloc[:1]
    
    # Encode categorical (handle unseen values)
    def safe_encode(le, val):
        val_str = str(val)
        if val_str in le.classes_:
            return le.transform([val_str])[0]
        else:
            return 0  # Default for unseen
    
    feat = {
        'last_start_x': last_start_x,
        'last_start_y': last_start_y,
        'last_dist_to_goal': last_row['dist_to_goal'],
        'last_dist_from_center': last_row['dist_from_center'],
        'last_angle': last_row['angle_from_origin'],
        'last_goal_view': last_row['goal_view_angle'],
        'last_time': last_row['normalized_time'],
        
        'ep_len': len(g),
        'ep_mean_x': g_safe['start_x'].mean(),
        'ep_mean_y': g_safe['start_y'].mean(),
        'ep_std_x': g_safe['start_x'].std() if len(g_safe) > 1 else 0,
        'ep_std_y': g_safe['start_y'].std() if len(g_safe) > 1 else 0,
        'ep_mean_dist_goal': g_safe['dist_to_goal'].mean(),
        'ep_mean_dist_center': g_safe['dist_from_center'].mean(),
        
        'ep_total_dist': g_safe['event_dist'].sum(),
        'ep_mean_dx': g_safe['event_dx'].mean(),
        'ep_mean_dy': g_safe['event_dy'].mean(),
        'ep_progression_x': last_start_x - g.iloc[0]['start_x'],
        'ep_progression_y': last_start_y - g.iloc[0]['start_y'],
        
        'last_team_id': safe_encode(le_team, last_row['team_id']),
        'last_player_id': safe_encode(le_player, last_row['player_id']),
        'last_type': safe_encode(le_type, last_row['type_name']),
        
        'n_passes': (g['type_name'] == 'Pass').sum(),
        'n_carries': (g['type_name'] == 'Carry').sum(),
        
        'in_final_third': int(last_start_x > 70),
        'in_penalty_area': int(last_start_x > 88.5 and 13.84 < last_start_y < 54.16),
    }
    
    return feat, last_start_x, last_start_y


# ============================================================
# 3. Main Inference
# ============================================================

def main():
    print("=" * 60)
    print("LightGBM Pass Prediction - Inference")
    print("=" * 60)
    
    # Load encoders
    print("\n[1/3] Loading model and encoders...")
    with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    
    le_team = encoders['le_team']
    le_player = encoders['le_player']
    le_type = encoders['le_type']
    feature_cols = encoders['feature_cols']
    
    # Load model
    model_path = os.path.join(MODEL_DIR, 'lgbm_fold0.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded: {model_path}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Load test metadata
    print("\n[2/3] Loading test data...")
    test_meta = pd.read_csv(TEST_META_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)
    
    print(f"Test episodes: {len(test_meta)}")
    
    # Predict
    print("\n[3/3] Running inference...")
    pred_map = {}
    
    for _, row in tqdm(test_meta.iterrows(), total=len(test_meta), desc="Inference"):
        game_episode = row['game_episode']
        
        # Load episode CSV
        rel_path = str(row['path'])
        rel_path = rel_path[2:] if rel_path.startswith('./') else rel_path
        full_path = os.path.join(DATA_ROOT, rel_path)
        
        g = pd.read_csv(full_path)
        
        # Build features
        feat, last_start_x, last_start_y = build_episode_features_single(
            g, le_team, le_player, le_type
        )
        
        if feat is None:
            # Fallback: predict center of field
            pred_map[game_episode] = (CENTER_X, CENTER_Y)
            continue
        
        # Create feature vector
        feat_df = pd.DataFrame([feat])
        
        # Ensure column order matches training
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0
        feat_df = feat_df[feature_cols].fillna(0)
        
        # Predict (dx, dy)
        pred = model.predict(feat_df)
        dx, dy = pred[0, 0], pred[0, 1]
        
        # Convert to absolute coordinates
        end_x = last_start_x + dx
        end_y = last_start_y + dy
        
        # Clip to stadium bounds
        end_x = np.clip(end_x, 0, STADIUM_X)
        end_y = np.clip(end_y, 0, STADIUM_Y)
        
        pred_map[game_episode] = (end_x, end_y)
    
    # Build submission
    preds_x = []
    preds_y = []
    missing = []
    
    for ge in submission['game_episode'].tolist():
        if ge not in pred_map:
            missing.append(ge)
            preds_x.append(CENTER_X)
            preds_y.append(CENTER_Y)
            continue
        px, py = pred_map[ge]
        preds_x.append(float(px))
        preds_y.append(float(py))
    
    if len(missing) > 0:
        print(f"Warning: {len(missing)} episodes not found in predictions")
    
    submission['end_x'] = preds_x
    submission['end_y'] = preds_y
    
    # Save submission
    base = "LGBM_2_submit"
    ext = ".csv"
    
    i = 0
    while True:
        out_name = f"{base}_{i}{ext}"
        if not os.path.exists(out_name):
            break
        i += 1
    
    submission[['game_episode', 'end_x', 'end_y']].to_csv(out_name, index=False)
    
    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"Saved: {out_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
