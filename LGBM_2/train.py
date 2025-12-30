"""
LightGBM-based Pass Prediction Model - Training Script
Episode-level aggregation for proper prediction
"""

import os
import pickle
import warnings

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# 1. Hyperparameters
# ============================================================

SEED = 42
N_SPLITS = 5
FOLD = 0

# LightGBM parameters
LGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'random_state': SEED,
    'verbose': -1
}

# Paths
TRAIN_PATH = "../data/train.csv"
MODEL_SAVE_DIR = "./models"

# Stadium constants
STADIUM_X, STADIUM_Y = 105.0, 68.0
CENTER_X, CENTER_Y = STADIUM_X / 2.0, STADIUM_Y / 2.0
GOAL_X = STADIUM_X
GOAL_POST_HALF = 3.66
GOAL_Y_L = CENTER_Y - GOAL_POST_HALF
GOAL_Y_R = CENTER_Y + GOAL_POST_HALF


# ============================================================
# 2. Feature Engineering Functions
# ============================================================

def add_geometry_features(df):
    """Add geometric features from start positions"""
    # 경기장 중심 기준 (52.5, 34)
    df['dist_from_center'] = np.sqrt(
        (df['start_x'] - CENTER_X)**2 + (df['start_y'] - CENTER_Y)**2
    )
    
    # 골문까지 거리 (골문 중심: 105, 34)
    df['dist_to_goal'] = np.sqrt(
        (df['start_x'] - GOAL_X)**2 + (df['start_y'] - CENTER_Y)**2
    )
    
    # 원점 기준 각도
    df['angle_from_origin'] = np.arctan2(
        df['start_y'] - CENTER_Y, df['start_x'] - CENTER_X
    )
    
    # 골문 시야각
    dx = GOAL_X - df['start_x']
    alpha_L = np.arctan2(GOAL_Y_L - df['start_y'], dx)
    alpha_R = np.arctan2(GOAL_Y_R - df['start_y'], dx)
    df['goal_view_angle'] = np.abs(alpha_R - alpha_L)
    
    return df


def add_time_features(df):
    """Add time-related features"""
    # 45분(2700초) 기준 정규화
    df['normalized_time'] = df['time_seconds'] % 2700
    df['time_in_half'] = df['normalized_time'] / 2700.0
    return df


def build_episode_features(df):
    """
    Build episode-level aggregated features.
    This is the key difference from event-level prediction.
    """
    
    print("Building episode-level features...")
    
    # Sort by time within each episode
    df = df.sort_values(['game_episode', 'time_seconds', 'action_id']).reset_index(drop=True)
    
    # Add basic features
    df = add_geometry_features(df)
    df = add_time_features(df)
    
    # Calculate event-level displacement
    df['event_dx'] = df['end_x'] - df['start_x']
    df['event_dy'] = df['end_y'] - df['start_y']
    df['event_dist'] = np.sqrt(df['event_dx']**2 + df['event_dy']**2)
    
    episode_features = []
    targets = []
    episode_keys = []
    game_ids = []
    
    for key, g in tqdm(df.groupby('game_episode'), desc="Processing episodes"):
        g = g.reset_index(drop=True)
        
        if len(g) < 2:
            continue
        
        # Find last Pass event for target
        if g.iloc[-1]['type_name'] != 'Pass':
            pass_idxs = g.index[g['type_name'] == 'Pass']
            if len(pass_idxs) == 0:
                continue
            g = g.loc[:pass_idxs[-1]].reset_index(drop=True)
            if len(g) < 2:
                continue
        
        # Target: last event's end position
        tx, ty = float(g.iloc[-1]['end_x']), float(g.iloc[-1]['end_y'])
        if np.isnan(tx) or np.isnan(ty):
            continue
        
        # Last event features (마지막 이벤트, 즉 예측 대상 직전까지)
        last_row = g.iloc[-1]
        last_start_x = last_row['start_x']
        last_start_y = last_row['start_y']
        
        # Leak-safe: exclude last event's end from aggregation
        g_safe = g.iloc[:-1] if len(g) > 1 else g.iloc[:1]
        
        # Aggregate features for episode
        feat = {
            # Last event position (input for prediction)
            'last_start_x': last_start_x,
            'last_start_y': last_start_y,
            'last_dist_to_goal': last_row['dist_to_goal'],
            'last_dist_from_center': last_row['dist_from_center'],
            'last_angle': last_row['angle_from_origin'],
            'last_goal_view': last_row['goal_view_angle'],
            'last_time': last_row['normalized_time'],
            
            # Episode statistics (leak-safe, excluding last)
            'ep_len': len(g),
            'ep_mean_x': g_safe['start_x'].mean(),
            'ep_mean_y': g_safe['start_y'].mean(),
            'ep_std_x': g_safe['start_x'].std() if len(g_safe) > 1 else 0,
            'ep_std_y': g_safe['start_y'].std() if len(g_safe) > 1 else 0,
            'ep_mean_dist_goal': g_safe['dist_to_goal'].mean(),
            'ep_mean_dist_center': g_safe['dist_from_center'].mean(),
            
            # Movement statistics
            'ep_total_dist': g_safe['event_dist'].sum(),
            'ep_mean_dx': g_safe['event_dx'].mean(),
            'ep_mean_dy': g_safe['event_dy'].mean(),
            'ep_progression_x': last_start_x - g.iloc[0]['start_x'],  # X 진행
            'ep_progression_y': last_start_y - g.iloc[0]['start_y'],  # Y 진행
            
            # Categorical (encoded later)
            'last_team_id': last_row['team_id'],
            'last_player_id': last_row['player_id'],
            'last_type': last_row['type_name'],
            
            # Count features
            'n_passes': (g['type_name'] == 'Pass').sum(),
            'n_carries': (g['type_name'] == 'Carry').sum(),
            
            # Zone features
            'in_final_third': int(last_start_x > 70),
            'in_penalty_area': int(last_start_x > 88.5 and 13.84 < last_start_y < 54.16),
        }
        
        episode_features.append(feat)
        
        # Target: delta (dx, dy) from last_start to end
        dx = tx - last_start_x
        dy = ty - last_start_y
        targets.append({'dx': dx, 'dy': dy, 'end_x': tx, 'end_y': ty})
        
        episode_keys.append(key)
        game_ids.append(g.iloc[0]['game_id'])
    
    features_df = pd.DataFrame(episode_features)
    targets_df = pd.DataFrame(targets)
    
    return features_df, targets_df, episode_keys, game_ids


# ============================================================
# 3. Training
# ============================================================

def train_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with MultiOutputRegressor"""
    
    lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)
    model = MultiOutputRegressor(lgbm)
    
    model.fit(X_train, y_train)
    
    # Validation
    val_pred = model.predict(X_val)
    
    # Euclidean distance (competition metric)
    distances = np.sqrt((val_pred[:, 0] - y_val.iloc[:, 0])**2 + 
                        (val_pred[:, 1] - y_val.iloc[:, 1])**2)
    mean_dist = distances.mean()
    
    return model, mean_dist


def main():
    print("=" * 60)
    print("LightGBM Pass Prediction - Training")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded {len(df)} events")
    
    # Build episode features
    print("\n[2/4] Building episode features...")
    features_df, targets_df, episode_keys, game_ids = build_episode_features(df)
    print(f"Total episodes: {len(features_df)}")
    
    # Encode categorical features
    print("\n[3/4] Encoding categorical features...")
    le_team = LabelEncoder()
    le_player = LabelEncoder()
    le_type = LabelEncoder()
    
    features_df['last_team_id'] = le_team.fit_transform(features_df['last_team_id'].astype(str))
    features_df['last_player_id'] = le_player.fit_transform(features_df['last_player_id'].astype(str))
    features_df['last_type'] = le_type.fit_transform(features_df['last_type'].astype(str))
    
    # Fill NaN
    features_df = features_df.fillna(0)
    
    # Feature columns
    feature_cols = [c for c in features_df.columns]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    
    # Save encoders
    encoders = {
        'le_team': le_team,
        'le_player': le_player,
        'le_type': le_type,
        'feature_cols': feature_cols
    }
    with open(os.path.join(MODEL_SAVE_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    
    # Split by game_id (GroupKFold)
    print("\n[4/4] Training with GroupKFold...")
    game_ids_arr = np.array(game_ids)
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(features_df, targets_df, groups=game_ids_arr)):
        if fold_i != FOLD:
            continue
        
        X_train = features_df.iloc[tr_idx]
        X_val = features_df.iloc[va_idx]
        
        # Target: predict (dx, dy) - delta from last_start
        y_train = targets_df[['dx', 'dy']].iloc[tr_idx]
        y_val = targets_df[['dx', 'dy']].iloc[va_idx]
        
        # Check group separation
        train_games = set(game_ids_arr[tr_idx])
        val_games = set(game_ids_arr[va_idx])
        overlap = train_games & val_games
        print(f"Train games: {len(train_games)}, Val games: {len(val_games)}, Overlap: {len(overlap)}")
        
        print(f"\nFold {fold_i}: Train {len(X_train)} | Val {len(X_val)}")
        
        # Train
        model, val_dist = train_model(X_train, y_train, X_val, y_val)
        
        print(f"Validation Euclidean Distance: {val_dist:.4f}")
        
        # Save model
        model_path = os.path.join(MODEL_SAVE_DIR, f'lgbm_fold{fold_i}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved: {model_path}")
        
        break  # Only train one fold for now
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
