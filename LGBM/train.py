import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# What model trains
PATH_TRAIN = "train.csv"

# Number of previous events to consider
K = 20

# Folder for saving models
MODEL_DIR = "models"

# Number of splits for cross-validation
N_SPLITS = 5

# Create run001, run002, ... directories for models
def get_next_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)

    existing = []
    for name in os.listdir(base_dir):
        if not name.startswith("run_"):
            continue
        try:
            existing.append(int(name.replace("run_", "")))
        except ValueError:
            continue

    next_n = (max(existing) + 1) if len(existing) > 0 else 1
    run_dir = os.path.join(base_dir, f"run_{next_n:03d}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

# Data loading
train = pd.read_csv(PATH_TRAIN)

# only for training
train["is_train"] = 1
events = train.copy()

# Categorizing and aligning data by time
events = events.sort_values(["game_episode", "time_seconds", "action_id"]).reset_index(drop=True)

events["event_idx"] = events.groupby("game_episode").cumcount()
events["n_events"] = events.groupby("game_episode")["event_idx"].transform("max") + 1
events["ep_idx_norm"] = events["event_idx"] / (events["n_events"] - 1).clip(lower=1)

# ----- What model trains -----TODO
# Time features
events["prev_time"] = events.groupby("game_episode")["time_seconds"].shift(1)
events["dt"] = (events["time_seconds"] - events["prev_time"]).fillna(0.0)

# Movement features
events["dx"] = events["end_x"] - events["start_x"]
events["dy"] = events["end_y"] - events["start_y"]
events["dist"] = np.sqrt(events["dx"] ** 2 + events["dy"] ** 2)
events["speed"] = events["dist"] / events["dt"].replace(0, 1e-3)

events["x_zone"] = (events["start_x"] / (105 / 7)).astype(int).clip(0, 6)
events["lane"] = pd.cut(
    events["start_y"],
    bins=[0, 68 / 3, 2 * 68 / 3, 68],
    labels=[0, 1, 2],
    include_lowest=True,
).astype(int)

# ----------------------
# 4. 라벨 및 episode-level 메타 (train 전용)
# ----------------------
train_events = events[events["is_train"] == 1].copy()
last_events = train_events.groupby("game_episode", as_index=False).tail(1).copy()

labels = last_events[["game_episode", "end_x", "end_y"]].rename(
    columns={"end_x": "target_x", "end_y": "target_y"}
)

ep_meta = last_events[["game_episode", "game_id", "team_id", "is_home", "period_id", "time_seconds"]].copy()
ep_meta = ep_meta.rename(columns={"team_id": "final_team_id"})

ep_meta["game_clock_min"] = np.where(
    ep_meta["period_id"] == 1,
    ep_meta["time_seconds"] / 60.0,
    45.0 + ep_meta["time_seconds"] / 60.0,
)

# ----------------------
# 5. 공격 팀 플래그
# ----------------------
events = events.merge(ep_meta[["game_episode", "final_team_id"]], on="game_episode", how="left")
events["is_final_team"] = (events["team_id"] == events["final_team_id"]).astype(int)

# ----------------------
# 6. 입력에서 마지막 이벤트 타깃 정보 가리기
# ----------------------
events["last_idx"] = events.groupby("game_episode")["event_idx"].transform("max")
events["is_last"] = (events["event_idx"] == events["last_idx"]).astype(int)

mask_last = events["is_last"] == 1
for col in ["end_x", "end_y", "dx", "dy", "dist", "speed"]:
    events.loc[mask_last, col] = np.nan

# ----------------------
# 7. 카테고리 인코딩
# ----------------------
events["type_name"] = events["type_name"].fillna("__NA_TYPE__")
events["result_name"] = events["result_name"].fillna("__NA_RES__")

le_type = LabelEncoder()
le_res = LabelEncoder()

events["type_id"] = le_type.fit_transform(events["type_name"])
events["res_id"] = le_res.fit_transform(events["result_name"])

if events["team_id"].dtype == "object":
    le_team = LabelEncoder()
    events["team_id_enc"] = le_team.fit_transform(events["team_id"])
else:
    events["team_id_enc"] = events["team_id"].astype(int)

# ----------------------
# 8. 마지막 K 이벤트만 사용 + pos_in_K (vectorized)
# ----------------------
events["rev_idx"] = events.groupby("game_episode")["event_idx"].transform(lambda s: s.max() - s)
lastK = events[events["rev_idx"] < K].copy()

lastK = lastK.sort_values(["game_episode", "event_idx"]).reset_index(drop=True)
lastK["k_len"] = lastK.groupby("game_episode")["event_idx"].transform("size").astype(int)
lastK["k_rank"] = lastK.groupby("game_episode").cumcount().astype(int)
lastK["pos_in_K"] = (K - lastK["k_len"] + lastK["k_rank"]).astype(int)
lastK = lastK.drop(columns=["k_len", "k_rank"])

# ----------------------
# 9. wide feature pivot
# ----------------------
num_cols = [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "dx",
    "dy",
    "dist",
    "speed",
    "dt",
    "ep_idx_norm",
    "x_zone",
    "lane",
    "is_final_team",
]

cat_cols = [
    "type_id",
    "res_id",
    "team_id_enc",
    "is_home",
    "period_id",
    "is_last",
]

feature_cols = num_cols + cat_cols
wide = lastK[["game_episode", "pos_in_K"] + feature_cols].copy()

# dtype 축소
for c in num_cols:
    if c in wide.columns:
        wide[c] = wide[c].astype("float32")
for c in cat_cols:
    if c in wide.columns:
        wide[c] = wide[c].astype("int16", errors="ignore")

wide_num = wide.pivot(index="game_episode", columns="pos_in_K", values=num_cols)
wide_cat = wide.pivot(index="game_episode", columns="pos_in_K", values=cat_cols)

wide_num.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_num.columns]
wide_cat.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_cat.columns]

X = pd.concat([wide_num, wide_cat], axis=1).reset_index()

X = X.merge(
    ep_meta[["game_episode", "game_id", "game_clock_min", "final_team_id", "is_home", "period_id"]],
    on="game_episode",
    how="left",
)
X = X.merge(labels, on="game_episode", how="left")

train_mask = X["game_episode"].isin(labels["game_episode"])
X_train = X[train_mask].copy()

y_train_x = X_train["target_x"].astype(float)
y_train_y = X_train["target_y"].astype(float)

groups = X_train["game_id"].values

drop_cols = ["game_episode", "game_id", "target_x", "target_y"]
X_train_feat = X_train.drop(columns=drop_cols).fillna(0)

# ----------------------
# 10. LightGBM 학습 + weight 저장
# ----------------------
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

gkf = GroupKFold(n_splits=N_SPLITS)

run_dir = get_next_run_dir(MODEL_DIR)

# 간단한 메타데이터 저장(재현성/관리 목적)
meta = {
    "K": K,
    "N_SPLITS": N_SPLITS,
    "params": params,
}
pd.Series(meta, dtype="object").to_json(os.path.join(run_dir, "meta.json"), force_ascii=False)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_feat, y_train_x, groups)):
    print(f"Fold {fold}")

    X_tr, X_val = X_train_feat.iloc[tr_idx], X_train_feat.iloc[val_idx]
    y_tr_x, y_val_x = y_train_x.iloc[tr_idx], y_train_x.iloc[val_idx]
    y_tr_y, y_val_y = y_train_y.iloc[tr_idx], y_train_y.iloc[val_idx]

    dtrain_x = lgb.Dataset(X_tr, label=y_tr_x)
    dvalid_x = lgb.Dataset(X_val, label=y_val_x, reference=dtrain_x)

    model_x = lgb.train(
        params,
        dtrain_x,
        num_boost_round=3000,
        valid_sets=[dtrain_x, dvalid_x],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )
    model_x.save_model(os.path.join(run_dir, f"lgbm_x_fold{fold}.txt"))

    dtrain_y = lgb.Dataset(X_tr, label=y_tr_y)
    dvalid_y = lgb.Dataset(X_val, label=y_val_y, reference=dtrain_y)

    model_y = lgb.train(
        params,
        dtrain_y,
        num_boost_round=3000,
        valid_sets=[dtrain_y, dvalid_y],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )
    model_y.save_model(os.path.join(run_dir, f"lgbm_y_fold{fold}.txt"))

print(f"Saved models to: {run_dir}")
