import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# ----------------------
# 0. 설정
# ----------------------
PATH_TRAIN = "train.csv"
PATH_SAMPLE_SUB = "sample_submission.csv"

TEST_DIR = "test"  # test 폴더 아래의 모든 CSV를 재귀적으로 로드

# 학습에서 저장한 weight(Booster) 파일들
MODEL_DIR = "models"

# 특정 run을 지정하고 싶으면 환경변수 LGBM_RUN_DIR을 설정
# 예: LGBM_RUN_DIR=models/run_003
RUN_DIR_ENV = os.environ.get("LGBM_RUN_DIR", "").strip()


def get_latest_run_dir(base_dir: str) -> str:
    runs = []
    if not os.path.isdir(base_dir):
        return ""
    for name in os.listdir(base_dir):
        if not name.startswith("run_"):
            continue
        try:
            n = int(name.replace("run_", ""))
        except ValueError:
            continue
        runs.append((n, os.path.join(base_dir, name)))
    if len(runs) == 0:
        return ""
    runs.sort(key=lambda x: x[0])
    return runs[-1][1]


RUN_DIR = RUN_DIR_ENV if RUN_DIR_ENV else get_latest_run_dir(MODEL_DIR)
if RUN_DIR == "":
    raise FileNotFoundError(
        "No run directory found under 'models/'. Run train.py first to create models/run_001, run_002, ..."
    )

MODEL_X_GLOB = os.path.join(RUN_DIR, "lgbm_x_fold*.txt")
MODEL_Y_GLOB = os.path.join(RUN_DIR, "lgbm_y_fold*.txt")

K = 20   # 마지막 K 이벤트 사용 (20~32 사이 선택)

# ----------------------
# 1. 데이터 로드
# ----------------------
train = pd.read_csv(PATH_TRAIN)
sample_sub = pd.read_csv(PATH_SAMPLE_SUB)

# test/ 폴더 내 모든 episode CSV 로드 (test_index.csv 없이 동작)
test_csv_paths = []
for root, _, files in os.walk(TEST_DIR):
    for fn in files:
        if fn.lower().endswith(".csv"):
            test_csv_paths.append(os.path.join(root, fn))

test_csv_paths = sorted(test_csv_paths)
if len(test_csv_paths) == 0:
    raise FileNotFoundError(f"No test csv files found under: {TEST_DIR}")

test_events_list = [pd.read_csv(p) for p in test_csv_paths]
test_events = pd.concat(test_events_list, ignore_index=True)

train["is_train"] = 1
test_events["is_train"] = 0

events = pd.concat([train, test_events], ignore_index=True)

# ----------------------
# 2. 기본 정렬 + episode 내 인덱스
# ----------------------
events = events.sort_values(["game_episode", "time_seconds", "action_id"]).reset_index(drop=True)

events["event_idx"] = events.groupby("game_episode").cumcount()
events["n_events"] = events.groupby("game_episode")["event_idx"].transform("max") + 1
events["ep_idx_norm"] = events["event_idx"] / (events["n_events"] - 1).clip(lower=1)

# ----------------------
# 3. 시간/공간 feature
# ----------------------
# Δt
events["prev_time"] = events.groupby("game_episode")["time_seconds"].shift(1)
events["dt"] = events["time_seconds"] - events["prev_time"]
events["dt"] = events["dt"].fillna(0.0)

# 이동량/거리
events["dx"] = events["end_x"] - events["start_x"]
events["dy"] = events["end_y"] - events["start_y"]
events["dist"] = np.sqrt(events["dx"]**2 + events["dy"]**2)

# 속도 (dt=0 보호)
events["speed"] = events["dist"] / events["dt"].replace(0, 1e-3)

# zone / lane (필요시 범위 조정)
events["x_zone"] = (events["start_x"] / (105/7)).astype(int).clip(0, 6)
events["lane"] = pd.cut(
    events["start_y"],
    bins=[0, 68/3, 2*68/3, 68],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

# ----------------------
# 4. 라벨 및 episode-level 메타 (train 전용)
# ----------------------
train_events = events[events["is_train"] == 1].copy()

last_events = (
    train_events
    .groupby("game_episode", as_index=False)
    .tail(1)
    .copy()
)

labels = last_events[["game_episode", "end_x", "end_y"]].rename(
    columns={"end_x": "target_x", "end_y": "target_y"}
)

# episode-level 메타 (마지막 이벤트 기준)
ep_meta = last_events[["game_episode", "game_id", "team_id", "is_home", "period_id", "time_seconds"]].copy()
ep_meta = ep_meta.rename(columns={"team_id": "final_team_id"})

# game_clock (분 단위, 0~90+)
ep_meta["game_clock_min"] = np.where(
    ep_meta["period_id"] == 1,
    ep_meta["time_seconds"] / 60.0,
    45.0 + ep_meta["time_seconds"] / 60.0
)

# ----------------------
# 5. 공격 팀 플래그 (final_team vs 상대)
# ----------------------
# final_team_id를 전체 events에 붙임
events = events.merge(
    ep_meta[["game_episode", "final_team_id"]],
    on="game_episode",
    how="left"
)

events["is_final_team"] = (events["team_id"] == events["final_team_id"]).astype(int)

# ----------------------
# 6. 입력용 events에서 마지막 이벤트 타깃 정보 가리기
# ----------------------
# is_last 플래그
events["last_idx"] = events.groupby("game_episode")["event_idx"].transform("max")
events["is_last"] = (events["event_idx"] == events["last_idx"]).astype(int)

# labels는 이미 뽑아놨으니, 입력쪽에서만 end_x, end_y, dx, dy, dist, speed 지움
mask_last = events["is_last"] == 1
for col in ["end_x", "end_y", "dx", "dy", "dist", "speed"]:
    events.loc[mask_last, col] = np.nan

# ----------------------
# 7. 카테고리 인코딩 (type_name, result_name, team_id 등)
# ----------------------
events["type_name"] = events["type_name"].fillna("__NA_TYPE__")
events["result_name"] = events["result_name"].fillna("__NA_RES__")

# 인코더는 train 데이터만으로 fit (test 사전 활용 해석 소지 최소화)
train_only = events[events["is_train"] == 1].copy()

le_type = LabelEncoder()
le_res = LabelEncoder()
le_type.fit(train_only["type_name"].astype(str))
le_res.fit(train_only["result_name"].astype(str))

type_map = {cls: i for i, cls in enumerate(le_type.classes_)}
res_map = {cls: i for i, cls in enumerate(le_res.classes_)}

# unknown은 -1로 처리
events["type_id"] = events["type_name"].astype(str).map(type_map).fillna(-1).astype(int)
events["res_id"] = events["result_name"].astype(str).map(res_map).fillna(-1).astype(int)

# team_id: train에서만 매핑을 만들고 test의 unknown은 -1
if train_only["team_id"].dtype == "object":
    team_map = {cls: i for i, cls in enumerate(pd.Series(train_only["team_id"].astype(str).unique()))}
    events["team_id_enc"] = events["team_id"].astype(str).map(team_map).fillna(-1).astype(int)
else:
    events["team_id_enc"] = events["team_id"].astype(int)

# ----------------------
# 8. 마지막 K 이벤트만 사용 (lastK)
# ----------------------
# rev_idx: 0이 마지막 이벤트
events["rev_idx"] = events.groupby("game_episode")["event_idx"].transform(
    lambda s: s.max() - s
)

lastK = events[events["rev_idx"] < K].copy()

# pos_in_K: 0~(K-1), 앞쪽 패딩 고려해서 뒤에 실제 이벤트가 모이게 (vectorized)
lastK = lastK.sort_values(["game_episode", "event_idx"]).reset_index(drop=True)
lastK["k_len"] = lastK.groupby("game_episode")["event_idx"].transform("size").astype(int)
lastK["k_rank"] = lastK.groupby("game_episode").cumcount().astype(int)
lastK["pos_in_K"] = (K - lastK["k_len"] + lastK["k_rank"]).astype(int)
lastK = lastK.drop(columns=["k_len", "k_rank"])

# ----------------------
# 9. wide feature pivot
# ----------------------
# 사용할 이벤트 피처 선택
num_cols = [
    "start_x", "start_y",
    "end_x", "end_y",
    "dx", "dy", "dist", "speed",
    "dt",
    "ep_idx_norm",
    "x_zone", "lane",
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

# 메모리 절약을 위해 dtype 축소 (CPU-only 환경에서 pivot/concat 부담 완화)
for c in num_cols:
    if c in wide.columns:
        wide[c] = wide[c].astype("float32")
for c in cat_cols:
    if c in wide.columns:
        wide[c] = wide[c].astype("int16", errors="ignore")

# 숫자형 pivot (중복이 없다는 가정에서 pivot 사용)
wide_num = wide.pivot(index="game_episode", columns="pos_in_K", values=num_cols)

# 범주형 pivot
wide_cat = wide.pivot(index="game_episode", columns="pos_in_K", values=cat_cols)

# 컬럼 이름 평탄화
wide_num.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_num.columns]
wide_cat.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_cat.columns]

X = pd.concat([wide_num, wide_cat], axis=1).reset_index()  # game_episode 포함

# episode-level 메타 붙이기
X = X.merge(ep_meta[["game_episode", "game_id", "game_clock_min", "final_team_id", "is_home", "period_id"]],
            on="game_episode", how="left")

# train 라벨 붙이기
X = X.merge(labels, on="game_episode", how="left")  # test는 NaN

# ----------------------
# 10. train/test 분리
# ----------------------
train_mask = X["game_episode"].isin(labels["game_episode"])
X_train = X[train_mask].copy()
X_test = X[~train_mask].copy()

y_train_x = X_train["target_x"].astype(float)
y_train_y = X_train["target_y"].astype(float)

# group용 game_id
groups = X_train["game_id"].values

# 모델 입력에서 빼야 할 컬럼들
drop_cols = [
    "game_episode",
    "game_id",
    "target_x",
    "target_y",
]

X_train_feat = X_train.drop(columns=drop_cols)
X_test_feat = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# NaN 채우기 (LGBM은 NaN 다루긴 하지만, 깔끔하게)
X_train_feat = X_train_feat.fillna(0)
X_test_feat = X_test_feat.fillna(0)

# ----------------------
# 11. LightGBM 모델 로드 (학습은 train.py에서 수행)
# ----------------------
model_x_paths = sorted(glob.glob(MODEL_X_GLOB))
model_y_paths = sorted(glob.glob(MODEL_Y_GLOB))

if len(model_x_paths) == 0 or len(model_y_paths) == 0:
    raise FileNotFoundError(
        "Model weight files not found in the selected run directory. "
        f"Selected RUN_DIR: {RUN_DIR}. Expected patterns: {MODEL_X_GLOB}, {MODEL_Y_GLOB}"
    )
if len(model_x_paths) != len(model_y_paths):
    raise ValueError(
        f"Mismatched number of x/y models: {len(model_x_paths)} vs {len(model_y_paths)}"
    )

models_x = [lgb.Booster(model_file=p) for p in model_x_paths]
models_y = [lgb.Booster(model_file=p) for p in model_y_paths]

# ----------------------
# 12. test 예측 + 앙상블
# ----------------------
pred_x_folds = []
pred_y_folds = []

for model_x, model_y in zip(models_x, models_y):
    pred_x_folds.append(model_x.predict(X_test_feat, num_iteration=model_x.best_iteration))
    pred_y_folds.append(model_y.predict(X_test_feat, num_iteration=model_y.best_iteration))

pred_x = np.mean(pred_x_folds, axis=0)
pred_y = np.mean(pred_y_folds, axis=0)

# 필드 범위로 클립
pred_x = np.clip(pred_x, 0, 105)
pred_y = np.clip(pred_y, 0, 68)

# ----------------------
# 13. submission 생성
# ----------------------
sub = sample_sub.copy()

# X_test에는 game_episode가 있으니, test_index와 align
pred_df = X_test[["game_episode"]].copy()
pred_df["end_x"] = pred_x
pred_df["end_y"] = pred_y

sub = sub.drop(columns=["end_x", "end_y"], errors="ignore")
sub = sub.merge(pred_df, on="game_episode", how="left")

# 기존 제출 파일을 덮어쓰지 않도록 번호를 증가시키며 저장
existing = glob.glob("submission_lgbm_lastK_*.csv")
nums = []
for p in existing:
    base = os.path.basename(p)
    try:
        n_str = base.replace("submission_lgbm_lastK_", "").replace(".csv", "")
        nums.append(int(n_str))
    except ValueError:
        pass

next_n = (max(nums) + 1) if len(nums) > 0 else 0
out_name = f"submission_lgbm_lastK_{next_n}.csv"
sub.to_csv(out_name, index=False)
print(f"Saved {out_name}")
