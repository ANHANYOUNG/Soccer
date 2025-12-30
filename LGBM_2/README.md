# LightGBM Pass Prediction Model (LGBM_2)

Episode-level LightGBM 기반 패스 도착 좌표 예측 모델

## 주요 개선점 (기존 LGBM_1 대비)

1. **Episode-level Aggregation**: 이벤트 단위가 아닌 에피소드 단위로 특성 집계
2. **Delta Prediction**: 절대 좌표 대신 (dx, dy) 상대 좌표 예측
3. **GroupKFold**: game_id 기준 분할로 누수 방지
4. **Leak-safe Features**: 마지막 이벤트 end 좌표 제외한 집계

## 파일 구조

```
LGBM_2/
├── train.py          # 학습 스크립트
├── inference.py      # 추론 스크립트
├── README.md         # 설명서
└── models/           # 모델 저장 디렉토리
    ├── lgbm_fold0.pkl
    └── encoders.pkl
```

## 특성 목록 (28개)

### Last Event Features (예측 대상 이벤트)
- `last_start_x`, `last_start_y`: 마지막 이벤트 시작 위치
- `last_dist_to_goal`: 골문까지 거리
- `last_dist_from_center`: 중앙으로부터 거리
- `last_angle`: 원점 기준 각도
- `last_goal_view`: 골문 시야각
- `last_time`: 정규화된 시간

### Episode Statistics (에피소드 집계)
- `ep_len`: 에피소드 길이
- `ep_mean_x`, `ep_mean_y`: 평균 위치
- `ep_std_x`, `ep_std_y`: 위치 표준편차
- `ep_mean_dist_goal`, `ep_mean_dist_center`: 평균 거리
- `ep_total_dist`: 총 이동 거리
- `ep_mean_dx`, `ep_mean_dy`: 평균 이동 방향
- `ep_progression_x`, `ep_progression_y`: 전체 진행 방향

### Categorical Features
- `last_team_id`, `last_player_id`, `last_type`: 인코딩된 범주형

### Count Features
- `n_passes`, `n_carries`: 이벤트 유형 개수

### Zone Features
- `in_final_third`, `in_penalty_area`: 구역 플래그

## 사용법

### 학습
```bash
cd LGBM_2
python train.py
```

### 추론
```bash
python inference.py
```

## 하이퍼파라미터

```python
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
}
```

## 기존 코드 문제점 수정

| 문제 | 수정 |
|------|------|
| train/test 구조 다름 (이벤트 vs 에피소드) | 에피소드 단위 집계 |
| 좌표계 불일치 (50/100 vs 52.5/105) | 105x68 기준 통일 |
| 랜덤 분할 (누수 가능) | GroupKFold by game_id |
| RMSE 평가 (지표 불일치) | Euclidean distance |
| player_stats 누수 | 사용 안함 |
