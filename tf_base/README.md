# Transformer Baseline

## DL_DEBUG.md 기반 디버깅 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Get Your Model to Run                              │
│  - Shape mismatch 확인                                       │
│  - Casting issue 확인                                        │
│  - Forward/Backward pass 성공 여부                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Overfit A Single Batch                             │
│  - Loss가 0으로 수렴하는지 확인                                │
│  - 안 되면: ERROR GOES UP / EXPLODES / OSCILLATES / PLATEAUS │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Evaluate (Full Training)                           │
│  - Bias = train_error                                       │
│  - Variance = valid_error - train_error                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Improve                                            │
│  - HIGH BIAS → 모델 키우기, 피처 추가                          │
│  - HIGH VARIANCE → regularization, 데이터 늘리기               │
└─────────────────────────────────────────────────────────────┘
```

---

## 사용법

### Step 1: 모델이 돌아가는지 확인
```bash
python train.py --mode check
```
**출력:** Shape 체크, Forward/Backward pass, NaN/Inf 검사

### Step 2: Single batch 오버피팅 테스트
```bash
python train.py --mode overfit
```
**기대 결과:** Loss가 0에 가까워져야 함 (모델이 학습 가능함을 증명)

### Step 3: Full training
```bash
python train.py --mode train --fold 0
```
**출력:** Bias-Variance 분석, Best valid_dist

---

## 디버깅용 설정 (NO REGULARIZATION)

| 설정 | 값 | 이유 |
|-----|-----|------|
| `dropout` | 0.0 | 순수 모델 성능 측정 |
| `weight_decay` | 0.0 | 순수 모델 성능 측정 |
| `normalization` | 없음 | Raw 데이터로 시작 |
| `augmentation` | 없음 | 순수 데이터로 시작 |

---

## 입력 피처 (최소 세트)

### 연속형 (5개) - Raw 값, 정규화 없음
| 이름 | 범위 | 설명 |
|------|------|------|
| `start_x` | 0~105 | 이벤트 시작 x 좌표 |
| `start_y` | 0~68 | 이벤트 시작 y 좌표 |
| `end_x_mask` | 0~105 | 마지막 이벤트는 0으로 마스킹 |
| `end_y_mask` | 0~68 | 마지막 이벤트는 0으로 마스킹 |
| `dt` | 0~ | 이전 이벤트와의 시간 차이 (초) |

### 범주형 (2개)
| 이름 | vocab 구조 |
|------|-----------|
| `type_name` | 0=PAD, 1~=실제값 |
| `result_name` | 0=PAD, 1~=실제값 |

**⚠️ 중요:** vocab은 1부터 시작 (0은 PAD 전용)

---

## 모델 구조 (최소, 디버깅용)

```
Input: cont(5) + type_emb(16) + res_emb(16) = 37
        ↓
    Linear(37 → 64)
        ↓
    Positional Encoding
        ↓
    TransformerEncoder (2 layers, Post-LN, ReLU)
        ↓
    Last-token Pooling
        ↓
    Linear(64) → ReLU → Linear(2)
        ↓
    Output: (end_x, end_y)
```

### 하이퍼파라미터 (디버깅용, 최소)
| 파라미터 | 값 |
|---------|-----|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `dim_ff` | 256 |
| `dropout` | 0.0 |
| `emb_dim` | 16 |
| `lr` | 3e-4 |
| `weight_decay` | 0.0 |

---

## Overfit 테스트 진단

| 증상 | 원인 | 해결 |
|------|------|------|
| Error GOES UP | Loss 부호 반대, 레이블 문제 | Loss 함수 확인 |
| Error EXPLODES | NaN/Inf, LR 너무 높음 | LR 낮추기, gradient clipping |
| Error OSCILLATES | LR 너무 높음, 데이터 문제 | LR 낮추기, 데이터 확인 |
| Error PLATEAUS | LR 너무 낮음, 모델 작음 | LR 높이기, 모델 키우기 |

---

## 파일 구조

```
tf_base/
├── train.py      # 학습 스크립트 (3 modes: check, overfit, train)
├── inference.py  # 추론 스크립트
├── README.md     # 이 파일
└── models/
    ├── vocab.pkl
    └── seed_*_fold_*/
        ├── best_model.pt
        └── config.pkl
```

## 디버깅 로그
```
베이스라인 완료
mode check, overfit으로 일단 확인 후 train에서 더 변경
TF라서 BN 했을 때는 성능 하락
D_model -> 128, N_LAYERS -> 6 (model size 키우기)