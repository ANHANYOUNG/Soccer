# Transformer Pass Prediction

## Overview
LSTM_2.ipynb를 기반으로 Transformer 아키텍처로 변환한 패스 도착점 예측 모델입니다.

## Files
- `train.py`: 학습 스크립트
- `inference.py`: 추론 스크립트
- `models/`: 학습된 모델 및 설정 파일 저장 디렉토리

## Model Architecture
- **Input Embedding**: 연속형 특성(12차원) + type_name 임베딩 + result_name 임베딩
- **Positional Encoding**: Sinusoidal positional encoding
- **Transformer Encoder**: 4 layers, 8 heads, d_model=128, ff_dim=512
- **Attention Pooling**: 시퀀스를 단일 벡터로 집계
- **Output Head**: MLP로 (x, y) 좌표 예측

## Hyperparameters
```python
D_MODEL = 128          # Transformer hidden dimension
N_HEADS = 8            # Number of attention heads
N_LAYERS = 4           # Number of transformer encoder layers
DIM_FEEDFORWARD = 512  # Feed-forward network dimension
DROPOUT = 0.3          # Dropout rate
EMB_DIM = 16           # Embedding dimension for categorical features
```

## Usage

### Training
```bash
cd Transformer
python train.py
```

학습 후 `models/` 디렉토리에 다음 파일들이 저장됩니다:
- `best_model.pt`: 학습된 모델 가중치
- `model_config.pkl`: 모델 설정
- `label_encoders.pkl`: LabelEncoder 객체들

### Inference
```bash
cd Transformer
python inference.py
```

추론 결과는 `Transformer_submit_X.csv` 파일로 저장됩니다.

## Features (12 continuous features)
1. `sx` (start_x): 시작 x 좌표
2. `sy` (start_y): 시작 y 좌표
3. `ex_mask` (end_x masked): 종료 x 좌표 (마지막 이벤트는 0으로 마스킹)
4. `ey_mask` (end_y masked): 종료 y 좌표 (마지막 이벤트는 0으로 마스킹)
5. `dt`: 이전 이벤트와의 시간 차이
6. `dist_to_goal`: 골 세그먼트까지의 거리
7. `theta_view`: 골 뷰 각도
8. `in_own_half`: 자기 진영 여부
9. `dist_p_box`: 페널티 박스까지의 거리
10. `prev_dx`: 이전 이벤트의 x 방향 이동
11. `prev_dy`: 이전 이벤트의 y 방향 이동
12. `prev_valid`: 이전 이벤트 정보 유효성 플래그

## Differences from LSTM
| Aspect | LSTM | Transformer |
|--------|------|-------------|
| Sequential Processing | Recurrent (순차적) | Parallel (병렬) |
| Long-range Dependencies | Gradient issues | Self-attention (더 효과적) |
| Positional Info | Implicit in state | Explicit positional encoding |
| Architecture | LSTM + Multi-head Attention Pool | Transformer Encoder + Attention Pool |

## Requirements
- torch
- pandas
- numpy
- scikit-learn
- tqdm


submit_1: Last Token 적용 최고 성능
submit_2,3: valid euclid dist 가장 낮은데 리더보드에서는 높음 -> fold에 과적합


## Debug 2025.12.30
```
1. single batch로 loss 잘 줄어드는지 확인

python3 train.py --mode overfit --overfit_epochs 500

1) dropout=0.1, data augment True -> overfitting
2) dropout=0.1, data augment True, weight decay=1e-4 -> overfitting
3) dropout=0.2, data augment True, weight decay=1e-4 -> training 을 못 함
4) D_MODEL = 512, N_HEADS = 16, N_LAYERS = 6, DIM_FEEDFORWARD = 1024 -> single batch loss 11~12 -> training 을 못 함
5) weight decay = 1e-5, std = 0.1
6) 제일 잘 나왔을 때로 회귀 
        BATCH_SIZE = 256
        LR = 1e-3
        WEIGHT_DECAY = 1e-5
        D_MODEL = 512    
        N_HEADS = 8          
        N_LAYERS = 4       
        DIM_FEEDFORWARD = 512 
        DROPOUT = 0.2         
        USE_AUGMENT = False  
        NOISE_STD = 0.5
        K_TRUNCATE = 50     
7) 원래대로 회귀, 학습 진행
        D_MODEL: 128 -> Average: 13.8467
        D_MODEL: 256 -> Average: 13.7814
```

## Debug 2025.12.30
```