# Transformer 5-Fold Ensemble Training Plan

## 전략 요약
- **K = 32** (OOF 분석 결과 기반)
- **5-fold cross-validation** (GroupKFold by game_id)
- **3 seeds** for diversity: [42, 123, 456]
- **Total models: 3 seeds × 5 folds = 15 models**

## 계산량 분석

### 학습 시간 예상
- 1개 모델 (150 epochs): ~30-40분 (GPU 기준)
- **총 15개 모델**: ~7.5-10시간

### GPU 메모리
- 모델 크기: ~5MB per model (매우 작음)
- Batch size 256: ~2-3GB VRAM
- **4 GPU 병렬 가능**: 4개 fold를 동시 학습 → 약 2-3시간으로 단축

### 추론 시간
- 15개 모델 앙상블 예측: ~5-10분
- 메모리: 15 models × 5MB = ~75MB (매우 적음)

## 학습 방법

### Option 1: 순차 학습 (단일 GPU)
```bash
cd /home/ahy0502/soccer/open_track1/Transformer

# train.py에서 FOLD = None으로 설정 (이미 설정됨)
python train.py
```
→ 모든 15개 모델을 순차적으로 학습 (~7-10시간)

### Option 2: 병렬 학습 (4 GPU) ✅ 추천
각 fold를 다른 GPU에서 동시에 학습:

```bash
# 터미널 1: Fold 0 (모든 seed)
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 &

# 터미널 2: Fold 1 (모든 seed)
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 &

# 터미널 3: Fold 2 (모든 seed)
CUDA_VISIBLE_DEVICES=2 python train.py --fold 2 &

# 터미널 4: Fold 3,4 (순차)
CUDA_VISIBLE_DEVICES=3 python train.py --fold 3
CUDA_VISIBLE_DEVICES=3 python train.py --fold 4
```

→ ~2-3시간으로 단축!

**필요한 수정**: `train.py`에 `--fold` 인자 추가 필요

## 추론 방법

```bash
cd /home/ahy0502/soccer/open_track1/Transformer
python inference.py
```

자동으로 `models/seed_*_fold_*` 디렉토리의 모든 모델(15개)을 로드하여 앙상블 예측:
1. 각 모델이 독립적으로 예측
2. 15개 예측의 평균을 최종 제출 파일로 생성
3. `Transformer_submit_X.csv` 저장

## 디렉토리 구조

```
Transformer/
├── train.py
├── inference.py
├── oof_analysis.py
└── models/
    ├── label_encoders.pkl (공통)
    ├── seed_42_fold_0/
    │   ├── best_model.pt
    │   └── model_config.pkl
    ├── seed_42_fold_1/
    ├── seed_42_fold_2/
    ├── seed_42_fold_3/
    ├── seed_42_fold_4/
    ├── seed_123_fold_0/
    ├── ...
    └── seed_456_fold_4/
```

## 추가 고려사항

### ✅ 이미 처리된 사항
1. **Data Leakage 방지**: GroupKFold로 game_id 기준 분리
2. **Delta Prediction**: 마지막 start_x, start_y 기준 상대좌표 예측
3. **Last Token Pooling**: 마지막 유효 토큰만 사용 (압도적 성능 향상)
4. **Gaussian NLL Loss**: 불확실성 학습
5. **K Truncation**: 최근 32개 이벤트에 집중

### ⚠️ 추가 검토 가능 사항
1. **Learning Rate Schedule**: ReduceLROnPlateau 사용 중 (적절함)
2. **Early Stopping**: 현재 없음 → 150 epoch 고정
   - Valid dist가 개선되지 않으면 조기 종료 추가 가능
   - 하지만 150 epoch도 충분히 빠름 (~30분)
3. **Test Time Augmentation (TTA)**: 현재 없음
   - 추론 시 노이즈 추가 예측 → 더 robust
   - 시간 2배 증가 but 성능 소폭 향상 가능
4. **Weighted Ensemble**: 현재 단순 평균
   - Valid 성능 기준 가중 평균 가능
   - 하지만 단순 평균도 충분히 효과적

### 🎯 현재 전략의 장점
1. **Stability**: 5-fold로 전체 데이터 활용
2. **Diversity**: 3 seeds로 초기화 다양성
3. **Efficiency**: K=32로 계산량 감소
4. **Robust**: 15 models 앙상블로 과적합 방지

## 예상 성능
- **OOF validation**: ~13.7-13.9 (K=32 기준)
- **Leaderboard**: OOF와 유사하거나 약간 개선 (앙상블 효과)
- **개선 폭**: 단일 모델 대비 ~0.1-0.2 거리 감소 예상

## 다음 단계
1. ✅ `train.py` 수정 완료 (5-fold 지원)
2. ✅ `inference.py` 수정 완료 (15 models 앙상블)
3. ⬜ `--fold` 인자 추가 (병렬 학습용) - 선택사항
4. ⬜ 학습 실행
5. ⬜ 추론 및 제출
