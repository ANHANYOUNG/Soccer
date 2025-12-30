Windows + PowerShell 기준 (프로젝트 폴더에서)

1) (권한 오류 방지: 현재 PowerShell 창에서만)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

2) venv 활성화
.\.venv\Scripts\Activate.ps1

3) pip 업데이트
python -m pip install --upgrade pip

4) 필수 패키지 설치
pip install numpy pandas scikit-learn tqdm
pip install torch

5) 설치 확인
python -c "import numpy,pandas,sklearn,tqdm,torch; print('ok')"



LSTM_1: 


2. LSTM_2_submit_0
축구장 영역 확실히 구분할 영역들은 구분 
ex) 하프라인, 패널티 박스, 골대를 점이 아닌 선분으로

    3. 데이터 로드 및 전처리

        a 로드 및 정렬
            train.csv를 읽고, 같은 game_episode 안에서 time_seconds, action_id 기준으로 정렬해 이벤트 시퀀스 순서를 확정

        b 범주형 결측 처리 및 인코딩
            type_name/result_name의 NaN을 특수 문자열로 채운 뒤, LabelEncoder로 type_id/res_id(정수 인덱스)로 변환(이 숫자는 크기 의미가 없고 “카테고리 인덱스”로서 이후 임베딩 입력이 됨)

        c episode 단위로 그룹화하여 학습 샘플 생성
            df를 game_episode로 groupby 해서 각 episode를 하나의 학습 샘플로 만들고, 길이가 너무 짧은 episode는 제외

        d 타깃 정의 및 누수 방지 정리
            타깃은 “마지막 Pass 이벤트의 end_x, end_y”로 정의
            대부분 마지막 이벤트가 Pass이지만 예외를 위해 마지막이 Pass가 아니면 마지막 Pass까지만 잘라 누수/정합성 문제를 제거
            입력 시퀀스에서 마지막 이벤트의 end_x/end_y는 정답이므로 ex_mask/ey_mask로 0 마스킹하고, is_end 플래그로 위치 표시

        e 피처(cont) 구성
            1. 범주형 시퀀스 (각 이벤트마다 1개씩)
                type_id: type_name을 LabelEncoder로 정수 인덱스로 변환한 값
                res_id: result_name을 LabelEncoder로 정수 인덱스로 변환한 값

            2. 연속형 시퀀스 feature cont (각 이벤트마다 14개, 순서 그대로) 기본/누수처리
                1. sx = start_x
                2. sy = start_y
                3. ex_mask = end_x (마지막 이벤트는 0으로 마스킹)
                4. ey_mask = end_y (마지막 이벤트는 0으로 마스킹)
                5. dt = 이전 이벤트와의 time_seconds 차이(첫 이벤트는 0)
                6. is_end = 마지막 이벤트에서 0, 나머지 1 (정답 위치 마스크 플래그)

                경기장에서의 고정 영역
                골문(선분 + 각도)
                7. dist_goal = (sx,sy)에서 골문 선분(x=105, y∈[30.34,37.66])까지 최소거리
                8. theta_mid_sin = theta_mid의 sin (theta_mid는 포스트 각도의 평균)
                9. theta_mid_cos = theta_mid의 cos
                10. theta_view = |alpha_R - alpha_L| (포스트 두 각의 차이, 골문 시야각)

                하프라인/박스
                11. d_half = |sx - 52.5| (하프라인까지 거리)
                12. in_own_half = sx < 52.5 이면 1 else 0 (자기 진영 플래그)
                13. in_p_box = 상대 페널티박스 내부 여부(1/0)
                14. dist_p_box = (sx,sy)에서 상대 페널티박스 직사각형까지 최소거리

        결과적으로 cont는 (T, 14) 형태의 시퀀스 (T는 episode 길이)

        f 최종 저장 구조
            각 episode에 대해
            episodes: {cont (T,14), type_id (T,), res_id (T,)}를 저장
            targets: (tx,ty) 1개를 저장
            episode_keys/game_id도 함께 저장해 이후 GroupKFold 등에서 사용
            이 셀의 산출물은 “episode별 시퀀스 입력 + 1개 좌표 타깃” 형태의 학습 데이터셋이며, 
            K=50 패딩/자르기는 다음 단계(DataLoader/모델 입력 단계)에서 적용

    4. Custom Dataset / DataLoader 정의 및 Validation 분할

        a EpisodeDataset 정의
            episodes/targets/keys를 묶어 “episode 단위 샘플”을 표준 형태로 반환하는 Dataset

            __len__은 전체 episode 개수를 반환
            getitem(idx)는 idx번째 episode를 꺼내 다음을 반환

            cont: (T, 14) 연속형 시퀀스 텐서
            type_id: (T,) 범주형(type) 인덱스 텐서
            res_id: (T,) 범주형(result) 인덱스 텐서
            y: (2,) 정답 좌표 텐서 (tx, ty)
            key: game_episode 식별자 문자열

        b collate_fn 정의(가변 길이 시퀀스 배치화)
            DataLoader가 샘플 여러 개를 한 배치로 묶을 때, episode 길이 T가 제각각이라 padding이 필요

            collate_fn은 batch(샘플 튜플들의 리스트)를 받아 다음을 생성
            lengths: 각 샘플의 원래 길이 T를 저장한 (B,) 텐서
            cont_pad: (B, T_max, 14)로 padding된 연속형 입력(부족한 구간은 0.0)
            type_pad: (B, T_max)로 padding된 type 인덱스(부족한 구간은 0)
            res_pad: (B, T_max)로 padding된 result 인덱스(부족한 구간은 0)
            y: (B, 2)로 stack된 정답 좌표(정답은 고정 길이라 padding 없음)
            keys: 배치에 포함된 episode 식별자 목록

            lengths는 이후 pack_padded_sequence 등에서 padding 구간을 무시하는데 사용

        c GroupKFold로 validation 분할(game_id 기준)
            episode_game_ids(episode가 속한 game_id)를 groups로 사용해 GroupKFold를 적용

            목적은 같은 경기(game_id)의 episode가 train과 valid에 동시에 들어가는 누수/과대평가를 방지

            N_SPLITS로 fold 수를 정하고, FOLD 값을 이용해 원하는 fold의 (train_idx, valid_idx)를 선택

            선택된 인덱스로 train_eps/train_tg/train_keys, valid_eps/valid_tg/valid_keys를 각각 구성

        d DataLoader 생성 및 shape sanity check
            train_ds/valid_ds를 만들고,
            train_loader는 shuffle=True(학습 일반화 목적)
            valid_loader는 shuffle=False(평가 재현성)

            배치 하나를 실제로 꺼내 cont_pad/type_pad/res_pad/lengths/y의 shape가 기대대로 나오는지 확인

            cont_pad: (B, T_max, 14)
            type_pad/res_pad: (B, T_max)
            lengths: (B,)
            y: (B, 2)
            keys: 식별자 리스트
    
    5. LSTM 베이스라인 모델 정의


        a 모델 목적과 입력/출력 형태
            episode 단위의 가변 길이 이벤트 시퀀스 입력을 받아, 각 episode의 마지막 패스 도착 좌표 (end_x, end_y)를 (B,2) 회귀로 예측

            입력은 DataLoader가 만든 배치 단위 텐서로 들어오며,
            cont_pad: (B, T_max, cont_dim) 연속형 시퀀스
            type_pad: (B, T_max) 이벤트 타입 인덱스 시퀀스
            res_pad: (B, T_max) 이벤트 결과 인덱스 시퀀스
            lengths: (B,) 각 episode의 원래 길이(T) 벡터
            출력은 out: (B, 2) 예측 좌표

        b 범주형 입력 처리(임베딩)
            type_id와 res_id는 LabelEncoder로 만든 “카테고리 인덱스”이므로 크기/대소 의미가 없다.
            이를 그대로 수치 피처로 쓰지 않고, nn.Embedding을 통해 각 인덱스를 학습 가능한 벡터(emb_dim)로 변환

        c LSTM 입력 벡터 구성(in_dim)
            각 timestep(이벤트 1개)에서 LSTM에 들어갈 입력은
            연속형 cont (cont_dim)
            type 임베딩 (emb_dim)
            res 임베딩 (emb_dim)
            을 concat한 벡터
            따라서 LSTM의 input_size = in_dim = cont_dim + 2*emb_dim로 설정

        d LSTM 백본(backbone) 설정
            nn.LSTM은 (B,T,in_dim) 시퀀스를 시간 순서대로 읽고 hidden state에 정보를 누적해 episode를 요약
            핵심 하이퍼파라미터는 hidden_size(요약 벡터 크기), num_layers(층 수), dropout(층 사이 드롭아웃)
            PyTorch 규칙상 dropout은 num_layers>1일 때만 의미가 있어 1층이면 0으로 강제
            bidirectional=False로 단방향을 사용해 “미래 이벤트를 거꾸로 읽는 효과”를 배제

        e padding 무시 처리(pack_padded_sequence)
            batch에는 padding(0) 구간이 포함되므로, LSTM이 이를 실제 이벤트로 학습하지 않도록 설정
            forward에서 pack_padded_sequence(x, lengths, ...)를 적용해 각 샘플의 진짜 길이만큼만 LSTM이 처리

        f episode 요약 벡터 선택(h_last)
            LSTM 출력에서 각 layer의 마지막 hidden state h_n을 얻고,
            맨 위 layer의 마지막 hidden인 h_n[-1]을 episode의 대표 요약 벡터로 사용

        g 회귀 헤드(head)로 좌표 출력
            episode 요약 벡터 h_last를 MLP(head)에 넣어 (x,y) 좌표 2개를 출력
            구조는 Linear(hidden→hidden) + ReLU + Linear(hidden→2)로,
            첫 Linear는 표현을 재조합하고, ReLU로 비선형성을 추가해 복잡한 좌표 변환을 학습하며, 마지막 Linear가 최종 2차원 출력

    6. 모델 학습 및 검증
        a 평가/학습 목표 분리(손실 vs 지표)
            학습 업데이트는 회귀 손실(criterion)로 수행하고, 검증 평가는 대회 지표인 유클리드 거리로 수행
            “모델을 어떻게 학습시키는가”는 SmoothL1Loss
            “성능을 어떻게 판단하는가”는 Euclidean Distance

        b Euclidean distance 계산 함수
            pred와 true는 (B,2) 좌표이며, 각 샘플 i에 대해 d_i = sqrt((pred_x - true_x)^2 + (pred_y - true_y)^2) 계산

            리더보드는 전체 샘플 평균 거리를 쓰는 형태이므로, 코드에서는 배치별 평균 거리(d.mean())를 구하고, 이를 배치 단위로 모아 epoch 평균(valid_euclid_dist) 계산

        c Optimizer/하이퍼파라미터 반영
            Adam optimizer

        d Train loop(파라미터 업데이트)
            model.train()으로 학습 모드 전환
            각 배치마다
            1) 입력/정답을 device(cpu/cuda)로 이동
            2) optimizer.zero_grad()로 gradient 초기화
            3) pred = model(...) forward
            4) loss = criterion(pred, y) 손실 계산
            5) loss.backward()로 gradient 계산
            6) optimizer.step()로 파라미터 업데이트

            배치 loss들을 tr_losses에 쌓아 epoch 평균 train_loss를 계산한다.

        e Validation loop(평가 전용)
            model.eval()로 평가 모드 전환
            torch.no_grad()로 gradient 계산을 끄기
            각 배치에서 pred를 얻고, euclidean_dist(pred, y)로 배치 평균 거리를 계산해 val_dists에 누적
            epoch 평균 valid_euclid_dist로 검증 성능을 산출

        f Best checkpoint 저장/복원
            매 epoch마다 valid_euclid_dist가 best_val보다 개선되면, 현재 모델의 state_dict를 CPU로 clone하여 best_state에 저장

            학습 종료 후 best_state를 다시 model에 load하여, “마지막 epoch”가 아닌 “검증 기준 최선” 모델을 최종 모델로 사용

        g tqdm 진행률 표시

    7. 평가 데이터셋 추론
        a 경로/상수/디바이스 준비
            test.csv(TEST_META_PATH)에서 각 episode 파일 경로를 읽고, sample_submission.csv(SUBMISSION_PATH)를 제출 템플릿으로 로드
            
            학습 때와 동일한 경기장/골대/페널티박스 기하 상수(105x68, 골대 선분, 박스 범위 등)를 그대로 선언
            
            모델을 cuda 가능 시 GPU로 올린 뒤 eval 모드로 전환

        b test_meta와 submission 템플릿 로드
            test_meta는 game_episode별 episode CSV의 상대경로(path)를 제공하고, submission은 최종 제출 행 순서(game_episode 순서)를 강제하는 역할
            
            최종 출력은 submission의 행 순서에 맞춰 end_x, end_y를 채워야 함

        c build_episode_from_df로 “학습과 동일한 전처리/피처 생성”을 재현
            
            각 episode DataFrame g에 대해
            1. 정렬: time_seconds, action_id 기준으로 이벤트 순서를 확정
            2. 결측 처리: type_name/result_name의 NaN을 특수 토큰으로 채움
            3. 미등록 라벨 처리: train LabelEncoder(le_type/le_res)에 없던 카테고리는 NA 토큰으로 치환(추론 에러 방지)
            4. 범주형 인코딩: type_id, res_id로 변환(Embedding 입력용 정수 인덱스)
            5. 연속형 피처 계산: dt, 좌표(sx,sy,ex,ey), 마지막 이벤트 end 마스킹(ex_mask/ey_mask), is_end 플래그
            6. 고정 영역/기하 피처: 골대 선분까지 거리(dist_to_goal), 골문 각도 표현(theta_mid_sin, theta_mid_cos, theta_view), 하프라인(d_half, in_own_half), 박스(dist_p_box, in_p_box)

            핵심: cont는 학습 때와 동일한 14개 피처 “같은 순서”로

        d episode별 예측 수행 및 pred_map 저장
            test_meta를 순회하며 각 row의 path로 episode CSV를 읽고, build_episode_from_df로 (cont, type_id, res_id)를 만든 뒤 텐서로 변환
            
            이때 배치 차원을 추가해 (1,T,…) 형태로 만들고 lengths=[T]를 함께 넣어 모델이 padding을 무시
            
            모델 출력 (1,2)를 (2,)로 변환해 pred_map[game_episode] = (pred_x, pred_y) 형태로 저장

        e 제출 템플릿 순서로 매칭하여 submission 완성
            submission의 game_episode 순서를 기준으로 pred_map에서 예측값을 찾아 end_x/end_y 리스트를 채우기
            
            만약 pred_map에 없는 key가 있으면 missing에 기록하고 0,0을 넣는데, 정상 파이프라인이면 missing은 0
            
            마지막에 submission["end_x"], submission["end_y"]를 채우고 inference 완료 행 수를 출력

    8. Submisiion
        a. 덮어쓰기 주의

3.  LSTM_2_submit_1
        layer 개수 증가 (1 -> 2)

        labelencoder 0, 1, 2...
        padding = 0
        label encoder의 첫번째 값을 padding으로 착각하는 문제

4. LSTM_2
    변경사항
        EPOCHS = 500
        BATCH_SIZE = 512
        NUM_LAYERS = 4
        DROPOUT = 0.25
    결과
        [epoch 396] train_loss=2.0921 | valid_euclid_dist=16.0016
                                                                           
        [epoch 397] train_loss=2.0106 | valid_euclid_dist=15.9003
                                                                                
        단순 볼륨 키운다고 성능 개선 X

5. LSTM_2_submit_2
        SEED = 42
        N_SPLITS = 5 # number of folds
        FOLD = 0 # which fold for validation
        K = 50 # number of events to consider before the target event if smaller than K, pad with zeros
        MIN_EVENTS = 2
        EPOCHS = 30
        BATCH_SIZE = 256
        LR = 1e-3
        WEIGHT_DECAY = 1e-5
        HIDDEN_SIZE = 256 # LSTM hidden size
        NUM_LAYERS = 1 # number of LSTM layers
        DROPOUT = 0.2
        NUM_WORKERS = 0

6. LSTM_2_submit_3
        SEED = 42
        N_SPLITS = 5 # number of folds
        FOLD = 0 # which fold for validation
        K = 50 # number of events to consider before the target event if smaller than K, pad with zeros
        MIN_EVENTS = 2
        EPOCHS = 30
        BATCH_SIZE = 256
        LR = 1e-3
        WEIGHT_DECAY = 1e-5
        HIDDEN_SIZE = 256 # LSTM hidden size
    수정 -> NUM_LAYERS = 2 # number of LSTM layers
        DROPOUT = 0.2
        NUM_WORKERS = 0

7. LSTM_2_submit_4
        파라미터 변경 X
        이전 이벤트 관련 feature추가


8. LSTM_2_submit_5
        feature 간소화 12개
        feature scailing 맞추기 -> 성능 급하락

        attention pooling 현재까지 최고 성능

9. LSTM_2_submit_6
        dropout 0.2 -> 0.4
        multihead attention pooling
        data augmentation -> 좌표에 noise
        valid 10 epoch 동안 개선 없으면 LR 절반으로

        과적합 해결방안 -> training을 어렵게 만들어라
        1. dropout 증가
        2. data augmentation
        3. reducing lr

        논리적으로 생각하기!!!!

단순 피처 증가가 방법이 아니다. 근본적인 이유 찾아야 됨
------------------------------------------------


