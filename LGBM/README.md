# K리그 최종 패스 좌표 예측 (baseline)

이 저장소는 제공 데이터만 사용해, game_episode 단위 입력으로 마지막 패스 도착 좌표(end_x, end_y)를 예측하는 최소 파이프라인입니다.

1) train.py
- 학습 데이터로 모델을 학습하고, 추론에 필요한 weight 파일을 models/ 폴더에 저장합니다.

2) inference.py
- models/의 weight 파일을 로드해 테스트 데이터에 대해 예측을 수행하고 제출용 CSV를 생성합니다.


## 현재 폴더 구조(핵심)
- train.csv
- test/ (하위 폴더에 episode CSV들이 존재)
- sample_submission.csv
- match_info.csv
- train.py
- inference.py
- models/ (train.py 실행 후 weight 저장 위치)
- requirements.txt


## 규정 준수 체크(이 프로젝트에서 지키는 것)
아래 항목은 대화에서 제공된 규정 문구를 기준으로 작성했습니다.

1) 외부 데이터 사용 금지
- 이 프로젝트는 폴더 내 제공 파일(train.csv, test/, sample_submission.csv, match_info.csv)만 읽습니다.

2) 원격 API 기반 모델 사용 금지
- inference.py, train.py는 로컬에서 LightGBM을 실행하며 외부 서버 API를 호출하지 않습니다.

3) 사전학습 모델 사용 가능 범위
- 현재 파이프라인은 사전학습 모델을 사용하지 않습니다.

4) 평가 데이터 누수(Data Leakage) 금지
- 예측 입력은 game_episode 내부 이벤트로부터 만든 피처만 사용하도록 구성합니다.
- 다른 episode(같은 경기의 다른 episode 포함) 정보를 입력으로 섞어 쓰지 않습니다.
	(주의: 데이터 합치기/인코더 학습 등 구현이 잘못되면 누수 위험이 생길 수 있으니, 코드를 수정할 때는 항상 이 조건을 유지해야 합니다.)

5) 코드/산출물 제출 요건(프로젝트 형태)
- 학습 코드와 추론 코드를 분리했습니다: train.py / inference.py
- 추론은 weight 파일이 필요합니다: models/ 폴더의 lgbm_x_fold*.txt, lgbm_y_fold*.txt
- 파일 경로는 상대 경로를 사용합니다.
- CSV 및 코드/주석은 UTF-8 인코딩을 전제로 합니다.
- 라이브러리 버전은 requirements.txt에 기록합니다.


## 내가 지금 해야 할 일(순서)
아래 순서대로만 진행하면 됩니다.

0) (처음 한 번) 패키지 설치
- requirements.txt에 적힌 패키지를 설치합니다.

1) 학습 실행
- train.py를 실행합니다.
- 기대 결과: models/ 폴더에 아래 파일들이 생성됩니다.
	- models/lgbm_x_fold0.txt ...
	- models/lgbm_y_fold0.txt ...

2) 추론 실행
- inference.py를 실행합니다.
- 기대 결과: 제출 파일이 생성됩니다.
	- submission_lgbm_lastK_0.csv, submission_lgbm_lastK_1.csv 처럼 실행할 때마다 번호가 증가합니다.

3) 제출 파일 확인
- 생성된 submission_lgbm_lastK_{번호}.csv 파일이 UTF-8 CSV인지 확인합니다.


## 출력 파일
- submission_lgbm_lastK_{n}.csv


