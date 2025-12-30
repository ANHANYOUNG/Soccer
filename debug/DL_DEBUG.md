# DL Debugging Guide

## 목차

- [논문과 다른 결과가 나오는 이유](#논문과-다른-결과가-나오는-이유)
- [Dataset의 중요성](#dataset의-중요성)
- [Troubleshooting](#troubleshooting)
- [1. Start Simple](#1-start-simple)
- [2. Implement](#2-implement)
- [3. Evaluate](#3-evaluate)
- [4. Improve](#4-improve)
- [5. Error Analysis](#5-error-analysis)
- [6. Tuning](#6-tuning)

## 논문과 다른 결과가 나오는 이유
![Diff_Paper](./01outline/difference_with_paper.png)
   - 구현 버그
   - 모델 하이퍼파라미터 차이
   - 데이터와 모델 안 맞음
   - 좋은 데이터가 아님
---

## Dataset의 중요성
![datasets](./01outline/dataset.png)
### PhD: Model, Algorithm
### Tesla: Datasets
---

## Troubleshooting
![troubleshooting](./01outline/troubleshooting.png)

---

## 1. Start Simple
![startsimple](./02startsimple/start%20simple.png), ![startsimple2](./02startsimple/startsimple2.png), ![startsimple3](./02startsimple/startsimple3.png)

## 2. Implement
![implement](./03implement/implement1.png)

## Get Your Model to Run
![common_issues](./03implement/01GYMR/common_issues.png)
### 1. **Shape Mismatch**
![shape_mismatch](./03implement/01GYMR/shape_mistmatch.png)

### 2. **Casting Issue**
![casting_issue](./03implement/01GYMR/casting_issue.png)

### 3. **OOM**
![oom](./03implement/01GYMR/oom.png)

### 4. **Others**
![others](./03implement/01GYMR/others.png)

---

## Overfit A Single Batch
![overfit](./03implement/02overfit_single/overfit.png)
### 1. **Error Goes Up**
  - 부호 반대
### 2. **Error Explodes**
  - Nan, Inf 수치 문제, LR 높을 때
### 3. **Error Oscillates**
  - LR 낮추고 데이터 라벨 섞였는지, Data Augmentation 잘못 됐는지 확인
### 4. **Error Plateaus**
  - LR 늘리기, Regulation 없애기
---

## Compare to A Known Result
### 1. 유사한 데이터셋의 공식 모델 구현체의 성능과 비교 — 한줄씩 보면서 같은 output을 내는지 확인
### 2. MNIST같은 벤치마크 데이터셋에서 공식 모델 구현체의 성능
### 3. 비공식 모델 구현 — GitHub같은곳의 소스들은 보통 버그가 있으니 unofficial은 조심(star 많다면 OK)
### 4. 논문에 나온 결과(코드 없을 경우)
### 5. MNIST같은 벤치마크 데이터셋에서 내 모델의 성능 — 내 모델이 간단한 세팅으로도 좋은 성능을 내는지 확인
### 6. 유사한 데이터셋의 유사한 모델의 성능 — 보통 어느정도 성능이 나오는지 확인
### 7. 정말 간단한 베이스라인(평균 출력이나 선형회귀 등…) — 내 모델이 무엇이든 학습이 되는지 확인

---

## 3. Evaluate
### 1. $Irreducible Error$
#### Human-performance baseline 같은 더 줄일 수 없는 최소한의 에러

### 2. $Bias \approx Error_{train} - Error_{irreducible}$ 
#### 클 수록 underfitting

### 3. $Variance \approx Error_{val} - Error_{train}$ 
#### 클수록 overfitting

### 4. $Val$ $Overfitting$
#### 또 validation data의 error를 줄이는 과정에서 validation에 오버피팅 될 수 있음 

---

## 4. Improve
### Underfitting
![underfitting](./04improve/underfitting.png)
### Overfitting
![overfitting](./04improve/overfitting.png)
### Distribution Shift
![distribution_shift](./04improve/distribution_shift.png)

---

## 5. Error Analysis
### 어떤 데이터를 모델이 틀리는가?
#### example
![ea](./05error_analysis/ea.png)

### Domain Adaptation (Pre-Training)
![da](./05error_analysis/da.png), ![type_da](./05error_analysis/type_da.png)

---

## 6. Tuning
![1](./06tuning/1.png)
![2](./06tuning/2.png)

### Method 1: Manual Hyperparam Optimization
![method1](./06tuning/method1.png)
#### 수동최적화

### Method 2: Grid Search
![method2](./06tuning/method2.png)
#### 좋은 하이퍼파라미터 범위에 대한 지식 필요
### Method 3: Random Search
![method3](./06tuning/method3.png)
#### 구현하기 쉬움

### Method 4: Coarse-to-Fine
![method4](./06tuning/method4.png)
#### 무작위 검색 -> 좋은 범위 좁혀 검색 -> 반복

### Method 5: Bayesian Hyperparam Optimization
![method5](./06tuning/method5.png)
#### 확률적 모델 만들기
