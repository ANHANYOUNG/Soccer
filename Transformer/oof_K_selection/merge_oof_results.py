"""
OOF 분석 결과 합치기
각 GPU에서 독립적으로 실행된 K별 결과를 하나로 통합
"""

import pandas as pd
import numpy as np
import os

K_VALUES = [12, 20, 32, 50]

print("=" * 70)
print("Merging OOF Analysis Results")
print("=" * 70)

results = {}
missing = []

# 각 K별 결과 파일 읽기
for k in K_VALUES:
    filename = f"oof_results_k{k}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0)
        results[k] = df[str(k)].values
        print(f"✓ Loaded: {filename}")
    else:
        missing.append(k)
        print(f"✗ Missing: {filename}")

if missing:
    print(f"\n⚠ Warning: Missing results for K = {missing}")
    print("Please wait for all processes to complete.")
    exit(1)

print("\n" + "=" * 70)
print("OOF ANALYSIS SUMMARY")
print("=" * 70)
print(f"{'K':<6} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} | Fold Results")
print("-" * 70)

best_scores = {}
for k in K_VALUES:
    vals = results[k]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    min_val = np.min(vals)
    max_val = np.max(vals)
    fold_str = ", ".join([f"{v:.2f}" for v in vals])
    print(f"{k:<6} | {mean_val:>10.4f} | {std_val:>10.4f} | {min_val:>10.4f} | {max_val:>10.4f} | [{fold_str}]")
    
    # Mean + 0.5*Std for LB stability
    best_scores[k] = mean_val + 0.5 * std_val

print("-" * 70)

# Recommendation
best_k = min(best_scores, key=best_scores.get)
print(f"\n[RECOMMENDATION] K = {best_k}")
print(f"  Mean: {np.mean(results[best_k]):.4f}")
print(f"  Std:  {np.std(results[best_k]):.4f}")
print(f"  Score (Mean + 0.5*Std): {best_scores[best_k]:.4f}")
print(f"  → Best for LB stability (낮은 평균 + 낮은 분산)")

# 전체 결과 저장
results_df = pd.DataFrame(results)
results_df.index = [f"fold_{i}" for i in range(len(results[K_VALUES[0]]))]
results_df.to_csv("oof_analysis_results.csv")
print(f"\n✓ Saved: oof_analysis_results.csv")
print("=" * 70)

# 추천 K 값 시각화
print("\n[Stability Ranking] (Lower is better)")
for i, (k, score) in enumerate(sorted(best_scores.items(), key=lambda x: x[1]), 1):
    mean = np.mean(results[k])
    std = np.std(results[k])
    print(f"{i}. K={k:2d}  Score={score:.4f}  (Mean={mean:.4f}, Std={std:.4f})")
