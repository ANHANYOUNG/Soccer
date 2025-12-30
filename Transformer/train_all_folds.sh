#!/bin/bash

# 5-Fold Scd /home/ahy0502/soccer/open_track1/Transformer

# Create log directory with incremental numbering
LOG_BASE_DIR="fold_log"
mkdir -p "$LOG_BASE_DIR"

# Find next available log directory number
LOG_NUM=0
while [ -d "${LOG_BASE_DIR}/fold_log_${LOG_NUM}" ]; do
    LOG_NUM=$((LOG_NUM + 1))
done

LOG_DIR="${LOG_BASE_DIR}/fold_log_${LOG_NUM}"
mkdir -p "$LOG_DIR"

echo "Log directory: $LOG_DIR"
echo ""

# Trap Ctrl+C to kill all background processes
trap 'echo -e "\n\n⚠️  Ctrl+C detected! Killing all training processes..."; kill $(jobs -p) 2>/dev/null; exit 1' INT

# 각 fold를 다른 GPU에서 백그라운드로 실행 (출력을 로그 파일로 저장)
echo "Starting Fold 0 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 > "${LOG_DIR}/fold_0.log" 2>&1 &
PID0=$! Training Script
# GPU 5개를 사용해 각 fold를 순차적으로 학습 (GPU당 1개 프로세스 보장)

# UTF-8 출력 안정화
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "========================================"
echo "Starting 5-Fold Sequential Training"
echo "========================================"
echo "Total: 3 seeds × 5 folds = 15 models"
echo "GPU 사용 전략:"
echo "  Fold 0 → GPU 0"
echo "  Fold 1 → GPU 1"
echo "  Fold 2 → GPU 2"
echo "  Fold 3 → GPU 3"
echo "  Fold 4 → GPU 4"
echo "각 fold는 3개 seed 순차 학습 (42→123→456)"
echo "========================================"
echo ""

cd /home/ahy0502/soccer/open_track1/Transformer || exit 1

# Trap Ctrl+C to kill all background processes
trap 'echo -e "\n\nCtrl+C detected! Killing all training processes..."; kill $(jobs -p) 2>/dev/null; exit 1' INT

# 각 fold를 다른 GPU에서 백그라운드로 실행 (출력을 로그 파일로 저장)
echo "Starting Fold 0 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 > fold_0.log 2>&1 &
PID0=$!

echo "Starting Fold 1 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 > "${LOG_DIR}/fold_1.log" 2>&1 &
PID1=$!

echo "Starting Fold 2 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python train.py --fold 2 > "${LOG_DIR}/fold_2.log" 2>&1 &
PID2=$!

echo "Starting Fold 3 on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python train.py --fold 3 > "${LOG_DIR}/fold_3.log" 2>&1 &
PID3=$!

echo "Starting Fold 4 on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python train.py --fold 4 > "${LOG_DIR}/fold_4.log" 2>&1 &
PID4=$!

echo ""
echo "All folds started!"
echo "========================================"
echo "Process IDs:"
echo "  Fold 0 (GPU 0): $PID0"
echo "  Fold 1 (GPU 1): $PID1"
echo "  Fold 2 (GPU 2): $PID2"
echo "  Fold 3 (GPU 3): $PID3"
echo "  Fold 4 (GPU 4): $PID4"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  ./check_progress.sh"
echo "  watch -n 10 ./check_progress.sh"
echo "  nvidia-smi"
echo ""
echo "Press Ctrl+C to stop all training"
echo ""

# 모든 프로세스가 끝날 때까지 대기
echo "Waiting for all folds to complete..."
echo ""

TOTAL_EPOCHS=150
TOTAL_SEEDS=3

# Helper function to extract current epoch from log
get_current_progress() {
    local fold=$1
    local log_file="${LOG_DIR}/fold_${fold}.log"
    if [ -f "$log_file" ]; then
        # 마지막 "[epoch XXX]" 패턴 찾기
        local last_epoch
        last_epoch=$(grep -oP '\[epoch \K[0-9]+' "$log_file" 2>/dev/null | tail -1)

        # 완료된 시드 수 세기 ("Best validation distance:" 패턴)
        local completed_seeds
        completed_seeds=$(grep -c "Best validation distance:" "$log_file" 2>/dev/null)

        if [ -n "$last_epoch" ]; then
            # 10진수로 강제 변환 (08, 09 같은 8진수 오류 방지)
            last_epoch=$((10#$last_epoch))
            # 현재 진행: (완료된 시드 * 150) + 현재 에포크
            local total_progress=$((completed_seeds * TOTAL_EPOCHS + last_epoch))
            echo "$total_progress"
        elif [ "$completed_seeds" -gt 0 ]; then
            echo $((completed_seeds * TOTAL_EPOCHS))
        else
            echo "0"
        fi
    else
        echo "0"
    fi
}

# Progress bar drawing function
draw_fold_progress() {
    local fold=$1
    local progress=$2
    local total=$((TOTAL_EPOCHS * TOTAL_SEEDS))

    # Handle empty or invalid progress
    if [ -z "$progress" ] || [ "$progress" = "N/A" ]; then
        progress=0
    fi

    local percent=0
    local filled=0
    local empty=30

    if [ "$total" -gt 0 ]; then
        percent=$((progress * 100 / total))
        filled=$((progress * 30 / total))
        empty=$((30 - filled))
    fi

    # Prevent overflow
    if [ "$filled" -gt 30 ]; then filled=30; fi
    if [ "$filled" -lt 0 ]; then filled=0; fi
    if [ "$empty" -gt 30 ]; then empty=30; fi
    if [ "$empty" -lt 0 ]; then empty=0; fi

    printf "Fold %d [GPU %d]: [" "$fold" "$fold"

    # 중요: tr은 멀티바이트(유니코드) 문자를 깨뜨릴 수 있으니 sed/awk 사용
    if [ "$filled" -gt 0 ]; then
        # 유니코드 블록
        awk -v n="$filled" 'BEGIN{for(i=0;i<n;i++) printf "█"}'
    fi
    if [ "$empty" -gt 0 ]; then
        awk -v n="$empty" 'BEGIN{for(i=0;i<n;i++) printf "░"}'
    fi

    printf "] %3d%% (%d/%d)\n" "$percent" "$progress" "$total"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training Progress (150 epochs × 3 seeds per fold)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Continuous monitoring until all processes complete
FIRST_DRAW=""
while true; do
    # Move cursor up 5 lines to overwrite previous output
    if [ "$FIRST_DRAW" = "done" ]; then
        tput cuu 5
    fi
    FIRST_DRAW="done"

    ALL_DONE=true

    for fold in {0..4}; do
        PROGRESS=$(get_current_progress "$fold")
        draw_fold_progress "$fold" "$PROGRESS"

        TOTAL_REQUIRED=$((TOTAL_EPOCHS * TOTAL_SEEDS))
        if [ -n "$PROGRESS" ] && [ "$PROGRESS" -lt "$TOTAL_REQUIRED" ]; then
            ALL_DONE=false
        fi
    done

    if [ "$ALL_DONE" = "true" ]; then
        break
    fi

    sleep 5
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for all processes to fully complete
wait "$PID0" "$PID1" "$PID2" "$PID3" "$PID4"

# Helper function to extract best validation distance from log
get_best_dist() {
    local fold=$1
    local log_file="${LOG_DIR}/fold_${fold}.log"
    if [ -f "$log_file" ]; then
        grep "Best validation distance:" "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}'
    else
        echo "N/A"
    fi
}

# Calculate average
calc_average() {
    local sum=0
    local count=0
    for dist in "$@"; do
        if [ "$dist" != "N/A" ]; then
            sum=$(echo "$sum + $dist" | bc -l)
            count=$((count + 1))
        fi
    done
    if [ "$count" -gt 0 ]; then
        echo "scale=4; $sum / $count" | bc -l
    else
        echo "N/A"
    fi
}

DIST0=$(get_best_dist 0)
DIST1=$(get_best_dist 1)
DIST2=$(get_best_dist 2)
DIST3=$(get_best_dist 3)
DIST4=$(get_best_dist 4)

echo ""
echo "========================================"
echo "All training completed!"
echo "========================================"
echo "Final Results:"
echo "  Fold 0: ${DIST0}"
echo "  Fold 1: ${DIST1}"
echo "  Fold 2: ${DIST2}"
echo "  Fold 3: ${DIST3}"
echo "  Fold 4: ${DIST4}"
echo ""
AVG=$(calc_average "$DIST0" "$DIST1" "$DIST2" "$DIST3" "$DIST4")
echo "  Average: ${AVG}"
echo ""
echo "Models saved in: models/seed_*_fold_*/"
echo "Logs saved in: ${LOG_DIR}/"
echo ""
echo "Next step: Run inference"
echo "  python inference.py"
echo "========================================"
