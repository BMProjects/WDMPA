#!/bin/bash
# WDMPA-Net 15-fold 训练脚本
# 预计时间: ~15小时

set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
DATA_ROOT="/home/bm/Data/MPIIFaceGaze_Processed"
LOG_DIR="$PROJECT_ROOT/results/logs/wdmpa"
RESULTS_FILE="$PROJECT_ROOT/results/wdmpa_results.csv"

mkdir -p "$LOG_DIR"

echo "model,fold,best_error,final_error,timestamp" > "$RESULTS_FILE"

echo "======================================"
echo "WDMPA-Net 15-fold 训练"
echo "开始时间: $(date)"
echo "======================================"

cd "$PROJECT_ROOT"

for fold in {0..14}; do
    echo ""
    echo "[$(date +%H:%M:%S)] 训练 Fold $fold..."
    
    LOG_FILE="$LOG_DIR/wdmpa_fold${fold}.log"
    
    PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
        --model wdmpa \
        --data-root "$DATA_ROOT" \
        --fold $fold \
        --epochs 60 \
        --output-dir runs/wdmpa_retrain/fold$fold \
        --name fold$fold 2>&1 | tee "$LOG_FILE"
    
    # 提取结果
    BEST_ERR=$(grep "New best:" "$LOG_FILE" | tail -1 | awk -F: '{print $NF}' | tr -d '°' | tr -d ' ')
    FINAL_ERR=$(grep "Error:" "$LOG_FILE" | tail -1 | sed 's/.*Error: //' | awk -F'°' '{print $1}')
    
    echo "wdmpa,$fold,$BEST_ERR,$FINAL_ERR,$(date +%Y%m%d_%H%M%S)" >> "$RESULTS_FILE"
    echo "  完成! Best: ${BEST_ERR}°"
done

echo ""
echo "======================================"
echo "WDMPA-Net 15-fold 训练完成!"
echo "结束时间: $(date)"
echo "======================================"
echo ""
cat "$RESULTS_FILE" | column -t -s','
