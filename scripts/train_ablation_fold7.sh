#!/bin/bash
# Fold 7 消融实验训练脚本
# 预计时间: ~6小时

set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
DATA_ROOT="/home/bm/Data/MPIIFaceGaze_Processed"
LOG_DIR="$PROJECT_ROOT/results/logs/ablation_fold7"
RESULTS_FILE="$PROJECT_ROOT/results/ablation_fold7_results.csv"
FOLD=7

mkdir -p "$LOG_DIR"

echo "model,fold,best_error,final_error,timestamp" > "$RESULTS_FILE"

echo "======================================"
echo "Fold 7 消融实验"
echo "开始时间: $(date)"
echo "======================================"

cd "$PROJECT_ROOT"

ABLATION_MODELS=(
    "wdmpa_awwd_fixed"
    "wdmpa_stride_conv"
    "wdmpa_channel_only"
    "wdmpa_spatial_only"
    "wdmpa_single_scale"
    "wdmpa_no_attention"
)

for model in "${ABLATION_MODELS[@]}"; do
    echo ""
    echo "[$(date +%H:%M:%S)] 训练 $model (Fold $FOLD)..."
    
    LOG_FILE="$LOG_DIR/${model}_fold7.log"
    
    PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
        --model $model \
        --data-root "$DATA_ROOT" \
        --fold $FOLD \
        --epochs 60 \
        --output-dir runs/ablation_fold7/$model \
        --name fold7 2>&1 | tee "$LOG_FILE"
    
    # 提取结果
    BEST_ERR=$(grep "New best:" "$LOG_FILE" | tail -1 | awk -F: '{print $NF}' | tr -d '°' | tr -d ' ')
    FINAL_ERR=$(grep "Error:" "$LOG_FILE" | tail -1 | sed 's/.*Error: //' | awk -F'°' '{print $1}')
    
    echo "$model,$FOLD,$BEST_ERR,$FINAL_ERR,$(date +%Y%m%d_%H%M%S)" >> "$RESULTS_FILE"
    echo "  完成! Best: ${BEST_ERR}°"
done

echo ""
echo "======================================"
echo "Fold 7 消融实验完成!"
echo "结束时间: $(date)"
echo "======================================"
echo ""
cat "$RESULTS_FILE" | column -t -s','
