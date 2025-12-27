#!/bin/bash
# WDMPA-Net 前台训练脚本 (改进版)
# 用法: ./scripts/run_experiments_foreground.sh
# 按 Ctrl+C 可停止整个脚本

DATA_ROOT="/home/bm/Data/MPIIFaceGaze_Processed"
EPOCHS=60
# 使用 train.py 新默认值: batch_size=256, lr=8e-4, workers=8, AMP=True
# 这些参数已针对 RTX 4090 优化，预计训练速度提升 3-5 倍
PROJECT_ROOT="/home/bm/Dev/WDMPA"
LOG_DIR="$PROJECT_ROOT/results/logs"
RESULTS_FILE="$PROJECT_ROOT/results/experiment_results.csv"

# 创建目录
mkdir -p $LOG_DIR

# Ctrl+C 处理：停止整个脚本
trap 'echo ""; echo "⚠️  训练被用户中断"; echo "已完成的结果保存在: $RESULTS_FILE"; exit 1' INT

# 检查结果文件
if [ ! -f "$RESULTS_FILE" ]; then
    echo "model,fold,best_error,final_error,timestamp" > $RESULTS_FILE
fi

echo "====================================="
echo "WDMPA-Net 实验 (RTX 4090 极限优化)"
echo "开始时间: $(date)"
echo "====================================="
echo "配置: Batch=512, LR=1.6e-3, AMP=True"
echo "Workers=12, Prefetch=4, Warmup=8 epochs"
echo "Epochs: $EPOCHS"
echo "按 Ctrl+C 停止整个脚本"
echo "====================================="

cd $PROJECT_ROOT

# 函数: 提取最佳错误
extract_best_error() {
    local log_file=$1
    grep "New best:" $log_file 2>/dev/null | tail -1 | awk -F: '{print $NF}' | tr -d '°' | tr -d ' '
}

# 函数: 提取最终错误
extract_final_error() {
    local log_file=$1
    grep "Error:" $log_file 2>/dev/null | tail -1 | sed 's/.*Error: //' | awk -F'°' '{print $1}'
}

# 函数: 检查是否已完成 (检查权重文件是否存在)
is_completed() {
    local output_dir=$1
    # 只有当 best.pkl 存在时才认为已完成
    local best_files=$(find "$output_dir" -name "best.pkl" 2>/dev/null | head -1)
    if [ -n "$best_files" ]; then
        return 0  # 已完成
    else
        return 1  # 未完成
    fi
}

# 函数: 训练单个模型
train_model() {
    local model=$1
    local fold=$2
    local output_dir=$3
    local log_file=$4
    
    # 检查是否已完成 (基于权重文件)
    if is_completed "$output_dir"; then
        echo "  ⏭️  $model fold $fold 已完成 (找到 best.pkl)，跳过"
        return 0
    fi
    
    echo "  [$(date +%H:%M:%S)] 训练 $model fold $fold..."
    
    # 训练 (使用 train.py 默认优化参数)
    PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
        --model $model \
        --data-root $DATA_ROOT \
        --fold $fold \
        --epochs $EPOCHS \
        --output-dir $output_dir \
        --name fold$fold 2>&1 | tee $log_file
    
    # 检查训练是否成功完成
    if is_completed "$output_dir"; then
        # 记录结果
        BEST_ERR=$(extract_best_error $log_file)
        FINAL_ERR=$(extract_final_error $log_file)
        echo "$model,$fold,$BEST_ERR,$FINAL_ERR,$(date +%Y%m%d_%H%M%S)" >> $RESULTS_FILE
        echo "    ✅ 完成! Best: ${BEST_ERR}°"
    else
        echo "    ❌ 训练未完成 (未找到 best.pkl)"
    fi
}

# 1. MobileNetV3-Small (15 folds)
echo ""
echo "[阶段 1/3] MobileNetV3-Small (15 folds)"
echo "====================================="
for fold in {0..14}; do
    train_model "mobilenetv3" $fold \
        "runs/baselines/mobilenetv3/fold$fold" \
        "$LOG_DIR/mobilenetv3_fold${fold}.log"
done

# 2. ShuffleNetV2 (15 folds)
echo ""
echo "[阶段 2/3] ShuffleNetV2 (15 folds)"
echo "====================================="
for fold in {0..14}; do
    train_model "shufflenetv2" $fold \
        "runs/baselines/shufflenetv2/fold$fold" \
        "$LOG_DIR/shufflenetv2_fold${fold}.log"
done

# 3. 消融实验 (fold 0)
echo ""
echo "[阶段 3/3] 消融实验 (fold 0)"
echo "====================================="
ABLATION_MODELS=("wdmpa_awwd_fixed" "wdmpa_stride_conv" "wdmpa_channel_only" "wdmpa_spatial_only" "wdmpa_single_scale" "wdmpa_no_attention")

for model in "${ABLATION_MODELS[@]}"; do
    train_model "$model" 0 \
        "runs/ablation/$model" \
        "$LOG_DIR/${model}.log"
done

echo ""
echo "====================================="
echo "所有训练完成!"
echo "结束时间: $(date)"
echo "====================================="
echo ""
echo "结果摘要:"
echo "====================================="
cat $RESULTS_FILE | column -t -s','
echo ""
echo "权重保存在: runs/"
