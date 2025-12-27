#!/bin/bash
# WDMPA-Net 完整实验批量训练脚本 (带日志记录)
# 用法: nohup ./scripts/run_all_experiments.sh > results/train_log.txt 2>&1 &

set -e

DATA_ROOT="/home/bm/Data/MPIIFaceGaze_Processed"
EPOCHS=60
BATCH_SIZE=32  # 与 WDMPA-Net 预训练配置一致 (configs/train_mpiigaze.yaml)
PROJECT_ROOT="/home/bm/Dev/WDMPA"
LOG_DIR="$PROJECT_ROOT/results/logs"
RESULTS_FILE="$PROJECT_ROOT/results/experiment_results.csv"

# 创建日志目录
mkdir -p $LOG_DIR

# 初始化结果文件
echo "model,fold,best_error,final_error,timestamp" > $RESULTS_FILE

echo "====================================="
echo "WDMPA-Net 完整实验"
echo "开始时间: $(date)"
echo "====================================="
echo "数据集: $DATA_ROOT"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "日志目录: $LOG_DIR"
echo "====================================="

cd $PROJECT_ROOT

# 函数: 提取最佳错误
extract_best_error() {
    local log_file=$1
    grep "New best:" $log_file | tail -1 | awk -F: '{print $NF}' | tr -d '°' | tr -d ' '
}

# 函数: 提取最终错误
extract_final_error() {
    local log_file=$1
    grep "Error:" $log_file | tail -1 | sed 's/.*Error: //' | awk -F'°' '{print $1}'
}

# 1. 基线模型训练 (15 folds each)
echo ""
echo "[阶段 1] 基线模型训练"
echo "====================================="

# MobileNetV3-Small (15 folds)
echo "训练 MobileNetV3-Small..."
for fold in {0..14}; do
    LOG_FILE="$LOG_DIR/mobilenetv3_fold${fold}.log"
    echo "  [$(date +%H:%M:%S)] MobileNetV3 fold $fold..."
    
    PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
        --model mobilenetv3 \
        --data-root $DATA_ROOT \
        --fold $fold \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output-dir runs/baselines/mobilenetv3/fold$fold \
        --name fold$fold 2>&1 | tee $LOG_FILE
    
    # 记录结果
    BEST_ERR=$(extract_best_error $LOG_FILE)
    FINAL_ERR=$(extract_final_error $LOG_FILE)
    echo "mobilenetv3,$fold,$BEST_ERR,$FINAL_ERR,$(date +%Y%m%d_%H%M%S)" >> $RESULTS_FILE
    echo "    完成! Best: ${BEST_ERR}°"
done
echo "MobileNetV3 完成!"

# ShuffleNetV2 (15 folds)
echo ""
echo "训练 ShuffleNetV2..."
for fold in {0..14}; do
    LOG_FILE="$LOG_DIR/shufflenetv2_fold${fold}.log"
    echo "  [$(date +%H:%M:%S)] ShuffleNetV2 fold $fold..."
    
    PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
        --model shufflenetv2 \
        --data-root $DATA_ROOT \
        --fold $fold \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output-dir runs/baselines/shufflenetv2/fold$fold \
        --name fold$fold 2>&1 | tee $LOG_FILE
    
    BEST_ERR=$(extract_best_error $LOG_FILE)
    FINAL_ERR=$(extract_final_error $LOG_FILE)
    echo "shufflenetv2,$fold,$BEST_ERR,$FINAL_ERR,$(date +%Y%m%d_%H%M%S)" >> $RESULTS_FILE
    echo "    完成! Best: ${BEST_ERR}°"
done
echo "ShuffleNetV2 完成!"

# 2. 消融实验 (只需 fold 0)
echo ""
echo "[阶段 2] 消融实验 (fold 0)"
echo "====================================="

ABLATION_MODELS=("wdmpa_awwd_fixed" "wdmpa_stride_conv" "wdmpa_channel_only" "wdmpa_spatial_only" "wdmpa_single_scale" "wdmpa_no_attention")

for model in "${ABLATION_MODELS[@]}"; do
    LOG_FILE="$LOG_DIR/${model}.log"
    echo "  [$(date +%H:%M:%S)] 训练 $model..."
    
    PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
        --model $model \
        --data-root $DATA_ROOT \
        --fold 0 \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output-dir runs/ablation/$model \
        --name $model 2>&1 | tee $LOG_FILE
    
    BEST_ERR=$(extract_best_error $LOG_FILE)
    FINAL_ERR=$(extract_final_error $LOG_FILE)
    echo "$model,0,$BEST_ERR,$FINAL_ERR,$(date +%Y%m%d_%H%M%S)" >> $RESULTS_FILE
    echo "    完成! Best: ${BEST_ERR}°"
done

echo ""
echo "====================================="
echo "所有训练完成!"
echo "结束时间: $(date)"
echo "====================================="
echo ""
echo "结果摘要:"
echo "====================================="
cat $RESULTS_FILE
echo ""
echo "结果保存位置:"
echo "  - 训练日志: $LOG_DIR/"
echo "  - 权重文件: runs/"
echo "  - 结果汇总: $RESULTS_FILE"
