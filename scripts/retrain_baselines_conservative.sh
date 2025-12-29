#!/bin/bash
# 保守参数重训练基线模型 (Fold 0)
# 用于Jetson Nano部署对比

set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
DATA_ROOT="/home/bm/Data/MPIIFaceGaze_Processed"

cd "$PROJECT_ROOT"

echo "========================================"
echo "基线模型重训练 (保守参数)"
echo "========================================"
echo "Fold: 0"
echo "Batch Size: 32"
echo "Learning Rate: 1e-4"
echo "Epochs: 60"
echo "========================================"
echo ""

# MobileNetV3
echo "[1/2] Training MobileNetV3..."
echo "开始时间: $(date)"
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
    --model mobilenetv3 \
    --data-root "$DATA_ROOT" \
    --fold 0 \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 60 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --warmup-epochs 5 \
    --output-dir runs/baselines_conservative/mobilenetv3/fold0 \
    --name fold0_conservative

echo "✅ MobileNetV3 完成"
echo ""

# ShuffleNetV2
echo "[2/2] Training ShuffleNetV2..."
echo "开始时间: $(date)"
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
    --model shufflenetv2 \
    --data-root "$DATA_ROOT" \
    --fold 0 \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 60 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --warmup-epochs 5 \
    --output-dir runs/baselines_conservative/shufflenetv2/fold0 \
    --name fold0_conservative

echo "✅ ShuffleNetV2 完成"
echo ""

echo "========================================"
echo "所有模型重训练完成!"
echo "结束时间: $(date)"
echo "========================================"
echo ""
echo "权重位置:"
find runs/baselines_conservative -name "best.pkl"
