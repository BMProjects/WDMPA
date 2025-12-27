#!/bin/bash
# ONNX模型导出脚本 (PC端)
# 用法: ./deploy/scripts/export_all_models.sh

set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
OUTPUT_DIR="$PROJECT_ROOT/deploy/onnx"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "==================================="
echo "导出所有模型为ONNX格式"
echo "==================================="

# 导出WDMPA-Net (fold 0)
echo "[1/4] 导出 WDMPA-Net..."
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model wdmpa \
    --weights weights/fold0_best.pkl \
    --output "$OUTPUT_DIR/wdmpa_fold0.onnx" \
    --opset 11

# 导出MobileNetV3 (fold 0)
echo "[2/4] 导出 MobileNetV3..."
WEIGHT_FILE=$(find runs/baselines/mobilenetv3/fold0 -name "best.pkl" | head -1)
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model mobilenetv3 \
    --weights "$WEIGHT_FILE" \
    --output "$OUTPUT_DIR/mobilenetv3_fold0.onnx" \
    --opset 11

# 导出ShuffleNetV2 (fold 0)
echo "[3/4] 导出 ShuffleNetV2..."
WEIGHT_FILE=$(find runs/baselines/shufflenetv2/fold0 -name "best.pkl" | head -1)
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model shufflenetv2 \
    --weights "$WEIGHT_FILE" \
    --output "$OUTPUT_DIR/shufflenetv2_fold0.onnx" \
    --opset 11

# 导出L2CS-Net
echo "[4/4] 导出 L2CS-Net..."
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model l2cs \
    --weights weights/l2cs_mpiigaze/fold0.pkl \
    --output "$OUTPUT_DIR/l2cs_net.onnx" \
    --opset 11

echo ""
echo "✅ 所有模型已导出到: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
