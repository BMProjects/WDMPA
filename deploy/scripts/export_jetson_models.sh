#!/bin/bash
# 导出Jetson部署所需的ONNX模型 (Fold 0)

set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
OUTPUT_DIR="$PROJECT_ROOT/deploy/onnx"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "导出Jetson部署模型 (Fold 0)"
echo "========================================"
echo ""

# 1. WDMPA-Net (原始预训练权重)
echo "[1/4] 导出 WDMPA-Net (原始预训练)..."
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model wdmpa \
    --weights weights/fold0_best.pkl \
    --output "$OUTPUT_DIR/wdmpa_fold0.onnx" \
    --opset 11
echo "✅ WDMPA-Net 完成"
echo ""

# 2. L2CS-Net (原始预训练权重)
echo "[2/4] 导出 L2CS-Net (原始预训练)..."
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model l2cs \
    --weights weights/l2cs_mpiigaze/fold0.pkl \
    --output "$OUTPUT_DIR/l2cs_fold0.onnx" \
    --opset 11
echo "✅ L2CS-Net 完成"
echo ""

# 3. MobileNetV3 (保守参数重训练)
echo "[3/4] 导出 MobileNetV3 (保守参数)..."
WEIGHT=$(find runs/baselines_conservative/mobilenetv3/fold0 -name "best.pkl" 2>/dev/null | head -1)
if [ -z "$WEIGHT" ]; then
    echo "⚠️  警告: 未找到MobileNetV3保守权重，请先运行 retrain_baselines_conservative.sh"
    exit 1
fi
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model mobilenetv3 \
    --weights "$WEIGHT" \
    --output "$OUTPUT_DIR/mobilenetv3_fold0.onnx" \
    --opset 11
echo "✅ MobileNetV3 完成"
echo ""

# 4. ShuffleNetV2 (保守参数重训练)
echo "[4/4] 导出 ShuffleNetV2 (保守参数)..."
WEIGHT=$(find runs/baselines_conservative/shufflenetv2/fold0 -name "best.pkl" 2>/dev/null | head -1)
if [ -z "$WEIGHT" ]; then
    echo "⚠️  警告: 未找到ShuffleNetV2保守权重，请先运行 retrain_baselines_conservative.sh"
    exit 1
fi
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model shufflenetv2 \
    --weights "$WEIGHT" \
    --output "$OUTPUT_DIR/shufflenetv2_fold0.onnx" \
    --opset 11
echo "✅ ShuffleNetV2 完成"
echo ""

echo "========================================"
echo "所有模型已导出!"
echo "========================================"
echo ""
ls -lh "$OUTPUT_DIR"
