#!/bin/bash
# TensorRT Engine Build Script for Jetson Nano
#
# Prerequisites:
#   - JetPack 4.6.6 installed
#   - trtexec available at /usr/src/tensorrt/bin/trtexec
#
# Usage:
#   ./deploy/build_tensorrt.sh wdmpa_fold0.onnx

set -e

ONNX_FILE=${1:-"deploy/wdmpa_fold0.onnx"}
OUTPUT_DIR=${2:-"deploy/engines"}
PRECISION=${3:-"fp16"}  # fp32, fp16, or int8

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get model name
MODEL_NAME=$(basename "$ONNX_FILE" .onnx)
ENGINE_FILE="$OUTPUT_DIR/${MODEL_NAME}_${PRECISION}.engine"

echo "=============================================="
echo "TensorRT Engine Builder"
echo "=============================================="
echo "Input ONNX:  $ONNX_FILE"
echo "Output:      $ENGINE_FILE"
echo "Precision:   $PRECISION"
echo "=============================================="

# Check if trtexec exists
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -f "$TRTEXEC" ]; then
    echo "Error: trtexec not found at $TRTEXEC"
    echo "Please ensure TensorRT is installed (part of JetPack)"
    exit 1
fi

# Build engine
if [ "$PRECISION" = "fp16" ]; then
    $TRTEXEC \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16 \
        --workspace=1024 \
        --verbose
elif [ "$PRECISION" = "int8" ]; then
    echo "INT8 requires calibration data. Using FP16 instead."
    $TRTEXEC \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16 \
        --workspace=1024
else
    $TRTEXEC \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --workspace=1024
fi

echo ""
echo "=============================================="
echo "Build complete!"
echo "Engine saved: $ENGINE_FILE"
echo "=============================================="

# Print engine info
ls -lh "$ENGINE_FILE"
