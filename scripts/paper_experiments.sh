#!/bin/bash
# Complete Jetson benchmark suite for paper experiments
#
# This script runs all required experiments:
# - Experiment A: Latency/FPS comparison
# - Experiment B: Power efficiency
# - Experiment C: Thermal stability
#
# Usage:
#   ./scripts/paper_experiments.sh

set -e
cd "$(dirname "$0")/.."

ONNX_DIR="deploy/onnx"
ENGINE_DIR="deploy/engines"
RESULTS_DIR="deploy/results/paper"
ABLATION_DIR="deploy/onnx/ablation"

mkdir -p "$ENGINE_DIR" "$RESULTS_DIR"

echo "============================================================"
echo "WDMPA-Net Paper Experiments for Jetson Nano"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# ============================================================
# Experiment A: Core latency comparison
# ============================================================
echo "[Experiment A] Core Latency Comparison"
echo "============================================================"

CORE_MODELS=(
    "wdmpa_fold0"
    "mobilenetv3_small"
    "shufflenetv2"
    "resnet50_gaze"
)

echo ""
echo "A.1 ONNX Runtime benchmarks..."
for model in "${CORE_MODELS[@]}"; do
    onnx_file="$ONNX_DIR/${model}.onnx"
    if [ -f "$onnx_file" ]; then
        echo "  Testing: $model (ONNX)"
        python deploy/jetson/benchmark.py \
            --model "$onnx_file" \
            --format onnx \
            --warmup 100 \
            --iterations 1000 \
            --output "$RESULTS_DIR/${model}_onnx.csv" 2>/dev/null || echo "    [FAILED]"
    fi
done

echo ""
echo "A.2 Building TensorRT FP16 engines..."
for model in "${CORE_MODELS[@]}"; do
    onnx_file="$ONNX_DIR/${model}.onnx"
    if [ -f "$onnx_file" ]; then
        echo "  Building: $model"
        ./deploy/jetson/build_tensorrt.sh "$onnx_file" "$ENGINE_DIR" fp16 2>/dev/null || echo "    [FAILED]"
    fi
done

echo ""
echo "A.3 TensorRT benchmarks..."
for model in "${CORE_MODELS[@]}"; do
    engine_file="$ENGINE_DIR/${model}_fp16.engine"
    if [ -f "$engine_file" ]; then
        echo "  Testing: $model (TensorRT FP16)"
        python deploy/jetson/benchmark.py \
            --model "$engine_file" \
            --format tensorrt \
            --warmup 100 \
            --iterations 1000 \
            --output "$RESULTS_DIR/${model}_trt.csv" 2>/dev/null || echo "    [FAILED]"
    fi
done

# ============================================================
# Experiment C: Thermal stability (10 minutes)
# ============================================================
echo ""
echo "[Experiment C] Thermal Stability Test"
echo "============================================================"
echo "Running 10-minute thermal test on WDMPA-Net..."

python deploy/jetson/thermal_test.py \
    --model "$ONNX_DIR/wdmpa_fold0.onnx" \
    --duration 600 \
    --interval 1.0 \
    --output "$RESULTS_DIR" 2>/dev/null || echo "  [FAILED]"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Experiments Complete!"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: $RESULTS_DIR"
ls -la "$RESULTS_DIR" 2>/dev/null || true

echo ""
echo "Next steps:"
echo "  1. Review CSV files for latency data"
echo "  2. Generate Pareto plot: python utils/paper_plots.py"
echo "  3. Check thermal_*.png for stability curve"
