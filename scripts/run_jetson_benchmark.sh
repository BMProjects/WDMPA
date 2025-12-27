#!/bin/bash
# Run all Jetson benchmarks and generate comparison data
#
# Usage:
#   ./scripts/run_jetson_benchmark.sh

set -e
cd "$(dirname "$0")/.."

ONNX_DIR="deploy/onnx"
ENGINE_DIR="deploy/engines"
RESULTS_DIR="deploy/results"

mkdir -p "$ENGINE_DIR" "$RESULTS_DIR"

echo "=============================================="
echo "WDMPA Jetson Benchmark Suite"
echo "=============================================="

# 1. ONNX Benchmarks
echo ""
echo "[1/3] Running ONNX benchmarks..."
for model in "$ONNX_DIR"/*.onnx; do
    name=$(basename "$model" .onnx)
    echo "  Testing: $name"
    python deploy/jetson/benchmark.py \
        --model "$model" \
        --warmup 100 \
        --iterations 1000 \
        --output "$RESULTS_DIR/${name}_onnx.csv" 2>/dev/null || echo "    Failed"
done

# 2. Build TensorRT Engines
echo ""
echo "[2/3] Building TensorRT engines..."
for model in "$ONNX_DIR"/*.onnx; do
    name=$(basename "$model" .onnx)
    echo "  Building: $name"
    ./deploy/jetson/build_tensorrt.sh "$model" "$ENGINE_DIR" fp16 2>/dev/null || echo "    Failed"
done

# 3. TensorRT Benchmarks
echo ""
echo "[3/3] Running TensorRT benchmarks..."
for engine in "$ENGINE_DIR"/*.engine; do
    if [ -f "$engine" ]; then
        name=$(basename "$engine" .engine)
        echo "  Testing: $name"
        python deploy/jetson/benchmark.py \
            --model "$engine" \
            --warmup 100 \
            --iterations 1000 \
            --output "$RESULTS_DIR/${name}_trt.csv" 2>/dev/null || echo "    Failed"
    fi
done

# Summary
echo ""
echo "=============================================="
echo "Results saved to: $RESULTS_DIR"
echo "=============================================="
ls -la "$RESULTS_DIR"
