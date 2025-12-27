#!/bin/bash
# =============================================================================
# Jetson Nano 完整实验脚本
# 
# 用法:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# 前提条件:
#   - ONNX 模型已传输到 onnx/ 目录
#   - 已安装: onnxruntime-gpu
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="${SCRIPT_DIR}/../onnx"
ENGINE_DIR="${SCRIPT_DIR}/../engines"
RESULTS_DIR="${SCRIPT_DIR}/../results"

mkdir -p "$ENGINE_DIR" "$RESULTS_DIR"

echo "============================================================"
echo "WDMPA-Net Jetson Nano 实验"
echo "============================================================"
echo "开始时间: $(date)"
echo "ONNX 目录: $ONNX_DIR"
echo ""

# ------------------------------------------------------------
# 实验 A: ONNX 延迟测试
# ------------------------------------------------------------
echo "[实验 A] ONNX Runtime 延迟测试"
echo "------------------------------------------------------------"

MODELS=(
    "wdmpa_fold0"
    "l2cs_net"
    "mobilenetv3_small"
    "shufflenetv2"
)

for model in "${MODELS[@]}"; do
    onnx_file="${ONNX_DIR}/${model}.onnx"
    if [ -f "$onnx_file" ]; then
        echo ""
        echo "测试: $model"
        python3 "${SCRIPT_DIR}/benchmark.py" \
            --model "$onnx_file" \
            --format onnx \
            --warmup 100 \
            --iterations 500 \
            --output "${RESULTS_DIR}/onnx_benchmark.csv"
    else
        echo "跳过: $onnx_file (文件不存在)"
    fi
done

# ------------------------------------------------------------
# 实验 B: TensorRT 构建与测试
# ------------------------------------------------------------
echo ""
echo "[实验 B] TensorRT FP16 测试"
echo "------------------------------------------------------------"

# 检查 trtexec 是否可用
if command -v trtexec &> /dev/null; then
    for model in "${MODELS[@]}"; do
        onnx_file="${ONNX_DIR}/${model}.onnx"
        engine_file="${ENGINE_DIR}/${model}_fp16.engine"
        
        if [ -f "$onnx_file" ] && [ ! -f "$engine_file" ]; then
            echo ""
            echo "构建 TensorRT 引擎: $model"
            trtexec \
                --onnx="$onnx_file" \
                --saveEngine="$engine_file" \
                --fp16 \
                --workspace=1024 \
                2>&1 | grep -E "(mean|throughput|latency)" || true
        fi
        
        if [ -f "$engine_file" ]; then
            echo ""
            echo "测试 TensorRT: $model"
            python3 "${SCRIPT_DIR}/benchmark.py" \
                --model "$engine_file" \
                --format tensorrt \
                --warmup 100 \
                --iterations 500 \
                --output "${RESULTS_DIR}/tensorrt_benchmark.csv"
        fi
    done
else
    echo "trtexec 不可用，跳过 TensorRT 测试"
fi

# ------------------------------------------------------------
# 实验 C: 热稳定性测试 (10分钟)
# ------------------------------------------------------------
echo ""
echo "[实验 C] 热稳定性测试 (10分钟)"
echo "------------------------------------------------------------"
echo "注意: 此测试需要较长时间，可按 Ctrl+C 跳过"
echo ""

wdmpa_onnx="${ONNX_DIR}/wdmpa_fold0.onnx"
if [ -f "$wdmpa_onnx" ]; then
    python3 "${SCRIPT_DIR}/thermal_test.py" \
        --model "$wdmpa_onnx" \
        --duration 600 \
        --output "${RESULTS_DIR}"
else
    echo "跳过: $wdmpa_onnx (文件不存在)"
fi

# ------------------------------------------------------------
# 完成
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "实验完成!"
echo "============================================================"
echo "结束时间: $(date)"
echo ""
echo "结果保存在: $RESULTS_DIR"
ls -la "$RESULTS_DIR" 2>/dev/null || true
