#!/bin/bash
# Jetson Nano一键测试脚本
# 自动运行所有性能测试并收集结果

set -e

echo "=========================================="
echo "WDMPA-Net Jetson Nano 性能测试"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

# 创建结果目录
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "结果目录: $RESULTS_DIR"
echo ""

# 检查ONNX模型
if [ ! -d "onnx" ] || [ -z "$(ls -A onnx/*.onnx 2>/dev/null)" ]; then
    echo "❌ 错误: 未找到ONNX模型"
    echo "请确保 onnx/ 目录包含模型文件"
    exit 1
fi

echo "找到以下模型:"
ls -lh onnx/*.onnx
echo ""

# 测试1: ONNX推理延迟
echo "[测试 1/3] ONNX推理性能测试..."
echo "--------------------"
python3 benchmark_onnx.py \
    --models onnx/ \
    --warmup 50 \
    --iterations 500 \
    --output "$RESULTS_DIR/latency.csv"

echo ""
echo "✅ 延迟测试完成"
echo ""

# 测试2: 内存使用
echo "[测试 2/3] 内存占用测试..."
echo "--------------------"
python3 memory_profiler.py \
    --models onnx/ \
    --output "$RESULTS_DIR/memory_usage.csv"

echo ""
echo "✅ 内存测试完成"
echo ""

# 测试3: 热稳定性 (可选，10分钟)
read -p "是否运行热稳定性测试? (10分钟) [y/N]: " run_thermal
if [[ "$run_thermal" =~ ^[Yy]$ ]]; then
    echo "[测试 3/3] 热稳定性测试 (10分钟)..."
    echo "--------------------"
    python3 thermal_stability.py \
        --model onnx/wdmpa_fold0.onnx \
        --duration 600 \
        --output "$RESULTS_DIR/thermal_curve.png"
    
    echo ""
    echo "✅ 热稳定性测试完成"
else
    echo "[测试 3/3] 跳过"
fi

# 生成摘要报告
echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="
echo "结束时间: $(date)"
echo ""

# 显示结果
echo "结果摘要:"
echo "--------------------"
if [ -f "$RESULTS_DIR/latency.csv" ]; then
    echo "延迟测试:"
    cat "$RESULTS_DIR/latency.csv" | column -t -s','
    echo ""
fi

if [ -f "$RESULTS_DIR/memory_usage.csv" ]; then
    echo "内存占用:"
    cat "$RESULTS_DIR/memory_usage.csv" | column -t -s','
    echo ""
fi

# 打包结果
echo "打包结果..."
tar -czf "${RESULTS_DIR}.tar.gz" "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "✅ 所有结果已保存"
echo "=========================================="
echo "目录: $RESULTS_DIR"
echo "打包: ${RESULTS_DIR}.tar.gz"
echo ""
echo "传回PC:"
echo "  scp ${RESULTS_DIR}.tar.gz user@pc:~/"
