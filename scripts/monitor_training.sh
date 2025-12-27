#!/bin/bash
# WDMPA-Net 训练监控脚本
# 用法: ./scripts/monitor_training.sh

PROJECT_ROOT="/home/bm/Dev/WDMPA"
LOG_FILE="$PROJECT_ROOT/results/train_log.txt"
RESULTS_FILE="$PROJECT_ROOT/results/experiment_results.csv"

clear
echo "=========================================="
echo "  WDMPA-Net 训练监控面板"
echo "=========================================="
echo ""

# 检查进程状态
PID=$(pgrep -f "run_all_experiments.sh")
if [ -n "$PID" ]; then
    echo "📊 进程状态: ✅ 运行中 (PID: $PID)"
    RUNTIME=$(ps -o etime= -p $PID | tr -d ' ')
    echo "⏱️  运行时间: $RUNTIME"
else
    echo "📊 进程状态: ❌ 未运行"
fi

echo ""

# 检查GPU状态
echo "🎮 GPU 状态:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  GPU 信息不可用"

echo ""

# 检查已完成的模型数
if [ -f "$RESULTS_FILE" ]; then
    COMPLETED=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    echo "✅ 已完成模型: $COMPLETED / 36"
    echo ""
    echo "最近完成的模型:"
    tail -3 "$RESULTS_FILE" | column -t -s','
fi

echo ""
echo "=========================================="
echo "📋 最新训练日志 (最后20行):"
echo "=========================================="
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "日志文件不存在"
fi

echo ""
echo "=========================================="
echo "💡 快捷命令:"
echo "   实时日志: tail -f $LOG_FILE"
echo "   查看结果: cat $RESULTS_FILE"
echo "   停止训练: kill $PID"
echo "=========================================="
