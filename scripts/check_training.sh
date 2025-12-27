#!/bin/bash
# WDMPA-Net 训练错误检测和通知脚本
# 用法: ./scripts/check_training.sh [--notify]
# 可以设置 cron 定期运行: */30 * * * * /home/bm/Dev/WDMPA/scripts/check_training.sh --notify

PROJECT_ROOT="/home/bm/Dev/WDMPA"
LOG_FILE="$PROJECT_ROOT/results/train_log.txt"
RESULTS_FILE="$PROJECT_ROOT/results/experiment_results.csv"
STATUS_FILE="$PROJECT_ROOT/results/training_status.txt"
NOTIFY=$1

check_status() {
    local status="OK"
    local message=""
    
    # 1. 检查进程是否运行
    PID=$(pgrep -f "run_all_experiments.sh")
    if [ -z "$PID" ]; then
        # 检查是否正常完成
        if grep -q "所有训练完成" "$LOG_FILE" 2>/dev/null; then
            status="COMPLETED"
            message="✅ 所有训练已完成!"
        else
            status="ERROR"
            message="❌ 训练进程意外终止!"
        fi
    else
        message="✅ 训练正在运行 (PID: $PID)"
    fi
    
    # 2. 检查日志中的错误 (排除正常的 "Error:" 输出)
    if [ -f "$LOG_FILE" ]; then
        ERROR_COUNT=$(grep -c -E "(Exception|CUDA out of memory|Traceback|RuntimeError|KeyError|ValueError)" "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$ERROR_COUNT" -gt 0 ]; then
            status="WARNING"
            message="$message\n⚠️ 检测到 $ERROR_COUNT 个潜在错误"
        fi
    fi
    
    # 3. 检查GPU内存
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_MEM" ] && [ "$GPU_MEM" -gt 22000 ]; then
        status="WARNING"
        message="$message\n⚠️ GPU内存使用过高: ${GPU_MEM}MB"
    fi
    
    # 4. 检查磁盘空间
    DISK_FREE=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$DISK_FREE" -lt 10 ]; then
        status="WARNING"
        message="$message\n⚠️ 磁盘空间不足: ${DISK_FREE}GB 剩余"
    fi
    
    # 5. 获取进度
    if [ -f "$RESULTS_FILE" ]; then
        COMPLETED=$(tail -n +2 "$RESULTS_FILE" | wc -l)
        message="$message\n📊 进度: $COMPLETED/36 模型已完成"
    fi
    
    # 输出状态
    echo "=========================================="
    echo "训练状态检查 - $(date)"
    echo "=========================================="
    echo -e "$message"
    
    # 保存状态到文件
    echo "$status" > "$STATUS_FILE"
    echo "$(date): $status" >> "$PROJECT_ROOT/results/status_history.txt"
    
    # 如果需要通知且有问题
    if [ "$NOTIFY" == "--notify" ] && [ "$status" != "OK" ]; then
        # 桌面通知 (如果可用)
        if command -v notify-send &> /dev/null; then
            notify-send "WDMPA训练状态: $status" "$message"
        fi
        
        # 声音提示 (如果可用)
        if command -v paplay &> /dev/null && [ "$status" == "ERROR" ]; then
            paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga 2>/dev/null
        fi
    fi
    
    return 0
}

# 检查最近错误的详细信息
show_recent_errors() {
    echo ""
    echo "=========================================="
    echo "最近的错误/警告 (如有):"
    echo "=========================================="
    if [ -f "$LOG_FILE" ]; then
        grep -E "(Error|Exception|CUDA|Traceback|error)" "$LOG_FILE" | tail -10
    fi
}

# 运行检查
check_status
show_recent_errors

echo ""
echo "=========================================="
echo "💡 提示:"
echo "   定期检查: watch -n 60 ./scripts/check_training.sh"
echo "   设置cron: crontab -e"
echo "   添加: */30 * * * * $PROJECT_ROOT/scripts/check_training.sh --notify"
echo "=========================================="
