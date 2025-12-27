#!/bin/bash
# 清理所有训练结果，准备重新开始
# 用法: ./scripts/clean_and_restart.sh

echo "⚠️  警告: 这将删除所有训练结果!"
echo ""
echo "将要删除:"
echo "  - runs/baselines/       (基线模型权重)"
echo "  - runs/ablation/        (消融实验权重)"
echo "  - results/logs/         (训练日志)"
echo "  - results/*.csv         (结果文件)"
echo ""
read -p "确认删除? (输入 'yes' 继续): " confirm

if [ "$confirm" != "yes" ]; then
    echo "已取消"
    exit 1
fi

echo ""
echo "正在清理..."

# 删除训练结果
rm -rf runs/baselines/
rm -rf runs/ablation/

# 删除日志
rm -rf results/logs/
rm -f results/experiment_results.csv
rm -f results/model_inventory.csv
rm -f results/train_log.txt
rm -f results/training_status.txt
rm -f results/status_history.txt

# 重新创建目录
mkdir -p runs/baselines/mobilenetv3
mkdir -p runs/baselines/shufflenetv2
mkdir -p runs/ablation
mkdir -p results/logs

echo ""
echo "✅ 清理完成!"
echo ""
echo "现在可以重新开始训练:"
echo "  ./scripts/run_experiments_foreground.sh"
