#!/bin/bash
# 综合实验数据导出脚本
# 导出所有实验的配置、日志和结果

set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
EXPORT_DIR="$PROJECT_ROOT/experiment_export_$(date +%Y%m%d_%H%M%S)"

cd "$PROJECT_ROOT"

echo "==========================================="
echo "实验数据导出"
echo "==========================================="
echo "导出目录: $EXPORT_DIR"
echo ""

# 创建导出目录结构
mkdir -p "$EXPORT_DIR"/{configs,results,logs,weights_info,analysis}

# 1. 导出实验配置
echo "[1/6] 导出实验配置..."
cat > "$EXPORT_DIR/configs/experiment_overview.md" << 'EOF'
# WDMPA-Net 实验配置总览

## 实验环境

- **硬件**: NVIDIA RTX 4090 (24GB VRAM)
- **软件**: PyTorch, uv, Python 3.13
- **数据集**: MPIIFaceGaze (预处理版)
- **评估**: 15-fold Leave-One-Person-Out (LOPO)

## 实验列表

### 1. 基线模型训练 (RTX 4090优化参数)

**时间**: 2025-12-27

**训练参数**:
- Batch Size: 512
- Learning Rate: 1.6e-3
- Optimizer: AdamW
- Weight Decay: 0.01
- Warmup Epochs: 8
- Total Epochs: 60
- AMP: True
- Workers: 12

**模型列表**:
1. MobileNetV3-Small (15 folds) - 完成
2. ShuffleNetV2 (15 folds) - 完成

### 2. WDMPA-Net重训练 (RTX 4090优化参数)

**时间**: 2025-12-27

**训练参数**: 同上

**模型**: WDMPA-Net (15 folds) - 完成

### 3. 消融实验 (Fold 0)

**时间**: 2025-12-27

**训练参数**: 同基线模型

**变体列表**:
1. wdmpa_awwd_fixed - 固定Haar小波
2. wdmpa_stride_conv - Stride Conv替代AWWD
3. wdmpa_channel_only - 仅通道注意力
4. wdmpa_spatial_only - 仅空间注意力
5. wdmpa_single_scale - 单尺度MPA
6. wdmpa_no_attention - 无注意力

### 4. 原始预训练权重

**WDMPA-Net**: weights/fold*_best.pkl (15 folds)
**L2CS-Net**: weights/l2cs_mpiigaze/ (15 folds)

训练参数: 保守参数 (batch=32, lr=1e-4)
EOF

echo "✅ 实验配置已导出"

# 2. 导出结果CSV
echo "[2/6] 导出结果CSV..."
cp results/experiment_results.csv "$EXPORT_DIR/results/" 2>/dev/null || true
cp results/wdmpa_results.csv "$EXPORT_DIR/results/" 2>/dev/null || true
echo "✅ 结果CSV已导出"

# 3. 整理日志文件
echo "[3/6] 整理日志文件..."
# 复制所有日志
cp -r results/logs "$EXPORT_DIR/" 2>/dev/null || true

# 创建日志索引
cat > "$EXPORT_DIR/logs/LOG_INDEX.md" << 'EOF'
# 训练日志索引

## MobileNetV3 (15 folds)
- mobilenetv3_fold0.log - mobilenetv3_fold14.log

## ShuffleNetV2 (15 folds)
- shufflenetv2_fold0.log - shufflenetv2_fold14.log

## WDMPA-Net (15 folds)
- wdmpa/wdmpa_fold0.log - wdmpa/wdmpa_fold14.log

## 消融实验 (Fold 0)
- wdmpa_awwd_fixed.log
- wdmpa_stride_conv.log
- wdmpa_channel_only.log
- wdmpa_spatial_only.log
- wdmpa_single_scale.log
- wdmpa_no_attention.log

## 日志内容

每个日志文件包含:
- 每个epoch的训练loss
- 每个epoch的验证loss和angular error
- 最佳模型保存记录
- 训练时间统计
EOF

echo "✅ 日志文件已整理"

# 4. 提取权重文件信息
echo "[4/6] 提取权重文件信息..."
cat > "$EXPORT_DIR/weights_info/weights_summary.txt" << 'EOF'
权重文件汇总
=================================================================

原始预训练权重:
EOF

find weights -name "*.pkl" -type f -exec ls -lh {} \; >> "$EXPORT_DIR/weights_info/weights_summary.txt"

cat >> "$EXPORT_DIR/weights_info/weights_summary.txt" << 'EOF'

新训练权重 (RTX 4090参数):
EOF

find runs -name "best.pkl" -type f -exec ls -lh {} \; >> "$EXPORT_DIR/weights_info/weights_summary.txt"

echo "✅ 权重信息已提取"

# 5. 生成分析报告
echo "[5/6] 生成分析报告..."

# 复制已有分析报告
cp ABLATION_RESULTS.md "$EXPORT_DIR/analysis/" 2>/dev/null || true
cp WDMPA_RETRAIN_RESULTS.md "$EXPORT_DIR/analysis/" 2>/dev/null || true
cp results_summary.txt "$EXPORT_DIR/analysis/" 2>/dev/null || true
cp training_summary.txt "$EXPORT_DIR/analysis/" 2>/dev/null || true

# 生成实验对比表
python3 << 'EOFPYTHON'
import csv
from pathlib import Path

# 读取结果
results = {}

# MobileNetV3/ShuffleNetV2
csv_file = Path('results/experiment_results.csv')
if csv_file.exists():
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            fold = row['fold']
            if model not in results:
                results[model] = {}
            results[model][fold] = {
                'best': row['best_error'],
                'final': row['final_error']
            }

# WDMPA-Net
wdmpa_csv = Path('results/wdmpa_results.csv')
if wdmpa_csv.exists():
    with open(wdmpa_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold = row['fold']
            if 'wdmpa' not in results:
                results['wdmpa'] = {}
            results['wdmpa'][fold] = {
                'best': row['best_error'],
                'final': row['final_error']
            }

# 生成对比表
output_dir = Path('experiment_export_*').resolve()
# Use most recent export dir
export_dirs = sorted(Path('.').glob('experiment_export_*'))
if export_dirs:
    output_file = export_dirs[-1] / 'analysis' / 'experiment_comparison.csv'
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fold', 'MobileNetV3', 'ShuffleNetV2', 'WDMPA-Net', 'Best Model'])
        
        for fold in range(15):
            fold_str = str(fold)
            mobilenet = float(results.get('mobilenetv3', {}).get(fold_str, {}).get('best', '999'))
            shufflenet = float(results.get('shufflenetv2', {}).get(fold_str, {}).get('best', '999'))
            wdmpa = float(results.get('wdmpa', {}).get(fold_str, {}).get('best', '999'))
            
            best = min(mobilenet, shufflenet, wdmpa)
            if best == mobilenet:
                best_model = 'MobileNetV3'
            elif best == shufflenet:
                best_model = 'ShuffleNetV2'
            else:
                best_model = 'WDMPA-Net'
            
            writer.writerow([fold, mobilenet, shufflenet, wdmpa, best_model])
    
    print(f"✅ 对比表已生成: {output_file}")
EOFPYTHON

echo "✅ 分析报告已生成"

# 6. 创建README
echo "[6/6] 创建README..."
cat > "$EXPORT_DIR/README.md" << 'EOF'
# WDMPA-Net 实验数据导出

本目录包含所有实验的完整数据，方便后续分析和对比。

## 目录结构

```
experiment_export_YYYYMMDD_HHMMSS/
├── README.md                    # 本文件
├── configs/                     # 实验配置
│   └── experiment_overview.md   # 实验总览
├── results/                     # 结果CSV
│   ├── experiment_results.csv   # 基线模型+消融实验
│   └── wdmpa_results.csv       # WDMPA重训练
├── logs/                        # 训练日志
│   ├── LOG_INDEX.md            # 日志索引
│   ├── mobilenetv3_fold*.log   # MobileNetV3日志
│   ├── shufflenetv2_fold*.log  # ShuffleNetV2日志
│   ├── wdmpa/                  # WDMPA日志
│   └── wdmpa_*.log             # 消融实验日志
├── weights_info/               # 权重文件信息
│   └── weights_summary.txt     # 权重文件列表
└── analysis/                   # 分析报告
    ├── ABLATION_RESULTS.md     # 消融实验分析
    ├── WDMPA_RETRAIN_RESULTS.md # WDMPA重训练分析
    ├── experiment_comparison.csv # 实验对比表
    ├── results_summary.txt     # 结果汇总
    └── training_summary.txt    # 训练汇总
```

## 使用说明

### 查看实验配置
```bash
cat configs/experiment_overview.md
```

### 查看结果汇总
```bash
cat analysis/results_summary.txt
cat analysis/training_summary.txt
```

### 分析训练日志
```bash
# 查看某个fold的训练过程
cat logs/mobilenetv3_fold0.log

# 提取最佳误差
grep "New best:" logs/mobilenetv3_fold0.log

# 查看最终结果
tail -20 logs/mobilenetv3_fold0.log
```

### 对比不同实验
```bash
# 查看实验对比表
cat analysis/experiment_comparison.csv | column -t -s','
```

## 关键数据

### 15-fold平均误差

| 模型 | 平均误差 |
|------|---------|
| MobileNetV3 | 4.46° |
| ShuffleNetV2 | 4.62° |
| WDMPA-Net | 4.96° |

### 消融实验 (Fold 0)

| 变体 | Best Error |
|------|-----------|
| 完整WDMPA | 2.68° |
| Stride Conv | 2.67° |
| AWWD Fixed | 2.72° |
| Single Scale | 2.72° |
| Spatial Only | 2.80° |
| Channel Only | 2.82° |
| No Attention | 2.91° |

## 导出时间

$(date)
EOF

echo "✅ README已创建"

# 压缩导出目录
echo ""
echo "压缩导出数据..."
tar -czf "${EXPORT_DIR}.tar.gz" -C "$(dirname $EXPORT_DIR)" "$(basename $EXPORT_DIR)"

echo ""
echo "==========================================="
echo "✅ 实验数据导出完成!"
echo "==========================================="
echo "导出目录: $EXPORT_DIR"
echo "压缩文件: ${EXPORT_DIR}.tar.gz"
echo "大小: $(du -sh ${EXPORT_DIR}.tar.gz | cut -f1)"
echo ""
echo "查看内容:"
echo "  tar -tzf ${EXPORT_DIR}.tar.gz"
echo ""
echo "解压:"
echo "  tar -xzf ${EXPORT_DIR}.tar.gz"
