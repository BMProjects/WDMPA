# Jetson Nano 部署模型准备方案

## 模型清单 (Fold 0)

### ✅ 已有权重（直接使用）

| 模型 | 权重路径 | 大小 | 训练参数 | 状态 |
|------|---------|------|---------|------|
| **WDMPA-Net** | `weights/fold0_best.pkl` | 11MB | 原始保守参数 | ✅ 使用 |
| **L2CS-Net** | `weights/l2cs_mpiigaze/fold0.pkl` | 91MB | 官方预训练 | ✅ 使用 |
| **MobileNetV3** | `runs/baselines/mobilenetv3/fold0/*/best.pkl` | 4.3MB | RTX 4090参数 | ⚠️ 重训练 |
| **ShuffleNetV2** | `runs/baselines/shufflenetv2/fold0/*/best.pkl` | 6.0MB | RTX 4090参数 | ⚠️ 重训练 |

**说明**:
- WDMPA和L2CS: 使用原始预训练权重 ✅
- MobileNetV3和ShuffleNetV2: 当前权重用RTX 4090激进参数训练，需要用保守参数重训练

---

## 重训练需求

### MobileNetV3 (Fold 0)

**当前问题**: batch=512, lr=1.6e-3 (激进参数)

**重训练参数**:
```bash
batch_size: 32
lr: 1e-4
epochs: 60
optimizer: AdamW
```

**命令**:
```bash
PYTHONPATH=/home/bm/Dev/WDMPA uv run python3 tools/train.py \
    --model mobilenetv3 \
    --data-root /home/bm/Data/MPIIFaceGaze_Processed \
    --fold 0 \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 60 \
    --output-dir runs/baselines_conservative/mobilenetv3/fold0 \
    --name fold0_conservative
```

**时间**: ~1.5小时

### ShuffleNetV2 (Fold 0)

**重训练参数**: 同上

**命令**:
```bash
PYTHONPATH=/home/bm/Dev/WDMPA uv run python3 tools/train.py \
    --model shufflenetv2 \
    --data-root /home/bm/Data/MPIIFaceGaze_Processed \
    --fold 0 \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 60 \
    --output-dir runs/baselines_conservative/shufflenetv2/fold0 \
    --name fold0_conservative
```

**时间**: ~1.5小时

**总重训练时间**: ~3小时

---

## 测试数据准备

### 方案1: 不需要真实数据（推荐）

**Jetson实验目标**: 测试推理延迟/FPS

**使用随机tensor作为输入**:
- 输入shape: [1, 3, 224, 224]
- 随机生成: `np.random.randn(1, 3, 224, 224).astype(np.float32)`

**优点**:
- 无需准备数据
- 测试延迟足够准确
- 节省传输时间

**Jetson脚本已自动生成随机数据**:
```python
# 在 benchmark_onnx.py 中
dummy_input = np.random.randn(*input_shape).astype(np.float32)
```

✅ **推荐使用此方案**

### 方案2: 准备真实测试数据（可选）

**如果需要验证精度**，准备100张Fold 0图片：

```bash
# 从MPIIFaceGaze提取p00的100张图片
python3 << 'EOF'
import random
import shutil
from pathlib import Path

data_root = Path('/home/bm/Data/MPIIFaceGaze_Processed')
output = Path('deploy/test_data/images')
output.mkdir(parents=True, exist_ok=True)

# 获取p00的所有图片
images = list((data_root / 'Image' / 'p00').glob('*.jpg'))
samples = random.sample(images, min(100, len(images)))

for i, img in enumerate(samples, 1):
    shutil.copy(img, output / f'test_{i:03d}.jpg')
    
print(f"✅ 已复制 {len(samples)} 张图片到 deploy/test_data/images/")
EOF
```

**数据大小**: ~20MB

**何时需要**:
- 如果要在Jetson上验证精度
- 如果要测试真实图片的推理效果

**何时不需要**:
- 仅测试延迟/FPS → 不需要
- Jetson内存受限 (2GB) → 不需要

---

## 完整执行流程

### 步骤1: 重训练基线模型 (保守参数)

```bash
cd /home/bm/Dev/WDMPA

# 创建训练脚本
cat > scripts/retrain_baselines_conservative.sh << 'EOF'
#!/bin/bash
set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
DATA_ROOT="/home/bm/Data/MPIIFaceGaze_Processed"

cd "$PROJECT_ROOT"

# MobileNetV3
echo "[1/2] Training MobileNetV3 (conservative)..."
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
    --model mobilenetv3 \
    --data-root "$DATA_ROOT" \
    --fold 0 \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 60 \
    --output-dir runs/baselines_conservative/mobilenetv3/fold0 \
    --name fold0_conservative

# ShuffleNetV2
echo "[2/2] Training ShuffleNetV2 (conservative)..."
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/train.py \
    --model shufflenetv2 \
    --data-root "$DATA_ROOT" \
    --fold 0 \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 60 \
    --output-dir runs/baselines_conservative/shufflenetv2/fold0 \
    --name fold0_conservative

echo "✅ 保守参数重训练完成"
EOF

chmod +x scripts/retrain_baselines_conservative.sh

# 执行训练
./scripts/retrain_baselines_conservative.sh
```

**时间**: ~3小时

### 步骤2: 导出所有ONNX模型

```bash
# 修改export脚本，使用正确的权重路径
cat > deploy/scripts/export_jetson_models.sh << 'EOF'
#!/bin/bash
set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
OUTPUT_DIR="$PROJECT_ROOT/deploy/onnx"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "导出Jetson部署模型 (Fold 0)"

# WDMPA-Net (原始预训练)
echo "[1/4] WDMPA-Net..."
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model wdmpa \
    --weights weights/fold0_best.pkl \
    --output "$OUTPUT_DIR/wdmpa_fold0.onnx" \
    --opset 11

# L2CS-Net (原始预训练)
echo "[2/4] L2CS-Net..."
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model l2cs \
    --weights weights/l2cs_mpiigaze/fold0.pkl \
    --output "$OUTPUT_DIR/l2cs_fold0.onnx" \
    --opset 11

# MobileNetV3 (保守参数重训练)
echo "[3/4] MobileNetV3..."
WEIGHT=$(find runs/baselines_conservative/mobilenetv3/fold0 -name "best.pkl" | head -1)
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model mobilenetv3 \
    --weights "$WEIGHT" \
    --output "$OUTPUT_DIR/mobilenetv3_fold0.onnx" \
    --opset 11

# ShuffleNetV2 (保守参数重训练)
echo "[4/4] ShuffleNetV2..."
WEIGHT=$(find runs/baselines_conservative/shufflenetv2/fold0 -name "best.pkl" | head -1)
PYTHONPATH=$PROJECT_ROOT uv run python3 tools/export.py \
    --model shufflenetv2 \
    --weights "$WEIGHT" \
    --output "$OUTPUT_DIR/shufflenetv2_fold0.onnx" \
    --opset 11

echo "✅ 所有模型已导出"
ls -lh "$OUTPUT_DIR"
EOF

chmod +x deploy/scripts/export_jetson_models.sh

# 执行导出
./deploy/scripts/export_jetson_models.sh
```

### 步骤3: 打包Jetson部署文件

```bash
# 使用已有的打包脚本
./deploy/scripts/package_for_jetson.sh
```

**输出**: `jetson_deploy.tar.gz` (~50MB)

### 步骤4: 传输到Jetson

```bash
scp jetson_deploy.tar.gz jetson@<IP>:~/
```

### 步骤5: Jetson上运行测试

```bash
# Jetson上
cd ~
tar -xzf jetson_deploy.tar.gz
cd jetson_deploy
./setup.sh

cd scripts
./run_all_benchmarks.sh
```

---

## 预期结果

### 模型精度 (Fold 0, 预期)

| 模型 | 权重来源 | 预期误差 |
|------|---------|---------|
| WDMPA-Net | 原始预训练 | ~2.5-3.0° |
| L2CS-Net | 原始预训练 | ~3.5-4.0° |
| MobileNetV3 | 保守重训练 | ~2.5-2.8° |
| ShuffleNetV2 | 保守重训练 | ~2.7-3.0° |

### Jetson性能 (预期)

| 模型 | 延迟 (ms) | FPS | 参数量 |
|------|----------|-----|--------|
| **WDMPA-Net** | **~15** | **~66** | 2.58M |
| L2CS-Net | ~60 | ~16 | ~25M |
| MobileNetV3 | ~10 | ~100 | 2.5M |
| ShuffleNetV2 | ~12 | ~83 | 2.3M |

**核心论点验证**:
- WDMPA vs L2CS: 精度相当，**速度快4x** ✅
- WDMPA vs 轻量级基线: 精度相当，速度略慢但架构创新 ✅

---

## 时间线

| 阶段 | 时间 |
|------|------|
| 重训练MobileNetV3 | 1.5h |
| 重训练ShuffleNetV2 | 1.5h |
| 导出ONNX (4个模型) | 10min |
| 打包传输 | 5min |
| **总计** | **~3.2h** |

---

## 检查清单

### 准备阶段
- [x] 确认WDMPA原始权重存在
- [x] 确认L2CS原始权重存在
- [ ] 重训练MobileNetV3 (保守参数)
- [ ] 重训练ShuffleNetV2 (保守参数)

### 导出阶段
- [ ] 导出WDMPA ONNX
- [ ] 导出L2CS ONNX
- [ ] 导出MobileNetV3 ONNX
- [ ] 导出ShuffleNetV2 ONNX

### 部署阶段
- [ ] 打包jetson_deploy.tar.gz
- [ ] 传输到Jetson Nano
- [ ] Jetson上运行测试
- [ ] 收集结果数据

---

## 答案：如何准备测试数据？

**推荐答案**: **不需要准备真实测试数据**

**理由**:
1. Jetson实验目标是测试**推理速度**（延迟/FPS）
2. 使用随机tensor足够准确
3. 节省传输和存储空间（Jetson只有2GB内存）
4. 脚本已自动处理

**如果需要验证精度**，可以使用方案2准备100张真实图片（20MB），但对于延迟测试非必需。
