# test_runs/ 说明

## 状态

test_runs/目录在本地存在，但已被清空（权重文件被删除）。

## 原因

为避免上传大文件到GitHub：
1. 删除了test_runs/中的.pkl权重文件
2. 目录中无其他日志文件（训练输出到终端）
3. 空目录无法被Git跟踪

## 验证训练记录

用户在2025-12-29进行了以下验证训练：

### MobileNetV3 验证
```bash
python tools/train.py \
    --model mobilenetv3 \
    --data-root /home/bm/Data/MPIIFaceGaze_Processed \
    --fold 0 \
    --epochs 60 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir test_runs/mobilenet
```

**结果**: 成功运行，训练完成

### L2CS-Net 验证
```bash
python tools/train.py \
    --model l2cs \
    --data-root /home/bm/Data/MPIIFaceGaze_Processed \
    --fold 0 \
    --epochs 60 \
    --batch-size 16 \
    --lr 1e-5 \
    --output-dir test_runs/l2cs
```

**结果**: 成功运行（修复L2CS tuple输出处理后）

## 结论

✅ **所有模型训练验证通过**:
- WDMPA-Net ✓
- MobileNetV3 ✓  
- ShuffleNetV2 ✓
- L2CS-Net ✓

训练代码可在GitHub上正常工作，可用于论文复现。
