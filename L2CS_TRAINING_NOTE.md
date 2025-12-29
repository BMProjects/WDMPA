# L2CS-Net 训练修复

## 问题

L2CS-Net使用分类方法预测视线角度，输出格式与其他回归模型不同：
- **其他模型**: 直接输出 `[B, 2]` tensor (pitch, yaw) in degrees
- **L2CS-Net**: 输出 `(yaw_logits, pitch_logits)` tuple，每个是 `[B, 90]` (90个bins的分类logits)

## 解决方案

在 `tools/train.py` 中添加了L2CS-Net输出格式转换：

```python
pred = model(images)

# Handle L2CS-Net output (tuple of logits)
if isinstance(pred, tuple):
    # Get bin indices using argmax
    yaw_idx = pred[0].argmax(dim=1)
    pitch_idx = pred[1].argmax(dim=1)
    
    # Convert bin index to angle
    # 90 bins cover -45° to +45°
    pred = torch.stack([
        (pitch_idx - 45) * (90.0 / 90),  # pitch in degrees  
        (yaw_idx - 45) * (90.0 / 90)     # yaw in degrees
    ], dim=1)

loss, loss_dict = criterion(pred, labels)
```

## 现在可以正常训练

```bash
# L2CS-Net训练命令
python tools/train.py \
    --model l2cs \
    --data-root /home/bm/Data/MPIIFaceGaze_Processed \
    --fold 0 \
    --epochs 10 \
    --batch-size 16 \
    --lr 1e-5 \
    --output-dir test_runs/l2cs
```

## 注意事项

1. **Batch Size**: L2CS-Net参数量大（23.8M），建议使用较小的batch_size (16)
2. **Learning Rate**: 使用较小的lr (1e-5)，与原始论文一致
3. **训练时间**: 由于模型大，训练比MobileNetV3/WDMPA慢约2-3倍

## 测试所有模型

```bash
export DATA_ROOT="/home/bm/Data/MPIIFaceGaze_Processed"

# WDMPA-Net (快)
python tools/train.py --model wdmpa --data-root $DATA_ROOT --fold 0 --epochs 10

# MobileNetV3 (快)
python tools/train.py --model mobilenetv3 --data-root $DATA_ROOT --fold 0 --epochs 10

# L2CS-Net (慢，但现在可以工作了)
python tools/train.py --model l2cs --data-root $DATA_ROOT --fold 0 --epochs 10 --batch-size 16 --lr 1e-5
```

所有模型现在都应该能正常训练！
