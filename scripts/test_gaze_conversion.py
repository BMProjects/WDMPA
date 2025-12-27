#!/usr/bin/env python3
"""测试 gaze_to_3d 函数修复前后的差异."""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wdmpa.utils.metrics import gaze_to_3d, angular_error

# 测试用例
print("=" * 60)
print("测试 gaze_to_3d 函数修复")
print("=" * 60)

# 测试1: 简单角度
print("\n测试 1: 正前方 (pitch=0, yaw=0)")
gaze = torch.tensor([[0.0, 0.0]])
vec_3d = gaze_to_3d(gaze)
print(f"输入: {gaze}")
print(f"3D向量: {vec_3d}")
print(f"预期: [0, 0, -1] (Z轴负方向)")

# 测试2: 向上看
print("\n测试 2: 向上看 (pitch=30, yaw=0)")
gaze = torch.tensor([[30.0, 0.0]])
vec_3d = gaze_to_3d(gaze)
print(f"输入: {gaze}")
print(f"3D向量: {vec_3d}")
print(f"预期: x≈0, y<0 (负), z<0")

# 测试3: 向右看
print("\n测试 3: 向右看 (pitch=0, yaw=30)")
gaze = torch.tensor([[0.0, 30.0]])
vec_3d = gaze_to_3d(gaze)
print(f"输入: {gaze}")
print(f"3D向量: {vec_3d}")
print(f"预期: x<0 (负), y≈0, z<0")

# 测试4: 角度误差计算
print("\n测试 4: 角度误差")
pred = torch.tensor([[10.0, 20.0]])
target = torch.tensor([[12.0, 18.0]])
error = angular_error(pred, target)
print(f"预测: {pred}")
print(f"真实: {target}")
print(f"角度误差: {error.item():.2f}°")
print(f"合理范围: 2-4° (之前错误公式导致异常低)")

print("\n" + "=" * 60)
print("如果修复正确，fold 0 的结果应该从 2.59° 上升到 4-5°")
print("=" * 60)
